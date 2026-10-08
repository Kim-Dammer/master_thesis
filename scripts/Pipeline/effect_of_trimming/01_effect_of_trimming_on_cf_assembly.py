"""
For each complex: take the stoichiometry with the highest CombFold confidence in the pae run (18),
take the best assembly of the same stoichiometry in the trimmed run (19), and compare the two with US-align.

US-align outputs go to <trimmed pool folder>/us_align_plddt_trimmed/.
"""
import ast
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from procompa import get_project_root
from tqdm import tqdm

PRJ_ROOT = get_project_root()
pipeline_dir = PRJ_ROOT / "data/Pipeline"
pae_run_dir = pipeline_dir / "18_CP_model_based_on_pae"
trimmed_run_dir = pipeline_dir / "19_CP_pae_plddt_trimmed"
pae_results_csv = pae_run_dir / "all_pdb_present_18_CP_model_based_on_pae_pool_pipeline_complexes_combfold_results.csv"
pdb_mapping_csv = PRJ_ROOT / "data/complete_complex_pdb_mapping_v2/all_pdb_matches_with_match_class.csv"   # has n_proteins
N_PROTEINS_ABOVE = 2    # keep complexes with MORE than this many proteins
usalign_bin = PRJ_ROOT / "tools/usalign/USalign/USalign"
USALIGN_DIR_NAME = "us_align_plddt_trimmed"
SKIP_EXISTING = False   # True: reuse existing US-align outputs instead of failing
MAX_WORKERS = 8
comparison_csv_path = trimmed_run_dir / "effect_of_trimming_on_cf_assembly.csv"

for path in (pae_results_csv, pdb_mapping_csv, usalign_bin, trimmed_run_dir / "CombFold"):
    assert path.exists(), f"missing: {path}"

NON_PROTEIN_PREFIX = "CHEBI:"   # small molecules; CombFold pool folders only contain proteins
HASHED_POOL_DIR = re.compile(r"__[0-9a-f]{12}_pool_output$")   # long names: truncated protein list + hash


# ---------------- stoichiometry parsing ----------------
def stoich_from_pred_cell(cell):
    """'{"P1":1,"P2":2},{"rank":1,...}' -> (('P1', 1), ('P2', 2))"""
    first_dict = re.match(r"^\{([^}]*)\}", cell)
    assert first_dict, f"cannot parse prediction cell: {cell[:80]}"
    entries = first_dict.group(1).replace('"', "").split(",")
    return tuple(sorted((acc, int(n)) for acc, n in (entry.rsplit(":", 1) for entry in entries)))


def stoich_from_identifiers(cell):
    """'P1(1)|CHEBI:597326(2)|P2(2)' -> (('P1', 1), ('P2', 2)); small molecules are dropped"""
    parsed = [re.fullmatch(r"(.+)\((\d+)\)", entry) for entry in cell.split("|")]
    assert all(parsed), f"cannot parse identifiers: {cell}"
    proteins = [(m.group(1), int(m.group(2))) for m in parsed if not m.group(1).startswith(NON_PROTEIN_PREFIX)]
    assert proteins and all(n > 0 for _, n in proteins), f"no valid protein stoichiometry: {cell}"
    return tuple(sorted(proteins))


def stoich_from_pool_dir_name(dir_name):
    """'O13297x1_Q01159x1_pool_output' -> (('O13297', 1), ('Q01159', 1))"""
    entries = dir_name.removesuffix("_pool_output").split("_")
    parsed = [re.fullmatch(r"(.+)x(\d+)", entry) for entry in entries]
    assert all(parsed), f"cannot parse pool folder name: {dir_name}"
    return tuple(sorted((m.group(1), int(m.group(2))) for m in parsed))


def stoich_from_chain_list(pool_dir):
    """stoichiometry from CombFold's chain.list (one 'UNIPROT_CHAIN.pdb' per chain); used for hashed folder names"""
    chain_list = pool_dir / "_unified_representation/assembly_output/chain.list"
    assert chain_list.exists(), f"missing chain.list: {chain_list}"
    chain_files = [Path(line).name for line in chain_list.read_text().split()]
    parsed = [re.fullmatch(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pdb", name) for name in chain_files]
    assert all(parsed), f"unexpected chain.list entries in {chain_list}: {chain_files}"
    stoich = tuple(sorted(Counter(m.group(1) for m in parsed).items()))
    # the truncated protein list in the folder name must agree with chain.list
    name_prefix = HASHED_POOL_DIR.sub("", pool_dir.name).split("_")
    for entry in name_prefix:
        m = re.fullmatch(r"(.+)x(\d+)", entry)
        assert m and (m.group(1), int(m.group(2))) in stoich, f"{pool_dir.name}: {entry} not in chain.list {stoich}"
    return stoich


def stoich_to_str(stoich):
    return "_".join(f"{acc}x{n}" for acc, n in stoich)


# ---------------- pool folders ----------------
def index_pool_folders(run_dir):
    """{stoichiometry: pool_output folder} for one run"""
    stoich_to_pool_dir = {}
    for pool_dir in sorted((run_dir / "CombFold").glob("*_pool_output")):
        if HASHED_POOL_DIR.search(pool_dir.name):
            stoich = stoich_from_chain_list(pool_dir)
        else:
            stoich = stoich_from_pool_dir_name(pool_dir.name)
        assert stoich not in stoich_to_pool_dir, \
            f"two pool folders with the same stoichiometry: {stoich_to_pool_dir.get(stoich)} and {pool_dir}"
        stoich_to_pool_dir[stoich] = pool_dir
    assert stoich_to_pool_dir, f"no pool folders in {run_dir / 'CombFold'}"
    print(f"{run_dir.name}: {len(stoich_to_pool_dir)} pool folders")
    return stoich_to_pool_dir


def best_assembly_in_pool_dir(pool_dir):
    """(pdb path, confidence) of the highest-confidence assembly in confidence.txt; None if there is no assembly"""
    confidence_file = pool_dir / "assembled_results/confidence.txt"
    if not confidence_file.exists():
        return None
    pdb_to_confidence = {}
    for line in confidence_file.read_text().splitlines():
        if line.strip():
            pdb_path, confidence = line.rsplit(None, 1)
            pdb_to_confidence[pdb_path] = float(confidence)
    if not pdb_to_confidence:
        return None
    best_pdb = max(pdb_to_confidence, key=pdb_to_confidence.get)
    assert Path(best_pdb).exists(), f"pdb listed in {confidence_file} does not exist: {best_pdb}"
    return best_pdb, pdb_to_confidence[best_pdb]


# ---------------- best stoichiometry per complex in the pae run ----------------
def highest_confidence(cell):
    return max(ast.literal_eval(cell).values()) if cell not in (None, "{}") else None


def best_stoich_of_complex(complex_row):
    """(source, stoichiometry, confidence) of the highest-confidence candidate; None if no candidate has an assembly.
    Same candidates as in the notebook / 08 script: pred_1/2/3, plus the true stoichiometry only if no pred matched it
    (correct_pred_rank == "none"). For "unknown" rows the true stoichiometry is incomplete (counts of 0)."""
    candidates = []
    for rank in (1, 2, 3):
        confidence = highest_confidence(complex_row[f"CF_{rank}_confidence"])
        if confidence is not None:
            candidates.append((f"pred_{rank}", stoich_from_pred_cell(complex_row[f"pred_{rank}"]), confidence))
    true_confidence = highest_confidence(complex_row["CF_true_confidence"])
    if complex_row["correct_pred_rank"] == "none" and true_confidence is not None:
        candidates.append(("true", stoich_from_identifiers(complex_row["identifiers"]), true_confidence))
    return max(candidates, key=lambda candidate: candidate[2]) if candidates else None


pae_results = pl.read_csv(pae_results_csv, infer_schema_length=0)   # all columns as text
assert pae_results["complex_ac"].is_unique().all(), "more than one row per complex"

# n_proteins comes from the pdb mapping file (as in the 08 viewer script); the results csv has no such column
n_proteins_per_complex = pl.read_csv(pdb_mapping_csv).select("complex_ac", "n_proteins").unique()
assert n_proteins_per_complex["complex_ac"].is_unique().all(), "a complex has more than one n_proteins value"
missing_n_proteins = set(pae_results["complex_ac"]) - set(n_proteins_per_complex["complex_ac"])
assert not missing_n_proteins, f"{len(missing_n_proteins)} complexes have no n_proteins: {sorted(missing_n_proteins)}"
n_complexes_before = pae_results.height
pae_results = (
    pae_results.join(n_proteins_per_complex, on="complex_ac", how="inner")
    .filter(pl.col("n_proteins") > N_PROTEINS_ABOVE)
)
print(f"{pae_results.height}/{n_complexes_before} complexes with n_proteins > {N_PROTEINS_ABOVE}")
pae_pool_dirs = index_pool_folders(pae_run_dir)
trimmed_pool_dirs = index_pool_folders(trimmed_run_dir)

comparison_rows = []
complexes_without_pae_assembly = []
for complex_row in pae_results.iter_rows(named=True):
    best_stoich = best_stoich_of_complex(complex_row)
    if best_stoich is None:
        complexes_without_pae_assembly.append(complex_row["complex_ac"])
        continue
    source, stoich, csv_confidence = best_stoich

    assert stoich in pae_pool_dirs, f"{complex_row['complex_ac']}: no pae pool folder for {stoich_to_str(stoich)}"
    pae_assembly = best_assembly_in_pool_dir(pae_pool_dirs[stoich])
    assert pae_assembly is not None, f"{complex_row['complex_ac']}: csv has an assembly, {pae_pool_dirs[stoich]} not"
    assert abs(pae_assembly[1] - csv_confidence) < 0.01, \
        f"{complex_row['complex_ac']}: confidence.txt {pae_assembly[1]} != csv {csv_confidence}"

    trimmed_pool_dir = trimmed_pool_dirs.get(stoich)
    trimmed_assembly = best_assembly_in_pool_dir(trimmed_pool_dir) if trimmed_pool_dir else None
    comparison_rows.append({
        "complex_ac": complex_row["complex_ac"],
        "n_proteins": complex_row["n_proteins"],
        "stoichiometry": stoich_to_str(stoich),
        "stoichiometry_source": source,
        "pae_pdb": pae_assembly[0],
        "pae_cf_confidence": pae_assembly[1],
        "trimmed_pool_dir_exists": trimmed_pool_dir is not None,
        "trimmed_pdb": trimmed_assembly[0] if trimmed_assembly else None,
        "trimmed_cf_confidence": trimmed_assembly[1] if trimmed_assembly else None,
    })

print(f"{len(complexes_without_pae_assembly)} complexes without any pae assembly (not compared): "
      f"{complexes_without_pae_assembly}")
trimming_comparison = pl.DataFrame(comparison_rows)
no_trimmed_assembly = trimming_comparison.filter(pl.col("trimmed_pdb").is_null())
print(f"{no_trimmed_assembly.height} complexes without a trimmed assembly of the same stoichiometry (no US-align):")
print(no_trimmed_assembly.select("complex_ac", "stoichiometry", "trimmed_pool_dir_exists"))


# ---------------- US-align ----------------
USALIGN_COLUMNS = {   # -outfmt 2 column -> output column; structure 1 = pae, structure 2 = trimmed
    "TM1": "tm_score_norm_by_pae", "TM2": "tm_score_norm_by_trimmed", "RMSD": "rmsd",
    "L1": "n_residues_pae", "L2": "n_residues_trimmed", "Lali": "n_residues_aligned",
}


def run_usalign(pae_pdb, trimmed_pdb):
    """US-align of the full complexes (-mm 1 -ter 0, chains are matched by US-align); returns the metrics"""
    usalign_dir = Path(trimmed_pdb).parent.parent / USALIGN_DIR_NAME / Path(trimmed_pdb).stem
    stdout_file = usalign_dir / "usalign_stdout.txt"
    if not (SKIP_EXISTING and stdout_file.exists()):
        usalign_dir.mkdir(parents=True, exist_ok=SKIP_EXISTING)
        command = [str(usalign_bin), pae_pdb, trimmed_pdb, "-mm", "1", "-ter", "0", "-outfmt", "2"]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        (usalign_dir / "command.txt").write_text(" ".join(command) + "\n")
        stdout_file.write_text(process.stdout)

    header, values = [line for line in stdout_file.read_text().splitlines() if line.strip()]
    metrics = dict(zip(header.lstrip("#").split("\t"), values.split("\t")))
    assert metrics["PDBchain1"].startswith(pae_pdb) and metrics["PDBchain2"].startswith(trimmed_pdb), \
        f"{stdout_file} belongs to other structures: {metrics['PDBchain1']} vs {metrics['PDBchain2']}"
    return {"usalign_dir": str(usalign_dir)} | {
        out: int(metrics[col]) if col.startswith("L") else float(metrics[col]) for col, out in USALIGN_COLUMNS.items()
    }


# complexes with the same stoichiometry share pool folders, so each pdb pair is aligned once
pdb_pairs = (
    trimming_comparison.filter(pl.col("trimmed_pdb").is_not_null())
    .select("pae_pdb", "trimmed_pdb").unique().rows()
)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    usalign_metrics = list(tqdm(executor.map(lambda pair: run_usalign(*pair), pdb_pairs),
                                total=len(pdb_pairs), desc="US-align"))

usalign_table = pl.DataFrame([
    {"pae_pdb": pae_pdb, "trimmed_pdb": trimmed_pdb} | metrics
    for (pae_pdb, trimmed_pdb), metrics in zip(pdb_pairs, usalign_metrics)
])
trimming_comparison = (
    trimming_comparison.join(usalign_table, on=["pae_pdb", "trimmed_pdb"], how="left")
    .with_columns((pl.col("trimmed_cf_confidence") - pl.col("pae_cf_confidence")).alias("cf_confidence_change"))
    .sort("tm_score_norm_by_trimmed", nulls_last=True)
)
assert trimming_comparison.height == len(comparison_rows), "join changed the number of complexes"

with pl.Config(tbl_rows=30, tbl_cols=-1, fmt_str_lengths=60):
    print(trimming_comparison.drop("pae_pdb", "trimmed_pdb", "usalign_dir"))
trimming_comparison.write_csv(comparison_csv_path)
print(f"{trimming_comparison.height} complexes -> {comparison_csv_path}")