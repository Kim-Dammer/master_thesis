#!/usr/bin/env python3
"""
Generate a self-contained HTML viewer for CP complexes (trimmed CombFold run).

Per complex:
  - Left panel : CombFold assembled model; hover shows UniProt ID per chain
  - Right panel: PDB reference structures (precomputed set cover) or the
                 pairwise AF input models CombFold was given

CombFold model selection (deterministic, no name/score guessing):
  The specs actually submitted to CombFold for a complex (true stoichiometry +
  Stoic predictions) are read from the pipeline's expanded CSV. Each spec maps
  to exactly one output folder, named the same way s2 names it. From those
  runs the model with the highest score in confidence.txt is shown
  (STOICHIOMETRY_CHOICE = "best"), or only the true-spec run is used ("true").

Strictness:
  - Chain -> UniProt comes only from CombFold's chain.list (asserted complete).
  - A model that lacks chains of its spec (partial assembly) is shown, but the
    missing copies are listed in a red warning in the top bar.
  - Every input folder file must be either an AFM_ pair model or a REP_
    full-length representative (asserted).
  - Caps (MAX_PDB_REFS, MAX_INPUT_PAIRS) are always shown in the UI when hit.
"""
import base64
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm
from procompa import get_project_root

PRJ_ROOT = get_project_root()
DATA     = PRJ_ROOT / "data"

# ── CombFold run to visualize ─────────────────────────────────────────────────
SETUP_NAME   = "19_CP_pae_plddt_trimmed"
RUN_DIR      = DATA / "Pipeline" / SETUP_NAME
CF_BASE      = RUN_DIR / "CombFold"
EXPANDED_CSV = RUN_DIR / f"{SETUP_NAME}_expanded.csv"
CF_SOURCE    = "pool"     # must match --combfold-source of the run
STOICHIOMETRY_CHOICE = "best"   # "best": top score among true + all pred runs | "true": true spec run only
OUT          = DATA / "Pipeline/viewer/02_structure_viewer_visualize_plddt_trimmed.html"

# ── annotation / reference inputs ─────────────────────────────────────────────
INFO_CSV          = DATA / "CP_complexes_no_struct_coverage/confident_CF_complexes_with_pdb_match_info.csv"
PDB_HITS_PARQUET  = DATA / "CP_complexes_no_struct_coverage/complex_pdb_hits.parquet"
SET_COVER_PARQUET = DATA / "CP_complexes_no_struct_coverage/minimal_complex_pdb_set_cover_max_identity.parquet"
ANNOT_CSV         = DATA / "CP_complexes_no_struct_coverage/complex_pdb_annotations_map.csv"
SIFTS_CSV         = DATA / "pdb" / "pdb_chain_uniprot.csv"
MMSEQ_FILTERED    = DATA / "CP_complexes_no_struct_coverage/sanity_checks/mmseq_no_Strcut_filtered.parquet"
MMSEQ_RAW         = PRJ_ROOT / "scripts/mmseq_homology_match/mmseqs/mmseqs_run_max_sensitivity/results/mmseqs_new_identity_similarity_max_sensitivity.parquet"

EXCLUDED_COMPLEXES = {"CPX-1602"}
MAX_PDB_REFS    = 20   # cap on reference PDBs per complex (shown in UI when hit)
MAX_INPUT_PAIRS = 10   # cap on embedded pair models per complex (rank-1 models first; shown in UI when hit)
N_THREADS       = 16
GZIP_LEVEL      = 6    # level 9 is several times slower for ~1% smaller output

assert STOICHIOMETRY_CHOICE in ("best", "true"), STOICHIOMETRY_CHOICE
assert CF_SOURCE in ("pool", "pair"), CF_SOURCE
for required_path in (CF_BASE, EXPANDED_CSV, INFO_CSV, PDB_HITS_PARQUET, SET_COVER_PARQUET,
                      ANNOT_CSV, SIFTS_CSV, MMSEQ_FILTERED, MMSEQ_RAW):
    assert required_path.exists(), f"required input not found: {required_path}"


# ── helpers ───────────────────────────────────────────────────────────────────

_SPEC_TOKEN_RE  = re.compile(r"^([A-Za-z0-9_]+)\((\d+)\)$")
_CHAIN_LIST_RE  = re.compile(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pdb$")
_AFM_FILE_RE    = re.compile(r"^AFM_([A-Z][A-Z0-9]{5,})_([A-Z][A-Z0-9]{5,})_unrelaxed_rank_(\d+)_model_\d+\.pdb$")
_REP_FILE_RE    = re.compile(r"^REP_[A-Za-z0-9]+_full_length\.pdb$")
_PDB_KEEP_RECORDS = frozenset({"ATOM", "HETATM", "MODEL", "ENDMDL", "TER", "END"})


def uniprot_proteins(identifiers: str) -> set[str]:
    """UniProt accessions from the Complex Portal identifier string."""
    proteins = set()
    for part in identifiers.split("|"):
        part = part.strip()
        if part.startswith(("CHEBI", "URS")):
            continue
        accession_match = re.match(r"^([A-Z][A-Z0-9]{5,})", part)
        if accession_match:
            proteins.add(accession_match.group(1))
    return proteins


def parse_spec(spec: str) -> dict[str, int]:
    """'P1(1),P2(2)' -> {'P1': 1, 'P2': 2}; any unparseable token is an error."""
    protein_counts: dict[str, int] = {}
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        token_match = _SPEC_TOKEN_RE.match(token)
        assert token_match, f"unparseable spec token {token!r} in {spec!r}"
        assert token_match.group(1) not in protein_counts, f"protein listed twice in spec {spec!r}"
        protein_counts[token_match.group(1)] = int(token_match.group(2))
    assert protein_counts, f"empty spec: {spec!r}"
    return protein_counts


def output_folder_name(protein_counts: dict[str, int]) -> str:
    """Exactly how s2 names the output folder (incl. hash for names > 200 chars)."""
    complex_name = "_".join(f"{p}x{protein_counts[p]}" for p in sorted(protein_counts))
    if len(complex_name) > 200:
        complex_name = complex_name[:180] + "_" + hashlib.sha1(complex_name.encode()).hexdigest()[:12]
    return f"{complex_name}_{CF_SOURCE}_output"


def stoic_label(protein_counts: dict[str, int]) -> str:
    return "  |  ".join(f"{p}x{protein_counts[p]}" for p in sorted(protein_counts))


def parse_confidence(confidence_path: Path) -> list[dict]:
    """All '<model_path> <score>' lines, best first. Any malformed line is an error."""
    scored_models = []
    for line in confidence_path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        assert len(fields) == 2, f"malformed line in {confidence_path}: {line!r}"
        scored_models.append({"path": Path(fields[0]), "score": float(fields[1])})
    return sorted(scored_models, key=lambda model: model["score"], reverse=True)


def compress(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode(), compresslevel=GZIP_LEVEL)).decode()


def strip_pdb(text: str) -> str:
    """Keep only coordinate records (drops REMARK blocks, e.g. embedded PAE)."""
    return "\n".join(line for line in text.splitlines() if line[:6].strip() in _PDB_KEEP_RECORDS)


def pdb_chain_order(pdb_text: str) -> list[str]:
    """Chain IDs of ATOM records in order of first appearance."""
    chain_ids: list[str] = []
    seen_chain_ids: set[str] = set()
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and len(line) > 21 and line[21] not in seen_chain_ids:
            seen_chain_ids.add(line[21])
            chain_ids.append(line[21])
    return chain_ids


def read_chain_list(output_folder: str) -> dict[str, str]:
    """CombFold's own chain -> UniProt manifest (one 'UNIPROT_CHAIN.pdb' per line)."""
    chain_list_path = CF_BASE / output_folder / "_unified_representation" / "assembly_output" / "chain.list"
    assert chain_list_path.exists(), f"chain.list missing: {chain_list_path}"
    chain_to_protein: dict[str, str] = {}
    for line in chain_list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        chain_list_match = _CHAIN_LIST_RE.match(line)
        assert chain_list_match, f"unparseable chain.list line in {output_folder}: {line!r}"
        protein, chain = chain_list_match.group(1), chain_list_match.group(2)
        assert chain not in chain_to_protein, f"chain {chain} listed twice in {chain_list_path}"
        chain_to_protein[chain] = protein
    assert chain_to_protein, f"empty chain.list: {chain_list_path}"
    return chain_to_protein


def build_cf_chain_map(model_text: str, output_folder: str,
                       protein_counts: dict[str, int]) -> tuple[dict[str, str], dict[str, int]]:
    """Return (model chain -> UniProt, {protein: missing copies}) for an assembled model."""
    chain_to_protein = read_chain_list(output_folder)
    model_chains = pdb_chain_order(model_text)
    unknown_chains = [ch for ch in model_chains if ch not in chain_to_protein]
    assert not unknown_chains, f"{output_folder}: model chains {unknown_chains} not in chain.list"
    proteins_not_in_spec = set(chain_to_protein.values()) - set(protein_counts)
    assert not proteins_not_in_spec, f"{output_folder}: chain.list has proteins not in spec: {proteins_not_in_spec}"

    present_copies = Counter(chain_to_protein[ch] for ch in model_chains)
    excess_copies = {p: n for p, n in present_copies.items() if n > protein_counts[p]}
    assert not excess_copies, f"{output_folder}: more copies in model than in spec: {excess_copies}"
    missing_copies = {p: n - present_copies[p] for p, n in protein_counts.items() if present_copies[p] < n}
    return {ch: chain_to_protein[ch] for ch in model_chains}, missing_copies


def read_input_models(input_folder: str) -> tuple[list[dict], int]:
    """Pairwise AF models CombFold was given (pdbs/ of the input folder).

    Ordered by (rank, protein1, protein2) so the cap keeps the rank-1 model of
    as many pairs as possible. Returns (embedded models, total pair models)."""
    pdbs_dir = CF_BASE / input_folder / "pdbs"
    assert pdbs_dir.is_dir(), f"input pdbs dir missing: {pdbs_dir}"
    pair_model_files: list[tuple[int, str, str, Path]] = []
    for pdb_file in pdbs_dir.glob("*.pdb"):
        afm_match = _AFM_FILE_RE.match(pdb_file.name)
        if afm_match:
            pair_model_files.append((int(afm_match.group(3)), afm_match.group(1), afm_match.group(2), pdb_file))
            continue
        assert _REP_FILE_RE.match(pdb_file.name), f"unexpected file in {pdbs_dir}: {pdb_file.name}"
    assert pair_model_files, f"no AFM_ pair models in {pdbs_dir}"
    pair_model_files.sort()

    embedded_models = []
    for model_rank, protein1, protein2, pdb_file in pair_model_files[:MAX_INPUT_PAIRS]:
        stripped_text = strip_pdb(pdb_file.read_text())
        pair_chains = pdb_chain_order(stripped_text)
        assert len(pair_chains) == 2, f"{pdb_file}: expected 2 chains, got {pair_chains}"
        pair_name = f"{protein1} – {protein2}" if protein1 != protein2 else f"{protein1} (homo)"
        embedded_models.append({
            "filename" : pdb_file.name,
            "label"    : f"{pair_name}  #{model_rank}",
            "proteins" : [protein1, protein2],
            # s2 writes protein1 as the first chain, protein2 as the second
            "chain_map": {pair_chains[0]: protein1, pair_chains[1]: protein2},
            "pdb_gz"   : compress(stripped_text),
        })
    return embedded_models, len(pair_model_files)


def sniff_delimiter(path: Path) -> str:
    """Comma vs tab, from the first non-comment line."""
    with open(path, encoding="utf-8", errors="ignore") as file_handle:
        for line in file_handle:
            if line.strip() and not line.startswith("#"):
                return "\t" if line.count("\t") > line.count(",") else ","
    raise AssertionError(f"no data lines in {path}")


def assert_unique(table: pl.DataFrame, key_cols: list[str], what: str) -> None:
    duplicated_keys = table.filter(pl.struct(key_cols).is_duplicated())
    assert duplicated_keys.is_empty(), f"{what}: non-unique {key_cols}:\n{duplicated_keys}"


def split_hit_id(hit_col: str = "hit_pdb_id") -> list[pl.Expr]:
    """'1abc_A' -> hit_pdb_lower='1abc', hit_chain='A' (chain keeps any further '_')."""
    hit_parts = pl.col(hit_col).str.splitn("_", 2)
    return [hit_parts.struct.field("field_0").str.to_lowercase().alias("hit_pdb_lower"),
            hit_parts.struct.field("field_1").alias("hit_chain")]


# ── complexes ─────────────────────────────────────────────────────────────────

print("Loading complexes...")
complex_info = (
    pl.read_csv(INFO_CSV)
    .select(["complex_ac", "identifiers", "CF_confidence_max", "match_class"])
    .unique()
    .filter(~pl.col("complex_ac").is_in(list(EXCLUDED_COMPLEXES)))
    .sort("CF_confidence_max", descending=True)
)
assert_unique(complex_info, ["complex_ac"], str(INFO_CSV))
target_proteins_by_complex: dict[str, set[str]] = {
    info_row["complex_ac"]: uniprot_proteins(info_row["identifiers"])
    for info_row in complex_info.iter_rows(named=True)
}
complex_acs = list(target_proteins_by_complex)
all_target_proteins = set().union(*target_proteins_by_complex.values())
print(f"  {len(complex_acs)} complexes")

# ── submitted CombFold specs -> output folders ────────────────────────────────

print("Loading submitted CombFold specs...")
expanded_runs = pl.read_csv(EXPANDED_CSV, infer_schema_length=0)
required_expanded_cols = {"#Complex ac", "combfold_submission", "stoic_pred_rank", "is_true_spec"}
assert required_expanded_cols <= set(expanded_runs.columns), (
    f"{EXPANDED_CSV} missing columns: {required_expanded_cols - set(expanded_runs.columns)}"
)
# complex_ac -> output folder -> {"labels": [...], "protein_counts": {...}}
# (true spec and a correct prediction share a folder -> labels merged)
cf_runs_by_complex: dict[str, dict[str, dict]] = {}
n_rows_without_spec = 0
for run_row in expanded_runs.iter_rows(named=True):
    spec = run_row["combfold_submission"]
    if not spec:
        n_rows_without_spec += 1
        continue
    pred_rank = run_row["stoic_pred_rank"]
    if pred_rank:
        run_label = f"pred_{int(float(pred_rank))}"
    else:
        assert str(run_row["is_true_spec"]).lower() == "true", (
            f"row without stoic_pred_rank is not the true spec: {run_row}"
        )
        run_label = "true"
    protein_counts = parse_spec(spec)
    folder_runs = cf_runs_by_complex.setdefault(run_row["#Complex ac"], {})
    run_entry = folder_runs.setdefault(output_folder_name(protein_counts),
                                       {"labels": [], "protein_counts": protein_counts})
    run_entry["labels"].append(run_label)
print(f"  {len(cf_runs_by_complex)} complexes with submitted specs "
      f"({n_rows_without_spec} expanded rows without a spec)")
complexes_without_runs = [ac for ac in complex_acs if ac not in cf_runs_by_complex]
if complexes_without_runs:
    print(f"  WARNING {len(complexes_without_runs)} complexes have no submitted spec: "
          f"{complexes_without_runs}", file=sys.stderr)

# ── PDB set cover ─────────────────────────────────────────────────────────────

print("Loading PDB set cover...")
set_cover = pl.read_parquet(SET_COVER_PARQUET)
assert {"complex_ac", "cover_100_pdbs", "cover_75_pdbs"} <= set(set_cover.columns), set_cover.columns
assert_unique(set_cover, ["complex_ac"], str(SET_COVER_PARQUET))
pdb_cover_by_complex: dict[str, dict] = {}
for cover_row in set_cover.filter(pl.col("complex_ac").is_in(complex_acs)).iter_rows(named=True):
    if cover_row["cover_100_pdbs"]:
        cover_level, cover_pdbs = "100", list(cover_row["cover_100_pdbs"])
    elif cover_row["cover_75_pdbs"]:
        cover_level, cover_pdbs = "75", list(cover_row["cover_75_pdbs"])
    else:
        cover_level, cover_pdbs = None, []
    pdb_cover_by_complex[cover_row["complex_ac"]] = {
        "level": cover_level, "pdb_ids": cover_pdbs[:MAX_PDB_REFS], "n_total": len(cover_pdbs),
    }
needed_pdbs_lower = {
    pdb_id.lower() for cover in pdb_cover_by_complex.values() for pdb_id in cover["pdb_ids"]
}
print(f"  {len(pdb_cover_by_complex)} complexes with cover, {len(needed_pdbs_lower)} distinct PDBs")

# ── PDB -> CP proteins (complex_pdb_hits) ─────────────────────────────────────

complex_pdb_hits = (
    pl.read_parquet(PDB_HITS_PARQUET)
    .filter(pl.col("complex_ac").is_in(complex_acs) & pl.col("pdb_id").is_not_null())
    .select(["complex_ac", "pdb_id", "proteins"])
)
assert_unique(complex_pdb_hits, ["complex_ac", "pdb_id"], str(PDB_HITS_PARQUET))
pdb_proteins_by_complex: dict[str, dict[str, list[str]]] = {}
for hit_row in complex_pdb_hits.iter_rows(named=True):
    pdb_proteins_by_complex.setdefault(hit_row["complex_ac"], {})[hit_row["pdb_id"]] = list(hit_row["proteins"] or [])

# ── Complex Portal annotations ────────────────────────────────────────────────

print("Loading annotations...")
annotations = pl.read_csv(ANNOT_CSV)
assert {"complex_ac", "pdb_id", "cp_annotation", "pdb_annotation"} <= set(annotations.columns), annotations.columns
complex_annotations = (
    annotations.filter(pl.col("complex_ac").is_not_null() & pl.col("cp_annotation").is_not_null())
    .select(["complex_ac", "cp_annotation"]).unique()
)
assert_unique(complex_annotations, ["complex_ac"], f"{ANNOT_CSV} cp_annotation")
cp_annotation_by_complex = dict(complex_annotations.iter_rows())
pdb_annotations = (
    annotations
    .filter(pl.col("complex_ac").is_not_null() & pl.col("pdb_id").is_not_null()
            & pl.col("pdb_annotation").is_not_null())
    .select(["complex_ac", pl.col("pdb_id").str.to_lowercase().alias("pdb_lower"), "pdb_annotation"])
    .unique()
)
assert_unique(pdb_annotations, ["complex_ac", "pdb_lower"], f"{ANNOT_CSV} pdb_annotation")
pdb_annotation_by_pair = {(ac, pdb_lower): text for ac, pdb_lower, text in pdb_annotations.iter_rows()}
print(f"  {len(cp_annotation_by_complex)} complex annotations, {len(pdb_annotation_by_pair)} complex/PDB annotations")

# ── SIFTS (restricted to needed PDBs) ─────────────────────────────────────────

print("Loading SIFTS...")
sifts_rows = pl.read_csv(SIFTS_CSV, comment_prefix="#", infer_schema_length=0,
                         separator=sniff_delimiter(SIFTS_CSV))
assert {"PDB", "CHAIN", "SP_PRIMARY"} <= set(sifts_rows.columns), (
    f"SIFTS columns mismatch, got {sifts_rows.columns}"
)
sifts_rows = sifts_rows.filter(
    pl.col("PDB").str.to_lowercase().is_in(list(needed_pdbs_lower))
    & pl.col("CHAIN").is_not_null() & pl.col("SP_PRIMARY").is_not_null()
)
# pdb_lower -> chain -> all UniProts mapped to that chain (chimeric chains have several)
sifts_chain_uniprots: dict[str, dict[str, list[str]]] = {}
for pdb_id, chain, chain_uniprots in (
    sifts_rows.group_by(["PDB", "CHAIN"]).agg(pl.col("SP_PRIMARY").unique().sort()).iter_rows()
):
    sifts_chain_uniprots.setdefault(pdb_id.lower(), {})[chain] = chain_uniprots
# pdb_lower -> accession -> raw SIFTS rows (evidence popup)
sifts_evidence: dict[str, dict[str, list[dict]]] = {}
for sifts_row in sifts_rows.iter_rows(named=True):
    sifts_evidence.setdefault(sifts_row["PDB"].lower(), {}).setdefault(sifts_row["SP_PRIMARY"], []).append(sifts_row)
pdbs_without_sifts = needed_pdbs_lower - set(sifts_chain_uniprots)
print(f"  {len(sifts_chain_uniprots)}/{len(needed_pdbs_lower)} needed PDBs in SIFTS"
      + (f"; missing: {sorted(pdbs_without_sifts)}" if pdbs_without_sifts else ""))

# ── MMseq homology (restricted to needed PDBs) ────────────────────────────────

print("Loading MMseq homology hits...")
homology_hits = (
    pl.read_parquet(MMSEQ_FILTERED, columns=["protein_id", "hit_pdb_id"])
    .with_columns(split_hit_id())
)
assert homology_hits["hit_chain"].null_count() == 0, "hit_pdb_id without '_<chain>' in filtered MMseq parquet"
# pdb_lower -> chain -> CP proteins with a homology hit on that chain
homology_chain_proteins: dict[str, dict[str, list[str]]] = {}
for pdb_lower, chain, hit_proteins in (
    homology_hits.filter(pl.col("hit_pdb_lower").is_in(list(needed_pdbs_lower)))
    .group_by(["hit_pdb_lower", "hit_chain"]).agg(pl.col("protein_id").unique().sort())
    .iter_rows()
):
    homology_chain_proteins.setdefault(pdb_lower, {})[chain] = hit_proteins

# Evidence rows: same filter as the filtered parquet (identity > 30 OR
# blast_identity > 30, alnlen > 30) applied to the raw parquet, all columns kept.
mmseq_evidence_rows = (
    pl.scan_parquet(MMSEQ_RAW)
    .filter(pl.col("protein_id").is_in(list(all_target_proteins)))
    .filter((pl.col("identity_percent") > 30) | (pl.col("blast_identity_percent") > 30))
    .filter(pl.col("alnlen") > 30)
    .with_columns(split_hit_id())
    .filter(pl.col("hit_pdb_lower").is_in(list(needed_pdbs_lower)))
    .collect()
)
assert mmseq_evidence_rows["hit_chain"].null_count() == 0, "hit_pdb_id without '_<chain>' in raw MMseq parquet"
# (protein, pdb_lower) -> raw rows
mmseq_evidence: dict[tuple[str, str], list[dict]] = {}
for mmseq_row in mmseq_evidence_rows.drop(["hit_pdb_lower", "hit_chain"]).iter_rows(named=True):
    hit_pdb_lower = mmseq_row["hit_pdb_id"].split("_", 1)[0].lower()
    mmseq_evidence.setdefault((mmseq_row["protein_id"], hit_pdb_lower), []).append(mmseq_row)
print(f"  {len(homology_chain_proteins)} PDBs with homology hits, "
      f"{len(mmseq_evidence)} protein/PDB evidence groups")


# ── per complex ───────────────────────────────────────────────────────────────

def select_cf_run(ac: str, log: list[str]) -> tuple[str, list[dict], dict] | None:
    """Return (output_folder, scored models, run entry) of the chosen run, or None."""
    folder_runs = cf_runs_by_complex.get(ac, {})
    if STOICHIOMETRY_CHOICE == "true":
        folder_runs = {folder: run for folder, run in folder_runs.items() if "true" in run["labels"]}
    chosen_run = None
    for output_folder, run_entry in folder_runs.items():
        run_label = "/".join(run_entry["labels"])
        confidence_path = CF_BASE / output_folder / "assembled_results" / "confidence.txt"
        if not confidence_path.exists():
            log.append(f"  NOTE run {run_label}: no confidence.txt ({output_folder})")
            continue
        scored_models = parse_confidence(confidence_path)
        if not scored_models:
            log.append(f"  NOTE run {run_label}: empty confidence.txt ({output_folder})")
            continue
        log.append(f"  run {run_label}: top score {scored_models[0]['score']:.2f}")
        if chosen_run is None or scored_models[0]["score"] > chosen_run[1][0]["score"]:
            chosen_run = (output_folder, scored_models, run_entry)
    return chosen_run


def process_complex(info_row: dict) -> tuple[str, dict]:
    ac              = info_row["complex_ac"]
    target_proteins = target_proteins_by_complex[ac]
    log = [f"\n{ac}  ref CF={info_row['CF_confidence_max']:.1f}  ({len(target_proteins)} proteins)"]

    # CombFold model
    cf_models, cf_chain_map, missing_copies, run_labels, run_stoic_label = [], {}, {}, [], ""
    chosen_run = select_cf_run(ac, log)
    if chosen_run is None:
        log.append(f"  WARNING {ac}: no CombFold run with results "
                   f"(STOICHIOMETRY_CHOICE={STOICHIOMETRY_CHOICE}) -- no CF model shown")
    else:
        output_folder, scored_models, run_entry = chosen_run
        top_model_path = scored_models[0]["path"]
        assembled_dir = CF_BASE / output_folder / "assembled_results"
        assert top_model_path.parent.resolve() == assembled_dir.resolve(), (
            f"{ac}: confidence.txt points outside its folder: {top_model_path}"
        )
        assert top_model_path.exists(), f"{ac}: top model missing: {top_model_path}"
        model_text = strip_pdb(top_model_path.read_text())
        cf_chain_map, missing_copies = build_cf_chain_map(model_text, output_folder, run_entry["protein_counts"])
        cf_models = [{"name": top_model_path.name, "score": scored_models[0]["score"],
                      "pdb_gz": compress(model_text), "folder": output_folder, "path": str(top_model_path)}]
        run_labels = run_entry["labels"]
        run_stoic_label = stoic_label(run_entry["protein_counts"])
        log.append(f"  CF model: {scored_models[0]['score']:.2f} from {'/'.join(run_labels)}")
        if missing_copies:
            log.append(f"  WARNING {ac}: partial assembly, missing copies {missing_copies}")
        spec_proteins = set(run_entry["protein_counts"])
        if spec_proteins != target_proteins:
            log.append(f"  NOTE {ac}: spec proteins differ from CP identifiers -- "
                       f"only in spec {sorted(spec_proteins - target_proteins)}, "
                       f"only in CP {sorted(target_proteins - spec_proteins)}")

    # input pair models of the chosen run
    input_models, n_input_models_total = [], 0
    if chosen_run is not None:
        input_folder = chosen_run[0].removesuffix("_output") + "_input"
        input_models, n_input_models_total = read_input_models(input_folder)
        input_kb = sum(len(m["pdb_gz"]) for m in input_models) // 1024
        log.append(f"  input pair models: {len(input_models)}/{n_input_models_total} embedded ({input_kb} KB)")

    # reference PDBs
    cover = pdb_cover_by_complex.get(ac, {"level": None, "pdb_ids": [], "n_total": 0})
    cover_pdbs = cover["pdb_ids"]
    if not cover_pdbs:
        log.append(f"  NOTE {ac}: no PDB set cover")
    elif cover["n_total"] > MAX_PDB_REFS:
        log.append(f"  WARNING {ac}: PDB cover has {cover['n_total']} entries, showing {MAX_PDB_REFS}")
    pdb_chain_map = {
        pdb_id: {chain: {"u": chain_uniprots, "cp": [u for u in chain_uniprots if u in target_proteins]}
                 for chain, chain_uniprots in sifts_chain_uniprots.get(pdb_id.lower(), {}).items()}
        for pdb_id in cover_pdbs
    }
    if cover_pdbs:
        assert any(pdb_chain_map.values()), f"{ac}: none of the cover PDBs {cover_pdbs} is in SIFTS"
    pdb_homology_map = {
        pdb_id: {chain: [p for p in hit_proteins if p in target_proteins]
                 for chain, hit_proteins in homology_chain_proteins.get(pdb_id.lower(), {}).items()}
        for pdb_id in cover_pdbs
    }
    pdb_protein_evidence: dict[str, dict[str, dict]] = {}
    for pdb_id in cover_pdbs:
        pdb_lower = pdb_id.lower()
        per_protein_evidence = {}
        for protein in sorted(target_proteins):
            direct_rows   = sifts_evidence.get(pdb_lower, {}).get(protein, [])
            homology_rows = mmseq_evidence.get((protein, pdb_lower), [])
            if direct_rows or homology_rows:
                per_protein_evidence[protein] = {"direct": direct_rows, "homology": homology_rows}
        pdb_protein_evidence[pdb_id] = per_protein_evidence
    complex_pdb_proteins = pdb_proteins_by_complex.get(ac, {})

    entry = {
        "complex_ac"           : ac,
        "identifiers"          : info_row["identifiers"],
        "cf_confidence"        : cf_models[0]["score"] if cf_models else None,
        "cf_confidence_ref"    : float(info_row["CF_confidence_max"]),
        "match_class"          : info_row["match_class"] or "unknown",
        "stoic_label"          : run_stoic_label,
        "run_labels"           : run_labels,
        "assembly_missing"     : missing_copies,
        "cf_models"            : cf_models,
        "cf_chain_map"         : cf_chain_map,
        "cp_proteins"          : sorted(target_proteins),
        "cp_annotation"        : cp_annotation_by_complex.get(ac),
        "pdb_ids"              : cover_pdbs,
        "pdb_cover_level"      : cover["level"],
        "pdb_n_total"          : cover["n_total"],
        "pdb_cap_hit"          : cover["n_total"] > MAX_PDB_REFS,
        "pdb_proteins"         : {pdb_id: complex_pdb_proteins.get(pdb_id, []) for pdb_id in cover_pdbs},
        "pdb_annotation"       : {pdb_id: pdb_annotation_by_pair.get((ac, pdb_id.lower())) for pdb_id in cover_pdbs},
        "pdb_chain_map"        : pdb_chain_map,
        "pdb_homology_map"     : pdb_homology_map,
        "pdb_protein_evidence" : pdb_protein_evidence,
        "input_models"         : input_models,
        "input_n_total"        : n_input_models_total,
        "input_cap_hit"        : n_input_models_total > len(input_models),
    }
    tqdm.write("\n".join(log))
    return ac, entry


# File reads + gzip release the GIL, complexes are independent -> threads.
complex_entries: dict[str, dict] = {}
with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    for ac, entry in tqdm(executor.map(process_complex, complex_info.iter_rows(named=True)),
                          total=complex_info.height, desc="Assembling complexes"):
        complex_entries[ac] = entry

# Dropdown order: trimmed CF score, best first; complexes without a model last.
complex_entries = dict(sorted(
    complex_entries.items(),
    key=lambda item: (item[1]["cf_confidence"] is None, -(item[1]["cf_confidence"] or 0.0)),
))

complexes_without_model = [ac for ac, entry in complex_entries.items() if not entry["cf_models"]]
complexes_partial = {ac: entry["assembly_missing"] for ac, entry in complex_entries.items()
                     if entry["assembly_missing"]}
print(f"\n{len(complexes_without_model)}/{len(complex_entries)} complexes without CF model: "
      f"{complexes_without_model}", file=sys.stderr)
print(f"{len(complexes_partial)}/{len(complex_entries)} complexes with partial assembly: "
      f"{complexes_partial}", file=sys.stderr)


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CP Complex Structure Viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body {
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:13px; color:rgb(36,36,36);
  background:#f0f0f0; padding:10px;
  height:100vh; display:flex; flex-direction:column; gap:8px;
}
.topbar {
  background:white; border:1px solid #ddd; border-radius:6px;
  padding:9px 14px; display:flex; align-items:center; gap:12px;
  flex-shrink:0; flex-wrap:wrap;
}
.topbar label { font-weight:600; white-space:nowrap; }
#complex-select {
  flex:1; min-width:220px; max-width:560px;
  padding:4px 8px; border:1px solid #ccc; border-radius:4px; font-size:13px;
}
.badge {
  display:inline-block; padding:2px 10px; border-radius:10px;
  font-size:11px; font-weight:600; white-space:nowrap;
}
.badge-cf    { background:#ddeeff; color:#004a80; border:1px solid #aaccee; }
.badge-match { background:#fff0d5; color:#6b3d00; border:1px solid #e8c97a; }
#stoic-label {
  font-size:11px; color:#aaa; font-family:monospace;
  max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.path-label {
  position:absolute; left:8px; bottom:6px; z-index:5; pointer-events:none;
  font-size:10px; color:#777; font-family:monospace;
  background:rgba(255,255,255,.85); padding:2px 6px; border-radius:4px;
  max-width:75%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.mapped-summary {
  flex-basis:100%; font-size:11px; font-family:monospace;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.mapped-summary.right { text-align:right; }
.mapped-summary .m-blue   { color:#0072B2; font-weight:600; }
.mapped-summary .m-off    { color:#999;    font-weight:600; }
.cp-annotation {
  flex-basis:100%; font-size:12px; font-weight:600; color:#333;
  white-space:normal; line-height:1.35;
}
.pdb-annotation {
  flex-basis:100%; font-size:11px; color:#666; font-style:italic;
  white-space:normal; line-height:1.3;
}
.cap-warning {
  font-size:11px; font-weight:600; color:#a83232;
  background:#fde3e3; border:1px solid #f0b3b3;
  border-radius:4px; padding:1px 8px; display:none;
  max-width:420px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.cap-warning.on { display:inline-block; }
.cover-level { font-size:11px; color:#777; white-space:nowrap; }
.color-ctrl {
  margin-left:auto; display:flex; align-items:center;
  gap:6px; font-weight:normal; white-space:nowrap; font-size:12px;
}
.color-ctrl select {
  padding:2px 6px; border:1px solid #ccc; border-radius:4px; font-size:12px;
}
.plddt-ctrl {
  display:flex; align-items:center; gap:6px; font-weight:normal;
  white-space:nowrap; font-size:12px;
}
.plddt-ctrl input[type=range] { width:100px; vertical-align:middle; }
.plddt-ctrl input[type=number] {
  width:46px; font-family:monospace; padding:1px 3px;
  border:1px solid #ccc; border-radius:3px; font-size:12px;
}
.plddt-ctrl input:disabled { opacity:.4; }
.plddt-mode-label { display:flex; align-items:center; gap:3px; cursor:pointer; }
.plddt-mode-label input[type=radio] { margin:0; cursor:pointer; }
.plddt-legend { display:flex; align-items:center; gap:4px; font-size:11px; color:#666; }
.plddt-legend .sw { display:inline-block; width:9px; height:9px; border-radius:2px; }
.panels { display:grid; grid-template-columns:1fr 1fr; gap:8px; flex:1; min-height:0; }
.panel {
  background:white; border:1px solid #ddd; border-radius:6px;
  display:flex; flex-direction:column; overflow:hidden;
}
.panel-hdr {
  padding:7px 12px; border-bottom:1px solid #eee;
  display:flex; align-items:center; gap:8px;
  flex-shrink:0; flex-wrap:wrap; min-height:40px;
}
.panel-hdr h3 { font-size:13px; font-weight:600; }
.panel-body { flex:1; position:relative; min-height:0; }
.viewer3d   { width:100%; height:100%; }
.overlay {
  position:absolute; inset:0; background:rgba(255,255,255,.82);
  display:none; align-items:center; justify-content:center;
  font-size:13px; color:#555; text-align:center; padding:20px;
}
.overlay.on { display:flex; }
.navbtn {
  background:#f2f2f2; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:1px 9px; font-size:13px; line-height:1.6;
}
.navbtn:hover:not(:disabled) { background:#e4e4e4; }
.navbtn:disabled { opacity:.35; cursor:default; }
#scr-page {
  position:fixed; top:10px; right:14px; z-index:100;
  background:#0072B2; color:white; border:none; border-radius:5px;
  padding:4px 12px; font-size:12px; font-weight:600; cursor:pointer;
  box-shadow:0 1px 4px rgba(0,0,0,.25);
}
#scr-page:hover { background:#005a8e; }
#scr-page.busy  { opacity:.6; cursor:wait; }
#model-info { font-size:12px; color:#555; white-space:nowrap; }
.pdb-list   { display:flex; gap:5px; flex-wrap:wrap; }
.pdb-btn {
  background:#f2f2f2; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:2px 10px;
  font-size:12px; font-family:monospace; font-weight:600; letter-spacing:.5px;
}
.pdb-btn:hover { background:#e4e4e4; }
.pdb-btn.active { background:#0072B2; color:white; border-color:#0072B2; }
.pdb-btn.cached { border-color:#009E73; }
.right-mode-tabs { display:flex; gap:3px; align-items:center; flex-shrink:0; }
.mode-tab {
  background:#f0f0f0; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:2px 9px; font-size:12px; font-weight:600; line-height:1.6;
  white-space:nowrap;
}
.mode-tab:hover:not(.active):not(:disabled) { background:#e4e4e4; }
.mode-tab.active  { background:#0072B2; color:white; border-color:#0072B2; }
.mode-tab:disabled { opacity:.38; cursor:not-allowed; }
.input-list { display:flex; gap:5px; flex-wrap:wrap; }
.input-btn {
  background:#f2f2f2; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:2px 10px;
  font-size:12px; font-family:monospace; font-weight:600;
  max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.input-btn:hover { background:#e4e4e4; }
.input-btn.active { background:#009E73; color:white; border-color:#009E73; }
.protein-link { cursor:pointer; text-decoration:underline dotted; }
.protein-link:hover { opacity:.7; }
.evidence-backdrop { position:fixed; inset:0; background:rgba(0,0,0,.35); display:none; z-index:50; }
.evidence-backdrop.on { display:block; }
.evidence-popup {
  position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
  background:white; border-radius:8px; box-shadow:0 4px 24px rgba(0,0,0,.25);
  width:min(640px,90vw); max-height:80vh; overflow:auto; z-index:51; display:none;
}
.evidence-popup.on { display:block; }
.evidence-popup-hdr {
  display:flex; justify-content:space-between; align-items:center;
  padding:10px 14px; border-bottom:1px solid #eee; font-weight:600; font-size:14px;
  position:sticky; top:0; background:white;
}
.evidence-popup-body { padding:12px 14px; font-size:12px; }
.evidence-section { margin-bottom:14px; }
.evidence-section h4 { font-size:12px; margin-bottom:6px; color:#555; }
.evidence-table { width:100%; border-collapse:collapse; font-family:monospace; font-size:11px; }
.evidence-table th, .evidence-table td { text-align:left; padding:3px 6px; border-bottom:1px solid #f0f0f0; }
.evidence-empty { color:#999; font-style:italic; }
</style>
</head>
<body>

<div class="topbar">
  <label>Complex:</label>
  <select id="complex-select"></select>
  <span class="badge badge-cf"    id="badge-cf"></span>
  <span class="badge badge-match" id="badge-match"></span>
  <span id="stoic-label" title=""></span>
  <span id="assembly-warning" class="cap-warning"></span>
  <div class="color-ctrl">
    Color:
    <select id="color-mode">
      <option value="single"    >Single (yellow)</option>
      <option value="by-chain"  >By chain</option>
      <option value="by-mapping">By mapping</option>
      <option value="plddt"     >By pLDDT</option>
    </select>
  </div>
  <div class="plddt-legend" id="plddt-legend" style="display:none">
    <span class="sw" style="background:#0053D6"></span>&gt;90
    <span class="sw" style="background:#65CBF3"></span>70-90
    <span class="sw" style="background:#FFDB13"></span>50-70
    <span class="sw" style="background:#FF7D45"></span>&lt;50
  </div>
  <div class="plddt-ctrl">
    <label class="plddt-mode-label">
      <input type="radio" name="plddt-mode" value="threshold" id="plddt-mode-threshold" checked>
      pLDDT &ge;
    </label>
    <input type="range" id="plddt-threshold" min="0" max="100" step="1" value="0">
    <input type="number" id="plddt-threshold-num" min="0" max="100" step="1" value="0">
    <label class="plddt-mode-label">
      <input type="radio" name="plddt-mode" value="top50" id="plddt-mode-top50">
      Top 50% per chain
    </label>
    <span style="color:#999">(AF models only)</span>
  </div>
</div>

<div class="panels">

  <div class="panel">
    <div class="panel-hdr">
      <h3>CombFold Assembly</h3>
      <button class="navbtn" id="prev-m">&#9664;</button>
      <span id="model-info">-</span>
      <button class="navbtn" id="next-m">&#9654;</button>
      <span id="cf-mapped-summary" class="mapped-summary"></span>
      <span id="cf-annotation" class="cp-annotation"></span>
    </div>
    <div class="panel-body">
      <div id="cf-viewer"  class="viewer3d"></div>
      <div id="cf-overlay" class="overlay">Loading...</div>
      <span id="cf-path" class="path-label" title=""></span>
    </div>
  </div>

  <div class="panel">
    <div class="panel-hdr">
      <div class="right-mode-tabs">
        <button class="mode-tab active" id="tab-pdb"   onclick="setRightMode('pdb')">PDB Ref</button>
        <button class="mode-tab"        id="tab-input" onclick="setRightMode('input')">Input Pairs</button>
      </div>
      <div class="pdb-list"   id="pdb-list"></div>
      <div class="input-list" id="input-list" style="display:none"></div>
      <span id="pdb-cover-level" class="cover-level"></span>
      <span id="pdb-cap-warning" class="cap-warning"></span>
      <span id="input-cap-warning" class="cap-warning" style="display:none"></span>
      <span id="pdb-mapped-summary" class="mapped-summary right"></span>
      <span id="pdb-annotation" class="pdb-annotation"></span>
    </div>
    <div class="panel-body">
      <div id="pdb-viewer"  class="viewer3d"></div>
      <div id="pdb-overlay" class="overlay">Loading...</div>
    </div>
  </div>

</div>

<button id="scr-page" title="Screenshot entire page">&#8595; PNG</button>
<div id="evidence-popup-backdrop" class="evidence-backdrop"></div>
<div id="evidence-popup" class="evidence-popup">
  <div class="evidence-popup-hdr">
    <span id="evidence-popup-title"></span>
    <button id="evidence-popup-close" class="navbtn">&times;</button>
  </div>
  <div id="evidence-popup-body" class="evidence-popup-body"></div>
</div>

<script>
const COMPLEXES = %%JSON%%;

/* Okabe-Ito palette */
const CC = ['#0072B2','#E69F00','#009E73','#CC79A7',
            '#56B4E9','#D55E00','#F0E442','#999999'];
const YELLOW = '#E69F00';
const GRAY   = '#bbbbbb';

const LABEL_STYLE = {
  backgroundColor: 'black', backgroundOpacity: 0.75,
  fontColor: 'white', fontSize: 12, padding: 4, inFront: true,
};

/* ── gzip decompress ──────────────────────────────────────────────────── */
async function ungzip(b64) {
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).text();
}

/* ── state ────────────────────────────────────────────────────────────── */
let cfV = null, pdbV = null, cur = null, mIdx = 0;
let colorMode      = 'single';
let plddtThreshold = 0;            // AF models only
let plddtMode      = 'threshold';  // 'threshold' | 'top50'
let cfRaw = null, pdbRaw = null, pdbFmtCur = 'pdb';
let currentPdbId = null;

let rightMode       = 'pdb';   // 'pdb' | 'input'
let currentInputIdx = -1;
let inputChainMap   = {};      // chain -> UniProt of the displayed input model (from Python)

/* Reference PDB cache for the CURRENT complex only; pdbCacheOwner guards
   against late fetches from a previously selected complex. */
let pdbCache      = {};
let pdbCacheOwner = null;

/* ── pLDDT (AF DB 4-bin palette); AF models only, never crystal PDBs ──── */
function plddtColor(atom) {
  const b = atom.b;
  if (b == null) return GRAY;
  if (b > 90)    return '#0053D6';
  if (b > 70)    return '#65CBF3';
  if (b > 50)    return '#FFDB13';
  return '#FF7D45';
}

/* Must run AFTER setStyle: clears style on atoms below the active cutoff. */
function applyPlddtFilter(v) {
  if (plddtMode === 'top50') applyTop50Filter(v);
  else                       applyThresholdFilter(v, plddtThreshold);
}

function applyThresholdFilter(v, threshold) {
  if (!threshold) return;
  const below = v.selectedAtoms({}).filter(a => a.b != null && a.b < threshold);
  if (below.length) v.setStyle({serial: below.map(a => a.serial)}, {});
}

/* Per chain: hide atoms below that chain's own median pLDDT. */
function applyTop50Filter(v) {
  const atoms = v.selectedAtoms({}).filter(a => a.b != null);
  if (!atoms.length) return;
  const byChain = {};
  atoms.forEach(a => (byChain[a.chain] ||= []).push(a.b));
  const chainMedian = {};
  for (const [ch, vals] of Object.entries(byChain)) {
    vals.sort((x, y) => x - y);
    const mid = vals.length >> 1;
    chainMedian[ch] = (vals.length % 2) ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  }
  const below = atoms.filter(a => a.b < chainMedian[a.chain]);
  if (below.length) v.setStyle({serial: below.map(a => a.serial)}, {});
}

/* ── right-panel mode: PDB Ref <-> Input Pairs ────────────────────────── */
function setRightMode(mode) {
  rightMode = mode;
  const isPdb = mode === 'pdb';
  document.getElementById('tab-pdb').classList.toggle('active',  isPdb);
  document.getElementById('tab-input').classList.toggle('active', !isPdb);
  document.getElementById('pdb-list').style.display          = isPdb ? '' : 'none';
  document.getElementById('pdb-cover-level').style.display   = isPdb ? '' : 'none';
  document.getElementById('pdb-cap-warning').style.display   = isPdb ? '' : 'none';
  document.getElementById('input-list').style.display        = isPdb ? 'none' : '';
  document.getElementById('input-cap-warning').style.display = isPdb ? 'none' : '';
  document.getElementById('pdb-mapped-summary').innerHTML = '';

  if (isPdb) {
    if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
    if (currentPdbId && pdbCache[currentPdbId]) {
      const {text, fmt} = pdbCache[currentPdbId];
      styleViewer(pdbV, text, fmt, 'pdb');
    } else if (currentPdbId) {
      loadPDB(currentPdbId);
    } else {
      pdbV.removeAllModels(); pdbV.render();
    }
    updateAnnotations();
    updateMappedSummaries();
  } else {
    const models = cur?.input_models || [];
    buildInputList(models);
    if (models.length > 0) {
      loadInputModel(0);
    } else {
      pdbV.removeAllModels(); pdbV.render();
    }
    updateAnnotations();
  }
}

function buildInputList(models) {
  const list = document.getElementById('input-list');
  list.innerHTML = '';
  models.forEach((m, i) => {
    const b = document.createElement('button');
    b.className   = 'input-btn';
    b.textContent = m.label;
    b.title       = m.filename;
    b.onclick     = () => loadInputModel(i);
    list.appendChild(b);
  });
}

async function loadInputModel(idx) {
  const models = cur?.input_models || [];
  if (!cur || idx < 0 || idx >= models.length) return;
  currentInputIdx = idx;
  document.querySelectorAll('.input-btn')
    .forEach((b, i) => b.classList.toggle('active', i === idx));

  const m = models[idx];
  spin('pdb-overlay', true, 'Loading input pair...');
  try {
    const text = await ungzip(m.pdb_gz);
    if (cur?.input_models?.[currentInputIdx] !== m) return;  // user moved on meanwhile
    inputChainMap = m.chain_map;
    styleInputViewer(text, m.proteins);
    spin('pdb-overlay', false);
    if (cfRaw) highlightCFPair(m.proteins);
    updateAnnotations();
    updateMappedSummaries();
  } catch (e) {
    console.error(e);
    spin('pdb-overlay', true, 'Failed to load: ' + e.message);
  }
}

/* Input pair: protein1 chain -> blue, protein2 chain -> green (homodimer: both blue). */
function styleInputViewer(text, proteins) {
  pdbV.removeAllModels();
  const model  = pdbV.addModel(text, 'pdb');
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))];
  const homo   = proteins[0] === proteins[1];

  chains.forEach(ch => {
    if (colorMode === 'plddt') {
      pdbV.setStyle({chain: ch}, {cartoon: {colorfunc: plddtColor}});
      return;
    }
    const color = (homo || inputChainMap[ch] === proteins[0]) ? CC[0] : CC[2];
    pdbV.setStyle({chain: ch}, {cartoon: {color}});
  });
  applyPlddtFilter(pdbV);

  pdbV.setHoverable({}, true,
    (atom, v) => {
      const u = inputChainMap[atom.chain];
      const plddt = atom.b != null ? '  |  pLDDT ' + atom.b.toFixed(1) : '';
      v.removeAllLabels();
      v.addLabel('Chain ' + atom.chain + ': ' + u + plddt, {...LABEL_STYLE, position: atom});
      v.render();
    },
    (atom, v) => { v.removeAllLabels(); v.render(); }
  );
  pdbV.zoomTo(); pdbV.render();
}

/* CF assembly in input mode: pair proteins in the same colors as the right
   panel (blue / green), all other chains yellow. */
function highlightCFPair(proteins) {
  if (!cfRaw) return;
  const homo = proteins[0] === proteins[1];
  cfV.removeAllModels();
  const model  = cfV.addModel(cfRaw, 'pdb');
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))];
  chains.forEach(ch => {
    const u = cur.cf_chain_map[ch];
    let color = YELLOW;
    if (u === proteins[0])               color = CC[0];
    else if (!homo && u === proteins[1]) color = CC[2];
    cfV.setStyle({chain: ch}, {cartoon: {color}});
  });
  applyPlddtFilter(cfV);
  cfV.setHoverable({}, true, cfHoverCB, cfUnhoverCB);
  cfV.zoomTo(); cfV.render();
  updateMappedSummaries();
}

/* ── hover callbacks (re-registered after every addModel) ─────────────── */
function cfHoverCB(atom, viewer) {
  const u = cur?.cf_chain_map?.[atom.chain];
  const plddt = atom.b != null ? '  |  pLDDT ' + atom.b.toFixed(1) : '';
  viewer.removeAllLabels();
  viewer.addLabel('Chain ' + atom.chain + ': ' + u + plddt, {...LABEL_STYLE, position: atom});
  viewer.render();
}
function cfUnhoverCB(atom, viewer) { viewer.removeAllLabels(); viewer.render(); }

function pdbHoverCB(atom, viewer) {
  const info     = cur?.pdb_chain_map?.[currentPdbId]?.[atom.chain];
  const homology = cur?.pdb_homology_map?.[currentPdbId]?.[atom.chain] || [];
  let text = 'Chain ' + atom.chain;
  if (info) {
    text += ' | SIFTS: ' + info.u.join(', ')
          + (info.cp.length ? ' (in complex: ' + info.cp.join(', ') + ')' : ' (not in complex)');
  }
  if (homology.length) text += ' | homology: ' + homology.join(', ');
  if (!info && !homology.length) text += ' (no mapping)';
  if (atom.b != null) text += '  |  B-factor ' + atom.b.toFixed(1);  /* crystal B-factor, not pLDDT */
  viewer.removeAllLabels();
  viewer.addLabel(text, {...LABEL_STYLE, position: atom});
  viewer.render();
}
function pdbUnhoverCB(atom, viewer) { viewer.removeAllLabels(); viewer.render(); }

function initViewers() {
  const opts = {backgroundColor: 'white', antialias: true};
  cfV  = $3Dmol.createViewer(document.getElementById('cf-viewer'),  opts);
  pdbV = $3Dmol.createViewer(document.getElementById('pdb-viewer'), opts);
}

/* ── coloring ─────────────────────────────────────────────────────────── */
function styleViewer(v, text, fmt, storeAs) {
  if (storeAs === 'cf')  cfRaw = text;
  if (storeAs === 'pdb') { pdbRaw = text; pdbFmtCur = fmt; }

  v.removeAllModels();
  const model  = v.addModel(text, fmt);
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();

  if (storeAs === 'cf') v.setHoverable({}, true, cfHoverCB, cfUnhoverCB);
  else                  v.setHoverable({}, true, pdbHoverCB, pdbUnhoverCB);

  chains.forEach((ch, i) => {
    if (colorMode === 'plddt') {
      v.setStyle({chain: ch}, {cartoon: {colorfunc: plddtColor}});
      return;
    }
    let color;
    if (colorMode === 'by-chain') {
      color = CC[i % CC.length];
    } else if (colorMode === 'single') {
      color = YELLOW;
    } else if (storeAs === 'cf') {
      /* by-mapping, CF: blue if the chain's protein is in the shown PDB, else yellow */
      const pdbProts = cur?.pdb_proteins?.[currentPdbId] || [];
      color = pdbProts.includes(cur.cf_chain_map[ch]) ? CC[0] : YELLOW;
    } else {
      /* by-mapping, PDB: blue if direct SIFTS match OR homology hit to a CP protein */
      const chainInfo   = cur?.pdb_chain_map?.[currentPdbId]?.[ch];
      const homologyHit = cur?.pdb_homology_map?.[currentPdbId]?.[ch] || [];
      color = ((chainInfo && chainInfo.cp.length) || homologyHit.length) ? CC[0] : GRAY;
    }
    v.setStyle({chain: ch}, {cartoon: {color}});
  });

  if (storeAs === 'cf') applyPlddtFilter(v);  /* AF-derived only */
  v.zoomTo(); v.render();
  updateMappedSummaries();
}

/* ── protein click-to-inspect popup ───────────────────────────────────── */
function renderGroup(el, groups) {
  el.innerHTML = '';
  let wroteAny = false;
  groups.forEach(([cls, label, list]) => {
    if (!list.length) return;
    if (wroteAny) el.appendChild(document.createTextNode('\u00A0\u00A0|\u00A0\u00A0'));
    const labelSpan = document.createElement('span');
    labelSpan.className = cls;
    labelSpan.textContent = label + ': ';
    el.appendChild(labelSpan);
    list.forEach((protein, i) => {
      if (i > 0) el.appendChild(document.createTextNode(', '));
      const a = document.createElement('span');
      a.className = cls + ' protein-link';
      a.textContent = protein;
      a.dataset.protein = protein;
      el.appendChild(a);
    });
    wroteAny = true;
  });
  if (!wroteAny) el.textContent = 'No proteins';
}

function buildEvidenceSection(title, rows, preferredCols) {
  const wrap = document.createElement('div');
  wrap.className = 'evidence-section';
  const h = document.createElement('h4');
  h.textContent = title + ' (' + rows.length + ')';
  wrap.appendChild(h);
  if (!rows.length) {
    const p = document.createElement('div');
    p.className = 'evidence-empty';
    p.textContent = 'No rows for this protein / PDB.';
    wrap.appendChild(p);
    return wrap;
  }
  const allCols = new Set();
  rows.forEach(r => Object.keys(r).forEach(k => allCols.add(k)));
  const cols = [
    ...preferredCols.filter(c => allCols.has(c)),
    ...[...allCols].filter(c => !preferredCols.includes(c)),
  ];
  const table = document.createElement('table');
  table.className = 'evidence-table';
  const thead = document.createElement('tr');
  cols.forEach(c => { const th = document.createElement('th'); th.textContent = c; thead.appendChild(th); });
  table.appendChild(thead);
  rows.forEach(r => {
    const tr = document.createElement('tr');
    cols.forEach(c => {
      const td = document.createElement('td');
      const val = r[c];
      td.textContent = (val === null || val === undefined) ? '' : String(val);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  wrap.appendChild(table);
  return wrap;
}

function showEvidencePopup(protein) {
  const ev = cur?.pdb_protein_evidence?.[currentPdbId]?.[protein];
  document.getElementById('evidence-popup-title').textContent =
    protein + '  vs  ' + (currentPdbId ? currentPdbId.toUpperCase() : '?');
  const body = document.getElementById('evidence-popup-body');
  body.innerHTML = '';
  body.appendChild(buildEvidenceSection('Direct SIFTS match', ev?.direct || [], ['CHAIN', 'SP_PRIMARY']));
  body.appendChild(buildEvidenceSection('MMseq homology hits', ev?.homology || [],
    ['hit_pdb_id', 'identity_percent', 'blast_identity_percent', 'alnlen']));
  document.getElementById('evidence-popup-backdrop').classList.add('on');
  document.getElementById('evidence-popup').classList.add('on');
}

function hideEvidencePopup() {
  document.getElementById('evidence-popup-backdrop').classList.remove('on');
  document.getElementById('evidence-popup').classList.remove('on');
}

document.addEventListener('click', (e) => {
  const t = e.target;
  if (t.classList?.contains('protein-link')) {
    showEvidencePopup(t.dataset.protein);
  } else if (t.id === 'evidence-popup-close' || t.id === 'evidence-popup-backdrop') {
    hideEvidencePopup();
  }
});

/* ── mapped-proteins summary ──────────────────────────────────────────── */
function updateMappedSummaries() {
  const cfEl  = document.getElementById('cf-mapped-summary');
  const pdbEl = document.getElementById('pdb-mapped-summary');
  cfEl.innerHTML = ''; pdbEl.innerHTML = '';

  if (rightMode === 'input') {
    const m = (cur?.input_models || [])[currentInputIdx];
    if (m) {
      const span = document.createElement('span');
      span.className = 'm-blue';
      span.textContent = 'Highlighted: ' + [...new Set(m.proteins)].join(' \u2013 ');
      cfEl.appendChild(span);
    }
    return;
  }
  if (!cur || colorMode !== 'by-mapping') return;

  const pdbProts      = cur.pdb_proteins?.[currentPdbId] || [];
  const modelProteins = [...new Set(Object.values(cur.cf_chain_map || {}))];
  renderGroup(cfEl, [
    ['m-blue', 'Mapped', modelProteins.filter(p => pdbProts.includes(p)).sort()],
    ['m-off',  'Novel',  modelProteins.filter(p => !pdbProts.includes(p)).sort()],
  ]);

  const chainMap = cur.pdb_chain_map?.[currentPdbId] || {};
  const homMap   = cur.pdb_homology_map?.[currentPdbId] || {};
  const hitSet = new Set();
  Object.values(chainMap).forEach(info => info.cp.forEach(p => hitSet.add(p)));
  Object.values(homMap).forEach(arr => arr.forEach(p => hitSet.add(p)));
  const cpProteins = cur.cp_proteins || [];
  renderGroup(pdbEl, [
    ['m-blue', 'Mapped', cpProteins.filter(p => hitSet.has(p))],
    ['m-off',  'No hit', cpProteins.filter(p => !hitSet.has(p))],
  ]);
}

/* ── annotations: Complex Portal; RCSB title (labelled) where CP has none ─ */
function updateAnnotations() {
  const cfEl  = document.getElementById('cf-annotation');
  const pdbEl = document.getElementById('pdb-annotation');
  cfEl.textContent = cur?.cp_annotation || '(no Complex Portal annotation)';

  if (rightMode === 'input') {
    const m = (cur?.input_models || [])[currentInputIdx];
    pdbEl.textContent = m ? 'Input pair: ' + m.label + '  [' + m.filename + ']'
                          : '(no input pair models)';
    return;
  }
  if (!currentPdbId) { pdbEl.textContent = ''; return; }
  const cpAnnot   = cur?.pdb_annotation?.[currentPdbId];
  const rcsbTitle = pdbCache[currentPdbId]?.title;
  if (cpAnnot)        pdbEl.textContent = cpAnnot;
  else if (rcsbTitle) pdbEl.textContent = 'RCSB title: ' + rcsbTitle;
  else                pdbEl.textContent = '(no annotation available)';
}

function spin(id, on, msg) {
  const el = document.getElementById(id);
  el.classList.toggle('on', on);
  if (msg !== undefined) el.textContent = msg;
}

/* ── load CF model ────────────────────────────────────────────────────── */
async function loadCF(idx) {
  if (!cur) return;
  const models = cur.cf_models;
  if (!models.length) {
    cfRaw = null;
    cfV.removeAllModels(); cfV.render();
    document.getElementById('model-info').textContent = 'No CombFold model';
    document.getElementById('prev-m').disabled = true;
    document.getElementById('next-m').disabled = true;
    return;
  }
  idx = Math.max(0, Math.min(idx, models.length - 1));
  mIdx = idx;
  const complexAtStart = cur;
  spin('cf-overlay', true, 'Loading...');
  try {
    const pdb = await ungzip(models[idx].pdb_gz);
    if (cur !== complexAtStart) return;  // user switched complex meanwhile
    styleViewer(cfV, pdb, 'pdb', 'cf');
    document.getElementById('model-info').textContent =
      (idx + 1) + ' / ' + models.length + '  CF ' + models[idx].score.toFixed(1);
  } catch (e) {
    console.error(e);
    document.getElementById('model-info').textContent = 'Load error';
  } finally { spin('cf-overlay', false); }
  document.getElementById('prev-m').disabled = idx <= 0;
  document.getElementById('next-m').disabled = idx >= models.length - 1;
}

/* ── RCSB: structure (.pdb, or .cif for entries without legacy PDB format) + title ── */
async function fetchPDBTitle(pid) {
  try {
    const r = await fetch('https://data.rcsb.org/rest/v1/core/entry/' + pid.toUpperCase());
    if (!r.ok) return null;
    const json = await r.json();
    return json?.struct?.title || null;
  } catch (e) {
    console.error('RCSB title fetch failed for', pid, e);
    return null;
  }
}

async function fetchPDBText(pid) {
  const structPromise = (async () => {
    const r1 = await fetch('https://files.rcsb.org/download/' + pid.toUpperCase() + '.pdb');
    if (r1.ok) return { text: await r1.text(), fmt: 'pdb' };
    const r2 = await fetch('https://files.rcsb.org/download/' + pid.toUpperCase() + '.cif');
    if (!r2.ok) throw new Error('RCSB 404 for ' + pid + ' (.pdb and .cif)');
    return { text: await r2.text(), fmt: 'mmcif' };
  })();
  const [struct, title] = await Promise.all([structPromise, fetchPDBTitle(pid)]);
  return { ...struct, title };
}

async function loadPDB(pid) {
  if (!pid) return;
  currentPdbId = pid;
  document.querySelectorAll('.pdb-btn')
    .forEach(b => b.classList.toggle('active', b.dataset.pid === pid));
  updateAnnotations();
  const owner = pdbCacheOwner;

  if (pdbCache[pid]) {
    spin('pdb-overlay', false);
    const {text, fmt} = pdbCache[pid];
    styleViewer(pdbV, text, fmt, 'pdb');
    if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
    return;
  }
  spin('pdb-overlay', true, 'Fetching from RCSB...');
  try {
    const result = await fetchPDBText(pid);
    if (owner !== pdbCacheOwner) return;      // complex changed meanwhile
    pdbCache[pid] = result;
    if (currentPdbId !== pid || rightMode !== 'pdb') return;
    styleViewer(pdbV, result.text, result.fmt, 'pdb');
    if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
    updateAnnotations();
    spin('pdb-overlay', false);
  } catch (e) {
    console.error(e);
    spin('pdb-overlay', true, 'Could not load ' + pid.toUpperCase() + ' - ' + e.message);
  }
}

/* Sequential, stops as soon as the user switches complex. */
async function prefetchOtherPDBs(owner, pids) {
  for (const pid of pids) {
    if (owner !== pdbCacheOwner) return;
    if (pdbCache[pid]) continue;
    try {
      const result = await fetchPDBText(pid);
      if (owner !== pdbCacheOwner) return;
      pdbCache[pid] = result;
      const btn = document.querySelector('.pdb-btn[data-pid="' + pid + '"]');
      if (btn) btn.classList.add('cached');
      if (currentPdbId === pid) updateAnnotations();
    } catch (e) {
      console.error('prefetch failed for', pid, e);
    }
  }
}

/* ── select complex ───────────────────────────────────────────────────── */
function setWarning(id, text) {
  const el = document.getElementById(id);
  el.textContent = text || '';
  el.title       = text || '';
  el.classList.toggle('on', !!text);
}

function selectComplex(ac) {
  cur = COMPLEXES[ac];
  if (!cur) return;

  cfRaw = null; pdbRaw = null;
  currentPdbId = null;
  currentInputIdx = -1;
  inputChainMap = {};
  pdbCache = {};
  pdbCacheOwner = ac;
  spin('pdb-overlay', false);

  const hasInputs = cur.input_models.length > 0;
  document.getElementById('tab-input').disabled = !hasInputs;
  document.getElementById('tab-input').title = hasInputs ? '' : 'No CombFold run -> no input models';

  document.getElementById('badge-cf').textContent =
    (cur.cf_confidence != null ? 'CF ' + cur.cf_confidence.toFixed(1) : 'CF \u2013')
    + '  (untrimmed ' + cur.cf_confidence_ref.toFixed(1) + ')';
  document.getElementById('badge-match').textContent = cur.match_class.replace(/_/g, ' ');

  const stoicText = cur.stoic_label
    ? cur.stoic_label + '  [' + cur.run_labels.join('/') + ']'
    : '(no CombFold run)';
  const sl = document.getElementById('stoic-label');
  sl.textContent = stoicText;
  sl.title = stoicText;

  const missing = Object.entries(cur.assembly_missing);
  setWarning('assembly-warning', missing.length
    ? 'partial assembly, missing: ' + missing.map(([p, n]) => p + ' x' + n).join(', ')
    : '');

  const cfModel0 = cur.cf_models[0] || null;
  const cfPathEl = document.getElementById('cf-path');
  cfPathEl.textContent = cfModel0 ? cfModel0.folder : '(no CombFold model)';
  cfPathEl.title       = cfModel0 ? cfModel0.path   : '';

  document.getElementById('pdb-cover-level').textContent =
    cur.pdb_cover_level ? 'cover ' + cur.pdb_cover_level + '%' : '';
  setWarning('pdb-cap-warning', cur.pdb_cap_hit
    ? 'showing ' + cur.pdb_ids.length + ' of ' + cur.pdb_n_total + ' PDBs' : '');
  setWarning('input-cap-warning', cur.input_cap_hit
    ? 'showing ' + cur.input_models.length + ' of ' + cur.input_n_total + ' pair models (rank 1 first)' : '');

  const list = document.getElementById('pdb-list');
  list.innerHTML = '';
  const pids = cur.pdb_ids;
  if (!pids.length) {
    list.textContent = 'no PDB';
  } else {
    pids.forEach(pid => {
      const b = document.createElement('button');
      b.className   = 'pdb-btn';
      b.dataset.pid = pid;
      b.textContent = pid.toUpperCase();
      b.onclick = () => loadPDB(pid);
      list.appendChild(b);
    });
  }

  setRightMode('pdb');
  mIdx = 0;
  loadCF(0);
  if (pids.length) {
    loadPDB(pids[0]);
    prefetchOtherPDBs(ac, pids.slice(1));
  }
  updateAnnotations();
  updateMappedSummaries();
}

/* ── dropdown (order = Python order: trimmed CF score, best first) ─────── */
function buildDropdown() {
  const sel = document.getElementById('complex-select');
  Object.values(COMPLEXES).forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.complex_ac;
    const ids = c.identifiers.length > 55 ? c.identifiers.slice(0, 52) + '...' : c.identifiers;
    opt.textContent = c.complex_ac + '  -  ' + ids;
    sel.appendChild(opt);
  });
  sel.onchange = () => selectComplex(sel.value);
}

/* ── event wiring ─────────────────────────────────────────────────────── */
document.getElementById('prev-m').onclick = () => loadCF(mIdx - 1);
document.getElementById('next-m').onclick = () => loadCF(mIdx + 1);

document.getElementById('scr-page').onclick = function() {
  const btn = this;
  btn.classList.add('busy');
  btn.textContent = '...';
  html2canvas(document.documentElement, {
    useCORS: true, allowTaint: true, scale: 3,
    width: window.innerWidth, height: window.innerHeight,
    windowWidth: window.innerWidth, windowHeight: window.innerHeight,
    ignoreElements: el => el === btn,
  }).then(canvas => {
    const a = document.createElement('a');
    a.href     = canvas.toDataURL('image/png');
    a.download = (cur?.complex_ac || 'viewer') + '_screenshot.png';
    a.click();
  }).finally(() => {
    btn.classList.remove('busy');
    btn.textContent = '\u2193 PNG';
  });
};

document.getElementById('color-mode').onchange = function() {
  colorMode = this.value;
  document.getElementById('plddt-legend').style.display = colorMode === 'plddt' ? '' : 'none';
  if (rightMode === 'input') {
    if (currentInputIdx >= 0) loadInputModel(currentInputIdx);
    return;
  }
  if (cfRaw)  styleViewer(cfV,  cfRaw,  'pdb',     'cf');
  if (pdbRaw) styleViewer(pdbV, pdbRaw, pdbFmtCur, 'pdb');
};

function refilterAfterPlddtChange() {
  if (rightMode === 'input' && currentInputIdx >= 0) loadInputModel(currentInputIdx);
  else if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
}

function setThresholdValue(raw) {
  plddtThreshold = Math.max(0, Math.min(100, Number(raw) || 0));
  document.getElementById('plddt-threshold').value     = plddtThreshold;
  document.getElementById('plddt-threshold-num').value = plddtThreshold;
}
document.getElementById('plddt-threshold').oninput = function() {
  setThresholdValue(this.value); refilterAfterPlddtChange();
};
document.getElementById('plddt-threshold-num').oninput = function() {
  setThresholdValue(this.value); refilterAfterPlddtChange();
};
document.querySelectorAll('input[name="plddt-mode"]').forEach(radio => {
  radio.onchange = function() {
    plddtMode = this.value;
    const isThreshold = plddtMode === 'threshold';
    document.getElementById('plddt-threshold').disabled     = !isThreshold;
    document.getElementById('plddt-threshold-num').disabled = !isThreshold;
    refilterAfterPlddtChange();
  };
});

/* ── init ─────────────────────────────────────────────────────────────── */
initViewers();
buildDropdown();
const firstComplex = Object.keys(COMPLEXES)[0];
if (firstComplex) {
  document.getElementById('complex-select').value = firstComplex;
  selectComplex(firstComplex);
}
</script>
</body>
</html>
"""

# ── write output ──────────────────────────────────────────────────────────────

json_blob = json.dumps(complex_entries, ensure_ascii=True, default=str)
assert "%%JSON%%" in HTML
html_out = HTML.replace("%%JSON%%", json_blob)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html_out, encoding="utf-8")
size_kb = OUT.stat().st_size / 1024
print(f"\n  Written: {OUT}")
print(f"   Size:    {size_kb:.0f} KB  (~{size_kb / 1024:.1f} MB)")