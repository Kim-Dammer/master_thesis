#!/usr/bin/env python3
"""
Trimmed YeastMap CombFold viewer, compared to the untrimmed run or to the CP core complex.

Dropdown : YeastMap complexes whose best trimmed output (over pred_1/2/3) has confidence > CONF_THRESHOLD,
           filterable by PDB match class (all_YM_pdb_matches.csv)
Left     : trimmed run, best output per predicted stoichiometry, arrows step through the
           stoichiometries from highest to lowest confidence
Right    : switch between
           - untrimmed : best output of the SAME stoichiometry (same output folder name)
           - CP core   : the Complex Portal complex the YeastMap prediction is based on (input_12_yeastmap.tsv),
                         stoichiometry closest to the left one (shared proteins), ties -> highest confidence
           - PDB (core): experimental structure of the CP core complex, if its match class
                         (all_pdb_matches_with_match_class.csv) is exact_pdb_match / exact_homology_match.
                         Fetched from RCSB (mmCIF, asymmetric unit) when the tab is opened. Chain mapping
                         CP protein -> PDB chain from exact_matches.csv (one chain per protein; other chains of
                         the same entity are treated as further copies of that protein). exact_pdb_match rows
                         missing in exact_matches.csv are mapped by UniProt id: CP protein == UniProt accession
                         of the chain's entity in the mmCIF _struct_ref (db_name UNP)
Colors   : by chain | by protein (same color for a protein in both panels + GO panel; PDB chains get the color
           of their mapped CP protein) | uniform | trimmed mapping | CP core proteins |
           core PDB chain mapping (blue = protein/chain mapped between CP core and PDB, amber = not mapped)
Click    : on an atom -> chain, UniProt, residue, pLDDT; every UniProt id links to uniprot.org
PDB      : match class of the YeastMap complex + PDB id (links to RCSB)
GO       : button opens a panel with
           - per protein: how many of its GO terms are shared with the other side (added <-> CP core)
           - term x protein matrix (BP / MF / CC), rows: shared, added only, core only
           - click a term -> its proteins are highlighted in both 3D panels
Gene names are fetched from the UniProt REST API when the page is opened (ids only if offline).
"""
import base64
import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from procompa import get_project_root
from tqdm import tqdm

DATA = get_project_root() / "data"
TRIMMED_RUN = "22_YM_pae_trimmed"
UNTRIMMED_RUN = "21_YM_pae"
CP_RUN = "23_all_CP_pae"
YEASTMAP_TSV = DATA / "Pipeline/12_Yeastmap/input_12_yeastmap.tsv"   # YeastMap complex -> CP complex
PDB_MATCHES_CSV = DATA / "YM_CP_complete_complex_pdb_mapping/all_YM_pdb_matches.csv"
GO_PARQUET = DATA / "GO/ym_protein_mapping.parquet"
CORE_PDB_MATCHES_CSV = DATA / "complete_complex_pdb_mapping_v2/all_pdb_matches_with_match_class.csv"   # CP complex -> PDB
CORE_PDB_CHAIN_HITS_CSV = DATA / "complete_complex_pdb_mapping_v2/homology_pdb_mapping/exact_matches.csv"   # CP protein -> PDB chain
CONF_THRESHOLD = 75
OUT = DATA / "Pipeline/viewer/09_trimmed_vs_untrimmed_viewer_cf_75.html"
BACKBONE_ATOMS = {"N", "CA", "C", "O"}   # enough for a cartoon, ~5x smaller than full atom
N_THREADS = 16                           # file reads are the slow part -> read complexes in parallel

PDB_MATCH_CLASSES = ["exact_homology_match", "superset", "partial", "no_sufficient_match", "no_homology_match"]
CORE_PDB_MATCH_CLASSES = ["exact_pdb_match", *PDB_MATCH_CLASSES]
CORE_PDB_SHOWN_CLASSES = {"exact_pdb_match", "exact_homology_match"}   # only these get the 'PDB (core)' structure
GO_ASPECT_SHORT = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}
GO_GROUPS = {"added", "true_complex", "true_only"}   # added = YM-added proteins, true_complex = all CP core proteins,
                                                     # true_only = CP core proteins not in the YM complex
GO_ROOT_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}   # 'molecular_function' etc. -> dropped, no information


def results_csv(run: str) -> Path:
    return DATA / "Pipeline" / run / f"all_pdb_present_{run}_pool_pipeline_complexes_combfold_results.csv"


for required_path in (results_csv(TRIMMED_RUN), results_csv(CP_RUN), YEASTMAP_TSV, PDB_MATCHES_CSV, GO_PARQUET,
                      CORE_PDB_MATCHES_CSV, CORE_PDB_CHAIN_HITS_CSV, DATA / "Pipeline" / UNTRIMMED_RUN / "CombFold"):
    assert required_path.exists(), f"missing: {required_path}"


# ── helpers ───────────────────────────────────────────────────────────────────

def output_folder_name(protein_counts: dict[str, int]) -> str:
    """Same naming as the pipeline: 'P1x1_P2x2_pool_output' (hashed if > 200 chars)."""
    complex_name = "_".join(f"{p}x{protein_counts[p]}" for p in sorted(protein_counts))
    if len(complex_name) > 200:
        complex_name = complex_name[:180] + "_" + hashlib.sha1(complex_name.encode()).hexdigest()[:12]
    return f"{complex_name}_pool_output"


def backbone_pdb(pdb_text: str) -> str:
    return "\n".join(line for line in pdb_text.splitlines()
                     if line.startswith("ATOM") and line[12:16].strip() in BACKBONE_ATOMS)


def compress(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode(), compresslevel=6)).decode()


def csv_best_confidence(col: str) -> pl.Expr:
    """'{0: 78.5017, 1: 78.8618}' -> 78.8618; '{}' -> null"""
    return (pl.col(col).str.extract_all(r": -?[\d.]+")
            .list.eval(pl.element().str.slice(2).cast(pl.Float64)).list.max())


def parse_pred(row: dict, pred: int) -> tuple[dict[str, int], dict]:
    """pred_k = '{"P1":1,"P2":2},{"rank":1,"n_copies":3,"probability":0.33}' -> (counts, info)"""
    protein_counts, pred_info = json.loads("[" + row[f"pred_{pred}"] + "]")
    assert pred_info["rank"] == pred, f"{row['complex_ac']} pred_{pred}: rank is {pred_info['rank']}"
    return protein_counts, pred_info


def identifier_counts(identifiers: str) -> dict[str, int]:
    """'P1(2)|P2(0)|CHEBI:123(1)' -> {'P1': 2, 'P2': 0} (small molecules / RNA dropped, 0 = unknown)"""
    protein_counts = {}
    for token in identifiers.split("|"):
        if token.startswith(("CHEBI", "URS")):
            continue
        token_match = re.match(r"^([A-Za-z0-9_-]+)\((\d+)\)$", token.strip())
        assert token_match, f"unparseable identifier {token!r} in {identifiers!r}"
        protein_counts[token_match.group(1)] = int(token_match.group(2))
    return protein_counts


def read_chain_list(output_dir: Path) -> dict[str, str]:
    """CombFold's chain.list: one 'UNIPROT_CHAIN.pdb' per line -> {chain: uniprot}."""
    chain_list_path = output_dir / "_unified_representation" / "assembly_output" / "chain.list"
    assert chain_list_path.exists(), f"chain.list missing: {chain_list_path}"
    chain_to_protein = {}
    for line in filter(None, map(str.strip, chain_list_path.read_text().splitlines())):
        chain_match = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pdb$", line)
        assert chain_match, f"unparseable chain.list line in {chain_list_path}: {line!r}"
        chain_to_protein[chain_match.group(2)] = chain_match.group(1)
    return chain_to_protein


def read_best_model(run: str, folder: str) -> dict | None:
    """Highest-confidence output of one CombFold folder; None if the folder has no outputs."""
    output_dir = DATA / "Pipeline" / run / "CombFold" / folder
    confidence_path = output_dir / "assembled_results" / "confidence.txt"
    if not confidence_path.exists():
        return None
    scored_lines = [line.split() for line in confidence_path.read_text().splitlines() if line.strip()]
    if not scored_lines:
        return None
    assert all(len(fields) == 2 for fields in scored_lines), f"malformed {confidence_path}"
    best_path, best_conf = max(((Path(p), float(s)) for p, s in scored_lines), key=lambda model: model[1])
    model_path = output_dir / "assembled_results" / best_path.name
    assert model_path.exists(), f"best model missing: {model_path}"

    model_text = backbone_pdb(model_path.read_text())
    chain_to_protein = read_chain_list(output_dir)
    model_chains = list(dict.fromkeys(line[21] for line in model_text.splitlines()))   # order of appearance
    assert set(model_chains) <= set(chain_to_protein), f"{model_path}: chains {set(model_chains) - set(chain_to_protein)} not in chain.list"
    plddts = [float(line[60:66]) for line in model_text.splitlines()]
    assert 0 <= min(plddts) and max(plddts) <= 100 and max(plddts) > 0, (
        f"{model_path}: B-factor column doesn't look like pLDDT (min {min(plddts)}, max {max(plddts)})")
    return {"conf": best_conf, "n_outputs": len(scored_lines), "folder": folder, "model": best_path.name,
            "model_chains": model_chains,
            "chains": {chain: chain_to_protein[chain] for chain in model_chains},
            "pdb_gz": compress(model_text)}


def core_candidates(cp_row: dict) -> dict[str, dict]:
    """All CP core stoichiometries with outputs: pred_1/2/3 + true (if run) -> {folder: {"labels", "counts", "model"}}."""
    candidates = {}
    for pred in (1, 2, 3):
        if cp_row[f"CF_{pred}_n_assemblies"] == 0:
            continue
        protein_counts, _ = parse_pred(cp_row, pred)
        folder = output_folder_name(protein_counts)
        if folder in candidates:
            candidates[folder]["labels"].append(f"pred {pred}")
            continue
        model = read_best_model(CP_RUN, folder)
        assert model is not None, f"{cp_row['complex_ac']} pred {pred}: no outputs in {CP_RUN}/{folder}"
        assert model["n_outputs"] == cp_row[f"CF_{pred}_n_assemblies"], f"{cp_row['complex_ac']} pred {pred}: output count differs from csv"
        candidates[folder] = {"labels": [f"pred {pred}"], "counts": protein_counts, "model": model}

    # true stoichiometry: unknown '(0)' counts were run as 1 copy (e.g. CPX-474: P07213(0) -> P07213x1)
    true_counts = {protein: n if n > 0 else 1 for protein, n in identifier_counts(cp_row["identifiers"]).items()}
    true_folder = output_folder_name(true_counts)
    if true_folder in candidates:
        candidates[true_folder]["labels"].append("true")
    elif cp_row["CF_true_n_assemblies"] > 0:
        model = read_best_model(CP_RUN, true_folder)
        assert model is not None, f"{cp_row['complex_ac']} true: no outputs in {CP_RUN}/{true_folder}"
        assert model["n_outputs"] == cp_row["CF_true_n_assemblies"], f"{cp_row['complex_ac']} true: output count differs from csv"
        candidates[true_folder] = {"labels": ["true"], "counts": true_counts, "model": model}
    return candidates


def closest_core(trimmed_counts: dict[str, int], candidates: dict[str, dict], core_proteins: set[str]) -> tuple[str, int]:
    """Core stoichiometry with the fewest copy differences on the shared proteins; ties -> highest confidence."""
    shared_proteins = set(trimmed_counts) & core_proteins
    def copy_difference(candidate: dict) -> int:
        return sum(abs(trimmed_counts[p] - candidate["counts"].get(p, 0)) for p in shared_proteins)
    best_folder = min(candidates, key=lambda f: (copy_difference(candidates[f]), -candidates[f]["model"]["conf"]))
    return best_folder, copy_difference(candidates[best_folder])


def core_pdb_reference(cp_ac: str, core_proteins: set[str]) -> dict | None:
    """PDB structure of the CP core complex for the 'PDB (core)' tab; None if the match class is not exact.

    mapping_source 'exact_matches': chain_to_protein = {pdb_chain: CP UniProt} from exact_matches.csv (for homology
    matches the chain is a homolog of that CP protein).
    mapping_source 'uniprot': exact_pdb_match rows missing in exact_matches.csv (match taken directly from Complex
    Portal) -> chain_to_protein None, the viewer maps chains by the UniProt accessions in the mmCIF _struct_ref."""
    core_match = core_pdb_match_by_cp_ac[cp_ac]
    if core_match["match_class"] not in CORE_PDB_SHOWN_CLASSES:
        return None
    pdb_id = core_match["pdb_id"]
    matching_hits = matching_hits_by_cp_pdb.get((cp_ac, pdb_id))
    if matching_hits is None:
        assert core_match["match_class"] == "exact_pdb_match", (
            f"{cp_ac} {pdb_id} ({core_match['match_class']}): no chain mapping in {CORE_PDB_CHAIN_HITS_CSV.name}")
        return {"pdb_id": pdb_id, "match_class": core_match["match_class"], "mapping_source": "uniprot", "chain_to_protein": None}

    chain_to_protein = {}
    for hit in matching_hits.split(","):
        hit_match = re.match(r"^([a-z0-9]+):([a-z0-9]{4})_([A-Za-z0-9]+)$", hit.strip())
        assert hit_match, f"{cp_ac} {pdb_id}: unparseable matching hit {hit!r}"
        protein, hit_pdb_id, chain = hit_match.group(1).upper(), hit_match.group(2).upper(), hit_match.group(3)
        assert hit_pdb_id == pdb_id, f"{cp_ac}: hit {hit!r} is not on {pdb_id}"
        assert chain not in chain_to_protein, f"{cp_ac} {pdb_id}: chain {chain} mapped twice"
        chain_to_protein[chain] = protein
    mapped_proteins = set(chain_to_protein.values())
    assert len(mapped_proteins) == len(chain_to_protein), f"{cp_ac} {pdb_id}: a protein mapped to more than one chain"
    assert mapped_proteins == core_proteins, (
        f"{cp_ac} {pdb_id}: mapped proteins {sorted(mapped_proteins)} != CP core proteins {sorted(core_proteins)}")
    return {"pdb_id": pdb_id, "match_class": core_match["match_class"], "mapping_source": "exact_matches",
            "chain_to_protein": chain_to_protein}


def go_terms_for_complex(ym_ac: str, core_kept: set[str], core_dropped: set[str], added: set[str]) -> list[list]:
    """GO rows of one YM complex -> [[go_id, 'BP', name, dataset_freq, [proteins]], ...] (root terms dropped).

    Proteins are the union of the 'added' and 'true_complex' groups; 'true_only' is only used as a consistency check
    (it must be exactly the CP core proteins that are not in the YM complex)."""
    term_proteins = {group: {} for group in GO_GROUPS}
    term_aspect_name = {}
    for go_row in go_rows_by_ym_ac[ym_ac].iter_rows(named=True):
        term_proteins[go_row["group"]][go_row["go_id"]] = set(go_row["proteins"])
        term_aspect_name[go_row["go_id"]] = (GO_ASPECT_SHORT[go_row["aspect"]], go_row["go_name"])

    core = core_kept | core_dropped
    for go_id, proteins in term_proteins["added"].items():
        assert proteins <= added, f"{ym_ac} {go_id}: 'added' proteins {proteins - added} not YM-only"
    for go_id, proteins in term_proteins["true_complex"].items():
        assert proteins <= core, f"{ym_ac} {go_id}: 'true_complex' proteins {proteins - core} not in CP core"
    for go_id in term_proteins["true_complex"].keys() | term_proteins["true_only"].keys():
        expected_true_only = term_proteins["true_complex"].get(go_id, set()) & core_dropped
        assert term_proteins["true_only"].get(go_id, set()) == expected_true_only, (
            f"{ym_ac} {go_id}: 'true_only' {term_proteins['true_only'].get(go_id)} != core proteins missing in YM {expected_true_only}")

    go_terms = []
    for go_id in sorted(term_aspect_name.keys() - GO_ROOT_TERMS):
        proteins = term_proteins["added"].get(go_id, set()) | term_proteins["true_complex"].get(go_id, set())
        assert proteins, f"{ym_ac} {go_id}: term without proteins in 'added' / 'true_complex'"
        aspect, go_name = term_aspect_name[go_id]
        go_terms.append([go_id, aspect, go_name, round(dataset_freq_by_go_id[go_id], 4), sorted(proteins)])
    return go_terms


# ── load tables ───────────────────────────────────────────────────────────────

trimmed_results = pl.read_csv(results_csv(TRIMMED_RUN))
assert trimmed_results["complex_ac"].is_unique().all(), f"{TRIMMED_RUN}: more than one row per complex"
cp_results = pl.read_csv(results_csv(CP_RUN))
assert cp_results["complex_ac"].is_unique().all(), f"{CP_RUN}: more than one row per complex"
cp_row_by_ac = {cp_row["complex_ac"]: cp_row for cp_row in cp_results.iter_rows(named=True)}

yeastmap = pl.read_csv(YEASTMAP_TSV, separator="\t", infer_schema_length=0)
assert yeastmap["#Complex ac"].is_unique().all(), "YeastMap complex listed twice in tsv"
cp_ac_by_ym_ac = dict(zip(yeastmap["#Complex ac"], yeastmap["#Complex_ac_db"]))

# PDB matches: one row per YM complex, pdb_id 'none' exactly for no_homology_match
pdb_matches = pl.read_csv(PDB_MATCHES_CSV, schema_overrides={"pdb_id": pl.String})
assert pdb_matches["complex_ac"].is_unique().all(), f"{PDB_MATCHES_CSV}: more than one row per complex"
assert set(pdb_matches["match_class"]) <= set(PDB_MATCH_CLASSES), f"unknown match_class: {set(pdb_matches['match_class']) - set(PDB_MATCH_CLASSES)}"
assert ((pdb_matches["pdb_id"] == "none") == (pdb_matches["match_class"] == "no_homology_match")).all(), \
    "pdb_id 'none' should occur exactly for no_homology_match"
pdb_match_by_ym_ac = {match_row["complex_ac"]: {"match_class": match_row["match_class"],
                                                "pdb_id": None if match_row["pdb_id"] == "none" else match_row["pdb_id"]}
                      for match_row in pdb_matches.iter_rows(named=True)}

# PDB match of the CP core complexes: one row per CP complex, pdb_id 'none' exactly for no_homology_match
core_pdb_matches = pl.read_csv(CORE_PDB_MATCHES_CSV, infer_schema_length=0)
assert core_pdb_matches["complex_ac"].is_unique().all(), f"{CORE_PDB_MATCHES_CSV}: more than one row per complex"
assert set(core_pdb_matches["match_class"]) <= set(CORE_PDB_MATCH_CLASSES), \
    f"unknown match_class: {set(core_pdb_matches['match_class']) - set(CORE_PDB_MATCH_CLASSES)}"
assert ((core_pdb_matches["pdb_id"] == "none") == (core_pdb_matches["match_class"] == "no_homology_match")).all(), \
    f"{CORE_PDB_MATCHES_CSV.name}: pdb_id 'none' should occur exactly for no_homology_match"
assert core_pdb_matches["pdb_id"].str.contains(r"^([0-9A-Za-z]{4}|none)$").all(), f"{CORE_PDB_MATCHES_CSV.name}: odd pdb_id"
core_pdb_match_by_cp_ac = {match_row["complex_ac"]: {"match_class": match_row["match_class"],   # pdb ids: mixed case in csv
                                                     "pdb_id": None if match_row["pdb_id"] == "none" else match_row["pdb_id"].upper()}
                           for match_row in core_pdb_matches.iter_rows(named=True)}

# CP protein -> PDB chain, one row per (CP complex, PDB): 'p25656:6k7g_C,p39524:6k7g_A'
core_pdb_chain_hits = (pl.read_csv(CORE_PDB_CHAIN_HITS_CSV, infer_schema_length=0)
                       .with_columns(pl.col("pdb_id").str.to_uppercase()))
assert core_pdb_chain_hits.select("complex_ac", "pdb_id").is_unique().all(), \
    f"{CORE_PDB_CHAIN_HITS_CSV.name}: more than one row per (complex, PDB)"
matching_hits_by_cp_pdb = {(cp_ac, pdb_id): matching_hits for cp_ac, pdb_id, matching_hits
                           in core_pdb_chain_hits.select("complex_ac", "pdb_id", "matching_hits").iter_rows()}

# GO: one row per (YM complex, group, term), proteins = the group's proteins annotated with the term
go_annotations = pl.read_parquet(GO_PARQUET)
assert go_annotations.select("complex_ac", "group", "aspect", "go_id").is_unique().all(), "duplicate GO rows"
assert set(go_annotations["group"]) == GO_GROUPS, f"unexpected GO groups: {set(go_annotations['group'])}"
assert set(go_annotations["aspect"]) == set(GO_ASPECT_SHORT), f"unexpected GO aspects: {set(go_annotations['aspect'])}"
assert (go_annotations["n_proteins"] == go_annotations["proteins"].list.len()).all(), "n_proteins != len(proteins)"
assert go_annotations.group_by("go_id").agg(pl.col("aspect").n_unique()).filter(pl.col("aspect") > 1).is_empty(), \
    "a GO id with more than one aspect"
go_cp_pairs = go_annotations.select("complex_ac", "cp_complex_ac").unique()
wrong_cp_in_go = [(ym_ac, cp_ac) for ym_ac, cp_ac in go_cp_pairs.iter_rows() if cp_ac_by_ym_ac.get(ym_ac) != cp_ac]
assert not wrong_cp_in_go, f"GO parquet CP complex differs from {YEASTMAP_TSV.name}: {wrong_cp_in_go[:10]}"
# in_true_complex of an added term == the term is annotated on a CP core protein
core_term_keys = go_annotations.filter(pl.col("group") == "true_complex").select("complex_ac", "go_id")
added_terms_checked = (go_annotations.filter(pl.col("group") == "added")
                       .join(core_term_keys.with_columns(pl.lit(True).alias("term_on_core_protein")),
                             on=["complex_ac", "go_id"], how="left"))
assert (added_terms_checked["in_true_complex"] == added_terms_checked["term_on_core_protein"].fill_null(False)).all(), \
    "in_true_complex doesn't match the true_complex rows"

# term specificity proxy: fraction of all proteins in the GO table (all complexes) that carry the term
protein_term_pairs = (go_annotations.filter(pl.col("group") != "true_only").explode("proteins")
                      .select(pl.col("proteins").alias("protein"), "go_id").unique())
n_go_table_proteins = protein_term_pairs["protein"].n_unique()
dataset_freq_by_go_id = dict(protein_term_pairs.group_by("go_id")
                             .agg((pl.len() / n_go_table_proteins).alias("dataset_freq")).iter_rows())
go_rows_by_ym_ac = {key[0]: go_part for key, go_part in go_annotations.partition_by("complex_ac", as_dict=True).items()}

# filter on the confidences already in the csv -> only read PDBs of complexes that end up in the viewer
confident_complexes = (
    trimmed_results
    .with_columns(*[csv_best_confidence(f"CF_{pred}_confidence").alias(f"best_conf_{pred}") for pred in (1, 2, 3)])
    .filter(pl.max_horizontal("best_conf_1", "best_conf_2", "best_conf_3") > CONF_THRESHOLD)
)
assert confident_complexes.height, f"no complex with trimmed confidence > {CONF_THRESHOLD}"
missing_cp = [ac for ac in confident_complexes["complex_ac"] if cp_ac_by_ym_ac.get(ac) not in cp_row_by_ac]
assert not missing_cp, f"no CP core complex (tsv or {CP_RUN} csv) for: {missing_cp}"
missing_pdb_match = [ac for ac in confident_complexes["complex_ac"] if ac not in pdb_match_by_ym_ac]
assert not missing_pdb_match, f"{len(missing_pdb_match)} complexes not in {PDB_MATCHES_CSV.name}: {missing_pdb_match[:10]}"
missing_core_pdb_match = sorted({cp_ac_by_ym_ac[ac] for ac in confident_complexes["complex_ac"]} - core_pdb_match_by_cp_ac.keys())
assert not missing_core_pdb_match, \
    f"{len(missing_core_pdb_match)} CP core complexes not in {CORE_PDB_MATCHES_CSV.name}: {missing_core_pdb_match[:10]}"
missing_go = [ac for ac in confident_complexes["complex_ac"] if ac not in go_rows_by_ym_ac]
assert not missing_go, f"{len(missing_go)} complexes not in {GO_PARQUET.name}: {missing_go[:10]}"
print(f"{confident_complexes.height}/{trimmed_results.height} complexes with trimmed conf > {CONF_THRESHOLD}")


# ── per complex ───────────────────────────────────────────────────────────────

def build_complex_entry(row: dict) -> dict:
    cp_row = cp_row_by_ac[cp_ac_by_ym_ac[row["complex_ac"]]]
    core_proteins = set(identifier_counts(cp_row["identifiers"]))
    ym_proteins = set(identifier_counts(row["identifiers"]))
    candidates = core_candidates(cp_row)

    stois_by_folder = {}   # two preds can be the same stoichiometry -> one entry, both pred numbers
    for pred in (1, 2, 3):
        if row[f"CF_{pred}_n_assemblies"] == 0:
            continue
        protein_counts, pred_info = parse_pred(row, pred)
        folder = output_folder_name(protein_counts)
        if folder in stois_by_folder:
            stois_by_folder[folder]["preds"].append(pred)
            continue
        trimmed_model = read_best_model(TRIMMED_RUN, folder)
        assert trimmed_model is not None, f"{row['complex_ac']} pred {pred}: no outputs in {folder}"
        assert trimmed_model["n_outputs"] == row[f"CF_{pred}_n_assemblies"], f"{row['complex_ac']} pred {pred}: output count differs from csv"
        assert abs(trimmed_model["conf"] - row[f"best_conf_{pred}"]) < 1e-3, (
            f"{row['complex_ac']} pred {pred}: confidence.txt {trimmed_model['conf']} != csv {row[f'best_conf_{pred}']}")

        left_proteins = [trimmed_model["chains"][chain] for chain in trimmed_model["model_chains"]]
        core_folder, core_difference = closest_core(protein_counts, candidates, core_proteins) if candidates else (None, None)
        stois_by_folder[folder] = {
            "preds": [pred],
            "probability": pred_info["probability"],   # of the first (highest-ranked) pred with this stoi
            "counts": protein_counts,
            "interesting": any(n > 1 for n in protein_counts.values()),
            "trimmed": trimmed_model,
            "untrimmed": read_best_model(UNTRIMMED_RUN, folder),
            "core_folder": core_folder,
            "core_difference": core_difference,       # summed copy-number difference on shared proteins
            "core_proteins_in_left": len(set(left_proteins) & core_proteins),
            "core_chains_in_left": sum(p in core_proteins for p in left_proteins),
        }

    stois = sorted(stois_by_folder.values(), key=lambda stoi: -stoi["trimmed"]["conf"])
    used_core_folders = {stoi["core_folder"] for stoi in stois}
    protein_groups = {"core_kept": sorted(core_proteins & ym_proteins),
                      "core_dropped": sorted(core_proteins - ym_proteins),
                      "added": sorted(ym_proteins - core_proteins)}
    return {
        "complex_ac": row["complex_ac"],
        "identifiers": row["identifiers"],
        "cp_complex_ac": cp_row["complex_ac"],
        "core_proteins": sorted(core_proteins),
        "core_models": {folder: c for folder, c in candidates.items() if folder in used_core_folders},
        "stois": stois,
        "pdb_match": pdb_match_by_ym_ac[row["complex_ac"]],
        "core_pdb_match": core_pdb_match_by_cp_ac[cp_row["complex_ac"]],
        "core_pdb": core_pdb_reference(cp_row["complex_ac"], core_proteins),
        "protein_groups": protein_groups,
        "go_terms": go_terms_for_complex(row["complex_ac"], *(set(protein_groups[g]) for g in ("core_kept", "core_dropped", "added"))),
    }


with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    viewer_complexes = list(tqdm(executor.map(build_complex_entry, confident_complexes.iter_rows(named=True)),
                                 total=confident_complexes.height, desc="Reading models"))

viewer_complexes.sort(key=lambda c: -c["stois"][0]["trimmed"]["conf"])
all_stois = [stoi for c in viewer_complexes for stoi in c["stois"]]
print(f"{len(all_stois)} stoichiometries")
print(f"  {sum(s['untrimmed'] is None for s in all_stois)} without untrimmed output -> right panel empty")
print(f"  {sum(s['core_folder'] is None for s in all_stois)} without any CP core output -> right panel empty")
print(f"  {sum(s['core_difference'] == 0 for s in all_stois)} with the same stoi in CP core (shared proteins)")
for match_class in PDB_MATCH_CLASSES:
    print(f"  PDB {match_class}: {sum(c['pdb_match']['match_class'] == match_class for c in viewer_complexes)} complexes")
print(f"  {sum(not c['protein_groups']['added'] for c in viewer_complexes)} complexes without added proteins")
for match_class in CORE_PDB_MATCH_CLASSES:
    print(f"  CP core PDB {match_class}: {sum(c['core_pdb_match']['match_class'] == match_class for c in viewer_complexes)} complexes")
cores_mapped_by_uniprot = sorted({(c["cp_complex_ac"], c["core_pdb"]["pdb_id"]) for c in viewer_complexes
                                  if c["core_pdb"] and c["core_pdb"]["mapping_source"] == "uniprot"})
print(f"  {len(cores_mapped_by_uniprot)} CP cores with exact_pdb_match not in {CORE_PDB_CHAIN_HITS_CSV.name} "
      f"-> chains mapped by UniProt id from the mmCIF in the viewer: {cores_mapped_by_uniprot}")


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trimmed YeastMap CombFold viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
:root {
  --bg:#e9ecf0; --panel:#ffffff; --line:#d3d8df; --ink:#1c2430; --muted:#5d6876; --accent:#1565c0;
  --b-bg:#c3d6ee; --b-ink:#0c3466; --int-bg:#f4b877; --int-ink:#5a2700; --core-bg:#93d2c4; --core-ink:#0a4238;
  --shared-bg:#a9dcc5; --shared-ink:#0b5a3e; --added-bg:#f5cf8e; --added-ink:#6b3b00;
  --coreonly-bg:#c4cfdd; --coreonly-ink:#22384f;
}
* { box-sizing:border-box; margin:0; }
body { font-family:-apple-system,"Segoe UI",sans-serif; font-size:13px; color:var(--ink); background:var(--bg);
       padding:10px; height:100vh; display:flex; flex-direction:column; gap:8px; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
button { cursor:pointer; }
.topbar, .panel, .go-panel { background:var(--panel); border:1px solid var(--line); border-radius:6px; }
.topbar { padding:8px 12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
#complex-select { flex:1; max-width:640px; padding:3px; }
.panels { display:grid; grid-template-columns:1fr 1fr; gap:8px; flex:1 1 0; min-height:0; }
.panel { display:flex; flex-direction:column; overflow:hidden; }
.panel-hdr { padding:7px 12px; border-bottom:1px solid #e6e9ed; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.panel-hdr h3 { font-size:13px; }
.stoi { flex-basis:100%; font-size:11px; color:var(--muted); line-height:1.7; }
.prot { white-space:nowrap; font-family:ui-monospace,Menlo,monospace; font-size:11px; }
.prot .gene { font-family:-apple-system,"Segoe UI",sans-serif; font-weight:700; color:var(--ink); }
.prot .gene:not(:empty)::after { content:' '; }
.sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:3px; vertical-align:-1px; }
.copies { color:var(--muted); margin-right:8px; font-family:ui-monospace,Menlo,monospace; }
.badge { display:inline-block; margin:1px 0; padding:1px 9px; border-radius:10px; font-size:11px;
         font-weight:600; background:var(--b-bg); color:var(--b-ink); }
.badge.interesting { background:var(--int-bg); color:var(--int-ink); }
.badge.core { background:var(--core-bg); color:var(--core-ink); }
.badge a { color:inherit; text-decoration:underline; }
.pdb-exact_pdb_match { background:#0b4d24; color:#fff; }
.pdb-exact_homology_match { background:#1e7b3c; color:#fff; }
.badge.warn { background:#d62728; color:#fff; }
#color-legend { font-size:11px; color:var(--muted); }
#color-legend .sw { width:10px; height:10px; }
.pdb-superset { background:#9ccc65; color:#20380a; }
.pdb-partial { background:#f2c94c; color:#4d3800; }
.pdb-no_sufficient_match { background:#c9ced6; color:#333b47; }
.pdb-no_homology_match { background:#e5958f; color:#5c120d; }
.tab { padding:2px 9px; border:1px solid #c3c9d1; border-radius:4px; background:#f1f3f6; font-weight:600; }
.tab.active { background:var(--accent); color:white; border-color:var(--accent); }
.body { flex:1; position:relative; min-height:0; }
.viewer { width:100%; height:100%; }
.folder, .click-info { position:absolute; bottom:6px; z-index:5; font-size:10px; font-family:ui-monospace,Menlo,monospace;
                       background:rgba(255,255,255,.9); padding:2px 6px; }
.folder { left:8px; color:var(--muted); }
.click-info { right:8px; color:var(--ink); font-size:12px; }
.empty { position:absolute; inset:0; display:none; align-items:center; justify-content:center; color:var(--muted); }
.empty.on { display:flex; }
#highlight-chip { display:none; background:#d62728; color:#fff; border-radius:10px; padding:1px 4px 1px 10px; font-size:11px; font-weight:600; }
#highlight-chip.on { display:inline-flex; gap:6px; align-items:center; }
#highlight-chip button { border:0; background:rgba(255,255,255,.25); color:#fff; border-radius:8px; padding:0 6px; }

/* GO panel */
.go-panel { display:none; flex:0 0 46vh; min-height:0; flex-direction:column; overflow:hidden; }
body.go-open .go-panel { display:flex; }
.go-hdr { padding:7px 12px; border-bottom:1px solid #e6e9ed; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.go-hdr h3 { font-size:13px; }
.go-hdr .spacer { flex:1; }
.go-body { flex:1; min-height:0; display:grid; grid-template-columns:minmax(300px, 400px) 1fr; }
.go-summary, .go-matrix { overflow:auto; min-height:0; }
.go-summary { border-right:1px solid #e6e9ed; padding:6px 10px; }
.go-summary p { color:var(--muted); font-size:11px; margin:2px 0 6px; }
table { border-collapse:collapse; }
.go-summary table { width:100%; }
.go-summary td, .go-summary th { padding:3px 4px; text-align:left; font-size:11px; border-bottom:1px solid #eef0f3; }
.go-summary th { color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--panel); }
.role { font-size:10px; color:var(--muted); }
.frac { display:inline-block; min-width:38px; font-variant-numeric:tabular-nums; }
.bar { display:inline-block; width:40px; height:6px; background:#e3e7ec; border-radius:3px; vertical-align:middle; overflow:hidden; }
.bar i { display:block; height:100%; background:#1e7b3c; }
.go-matrix th, .go-matrix td { font-size:11px; padding:2px 5px; border-bottom:1px solid #eef0f3; }
.go-matrix thead th { position:sticky; background:var(--panel); z-index:2; }
.go-matrix thead tr:first-child th { top:0; height:20px; }
.go-matrix thead tr:nth-child(2) th { top:20px; vertical-align:bottom; text-align:left; }
.group-hdr { text-align:center !important; color:var(--muted); font-weight:600; border-bottom:2px solid var(--line); }
.prot-col { writing-mode:vertical-rl; transform:rotate(180deg); white-space:nowrap; padding:4px 2px !important; }
.prot-col .prot { font-size:10px; }
td.cell { width:16px; min-width:16px; padding:0 !important; border-left:1px solid #f1f3f5; }
td.cell i { display:block; width:12px; height:12px; margin:auto; border-radius:2px; }
.group-start { border-left:2px solid var(--line) !important; }
.go-matrix tbody tr { cursor:pointer; }
.go-matrix tbody tr:hover { background:#f3f6fa; }
.go-matrix tbody tr.selected { background:#fde3e3; }
.status { display:inline-block; padding:0 7px; border-radius:8px; font-size:10px; font-weight:700; white-space:nowrap; }
.status.shared { background:var(--shared-bg); color:var(--shared-ink); }
.status.added { background:var(--added-bg); color:var(--added-ink); }
.status.core { background:var(--coreonly-bg); color:var(--coreonly-ink); }
.term-name { max-width:420px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.go-id { font-family:ui-monospace,Menlo,monospace; font-size:10px; color:var(--muted); }
.freq { font-variant-numeric:tabular-nums; color:var(--muted); text-align:right; }
.generic .term-name { color:var(--muted); }
.go-empty { padding:20px; color:var(--muted); }
</style></head>
<body>

<div class="topbar">
  <select id="pdb-filter" title="filter complexes by PDB match class of the YeastMap complex"></select>
  <select id="core-pdb-filter" title="filter complexes by PDB match class of their CP core complex"></select>
  <button id="prev-complex">&#9664;</button>
  <select id="complex-select"></select>
  <button id="next-complex">&#9654;</button>
  <span id="pdb-info"></span>
  <label>Color <select id="color-mode">
    <option value="chain">by chain</option>
    <option value="protein">by protein</option>
    <option value="uniform">uniform</option>
    <option value="mapping">trimmed mapping</option>
    <option value="core">CP core proteins</option>
    <option value="pdbmap">core PDB chain mapping</option>
  </select></label>
  <span id="color-legend"></span>
  <label>pLDDT &ge; <input type="number" id="plddt-threshold" value="70" min="0" max="100" step="1" style="width:55px" disabled></label>
  <label><input type="checkbox" id="show-all" checked> show all</label>
  <button id="go-toggle" class="tab">GO terms</button>
  <span id="highlight-chip"><span id="highlight-text"></span><button id="highlight-clear" title="clear highlight">&times;</button></span>
</div>

<div class="panels">
  <div class="panel">
    <div class="panel-hdr">
      <h3>Trimmed</h3>
      <button id="prev-stoi">&#9664;</button><button id="next-stoi">&#9654;</button>
      <span id="left-stats"></span>
      <span id="left-stoi" class="stoi"></span>
    </div>
    <div class="body">
      <div id="left-viewer" class="viewer"></div>
      <span id="left-folder" class="folder"></span><span id="left-click" class="click-info"></span>
    </div>
  </div>
  <div class="panel">
    <div class="panel-hdr">
      <button class="tab active" data-mode="untrimmed">Untrimmed</button>
      <button class="tab" data-mode="core">CP core</button>
      <button class="tab" data-mode="pdb" title="experimental structure of the CP core complex (exact PDB / exact homology match), loaded from RCSB">PDB (core)</button>
      <span id="right-stats"></span>
      <span id="right-stoi" class="stoi"></span>
    </div>
    <div class="body">
      <div id="right-viewer" class="viewer"></div>
      <span id="right-folder" class="folder"></span><span id="right-click" class="click-info"></span>
      <div id="right-empty" class="empty"></div>
    </div>
  </div>
</div>

<div class="go-panel">
  <div class="go-hdr">
    <h3>GO terms</h3>
    <span id="aspect-tabs"></span>
    <label><input type="checkbox" class="status-filter" value="shared" checked> shared</label>
    <label><input type="checkbox" class="status-filter" value="added" checked> added only</label>
    <label><input type="checkbox" class="status-filter" value="core" checked> core only</label>
    <label title="dataset frequency = fraction of all proteins in the GO table carrying the term; generic terms (e.g. nucleus, protein binding) make proteins look related">
      hide terms on &gt; <input type="number" id="freq-cutoff" value="100" min="1" max="100" step="1" style="width:50px">% of dataset proteins</label>
    <input type="search" id="go-search" placeholder="filter term names" style="width:170px">
    <span class="spacer"></span>
    <span id="go-overview"></span>
  </div>
  <div class="go-body">
    <div class="go-summary" id="go-summary"></div>
    <div class="go-matrix" id="go-matrix"></div>
  </div>
</div>

<script>
const COMPLEXES = %%JSON%%;
const LIGHT_BLUE = '#8fb8e6';   // PDB chain of the same entity as a mapped chain (= another copy of that protein)
const BLUE = '#1565c0', AMBER = '#e69f00', CORE_COLOR = '#00876c', HIGHLIGHT = '#d62728', DIM = '#d3d7dc';
const PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
                 '#0b4f8a', '#b35806', '#1a6b1a', '#8e1a1b', '#5e3c99', '#5d3a2f', '#a6457f', '#0e7c86', '#7a7a10', '#3d3d3d'];
const RANK_NAMES = ['most likely stoi', '2nd most likely stoi', '3rd most likely stoi'];
const PDB_CLASS_NAMES = {exact_pdb_match: 'exact PDB match', exact_homology_match: 'exact homology match', superset: 'superset', partial: 'partial',
                         no_sufficient_match: 'no sufficient match', no_homology_match: 'no homology match'};
const ASPECTS = {BP: 'biological process', MF: 'molecular function', CC: 'cellular component'};
const STATUS_NAMES = {shared: 'shared', added: 'added only', core: 'core only'};
const GENERIC_FREQ = 0.05;   // terms on > 5% of the GO-table proteins (~99th percentile) are greyed out as generic
const leftViewer  = $3Dmol.createViewer('left-viewer',  {backgroundColor: 'white'});
const rightViewer = $3Dmol.createViewer('right-viewer', {backgroundColor: 'white'});
const $ = id => document.getElementById(id);

let complexIndex = 0, complex = null, stoiIndex = 0, rightMode = 'untrimmed';
let leftModel = null, rightModel = null;     // models currently drawn (chain -> UniProt for colors + clicks)
let leftLoadId = 0, rightLoadId = 0;         // ignore results of older clicks
let visibleComplexes = [];                   // indices into COMPLEXES passing the PDB filters
let rightError = null;                       // message if the core PDB could not be loaded
const pdbCache = new Map();                  // PDB id -> Promise<{text, chainOrder, entityOfChain, accessionsOfEntity}>
const parsedPdbById = new Map();             // PDB id -> resolved value of pdbCache (for synchronous use in applyColors)
let goAspect = 'BP', highlight = null;       // highlight = {goId, name, proteins: Set}
const geneNames = new Map();                 // UniProt -> gene name (filled from the UniProt REST API)

async function ungzip(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
}

const esc = s => String(s).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
function badge(text, cls = '') { return `<span class="badge ${cls}">${text}</span> `; }

/* ── core PDB reference: fetched from RCSB, chain -> entity from the mmCIF _atom_site loop ── */
const CIF_TOKEN = /'(?:[^']|'(?=\\S))*'|"(?:[^"]|"(?=\\S))*"|\\S+/g;   // quotes may contain ' / " not followed by a space
const unquote = token => /^(['"]).*\\1$/s.test(token) ? token.slice(1, -1) : token;

function cifTokens(lines) {
  // tokens of a block of mmCIF lines; a ';' text field (several lines) is one token
  const tokens = [];
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].startsWith(';')) { tokens.push(...(lines[i].match(CIF_TOKEN) || []).map(unquote)); continue; }
    const textLines = [lines[i].slice(1)];
    while (++i < lines.length && !lines[i].startsWith(';')) textLines.push(lines[i]);
    if (i === lines.length) throw new Error('unterminated ; text field');
    tokens.push(textLines.join('\\n'));
  }
  return tokens;
}

function parseCifCategory(lines, category) {
  // rows ([{item: value}]) of one category, written as loop_ or as key-value pairs; null if the category is absent
  const prefix = `_${category}.`;
  const start = lines.findIndex(line => line.startsWith(prefix));
  if (start < 0) return null;
  const isLoop = start > 0 && lines[start - 1].trim() === 'loop_';
  let end = start;
  const columns = [];
  if (isLoop) while (lines[end].startsWith(prefix)) columns.push(lines[end++].trim().slice(prefix.length));
  for (; end < lines.length; end++) {   // block ends at '#', loop_, data_ or (loop) the next item name
    const line = lines[end];
    if (line.startsWith(';')) { while (++end < lines.length && !lines[end].startsWith(';')); continue; }
    if (line.startsWith('#') || line.startsWith('loop_') || line.startsWith('data_')) break;
    if (line.startsWith('_') && (isLoop || !line.startsWith(prefix))) break;
  }
  const tokens = cifTokens(lines.slice(isLoop ? start + columns.length : start, end));
  if (!isLoop) {
    const row = {};
    if (tokens.length % 2) throw new Error(`_${category}: odd number of key/value tokens`);
    for (let i = 0; i < tokens.length; i += 2) {
      if (!tokens[i].startsWith(prefix)) throw new Error(`_${category}: unexpected token ${tokens[i]}`);
      row[tokens[i].slice(prefix.length)] = tokens[i + 1];
    }
    return [row];
  }
  if (tokens.length % columns.length) throw new Error(`_${category}: ${tokens.length} values for ${columns.length} columns`);
  const rows = [];
  for (let i = 0; i < tokens.length; i += columns.length)
    rows.push(Object.fromEntries(columns.map((column, j) => [column, tokens[i + j]])));
  return rows;
}

function uniprotOfEntities(lines) {
  // entity id -> Set of UniProt accessions (_struct_ref, db_name UNP; chimeric entities have several)
  const accessionsOfEntity = {};
  for (const ref of parseCifCategory(lines, 'struct_ref') || []) {
    if (ref.db_name !== 'UNP') continue;
    if (!ref.entity_id || !ref.pdbx_db_accession) throw new Error('_struct_ref UNP row without entity_id / pdbx_db_accession');
    (accessionsOfEntity[ref.entity_id] = accessionsOfEntity[ref.entity_id] || new Set()).add(ref.pdbx_db_accession);
  }
  return accessionsOfEntity;
}

function parseAtomSiteChains(cifText) {
  // auth chain ids (= the ids 3Dmol uses for cif) in order of appearance + their polymer entity, first model only
  const lines = cifText.split('\\n');
  let lineIndex = lines.findIndex(line => line.startsWith('_atom_site.'));
  if (lineIndex < 0) throw new Error('no _atom_site loop in mmCIF');
  const columns = [];
  while (lines[lineIndex].startsWith('_atom_site.')) columns.push(lines[lineIndex++].trim().slice('_atom_site.'.length));
  const [groupCol, chainCol, entityCol, modelCol] = ['group_PDB', 'auth_asym_id', 'label_entity_id', 'pdbx_PDB_model_num']
    .map(name => columns.indexOf(name));
  if (groupCol < 0 || chainCol < 0 || entityCol < 0) throw new Error('_atom_site lacks group_PDB / auth_asym_id / label_entity_id');
  const entityOfChain = {};
  let firstModel = null;
  for (; lineIndex < lines.length; lineIndex++) {
    const line = lines[lineIndex];
    if (line.startsWith('#') || line.startsWith('_') || line.startsWith('loop_') || line.startsWith('data_')) break;
    if (!line.trim()) continue;
    const tokens = line.match(CIF_TOKEN);
    if (tokens.length !== columns.length) throw new Error(`_atom_site row with ${tokens.length} fields, expected ${columns.length}`);
    if (modelCol >= 0) {
      firstModel = firstModel ?? tokens[modelCol];
      if (tokens[modelCol] !== firstModel) continue;
    }
    if (tokens[groupCol] !== 'ATOM') continue;   // ligands / waters share the auth chain id but have their own entity
    const chain = tokens[chainCol], entity = tokens[entityCol];
    if (chain in entityOfChain && entityOfChain[chain] !== entity)
      throw new Error(`auth chain ${chain} has polymer atoms of entities ${entityOfChain[chain]} and ${entity}`);
    entityOfChain[chain] = entity;
  }
  const chainOrder = Object.keys(entityOfChain);
  if (!chainOrder.length) throw new Error('no ATOM records');
  return {chainOrder, entityOfChain, accessionsOfEntity: uniprotOfEntities(lines)};
}

function fetchPdbReference(pdbId) {
  if (!pdbCache.has(pdbId)) {
    const request = fetch(`https://files.rcsb.org/download/${pdbId}.cif`)
      .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.text(); })
      .then(text => ({text, ...parseAtomSiteChains(text)}))
      .then(parsed => { parsedPdbById.set(pdbId, parsed); return parsed; });
    request.catch(() => pdbCache.delete(pdbId));   // allow a retry after e.g. a network error
    pdbCache.set(pdbId, request);
  }
  return pdbCache.get(pdbId);
}

function buildPdbModel(corePdb, parsed, coreProteins) {
  const model = corePdb.mapping_source === 'uniprot' ? uniprotChainMapping(corePdb, parsed, coreProteins)
                                                     : exactMatchesChainMapping(corePdb, parsed);
  const mappedProteins = new Set(Object.values(model.chains));
  return {...model, kind: 'pdb', pdb_id: corePdb.pdb_id, match_class: corePdb.match_class, mapping_source: corePdb.mapping_source,
          model_chains: parsed.chainOrder, entityOfChain: parsed.entityOfChain, chainsNotIn3Dmol: [],
          missingProteins: [...coreProteins].filter(protein => !mappedProteins.has(protein))};
}

function uniprotChainMapping(corePdb, parsed, coreProteins) {
  // every chain whose entity carries exactly one CP core UniProt accession -> that protein (all copies count as mapped)
  const chains = {}, ambiguousEntities = new Set();
  for (const chain of parsed.chainOrder) {
    const entity = parsed.entityOfChain[chain];
    const coreAccessions = [...(parsed.accessionsOfEntity[entity] || [])].filter(accession => coreProteins.has(accession));
    if (coreAccessions.length > 1) ambiguousEntities.add(entity);
    else if (coreAccessions.length === 1) chains[chain] = coreAccessions[0];
  }
  return {chains, mappedChains: new Set(Object.keys(chains)), missingChains: [], ambiguousEntities: [...ambiguousEntities]};
}

function exactMatchesChainMapping(corePdb, parsed) {
  // chains: mapped chain -> its CP protein; other chains of the same entity -> same protein (further copies),
  // unless two different CP proteins are mapped onto chains of that entity (then only the mapped chains get a protein)
  const mappedChainToProtein = corePdb.chain_to_protein;
  const proteinsOfEntity = {};
  for (const [chain, protein] of Object.entries(mappedChainToProtein)) {
    const entity = parsed.entityOfChain[chain];
    if (entity === undefined) continue;   // reported as missing chain below
    (proteinsOfEntity[entity] = proteinsOfEntity[entity] || new Set()).add(protein);
  }
  const ambiguousEntities = Object.keys(proteinsOfEntity).filter(entity => proteinsOfEntity[entity].size > 1);
  const chains = {};
  for (const chain of parsed.chainOrder) {
    const entityProteins = proteinsOfEntity[parsed.entityOfChain[chain]];
    if (chain in mappedChainToProtein) chains[chain] = mappedChainToProtein[chain];
    else if (entityProteins && entityProteins.size === 1) chains[chain] = [...entityProteins][0];
  }
  return {chains, mappedChains: new Set(Object.keys(mappedChainToProtein)),
          missingChains: Object.keys(mappedChainToProtein).filter(chain => !(chain in parsed.entityOfChain)), ambiguousEntities};
}

/* ── proteins: colors, UniProt links, gene names ── */
function proteinOrder(c) { return [...c.protein_groups.core_kept, ...c.protein_groups.core_dropped, ...c.protein_groups.added]; }
function proteinColor(acc) { return complex.colorOf.get(acc) || '#888888'; }
function protLabel(acc, withSwatch = true) {
  const sw = withSwatch ? `<i class="sw" style="background:${proteinColor(acc)}"></i>` : '';
  return `<a class="prot" href="https://www.uniprot.org/uniprotkb/${acc.split('-')[0]}/entry" target="_blank" rel="noopener" `
       + `title="open ${acc} in UniProt">${sw}<span class="gene" data-gene="${acc}">${esc(geneNames.get(acc) || '')}</span>${acc}</a>`;
}
function fillGeneNames() {
  document.querySelectorAll('[data-gene]').forEach(el => { el.textContent = geneNames.get(el.dataset.gene) || ''; });
}
async function fetchGeneNames(accessions) {
  const missing = accessions.filter(acc => !geneNames.has(acc));
  if (!missing.length) return;
  missing.forEach(acc => geneNames.set(acc, ''));   // mark as requested
  const query = missing.map(acc => 'accession:' + acc.split('-')[0]).join(' OR ');
  const url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession,gene_primary,gene_oln&format=json&size=500&query='
            + encodeURIComponent(query);
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('HTTP ' + response.status);
    for (const entry of (await response.json()).results) {
      const gene = (entry.genes || [])[0] || {};
      const name = gene.geneName ? gene.geneName.value : (gene.orderedLocusNames || [{}])[0].value;
      if (name) geneNames.set(entry.primaryAccession, name);
    }
    fillGeneNames();
  } catch (err) {
    console.warn('UniProt gene names not loaded (showing ids only):', err);
  }
}

function stoiHtml(counts) {
  return Object.keys(counts).sort().map(p => protLabel(p) + `<span class="copies">x${counts[p]}</span>`).join('');
}
function chainStats(model) {
  const nUnique = new Set(model.model_chains.map(c => model.chains[c])).size;
  return badge(model.model_chains.length + ' chains') + badge(nUnique + ' unique proteins');
}

/* what the right panel shows: untrimmed model of the same stoi, or the matched CP core model */
function rightInfo() {
  const stoi = complex.stois[stoiIndex];
  if (rightMode === 'pdb') return null;   // drawn from rightModel (fetched), see pdbStats
  if (rightMode === 'untrimmed') return stoi.untrimmed ? {model: stoi.untrimmed, counts: stoi.counts} : null;
  return complex.core_models[stoi.core_folder] || null;
}

function rcsbLink(pdbId) {
  return `<a href="https://www.rcsb.org/structure/${pdbId}" target="_blank" rel="noopener" title="open in RCSB">${pdbId.toUpperCase()}</a>`;
}
function pdbBadge(c) {
  const match = c.pdb_match, coreMatch = c.core_pdb_match;
  return badge('PDB: ' + PDB_CLASS_NAMES[match.match_class] + (match.pdb_id ? ' ' + rcsbLink(match.pdb_id) : ''), 'pdb-' + match.match_class)
    + badge(`core ${c.cp_complex_ac} PDB: ` + PDB_CLASS_NAMES[coreMatch.match_class] + (coreMatch.pdb_id ? ' ' + rcsbLink(coreMatch.pdb_id) : ''),
            'pdb-' + coreMatch.match_class);
}

/* header of the PDB (core) tab: mapping CP protein -> chain(s), warnings if the mapping is missing / inconsistent */
function pdbStats(model) {
  const nCore = complex.core_proteins.length;
  const mappedProteins = new Set(Object.values(model.chains));
  const homology = model.match_class === 'exact_homology_match';
  let html = badge('PDB ' + rcsbLink(model.pdb_id), 'pdb-' + model.match_class) + badge(PDB_CLASS_NAMES[model.match_class], 'pdb-' + model.match_class)
    + badge(model.model_chains.length + ' polymer chains')
    + badge(`mapped: ${model.mappedChains.size} chains / ${mappedProteins.size} of ${nCore} CP core proteins`, 'core')
    + badge(`${Object.keys(model.chains).length - model.mappedChains.size} same-entity copies`, 'core')
    + badge(`${model.model_chains.length - Object.keys(model.chains).length} chains not mapped`);
  if (homology) html += badge('chains are homologs of the CP proteins', 'interesting');
  html += badge(model.mapping_source === 'uniprot' ? 'mapped by UniProt id (mmCIF _struct_ref)' : 'mapped via exact_matches.csv');
  if (model.missingProteins.length) html += badge('CP core proteins not in PDB: ' + model.missingProteins.join(', '), 'warn');
  if (model.missingChains.length) html += badge('mapped chains not in mmCIF: ' + model.missingChains.join(', '), 'warn');
  if (model.chainsNotIn3Dmol.length) html += badge('chains not found in 3Dmol model: ' + model.chainsNotIn3Dmol.join(', '), 'warn');
  if (model.ambiguousEntities.length) html += badge('entities with >1 CP protein (left unmapped): ' + model.ambiguousEntities.join(', '), 'warn');
  return html;
}
function pdbMappingHtml(model) {
  // per CP core protein: mapped chain (+ same-entity copies)
  const chainsOfProtein = {};
  for (const chain of model.model_chains) if (model.chains[chain]) (chainsOfProtein[model.chains[chain]] = chainsOfProtein[model.chains[chain]] || []).push(chain);
  return complex.core_proteins.map(protein => {
    const chains = chainsOfProtein[protein] || [];
    const mapped = chains.filter(chain => model.mappedChains.has(chain)), copies = chains.filter(chain => !model.mappedChains.has(chain));
    const text = chains.length ? 'chain ' + mapped.join(',') + (copies.length ? ` (+${copies.join(',')})` : '') : 'not mapped';
    return protLabel(protein) + `<span class="copies">${text}</span>`;
  }).join('');
}

function updateStats() {
  const stoi = complex.stois[stoiIndex];
  const trimmed = stoi.trimmed;
  const rank = badge(RANK_NAMES[stoiIndex] + ` (${stoiIndex + 1}/${complex.stois.length})`);
  const interesting = stoi.interesting ? badge('interesting stoi', 'interesting') : '';
  const core = complex.core_models[stoi.core_folder];

  $('left-stats').innerHTML = badge('CF ' + trimmed.conf.toFixed(2)) + badge('pred ' + stoi.preds.join('/'))
    + badge('p = ' + stoi.probability.toFixed(2)) + rank + badge(trimmed.n_outputs + ' outputs')
    + chainStats(trimmed) + interesting
    + badge(`CP core proteins: ${stoi.core_proteins_in_left}/${complex.core_proteins.length}`, 'core')
    + badge(`their chains: ${stoi.core_chains_in_left}` + (core ? ` (CP core: ${core.model.model_chains.length})` : ''), 'core');
  $('left-stoi').innerHTML = stoiHtml(stoi.counts);
  $('left-folder').textContent = trimmed.folder + ' / ' + trimmed.model;

  if (rightMode === 'pdb') {
    const corePdb = complex.core_pdb;
    const loaded = rightModel && rightModel.kind === 'pdb' && corePdb && rightModel.pdb_id === corePdb.pdb_id ? rightModel : null;
    const coreMatch = complex.core_pdb_match;
    $('right-stats').innerHTML = loaded ? pdbStats(loaded) : '';
    $('right-stoi').innerHTML = loaded ? pdbMappingHtml(loaded) : '';
    $('right-folder').textContent = loaded ? `RCSB ${loaded.pdb_id}.cif (asymmetric unit, first model)` : '';
    $('right-empty').textContent = !corePdb
      ? `CP core ${complex.cp_complex_ac}: ${PDB_CLASS_NAMES[coreMatch.match_class]}${coreMatch.pdb_id ? ' (' + coreMatch.pdb_id + ')' : ''} -> no exact PDB reference`
      : rightError ? `could not load ${corePdb.pdb_id} from RCSB: ${rightError}` : `loading ${corePdb.pdb_id} from RCSB ...`;
    $('right-empty').classList.toggle('on', !loaded);
    return;
  }
  const right = rightInfo();
  if (!right) {
    $('right-stats').innerHTML = '';
    $('right-empty').textContent = rightMode === 'core' ? 'no CP core output for ' + complex.cp_complex_ac
                                                        : 'no untrimmed output for this stoichiometry';
  } else if (rightMode === 'untrimmed') {
    const delta = right.model.conf - trimmed.conf;
    $('right-stats').innerHTML = badge(`CF ${right.model.conf.toFixed(2)} (${delta >= 0 ? '+' : ''}${delta.toFixed(2)} vs trimmed)`)
      + badge('pred ' + stoi.preds.join('/')) + rank + badge(right.model.n_outputs + ' outputs')
      + chainStats(right.model) + interesting;
  } else {
    const match = stoi.core_difference === 0 ? 'same stoi as left (shared proteins)'
                                             : `closest stoi (${stoi.core_difference} copies differ)`;
    $('right-stats').innerHTML = badge('CF ' + right.model.conf.toFixed(2))
      + badge(complex.cp_complex_ac + ' ' + right.labels.join('/'), 'core') + badge(match, 'core')
      + badge(right.model.n_outputs + ' outputs') + chainStats(right.model)
      + (Object.values(right.counts).some(n => n > 1) ? badge('interesting stoi', 'interesting') : '');
  }
  $('right-stoi').innerHTML = right ? stoiHtml(right.counts) : '';
  $('right-folder').textContent = right ? right.model.folder + ' / ' + right.model.model : '';
  $('right-empty').classList.toggle('on', !right);
}

function onAtomClick(side, model, atom) {
  if (model.kind === 'pdb') {
    const protein = model.chains[atom.chain];
    const proteinText = !protein ? 'not mapped'
      : (model.mappedChains.has(atom.chain) ? '' : 'copy of ') + (model.match_class === 'exact_homology_match' ? 'homolog of ' : '') + protLabel(protein);
    $(side + '-click').innerHTML = `chain ${atom.chain} (entity ${model.entityOfChain[atom.chain] ?? '-'})  ${proteinText}  `
                                 + `${atom.resn} ${atom.resi}  B ${atom.b.toFixed(1)}`;
    return;
  }
  $(side + '-click').innerHTML =
    `chain ${atom.chain}  ${protLabel(model.chains[atom.chain])}  ${atom.resn} ${atom.resi}  pLDDT ${atom.b.toFixed(1)}`;
}

function drawModel(viewer, model, side, text, format) {
  viewer.clear();
  $(side + '-click').textContent = '';
  if (!model) return;
  viewer.addModel(text, format);
  viewer.setClickable({}, true, atom => onAtomClick(side, model, atom));
}

async function loadLeft() {
  const id = ++leftLoadId, model = complex.stois[stoiIndex].trimmed;
  const text = await ungzip(model.pdb_gz);
  if (id !== leftLoadId) return false;   // a newer click started -> drop this one before touching the viewer
  drawModel(leftViewer, model, 'left', text, 'pdb'); leftModel = model;
  return true;
}

async function loadRight() {
  const id = ++rightLoadId;
  let model = null, text = null, format = 'pdb', error = null;
  if (rightMode === 'pdb') {
    rightModel = null; rightError = null; updateStats();   // shows 'loading ...'
    if (complex.core_pdb) {
      try {
        const parsed = await fetchPdbReference(complex.core_pdb.pdb_id);
        model = buildPdbModel(complex.core_pdb, parsed, new Set(complex.core_proteins)); text = parsed.text; format = 'cif';
      } catch (err) {
        error = String(err.message || err);
        console.error('core PDB not loaded:', err);
      }
    }
  } else {
    const right = rightInfo();
    model = right ? right.model : null;
    text = model ? await ungzip(model.pdb_gz) : null;
  }
  if (id !== rightLoadId) return false;
  drawModel(rightViewer, model, 'right', text, format); rightModel = model; rightError = error;
  if (model && model.kind === 'pdb') {
    // check that 3Dmol uses the same (auth) chain ids as the _atom_site parse / exact_matches.csv
    const chainsIn3Dmol = new Set(rightViewer.selectedAtoms({}).map(atom => atom.chain));
    model.chainsNotIn3Dmol = model.model_chains.filter(chain => !chainsIn3Dmol.has(chain));
  }
  updateStats();
  return true;
}

/* ── 3D colors ── */
function cartoonStyle(mode, model, isLeft, trimmedResidues, coreProteins, pdbMappedProteins) {
  const proteinOf = a => model.chains[a.chain];
  const isPdb = model.kind === 'pdb';
  if (highlight)               return {colorfunc: a => highlight.proteins.has(proteinOf(a)) ? HIGHLIGHT : DIM};
  if (mode === 'chain') {
    const chainColor = Object.fromEntries(model.model_chains.map((c, i) => [c, PALETTE[i % PALETTE.length]]));
    return {colorfunc: a => chainColor[a.chain]};
  }
  if (mode === 'protein')      return {colorfunc: a => proteinColor(proteinOf(a))};
  if (mode === 'uniform')      return {color: AMBER};
  if (mode === 'mapping' && isPdb) return {color: DIM};   // PDB residue numbers are not UniProt positions
  if (mode === 'mapping')      return isLeft ? {color: BLUE}
                                             : {colorfunc: a => trimmedResidues.has(proteinOf(a) + ':' + a.resi) ? BLUE : AMBER};
  if (mode === 'pdbmap' && isPdb) return {colorfunc: a => model.mappedChains.has(a.chain) ? BLUE : proteinOf(a) ? LIGHT_BLUE : AMBER};
  if (mode === 'pdbmap')       return {colorfunc: a => pdbMappedProteins.has(proteinOf(a)) ? BLUE : AMBER};
  return {colorfunc: a => coreProteins.has(proteinOf(a)) ? CORE_COLOR : AMBER};
}

function applyColors() {
  if (!leftModel) return;
  const mode = $('color-mode').value;
  const coreProteins = new Set(complex.core_proteins);
  const pdbMappedProteins = corePdbMappedProteins(mode);
  updateLegend(mode);
  // residues are matched by (UniProt, residue number), so all copies of a protein are treated the same
  const trimmedResidues = new Set(leftViewer.selectedAtoms({atom: 'CA'}).map(a => leftModel.chains[a.chain] + ':' + a.resi));
  for (const [viewer, model, isLeft] of [[leftViewer, leftModel, true], [rightViewer, rightModel, false]]) {
    if (!model) continue;
    viewer.setStyle({}, {cartoon: cartoonStyle(mode, model, isLeft, trimmedResidues, coreProteins, pdbMappedProteins)});
  }
  hideLowPlddt();
  leftViewer.render(); rightViewer.render();
}

function hideLowPlddt() {
  // pLDDT is read from the B-factor column; the trimmed mapping above still uses all residues
  // (not applied to the PDB structure: its B-factor column holds crystallographic B-factors)
  if ($('show-all').checked) return;
  const threshold = Number($('plddt-threshold').value) || 0;
  const predictedViewers = rightModel && rightModel.kind === 'pdb' ? [leftViewer] : [leftViewer, rightViewer];
  predictedViewers.forEach(v => v.setStyle({predicate: a => a.b < threshold}, {}));
}

function corePdbMappedProteins(mode) {
  // CP proteins with a chain in the core PDB; for the UniProt-based mapping this needs the mmCIF -> fetched on demand
  const corePdb = complex.core_pdb;
  if (!corePdb) return new Set();
  if (corePdb.mapping_source === 'exact_matches') return new Set(Object.values(corePdb.chain_to_protein));
  const parsed = parsedPdbById.get(corePdb.pdb_id);
  if (parsed) return new Set(Object.values(buildPdbModel(corePdb, parsed, new Set(complex.core_proteins)).chains));
  if (mode === 'pdbmap') {
    const requestedFor = complex;
    fetchPdbReference(corePdb.pdb_id).then(() => { if (complex === requestedFor) applyColors(); })
      .catch(err => console.error('core PDB not loaded:', err));
  }
  return new Set();
}

function updateLegend(mode) {
  const swatch = (color, text) => `<i class="sw" style="background:${color}"></i>${text} `;
  if (mode === 'pdbmap') {
    $('color-legend').innerHTML = !complex.core_pdb ? 'core has no exact PDB reference'
      : complex.core_pdb.mapping_source === 'uniprot' && !parsedPdbById.has(complex.core_pdb.pdb_id)
        ? `loading ${complex.core_pdb.pdb_id} for the UniProt mapping ...`
      : swatch(BLUE, 'protein / chain mapped to ' + complex.core_pdb.pdb_id)
        + (complex.core_pdb.mapping_source === 'exact_matches' ? swatch(LIGHT_BLUE, 'same-entity copy (PDB)') : '')
        + swatch(AMBER, 'not mapped');
  } else if (mode === 'mapping' && rightMode === 'pdb') {
    $('color-legend').textContent = 'trimmed mapping: residue numbers not comparable with the PDB -> PDB grey';
  } else {
    $('color-legend').textContent = '';
  }
}

function setHighlight(newHighlight) {
  highlight = newHighlight;
  $('highlight-chip').classList.toggle('on', !!highlight);
  $('highlight-text').textContent = highlight ? 'highlighted: ' + highlight.name : '';
  document.querySelectorAll('#go-matrix tbody tr').forEach(tr =>
    tr.classList.toggle('selected', !!highlight && tr.dataset.go === highlight.goId));
  applyColors();
}

/* ── GO panel ── */
function goTermsWithStatus() {
  const added = new Set(complex.protein_groups.added);
  return complex.go_terms.map(([goId, aspect, name, freq, proteins]) => {
    const inAdded = proteins.some(p => added.has(p)), inCore = proteins.some(p => !added.has(p));
    return {goId, aspect, name, freq, proteins: new Set(proteins),
            status: inAdded && inCore ? 'shared' : inAdded ? 'added' : 'core'};
  });
}

function renderGo() {
  if (!document.body.classList.contains('go-open')) return;
  const groups = complex.protein_groups;
  const freqCutoff = (Number($('freq-cutoff').value) || 100) / 100;
  const specificTerms = goTermsWithStatus().filter(t => t.freq <= freqCutoff);   // the frequency filter also feeds the summary

  // aspect tabs with counts
  $('aspect-tabs').innerHTML = Object.keys(ASPECTS).map(aspect =>
    `<button class="tab ${aspect === goAspect ? 'active' : ''}" data-aspect="${aspect}" title="${ASPECTS[aspect]}">`
    + `${aspect} ${specificTerms.filter(t => t.aspect === aspect).length}</button>`).join(' ');
  $('aspect-tabs').querySelectorAll('button').forEach(b => b.onclick = () => { goAspect = b.dataset.aspect; renderGo(); });

  // per protein: its terms shared with the other side (added <-> CP core), per aspect
  const addedSet = new Set(groups.added);
  const termsOf = (protein, aspect) => specificTerms.filter(t => t.aspect === aspect && t.proteins.has(protein));
  const sharedCount = (protein, aspect) => termsOf(protein, aspect).filter(t => t.status === 'shared').length;
  const fracCell = (protein, aspect) => {
    const total = termsOf(protein, aspect).length, shared = sharedCount(protein, aspect);
    if (!total) return '<td class="role">no terms</td>';
    return `<td><span class="frac">${shared}/${total}</span><span class="bar"><i style="width:${100 * shared / total}%"></i></span></td>`;
  };
  const roleRows = (proteins, role) => proteins.map(p =>
    `<tr><td>${protLabel(p)}<div class="role">${role}</div></td>${Object.keys(ASPECTS).map(a => fracCell(p, a)).join('')}</tr>`).join('');
  $('go-summary').innerHTML =
    `<p>Terms of each protein that also occur on the other side: added proteins vs all CP core proteins, `
    + `CP core proteins vs added proteins. Respects the frequency cutoff.</p>`
    + `<table><thead><tr><th>protein</th>${Object.keys(ASPECTS).map(a => `<th>${a}</th>`).join('')}</tr></thead><tbody>`
    + roleRows(groups.added, 'added by YeastMap') + roleRows(groups.core_kept, 'CP core, in YM')
    + roleRows(groups.core_dropped, 'CP core, not in YM') + '</tbody></table>';

  // overview: how many added proteins share at least one term with the core
  $('go-overview').innerHTML = groups.added.length
    ? 'added proteins sharing &ge;1 term with core: ' + Object.keys(ASPECTS).map(a =>
        badge(`${a} ${groups.added.filter(p => sharedCount(p, a) > 0).length}/${groups.added.length}`, 'core')).join('')
    : badge('no added proteins');

  // term x protein matrix
  const shownStatus = new Set([...document.querySelectorAll('.status-filter:checked')].map(cb => cb.value));
  const search = $('go-search').value.trim().toLowerCase();
  const statusOrder = {shared: 0, added: 1, core: 2};
  const rows = specificTerms
    .filter(t => t.aspect === goAspect && shownStatus.has(t.status)
                 && (!search || t.name.toLowerCase().includes(search) || t.goId.toLowerCase().includes(search)))
    .sort((a, b) => statusOrder[a.status] - statusOrder[b.status] || b.proteins.size - a.proteins.size
                    || a.freq - b.freq || a.name.localeCompare(b.name));
  const columnGroups = [['CP core, in YM', groups.core_kept], ['CP core, not in YM', groups.core_dropped],
                        ['added by YeastMap', groups.added]].filter(([, proteins]) => proteins.length);
  const columns = columnGroups.flatMap(([, proteins]) => proteins.map((p, i) => ({protein: p, groupStart: i === 0})));
  if (!rows.length) {
    $('go-matrix').innerHTML = `<div class="go-empty">no ${ASPECTS[goAspect]} terms with the current filters</div>`;
    return;
  }
  const header1 = '<tr><th colspan="4"></th>' + columnGroups.map(([label, proteins]) =>
    `<th class="group-hdr group-start" colspan="${proteins.length}">${label}</th>`).join('') + '</tr>';
  const header2 = '<tr><th>status</th><th>term</th><th>GO id</th><th title="fraction of all proteins in the GO table carrying the term">dataset</th>'
    + columns.map(col => `<th class="prot-col ${col.groupStart ? 'group-start' : ''}">${protLabel(col.protein)}</th>`).join('') + '</tr>';
  const body = rows.map(t => {
    const generic = t.freq > GENERIC_FREQ ? 'generic' : '';
    const cells = columns.map(col => `<td class="cell ${col.groupStart ? 'group-start' : ''}">`
      + (t.proteins.has(col.protein) ? `<i style="background:${proteinColor(col.protein)}"></i>` : '') + '</td>').join('');
    return `<tr data-go="${t.goId}" class="${generic} ${highlight && highlight.goId === t.goId ? 'selected' : ''}">`
      + `<td><span class="status ${t.status}">${STATUS_NAMES[t.status]}</span></td>`
      + `<td class="term-name" title="${esc(t.name)}">${esc(t.name)}</td>`
      + `<td><a class="go-id" href="https://www.ebi.ac.uk/QuickGO/term/${t.goId}" target="_blank" rel="noopener">${t.goId}</a></td>`
      + `<td class="freq">${(100 * t.freq).toFixed(t.freq < 0.01 ? 1 : 0)}%</td>${cells}</tr>`;
  }).join('');
  $('go-matrix').innerHTML = `<table><thead>${header1}${header2}</thead><tbody>${body}</tbody></table>`;
  const termById = new Map(rows.map(t => [t.goId, t]));
  $('go-matrix').querySelectorAll('tbody tr').forEach(tr => tr.onclick = event => {
    if (event.target.closest('a')) return;   // links open UniProt / QuickGO, they don't toggle the highlight
    const term = termById.get(tr.dataset.go);
    setHighlight(highlight && highlight.goId === term.goId ? null : {goId: term.goId, name: term.name, proteins: term.proteins});
  });
}

function toggleGoPanel() {
  const open = document.body.classList.toggle('go-open');
  $('go-toggle').classList.toggle('active', open);
  renderGo();
  requestAnimationFrame(() => { leftViewer.resize(); rightViewer.resize(); leftViewer.render(); rightViewer.render(); });
}

/* ── navigation ── */
async function showStoi(index) {
  stoiIndex = index;
  $('prev-stoi').disabled = index === 0;
  $('next-stoi').disabled = index === complex.stois.length - 1;
  updateStats();
  const [leftDone, rightDone] = await Promise.all([loadLeft(), loadRight()]);
  if (!leftDone || !rightDone) return;
  updateStats();
  applyColors();
  leftViewer.zoomTo(); rightViewer.zoomTo();
  leftViewer.render(); rightViewer.render();
}

async function setRightMode(mode) {
  rightMode = mode;
  document.querySelectorAll('.tab[data-mode]').forEach(tab => tab.classList.toggle('active', tab.dataset.mode === mode));
  updateStats();
  if (!(await loadRight())) return;
  applyColors();
  rightViewer.zoomTo(); rightViewer.render();
}

function selectComplex(index) {
  complexIndex = index;
  complex = COMPLEXES[index];
  if (!complex.colorOf) complex.colorOf = new Map(proteinOrder(complex).map((p, i) => [p, PALETTE[i % PALETTE.length]]));
  const position = visibleComplexes.indexOf(index);
  $('complex-select').value = index;
  $('prev-complex').disabled = position <= 0;
  $('next-complex').disabled = position === visibleComplexes.length - 1;
  $('pdb-info').innerHTML = pdbBadge(complex);
  highlight = null;
  $('highlight-chip').classList.remove('on');
  renderGo();
  showStoi(0);
  fetchGeneNames(proteinOrder(complex));
}

function optionText(c) {
  const ids = c.identifiers.length > 50 ? c.identifiers.slice(0, 47) + '...' : c.identifiers;
  const pdb = c.pdb_match.pdb_id ? ' ' + c.pdb_match.pdb_id.toUpperCase() : '';
  const corePdb = c.core_pdb_match.pdb_id ? ' ' + c.core_pdb_match.pdb_id.toUpperCase() : '';
  return `${c.complex_ac}  (CF ${c.stois[0].trimmed.conf.toFixed(1)}, PDB ${PDB_CLASS_NAMES[c.pdb_match.match_class]}${pdb}, `
       + `core ${c.cp_complex_ac} PDB ${PDB_CLASS_NAMES[c.core_pdb_match.match_class]}${corePdb})  ${ids}`;
}

function rebuildComplexOptions() {
  const pdbClass = $('pdb-filter').value, corePdbClass = $('core-pdb-filter').value;
  visibleComplexes = COMPLEXES.map((_, i) => i).filter(i =>
    (pdbClass === 'all' || COMPLEXES[i].pdb_match.match_class === pdbClass)
    && (corePdbClass === 'all' || COMPLEXES[i].core_pdb_match.match_class === corePdbClass));
  const select = $('complex-select');
  select.innerHTML = '';
  visibleComplexes.forEach(i => select.add(new Option(optionText(COMPLEXES[i]), i)));
  if (!visibleComplexes.length) select.add(new Option('no complex with this combination of PDB filters', ''));
  select.disabled = !visibleComplexes.length;
}

function fillClassFilter(selectId, label, matchKey) {
  $(selectId).add(new Option(`${label}: all (${COMPLEXES.length})`, 'all'));
  Object.keys(PDB_CLASS_NAMES).forEach(pdbClass => {
    const n = COMPLEXES.filter(c => c[matchKey].match_class === pdbClass).length;
    if (n) $(selectId).add(new Option(`${label}: ${PDB_CLASS_NAMES[pdbClass]} (${n})`, pdbClass));
  });
}
fillClassFilter('pdb-filter', 'YM PDB', 'pdb_match');
fillClassFilter('core-pdb-filter', 'core PDB', 'core_pdb_match');
rebuildComplexOptions();

function onPdbFilterChange() {
  rebuildComplexOptions();
  if (!visibleComplexes.length) {   // the two filters combined can be empty -> keep showing the current complex
    $('prev-complex').disabled = $('next-complex').disabled = true;
    return;
  }
  selectComplex(visibleComplexes.includes(complexIndex) ? complexIndex : visibleComplexes[0]);
}
$('pdb-filter').onchange = onPdbFilterChange;
$('core-pdb-filter').onchange = onPdbFilterChange;
$('complex-select').onchange = () => selectComplex(Number($('complex-select').value));
$('prev-complex').onclick = () => selectComplex(visibleComplexes[visibleComplexes.indexOf(complexIndex) - 1]);
$('next-complex').onclick = () => selectComplex(visibleComplexes[visibleComplexes.indexOf(complexIndex) + 1]);
$('prev-stoi').onclick = () => showStoi(stoiIndex - 1);
$('next-stoi').onclick = () => showStoi(stoiIndex + 1);
document.querySelectorAll('.tab[data-mode]').forEach(tab => tab.onclick = () => setRightMode(tab.dataset.mode));
$('color-mode').onchange = applyColors;
$('plddt-threshold').oninput = applyColors;
$('show-all').onchange = function() {
  $('plddt-threshold').disabled = this.checked;
  applyColors();
};
$('go-toggle').onclick = toggleGoPanel;
$('highlight-clear').onclick = () => setHighlight(null);
document.querySelectorAll('.status-filter').forEach(cb => cb.onchange = renderGo);
$('freq-cutoff').oninput = renderGo;
$('go-search').oninput = renderGo;

selectComplex(visibleComplexes[0]);
</script>
</body></html>
"""

assert "%%JSON%%" in HTML
OUT.parent.mkdir(parents=True, exist_ok=True)
viewer_json = json.dumps(viewer_complexes).replace("</", "<\\/")   # a '</script>' inside a string would end the script
OUT.write_text(HTML.replace("%%JSON%%", viewer_json), encoding="utf-8")
print(f"Written: {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")