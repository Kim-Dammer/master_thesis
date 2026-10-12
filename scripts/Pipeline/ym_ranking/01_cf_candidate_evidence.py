#!/usr/bin/env python3
"""
Evidence per YeastMap complex that its CombFold prediction makes sense (read by the 09 viewer).
Same complexes as the viewer: trimmed run, best output over pred_1/2/3 with confidence > CONF_THRESHOLD;
the best (highest-confidence) trimmed stoichiometry is scored.

structure: can the CP core part be trusted?
  tm_core_model_vs_pdb   : best CombFold model of the CP core complex (CP_RUN, over pred_1/2/3/true) vs the core PDB
  tm_trimmed_core_vs_pdb : the CP core chains of the best trimmed YeastMap model vs the core PDB
                           (did the added proteins change how the core is assembled?)
  Only for CP cores with exact_pdb_match / exact_homology_match. The PDB (RCSB mmCIF, asymmetric unit, first model,
  no ligands / waters) is reduced to the chains mapped to CP core proteins, same mapping as the viewer:
    - exact_matches.csv: one chain per protein + the other chains of the same entity (further copies)
    - exact_pdb_match without a row there: chains whose entity has the protein's UniProt accession in _struct_ref
  Cores that cannot be fully mapped (e.g. 1X3Z: Png1 referenced as GenBank in _struct_ref) are skipped: no TM-scores,
  reason in core_pdb_skip_reason, listed in the log.
  US-align -mm 1 -ter 0 (chains matched by US-align). tm_* = TM-score normalised by the CombFold structure
  (how much of the prediction is reproduced in the PDB); *_norm_by_pdb normalised by the reduced PDB.

biology, per added protein (added = in the YeastMap complex, not in the CP core):
  GO : the most specific GO term it shares with any CP core protein. Terms are propagated over is_a + part_of;
       specificity = fraction of annotated yeast proteins carrying the term (same aspect). NOT and IPI annotations
       are excluded (as in notebook 30). 'supported' = shared term on <= SPECIFIC_TERM_MAX_FREQ of the proteins.
  IPI: GOA annotations with evidence IPI ('inferred from physical interaction') between the added protein and a
       CP core protein, in either direction (with/from column), with their PMIDs.

novelty: PDB match class of the YeastMap complex itself (all_YM_pdb_matches.csv).

Outputs (OUT_DIR): candidate_evidence.csv (one row per YeastMap complex),
                   added_protein_evidence.csv (one row per YeastMap complex x added protein),
                   usalign/<pair>/ (US-align input + stdout), PDB mmCIFs are cached in MMCIF_DIR.
"""
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import gemmi
import polars as pl
from goatools.obo_parser import GODag
from procompa import get_project_root
from tqdm import tqdm

PRJ_ROOT = get_project_root()
DATA = PRJ_ROOT / "data"
TRIMMED_RUN = "22_YM_pae_trimmed"
CP_RUN = "23_all_CP_pae"
YEASTMAP_TSV = DATA / "Pipeline/12_Yeastmap/input_12_yeastmap.tsv"   # YeastMap complex -> CP complex
YM_PDB_MATCHES_CSV = DATA / "YM_CP_complete_complex_pdb_mapping/all_YM_pdb_matches.csv"
CORE_PDB_MATCHES_CSV = DATA / "complete_complex_pdb_mapping_v2/all_pdb_matches_with_match_class.csv"
CORE_PDB_CHAIN_HITS_CSV = DATA / "complete_complex_pdb_mapping_v2/homology_pdb_mapping/exact_matches.csv"
GAF_PATH = DATA / "GO/goa_yeast.gaf.gz"
GO_OBO_PATH = DATA / "GO/go-basic.obo"
USALIGN_BIN = PRJ_ROOT / "tools/usalign/USalign/USalign"
MMCIF_DIR = DATA / "pdb_mmcif"                                    # downloaded PDB entries (cache)
OUT_DIR = DATA / "Pipeline/viewer/10_cf_candidate_evidence"

CONF_THRESHOLD = 75               # same as the viewer
CORE_PDB_SHOWN_CLASSES = {"exact_pdb_match", "exact_homology_match"}
SPECIFIC_TERM_MAX_FREQ = 0.01     # shared GO term counts as support if on <= 1% of the annotated yeast proteins
GO_EXCLUDED_EVIDENCE = {"IPI"}    # as in notebook 30; IPI is used separately as interaction evidence
GO_PROPAGATION_RELATIONS = {"part_of"}   # + is_a
GO_ASPECT_SHORT = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}
SKIP_EXISTING = True              # reuse finished US-align outputs (stdout checked to belong to the same files)
MAX_WORKERS = 16                  # one US-align per thread (= --cpus-per-task)

for required_path in (YEASTMAP_TSV, YM_PDB_MATCHES_CSV, CORE_PDB_MATCHES_CSV, CORE_PDB_CHAIN_HITS_CSV, GAF_PATH,
                      GO_OBO_PATH, USALIGN_BIN, DATA / "Pipeline" / TRIMMED_RUN / "CombFold", DATA / "Pipeline" / CP_RUN / "CombFold"):
    assert required_path.exists(), f"missing: {required_path}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MMCIF_DIR.mkdir(parents=True, exist_ok=True)


# ── CombFold outputs (same conventions as the 09 viewer) ──────────────────────

def results_csv(run: str) -> Path:
    return DATA / "Pipeline" / run / f"all_pdb_present_{run}_pool_pipeline_complexes_combfold_results.csv"


def output_folder_name(protein_counts: dict[str, int]) -> str:
    """'P1x1_P2x2_pool_output' (hashed if > 200 chars), as in the pipeline"""
    complex_name = "_".join(f"{p}x{protein_counts[p]}" for p in sorted(protein_counts))
    if len(complex_name) > 200:
        complex_name = complex_name[:180] + "_" + hashlib.sha1(complex_name.encode()).hexdigest()[:12]
    return f"{complex_name}_pool_output"


def parse_pred(row: dict, pred: int) -> dict[str, int]:
    """pred_k = '{"P1":1,"P2":2},{"rank":1,...}' -> {'P1': 1, 'P2': 2}"""
    protein_counts, pred_info = json.loads("[" + row[f"pred_{pred}"] + "]")
    assert pred_info["rank"] == pred, f"{row['complex_ac']} pred_{pred}: rank is {pred_info['rank']}"
    return protein_counts


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
    """CombFold's chain.list: one 'UNIPROT_CHAIN.pdb' per line -> {chain: uniprot}"""
    chain_list_path = output_dir / "_unified_representation" / "assembly_output" / "chain.list"
    assert chain_list_path.exists(), f"chain.list missing: {chain_list_path}"
    chain_to_protein = {}
    for line in filter(None, map(str.strip, chain_list_path.read_text().splitlines())):
        chain_match = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pdb$", Path(line).name)
        assert chain_match, f"unparseable chain.list line in {chain_list_path}: {line!r}"
        chain_to_protein[chain_match.group(2)] = chain_match.group(1)
    return chain_to_protein


def best_model(run: str, folder: str) -> dict | None:
    """Highest-confidence output of one CombFold folder: {path, conf, chain_to_protein}; None if no outputs."""
    output_dir = DATA / "Pipeline" / run / "CombFold" / folder
    confidence_path = output_dir / "assembled_results" / "confidence.txt"
    if not confidence_path.exists():
        return None
    scored_lines = [line.split() for line in confidence_path.read_text().splitlines() if line.strip()]
    if not scored_lines:
        return None
    assert all(len(fields) == 2 for fields in scored_lines), f"malformed {confidence_path}"
    best_name, best_conf = max(((Path(p).name, float(s)) for p, s in scored_lines), key=lambda model: model[1])
    model_path = output_dir / "assembled_results" / best_name
    assert model_path.exists(), f"best model missing: {model_path}"
    return {"path": model_path, "conf": best_conf, "folder": folder, "chain_to_protein": read_chain_list(output_dir)}


def best_conf_in_cell(cell: str | None) -> float | None:
    """'{0: 78.5017, 1: 78.8618}' -> 78.8618; '{}' / null -> None"""
    values = [float(v) for v in re.findall(r": (-?[\d.]+)", cell or "")]
    return max(values) if values else None


def best_trimmed_stoichiometry(ym_row: dict) -> tuple[dict[str, int], dict] | None:
    """(counts, model) of the highest-confidence trimmed prediction; None if <= CONF_THRESHOLD"""
    pred_confs = {pred: best_conf_in_cell(ym_row[f"CF_{pred}_confidence"]) for pred in (1, 2, 3)
                  if int(ym_row[f"CF_{pred}_n_assemblies"]) > 0}
    pred_confs = {pred: conf for pred, conf in pred_confs.items() if conf is not None}
    if not pred_confs or max(pred_confs.values()) <= CONF_THRESHOLD:
        return None
    best_pred = max(pred_confs, key=pred_confs.get)
    protein_counts = parse_pred(ym_row, best_pred)
    model = best_model(TRIMMED_RUN, output_folder_name(protein_counts))
    assert model is not None, f"{ym_row['complex_ac']} pred {best_pred}: no outputs"
    assert abs(model["conf"] - pred_confs[best_pred]) < 1e-3, f"{ym_row['complex_ac']}: confidence.txt != csv"
    return protein_counts, model


def best_core_model(cp_row: dict) -> dict | None:
    """Highest-confidence CombFold model of the CP core over pred_1/2/3 + true ('(0)' counts were run as 1 copy)"""
    folders = {output_folder_name(parse_pred(cp_row, pred)) for pred in (1, 2, 3) if int(cp_row[f"CF_{pred}_n_assemblies"]) > 0}
    if int(cp_row["CF_true_n_assemblies"]) > 0:
        folders.add(output_folder_name({p: n if n > 0 else 1 for p, n in identifier_counts(cp_row["identifiers"]).items()}))
    models = [best_model(CP_RUN, folder) for folder in sorted(folders)]
    assert all(models), f"{cp_row['complex_ac']}: a stoichiometry with assemblies in the csv has no outputs"
    return max(models, key=lambda model: model["conf"]) if models else None


# ── core PDB: chain mapping + reduced structure ───────────────────────────────

def download_mmcif(pdb_id: str) -> Path:
    mmcif_path = MMCIF_DIR / f"{pdb_id}.cif.gz"
    if not mmcif_path.exists():
        request = urllib.request.Request(f"https://files.rcsb.org/download/{pdb_id}.cif.gz", headers={"User-Agent": "Mozilla/5.0"})
        partial_path = mmcif_path.with_suffix(".part")
        with urllib.request.urlopen(request, timeout=120) as response, open(partial_path, "wb") as partial_file:
            shutil.copyfileobj(response, partial_file)
        partial_path.rename(mmcif_path)   # no half-written file in the cache if the download dies
    return mmcif_path


def chain_entities(structure: gemmi.Structure) -> dict[str, str]:
    """auth chain -> polymer entity id (first model)"""
    entity_of_subchain = {subchain: entity.name for entity in structure.entities
                          if entity.entity_type == gemmi.EntityType.Polymer for subchain in entity.subchains}
    entity_of_chain = {}
    for chain in structure[0]:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        assert polymer.subchain_id() in entity_of_subchain, f"chain {chain.name}: subchain {polymer.subchain_id()} has no polymer entity"
        entity_of_chain[chain.name] = entity_of_subchain[polymer.subchain_id()]
    return entity_of_chain


def uniprot_of_entities(mmcif_path: Path) -> dict[str, set[str]]:
    """entity id -> UniProt accessions in _struct_ref (db_name UNP; chimeric entities have several)"""
    struct_ref = gemmi.cif.read(str(mmcif_path)).sole_block().find("_struct_ref.", ["entity_id", "db_name", "pdbx_db_accession"])
    accessions_of_entity = defaultdict(set)
    for ref_row in struct_ref:
        if ref_row.str(1) == "UNP":
            accessions_of_entity[ref_row.str(0)].add(ref_row.str(2))
    return accessions_of_entity


def map_core_pdb_chains(cp_ac: str, core_pdb: dict, core_proteins: set[str]) -> tuple[Path, dict[str, str] | None, str | None]:
    """(mmCIF path, {auth chain: CP core protein}, None) - same logic as the viewer's PDB (core) tab;
    (mmCIF path, None, skip reason) if the CP core proteins cannot all be mapped to chains"""
    mmcif_path = download_mmcif(core_pdb["pdb_id"])
    structure = gemmi.read_structure(str(mmcif_path))
    structure.setup_entities()
    entity_of_chain = chain_entities(structure)
    chain_to_protein = {}
    if core_pdb["mapping_source"] == "exact_matches":
        missing_chains = set(core_pdb["chain_to_protein"]) - set(entity_of_chain)
        if missing_chains:
            return mmcif_path, None, f"mapped chains {sorted(missing_chains)} not polymer chains in the mmCIF"
        proteins_of_entity = defaultdict(set)
        for chain, protein in core_pdb["chain_to_protein"].items():
            proteins_of_entity[entity_of_chain[chain]].add(protein)
        for chain, entity in entity_of_chain.items():
            if chain in core_pdb["chain_to_protein"]:
                chain_to_protein[chain] = core_pdb["chain_to_protein"][chain]
            elif len(proteins_of_entity[entity]) == 1:   # further copy of a mapped protein
                chain_to_protein[chain] = next(iter(proteins_of_entity[entity]))
    else:   # mapping by UniProt id
        accessions_of_entity = uniprot_of_entities(mmcif_path)
        for chain, entity in entity_of_chain.items():
            core_accessions = accessions_of_entity[entity] & core_proteins
            if len(core_accessions) > 1:
                return mmcif_path, None, f"entity {entity} carries several CP core proteins {sorted(core_accessions)}"
            if core_accessions:
                chain_to_protein[chain] = next(iter(core_accessions))
    missing_proteins = core_proteins - set(chain_to_protein.values())
    if missing_proteins:
        return mmcif_path, None, f"CP core proteins without a chain ({core_pdb['mapping_source']}): {sorted(missing_proteins)}"
    return mmcif_path, chain_to_protein, None


def write_reduced_pdb(mmcif_path: Path, keep_chains: set[str], out_path: Path) -> None:
    """first model, only the given polymer chains, no ligands / waters / hydrogens / altlocs -> mmCIF"""
    structure = gemmi.read_structure(str(mmcif_path))
    structure.setup_entities()
    while len(structure) > 1:
        del structure[1]
    structure.remove_ligands_and_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()
    for chain_name in [chain.name for chain in structure[0]]:
        if chain_name not in keep_chains:
            structure[0].remove_chain(chain_name)
    kept_chains = {chain.name for chain in structure[0]}
    assert kept_chains == keep_chains, f"{mmcif_path.name}: kept {sorted(kept_chains)} != wanted {sorted(keep_chains)}"
    structure.make_mmcif_document().write_file(str(out_path))


def write_core_chains_of_model(model_path: Path, chain_to_protein: dict[str, str], core_proteins: set[str], out_path: Path) -> None:
    """CombFold model reduced to the chains of CP core proteins"""
    core_chains = {chain for chain, protein in chain_to_protein.items() if protein in core_proteins}
    kept_lines = [line for line in model_path.read_text().splitlines()
                  if line.startswith(("ATOM", "HETATM")) and line[21] in core_chains]
    assert kept_lines, f"{model_path}: no chain of a CP core protein"
    out_path.write_text("\n".join(kept_lines) + "\nEND\n")


# ── US-align ──────────────────────────────────────────────────────────────────

USALIGN_COLUMNS = {   # -outfmt 2 column -> output suffix; structure 1 = CombFold, structure 2 = reduced PDB
    "TM1": "tm", "TM2": "tm_norm_by_pdb", "RMSD": "rmsd", "L1": "n_residues_cf", "L2": "n_residues_pdb", "Lali": "n_residues_aligned",
}


def run_usalign(cf_path: Path, pdb_path: Path, usalign_dir: Path) -> dict:
    stdout_file = usalign_dir / "usalign_stdout.txt"
    if not (SKIP_EXISTING and stdout_file.exists()):
        command = [str(USALIGN_BIN), str(cf_path), str(pdb_path), "-mm", "1", "-ter", "0", "-outfmt", "2"]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        (usalign_dir / "command.txt").write_text(" ".join(command) + "\n")
        stdout_file.write_text(process.stdout)
    header, values = [line for line in stdout_file.read_text().splitlines() if line.strip() and not line.startswith("Warning")]
    metrics = dict(zip(header.lstrip("#").split("\t"), values.split("\t")))
    assert metrics["PDBchain1"].startswith(str(cf_path)) and metrics["PDBchain2"].startswith(str(pdb_path)), \
        f"{stdout_file} belongs to other structures: {metrics['PDBchain1']} vs {metrics['PDBchain2']}"
    return {suffix: int(metrics[col]) if col.startswith("L") else float(metrics[col]) for col, suffix in USALIGN_COLUMNS.items()}


# ── tables ────────────────────────────────────────────────────────────────────

trimmed_results = pl.read_csv(results_csv(TRIMMED_RUN), infer_schema_length=0)
assert trimmed_results["complex_ac"].is_unique().all(), f"{TRIMMED_RUN}: more than one row per complex"
cp_results = pl.read_csv(results_csv(CP_RUN), infer_schema_length=0)
assert cp_results["complex_ac"].is_unique().all(), f"{CP_RUN}: more than one row per complex"
cp_row_by_ac = {cp_row["complex_ac"]: cp_row for cp_row in cp_results.iter_rows(named=True)}

yeastmap = pl.read_csv(YEASTMAP_TSV, separator="\t", infer_schema_length=0)
assert yeastmap["#Complex ac"].is_unique().all(), "YeastMap complex listed twice in tsv"
cp_ac_by_ym_ac = dict(zip(yeastmap["#Complex ac"], yeastmap["#Complex_ac_db"]))

ym_pdb_matches = pl.read_csv(YM_PDB_MATCHES_CSV, infer_schema_length=0)
assert ym_pdb_matches["complex_ac"].is_unique().all(), f"{YM_PDB_MATCHES_CSV.name}: more than one row per complex"
ym_pdb_match_by_ac = {ac: (match_class, pdb_id) for ac, match_class, pdb_id
                      in ym_pdb_matches.select("complex_ac", "match_class", "pdb_id").iter_rows()}

core_pdb_matches = pl.read_csv(CORE_PDB_MATCHES_CSV, infer_schema_length=0)
assert core_pdb_matches["complex_ac"].is_unique().all(), f"{CORE_PDB_MATCHES_CSV.name}: more than one row per complex"
core_pdb_match_by_cp_ac = {ac: (match_class, None if pdb_id == "none" else pdb_id.upper()) for ac, match_class, pdb_id
                           in core_pdb_matches.select("complex_ac", "match_class", "pdb_id").iter_rows()}
core_pdb_chain_hits = pl.read_csv(CORE_PDB_CHAIN_HITS_CSV, infer_schema_length=0).with_columns(pl.col("pdb_id").str.to_uppercase())
assert core_pdb_chain_hits.select("complex_ac", "pdb_id").is_unique().all(), f"{CORE_PDB_CHAIN_HITS_CSV.name}: duplicate (complex, PDB)"
matching_hits_by_cp_pdb = {(ac, pdb_id): hits for ac, pdb_id, hits
                           in core_pdb_chain_hits.select("complex_ac", "pdb_id", "matching_hits").iter_rows()}


def core_pdb_reference(cp_ac: str) -> dict | None:
    """as in the viewer: exact_matches.csv mapping, else (exact_pdb_match only) mapping by UniProt id"""
    match_class, pdb_id = core_pdb_match_by_cp_ac[cp_ac]
    if match_class not in CORE_PDB_SHOWN_CLASSES:
        return None
    matching_hits = matching_hits_by_cp_pdb.get((cp_ac, pdb_id))
    if matching_hits is None:
        assert match_class == "exact_pdb_match", f"{cp_ac} {pdb_id} ({match_class}): no chain mapping in {CORE_PDB_CHAIN_HITS_CSV.name}"
        return {"pdb_id": pdb_id, "match_class": match_class, "mapping_source": "uniprot", "chain_to_protein": None}
    chain_to_protein = {}
    for hit in matching_hits.split(","):
        hit_match = re.match(r"^([a-z0-9]+):([a-z0-9]{4})_([A-Za-z0-9]+)$", hit.strip())
        assert hit_match and hit_match.group(2).upper() == pdb_id, f"{cp_ac} {pdb_id}: unexpected matching hit {hit!r}"
        chain_to_protein[hit_match.group(3)] = hit_match.group(1).upper()
    return {"pdb_id": pdb_id, "match_class": match_class, "mapping_source": "exact_matches", "chain_to_protein": chain_to_protein}


# ── GO: propagated annotations, term frequency, IPI pairs ────────────────────

go_dag = GODag(str(GO_OBO_PATH), optional_attrs={"relationship"}, prt=None)


@cache
def go_ancestors(go_id: str) -> frozenset[str]:
    """the term itself + all ancestors over is_a and GO_PROPAGATION_RELATIONS (canonical ids)"""
    term = go_dag[go_id]
    parents = set(term.parents) | {parent for relation in GO_PROPAGATION_RELATIONS for parent in term.relationship.get(relation, set())}
    return frozenset({term.id}).union(*(go_ancestors(parent.id) for parent in parents))


gaf_table = (
    pl.read_csv(gzip.open(GAF_PATH).read(), separator="\t", comment_prefix="!", has_header=False, quote_char=None, infer_schema_length=0)
    .select(pl.nth(0).alias("db"), pl.nth(1).alias("protein"), pl.nth(3).alias("qualifier"), pl.nth(4).alias("go_id"),
            pl.nth(5).alias("reference"), pl.nth(6).alias("evidence"), pl.nth(7).alias("with_from"))
    .filter((pl.col("db") == "UniProtKB") & ~pl.col("qualifier").fill_null("").str.contains("NOT"))
)
unknown_go_ids = set(gaf_table["go_id"]) - set(go_dag)
assert not unknown_go_ids, f"{len(unknown_go_ids)} GAF term ids not in {GO_OBO_PATH.name}: {sorted(unknown_go_ids)[:10]}"

# propagated terms per protein and aspect
go_terms_of_protein = defaultdict(set)   # (protein, aspect) -> set of term ids
for protein, go_id in gaf_table.filter(~pl.col("evidence").is_in(GO_EXCLUDED_EVIDENCE)).select("protein", "go_id").unique().iter_rows():
    aspect = GO_ASPECT_SHORT[go_dag[go_id].namespace]
    go_terms_of_protein[(protein, aspect)] |= go_ancestors(go_id)
n_annotated_proteins = defaultdict(int)
term_protein_counts = defaultdict(int)
for (protein, aspect), go_ids in go_terms_of_protein.items():
    n_annotated_proteins[aspect] += 1
    for go_id in go_ids:
        term_protein_counts[go_id] += 1
term_freq = {go_id: count / n_annotated_proteins[GO_ASPECT_SHORT[go_dag[go_id].namespace]] for go_id, count in term_protein_counts.items()}
print(f"GO: propagated annotations for {dict(n_annotated_proteins)} proteins per aspect")

# IPI: protein <-> with/from partner (UniProtKB entries; isoform / PRO-chain suffix dropped: P12345-2 -> P12345), with PMIDs
ipi_rows = gaf_table.filter(pl.col("evidence") == "IPI").select("protein", "with_from", "reference")
with_from_entries = ipi_rows["with_from"].drop_nulls().str.split("|").explode().str.split(",").explode()
partner_db_counts = with_from_entries.str.extract(r"^([^:]+):").alias("db").value_counts().sort("count", descending=True)
print(f"IPI with/from entries per database (only UniProtKB is used): {dict(partner_db_counts.iter_rows())}")
pmids_of_pair = defaultdict(set)   # frozenset({protein, partner}) -> PMIDs
for protein, with_from, reference in ipi_rows.iter_rows():
    pmids = {ref for ref in (reference or "").split("|") if ref.startswith("PMID:")}
    for partner_entry in re.split(r"[|,]", with_from or ""):
        partner_match = re.match(r"^UniProtKB:([A-Z0-9]+)(?:-.+)?$", partner_entry.strip())
        if partner_match and partner_match.group(1) != protein:
            pmids_of_pair[frozenset({protein, partner_match.group(1)})] |= pmids
print(f"IPI: {len(pmids_of_pair)} protein pairs")


def go_support(added_protein: str, core_proteins: set[str], aspect: str) -> dict:
    """most specific term shared between the added protein and any CP core protein (propagated)"""
    core_terms = set().union(*(go_terms_of_protein.get((protein, aspect), set()) for protein in core_proteins))
    shared_terms = go_terms_of_protein.get((added_protein, aspect), set()) & core_terms
    if not shared_terms:
        return {f"go_{aspect}_term": None, f"go_{aspect}_name": None, f"go_{aspect}_freq": None}
    best_term = min(shared_terms, key=lambda go_id: (term_freq[go_id], go_id))
    return {f"go_{aspect}_term": best_term, f"go_{aspect}_name": go_dag[best_term].name, f"go_{aspect}_freq": term_freq[best_term]}


# ── per YeastMap complex ──────────────────────────────────────────────────────

candidates = []   # one dict per YeastMap complex
for ym_row in trimmed_results.iter_rows(named=True):
    best_trimmed = best_trimmed_stoichiometry(ym_row)
    if best_trimmed is None:
        continue
    protein_counts, trimmed_model = best_trimmed
    cp_ac = cp_ac_by_ym_ac[ym_row["complex_ac"]]
    assert cp_ac in cp_row_by_ac, f"{ym_row['complex_ac']}: CP core {cp_ac} not in {CP_RUN}"
    assert cp_ac in core_pdb_match_by_cp_ac, f"{cp_ac} not in {CORE_PDB_MATCHES_CSV.name}"
    core_proteins = set(identifier_counts(cp_row_by_ac[cp_ac]["identifiers"]))
    ym_proteins = set(identifier_counts(ym_row["identifiers"]))
    candidates.append({"ym_ac": ym_row["complex_ac"], "cp_ac": cp_ac, "protein_counts": protein_counts,
                       "trimmed_model": trimmed_model, "core_model": best_core_model(cp_row_by_ac[cp_ac]),
                       "core_proteins": core_proteins, "added_proteins": sorted(ym_proteins - core_proteins),
                       "core_pdb": core_pdb_reference(cp_ac)})
print(f"{len(candidates)}/{trimmed_results.height} YeastMap complexes with trimmed confidence > {CONF_THRESHOLD}")

# core PDBs: download + chain mapping, all checked before any US-align runs
core_pdb_chains_by_cp_ac = {}
for candidate in tqdm([c for c in candidates if c["core_pdb"]], desc="core PDB mapping"):
    if candidate["cp_ac"] not in core_pdb_chains_by_cp_ac:
        core_pdb_chains_by_cp_ac[candidate["cp_ac"]] = map_core_pdb_chains(candidate["cp_ac"], candidate["core_pdb"], candidate["core_proteins"])

skipped_core_pdbs = {cp_ac: (core_pdb_chains[0].name, core_pdb_chains[2]) for cp_ac, core_pdb_chains in core_pdb_chains_by_cp_ac.items()
                     if core_pdb_chains[1] is None}
print(f"{len(skipped_core_pdbs)}/{len(core_pdb_chains_by_cp_ac)} core PDBs skipped (not fully mappable, no TM-scores):")
for cp_ac, (mmcif_name, skip_reason) in sorted(skipped_core_pdbs.items()):
    print(f"  {cp_ac} {mmcif_name}: {skip_reason}")

# US-align jobs: (key, CombFold structure, reduced PDB, output dir); identical pairs are aligned once
usalign_jobs = {}
for candidate in candidates:
    if not candidate["core_pdb"] or candidate["cp_ac"] in skipped_core_pdbs:
        continue
    mmcif_path, pdb_chain_to_protein, _ = core_pdb_chains_by_cp_ac[candidate["cp_ac"]]
    reduced_pdb_path = OUT_DIR / "reduced_pdb" / f"{candidate['core_pdb']['pdb_id']}_{candidate['cp_ac']}_core_chains.cif"
    if not reduced_pdb_path.exists():
        reduced_pdb_path.parent.mkdir(exist_ok=True)
        write_reduced_pdb(mmcif_path, set(pdb_chain_to_protein), reduced_pdb_path)
    if candidate["core_model"]:
        core_dir = OUT_DIR / "usalign" / f"core_model_{candidate['cp_ac']}"
        usalign_jobs[("core_model", candidate["cp_ac"])] = (candidate["core_model"]["path"], reduced_pdb_path, core_dir)
    trimmed_dir = OUT_DIR / "usalign" / f"trimmed_core_{candidate['ym_ac']}"
    trimmed_dir.mkdir(parents=True, exist_ok=True)
    trimmed_core_path = trimmed_dir / "trimmed_model_core_chains.pdb"
    write_core_chains_of_model(candidate["trimmed_model"]["path"], candidate["trimmed_model"]["chain_to_protein"],
                               candidate["core_proteins"], trimmed_core_path)
    usalign_jobs[("trimmed_core", candidate["ym_ac"])] = (trimmed_core_path, reduced_pdb_path, trimmed_dir)
for _, _, usalign_dir in usalign_jobs.values():
    usalign_dir.mkdir(parents=True, exist_ok=True)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_key = {executor.submit(run_usalign, *job): key for key, job in usalign_jobs.items()}
    usalign_metrics = {future_to_key[future]: future.result()
                       for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="US-align")}

# ── output tables ─────────────────────────────────────────────────────────────

added_protein_rows = []
candidate_rows = []
for candidate in candidates:
    added_rows_of_complex = []
    for added_protein in candidate["added_proteins"]:
        ipi_partners = sorted(core for core in candidate["core_proteins"] if frozenset({added_protein, core}) in pmids_of_pair)
        ipi_pmids = sorted(set().union(*(pmids_of_pair[frozenset({added_protein, core})] for core in ipi_partners)))
        evidence_row = {"complex_ac": candidate["ym_ac"], "cp_complex_ac": candidate["cp_ac"], "added_protein": added_protein,
                        "ipi_core_partners": ";".join(ipi_partners), "ipi_pmids": ";".join(ipi_pmids)}
        for aspect in ("BP", "CC", "MF"):
            evidence_row |= go_support(added_protein, candidate["core_proteins"], aspect)
        added_rows_of_complex.append(evidence_row)
    added_protein_rows += added_rows_of_complex

    def n_supported(aspect: str) -> int:
        return sum(row[f"go_{aspect}_freq"] is not None and row[f"go_{aspect}_freq"] <= SPECIFIC_TERM_MAX_FREQ
                   for row in added_rows_of_complex)

    ym_class, ym_pdb_id = ym_pdb_match_by_ac[candidate["ym_ac"]]
    core_class, core_pdb_id = core_pdb_match_by_cp_ac[candidate["cp_ac"]]
    candidate_row = {
        "complex_ac": candidate["ym_ac"], "cp_complex_ac": candidate["cp_ac"],
        "stoichiometry": output_folder_name(candidate["protein_counts"]).removesuffix("_pool_output"),
        "trimmed_cf_confidence": candidate["trimmed_model"]["conf"],
        "core_cf_confidence": candidate["core_model"]["conf"] if candidate["core_model"] else None,
        "ym_pdb_class": ym_class, "ym_pdb_id": None if ym_pdb_id == "none" else ym_pdb_id,
        "core_pdb_class": core_class, "core_pdb_id": core_pdb_id,
        "core_pdb_mapping_source": candidate["core_pdb"]["mapping_source"] if candidate["core_pdb"] else None,
        "core_pdb_skip_reason": skipped_core_pdbs[candidate["cp_ac"]][1] if candidate["cp_ac"] in skipped_core_pdbs else None,
        "n_core_proteins": len(candidate["core_proteins"]), "n_added_proteins": len(candidate["added_proteins"]),
        "n_added_with_ipi": sum(bool(row["ipi_core_partners"]) for row in added_rows_of_complex),
        "n_added_go_bp_support": n_supported("BP"), "n_added_go_cc_support": n_supported("CC"),
        "n_added_go_mf_support": n_supported("MF"),
    }
    for prefix, key in (("core_model_vs_pdb", ("core_model", candidate["cp_ac"])), ("trimmed_core_vs_pdb", ("trimmed_core", candidate["ym_ac"]))):
        metrics = usalign_metrics.get(key, {})
        candidate_row |= {f"{prefix}_{suffix}": metrics.get(suffix) for suffix in USALIGN_COLUMNS.values()}
    candidate_rows.append(candidate_row)

candidate_evidence = pl.DataFrame(candidate_rows, infer_schema_length=None).sort("trimmed_core_vs_pdb_tm", descending=True, nulls_last=True)
added_protein_evidence = pl.DataFrame(added_protein_rows, infer_schema_length=None)
candidate_evidence.write_csv(OUT_DIR / "candidate_evidence.csv")
added_protein_evidence.write_csv(OUT_DIR / "added_protein_evidence.csv")
with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=250):
    print(candidate_evidence.select("complex_ac", "cp_complex_ac", "trimmed_cf_confidence", "core_pdb_class", "core_pdb_id",
                                    "core_model_vs_pdb_tm", "trimmed_core_vs_pdb_tm", "n_added_proteins", "n_added_with_ipi",
                                    "n_added_go_bp_support", "ym_pdb_class"))
print(f"written: {OUT_DIR / 'candidate_evidence.csv'} ({candidate_evidence.height} complexes), "
      f"{OUT_DIR / 'added_protein_evidence.csv'} ({added_protein_evidence.height} added proteins)")