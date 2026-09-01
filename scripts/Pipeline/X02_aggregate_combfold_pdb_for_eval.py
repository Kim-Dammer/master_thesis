import argparse
import ast
import json
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import gemmi
import polars as pl
from rcsbapi.data import DataQuery as Query
from tqdm import tqdm

from procompa.combfold_eval.config import Config
from procompa.combfold_eval.reference import _assembly_ids, _download

logger = logging.getLogger(__name__)

reference_pdb_dir = Path("/cluster/project/beltrao/kdammer/master_thesis/data/reference_pdb")
cfg = Config()


def download_pdb(pdb_id, reference_pdb):
    """
    Worker function to process a single PDB entry.

    Checks whether it already exists in the reference directory,
    if not, creates the directory and downloads the AU and all assemblies.
    """

    pdb_id_dir = reference_pdb / pdb_id
    assert not pdb_id_dir.exists(), f"{pdb_id} already exists in {reference_pdb}"
    pdb_id_dir.mkdir(parents=True, exist_ok=True)

    try:
        au_path = pdb_id_dir / f"{pdb_id}.cif"
        _download(f"{cfg.rcsb_files}/{pdb_id}.cif", au_path, cfg.download_timeout_s)

        for a_id in _assembly_ids(pdb_id, cfg):
            assembly_id_path = pdb_id_dir / f"{pdb_id}-assembly{a_id}.cif"
            _download(f"{cfg.rcsb_files}/{pdb_id}-assembly{a_id}.cif", assembly_id_path, cfg.download_timeout_s)
    except Exception:
        shutil.rmtree(pdb_id_dir, ignore_errors=True)  # don't leave a partial dir
        raise


def _parse_entities_and_refs(cif_path):
    """Low-level mmCIF parsing: entity table + struct_ref table for one file."""
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()

    entities = {}
    for row in block.find('_entity.', ['id', 'type', 'src_method',
                                        'pdbx_description',
                                        'pdbx_number_of_molecules']):
        entities[row[0]] = {
            'type': row[1],
            'src_method': row[2],
            'desc': row[3],
            'n_mol': int(row[4]) if row[4] not in ('?', '.') else None,
        }

    refs = {}
    for row in block.find('_struct_ref.', ['entity_id', 'db_name',
                                            'pdbx_db_accession']):
        refs[row[0]] = {'db_name': row[1], 'accession': row[2]}

    return entities, refs


@lru_cache(maxsize=None)
def _rcsb_uniprot_map(pdb_id):
    """entity_id -> UniProt accession, via RCSB (SIFTS-integrated).
    Returns only entities RCSB could map to UniProt; others absent."""
    pdb_id = pdb_id.upper()
    query = Query(
        input_type="entries",
        input_ids=[pdb_id],
        return_data_list=[
            "polymer_entities.rcsb_id",  # e.g. "1X3Z_1"
            "polymer_entities.rcsb_polymer_entity_container_identifiers"
            ".reference_sequence_identifiers.database_name",
            "polymer_entities.rcsb_polymer_entity_container_identifiers"
            ".reference_sequence_identifiers.database_accession",
        ],
    )
    result = query.exec()

    mapping = {}
    entities = result["data"]["entries"][0]["polymer_entities"]
    for ent in entities:
        entity_id = ent["rcsb_id"].split("_")[-1]  # "1X3Z_1" -> "1"
        ids = (ent["rcsb_polymer_entity_container_identifiers"]
               .get("reference_sequence_identifiers") or [])
        unp = [r["database_accession"] for r in ids
               if r["database_name"] == "UniProt"]
        if unp:
            mapping[entity_id] = unp[0]
    return mapping


def _build_stoichiometry(cif_path, entities, refs, fallback_refs=None,
                         unp_overrides=None):
    fallback_refs = fallback_refs or {}
    unp_overrides = unp_overrides or {}
    stoich = {}

    for eid, e in entities.items():
        if e['type'] != 'polymer' or e['src_method'] not in ('man', 'nat'):
            continue

        ref = refs.get(eid)
        if not ref and eid in fallback_refs:
            ref = fallback_refs[eid]

        if not ref:
            logger.warning(f"{cif_path}: entity {eid} ({e['desc']!r}) has no "
                            f"_struct_ref row (local or AU fallback) — dropping")
            continue
        if ref['db_name'] == 'PDB':
            logger.debug(f"{cif_path}: entity {eid} self-referential, excluding")
            continue

        accession = ref['accession']
        if ref['db_name'] != 'UNP':
            if eid in unp_overrides:
                logger.info(f"{cif_path}: entity {eid} {ref['db_name']} accession "
                            f"{accession!r} resolved to UniProt "
                            f"{unp_overrides[eid]!r} via RCSB")
                accession = unp_overrides[eid]
            else:
                logger.warning(f"{cif_path}: entity {eid} accession {accession!r} "
                               f"is {ref['db_name']}, not UniProt, and RCSB had no "
                               f"UniProt mapping — keeping raw accession")

        if e['n_mol'] is None:
            logger.warning(f"{cif_path}: entity {eid} ({accession}) has no "
                            f"pdbx_number_of_molecules — skipping, incomplete")
            continue

        stoich[accession] = e['n_mol']

    return stoich


def get_stoichiometry(cif_path, fallback_refs=None):
    entities, refs = _parse_entities_and_refs(cif_path)
    return _build_stoichiometry(cif_path, entities, refs, fallback_refs)


def get_pdb_stoichiometries(pdb_id, reference_pdb):
    pdb_id = str(pdb_id).lower()
    pdb_id_dir = Path(reference_pdb) / pdb_id
    if not pdb_id_dir.exists():
        raise FileNotFoundError(f"{pdb_id}: expected directory {pdb_id_dir} does not exist")

    result = {}

    au_path = pdb_id_dir / f"{pdb_id}.cif"
    if not au_path.exists():
        raise FileNotFoundError(f"{pdb_id}: missing AU file {au_path}")
    au_entities, au_refs = _parse_entities_and_refs(au_path)

    # Only hit the API if some polymer entity isn't already UniProt/PDB
    needs_mapping = any(
        r['db_name'] not in ('UNP', 'PDB') for r in au_refs.values()
    )
    unp_overrides = _rcsb_uniprot_map(pdb_id) if needs_mapping else {}

    result["au"] = _build_stoichiometry(au_path, au_entities, au_refs,
                                        unp_overrides=unp_overrides)

    assembly_pattern = re.compile(rf"^{re.escape(pdb_id)}-assembly(\d+)\.cif$")
    for path in sorted(pdb_id_dir.glob(f"{pdb_id}-assembly*.cif")):
        m = assembly_pattern.match(path.name)
        if not m:
            raise ValueError(f"{pdb_id}: unexpected filename {path.name} matched glob but not assembly-id pattern")
        a_id = m.group(1)
        a_entities, a_refs = _parse_entities_and_refs(path)
        result[a_id] = _build_stoichiometry(path, a_entities, a_refs,
                                            fallback_refs=au_refs,
                                            unp_overrides=unp_overrides)

    return result


# %% CP_stochiometry + reference PDB matching

# --- sentinels ---
UNKNOWN = "unknown"
AMBIGUOUS = "ambiguous"
UNKNOWN_INPUTS = {"unknown", "unkown"}   # spellings that may appear in correct_pred_rank
NONE_INPUTS = {None, "none"}

# Canonical UniProt accession only (isoforms P32628-1 and -PRO_ forms intentionally excluded)
UNP_RE = re.compile(
    r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
    r"|^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
)
TOKEN_RE = re.compile(r"^(.+?)\((\d+)\)$")   # "P32628(1)" -> ("P32628", "1")


def parse_identifiers_to_stoich(identifiers, pdb_id=None):
    """Pipe-separated identifiers -> stoich dict, with:
      - any non-UNP token (CHEBI, EBI-, isoforms, -PRO_, junk) -> AMBIGUOUS
      - any (0) count -> UNKNOWN (ComplexPortal 'unknown stoichiometry')
      - a genuine all-UNP, all->=1 dict -> warn (least-trusted fallback path)
    """
    if identifiers is None:
        return AMBIGUOUS

    stoich = {}
    for tok in identifiers.split("|"):
        tok = tok.strip()
        m = TOKEN_RE.match(tok)
        if not m:
            return AMBIGUOUS
        acc, count = m.group(1), int(m.group(2))
        if not UNP_RE.match(acc):
            return AMBIGUOUS
        stoich[acc] = count

    if not stoich:
        return AMBIGUOUS

    if any(c == 0 for c in stoich.values()):
        if any(c > 0 for c in stoich.values()):
            logger.warning(f"{pdb_id}: identifiers mix zero and non-zero counts "
                           f"{identifiers!r} — treating as {UNKNOWN}")
        return UNKNOWN

    logger.warning(f"{pdb_id}: derived CP_stochiometry from identifiers "
                   f"{identifiers!r} -> {stoich} — verify")
    return stoich


def _first_pred_dict(val):
    """Extract the stoich dict (first of the two dicts) from a pred_{rank} cell,
    e.g. '{..stoich..},{"rank":..}' -> {..stoich..}."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return dict(val[0])
    if isinstance(val, dict):
        return dict(val)
    s = str(val).strip()
    try:
        result = json.loads("[" + s + "]")[0]
    except Exception:
        result = ast.literal_eval("(" + s + ")")[0]
    if "rank" in result:  # guard: stoich dict should be first, not the meta dict
        raise ValueError(f"pred cell parsed to meta dict first, expected stoich: {s!r}")
    return result


def _find_match(cp, pdb_stoich):
    """Best-matching key by exact dict equality: assembly over AU,
    lowest numeric assembly index. None if no match."""
    matches = [k for k, s in pdb_stoich.items() if s == cp]
    assemblies = sorted((k for k in matches if k != "au"), key=int)
    ordered = assemblies + (["au"] if "au" in matches else [])
    return ordered[0] if ordered else None


def _default_model(pdb_stoich):
    """Default reference: assembly 1 if present, else AU."""
    return "1" if "1" in pdb_stoich else "au"


def _find_match_by_pdb_stoich(row, pdb_stoich):
    """If Complex portal doent have a stoichiometry that the one from pdb to compare (if this stoichiometry was already run by Stoic CF pipeline)"""
    asm_keys = sorted((k for k in pdb_stoich if k != "au"), key=int)
    if "au" in pdb_stoich:
        asm_keys.append("au")
    pred_cols = sorted(
        (c for c in row if c.startswith("pred_") and c[5:].isdigit()),
        key=lambda c: int(c[5:]),
    )
    for asm_key in asm_keys:
        asm_stoich = pdb_stoich[asm_key]
        if not isinstance(asm_stoich, dict):
            continue
        for col in pred_cols:
            cf_s = _first_pred_dict(row[col])
            if isinstance(cf_s, dict) and cf_s == asm_stoich:
                return asm_stoich, asm_key
    return None, None    


def classify_row(row):
    """-> (CP_stochiometry, reference_pdb_model, found_match_reference_pdb_model)"""
    rank = row["correct_pred_rank"]
    pdb_stoich = row["pdb_stoichiometry"]
    rank_norm = rank.lower() if isinstance(rank, str) else rank

    if pdb_stoich is None:  # upstream stoichiometry extraction failed
        cp = UNKNOWN if rank_norm in UNKNOWN_INPUTS else None
        return cp, None, False

    # --- Step 1: CP_stochiometry ---
    if isinstance(rank_norm, str) and rank_norm.isdigit():
        col = f"pred_{rank_norm}"
        if col not in row:
            raise KeyError(f"{row.get('pdb_id')}: correct_pred_rank={rank!r} but no column {col!r}")
        cp = _first_pred_dict(row[col])
    elif rank_norm in NONE_INPUTS:
        cp = parse_identifiers_to_stoich(row["identifiers"], pdb_id=row.get("pdb_id"))
    elif rank_norm in UNKNOWN_INPUTS:
        matched_stoich, matched_key = _find_match_by_pdb_stoich(row, pdb_stoich)
        if matched_key is not None:
            return matched_stoich, matched_key, True
        cp = UNKNOWN
    else:
        raise ValueError(f"{row.get('pdb_id')}: unexpected correct_pred_rank {rank!r}")

    # --- Step 2: match (dict -> match else default+False; sentinel -> default+False) ---
    if isinstance(cp, dict):
        key = _find_match(cp, pdb_stoich)
        if key is not None:
            return cp, key, True
        return cp, _default_model(pdb_stoich), False
    return cp, _default_model(pdb_stoich), False   # UNKNOWN or AMBIGUOUS


def main(combfold_results_csv, n_workers=8):
    all_combfold_runs_summary_results = pl.read_csv(
        combfold_results_csv
    )
    all_pdb_matches_with_match_class = pl.read_csv(
        "/cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/all_pdb_matches_with_match_class.csv"
    )

    # Join to add pdb_id, match_class and n_proteins columns
    df = all_combfold_runs_summary_results.join(
        all_pdb_matches_with_match_class, on="complex_ac", how="left"
    )

    # Filter on match_class == exact_pdb_match as we only evaluate on those
    df = df.filter(pl.col("match_class") == "exact_pdb_match")

    n_null = df["pdb_id"].null_count()
    assert n_null == 0, f"{n_null} rows have null pdb_id after exact_pdb_match filter — unmatched complex_ac?"

    requested_entries = list(df["pdb_id"])
    requested_entries = [x.lower() for x in requested_entries]
    available_pdbs = set(os.listdir(reference_pdb_dir)) # this also has some random stuff
    to_download = set(requested_entries) - available_pdbs

    # Download the missing PDBs in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(download_pdb, pdb_id, reference_pdb_dir)
            for pdb_id in to_download
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDBs"):
            future.result()

    # Add pdb stoichs to df 
    df = df.with_columns(
        pl.col("pdb_id")
        .map_elements(
            lambda pdb_id: get_pdb_stoichiometries(pdb_id, reference_pdb_dir),
            return_dtype=pl.Object,
        )
        .alias("pdb_stoichiometry")
    )

    # Choosing pdb model to use for downstream alignment 
    cp_col, model_col, found_col = [], [], []
    for row in df.iter_rows(named=True):
        cp, model, found = classify_row(row)
        cp_col.append(cp)
        model_col.append(model)
        found_col.append(found)

    df = df.with_columns(
        pl.Series("CP_stochiometry", cp_col, dtype=pl.Object),
        pl.Series("reference_pdb_model", model_col),
        pl.Series("found_match_reference_pdb_model", found_col),
    )

    output_dir = Path(combfold_results_csv).parent / "cf_pdb_structure_similarity"
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Object]
    df = df.with_columns(
        pl.col(c).map_elements(lambda x: json.dumps(x) if x is not None else None, return_dtype=pl.Utf8)
        for c in obj_cols
    )
    df.write_parquet(output_dir / "aggregate_cf_for_pdb_eval.parquet")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Aggregate CombFold PDB results for evaluation")
    parser.add_argument(
        "--cf_results_summary",
        type=str,
        required=True,
        help="Path to the combfold results CSV, e.g. Pipeline/t11_RM_TM_updated_CF_pipeline/all_pdb_present_t11_CF_test_pool_pipeline_complexes_combfold_results.csv",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of parallel workers for downloading PDBs",
    )
    args = parser.parse_args()
    main(combfold_results_csv=args.cf_results_summary, n_workers=args.n_workers)