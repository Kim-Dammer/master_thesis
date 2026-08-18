#!/usr/bin/env python3
"""
Evaluate CombFold predictions against reference PDB structures.

For each (complex, CombFold output) pair this script:
  1. resolves the reference PDB/CIF path and the CombFold output paths,
  2. builds a UniProt -> (CF chains, PDB chains) mapping using SIFTS,
  3. runs US-align at the complex level and per chain pair,
  4. parses the US-align stdout into metrics columns,
  5. annotates close-by chain pairs (interfaces) for both CF and reference,
  6. flags Stoic training-set leakage (exact PDB, and identical protein set).
"""

import argparse
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gemmi
import polars as pl
from Bio.PDB import MMCIFParser, PDBParser
from tqdm import tqdm

logger = logging.getLogger(__name__)

RELEVANT_COLS = [
    "complex_ac",
    "identifiers",
    "n_proteins",
    "pdb_id",
    "reference_pdb_model",
    "CP_stochiometry",
    "Combfold_result_path",
    "n_pairs_used",
    "pairs_used",
    "correct_pred_rank",
    "n_combfold_outputs",
]

METRIC_DTYPES = {
    "len_structure1": pl.Int64,
    "len_structure2": pl.Int64,
    "aligned_length": pl.Int64,
    "rmsd": pl.Float64,
    "seq_id": pl.Float64,
    "tm_score": pl.Float64,
}

CF_OUTPUT_RE = r"assembled_results/output_clustered_(\d+)\.pdb$"

DATA_ROOT = Path("/cluster/project/beltrao/kdammer/master_thesis/data")
DEFAULT_REFERENCE_PDB_DIR = DATA_ROOT / "reference_pdb"
DEFAULT_SIFTS_CSV = DATA_ROOT / "pdb" / "pdb_chain_uniprot.csv"
DEFAULT_STOIC_CSV = DATA_ROOT / "Stoic" / "data_file_stoic.csv"
DEFAULT_USALIGN_BIN = Path("/cluster/project/beltrao/kdammer/master_thesis/tools/usalign/USalign/USalign")
DEFAULT_OUTPUT_NAME = "cf_pdb_eval_all_metrics.parquet"


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def get_reference_pdb_path(pdb_id: str, reference_pdb_model: str, reference_pdb_dir: Path) -> Path:
    """Resolve the reference structure file (.pdb or .cif) for a PDB ID / assembly."""
    pdb_id = pdb_id.lower()
    if reference_pdb_model == "au":
        stem = reference_pdb_dir / pdb_id / pdb_id
    else:
        stem = reference_pdb_dir / pdb_id / f"{pdb_id}-assembly{reference_pdb_model}"

    for suffix in (".pdb", ".cif"):
        if stem.with_suffix(suffix).exists():
            return stem.with_suffix(suffix)
    raise FileNotFoundError(f"Neither .pdb nor .cif file found for {pdb_id} at {stem}.")


def get_combfold_output_paths(combfold_result_path: str, n_combfold_outputs: int) -> list[Path]:
    """
    List assembled CombFold outputs under `{combfold_result_path}/assembled_results/`,
    sorted for a stable order.

    Raises if the count does not match `n_combfold_outputs` -- we don't want to
    silently proceed with a mismatched count.
    """
    if n_combfold_outputs == 0:
        return []

    assembled_results_dir = Path(combfold_result_path) / "assembled_results"
    if not assembled_results_dir.is_dir():
        raise FileNotFoundError(f"assembled_results dir does not exist: {assembled_results_dir}")

    assert all(p.suffix in (".txt", ".pdb") for p in assembled_results_dir.iterdir()), (
        f"Unexpected file extension found in {assembled_results_dir}. Expected only .txt or .pdb files."
    )
    output_paths = sorted(assembled_results_dir.glob("*.pdb"))

    if len(output_paths) != n_combfold_outputs:
        raise ValueError(
            f"Expected {n_combfold_outputs} CombFold outputs in {assembled_results_dir}, "
            f"found {len(output_paths)}."
        )
    return output_paths


def build_long_df(df: pl.DataFrame, reference_pdb_dir: Path) -> pl.DataFrame:
    """Resolve reference + CombFold paths, then explode to one row per CombFold output."""
    df = df.select(RELEVANT_COLS).with_columns(
        pl.struct(["pdb_id", "reference_pdb_model"])
        .map_elements(
            lambda row: str(get_reference_pdb_path(row["pdb_id"], row["reference_pdb_model"], reference_pdb_dir)),
            return_dtype=pl.Utf8,
        )
        .alias("reference_pdb_path")
    )

    df = df.with_columns(
        pl.struct(["Combfold_result_path", "n_combfold_outputs"])
        .map_elements(
            lambda row: [
                str(p) for p in get_combfold_output_paths(row["Combfold_result_path"], row["n_combfold_outputs"])
            ],
            return_dtype=pl.List(pl.Utf8),
        )
        .alias("combfold_output_path")
    )

    df_long = df.explode("combfold_output_path")

    # sanity checks: row count scales by n_combfold_outputs (0 -> 1 row from explode),
    # and only rows with n_combfold_outputs == 0 may have a null combfold_output_path
    expected_n_rows = df["n_combfold_outputs"].clip(lower_bound=1).sum()
    assert df_long.height == expected_n_rows, (
        f"Expected {expected_n_rows} rows after exploding, got {df_long.height}."
    )

    expected_null_outputs = (df["n_combfold_outputs"] == 0).sum()
    actual_null_outputs = df_long["combfold_output_path"].null_count()
    assert actual_null_outputs == expected_null_outputs, (
        f"Expected {expected_null_outputs} null combfold_output_path rows (n_combfold_outputs == 0), "
        f"got {actual_null_outputs}."
    )
    assert df_long["reference_pdb_path"].null_count() == 0, "Found null reference_pdb_path."

    logger.info(f"Built long df with {df_long.height} rows from {df.height} complexes.")
    return df_long


# --------------------------------------------------------------------------- #
# chain mapping (SIFTS)
# --------------------------------------------------------------------------- #
def build_sifts_chain_lookup(sifts_csv: Path, relevant_pdb_ids: set[str]) -> dict[str, dict[str, list[str]]]:
    """pdb_id -> {uniprot: [chain ids]} from pdb_chain_uniprot.csv, restricted to relevant PDBs."""
    sifts = (
        pl.read_csv(sifts_csv, skip_rows=1, columns=["PDB", "CHAIN", "SP_PRIMARY"])
        .rename(str.lower)
        .filter(pl.col("pdb").is_in(relevant_pdb_ids))
    )

    grouped = (
        sifts.select(["pdb", "chain", "sp_primary"])
        .unique()
        .group_by(["pdb", "sp_primary"])
        .agg(pl.col("chain").unique().alias("chains"))
    )

    lookup: dict[str, dict[str, list[str]]] = {}
    for pdb_id, uniprot, chains in grouped.iter_rows():
        lookup.setdefault(pdb_id, {})[uniprot] = chains
    return lookup


def get_structure_chain_ids(struct_path: str) -> set[str]:
    """Chain IDs that actually exist in this file.

    pdb_chain_uniprot.csv is built on the AU, which can contain more chains than
    the biological assembly.
    """
    parser = MMCIFParser(QUIET=True) if struct_path.endswith(".cif") else PDBParser(QUIET=True)
    structure = parser.get_structure("x", struct_path)
    return {chain.id for chain in structure[0]}


def build_chain_mapping(
    reference_pdb_path: str,
    combfold_output_path: str | None,
    sifts_lookup: dict[str, dict[str, list[str]]],
    complex_ac: str | None = None,
) -> dict[str, tuple[list[str], list[str]]]:
    """uniprot -> (CF chain ids, reference chain ids). Empty dict on any mismatch."""
    if combfold_output_path is None:
        logger.warning(f"{complex_ac or '?'}: no combfold_output_path (n_combfold_outputs == 0) -- empty mapping.")
        return {}

    pdb_id = Path(reference_pdb_path).parent.name
    assembly_chain_ids = get_structure_chain_ids(reference_pdb_path)
    pdb_chains = {
        unp: sorted(set(chains) & assembly_chain_ids) for unp, chains in sifts_lookup[pdb_id].items()
    }

    chain_list = (
        Path(combfold_output_path).parent.parent / "_unified_representation" / "assembly_output" / "chain.list"
    )
    cf_chains: dict[str, list[str]] = {}
    for line in chain_list.read_text().split():
        m = re.match(r"^(\w+)_(\w+)\.pdb$", line)
        if m is None:
            logger.warning(
                f"{complex_ac or '?'}: unparseable chain.list line {line!r} in {chain_list} -- empty mapping."
            )
            return {}
        unp, chain_id = m.groups()
        cf_chains.setdefault(unp, []).append(chain_id)

    if set(cf_chains) != set(pdb_chains):
        logger.warning(
            f"{complex_ac or '?'}: uniprot mismatch cf vs pdb: {set(cf_chains) ^ set(pdb_chains)} -- empty mapping."
        )
        return {}

    for unp in cf_chains:
        if len(cf_chains[unp]) != len(pdb_chains[unp]):
            logger.warning(
                f"{complex_ac or '?'}: {unp} copy-number mismatch: "
                f"CF has {cf_chains[unp]}, PDB assembly has {pdb_chains[unp]} -- empty mapping."
            )
            return {}

    # NOTE: for homomers (len > 1), which CF chain corresponds to which PDB chain
    # is NOT resolved here -- both chain lists are returned, correspondence left to
    # the RMSD step (permutation search over copies).
    return {unp: (cf_chains[unp], pdb_chains[unp]) for unp in cf_chains}


# --------------------------------------------------------------------------- #
# US-align
# --------------------------------------------------------------------------- #
def _pred_index(combfold_output_path: str) -> str:
    """output_clustered_3.pdb -> '3'"""
    return Path(combfold_output_path).stem.split("_")[-1]


def run_usalign_for_row(
    row: dict,
    usalign_bin: str,
    mm: int = 1,
    ter: int = 1,
    skip_existing: bool = False,
) -> dict | None:
    """Run US-align for one CombFold output: once for the complex, once per chain pair."""
    mapping = row["chain_mapping"]
    cf_chains = [c for cf, _ in mapping.values() for c in cf]
    ref_chains = [c for _, ref in mapping.values() for c in ref]

    assert len(cf_chains) == len(ref_chains), "Mismatch in number of chains between CF and reference."
    if not cf_chains:
        logger.warning(f"{row['combfold_output_path']}: no chains to align -- skipping.")
        return None

    # US-align outputs live inside the CombFold dir for that complex
    combfold_output_dir = Path(row["combfold_output_path"]).parent.parent
    out_base = combfold_output_dir / "usalign_outputs"
    out_base.mkdir(exist_ok=True)  # shared by different prediction outputs
    out_dir = out_base / f"usalign_outputs_pred{_pred_index(row['combfold_output_path'])}"
    if out_dir.exists() and skip_existing:
        logger.info(f"{out_dir} exists -- skipping (--skip-existing).")
        return None
    out_dir.mkdir(exist_ok=skip_existing)

    def _run(target_dir: Path, chain1: str, chain2: str) -> str:
        target_dir.mkdir(exist_ok=True)
        cmd = [
            usalign_bin,
            row["combfold_output_path"],
            row["reference_pdb_path"],
            "-mm", str(mm),
            "-ter", str(ter),
            "-mol", "prot",
            "-chain1", chain1,
            "-chain2", chain2,
            "-o", str(target_dir / "usalign"),
        ]
        logger.info(f"Running USalign: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        (target_dir / "usalign_stdout.txt").write_text(proc.stdout)
        return proc.stdout

    results = {"complex": _run(out_dir / "complex", ",".join(cf_chains), ",".join(ref_chains))}

    for prot_id, (cf, ref) in mapping.items():
        assert len(cf) == len(ref), f"Mismatch in number of chains for {prot_id} between CF and reference."
        for cf_chain, ref_chain in zip(cf, ref):
            key = f"chain_{cf_chain}_{ref_chain}_{prot_id}"
            results[key] = _run(out_dir / key, cf_chain, ref_chain)

    return results


def parse_usalign_output(filepath: str) -> dict:
    """
    Parse a US-align stdout file and extract key metrics.

    tm_score is the one normalized by the length of Structure_2 (the reference).
    """
    text = Path(filepath).read_text()

    patterns = {
        "len_structure1": r"Length of Structure_1:\s*(\d+)\s*residues",
        "len_structure2": r"Length of Structure_2:\s*(\d+)\s*residues",
        "aligned_length": r"Aligned length=\s*(\d+)",
        "seq_id": r"Seq_ID=n_identical/n_aligned=\s*([\d.]+)",
        "rmsd": r"RMSD=\s*([\d.]+)",
        "tm_score": r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_2",
    }

    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"Could not find '{key}' in {filepath}")
        value = match.group(1)
        result[key] = int(value) if METRIC_DTYPES[key] == pl.Int64 else float(value)
    return result


def parse_usalign_per_chain(pred_dir: str) -> dict:
    """Collect per-chain-pair metrics from a usalign_outputs_pred* dir into parallel lists."""
    pred_path = Path(pred_dir)
    ids: list[str] = []
    values: dict[str, list] = {k: [] for k in METRIC_DTYPES}

    def _pack() -> dict:
        return {"usalign_per_chain_ID": ids} | {f"usalign_per_chain_{k}": v for k, v in values.items()}

    if not pred_path.is_dir():
        logger.warning(f"usalign per-chain dir missing: {pred_dir}")
        return _pack()

    for d in sorted(p for p in pred_path.iterdir() if p.is_dir() and p.name != "complex"):
        parts = d.name.split("_")
        assert len(parts) >= 4 and parts[0] == "chain", f"unexpected chain dir name: {d}"
        chain_a, chain_b, unp = parts[1], parts[2], "_".join(parts[3:])
        # assert len(chain_a) == 1 and len(chain_b) == 1, f"unexpected chain letters in: {d}" #! this appears to not always be the case, can have multi char chain names
        ids.append(f"{unp}_{chain_a}_{chain_b}")

        stdout_path = d / "usalign_stdout.txt"
        try:
            parsed = parse_usalign_output(str(stdout_path))
        except Exception as e:
            logger.warning(f"failed to parse usalign output at {stdout_path}: {e}")
            parsed = {k: None for k in METRIC_DTYPES}

        for k in METRIC_DTYPES:
            values[k].append(parsed.get(k))

    return _pack()


def add_usalign_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Add complex-level and per-chain US-align metric columns."""
    df = df.with_columns(
        pl.col("combfold_output_path")
        .str.replace(CF_OUTPUT_RE, "usalign_outputs/usalign_outputs_pred${1}/complex/usalign_stdout.txt")
        .alias("usalign_stdout_path"),
        pl.col("combfold_output_path")
        .str.replace(CF_OUTPUT_RE, "usalign_outputs/usalign_outputs_pred${1}")
        .alias("usalign_pred_dir"),
    )

    #TODO: currently unsure why no usalign output exists for these
    missing = ~df["usalign_pred_dir"].map_elements(lambda p: Path(p).is_dir(), return_dtype=pl.Boolean)
    if missing.any():
        for row in df.filter(missing).iter_rows(named=True):
            logger.warning(f"usalign output dir not found, skipping row: {row}")
        df = df.filter(~missing)

    df = df.with_columns(
        pl.col("usalign_stdout_path")
        .map_elements(parse_usalign_output, return_dtype=pl.Struct(METRIC_DTYPES))
        .alias("cpx")
    ).unnest("cpx")
    df = df.rename({c: f"usalign_cpx_{c}" for c in METRIC_DTYPES})

    per_chain_schema = pl.Struct(
        {"usalign_per_chain_ID": pl.List(pl.Utf8)}
        | {f"usalign_per_chain_{k}": pl.List(v) for k, v in METRIC_DTYPES.items()}
    )
    return df.with_columns(
        pl.col("usalign_pred_dir")
        .map_elements(parse_usalign_per_chain, return_dtype=per_chain_schema)
        .alias("per_chain")
    ).unnest("per_chain")


# --------------------------------------------------------------------------- #
# interfaces
# --------------------------------------------------------------------------- #
def get_close_by_chain_pairs(path: str, cutoff: float = 8.0, min_contacts: int = 5) -> dict:
    """Chain pairs with at least `min_contacts` CA-CA contacts within `cutoff` angstrom."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    model = st[0]

    ns = gemmi.NeighborSearch(model, st.cell, cutoff).populate()
    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.SameChain

    contact_count: dict[tuple[str, str], int] = defaultdict(int)
    for c in cs.find_contacts(ns):
        if c.partner1.atom.name != "CA" or c.partner2.atom.name != "CA":
            continue
        contact_count[tuple(sorted((c.partner1.chain.name, c.partner2.chain.name)))] += 1

    return {pair: n for pair, n in contact_count.items() if n >= min_contacts}


def add_close_by_chain_pairs(df: pl.DataFrame, cutoff: float = 8.0, min_contacts: int = 5) -> pl.DataFrame:
    """Add cf_/ref_close_by_chainpairs columns, keyed by UniProt pair instead of chain letters."""

    def invert(mapping: dict) -> tuple[dict, dict]:
        cf_to_unp, ref_to_unp = {}, {}
        for unp_id, (cf_chains, ref_chains) in mapping.items():
            cf_to_unp.update({c: unp_id for c in cf_chains})
            ref_to_unp.update({c: unp_id for c in ref_chains})
        return cf_to_unp, ref_to_unp

    def to_unp_keys(pair_dict: dict, chain_to_unp: dict) -> dict:
        return {f"{chain_to_unp.get(c1, c1)}_{chain_to_unp.get(c2, c2)}": v for (c1, c2), v in pair_dict.items()}

    cf_out, ref_out = [], []
    for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Close-by chain pairs", unit="row"):
        cf_to_unp, ref_to_unp = invert(row["chain_mapping"])
        cf_interfaces, ref_interfaces = {}, {}

        if row["combfold_output_path"] is None:
            logger.warning(f"{row['complex_ac'] or '?'}: no combfold_output_path -- skipping.")
        else:
            assert Path(row["combfold_output_path"]).exists(), (
                f"Combfold output path does not exist: {row['combfold_output_path']}"
            )
            cf_interfaces = get_close_by_chain_pairs(row["combfold_output_path"], cutoff, min_contacts)

        if row["reference_pdb_path"] is None:
            logger.warning(f"{row['complex_ac'] or '?'}: no reference_pdb_path -- skipping.")
        else:
            assert Path(row["reference_pdb_path"]).exists(), (
                f"Reference PDB path does not exist: {row['reference_pdb_path']}"
            )
            ref_interfaces = get_close_by_chain_pairs(row["reference_pdb_path"], cutoff, min_contacts)

        cf_out.append(to_unp_keys(cf_interfaces, cf_to_unp))
        ref_out.append(to_unp_keys(ref_interfaces, ref_to_unp))

    return df.with_columns(
        pl.Series("cf_close_by_chainpairs", cf_out, dtype=pl.Object),
        pl.Series("ref_close_by_chainpairs", ref_out, dtype=pl.Object),
    )


# --------------------------------------------------------------------------- #
# Stoic leakage flags
# --------------------------------------------------------------------------- #
def add_stoic_flags(df: pl.DataFrame, stoic_csv: Path, sifts_csv: Path) -> pl.DataFrame:
    """Flag whether the exact PDB, or any PDB with an identical protein set, is in Stoic training."""
    stoic_training = pl.read_csv(stoic_csv).filter(pl.col("split") == "train")

    df = df.with_columns(
        exact_pdb_in_stoic=pl.col("pdb_id")
        .str.to_lowercase()
        .is_in(stoic_training["pdb_id"].str.to_lowercase().implode())
    )

    pdb_uniprot_mapping = pl.read_csv(
        sifts_csv,
        skip_rows=1,
        schema_overrides={c: pl.Utf8 for c in ("RES_BEG", "RES_END", "PDB_BEG", "PDB_END", "SP_BEG", "SP_END")},
    ).rename(str.lower)

    pdb_to_proteins = pdb_uniprot_mapping.group_by("pdb").agg(
        pl.col("sp_primary").unique().alias("proteins"),
        pl.col("chain").n_unique().alias("n_chains"),
    )

    # stoic training only holds PDB IDs, so attach the protein sets
    stoic_training = stoic_training.select(["pdb_id", "split"]).join(
        pdb_to_proteins, left_on="pdb_id", right_on="pdb", how="left"
    )

    def canon(x):
        return x.list.sort().list.join(",")

    in_stoic_training = set(canon(stoic_training["proteins"]).to_list())

    return (
        df.with_columns(
            pl.col("CP_stochiometry")
            .map_elements(lambda s: list(json.loads(s).keys()), return_dtype=pl.List(pl.Utf8))
            .alias("CP_stochiometry_protlist")
        )
        .with_columns(canon(pl.col("CP_stochiometry_protlist")))
        .with_columns(
            pl.col("CP_stochiometry_protlist")
            .is_in(in_stoic_training)
            .alias("pdb_with_identical_proteins_set_to_CF_in_stoic")
        )
        .drop("CP_stochiometry_protlist")
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def serialize_object_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Object-dtype columns can't be written to parquet; dump them to JSON strings."""
    obj_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Object]
    if not obj_cols:
        return df
    return df.with_columns(
        pl.col(c).map_elements(lambda x: json.dumps(x) if x is not None else None, return_dtype=pl.Utf8)
        for c in obj_cols
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-x3-output-path", type=Path, required=True,
                        help="Input parquet from step X3 (…_with_combfold_results_paths.parquet).")
    parser.add_argument("--reference-pdb-dir", type=Path, default=DEFAULT_REFERENCE_PDB_DIR,
                        help="Dir containing per-PDB subdirs with reference .pdb/.cif files.")
    parser.add_argument("--sifts-csv", type=Path, default=DEFAULT_SIFTS_CSV,
                        help="SIFTS pdb_chain_uniprot.csv.")
    parser.add_argument("--stoic-csv", type=Path, default=DEFAULT_STOIC_CSV,
                        help="Stoic data_file_stoic.csv (used for leakage flags).")
    parser.add_argument("--usalign-bin", type=Path, default=DEFAULT_USALIGN_BIN,
                        help="Path to the USalign binary.")
    parser.add_argument("--output-path", type=Path, default=None,
                        help=f"Output parquet path. Defaults to "
                             f"<step_x3_output_path.parent>/{DEFAULT_OUTPUT_NAME}")

    parser.add_argument("--max-workers", type=int, default=8, help="Threads for running US-align.")
    parser.add_argument("--mm", type=int, default=1, help="US-align -mm option.")
    parser.add_argument("--ter", type=int, default=1, help="US-align -ter option.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip complexes whose usalign_outputs_pred* dir already exists "
                             "(default: fail, so reruns don't silently mix old and new results).")
    parser.add_argument("--skip-usalign", action="store_true",
                        help="Don't run US-align; only parse existing outputs.")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Contact distance cutoff (angstrom).")
    parser.add_argument("--min-contacts", type=int, default=5, help="Min CA-CA contacts to call an interface.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_path = args.output_path or args.step_x3_output_path.parent / DEFAULT_OUTPUT_NAME

    df = pl.read_parquet(args.step_x3_output_path)
    assert df["found_match_reference_pdb_model"].all(), (
        "Not all rows have a matching reference PDB model. This should have been filtered in step X3."
    )

    df_long = build_long_df(df, args.reference_pdb_dir)

    sifts_lookup = build_sifts_chain_lookup(
        args.sifts_csv, set(df_long["pdb_id"].str.to_lowercase().unique().to_list())
    )
    df_long = df_long.with_columns(
        pl.struct(["reference_pdb_path", "combfold_output_path", "complex_ac"])
        .map_elements(
            lambda r: build_chain_mapping(
                r["reference_pdb_path"], r["combfold_output_path"], sifts_lookup, r["complex_ac"]
            ),
            return_dtype=pl.Object,
        )
        .alias("chain_mapping")
    )

    if args.skip_usalign:
        logger.info("--skip-usalign set; parsing existing US-align outputs only.")
    else:
        rows = list(df_long.iter_rows(named=True))
        logger.info(f"Running US-align for {rows}.")
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            list(
                tqdm(
                    ex.map(
                        lambda r: run_usalign_for_row(
                            r, str(args.usalign_bin), args.mm, args.ter, args.skip_existing
                        ),
                        rows,
                    ),
                    total=len(rows),
                    desc="US-align",
                    unit="pred",
                )
            )

    df_long = add_usalign_metrics(df_long)
    df_long = add_close_by_chain_pairs(df_long, args.cutoff, args.min_contacts)
    df_long = add_stoic_flags(df_long, args.stoic_csv, args.sifts_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialize_object_cols(df_long).write_parquet(output_path)
    logger.info(f"Wrote {df_long.height} rows to {output_path}")


if __name__ == "__main__":
    main()