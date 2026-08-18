#!/usr/bin/env python3
"""Attach CombFold result paths and output counts to the PDB-eval aggregate table."""

import argparse
import json
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def format_dir_name(dir_name: str) -> dict[str, int]:
    """
    Convert a CombFold output dir name into a stoichiometry dict.

    e.g. "P25604x1_P42939x1_Q02767x1_Q99176x1_pool_output" ->
         {"P25604": 1, "P42939": 1, "Q02767": 1, "Q99176": 1}

    Dir names end in either "_pool_output" or "_pair_output".
    """
    assert dir_name.endswith("_pool_output") or dir_name.endswith("_pair_output"), (
        f"Directory name {dir_name} does not end with _pool_output or _pair_output"
    )
    prot_dict = {}
    for prot in dir_name.split("_")[:-2]:
        prot_id, count = prot.split("x")
        prot_dict[prot_id] = int(count)
    return prot_dict


def build_lookup(combfold_output_dir: Path) -> dict[frozenset, Path]:
    """Map stoichiometry (as a frozenset of items) -> CombFold output dir path."""
    pairs = []
    for subdir in combfold_output_dir.iterdir():
        if not subdir.name.endswith("_output"):
            continue
        if "__" in subdir.name:
            logger.warning(f"Skipping ambiguous dir name for very large complex: {subdir.name}")
            continue
        pairs.append((format_dir_name(subdir.name), subdir))
    return {frozenset(d.items()): path for d, path in pairs}


def count_combfold_outputs(combfold_result_path: str) -> int:
    """Count output_* dirs in {combfold_result_path}/assembled_results/."""
    path = Path(combfold_result_path)
    assert path is not None and path.is_dir(), f"Combfold result path {path} is not a directory"

    assembled_results_dir = path / "assembled_results"
    if not (path / "_unified_representation").is_dir():
        logger.warning(
            f"{path} does not contain _unified_representation subdir. Probably pair not in database."
        )
    if not assembled_results_dir.is_dir():
        logger.warning(f"{path} does not contain assembled_results subdir. Combfold failed to assemble.")
        return 0
    return len(list(assembled_results_dir.glob("output_*")))


def process(step_x2_output_path: Path, combfold_output_dir: Path, keep_unmatched: bool) -> pl.DataFrame:
    df = pl.read_parquet(step_x2_output_path)

    if not keep_unmatched:
        df = df.filter(pl.col("found_match_reference_pdb_model") == True)

    #TODO This should use the _input dir for too long complex predictions to infer stoichiometry instead of filtering those out
    lookup = build_lookup(combfold_output_dir)

    def lookup_path(s: str) -> str | None:
        match = lookup.get(frozenset(json.loads(s).items()))
        return str(match) if match is not None else None

    df = df.with_columns(
        pl.col("CP_stochiometry")
        .map_elements(lookup_path, return_dtype=pl.Utf8)
        .alias("Combfold_result_path")
    )

    dropped = df.filter(pl.col("Combfold_result_path").is_null())
    if dropped.height:
        logger.warning(
            f"Dropped {dropped.height} rows with no matching CombFold output:\n"
            f"{dropped.select('complex_ac', 'identifiers', 'CP_stochiometry')}"
        )
    df = df.filter(pl.col("Combfold_result_path").is_not_null())

    df = df.with_columns(
        pl.col("Combfold_result_path")
        .map_elements(count_combfold_outputs, return_dtype=pl.Int64)
        .alias("n_combfold_outputs")
    )

    # Object-dtype columns can't be serialized to parquet directly; dump to JSON strings.
    obj_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Object]
    if obj_cols:
        df = df.with_columns(
            pl.col(c).map_elements(lambda x: json.dumps(x) if x is not None else None, return_dtype=pl.Utf8)
            for c in obj_cols
        )

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combfold-output-dir",
        type=Path,
        help="Root dir containing CombFold *_pool_output / *_pair_output subdirs.",
    )
    parser.add_argument(
        "--step-x2-output-path",
        type=Path,
        help="Input parquet from step X2.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to "
        "<step_x2_output_path.parent>/aggregate_cf_for_pdb_eval_with_combfold_results_paths.parquet",
    )
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep rows where found_match_reference_pdb_model is False (default: filter them out).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    output_dir = args.step_x2_output_path.parent
    assert output_dir.is_dir(), f"Output dir {output_dir} is not a directory"
    output_path = args.output_path or output_dir / "aggregate_cf_for_pdb_eval_with_combfold_results_paths.parquet"

    df = process(args.step_x2_output_path, args.combfold_output_dir, args.keep_unmatched)
    df.write_parquet(output_path)
    logger.info(f"Wrote {df.height} rows to {output_path}")


if __name__ == "__main__":
    main()