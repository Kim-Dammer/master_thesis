#!/usr/bin/env python3
"""
aggregate_stoic.py
==================

Walk a folder of Stoic prediction results (one subfolder per complex, named
``CPX-<id>``, each containing a ``results.json`` and a number of
``af3_input_*.json`` files) and produce a single CSV with **one row per
complex**.

Each row carries:

* ``cpx_id`` (folder name),
* ``uniprot_ids`` — sorted, space-separated UniProt IDs of all proteins
  involved (matching the ``true_complex`` / ``predicted_complex`` format),
* ``n_predictions`` — actual number of Stoic predictions found for that CPX,
* up to ``--max-preds`` (default 10) prediction pairs sorted by probability
  descending (best first). Each prediction occupies two columns:

    - ``pred_N_stoichiometry`` — JSON dict ``{UniProt_ID: stoichiometry}``,
    - ``pred_N_score``         — JSON dict ``{"rank": int, "probability": float}``.

* all remaining columns from ``first_setup_pipeline_complexes.csv``, joined
  on ``#Complex ac == cpx_id``. If a CPX appears in multiple pipeline-CSV
  rows, the row with the highest ``jaccard_similarity`` is kept (ties
  broken by ``confidence_score`` descending).

Sequences in ``results.json`` are mapped to UniProt IDs via exact equality
against ``uniprot_mapped_sequences.csv``. Any sequence that fails to map
raises ``KeyError`` and stops the script (fail-loud behaviour).

EXAMPLE COMMAND (the user's current paths)
------------------------------------------
::

    python aggregate_stoic.py \\
        --stoic-dir    /.../data/Pipeline/first_setup/stoic_results \\
        --pipeline-csv /.../data/Pipeline/first_setup/first_setup_pipeline_complexes.csv \\
        --uniprot-map  /.../data/Pipeline/first_setup/uniprot_mapped_sequences.csv \\
        --out          /.../data/Pipeline/first_setup/stoic_results_aggregated.csv

All four ``--…`` arguments default to the paths above, so the command can
also be run without any arguments on that filesystem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from procompa import get_project_root

# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #
DEFAULT_STOIC_DIR = (
    get_project_root() / "data" / "Pipeline" / "first_setup" / "stoic_results"
)
DEFAULT_PIPELINE_CSV = (
    get_project_root() / "data" / "Pipeline" / "first_setup" / "first_setup_pipeline_complexes.csv"
)
DEFAULT_UNIPROT_MAP = (
    get_project_root() / "data" / "Pipeline" / "first_setup" / "uniprot_mapped_sequences.csv"
)
DEFAULT_OUT = (
    get_project_root() / "data" / "Pipeline" / "first_setup" / "stoic_results_aggregated.csv"
)
DEFAULT_MAX_PREDS = 10

# Reserved (non-sequence) keys in each prediction dict of results.json.
PREDICTION_META_KEYS = {"rank", "probability"}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_uniprot_mapping(path: Path) -> dict[str, str]:
    """Return ``{sequence: uniprot_id}`` from a two-column CSV.

    Raises if the same sequence maps to multiple distinct UniProt IDs.
    """
    df = pd.read_csv(path)
    expected_cols = {"uniprot_id", "sequence"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}; "
            f"found columns: {list(df.columns)}"
        )

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        seq = str(row["sequence"]).strip()
        uid = str(row["uniprot_id"]).strip()
        if not seq or not uid:
            continue
        if seq in mapping and mapping[seq] != uid:
            raise ValueError(
                f"Duplicate sequence in {path} maps to two different "
                f"UniProt IDs: {mapping[seq]!r} vs {uid!r}. "
                f"Sequence (first 60 chars): {seq[:60]!r}"
            )
        mapping[seq] = uid
    return mapping


def load_pipeline_csv(path: Path) -> pd.DataFrame:
    """Load the pipeline CSV, deduplicate by ``#Complex ac``.

    For duplicate CPX IDs, keep the row with the highest ``jaccard_similarity``
    (ties broken by ``confidence_score`` descending).
    """
    df = pd.read_csv(path)
    if "#Complex ac" not in df.columns:
        raise ValueError(
            f"{path} must have a '#Complex ac' column; "
            f"found columns: {list(df.columns)}"
        )

    # Sort so the "best" row per CPX appears first, then keep first per CPX.
    sort_cols = []
    ascending = []
    if "jaccard_similarity" in df.columns:
        sort_cols.append("jaccard_similarity")
        ascending.append(False)
    if "confidence_score" in df.columns:
        sort_cols.append("confidence_score")
        ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    df = df.drop_duplicates(subset="#Complex ac", keep="first").reset_index(drop=True)
    return df


def parse_prediction_entry(
    entry: dict[str, Any],
    seq_to_uniprot: dict[str, str],
    cpx_id: str,
    entry_idx: int,
) -> tuple[float, int, dict[str, int]]:
    """Convert one prediction dict into ``(probability, rank, stoich)``.

    ``stoich`` is ``{uniprot_id: stoichiometry}``.

    Raises ``KeyError`` if a sequence is not in the mapping.
    Raises ``ValueError`` if ``rank`` / ``probability`` are missing.
    """
    if "probability" not in entry or "rank" not in entry:
        raise ValueError(
            f"[{cpx_id}] prediction #{entry_idx} is missing 'rank' or "
            f"'probability'. Entry keys: {list(entry.keys())[:5]}..."
        )

    probability = float(entry["probability"])
    # Stoic encodes ``rank`` as a float in JSON; cast to int because integers
    # are what the JSON file actually represents semantically.
    rank = int(entry["rank"])

    stoich: dict[str, int] = {}
    for key, value in entry.items():
        if key in PREDICTION_META_KEYS:
            continue
        # The remaining keys are full protein sequences; the value is the
        # stoichiometry (copy count) for that sequence in this prediction.
        seq = str(key)
        if seq not in seq_to_uniprot:
            raise KeyError(
                f"[{cpx_id}] prediction #{entry_idx} contains a sequence "
                f"that is not present in the UniProt mapping file. "
                f"Sequence (first 60 chars): {seq[:60]!r} "
                f"(length {len(seq)})."
            )
        uid = seq_to_uniprot[seq]
        if uid in stoich:
            raise ValueError(
                f"[{cpx_id}] prediction #{entry_idx} maps two different "
                f"sequence keys to the same UniProt ID {uid!r}; this should "
                f"not happen in a Stoic results.json."
            )
        stoich[uid] = int(value)

    return probability, rank, stoich


def process_cpx_folder(
    folder: Path,
    seq_to_uniprot: dict[str, str],
    max_preds: int,
) -> dict[str, Any]:
    """Process a single CPX-* folder and return a row dict for the output."""
    cpx_id = folder.name
    results_path = folder / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"[{cpx_id}] missing results.json at {results_path}")

    with results_path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(
            f"[{cpx_id}] results.json is expected to be a JSON list of "
            f"prediction dicts; got {type(data).__name__}."
        )
    if not data:
        raise ValueError(f"[{cpx_id}] results.json is an empty list.")

    parsed: list[tuple[float, int, dict[str, int]]] = []
    for i, entry in enumerate(data):
        parsed.append(parse_prediction_entry(entry, seq_to_uniprot, cpx_id, i))

    # Sort by probability descending (best first). Ties: keep input order.
    parsed.sort(key=lambda t: t[0], reverse=True)

    n_predictions = len(parsed)
    kept = parsed[:max_preds]
    if n_predictions > max_preds:
        print(
            f"[{cpx_id}] WARNING: results.json has {n_predictions} "
            f"predictions, keeping top {max_preds} by probability.",
            file=sys.stderr,
        )

    # Union of all UniProt IDs across all predictions.
    uniprot_set: set[str] = set()
    for _, _, stoich in parsed:
        uniprot_set.update(stoich.keys())
    uniprot_ids = " ".join(sorted(uniprot_set))

    row: dict[str, Any] = {
        "cpx_id": cpx_id,
        "uniprot_ids": uniprot_ids,
        "n_predictions": n_predictions,
    }

    for slot in range(1, max_preds + 1):
        if slot <= len(kept):
            probability, rank, stoich = kept[slot - 1]
            row[f"pred_{slot}_stoichiometry"] = json.dumps(
                stoich, sort_keys=True, separators=(",", ":")
            )
            row[f"pred_{slot}_score"] = json.dumps(
                {"rank": rank, "probability": probability},
                separators=(",", ":"),
            )
        else:
            row[f"pred_{slot}_stoichiometry"] = ""
            row[f"pred_{slot}_score"] = ""

    return row


def attach_pipeline_columns(
    stoic_rows: list[dict[str, Any]],
    pipeline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join the Stoic rows with the pipeline CSV on cpx_id."""
    stoic_df = pd.DataFrame(stoic_rows)

    # Pipeline CSV: rename '#Complex ac' to cpx_id for the join; keep all
    # other columns (do NOT drop columns the user wants from that CSV).
    pipe = pipeline_df.rename(columns={"#Complex ac": "cpx_id"})
    # Avoid column collisions on cpx_id-derived names: none of the pipeline
    # CSV columns clash with our newly created ones, but check defensively.
    overlap = (set(pipe.columns) & set(stoic_df.columns)) - {"cpx_id"}
    if overlap:
        # Rename pipeline columns with a suffix so we never silently
        # overwrite Stoic output columns.
        pipe = pipe.rename(columns={c: f"{c}__pipeline" for c in overlap})

    merged = stoic_df.merge(pipe, on="cpx_id", how="left")

    # Warn about CPX folders not found in the pipeline CSV.
    pipeline_cpx_ids = set(pipe["cpx_id"].dropna().astype(str))
    missing = sorted(set(stoic_df["cpx_id"]) - pipeline_cpx_ids)
    if missing:
        print(
            f"WARNING: {len(missing)} CPX folder(s) not found in pipeline "
            f"CSV (pipeline columns will be NaN): {missing}",
            file=sys.stderr,
        )

    # Also report CPX rows in the pipeline CSV without a folder.
    folder_cpx_ids = set(stoic_df["cpx_id"])
    only_in_csv = sorted(pipeline_cpx_ids - folder_cpx_ids)
    if only_in_csv:
        print(
            f"INFO: {len(only_in_csv)} CPX row(s) in pipeline CSV have no "
            f"corresponding Stoic folder (omitted from output): "
            f"{only_in_csv}",
            file=sys.stderr,
        )

    return merged


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Aggregate Stoic prediction results into a per-complex CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--stoic-dir", default=DEFAULT_STOIC_DIR,
                   help="Folder containing CPX-* subfolders.")
    p.add_argument("--pipeline-csv", default=DEFAULT_PIPELINE_CSV,
                   help="first_setup_pipeline_complexes.csv path.")
    p.add_argument("--uniprot-map", default=DEFAULT_UNIPROT_MAP,
                   help="uniprot_mapped_sequences.csv (columns: uniprot_id,sequence).")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="Output CSV path.")
    p.add_argument("--max-preds", type=int, default=DEFAULT_MAX_PREDS,
                   help="Number of prediction-column pairs to emit (default: 10).")
    args = p.parse_args(argv)

    stoic_dir = Path(args.stoic_dir)
    if not stoic_dir.is_dir():
        print(f"ERROR: --stoic-dir does not exist: {stoic_dir}", file=sys.stderr)
        return 1

    print(f"Loading UniProt mapping: {args.uniprot_map}")
    seq_to_uniprot = load_uniprot_mapping(Path(args.uniprot_map))
    print(f"  -> {len(seq_to_uniprot)} unique sequences mapped.")

    print(f"Loading pipeline CSV: {args.pipeline_csv}")
    pipeline_df = load_pipeline_csv(Path(args.pipeline_csv))
    print(f"  -> {len(pipeline_df)} rows after deduplicating by '#Complex ac'.")

    # Iterate CPX folders deterministically (sorted by natural CPX number when
    # possible, else lexicographic).
    def cpx_sort_key(folder: Path) -> tuple[int, str]:
        name = folder.name
        if name.startswith("CPX-"):
            tail = name.split("-", 1)[1]
            try:
                return (int(tail), name)
            except ValueError:
                pass
        return (10**12, name)

    cpx_folders = sorted(
        [f for f in stoic_dir.iterdir() if f.is_dir() and f.name.startswith("CPX-")],
        key=cpx_sort_key,
    )
    print(f"Found {len(cpx_folders)} CPX-* folder(s) in {stoic_dir}.")

    stoic_rows: list[dict[str, Any]] = []
    n_preds_hist: dict[int, int] = {}
    for folder in cpx_folders:
        row = process_cpx_folder(folder, seq_to_uniprot, args.max_preds)
        stoic_rows.append(row)
        n_preds_hist[row["n_predictions"]] = (
            n_preds_hist.get(row["n_predictions"], 0) + 1
        )
        print(
            f"  {row['cpx_id']}: n_predictions={row['n_predictions']}, "
            f"uniprot_ids={row['uniprot_ids']}"
        )

    if not stoic_rows:
        print("ERROR: no CPX-* folders processed; nothing to write.",
              file=sys.stderr)
        return 1

    merged = attach_pipeline_columns(stoic_rows, pipeline_df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print()
    print(f"Wrote {len(merged)} rows x {len(merged.columns)} columns -> {out_path}")
    print(f"n_predictions histogram: {dict(sorted(n_preds_hist.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
