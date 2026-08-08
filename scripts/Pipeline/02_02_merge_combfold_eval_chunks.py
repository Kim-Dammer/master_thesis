#!/usr/bin/env python3
"""merge_combfold_eval_chunks.py

Merge per-chunk combfold_eval outputs (produced by the SLURM array driven by
submit_combfold_eval.sh / submit_combfold_eval_array.sbatch) back into a
single set of complex_summary.csv / per_chain.csv / per_interface.csv.

Usage:
    python3 merge_combfold_eval_chunks.py \
        --chunks-dir /path/to/combfold_eval_out \
        --out-dir /path/to/combfold_eval_out \
        [--pattern "chunk_*"]

Each "<chunks-dir>/<pattern>/" directory is expected to contain zero or more
of complex_summary.csv, per_chain.csv, per_interface.csv (a chunk may be
missing a file if that chunk's array task failed or produced no rows --
these are warned about, not treated as fatal, since a partial run may still
be worth inspecting).

For each of the three files, this script:
  1. Reads every chunk's copy that exists.
  2. Concatenates them and checks the merged row count equals the sum of
     per-chunk row counts (catches truncated/corrupted reads).
  3. Checks no complex_ac value appears in more than one chunk (chunks are
     built from disjoint slices of a sorted, deduplicated complex_ac list,
     so any overlap indicates a chunk-slicing bug, not expected behavior).
  4. Sorts the merged rows by whichever of a fixed key-column list are
     present, for deterministic output order independent of chunk-glob order.

Exit code is 1 if any check fails for any file, 0 otherwise.
"""
import argparse
import glob
import os
import sys

import pandas as pd

OUTPUT_FILES = ["complex_summary.csv", "per_chain.csv", "per_interface.csv"]
SORT_KEYS = ["complex_ac", "stoich_source", "cf_folder_type", "pdb_id", "cf_cluster"]


def merge_one_file(fname: str, chunk_dirs: list):
    """Merge a single output filename across all chunk dirs.

    Returns (merged_df, failed) where merged_df is None if no chunk had this
    file at all (nothing to merge), and failed is 1 if any consistency check
    failed, else 0.
    """
    frames = []  # list of (chunk_dir, DataFrame) -- kept paired for the
    # disjointness check below, so we never need to re-read any CSV a second
    # time just to know which chunk a row came from.
    missing = []
    for cdir in chunk_dirs:
        fpath = os.path.join(cdir, fname)
        if not os.path.isfile(fpath):
            missing.append(cdir)
            continue
        df = pd.read_csv(fpath)
        frames.append((cdir, df))

    if missing:
        print(f"[merge] WARNING: {fname} missing in {len(missing)} chunk(s): "
              f"{', '.join(os.path.basename(m) for m in missing)}", file=sys.stderr)

    if not frames:
        print(f"[merge] WARNING: no chunks contained {fname}; skipping.", file=sys.stderr)
        return None, 0

    failed = 0

    # Disjointness check: no complex_ac should appear in more than one chunk.
    if all("complex_ac" in df.columns for _, df in frames):
        seen = {}
        for cdir, df in frames:
            for ac in df["complex_ac"].astype(str).str.strip().unique():
                if ac in seen and seen[ac] != cdir:
                    print(f"[merge] ERROR: {fname}: complex_ac '{ac}' appears in both "
                          f"{os.path.basename(seen[ac])} and {os.path.basename(cdir)} "
                          f"(chunks should be disjoint by construction)", file=sys.stderr)
                    failed = 1
                seen[ac] = cdir

    expected_rows = sum(len(df) for _, df in frames)
    merged = pd.concat([df for _, df in frames], ignore_index=True)
    if len(merged) != expected_rows:
        print(f"[merge] ERROR: {fname}: merged row count {len(merged)} != "
              f"sum of per-chunk row counts {expected_rows}", file=sys.stderr)
        failed = 1

    sort_cols = [c for c in SORT_KEYS if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return merged, failed



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks-dir", required=True, help="Directory containing chunk_* subfolders.")
    ap.add_argument("--out-dir", required=True, help="Where to write the merged CSVs.")
    ap.add_argument("--pattern", default="chunk_*", help="Glob pattern for chunk subfolders (default: chunk_*).")
    a = ap.parse_args()

    chunk_dirs = sorted(
        d for d in glob.glob(os.path.join(a.chunks_dir, a.pattern)) if os.path.isdir(d)
    )
    if not chunk_dirs:
        print(f"[merge] ERROR: no directories matched {os.path.join(a.chunks_dir, a.pattern)}", file=sys.stderr)
        return 1
    print(f"[merge] found {len(chunk_dirs)} chunk dir(s): "
          f"{', '.join(os.path.basename(d) for d in chunk_dirs)}")

    os.makedirs(a.out_dir, exist_ok=True)

    any_failed = 0
    for fname in OUTPUT_FILES:
        merged, failed = merge_one_file(fname, chunk_dirs)
        if merged is None:  # skipped, no chunks had this file
            continue
        any_failed = any_failed or failed
        out_path = os.path.join(a.out_dir, fname)
        merged.to_csv(out_path, index=False)
        status = "OK" if not failed else "FAILED CHECKS (see above)"
        print(f"[merge] {fname}: {len(merged)} rows -> {out_path} [{status}]")

    if any_failed:
        print("[merge] one or more checks failed -- inspect the errors above before trusting the merged output.",
              file=sys.stderr)
        return 1
    print("[merge] all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
