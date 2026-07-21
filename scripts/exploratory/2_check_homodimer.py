#!/usr/bin/env python
"""
check_homodimer.py — Definitively check which homodimer structures for a
protein actually exist in the foldcomp DB, broken down by input_type
(pair/pool) and sample.

Usage:
    python check_homodimer.py P00445
    python check_homodimer.py p00445
    python check_homodimer.py P00445 --db /path/to/predictions-db

Queries the foldcomp DB directly (not the parquet metadata), so it reports
ground truth: a sample is "OK" only if BOTH chain structures are present in
foldcomp. Scores existing in the parquet does NOT count — that's the exact
discrepancy this tool exposes.

NOTE: summary_models.parquet does NOT contain db_id1/db_id2 columns. Those
are synthesized by yp.get_models() as {input_name}_{sample}_{chain_id}. We
construct them the same way here, so the keys match what yp produces.
"""

import argparse
import os
import sys

import polars as pl
import foldcomp

DEFAULT_DB_DIR = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
DEFAULT_FC_DB = os.path.join(DEFAULT_DB_DIR, "predictions-db", "predictions-db")
DEFAULT_PARQUET = os.path.join(DEFAULT_DB_DIR, "summary_models.parquet")


def main():
    parser = argparse.ArgumentParser(
        description="Check which homodimer structures exist in foldcomp for a protein."
    )
    parser.add_argument("protein", help="UniProt ID (case-insensitive, e.g. P00445 or p00445)")
    parser.add_argument("--db", default=DEFAULT_FC_DB,
                        help=f"Path to foldcomp predictions-db (default: {DEFAULT_FC_DB})")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                        help=f"Path to summary_models.parquet (default: {DEFAULT_PARQUET})")
    args = parser.parse_args()

    protein_orig = args.protein
    protein_lo = protein_orig.lower()

    fc_db = args.db
    parquet_path = args.parquet

    if not os.path.exists(fc_db):
        print(f"ERROR: foldcomp DB not found at {fc_db}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(parquet_path):
        print(f"ERROR: parquet not found at {parquet_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Protein:        {protein_orig} (query: {protein_lo})")
    print(f"Foldcomp DB:    {fc_db}")
    print(f"Parquet:        {parquet_path}")
    print()

    # --- Pull all rows for this protein's self-pair from the parquet ---
    # Lazy scan with predicate pushdown: only reads matching rows from the
    # ~69M-row parquet, not the whole file. Construct db_id keys the same
    # way yp.get_models() does: {input_name}_{sample}_{chain_id}.
    # CRITICAL: with_columns MUST come before select, since db_id1/db_id2
    # don't exist in the raw parquet — they're synthesized here.
    rows = (
        pl.scan_parquet(parquet_path)
        .filter(
            (pl.col("af3_id1") == protein_lo) &
            (pl.col("af3_id2") == protein_lo)
        )
        .with_columns(
            (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8)
             + "_" + pl.col("chain_id1")).alias("db_id1"),
            (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8)
             + "_" + pl.col("chain_id2")).alias("db_id2"),
        )
        .select("input_type", "sample", "ranking_score", "chain_pair_iptm",
                "db_id1", "db_id2")
        .sort(["input_type", "sample"])
        .collect()
    )

    if rows.height == 0:
        print(f"No rows found in summary_models.parquet for {protein_lo} self-pair.")
        print("This protein has no homodimer predictions (pair or pool) in this DB.")
        sys.exit(0)

    print(f"summary_models.parquet rows for {protein_lo} self-pair: {rows.height}")
    print()

    # --- Check every db_id against foldcomp in ONE batched call ---
    # Single foldcomp.open() with all keys at once — one DB open, one pass.
    all_db_ids = sorted(set(rows["db_id1"].to_list() + rows["db_id2"].to_list()))
    found = set()
    with foldcomp.open(fc_db, ids=all_db_ids) as db:
        for name, _ in db:
            found.add(name)

    # --- Report per-row status ---
    print(f"{'type':<6} {'sample':<7} {'rank_score':<11} {'iptm':<6} "
          f"{'chain A':<28} {'chain B':<28} {'status'}")
    print("-" * 110)

    counts = {"pair": {"ok": 0, "missing": 0}, "pool": {"ok": 0, "missing": 0}}
    found_list = list(found)

    for row in rows.iter_rows(named=True):
        k1, k2 = row["db_id1"], row["db_id2"]
        k1_ok = k1 in found
        k2_ok = k2 in found
        if k1_ok and k2_ok:
            status = "OK"
            counts[row["input_type"]]["ok"] += 1
        elif k1_ok or k2_ok:
            status = f"PARTIAL ({'A' if k1_ok else 'B'} only)"
            counts[row["input_type"]]["missing"] += 1
        else:
            status = "MISSING"
            counts[row["input_type"]]["missing"] += 1

        print(f"{row['input_type']:<6} {row['sample']:<7} "
              f"{row['ranking_score']:<11.3f} {row['chain_pair_iptm']:<6.2f} "
              f"{k1:<28} {k2:<28} {status}")

    # --- Summary ---
    print()
    print("SUMMARY:")
    for itype in ["pair", "pool"]:
        ok = counts[itype]["ok"]
        miss = counts[itype]["missing"]
        total = ok + miss
        if total == 0:
            print(f"  {itype}: no samples in parquet")
        else:
            pct = 100.0 * ok / total if total else 0
            print(f"  {itype}: {ok}/{total} samples present in foldcomp ({pct:.0f}%)")

    # Best available sample per type — compute from already-loaded rows
    # and the found set. No second parquet scan, no second foldcomp open.
    for itype in ["pair", "pool"]:
        best = (
            rows.filter(pl.col("input_type") == itype)
            .filter(pl.col("db_id1").is_in(found_list))
            .filter(pl.col("db_id2").is_in(found_list))
            .sort("ranking_score", descending=True)
            .head(1)
        )
        if best.height > 0:
            r = best.row(0, named=True)
            print(f"  Best available {itype} sample: {r['sample']} "
                  f"(ranking_score={r['ranking_score']:.3f}, iptm={r['chain_pair_iptm']:.2f})")
        else:
            print(f"  Best available {itype} sample: NONE (no structures in foldcomp)")

    # Verdict
    print()
    pair_ok = counts["pair"]["ok"]
    pair_total = counts["pair"]["ok"] + counts["pair"]["missing"]
    if pair_total > 0 and pair_ok == 0:
        print(f"VERDICT: {protein_orig} has pair SCORES in the parquet but ZERO pair "
              f"STRUCTURES in foldcomp — this is the ingestion gap, not a lookup bug.")
    elif pair_total > 0 and pair_ok < pair_total:
        print(f"VERDICT: {protein_orig} has PARTIAL pair coverage ({pair_ok}/{pair_total}) "
              f"in foldcomp — some samples ingested, some missing.")
    elif pair_total > 0:
        print(f"VERDICT: {protein_orig} has FULL pair coverage ({pair_ok}/{pair_total}) "
              f"in foldcomp — all samples present.")


if __name__ == "__main__":
    main()
