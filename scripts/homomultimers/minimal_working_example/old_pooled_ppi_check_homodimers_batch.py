#!/usr/bin/env python
"""
check_homodimers_batch.py — Survey pair/pool structure coverage across many
homodimers in ONE pass. Answers "for how many proteins do I get pair vs pool
vs missing" without running CombFold at all.

Usage:
    # proteins as arguments
    python check_homodimers_batch.py P00445 Q01217 P40202 O13297

    # proteins from a file (one ID per line)
    python check_homodimers_batch.py --from-file my_proteins.txt

    # save full per-protein table to CSV
    uv run check_homodimers_batch.py --from-file /cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_CP_proteins.csv --out coverage.csv

Efficiency: ONE parquet scan (predicate-pushed to self-pairs of the requested
proteins) + ONE foldcomp.open (all keys across all proteins batched). Runs in
seconds on a login node regardless of how many proteins you check.
"""

import argparse
import os
import sys
from pathlib import Path

import polars as pl
import foldcomp

DEFAULT_DB_DIR = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
DEFAULT_FC_DB = os.path.join(DEFAULT_DB_DIR, "predictions-db", "predictions-db")
DEFAULT_PARQUET = os.path.join(DEFAULT_DB_DIR, "summary_models.parquet")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-survey homodimer pair/pool coverage in foldcomp."
    )
    parser.add_argument("proteins", nargs="*", help="Protein IDs to check")
    parser.add_argument("--from-file", dest="from_file", default=None,
                        help="File with protein IDs (one per line)")
    parser.add_argument("--db", default=DEFAULT_FC_DB,
                        help=f"Path to foldcomp predictions-db (default: {DEFAULT_FC_DB})")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                        help=f"Path to summary_models.parquet (default: {DEFAULT_PARQUET})")
    parser.add_argument("--out", default=None,
                        help="Write per-protein CSV table to this path")
    args = parser.parse_args()

    # --- Collect protein list ---
    proteins = list(args.proteins)
    if args.from_file:
        with open(args.from_file) as fh:
            proteins.extend(line.strip() for line in fh if line.strip())
    if not proteins:
        parser.error("provide protein IDs as arguments or via --from-file")

    # Deduplicate, preserve order, lowercase for DB matching
    seen = set()
    proteins_lo = []
    orig_map = {}
    for p in proteins:
        lo = p.lower()
        if lo not in seen:
            seen.add(lo)
            proteins_lo.append(lo)
            orig_map[lo] = p

    fc_db = args.db
    parquet_path = args.parquet

    if not os.path.exists(fc_db):
        print(f"ERROR: foldcomp DB not found at {fc_db}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(parquet_path):
        print(f"ERROR: parquet not found at {parquet_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking {len(proteins_lo)} proteins...")
    print(f"Foldcomp DB: {fc_db}")
    print(f"Parquet:     {parquet_path}")
    print()

    # --- ONE parquet scan: all self-pairs for all requested proteins ---
    # Predicate pushdown: only reads matching rows from the ~69M-row file.
    # Construct db_id keys the same way yp.get_models() does.
    rows = (
        pl.scan_parquet(parquet_path)
        .filter(
            pl.col("af3_id1").is_in(proteins_lo) &
            (pl.col("af3_id1") == pl.col("af3_id2"))
        )
        .with_columns(
            (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8)
             + "_" + pl.col("chain_id1")).alias("db_id1"),
            (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8)
             + "_" + pl.col("chain_id2")).alias("db_id2"),
        )
        .select("af3_id1", "input_type", "sample", "ranking_score",
                "chain_pair_iptm", "db_id1", "db_id2")
        .collect()
    )

    if rows.height == 0:
        print("No self-pair rows found in parquet for any of the requested proteins.")
        sys.exit(0)

    # --- foldcomp probe via the .lookup index file (instant, no decompression) ---
    # foldcomp.open() decompresses every entry it returns, which is slow and
    # can hit "Error decompressing" on corrupted entries. The .lookup file is
    # a plain-text index of every key in the DB — grep it once, build a set,
    # and check membership. Same ground truth, orders of magnitude faster.
    all_db_ids = sorted(set(rows["db_id1"].to_list() + rows["db_id2"].to_list()))
    lookup_path = fc_db + ".lookup"
    print(f"Reading DB index: {lookup_path}")
    db_keys = set()
    with open(lookup_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                db_keys.add(parts[1])
    print(f"DB index has {len(db_keys):,} keys. Checking {len(all_db_ids)} requested keys...")
    found = set(k for k in all_db_ids if k in db_keys)
    print(f"Found {len(found)} of {len(all_db_ids)} keys in DB index.")
    print()
    found_list = list(found)

    # --- Per-protein aggregation ---
    # Add a "both_present" flag per row, then group by protein + input_type
    rows_scored = rows.with_columns(
        pl.when(
            pl.col("db_id1").is_in(found_list) &
            pl.col("db_id2").is_in(found_list)
        ).then(True).otherwise(False).alias("both_present")
    )

    # Per protein x input_type: count ok/total, best ranking_score among ok
    per_protein = (
        rows_scored
        .group_by(["af3_id1", "input_type"])
        .agg(
            pl.len().alias("n_samples"),
            pl.col("both_present").sum().alias("n_ok"),
            pl.col("ranking_score").filter(pl.col("both_present")).max().alias("best_ok_score"),
            pl.col("chain_pair_iptm").filter(pl.col("both_present")).max().alias("best_ok_iptm"),
        )
        .sort(["af3_id1", "input_type"])
    )

    # Pivot into one row per protein with pair and pool columns
    pair_df = per_protein.filter(pl.col("input_type") == "pair").rename({
        "n_samples": "pair_total", "n_ok": "pair_ok",
        "best_ok_score": "pair_best_score", "best_ok_iptm": "pair_best_iptm"
    }).drop("input_type")
    pool_df = per_protein.filter(pl.col("input_type") == "pool").rename({
        "n_samples": "pool_total", "n_ok": "pool_ok",
        "best_ok_score": "pool_best_score", "best_ok_iptm": "pool_best_iptm"
    }).drop("input_type")

    summary = pair_df.join(pool_df, on="af3_id1", how="full", suffix="_pool")

    # Fill nulls for proteins that have no pair or no pool rows
    summary = summary.with_columns([
        pl.col("pair_total").fill_null(0),
        pl.col("pair_ok").fill_null(0),
        pl.col("pool_total").fill_null(0),
        pl.col("pool_ok").fill_null(0),
    ])

    # Classify each protein
    summary = summary.with_columns(
        pl.when(pl.col("pair_ok") > 0).then(pl.lit("PAIR_AVAILABLE"))
        .when(pl.col("pool_ok") > 0).then(pl.lit("POOL_FALLBACK"))
        .otherwise(pl.lit("MISSING"))
        .alias("verdict")
    )

    # Add original-case protein name
    summary = summary.with_columns(
        pl.col("af3_id1").replace(orig_map).alias("protein")
    )

    # Reorder columns for readability
    summary = summary.select([
        "protein", "verdict",
        "pair_ok", "pair_total", "pair_best_score", "pair_best_iptm",
        "pool_ok", "pool_total", "pool_best_score", "pool_best_iptm",
    ]).sort("protein")

    # --- Print table ---
    print(f"{'protein':<10} {'verdict':<16} {'pair_ok':<8} {'pool_ok':<8} "
          f"{'pair_best':<10} {'pool_best':<10}")
    print("-" * 70)
    for row in summary.iter_rows(named=True):
        pbs = f"{row['pair_best_score']:.3f}" if row['pair_best_score'] is not None else "-"
        pos = f"{row['pool_best_score']:.3f}" if row['pool_best_score'] is not None else "-"
        print(f"{row['protein']:<10} {row['verdict']:<16} "
              f"{row['pair_ok']}/{row['pair_total']:<7} "
              f"{row['pool_ok']}/{row['pool_total']:<7} "
              f"{pbs:<10} {pos:<10}")

    # --- Aggregate counts ---
    print()
    print("AGGREGATE:")
    counts = summary.group_by("verdict").len().sort("verdict")
    for row in counts.iter_rows(named=True):
        print(f"  {row['verdict']:<16} {row['len']} proteins")
    total = summary.height
    n_pair = summary.filter(pl.col("verdict") == "PAIR_AVAILABLE").height
    n_pool = summary.filter(pl.col("verdict") == "POOL_FALLBACK").height
    n_miss = summary.filter(pl.col("verdict") == "MISSING").height
    print()
    print(f"  Total:  {total} proteins")
    print(f"  Pair:   {n_pair} ({100*n_pair/total:.0f}%) — have pair structures in foldcomp")
    print(f"  Pool:   {n_pool} ({100*n_pool/total:.0f}%) — no pair, but pool available (fallback)")
    print(f"  Missing:{n_miss} ({100*n_miss/total:.0f}%) — no structures at all")

    # --- Also flag proteins in the request list that had NO parquet rows ---
    found_in_parquet = set(summary["protein"].to_list())
    not_in_parquet = [orig_map[lo] for lo in proteins_lo if orig_map[lo] not in found_in_parquet]
    if not_in_parquet:
        print()
        print(f"  WARNING: {len(not_in_parquet)} proteins had no self-pair rows in parquet:")
        for p in not_in_parquet:
            print(f"    {p}")

    # --- Save CSV if requested ---
    if args.out:
        summary.write_csv(args.out)
        print(f"\nPer-protein table written: {args.out}")


if __name__ == "__main__":
    main()
