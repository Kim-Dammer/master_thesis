#!/usr/bin/env python
"""
Audit Bug #3: Biopython chain-ID collision risk in save_pair_pdb()
(s2_run_CombFold.sbatch).

save_pair_pdb() keeps the FIRST-loaded FoldComp structure's chain ID exactly
as FoldComp stored it natively, and only renames the SECOND-loaded
structure's chain to a freshly generated id (from af3io.input.enumerate_chains(),
which starts at 'A'). It never inspects or renames the first structure's
native chain ID. If FoldComp's native ID for the entry that lands first
happens to equal the generated id assigned to the entry that lands second,
Bio.PDB's struct0[0].add(chain0) raises an uncaught PDBConstructionException
and crashes the whole s2_run_CombFold.sbatch job.

This script empirically checks the actual distribution of native FoldComp
chain IDs to assess how live that risk is, WITHOUT touching save_pair_pdb()
itself.

Two steps:

  Step A (DuckDB): summary_models.parquet is a single columnar file with
    millions of rows but only ~14 columns. DuckDB's projection pushdown lets
    us pull just (input_name, sample, chain_id1, chain_id2) without loading
    the rest of the table, then reduce row-level duplication (many pairwise
    rows share the same underlying monomer/chain entry) down to the actual
    distinct set of FoldComp keys used anywhere in the pipeline. A random
    sample (fixed seed, reservoir sampling) of that distinct-key universe is
    then drawn -- DuckDB cannot see inside the FoldComp binary DB at all, so
    this step only produces the list of keys to check in Step B.

  Step B (foldcomp + Bio.PDB): for each sampled key, open the real FoldComp
    structure DB (batched, to bound memory), parse the returned PDB text with
    Bio.PDB (same parser save_pair_pdb() uses), and record the native chain
    ID of the resulting structure's first (only) chain. Checkpointed to CSV
    so a long job can resume after a partial failure.

fc_key convention (must match s2_run_CombFold.sbatch exactly):
    fc_key = f"{input_name}_{sample}_{chain_id}"
(no input_type component -- confirmed from the pipeline's own key
construction for both homodimer and heteropair candidate blocks).

Usage:
    python audit_chain_collision.py \
        --pooled-ppi-db /cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07 \
        --out-dir . \
        --sample-size 50000 \
        --seed 42

Outputs (written to --out-dir):
    chain_collision_sample_keys.csv     Step A output (the sampled keys)
    chain_collision_audit_results.csv   Step B output (checkpointed, resumable)
"""
import argparse
import csv
import io
import os
import sys
import time
from collections import Counter
from pathlib import Path

import Bio.PDB


def step_a_sample_keys(pooled_ppi_db: str, out_csv: Path, sample_size: int, seed: int) -> int:
    """Query summary_models.parquet via DuckDB, reduce to the distinct
    (input_name, sample, chain_id) universe, sample it, and write the sample
    to out_csv. Returns the size of the full distinct-key universe (before
    sampling) so it can be reported to the user.
    """
    import duckdb

    parquet_path = os.path.join(pooled_ppi_db, "summary_models.parquet")
    if not os.path.exists(parquet_path):
        print(f"ERROR: parquet file not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    con = duckdb.connect()

    con.execute(f"""
        CREATE TEMP TABLE keys AS
        SELECT input_name, "sample", chain_id1 AS chain_id
        FROM read_parquet('{parquet_path}')
        UNION
        SELECT input_name, "sample", chain_id2 AS chain_id
        FROM read_parquet('{parquet_path}')
    """)

    n_distinct = con.execute("SELECT COUNT(*) FROM keys").fetchone()[0]
    print(f"[Step A] Distinct (input_name, sample, chain_id) keys in "
          f"summary_models.parquet: {n_distinct}")

    if n_distinct <= sample_size:
        print(f"[Step A] Distinct-key universe ({n_distinct}) <= requested "
              f"sample size ({sample_size}) -- using all of it (exhaustive).")
        sampled = con.execute("SELECT input_name, \"sample\", chain_id FROM keys").fetchall()
    else:
        sampled = con.execute(
            f"SELECT input_name, \"sample\", chain_id FROM keys "
            f"USING SAMPLE {sample_size} (reservoir, {seed})"
        ).fetchall()

    print(f"[Step A] Sampled {len(sampled)} keys (seed={seed}).")

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["input_name", "sample", "chain_id", "fc_key"])
        for input_name, sample, chain_id in sampled:
            fc_key = f"{input_name}_{sample}_{chain_id}"
            writer.writerow([input_name, sample, chain_id, fc_key])
    print(f"[Step A] Wrote sample keys: {out_csv}")

    con.close()
    return n_distinct


def _load_sample_keys(sample_csv: Path):
    rows = []
    with open(sample_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _load_already_checked(results_csv: Path):
    checked = set()
    if results_csv.exists():
        with open(results_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                checked.add(row["fc_key"])
    return checked


def step_b_check_native_chain_ids(fc_db: str, sample_csv: Path, results_csv: Path,
                                   batch_size: int) -> None:
    """For each sampled fc_key, open the real FoldComp DB and record the
    native chain ID Bio.PDB parses out, exactly mirroring save_pair_pdb()'s
    own parsing path. Checkpoints after every batch so the job can resume.
    """
    import foldcomp

    rows = _load_sample_keys(sample_csv)
    by_key = {r["fc_key"]: r for r in rows}
    all_keys = list(by_key.keys())

    already_checked = _load_already_checked(results_csv)
    remaining = [k for k in all_keys if k not in already_checked]
    print(f"[Step B] {len(all_keys)} total sampled keys, "
          f"{len(already_checked)} already checked (resume), "
          f"{len(remaining)} remaining.")

    write_header = not results_csv.exists()
    parser = Bio.PDB.PDBParser(QUIET=True)

    with open(results_csv, "a", newline="") as out_fh:
        writer = csv.writer(out_fh)
        if write_header:
            writer.writerow([
                "fc_key", "input_name", "sample", "chain_id",
                "native_chain_id", "found", "returned_name_matches_requested",
            ])

        n_batches = (len(remaining) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            batch = remaining[bi * batch_size:(bi + 1) * batch_size]
            t0 = time.time()
            seen = set()
            order_checked = 0
            order_matched = 0
            try:
                with foldcomp.open(fc_db, ids=batch) as db:
                    for i, (name, pdb) in enumerate(db):
                        seen.add(name)
                        if i < len(batch):
                            order_checked += 1
                            if name == batch[i]:
                                order_matched += 1
                        meta = by_key.get(name)
                        if meta is None:
                            # Shouldn't happen (returned a key we didn't ask for)
                            continue
                        try:
                            struct = parser.get_structure(name, io.StringIO(pdb))
                            chain = next(struct[0].get_chains())
                            native_chain_id = chain.id
                        except Exception as exc:  # pragma: no cover - defensive
                            print(f"  WARNING: failed to parse structure for {name}: {exc}")
                            native_chain_id = None
                        writer.writerow([
                            name, meta["input_name"], meta["sample"], meta["chain_id"],
                            native_chain_id, True,
                            (i < len(batch) and name == batch[i]),
                        ])
            except Exception as exc:
                print(f"  ERROR: foldcomp.open failed for batch {bi + 1}/{n_batches}: {exc}",
                      file=sys.stderr)
                out_fh.flush()
                raise

            missing = [k for k in batch if k not in seen]
            for k in missing:
                meta = by_key[k]
                writer.writerow([k, meta["input_name"], meta["sample"], meta["chain_id"],
                                  None, False, None])

            out_fh.flush()
            dt = time.time() - t0
            order_note = (f"{order_matched}/{order_checked} order-matched"
                          if order_checked else "n/a")
            print(f"[Step B] batch {bi + 1}/{n_batches}: requested={len(batch)} "
                  f"found={len(seen)} missing={len(missing)} ({order_note}) "
                  f"[{dt:.1f}s]")


def summarize(results_csv: Path, n_distinct_universe: int) -> None:
    rows = []
    with open(results_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    n_total = len(rows)
    n_found = sum(1 for r in rows if r["found"] == "True")
    n_missing = n_total - n_found

    chain_id_counts = Counter(r["native_chain_id"] for r in rows if r["found"] == "True")

    order_checked = [r for r in rows if r["returned_name_matches_requested"] not in ("", None)]
    order_matched = sum(1 for r in order_checked
                        if r["returned_name_matches_requested"] == "True")

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY: Bug #3 chain-ID collision risk in save_pair_pdb()")
    print("=" * 70)
    print(f"Full distinct-key universe (Step A, before sampling): {n_distinct_universe}")
    print(f"Keys audited in Step B (sampled): {n_total}")
    print(f"  found in FoldComp DB: {n_found}")
    print(f"  missing from FoldComp DB: {n_missing}")
    print()
    print("Native chain ID distribution (found keys only):")
    for chain_id, count in chain_id_counts.most_common():
        pct = 100.0 * count / max(n_found, 1)
        print(f"  {chain_id!r}: {count} ({pct:.3f}%)")
    print()

    non_a = [r for r in rows if r["found"] == "True" and r["native_chain_id"] != "A"]
    print(f"Keys with native chain ID != 'A' (live collision candidates): {len(non_a)}")
    if non_a:
        print("  (input_name, sample, chain_id, fc_key, native_chain_id):")
        for r in non_a[:200]:
            print(f"    {r['input_name']}, {r['sample']}, {r['chain_id']}, "
                  f"{r['fc_key']}, {r['native_chain_id']}")
        if len(non_a) > 200:
            print(f"    ... and {len(non_a) - 200} more (see {results_csv} for full list)")
    else:
        print("  None found in this sample -- consistent with FoldComp always assigning "
              "native chain ID 'A' to single-chain monomer structures (would need the "
              "exhaustive/full-universe check to fully rule out rare exceptions).")

    if order_checked:
        pct = 100.0 * order_matched / len(order_checked)
        print(f"\nOrder-preservation diagnostic: {order_matched}/{len(order_checked)} "
              f"({pct:.1f}%) of returned entries matched the requested id at that "
              "position within their batch.")

    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled-ppi-db",
                    default="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07",
                    help="Path to the pooled PPI data directory containing "
                         "summary_models.parquet and predictions-db/.")
    ap.add_argument("--out-dir", default=".", help="Directory to write output CSVs to.")
    ap.add_argument("--sample-size", type=int, default=50000,
                    help="Number of distinct (input_name, sample, chain_id) keys to sample "
                         "for the Step B ground-truth check. If the full distinct-key "
                         "universe is smaller than this, all of it is used (exhaustive).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reservoir sampling.")
    ap.add_argument("--batch-size", type=int, default=2000,
                    help="Number of FoldComp ids to request per foldcomp.open() call in Step B.")
    ap.add_argument("--skip-step-a", action="store_true",
                    help="Skip Step A and reuse an existing chain_collision_sample_keys.csv "
                         "in --out-dir (e.g. to resume/retry Step B only).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "chain_collision_sample_keys.csv"
    results_csv = out_dir / "chain_collision_audit_results.csv"
    fc_db = os.path.join(args.pooled_ppi_db, "predictions-db", "predictions-db")

    if args.skip_step_a:
        if not sample_csv.exists():
            print(f"ERROR: --skip-step-a given but {sample_csv} does not exist.",
                  file=sys.stderr)
            sys.exit(1)
        n_distinct_universe = len(_load_sample_keys(sample_csv))
        print(f"[Step A] Skipped (reusing {sample_csv}).")
    else:
        n_distinct_universe = step_a_sample_keys(
            args.pooled_ppi_db, sample_csv, args.sample_size, args.seed)

    step_b_check_native_chain_ids(fc_db, sample_csv, results_csv, args.batch_size)

    summarize(results_csv, n_distinct_universe)


if __name__ == "__main__":
    main()
