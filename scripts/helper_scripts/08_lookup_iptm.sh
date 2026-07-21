#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_iptm
#SBATCH --output=logs/lookup_iptm_%j.out
#SBATCH --error=logs/lookup_iptm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G

# ===========================================================================
# lookup_iptm.sh — Map all 4 ipTM scores to protein pairs with batching.
#
# Reads the yeast protein pairs parquet, joins against the pooled-PPI yeast
# DB (data-26.04) for all 4 iptm columns (order-independent, takes MAX per
# pair per column), and writes each batch to a separate part file to keep
# memory flat. Parts are concatenated at the end into the final output.
#
# The 4 iptm columns:
#     chain_pair_iptm_best
#     chain_pair_iptm_mean
#     chain_pair_iptm_best_corrected
#     chain_pair_iptm_mean_corrected
#
# Memory profile: only the lookup table + one batch are in memory at a time.
#
# Usage:
#    sbatch 08_lookup_iptm.sh
# ===========================================================================

set -euo pipefail

# --- Config ----------------------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# ---------------------------------------------------------------------------

source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting ipTM lookup (4 columns) with memory-efficient batching..."

python3 - <<'PYEOF'
import os
import gc
from pathlib import Path
import sys
import psutil
import polars as pl
import pooled_ppi

# --- Paths -----------------------------------------------------------------
PAIRS_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM/yeast_protein_pairs.parquet")
POOLED_PPI_DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.04"
OUTPUT_DIR = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM")
OUTPUT_PATH = OUTPUT_DIR / "yeast_pairs_iptm_4cols.parquet"
PARTS_PREFIX = "yeast_pairs_iptm_4cols_part_"

BATCH_SIZE = 100_000

# The 4 iptm columns to fetch from the pooled-PPI DB
IPTM_COLUMNS = [
    "chain_pair_iptm_best",
    "chain_pair_iptm_mean",
    "chain_pair_iptm_best_corrected",
    "chain_pair_iptm_mean_corrected",
]

def mem_gb():
    """Current RSS in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

# --- Load & normalize the pooled-PPI DB ------------------------------------
print(f"Loading pooled-PPI DB from {POOLED_PPI_DB} ...")
pp = pooled_ppi.PooledPredictionsDb(POOLED_PPI_DB)
print(f"  Memory after pandas load: {mem_gb():.1f} GB")

pp_df = pl.from_pandas(pp.pairs)
print(f"  DB shape: {pp_df.shape}")
print(f"  Memory after pandas->polars: {mem_gb():.1f} GB")

# Free the pandas copy immediately
del pp
gc.collect()
print(f"  Memory after freeing pandas: {mem_gb():.1f} GB")

# Verify all 4 iptm columns exist
missing_cols = [c for c in IPTM_COLUMNS if c not in pp_df.columns]
if missing_cols:
    print(f"  WARNING: Missing iptm columns in DB: {missing_cols}")
available_iptm_cols = [c for c in IPTM_COLUMNS if c in pp_df.columns]
if not available_iptm_cols:
    sys.exit("  ERROR: No iptm columns found in pooled-PPI DB!")
print(f"  Available iptm columns: {available_iptm_cols}")

# Normalize DB to canonical order (p1 <= p2) and take MAX per iptm column per pair
print("Normalizing DB and taking max per iptm column per pair ...")
select_cols = ["p1", "p2"] + available_iptm_cols
lookup_df = (
    pp_df
    .with_columns([
        pl.min_horizontal("uniprot_id1", "uniprot_id2").alias("p1"),
        pl.max_horizontal("uniprot_id1", "uniprot_id2").alias("p2"),
    ])
    .select(select_cols)
    .group_by(["p1", "p2"])
    .agg([pl.col(c).max().alias(c) for c in available_iptm_cols])
)
print(f"  Lookup table: {lookup_df.height} unique pairs with {len(available_iptm_cols)} iptm columns")

# Free the raw polars DB copy
del pp_df
gc.collect()
print(f"  Memory after building lookup_df: {mem_gb():.1f} GB")

# --- Load target pairs -----------------------------------------------------
print(f"Reading target pairs from {PAIRS_PATH} ...")
pairs_df = pl.read_parquet(PAIRS_PATH)
print(f"  {pairs_df.height} rows, columns: {pairs_df.columns}")
print(f"  Memory after loading pairs: {mem_gb():.1f} GB")

# Normalize target pairs to canonical order
pairs_norm = (
    pairs_df.with_columns([
        pl.min_horizontal("protein_A", "protein_B").alias("p1"),
        pl.max_horizontal("protein_A", "protein_B").alias("p2"),
    ])
    .with_row_index("row_id")
)

# --- Diagnostic: coverage check --------------------------------------------
target_unique = pairs_norm.select(["p1", "p2"]).unique()
db_unique = lookup_df.select(["p1", "p2"])

target_set = set(zip(target_unique["p1"].to_list(), target_unique["p2"].to_list()))
db_set = set(zip(db_unique["p1"].to_list(), db_unique["p2"].to_list()))
overlap = target_set & db_set
print(f"  Unique target pairs: {len(target_set)}")
print(f"  Unique DB pairs:     {len(db_set)}")
print(f"  Overlap (found):     {len(overlap)}")
print(f"  Missing from DB:     {len(target_set) - len(overlap)}")

del target_unique, db_unique, target_set, db_set, overlap
gc.collect()

# --- Clean up any leftover part files from a previous run ------------------
for p in sorted(OUTPUT_DIR.glob(f"{PARTS_PREFIX}*.parquet")):
    p.unlink()
    print(f"  Removed old part: {p.name}")

# --- Batch join — write each batch to its own file -------------------------
print(f"Processing in batches of {BATCH_SIZE} ...")

n_total = pairs_norm.height
n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
part_paths = []

# Output columns: original pair columns + 4 iptm columns
output_cols = ["protein_A", "protein_B"] + available_iptm_cols

for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, n_total)
    batch = pairs_norm.slice(start, end - start)

    # Join against lookup table
    batch_result = (
        batch.join(lookup_df, on=["p1", "p2"], how="left")
        .select(output_cols)
    )

    # Write this batch to its own file
    part_path = OUTPUT_DIR / f"{PARTS_PREFIX}{batch_idx:04d}.parquet"
    batch_result.write_parquet(part_path)
    part_paths.append(part_path)

    found_in_batch = batch_result.filter(pl.col(available_iptm_cols[0]).is_not_null()).height
    print(f"  Batch {batch_idx + 1}/{n_batches} (rows {start}-{end}): "
          f"{found_in_batch}/{batch.height} pairs found, "
          f"memory: {mem_gb():.1f} GB")

    del batch, batch_result
    gc.collect()

# Free the input pairs
del pairs_norm, pairs_df
gc.collect()

# --- Assemble final output from parts --------------------------------------
print("Assembling final output from parts ...")
final_df = pl.concat([pl.read_parquet(p) for p in part_paths])

# Add null columns for any iptm cols missing from the DB
for c in missing_cols:
    final_df = final_df.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(c)
    )

final_df.write_parquet(OUTPUT_PATH)

# Report coverage
for c in IPTM_COLUMNS:
    if c in final_df.columns:
        n_valid = final_df.filter(pl.col(c).is_not_null()).height
        print(f"  {c}: {n_valid}/{final_df.height} non-null ({100*n_valid/final_df.height:.1f}%)")

print(f"\nDone! Output: {OUTPUT_PATH}")
print(f"  Shape: {final_df.shape}")
print(f"  Peak memory: {mem_gb():.1f} GB")

# Clean up part files
for p in part_paths:
    p.unlink()
print("Part files removed.")
PYEOF

echo "[$(date)] Script run complete."
