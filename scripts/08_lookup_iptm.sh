#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_iptm
#SBATCH --output=logs/lookup_iptm_%j.out
#SBATCH --error=logs/lookup_iptm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G

# ===========================================================================
# lookup_iptm.sbatch — Map best ipTM scores to protein pairs with batching.
#
# Reads a parquet of (protein_A, protein_B) pairs, joins against the
# pooled-PPI yeast DB (order-independent, takes MAX iptm per pair),
# and writes each batch to a separate part file to keep memory flat.
# Parts are concatenated at the end into the final output.
#
# Memory profile: only the lookup table + one batch are in memory at a time.
#
# Usage:
#    sbatch 08_lookup_iptm_pairs.sh
# ===========================================================================

set -euo pipefail

# --- Config ----------------------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# ---------------------------------------------------------------------------

source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting ipTM lookup with memory-efficient batching..."

python - <<'EOF'
import os
import gc
from pathlib import Path
import sys
import psutil
import polars as pl
import pooled_ppi

# --- Paths -----------------------------------------------------------------
PAIRS_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM/yeast_protein_pairs.parquet")
POOLED_PPI_DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.03"
OUTPUT_DIR = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM")
OUTPUT_PATH = OUTPUT_DIR / "yeast_pairs_iptm_mapped.parquet"
PARTS_PREFIX = "yeast_pairs_iptm_part_"

BATCH_SIZE = 100_000

def mem_gb():
    """Current RSS in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

# --- Load & normalize the pooled-PPI DB ------------------------------------
print(f"Loading pooled-PPI DB from {POOLED_PPI_DB} ...")
pp = pooled_ppi.PooledPredictionsDb(POOLED_PPI_DB)
print(f"  Memory after pandas load: {mem_gb():.1f} GB")

pp_df = pl.from_pandas(pp.pairs)
print(f"  DB shape: {pp_df.shape}")
print(f"  Memory after pandas→polars: {mem_gb():.1f} GB")

# Free the pandas copy immediately
del pp
gc.collect()
print(f"  Memory after freeing pandas: {mem_gb():.1f} GB")

# Discover ipTM column
iptm_col = None
for candidate in [
    "chain_pair_iptm_mean_corrected",
    "chain_pair_iptm",
    "iptm",
    "ipTM",
    "average_iptm",
]:
    if candidate in pp_df.columns:
        iptm_col = candidate
        break

if iptm_col is None:
    iptm_candidates = [c for c in pp_df.columns if "iptm" in c.lower()]
    if iptm_candidates:
        iptm_col = iptm_candidates[0]
    else:
        sys.exit("Cannot find an ipTM column in the pooled-PPI DB.")

print(f"  Using ipTM column: {iptm_col}")

# Normalize DB to canonical order (p1 <= p2) and take BEST (max) iptm per pair
print("Normalizing DB and taking max iptm per pair ...")
lookup_df = (
    pp_df.select(["uniprot_id1", "uniprot_id2", iptm_col])
    .with_columns([
        pl.min_horizontal("uniprot_id1", "uniprot_id2").alias("p1"),
        pl.max_horizontal("uniprot_id1", "uniprot_id2").alias("p2"),
    ])
    .select(["p1", "p2", iptm_col])
    .group_by(["p1", "p2"])
    .agg(pl.col(iptm_col).max().alias("best_iptm"))
)
print(f"  Lookup table: {lookup_df.height} unique pairs with best iptm")

# Free the raw polars DB copy — only keep the compact lookup table
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

for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, n_total)
    batch = pairs_norm.slice(start, end - start)

    # Join against lookup table
    batch_result = (
        batch.join(lookup_df, on=["p1", "p2"], how="left")
        .select(["protein_A", "protein_B", "best_iptm"])
    )

    # Write this batch to its own file — no accumulation in memory
    part_path = OUTPUT_DIR / f"{PARTS_PREFIX}{batch_idx:04d}.parquet"
    batch_result.write_parquet(part_path)
    part_paths.append(part_path)

    found_in_batch = batch_result.filter(pl.col("best_iptm").is_not_null()).height
    print(f"  Batch {batch_idx + 1}/{n_batches} (rows {start}-{end}): "
          f"{found_in_batch}/{batch.height} pairs found, "
          f"memory: {mem_gb():.1f} GB")

    del batch, batch_result
    gc.collect()

# Free the input pairs — no longer needed
del pairs_norm, pairs_df
gc.collect()

# --- Assemble final output from parts --------------------------------------
print("Assembling final output from parts ...")
final_df = pl.concat([pl.read_parquet(p) for p in part_paths])
final_df.write_parquet(OUTPUT_PATH)

n_found = final_df.filter(pl.col("best_iptm").is_not_null()).height
n_missing = final_df.filter(pl.col("best_iptm").is_null()).height
print(f"\nDone! {n_found}/{final_df.height} pairs have iptm scores, {n_missing} missing.")
print(f"Final output: {OUTPUT_PATH}")
print(f"  Shape: {final_df.shape}")
print(f"  Peak memory: {mem_gb():.1f} GB")

# Clean up part files
for p in part_paths:
    p.unlink()
print("Part files removed.")
EOF

echo "[$(date)] Script run complete."
