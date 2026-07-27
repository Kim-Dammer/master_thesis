#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_pae_ptm
#SBATCH --output=logs/lookup_pae_ptm_%j.out
#SBATCH --error=logs/lookup_pae_ptm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G

# ===========================================================================
# 08_02_lookup_PAE_ptm.sh
#
# Map two nested-list model columns to protein pairs:
#     chain_pair_pae_min  - List(List(Float64))
#     chain_ptm           - List(Float64)
#
# Source DB:  data-26.07/summary_models.parquet
# Pair keys in DB: af3_id1, af3_id2   (homodimer <=> af3_id1 == af3_id2)
#
# Pair set = all pairs in yeast_protein_pairs.parquet
#            UNION all homodimers present in the DB (af3_id1 == af3_id2).
#
# Aggregation: NONE. Join is order-independent (canonical p1<=p2) and keeps
# ALL matching DB rows as-is, so a pair with N predictions -> N output rows.
#
# Usage:
#    sbatch 08_02_lookup_PAE_ptm.sh
# ===========================================================================

set -euo pipefail

# --- Config ----------------------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# ---------------------------------------------------------------------------

source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting PAE/ptm lookup (chain_pair_pae_min, chain_ptm)..."

python3 - <<'PYEOF'
import os
import gc
import sys
from pathlib import Path
import psutil
import polars as pl

# --- Paths -----------------------------------------------------------------
PAIRS_PATH   = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/iPTM/yeast_protein_pairs.parquet")
MODELS_PATH  = Path("/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07/summary_models.parquet")
OUTPUT_DIR   = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/PAE")
OUTPUT_PATH  = OUTPUT_DIR / "yeast_pairs_pae.parquet"

# DB pair-key columns (homodimer <=> these are equal)
DB_ID1 = "af3_id1"
DB_ID2 = "af3_id2"

# Nested-list value columns to fetch
VALUE_COLUMNS = [
    "input_type",  # String
    "input_name",  # String
    "seed",  # Int64
    "sample",  # Int64
    "chain_pair_pae_min_min",  # Float64
    "chain_pair_pae_min_max",  # Float64
    "chain_pair_pae_min_mean"   # Float64
]

def mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

# --- Inspect DB schema (metadata only, no data load) -----------------------
print(f"Scanning DB schema: {MODELS_PATH}")
db_lazy = pl.scan_parquet(MODELS_PATH)
db_schema = db_lazy.collect_schema()
db_cols = db_schema.names()
print("  DB columns:")
for c in db_cols:
    print(f"    {c} - {db_schema[c]}")

# Fail loudly if the expected columns are not present
required = [DB_ID1, DB_ID2] + VALUE_COLUMNS
missing = [c for c in required if c not in db_cols]
if missing:
    sys.exit(f"ERROR: expected columns missing from {MODELS_PATH.name}: {missing}\n"
             f"Available columns: {db_cols}")

# --- Build the DB lookup (canonical order, keep ALL rows) ------------------
# Select only what we need, normalize pair order to p1 <= p2 so the join is
# order-independent. No aggregation -> nested lists preserved verbatim.
print("Building canonical-order lookup from DB (no aggregation)...")
lookup_lazy = (
    db_lazy
    .select([DB_ID1, DB_ID2] + VALUE_COLUMNS)
    .with_columns([
        pl.col(DB_ID1).str.to_lowercase(),
        pl.col(DB_ID2).str.to_lowercase(),
    ])
    .with_columns([
        pl.min_horizontal(DB_ID1, DB_ID2).alias("p1"),
        pl.max_horizontal(DB_ID1, DB_ID2).alias("p2"),
    ])
    .select(["p1", "p2"] + VALUE_COLUMNS)
)

lookup_df = lookup_lazy.collect(engine="streaming")
print(f"  Lookup rows (all predictions): {lookup_df.height}")
print(f"  Memory after lookup collect: {mem_gb():.1f} GB")

# Homodimers present in the DB (p1 == p2 after canonicalization)
db_homodimers = (
    lookup_df.select(["p1", "p2"])
    .filter(pl.col("p1") == pl.col("p2"))
    .unique()
)
print(f"  Distinct DB homodimers (af3_id1 == af3_id2): {db_homodimers.height}")

# --- Load target pairs -----------------------------------------------------
print(f"Reading target pairs: {PAIRS_PATH}")
pairs_df = pl.read_parquet(PAIRS_PATH)
print(f"  {pairs_df.height} rows, columns: {pairs_df.columns}")

# Normalize target pairs to canonical order
pairs_norm = pairs_df.with_columns([
    pl.col("protein_A").str.to_lowercase(),
    pl.col("protein_B").str.to_lowercase(),
]).with_columns([
    pl.min_horizontal("protein_A", "protein_B").alias("p1"),
    pl.max_horizontal("protein_A", "protein_B").alias("p2"),
]).select(["p1", "p2"])

# --- Union: yeast_protein_pairs  +  ALL DB homodimers ----------------------
# Represent every entry as a (p1, p2) canonical pair, then de-duplicate the
# KEY set (we still keep all prediction rows via the join below).
all_keys = pl.concat([
    pairs_norm.select(["p1", "p2"]),
    db_homodimers.select(["p1", "p2"]),
]).unique()

n_pairs_only   = pairs_norm.select(["p1", "p2"]).unique().height
n_after_union  = all_keys.height
print(f"  Unique keys from pairs list:      {n_pairs_only}")
print(f"  Unique keys after +DB homodimers: {n_after_union}")
print(f"  Homodimers added:                 {n_after_union - n_pairs_only}")

del pairs_df, pairs_norm, db_homodimers
gc.collect()

# --- Coverage diagnostic ---------------------------------------------------
db_key_set = set(zip(
    lookup_df["p1"].to_list(),
    lookup_df["p2"].to_list(),
))
tgt_key_set = set(zip(
    all_keys["p1"].to_list(),
    all_keys["p2"].to_list(),
))
overlap = tgt_key_set & db_key_set
print(f"  Target keys: {len(tgt_key_set)} | in DB: {len(overlap)} | "
      f"missing: {len(tgt_key_set) - len(overlap)}")
del db_key_set, tgt_key_set, overlap
gc.collect()

# --- Join: keep ALL matching DB rows per key -------------------------------
# left join keys -> lookup. Keys with multiple predictions expand to multiple
# rows; keys absent from the DB yield one row with null list columns.
print("Joining (keep all matching rows)...")
result = all_keys.join(lookup_df, on=["p1", "p2"], how="left")


# Rename canonical keys back to protein_A / protein_B for output clarity
result = result.rename({"p1": "protein_A", "p2": "protein_B"})

# Label each pair as homomer vs heteromer
result = result.with_columns(
    pl.when(pl.col("protein_A") == pl.col("protein_B"))
    .then(pl.lit("homomer"))
    .otherwise(pl.lit("heteromer"))
    .alias("complex_type")
)

print(f"  Result rows: {result.height}")
print(f"  Memory before write: {mem_gb():.1f} GB")

del lookup_df, all_keys
gc.collect()

# --- Write -----------------------------------------------------------------
result.write_parquet(OUTPUT_PATH)

# Coverage report
for c in VALUE_COLUMNS:
    n_valid = result.filter(pl.col(c).is_not_null()).height
    print(f"  {c}: {n_valid}/{result.height} non-null "
          f"({100*n_valid/result.height:.1f}%)")

print(f"\nDone! Output: {OUTPUT_PATH}")
print(f"  Shape: {result.shape}")
print(f"  Columns: {result.columns}")
print(f"  Peak memory: {mem_gb():.1f} GB")
PYEOF

echo "[$(date)] Script run complete."
