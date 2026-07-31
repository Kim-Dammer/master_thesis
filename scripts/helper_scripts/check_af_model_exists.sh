#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=check_af_model_exists
#SBATCH --output=logs/check_af_model_exists_%j.out
#SBATCH --error=logs/check_af_model_exists_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G

# check_af_model_exists.sh  <input_pairs.parquet>  <output_dir>
#
# Adds a boolean column `AF_model_exists` to a table of yeast protein pairs:
#   True  = the pair was predicted together in a POOL run in the pooled-ppi
#           structure DB (input_type == "pool")
#   False = not found in any pool run
# Matching is case-insensitive and order-independent (Protein_1/Protein_2 vs
# af3_id1/af3_id2). The 124 GB models.sqlite is never opened — this is a pure
# parquet-index lookup.
#
# Usage:
#   ./check_af_model_exists.sh  my_pairs.parquet  ./out
#   POOLED_PPI_DATA_DIR=/other/dir ./check_af_model_exists.sh  in.parquet  out/
set -euo pipefail
INPUT="${1:?need input parquet as arg 1}"
OUTDIR="${2:?need output dir as arg 2}"
mkdir -p "$OUTDIR"

python - "$INPUT" "$OUTDIR" <<'PY'
import sys, os, polars as pl
input_path, outdir = sys.argv[1], sys.argv[2]

# --- locate the structure-DB summary parquet -------------------------------
DATA_DIR = os.environ.get(
    "POOLED_PPI_DATA_DIR",
    "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07")
need = {"input_type", "af3_id1", "af3_id2"}

def has_cols(path):
    return os.path.exists(path) and need <= set(
        pl.scan_parquet(path).collect_schema().names())

db_file = f"{DATA_DIR}/summary_pairs.parquet"
if not has_cols(db_file):                       # fall back to per-model summary
    db_file = f"{DATA_DIR}/summary_models.parquet"
print(f"DB file: {db_file}")

# --- order-independent, case-insensitive pair key --------------------------
def canon(a, b):
    a, b = a.str.to_lowercase(), b.str.to_lowercase()
    return (pl.when(a <= b).then(pl.concat_str([a, b], separator="__"))
              .otherwise(pl.concat_str([b, a], separator="__")))

# diagnostic: confirm "pool" is actually present as an input_type value
types = (pl.scan_parquet(db_file).select("input_type").unique()
           .collect().get_column("input_type").to_list())
print(f"input_type values in DB: {types}")

# unique set of pool pairs (read 2 cols, keep pool rows, dedup)
db_keys = (pl.scan_parquet(db_file)
             .filter(pl.col("input_type") == "pool")
             .select(key=canon(pl.col("af3_id1"), pl.col("af3_id2")))
             .unique().collect().get_column("key"))
print(f"Unique pool pairs in DB: {db_keys.len():,}")

# --- annotate the input pairs ----------------------------------------------
pairs = pl.read_parquet(input_path)
out = pairs.with_columns(
    AF_model_exists=canon(pl.col("Protein_1"), pl.col("Protein_2")).is_in(db_keys.implode()))
n = int(out["AF_model_exists"].sum())
print(f"Input pairs: {out.height:,} | in pool DB: {n:,} | missing: {out.height - n:,}")

stem = os.path.splitext(os.path.basename(input_path))[0]
out_path = os.path.join(outdir, f"{stem}_AF_model_exists.parquet")
out.write_parquet(out_path)
print(f"Written: {out_path}")
PY
