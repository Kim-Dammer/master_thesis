#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=pair_heteromer_batches
#SBATCH --output=logs/pair_heteromer_batches_%j.out
#SBATCH --error=logs/pair_heteromer_batches_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=50G

# For all heterodimers (af3_id1 != af3_id2) with input_type='pair',
# show which batch_id they came from. if all the same then from probabaly from paola
#
# Output: pair_heteromer_batches.csv (batch_id, n_models, n_unique_pairs)

set -euo pipefail

DATA_DIR="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
MODELS_PARQUET="${DATA_DIR}/summary_models.parquet"

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

mkdir -p logs

export MODELS_PARQUET

python3 - << 'PYEOF'
import os
import polars as pl

path = os.environ["MODELS_PARQUET"]

# One lazy scan: project only 4 columns, filter heteromers + pair in one
# predicate (pushed down to Parquet row groups), group by batch_id.
result = (
    pl.scan_parquet(path)
      .select("af3_id1", "af3_id2", "input_type", "batch_id")
      .filter(
          (pl.col("af3_id1") != pl.col("af3_id2")) &
          (pl.col("input_type") == "pair")
      )
      .group_by("batch_id")
      .agg(
          pl.len().alias("n_models"),
          pl.struct("af3_id1", "af3_id2").n_unique().alias("n_unique_pairs"),
      )
      .sort("n_models", descending=True)
      .collect()
)

print(f"=== HETEROMER pair-type models by batch_id ===")
print(f"  total models: {result['n_models'].sum():,}")
print(f"  total unique pairs: {result['n_unique_pairs'].sum():,}")
print(f"  batches: {result.height}")
print()
print(result)

out = "pair_heteromer_batches.csv"
result.write_csv(out)
print(f"\nWrote -> {out}")
PYEOF

echo "[sbatch] done."
