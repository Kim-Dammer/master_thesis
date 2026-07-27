#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=best_model
#SBATCH --output=logs/best_model_%j.out
#SBATCH --error=logs/best_model_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
 
 
# ===========================================================================
# 02_get_best_model.sh
# sbatch 02_get_best_model.sh /cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07/summary_models.parquet /cluster/project/beltrao/kdammer/master_thesis/data/AF_pooled_data/best_model_per_pool.parquet
# ===========================================================================


set -euo pipefail

# --- Config ----------------------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# ---------------------------------------------------------------------------

source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting best model lookup..."

python3 - "$1" "$2" <<'PYEOF'
import sys
import polars as pl

infile  = sys.argv[1] if len(sys.argv) > 1 else "input.parquet"
outfile = sys.argv[2] if len(sys.argv) > 2 else "output.parquet"

lf = pl.scan_parquet(infile)

result = (
    lf.with_columns([
          pl.min_horizontal("af3_id1", "af3_id2").alias("_lo"),
          pl.max_horizontal("af3_id1", "af3_id2").alias("_hi"),
      ])
      .drop(["af3_id1", "af3_id2"])
      .rename({"_lo": "af3_id1", "_hi": "af3_id2"})
      .sort("ranking_score", descending=True)
      .unique(subset=["af3_id1", "af3_id2", "input_name"], keep="first")
      .collect(engine="streaming")
)

result.write_parquet(outfile)
print(f"wrote {result.height} rows to {outfile}", file=sys.stderr)
PYEOF

echo "[$(date)] Script run complete."