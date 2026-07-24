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
# 02_get_best_model.sh
# sbatch 02_get_best_model.sh /cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07/summary_models.parquet
# ===========================================================================


set -euo pipefail

# --- Config ----------------------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# ---------------------------------------------------------------------------

source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting PAE/ptm lookup (chain_pair_pae_min, chain_ptm)..."

python3 - <<'PYEOF'

import sys
import polars as pl

infile  = sys.argv[1] if len(sys.argv) > 1 else "input.parquet"
outfile = sys.argv[2] if len(sys.argv) > 2 else "output.parquet"

lf = pl.scan_parquet(infile)   # use pl.scan_csv(infile) for CSV

result = (
    lf.filter(pl.col("input type") == "pool")
      .with_columns([
          pl.min_horizontal("af3_id1", "af3_id2").alias("_lo"),
          pl.max_horizontal("af3_id1", "af3_id2").alias("_hi"),
      ])
      .drop(["af3_id1", "af3_id2"])
      .rename({"_lo": "af3_id1", "_hi": "af3_id2"})
      .sort("ranking_score", descending=True)
      .unique(subset=["af3_id1", "af3_id2", "input_name"], keep="first")
      .collect(streaming=True)
)

result.write_parquet(outfile)   # or result.write_csv(outfile)
print(f"wrote {result.height} rows to {outfile}", file=sys.stderr)

PYEOF

echo "[$(date)] Script run complete."