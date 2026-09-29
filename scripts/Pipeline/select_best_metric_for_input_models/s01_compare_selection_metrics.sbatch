#!/bin/bash
#SBATCH --account es_biol
#SBATCH --partition es_biol
#SBATCH --job-name=cmp_metrics
#SBATCH --output=logs/cmp_metrics_%j.out
#SBATCH --error=logs/cmp_metrics_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=10G

# Wraps 01_compare_selection_metrics.py -- login node was struggling with the summary_models.parquet / summary_confidences.parquet scan sizes.
# Usage: sbatch s01_compare_selection_metrics.sbatch

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting metric comparison..."
uv run 01_compare_selection_metrics.py
echo "[$(date)] Done."
