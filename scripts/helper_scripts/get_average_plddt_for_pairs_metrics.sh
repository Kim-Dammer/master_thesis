#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=get_avg_plddt
#SBATCH --output=logs/get_avg_plddt_%j.out
#SBATCH --error=logs/get_avg_plddt_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20G

# get_average_plddt_for_pairs_metrics.sh  <pairs.csv>  <out.csv>  [extra python args...]
#
# Runs get_average_plddt_for_pairs_metrics.py on a compute node instead of the
# login node - this script scans the full summary_models.parquet plus does
# chunked lookups against the 124GB models.sqlite, which is too heavy to run
# interactively on a shared login node.
#
# Usage:
#   sbatch get_average_plddt_for_pairs_metrics.sh in.csv out.csv
#   sbatch get_average_plddt_for_pairs_metrics.sh in.csv out.csv --skip-verify
set -euo pipefail
PAIRS_CSV="${1:?need pairs csv as arg 1}"
OUT_CSV="${2:?need output csv as arg 2}"
shift 2
EXTRA_ARGS=("$@")

mkdir -p logs
mkdir -p "$(dirname "$OUT_CSV")"

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

SCRIPT_DIR="/cluster/project/beltrao/kdammer/master_thesis/scripts/helper_scripts"

python "${SCRIPT_DIR}/get_average_plddt_for_pairs_metrics.py" \
    --pairs-csv "$PAIRS_CSV" \
    --out-csv "$OUT_CSV" \
    "${EXTRA_ARGS[@]}"
