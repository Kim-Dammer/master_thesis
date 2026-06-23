#!/bin/bash
#SBATCH --job-name=ss_analyze
#SBATCH --output=logs/ss_analyze_%j.out
#SBATCH --error=logs/ss_analyze_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

# ===========================================================================
# 11_analyze.sbatch — Dependency wrapper for the second-setup analyze step.
#
# Submitted by 11_second_setup_pipeline.py with --dependency=afterany on all
# CombFold jobs. Runs the analyze stage which parses CombFold outputs and
# writes the final enriched CSV.
#
# Args:
#   $1  input_csv  — path to all_pdb_present_first_setup_pipeline_complexes.csv
#   $2  out_dir    — second_setup output directory
#
# Submitted from: 11_second_setup_pipeline.py
# ===========================================================================

set -euo pipefail

INPUT_CSV="${1:?missing input_csv}"
OUT_DIR="${2:?missing out_dir}"

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCHESTRATOR="${SCRIPT_DIR}/11_second_setup_pipeline.py"

mkdir -p "${OUT_DIR}/logs"

source "${VENV}/bin/activate"

echo "[$(date)] Running analyze stage..."
echo "  input_csv: ${INPUT_CSV}"
echo "  out_dir:   ${OUT_DIR}"

python "${ORCHESTRATOR}" \
    --input-csv "${INPUT_CSV}" \
    --out-dir   "${OUT_DIR}" \
    --mode      analyze

echo "[$(date)] Analyze complete."
