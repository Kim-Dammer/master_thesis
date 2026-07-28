#!/bin/bash
#SBATCH --job-name=ss_analyze
#SBATCH --output=logs/ss_analyze_%j.out
#SBATCH --error=logs/ss_analyze_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

# ===========================================================================
# s3_analyze_CF_results.sbatch — Dependency wrapper for the analyze step.
#
# Submitted by 01_combfold_with_stoic.py (stage_submit_analyze_dependency)
# with --dependency=afterany on all CombFold jobs. Runs the analyze stage
# which parses CombFold outputs and writes the final enriched CSV.
#
# Args:
#   $1  input_csv  — path to the pipeline's cleaned/expanded complexes CSV
#   $2  out_dir    — pipeline output directory
#   $3  source     — 'pair' or 'pool' (optional, default 'pool'). MUST match
#                    the --combfold-source the CombFold jobs were actually
#                    submitted with, since s2 writes results into
#                    '<complex_name>_<source>_output/'. A mismatch here does
#                    NOT error - it silently marks every row
#                    combfold_successfully=False with empty confidence.
#
# Submitted from: 01_combfold_with_stoic.py
# ===========================================================================

set -euo pipefail

INPUT_CSV="${1:?missing input_csv}"
OUT_DIR="${2:?missing out_dir}"
SOURCE="${3:-pool}"

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCHESTRATOR="${SCRIPT_DIR}/01_combfold_with_stoic.py"

mkdir -p "${OUT_DIR}/logs"

source "${VENV}/bin/activate"

echo "[$(date)] Running analyze stage..."
echo "  input_csv: ${INPUT_CSV}"
echo "  out_dir:   ${OUT_DIR}"
echo "  source:    ${SOURCE}"

python "${ORCHESTRATOR}" \
    --input-csv       "${INPUT_CSV}" \
    --out-dir         "${OUT_DIR}" \
    --combfold-source "${SOURCE}" \
    --mode            analyze

echo "[$(date)] Analyze complete."
