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
#   $1  input_csv    — path to the pipeline's cleaned/expanded complexes CSV
#   $2  out_dir      — pipeline output directory
#   $3  source       — 'pair' or 'pool' (optional, default 'pool'). MUST match
#                      the --combfold-source the CombFold jobs were actually
#                      submitted with, since s2 writes results into
#                      '<complex_name>_<source>_output/'. A mismatch here does
#                      NOT error - it silently marks every row
#                      combfold_successfully=False with empty confidence.
#   $4  orchestrator — (optional) absolute path to 01_combfold_with_stoic.py.
#                      01_combfold_with_stoic.py's stage_submit_analyze_dependency
#                      always passes this explicitly (its own resolved
#                      __file__ path, computed once outside SLURM). If
#                      omitted (e.g. manual/interactive invocation), falls
#                      back to self-discovery via dirname "${BASH_SOURCE[0]}"
#                      -- NOTE: that fallback only works when this script is
#                      run directly with bash, NOT via sbatch. Under sbatch,
#                      SLURM copies the submitted script into a spool
#                      directory before executing it, so BASH_SOURCE[0]
#                      resolves to that spool copy, not this script's real
#                      location -- self-discovery would then fail with
#                      "No such file or directory" for 01_combfold_with_stoic.py.
#                      Always pass $4 explicitly when submitting via sbatch.
#
# Submitted from: 01_combfold_with_stoic.py
# ===========================================================================

set -euo pipefail

INPUT_CSV="${1:?missing input_csv}"
OUT_DIR="${2:?missing out_dir}"
SOURCE="${3:-pool}"

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCHESTRATOR="${4:-${SCRIPT_DIR}/01_combfold_with_stoic.py}"

if [[ ! -f "${ORCHESTRATOR}" ]]; then
    echo "ERROR: orchestrator script not found: ${ORCHESTRATOR}"
    echo "  (this script was likely run under sbatch without the 4th"
    echo "   'orchestrator path' argument -- BASH_SOURCE-based self-discovery"
    echo "   does not work under sbatch, since SLURM executes a spooled copy"
    echo "   of this script, not the original file. Pass the orchestrator's"
    echo "   absolute path explicitly as \$4.)"
    exit 1
fi

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
