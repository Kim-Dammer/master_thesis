#!/bin/bash
# Run this directly on the LOGIN node (not via sbatch):
# Runs X02 here (needs internet: RCSB downloads + API), then submits
# X_submit_cf_eval_STEPS_3_4.sbatch to a compute node for X03 + X04.

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

CF_RESULTS_CSV="/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/8_benchmark_part_two/all_pdb_present_8_benchmark_part_two_pool_pipeline_complexes_combfold_results.csv"
COMBFOLD_OUTPUT_DIR="/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/8_benchmark_part_two/CombFold"
echo "=== X02: aggregate combfold + pdb for eval (login node, n_workers=8) ==="
python X02_aggregate_combfold_pdb_for_eval.py \
    --cf_results_summary "${CF_RESULTS_CSV}" \
    --n_workers 8

# X02's output path is fixed (not configurable) to a cf_pdb_structure_similarity subdir of the CF_RESULTS_CSV's directory:
CF_RESULTS_DIR="$(dirname "${CF_RESULTS_CSV}")"
X02_OUT="${CF_RESULTS_DIR}/cf_pdb_structure_similarity/aggregate_cf_for_pdb_eval.parquet"

if [[ ! -f "${X02_OUT}" ]]; then
    echo "ERROR: expected X02 output not found at ${X02_OUT}" >&2
    exit 1
fi

echo "=== submitting X03/X04 compute job ==="
sbatch \
    --export=ALL,CF_RESULTS_CSV="${CF_RESULTS_CSV}",COMBFOLD_OUTPUT_DIR="${COMBFOLD_OUTPUT_DIR}" \
    X_submit_cf_eval_STEPS_3_4.sbatch