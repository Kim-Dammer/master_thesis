#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=cf_x3_x4
#SBATCH --output=logs/cf_x3_x4_%j.out
#SBATCH --error=logs/cf_x3_x4_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

# CF_RESULTS_CSV and COMBFOLD_OUTPUT_DIR are injected by X_run_cf_eval_pipeline.sh
# via `sbatch --export=...`. This job is not meant to be submitted standalone --
# use X_run_cf_eval_pipeline.sh, which runs X02 on the login node first and then
# calls sbatch on this file.
: "${CF_RESULTS_CSV:?Not set -- submit via X_run_cf_eval_pipeline.sh, not sbatch directly}"
: "${COMBFOLD_OUTPUT_DIR:?Not set -- submit via X_run_cf_eval_pipeline.sh, not sbatch directly}"

CF_RESULTS_DIR="$(dirname "${CF_RESULTS_CSV}")"
X02_OUT="${CF_RESULTS_DIR}/cf_pdb_structure_similarity/aggregate_cf_for_pdb_eval.parquet"
X03_OUT="${CF_RESULTS_DIR}/cf_pdb_structure_similarity/aggregate_cf_for_pdb_eval_with_combfold_results_paths.parquet"

echo "=== X03: find combfold outputs ==="
python X03_find_combfold_outputs.py \
    --step-x2-output-path "${X02_OUT}" \
    --combfold-output-dir "${COMBFOLD_OUTPUT_DIR}"
    # --keep-unmatched left off: default keeps only found_match_reference_pdb_model==True

echo "=== X04: compute structure similarity metrics ==="
python X04_compute_structure_similarity_metrics.py \
    --step-x3-output-path "${X03_OUT}" \
    --max-workers 16 \
    --ter 1 \
    --skip-existing \
    # -v
    # --reference-pdb-dir, --sifts-csv, --stoic-csv, --usalign-bin, --mm, --cutoff,
    # --min-contacts all left at script defaults

echo "=== done ==="
