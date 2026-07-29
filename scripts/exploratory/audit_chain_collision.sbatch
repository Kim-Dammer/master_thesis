#!/bin/bash
#SBATCH --account es_biol
#SBATCH --partition es_biol
#SBATCH --job-name=audit_chain
#SBATCH --output=logs/audit_chain_%j.out
#SBATCH --error=logs/audit_chain_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80G

# ============================================================
# Audit Bug #3: Biopython chain-ID collision risk in
# save_pair_pdb() (s2_run_CombFold.sbatch).
#
# Samples a large random subset of the distinct FoldComp
# (input_name, sample, chain_id) keys used anywhere in
# summary_models.parquet (via DuckDB, cheap column-projected
# scan) and, for each sampled key, opens the real FoldComp DB
# and records the native Bio.PDB chain ID -- exactly mirroring
# what save_pair_pdb() itself does -- to check how often a
# non-'A' native chain ID occurs (the collision precondition).
#
# Does NOT modify save_pair_pdb() or run any part of the real
# pipeline. Report-only.
#
# Usage:
#   sbatch audit_chain_collision.sbatch
#   sbatch audit_chain_collision.sbatch --sample-size 100000
# (any extra args are forwarded to audit_chain_collision.py)
# ============================================================

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
POOLED_PPI_DB="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"

# NOTE: SLURM copies the submitted script into a spool directory before
# running it, so "$0" / readlink -f "$0" inside an sbatch script resolves to
# that spool copy, NOT to where you actually placed this file. Use
# SLURM_SUBMIT_DIR (the directory 'sbatch' was invoked from) instead, which
# SLURM always sets correctly. Falls back to the current directory if run
# directly with bash (outside SLURM) for local testing.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${SCRIPT_DIR}/chain_collision_audit_out"

source "${VENV}/bin/activate"

mkdir -p "${SCRIPT_DIR}/logs" "${OUT_DIR}"

echo "[$(date)] Starting chain-ID collision audit."
echo "Pooled PPI DB: ${POOLED_PPI_DB}"
echo "Output dir: ${OUT_DIR}"

python "${SCRIPT_DIR}/audit_chain_collision.py" \
    --pooled-ppi-db "${POOLED_PPI_DB}" \
    --out-dir "${OUT_DIR}" \
    --sample-size 50000 \
    --seed 42 \
    --batch-size 2000 \
    "$@"

echo "[$(date)] Audit finished. Results in: ${OUT_DIR}"
