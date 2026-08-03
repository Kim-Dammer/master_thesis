#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=pair_vs_pool_plddt
#SBATCH --output=logs/pair_vs_pool_plddt_%j.out
#SBATCH --error=logs/pair_vs_pool_plddt_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=100G

#TODO: check plDDT extraction (expecially for pairwise generated models)
set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

mkdir -p logs

export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

echo "[$(date)] Starting homodimer pair-vs-pool + pLDDT extraction..."

python /cluster/project/beltrao/kdammer/master_thesis/scripts/homomultimers/0_compare_pools_pairs.py

echo "[$(date)] Done."
