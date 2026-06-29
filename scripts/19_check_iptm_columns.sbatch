#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_iptm
#SBATCH --output=logs/19_check_iptm_cols%j.out
#SBATCH --error=logs/19_check_iptm_cols%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

python3 19_check_iptm_cols.py
