#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=trimming_usalign
#SBATCH --output=logs/trimming_usalign_%j.out
#SBATCH --error=logs/trimming_usalign_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1   # print() shows up in the log immediately, not at the end

cd /cluster/project/beltrao/kdammer/master_thesis/scripts/Pipeline/effect_of_trimming
uv run effect_of_trimming_on_cf_assembly.py
