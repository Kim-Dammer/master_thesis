#!/bin/bash
#SBATCH --job-name=homomultimer_usalign
#SBATCH --output=logs/homomultimer_usalign_%j.out
#SBATCH --error=logs/homomultimer_usalign_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00

mkdir -p logs
cd /cluster/project/beltrao/kdammer/master_thesis/scripts/homomultimer_pool_pair_comparison
uv run run_homomultimer_usalign.py