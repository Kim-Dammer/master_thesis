#!/bin/bash
#SBATCH --job-name=usalign_pool_only
#SBATCH --output=logs/usalign_pool_only_%j.out
#SBATCH --error=logs/usalign_pool_only_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

mkdir -p logs
cd /cluster/project/beltrao/kdammer/master_thesis
uv run python scripts/homomultimer_pool_pair_comparison/run_usalign_pool_only.py
