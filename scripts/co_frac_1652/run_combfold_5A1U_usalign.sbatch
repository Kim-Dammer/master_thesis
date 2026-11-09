#!/bin/bash
#SBATCH --job-name=combfold_5A1U_usalign
#SBATCH --output=logs/combfold_5A1U_usalign_%j.out
#SBATCH --error=logs/combfold_5A1U_usalign_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00

mkdir -p logs
cd /cluster/project/beltrao/kdammer/master_thesis/scripts/combfold_5A1U_comparison
uv run run_combfold_5A1U_usalign.py
