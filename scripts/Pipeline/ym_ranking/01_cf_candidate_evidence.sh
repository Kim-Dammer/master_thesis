#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=cf_candidate_evidence
#SBATCH --output=logs/cf_candidate_evidence_%j.out
#SBATCH --error=logs/cf_candidate_evidence_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1   # print() shows up in the log immediately, not at the end

module load eth_proxy       # compute nodes need the proxy to download the core PDB mmCIFs from RCSB

cd /cluster/project/beltrao/kdammer/master_thesis/scripts/Pipeline/ym_ranking
uv run 01_cf_candidate_evidence.py
