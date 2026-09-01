#!/bin/bash
#SBATCH --job-name=add_iptm_corrected
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20G
#SBATCH --time=01:00:00
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --output=add_iptm_corrected_%j.out
#SBATCH --error=add_iptm_corrected_%j.err

/cluster/project/beltrao/kdammer/master_thesis/.venv/bin/python \
    /cluster/project/beltrao/kdammer/master_thesis/scripts/helper_scripts/add_iptm_corrected.py