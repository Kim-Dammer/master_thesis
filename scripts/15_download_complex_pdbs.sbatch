#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=download_pdbs
#SBATCH --output=/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files/logs/download_pdbs_%j.out
#SBATCH --error=/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files/logs/download_pdbs_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# ===========================================================================
# Download PDB files cross-referenced from Complex Portal for all yeast
# complexes. Produces three subfolders:
#   identity/                 — PDB represents the complete complex
#   subset/                   — Complex is part of a larger PDB assembly
#   experimental_evidence/    — PDB cited as evidence (untagged in TSV)
#
# Just run:  sbatch 15_download_complex_pdbs.sbatch
# ===========================================================================

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
SCRIPT_DIR="/cluster/project/beltrao/kdammer/master_thesis/scripts"
OUT_DIR="/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files"
TSV="/cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv"

mkdir -p "${OUT_DIR}/logs"

source "${VENV}/bin/activate"

echo "[$(date)] Starting PDB download..."

python "${SCRIPT_DIR}/16_download_complex_pdbs.py" \
    --tsv "${TSV}" \
    --out-dir "${OUT_DIR}"

echo "[$(date)] Download complete."
