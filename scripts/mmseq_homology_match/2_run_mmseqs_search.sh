#!/bin/bash
#
# Run mmseqs2 search: all query proteins vs pdb_seqres, in one batched call.
# Outputs a TSV with per-hit identity, e-value, and the raw aligned sequences
# (qaln/taln), which compute_mmseqs_similarity.py then uses to add a
# BLOSUM62-based %similarity column.
#
# Expects QUERY_FASTA to already exist - build it first with:
#   python build_query_fasta_from_json.py --input_dir <data dir> --output <QUERY_FASTA path>


# --- Slurm Settings ---
#SBATCH --job-name=mmseqs_search
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm_logs/mmseqs_%j.out
#SBATCH --error=slurm_logs/mmseqs_%j.err

set -euo pipefail

# --- USER CONFIGURATION ---
# Combined query FASTA, already built by build_query_fasta_from_json.py, e.g.:
#   python build_query_fasta_from_json.py --input_dir .../data --output all_queries.fasta
QUERY_FASTA="/cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/all_queries.fasta"

# Where to write mmseqs2 databases and results
WORK_DIR="/cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_e_value_100"

PDB_SEQRES_PATH="/cluster/project/alphafold/pdb_seqres/pdb_seqres.txt"

VENV_ACTIVATE="/cluster/project/beltrao/kdammer/master_thesis/.venv/bin/activate"

THREADS=8
SENSITIVITY=8.5   # 1 (fast/low-sensitivity) to 7.5 (max sensitivity, matches jackhmmer-ish recall)

# --- SCRIPT LOGIC ---

mkdir -p "$WORK_DIR"
mkdir -p slurm_logs

echo "=== Step 0: activate env, check mmseqs2 is available ==="
source "$VENV_ACTIVATE"
which mmseqs || { echo "ERROR: mmseqs2 not found on PATH. Install with: uv pip install mmseqs2 (or conda/module load if that fails)"; exit 1; }
mmseqs version

if [ ! -f "$QUERY_FASTA" ]; then
    echo "ERROR: query FASTA not found at $QUERY_FASTA"
    echo "Build it first with: python build_query_fasta_from_json.py --input_dir <data dir> --output $QUERY_FASTA"
    exit 1
fi
n_queries=$(grep -c '^>' "$QUERY_FASTA")
echo "Using query FASTA: $QUERY_FASTA ($n_queries sequences)"

echo "=== Step 1: build mmseqs2 databases (skipped if already built) ==="
PDB_DB="$WORK_DIR/pdb_seqresDB"
QUERY_DB="$WORK_DIR/queryDB"

if [ ! -f "${PDB_DB}.dbtype" ]; then
    echo "Building PDB target database (one-time, ~few minutes)..."
    mmseqs createdb "$PDB_SEQRES_PATH" "$PDB_DB"
else
    echo "PDB target database already exists, skipping build."
fi

mmseqs createdb "$QUERY_FASTA" "$QUERY_DB"

echo "=== Step 2: run the search ==="
RESULT_TSV="/cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_e_value_100/mmseqs_results.tsv"
TMP_DIR="$WORK_DIR/tmp"
mkdir -p "$TMP_DIR"

# Columns: query,target,pident,alnlen,evalue,qlen,tlen,qaln,taln
# qaln/taln are the gapped, aligned query/target sequences - needed by
# compute_mmseqs_similarity.py to derive a BLOSUM62 similarity score.
mmseqs easy-search "$QUERY_FASTA" "$PDB_DB" "$RESULT_TSV" "$TMP_DIR" \
    --threads "$THREADS" \
    -s "$SENSITIVITY" \
    --start-sens 4 \
    --sens-steps 5 \
    --num-iterations 3 \
    --max-seqs 3000 \
    --max-seq-id 1.0 \
    --comp-bias-corr 0 \
    --mask 0 \
    -e 1000 \
    -a 1 \
    --format-mode 4 \
    --format-output "query,target,pident,alnlen,evalue,qlen,tlen,qaln,taln"

echo "=== Done. Raw results: $RESULT_TSV ==="
echo "Next: run compute_mmseqs_similarity.py on this file to add BLOSUM62 similarity."