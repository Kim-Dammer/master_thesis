#!/bin/bash
#SBATCH --job-name=stoic_pipeline
#SBATCH --output=logs/stoic_%j.out
#SBATCH --error=logs/stoic_%j.err
#SBATCH --time=24:00:00
##SBATCH --partition=gpu
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="/cluster/project/beltrao/kdammer/master_thesis/local_models/stoic"
export HF_HOME="/cluster/project/beltrao/kdammer/master_thesis/local_models"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on Node: $(hostname)"
echo "Assigned GPU(s):"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=========================================="

# Single flat fasta directory
FASTA_DIR="../data/Pipeline/first_setup_pipeline_complexes"
OUTPUT_DIR="../data/Pipeline/stoic_results/first_setup_pipeline_complexes"
mkdir -p "$OUTPUT_DIR"

echo "Reading from: ${FASTA_DIR}"
echo "Saving to: ${OUTPUT_DIR}"

TOTAL_FILES=$(ls -1 "$FASTA_DIR"/*.fasta | wc -l)  #fasta files if not present can be created with the helper script 01_create_fasta_files_for_stoic.py
CURRENT_FILE=0

# TASK_ID/TASK_COUNT default to 0/1 (i.e. every file, identical to the
# original single-job loop) when SLURM_ARRAY_TASK_ID/_COUNT are unset -- e.g.
# if this script is ever run directly without --array.
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
echo "Array task ${TASK_ID}/${TASK_COUNT}: claiming every ${TASK_COUNT}th complex starting at index ${TASK_ID} (${TOTAL_FILES} complexes total)"

shopt -s nullglob
ALL_FILES=("$FASTA_DIR"/*.fasta)
shopt -u nullglob

for (( i=TASK_ID; i<TOTAL_FILES; i+=TASK_COUNT )); do
    FASTA_FILE="${ALL_FILES[$i]}"
    [ -e "$FASTA_FILE" ] || continue

    # Get complex name (e.g., "CPX-21" from "CPX-21.fasta")
    CPX_NAME=$(basename "$FASTA_FILE" .fasta)

    # Per-complex output dir
    COMPLEX_OUTPUT_DIR="$OUTPUT_DIR/$CPX_NAME"
    mkdir -p "$COMPLEX_OUTPUT_DIR"

    ((CURRENT_FILE++))
    echo "(task ${TASK_ID}, local $CURRENT_FILE) Processing: $CPX_NAME"

    uv run stoic_predict_stoichiometry \
        --model "$MODEL_PATH" \
        --input-path "$FASTA_FILE" \
        --top-n 10 \
        --output-dir "$COMPLEX_OUTPUT_DIR"
done

echo "Task ${TASK_ID} complete! ($CURRENT_FILE complexes processed)"