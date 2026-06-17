#!/bin/bash
#SBATCH --job-name=stoic_benchmark
#SBATCH --output=logs/stoic_%A_batch_%a.out
#SBATCH --error=logs/stoic_%A_batch_%a.err
#SBATCH --time=24:00:00
##SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --array=1-21

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="/cluster/project/beltrao/kdammer/master_thesis/local_models/stoic"
export HF_HOME="/cluster/project/beltrao/kdammer/master_thesis/local_models"

echo "=========================================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on Node: $(hostname)"
echo "Assigned GPU(s):"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=========================================="


BATCH_DIR="../data/Stoic/benchmark_stoic_batches/benchmark_stoic_batch_${SLURM_ARRAY_TASK_ID}"


OUTPUT_DIR="./stoic_results/benchmark_stoic_batch_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTPUT_DIR"

echo "Processing batch ${SLURM_ARRAY_TASK_ID}..."
echo "Reading from: ${BATCH_DIR}"
echo "Saving to: ${OUTPUT_DIR}"

TOTAL_FILES=$(ls -1 "$BATCH_DIR"/*.fasta | wc -l)
CURRENT_FILE=0

for FASTA_FILE in "$BATCH_DIR"/*.fasta; do
    [ -e "$FASTA_FILE" ] || continue
    
    # 1. Get the PDB name (e.g., "8fcj" from "8fcj.fasta")
    PDB_NAME=$(basename "$FASTA_FILE" .fasta)
    
    # 2. Create a specific sub-folder for this protein
    # This results in: ./stoic_results/benchmark_stoic_batch_1/8fcj/
    PROTEIN_OUTPUT_DIR="$OUTPUT_DIR/$PDB_NAME"
    mkdir -p "$PROTEIN_OUTPUT_DIR"
    
    ((CURRENT_FILE++))
    echo "[Batch ${SLURM_ARRAY_TASK_ID}] ($CURRENT_FILE/$TOTAL_FILES) Processing: $PDB_NAME"
    
    # 3. Run prediction pointing to the NEW unique sub-folder
    uv run stoic_predict_stoichiometry \
        --model "$MODEL_PATH" \
        --input-path "$FASTA_FILE" \
        --top-n 3 \
        --output-dir "$PROTEIN_OUTPUT_DIR"
        
done

echo "Batch ${SLURM_ARRAY_TASK_ID} complete!"