#!/bin/bash
#
# This script intelligently finds unique proteins that have NOT already been
# processed and submits a Slurm job array to analyze only the new ones.

# --- Slurm Settings ---
#SBATCH --job-name=template_analysis
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G

# --- USER CONFIGURATION ---
#INPUT_DATA_DIR="/cluster/project/beltrao/JvG/Projects/ubn_conf_changes/ubn_conf_changes/output/data/postAF3/uniqSeqs"
#OUTPUT_RESULTS_DIR="$SLURM_SUBMIT_DIR/chunk_analysis_results/uniqSeqsRes"
INPUT_DATA_DIR="/cluster/project/beltrao/kdammer/master_thesis/scripts/af3-template-analysis_from_ubnAF3Benchmark/af3-template-analysis/data"
OUTPUT_RESULTS_DIR="$SLURM_SUBMIT_DIR/chunk_analysis_results/uniqSeqs"
CHUNK_SIZE=1

# --- CONTAINER AND SCRIPT CONFIGURATION ---
AF3_IMAGE="/cluster/project/beltrao/shared/alphafold3/images/alphafold3_2e2ffc1.sif"
PROJECT_DIR="/cluster/project/beltrao/kdammer"
DATABASE_DIR="/cluster/project/alphafold"
PYTHON_SCRIPT="$SLURM_SUBMIT_DIR/scripts/run_chunk_analysis_mmseq2.py"
#TRUE_IDENTITY_SCRIPT="$SLURM_SUBMIT_DIR/scripts/calculate_true_identity.py"
PDB_SEQRES_PATH="/cluster/project/alphafold/pdb_seqres/pdb_seqres.txt"

# --- SCRIPT LOGIC ---

# Master job logic
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    mkdir -p "$OUTPUT_RESULTS_DIR"
    SLURM_LOG_DIR="$SLURM_SUBMIT_DIR/slurm_logs"
    mkdir -p "$SLURM_LOG_DIR"
    echo "Slurm logs will be saved in: $SLURM_LOG_DIR"

    echo "Scanning for unique, UNPROCESSED proteins (case-insensitive)..."
    
    declare -A seen_proteins
    UNPROCESSED_FILES=()
    for file in $(find "$INPUT_DATA_DIR" -maxdepth 1 -type f -name "*.json" | sort); do
        # --- FIXED: Convert protein_id to lowercase for checking ---
        protein_id_orig=$(basename "$file" | cut -d'_' -f1)
        protein_id_lower=$(echo "$protein_id_orig" | tr '[:upper:]' '[:lower:]')
        
        expected_output_dir="${OUTPUT_RESULTS_DIR}/${protein_id_lower}"

        # Use the lowercase version as the key for the 'seen' check
        if [[ -z "${seen_proteins[$protein_id_lower]}" ]]; then
            if [ ! -d "$expected_output_dir" ]; then
                UNPROCESSED_FILES+=("$file")
                echo "  [QUEUE] Found new protein to process: ${protein_id_orig}"
            else
                echo "  [SKIP]  Skipping already processed protein: ${protein_id_orig}"
            fi
            # Mark the lowercase version as seen
            seen_proteins["$protein_id_lower"]=1
        fi
    done

    TOTAL_FILES=${#UNPROCESSED_FILES[@]}

    if [ "$TOTAL_FILES" -eq 0 ]; then
        echo "All proteins have already been processed. No new jobs to submit. Exiting."
        exit 0
    fi

    NUM_JOBS=$(( (TOTAL_FILES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    echo "Found $TOTAL_FILES new proteins to process. Submitting a job array with $NUM_JOBS tasks."

    # Write the file list to a manifest file instead of an env var -
    # exporting thousands of paths as one env var blows past the OS
    # ARG_MAX limit ("Argument list too long") when sbatch is invoked.
    MANIFEST_FILE="$SLURM_LOG_DIR/file_manifest_$$.txt"
    printf '%s\n' "${UNPROCESSED_FILES[@]}" > "$MANIFEST_FILE"
    export MANIFEST_FILE

    # Submit array job and capture its job ID
    ARRAY_JOB_ID=$(sbatch --parsable --array=1-$NUM_JOBS \
           --output="$SLURM_LOG_DIR/%A_%a.out" \
           --error="$SLURM_LOG_DIR/%A_%a.err" \
           --export=ALL,MANIFEST_FILE="$MANIFEST_FILE" \
           "$0")

    echo "Array job $ARRAY_JOB_ID submitted."

    # DO THIS BY RUNNING THE SCRIPT INSTEAD
    # # Submit post-processing job (calculate_true_identity.py) dependent on array completion
    # sbatch --dependency=afterok:$ARRAY_JOB_ID \
    #        --job-name=true_identity \
    #        --time=02:00:00 \
    #        --cpus-per-task=8 \
    #        --mem-per-cpu=8G \
    #        --output="$SLURM_LOG_DIR/true_identity_%j.out" \
    #        --error="$SLURM_LOG_DIR/true_identity_%j.err" \
    #        --wrap="singularity exec \
    #            --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    #            --bind ${DATABASE_DIR}:${DATABASE_DIR} \
    #            --bind \$(pwd):\$(pwd) \
    #            ${AF3_IMAGE} \
    #            python3 ${TRUE_IDENTITY_SCRIPT} \
    #                --output_dir ${OUTPUT_RESULTS_DIR} \
    #                --pdb_seqres ${PDB_SEQRES_PATH}"

    # echo "Post-processing job submitted (depends on $ARRAY_JOB_ID)."
    exit
fi

# --- ARRAY TASK ---
# This part now receives the list of unprocessed files from the master job.

# Read the exported list of files into a Bash array
mapfile -t UNPROCESSED_FILES < "$MANIFEST_FILE"

TASK_ID=$SLURM_ARRAY_TASK_ID
START_INDEX=$(( (TASK_ID - 1) * CHUNK_SIZE ))
END_INDEX=$(( START_INDEX + CHUNK_SIZE - 1 ))
TOTAL_FILES=${#UNPROCESSED_FILES[@]}

if [ "$END_INDEX" -ge "$TOTAL_FILES" ]; then
    END_INDEX=$((TOTAL_FILES - 1))
fi

CHUNK_FILES=()
for i in $(seq $START_INDEX $END_INDEX); do
    CHUNK_FILES+=("${UNPROCESSED_FILES[$i]}")
done

echo "--- Starting Slurm Array Task ${TASK_ID} ---"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Processing ${#CHUNK_FILES[@]} unique proteins in this chunk."

singularity exec \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    --bind ${DATABASE_DIR}:${DATABASE_DIR} \
    --bind $(pwd):$(pwd) \
    ${AF3_IMAGE} \
    sh -c "python3 ${PYTHON_SCRIPT} \
        --output_dir ${OUTPUT_RESULTS_DIR} \
        --job_id ${TASK_ID} \
        --input_files ${CHUNK_FILES[*]}"

echo "--- Task ${TASK_ID} finished at $(date) ---"