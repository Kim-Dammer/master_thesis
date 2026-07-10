#!/bin/bash
#
# This script finds protein result folders that need true identity calculation
# and submits a Slurm job array to process them in parallel.

# --- Slurm Settings ---
#SBATCH --job-name=true_identity
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

# --- USER CONFIGURATION ---
RESULTS_DIR="$SLURM_SUBMIT_DIR/chunk_analysis_results/uniqSeqs"

# --- SCRIPT CONFIGURATION ---
PROJECT_DIR="/cluster/project/beltrao/JvG"
DATABASE_DIR="/cluster/project/alphafold"
PYTHON_SCRIPT="$SLURM_SUBMIT_DIR/scripts/run_single_true_identity.py"
PDB_SEQRES_PATH="/cluster/project/alphafold/pdb_seqres/pdb_seqres.txt"

# Venv config (uv-managed)
VENV_ACTIVATE="/cluster/project/beltrao/kdammer/master_thesis/.venv/bin/activate"

# --- SCRIPT LOGIC ---

# Master job logic (submits the array)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_LOG_DIR="$SLURM_SUBMIT_DIR/slurm_logs"
    mkdir -p "$SLURM_LOG_DIR"
    echo "Slurm logs will be saved in: $SLURM_LOG_DIR"

    echo "Scanning for protein folders that need true identity calculation..."

    UNPROCESSED_FOLDERS=()
    for folder in $(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort); do
        if [ ! -f "$folder/true_identity.csv" ]; then
            if ls "$folder"/*_templates.txt 1> /dev/null 2>&1; then
                UNPROCESSED_FOLDERS+=("$folder")
                echo "  [QUEUE] $(basename "$folder")"
            else
                echo "  [SKIP]  $(basename "$folder") - no HMMER output files"
            fi
        else
            echo "  [DONE]  $(basename "$folder") - already processed"
        fi
    done

    TOTAL_FOLDERS=${#UNPROCESSED_FOLDERS[@]}

    if [ "$TOTAL_FOLDERS" -eq 0 ]; then
        echo "All protein folders have already been processed. No new jobs to submit. Exiting."
        exit 0
    fi

    echo "Found $TOTAL_FOLDERS protein folders to process."

    export FOLDERS_TO_PROCESS="${UNPROCESSED_FOLDERS[*]}"

    ARRAY_JOB_ID=$(sbatch --parsable --array=1-"$TOTAL_FOLDERS" \
        --output="$SLURM_LOG_DIR/true_identity_%A_%a.out" \
        --error="$SLURM_LOG_DIR/true_identity_%A_%a.err" \
        "$0")

    echo "Array job $ARRAY_JOB_ID submitted."

    sbatch --dependency=afterany:$ARRAY_JOB_ID \
        --job-name=ti_summary \
        --time=00:10:00 \
        --cpus-per-task=1 \
        --mem=1G \
        --output="$SLURM_LOG_DIR/true_identity_summary_%j.out" \
        --error="$SLURM_LOG_DIR/true_identity_summary_%j.err" \
        --wrap="
echo '=== True Identity Error Summary ===' > $RESULTS_DIR/error_summary.log
echo 'Generated at: \$(date)' >> $RESULTS_DIR/error_summary.log
echo '' >> $RESULTS_DIR/error_summary.log

SUCCESS=0
FAILED=0
for folder in $RESULTS_DIR/*/; do
    if [ -f \"\${folder}true_identity.csv\" ]; then
        ((SUCCESS++))
    elif [ -f \"\${folder}true_identity.log\" ]; then
        ((FAILED++))
        echo \"FAILED: \$(basename \$folder)\" >> $RESULTS_DIR/error_summary.log
        echo '  Last 5 lines of log:' >> $RESULTS_DIR/error_summary.log
        tail -5 \"\${folder}true_identity.log\" | sed 's/^/    /' >> $RESULTS_DIR/error_summary.log
        echo '' >> $RESULTS_DIR/error_summary.log
    fi
done

echo '' >> $RESULTS_DIR/error_summary.log
echo \"Summary: \$SUCCESS successful, \$FAILED failed\" >> $RESULTS_DIR/error_summary.log
cat $RESULTS_DIR/error_summary.log
"

    echo "Summary job submitted (depends on $ARRAY_JOB_ID)."
    echo "Error summary will be written to: $RESULTS_DIR/error_summary.log"
    exit 0
fi

# --- ARRAY TASK ---

# Read the exported list of folders into a Bash array
read -r -a UNPROCESSED_FOLDERS <<< "$FOLDERS_TO_PROCESS"

TASK_ID=$SLURM_ARRAY_TASK_ID
FOLDER_INDEX=$((TASK_ID - 1))
PROTEIN_FOLDER="${UNPROCESSED_FOLDERS[$FOLDER_INDEX]}"

if [ -z "$PROTEIN_FOLDER" ]; then
    echo "Error: No folder assigned to task $TASK_ID"
    exit 1
fi

echo "--- Starting True Identity Task ${TASK_ID} ---"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Processing folder: $(basename "$PROTEIN_FOLDER")"

# --- MVP venv + biopython check (logs to Slurm output) ---
echo "=== ENV CHECK (venv + biopython) ==="
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Time: $(date)"

source "$VENV_ACTIVATE"
ACT_RC=$?

echo "venv activate rc=$ACT_RC"
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
echo "which python=$(which python 2>/dev/null || echo 'NOT FOUND')"
python -V 2>&1

python -c "import Bio; print('Biopython OK:', Bio.__version__)"
RC=$?

if [ $ACT_RC -ne 0 ] || [ $RC -ne 0 ]; then
    echo "ERROR: Venv/Biopython check failed (activate rc=$ACT_RC, import rc=$RC)"
    exit 2
fi
echo "=== ENV CHECK PASSED ==="
# --- end check ---

# Run python directly (NO singularity) using the activated venv's python
python "${PYTHON_SCRIPT}" \
    --protein_folder "${PROTEIN_FOLDER}" \
    --pdb_seqres "${PDB_SEQRES_PATH}"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Task ${TASK_ID} failed with exit code $EXIT_CODE"
fi

echo "--- Task ${TASK_ID} finished at $(date) with exit code $EXIT_CODE ---"
exit $EXIT_CODE