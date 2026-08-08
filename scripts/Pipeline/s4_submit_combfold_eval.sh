#!/bin/bash
# s4_run_combfold_eval.sh -- combined driver for a disconnect-proof combfold_eval
# production run on Euler: pre-warms the reference-PDB cache on the login node
# (synchronously, in the foreground -- compute nodes have no internet access),
# then submits s4_submit_combfold_eval_array.sbatch as a SLURM job array where
# each task scores one fixed-size chunk of complexes independently into its
# own output subfolder.
#
# Usage:
#   ./s4_run_combfold_eval.sh \
#       --mapping /path/to/mapping.csv \
#       --combfold-base /path/to/CombFold \
#       --out-dir /path/to/combfold_eval_out \
#       [--manifest /path/to/manifest.csv] \
#       [--chunk-size 20] [--throttle 10] \
#       [--only complex_ac1,complex_ac2,...] \
#       [--uniprot-csv /path/to/uniprot.csv] \
#       [--ref-cache /path/to/ref_cache] \
#       [-- any other 02_combfold_eval.py flag, e.g. --dockq-allowed-mismatches 12]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/02_combfold_eval.py"
ARRAY_SBATCH="${SCRIPT_DIR}/s5_combfold_eval_array.sbatch"
VENV="${VENV:-/cluster/project/beltrao/kdammer/master_thesis/.venv}"

MAPPING="/cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/all_pdb_matches_with_match_class.csv"
MANIFEST="/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/t6_CF_test_Example_for_RM_TM/all_pdb_present_t4_CF_test_pipeline_complexes_combfold_results.csv"
COMBFOLD_BASE=""
OUT_BASE=""
CHUNK_SIZE=20
THROTTLE=10
ONLY_FILTER=""
UNIPROT_CSV="/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_CF_YM_yeast_proteins_uniprot_mapped_sequences.csv"
REF_CACHE="/cluster/project/beltrao/kdammer/master_thesis/data/reference_pdb"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mapping) MAPPING="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --combfold-base) COMBFOLD_BASE="$2"; shift 2 ;;
        --out-dir) OUT_BASE="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --throttle) THROTTLE="$2"; shift 2 ;;
        --only) ONLY_FILTER="$2"; shift 2 ;;
        --uniprot-csv) UNIPROT_CSV="$2"; shift 2 ;;
        --ref-cache) REF_CACHE="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

: "${MAPPING:?--mapping is required}"
: "${COMBFOLD_BASE:?--combfold-base is required}"
: "${OUT_BASE:?--out-dir is required}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "[s4_run_combfold_eval] ERROR: ${SCRIPT} not found." >&2
    exit 1
fi
if [[ ! -f "${ARRAY_SBATCH}" ]]; then
    echo "[s4_run_combfold_eval] ERROR: ${ARRAY_SBATCH} not found." >&2
    exit 1
fi

if [[ -n "${UNIPROT_CSV}" ]]; then
    EXTRA_ARGS+=(--uniprot-csv "${UNIPROT_CSV}")
fi
EXTRA_ARGS_STR="${EXTRA_ARGS[*]:-}"

source "${VENV}/bin/activate"
mkdir -p "${OUT_BASE}" logs

echo "=== Step 1/3: pre-warming reference-PDB cache on login node ==="
REFS_ONLY_ARGS=(--refs-only --mapping "${MAPPING}" --n-workers 16)
[[ -n "${ONLY_FILTER}" ]] && REFS_ONLY_ARGS+=(--only "${ONLY_FILTER}")
[[ -n "${REF_CACHE}" ]] && REFS_ONLY_ARGS+=(--ref-cache "${REF_CACHE}")
uv run "${SCRIPT}" "${REFS_ONLY_ARGS[@]}"

echo "=== Step 2/3: computing chunk count (chunk_size=${CHUNK_SIZE}) ==="
N_CHUNKS=$(python3 - "$MAPPING" "$CHUNK_SIZE" "$ONLY_FILTER" <<'EOF'
import math
import sys

import pandas as pd

mapping_path, chunk_size, only_filter = sys.argv[1:4]
chunk_size = int(chunk_size)

df = pd.read_csv(mapping_path)
if "complex_ac" not in df.columns:
    sys.exit("mapping file must have a 'complex_ac' column")
# Same filtering/sorting logic as s4_submit_combfold_eval_array.sbatch's inline
# heredoc -- kept in sync manually since there's no shared chunk-list file;
# both must agree so this chunk count matches what each array task computes.
complexes = sorted(df["complex_ac"].astype(str).str.strip().unique())

if only_filter:
    keep = {x.strip() for x in only_filter.split(",") if x.strip()}
    complexes = [c for c in complexes if c in keep]

print(max(1, math.ceil(len(complexes) / chunk_size)))
EOF
)
echo "Total complexes will be split into ${N_CHUNKS} chunk(s) of up to ${CHUNK_SIZE} each."

echo "=== Step 3/3: submitting SLURM array (0-$((N_CHUNKS - 1))%${THROTTLE}) ==="
EXPORT_VARS="ALL,VENV=${VENV},SCRIPT=${SCRIPT},MAPPING=${MAPPING},MANIFEST=${MANIFEST},COMBFOLD_BASE=${COMBFOLD_BASE},OUT_BASE=${OUT_BASE},CHUNK_SIZE=${CHUNK_SIZE},ONLY_FILTER=${ONLY_FILTER},REF_CACHE=${REF_CACHE},EXTRA_ARGS=${EXTRA_ARGS_STR}"

JOB_ID=$(sbatch --parsable \
    --array="0-$((N_CHUNKS - 1))%${THROTTLE}" \
    --export="${EXPORT_VARS}" \
    "${ARRAY_SBATCH}")

echo ""
echo "Submitted array job ${JOB_ID} (${N_CHUNKS} tasks, throttled to ${THROTTLE} concurrent)."
echo ""
echo "Track progress:"
echo "  squeue -j ${JOB_ID}"
echo "Per-task resource usage once tasks finish (calibrate --cpus-per-task/--mem-per-cpu/--time!):"
echo "  seff ${JOB_ID}_0"
echo "Logs:"
echo "  logs/combfold_eval_${JOB_ID}_*.out / .err"
echo ""
echo ""
echo "=== Step 4/4: scheduling automatic merge after all array tasks finish ==="
MERGE_SCRIPT="${SCRIPT_DIR}/02_02_merge_combfold_eval_chunks.py"
if [[ ! -f "${MERGE_SCRIPT}" ]]; then
    echo "[s4_run_combfold_eval] WARNING: ${MERGE_SCRIPT} not found -- skipping merge scheduling." >&2
else
    MERGE_JOB_ID=$(sbatch --parsable \
        --account=es_biol \
        --partition=es_biol \
        --job-name=combfold_eval_merge \
        --dependency="afterok:${JOB_ID}" \
        --kill-on-invalid-dep=yes \
        --cpus-per-task=1 \
        --mem-per-cpu=4G \
        --time=00:30:00 \
        --output=logs/combfold_eval_merge_%j.out \
        --error=logs/combfold_eval_merge_%j.err \
        --wrap="source ${VENV}/bin/activate && uv run ${MERGE_SCRIPT} --chunks-dir ${OUT_BASE} --out-dir ${OUT_BASE}")
    echo "Scheduled merge job ${MERGE_JOB_ID}, will run once array job ${JOB_ID} completes successfully."
    echo "  squeue -j ${MERGE_JOB_ID}"
    echo "  logs/combfold_eval_merge_${MERGE_JOB_ID}.out / .err"
fi
echo "Submission complete — jobs queued"