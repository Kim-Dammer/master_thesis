#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=combfold_eval
#SBATCH --output=logs/combfold_eval_%A_%a.out
#SBATCH --error=logs/combfold_eval_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#
# One array task == one chunk of CHUNK_SIZE complexes, scored independently
# into its own --out-dir. complex_summary.csv/per_chain.csv/per_interface.csv
# are overwritten wholesale on every run, so sharing an --out-dir across
# concurrent array tasks would clobber results -- each task therefore gets
# ${OUT_BASE}/chunk_${SLURM_ARRAY_TASK_ID}/.
#
# This script assumes the reference-PDB cache (REF_CACHE) has already been
# pre-warmed (e.g. via `run_combfold_eval.py --refs-only` on a login node) --
# compute nodes on Euler have no internet access by default.
#
# Not meant to be run directly: submitted by submit_combfold_eval.sh, which
# exports VENV, SCRIPT, MAPPING, MANIFEST, COMBFOLD_BASE, OUT_BASE,
# CHUNK_SIZE, ONLY_FILTER, REF_CACHE, EXTRA_ARGS via --export.
# Resource defaults above are first-pass estimates (see PLAN.md) -- calibrate
# with `seff <jobid>_<task_id>` on a small test array before trusting them at
# full scale, and adjust --cpus-per-task/--mem-per-cpu/--time accordingly.
set -euo pipefail

source "${VENV}/bin/activate"

: "${SCRIPT:?SCRIPT not set}"
: "${CHUNK_SIZE:?CHUNK_SIZE not set}"
: "${MAPPING:?MAPPING not set}"
: "${COMBFOLD_BASE:?COMBFOLD_BASE not set}"
: "${OUT_BASE:?OUT_BASE not set}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?not running under a SLURM array}"

echo "[$(date)] Task ${TASK_ID}: computing complex_ac slice (chunk_size=${CHUNK_SIZE})..."

COMPLEXES=$(python3 - "$MAPPING" "${MANIFEST:-}" "$TASK_ID" "$CHUNK_SIZE" "${ONLY_FILTER:-}" <<'EOF'
import sys
import pandas as pd

mapping_path, manifest_path, task_id, chunk_size, only_filter = sys.argv[1:6]
task_id, chunk_size = int(task_id), int(chunk_size)

df = pd.read_csv(mapping_path)
if "complex_ac" not in df.columns:
    sys.exit("mapping file must have a 'complex_ac' column")
# sorted() is required so every array task derives the SAME global ordering
# independently -- this is what makes disjoint, gap-free chunk slicing work
# without a shared chunk-list file.
complexes = sorted(df["complex_ac"].astype(str).str.strip().unique())

# Restrict to complexes present in the manifest -- MUST match s4 exactly.
if manifest_path:
    mf = pd.read_csv(manifest_path)
    if "complex_ac" not in mf.columns:
        sys.exit("manifest file must have a 'complex_ac' column")
    manifest_acs = {str(x).strip() for x in mf["complex_ac"]}
    complexes = [c for c in complexes if c in manifest_acs]

if only_filter:
    keep = {x.strip() for x in only_filter.split(",") if x.strip()}
    complexes = [c for c in complexes if c in keep]

start = task_id * chunk_size
chunk = complexes[start:start + chunk_size]
print(",".join(chunk))
EOF
)

if [[ -z "${COMPLEXES}" ]]; then
    echo "[$(date)] Task ${TASK_ID}: empty slice (past the end of the complex list) -- nothing to do."
    exit 0
fi

OUT_DIR="${OUT_BASE}/chunk_${TASK_ID}"
mkdir -p "${OUT_DIR}"
N_IN_CHUNK=$(echo "${COMPLEXES}" | tr ',' '\n' | wc -l)
echo "[$(date)] Task ${TASK_ID}: scoring ${N_IN_CHUNK} complexes -> ${OUT_DIR}"

MANIFEST_ARG=()
[[ -n "${MANIFEST:-}" ]] && MANIFEST_ARG=(--manifest "${MANIFEST}")

REF_CACHE_ARG=()
[[ -n "${REF_CACHE:-}" ]] && REF_CACHE_ARG=(--ref-cache "${REF_CACHE}")

uv run "${SCRIPT}" \
    --mapping "${MAPPING}" \
    "${MANIFEST_ARG[@]}" \
    --combfold-base "${COMBFOLD_BASE}" \
    --out-dir "${OUT_DIR}" \
    --only "${COMPLEXES}" \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    "${REF_CACHE_ARG[@]}" \
    ${EXTRA_ARGS:-}

echo "[$(date)] Task ${TASK_ID}: done."
