#!/bin/bash
##SBATCH --account es_biol
#SBATCH --partition es_biol
#SBATCH --job-name=combfold
#SBATCH --output=logs/combfold_%j.out
#SBATCH --error=logs/combfold_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=5G
#SBATCH --gpus=1

# ============================================================
# CombFold assembly script
#
# Usage:
# sbatch run_CombFold.sbatch 'Q12186(1),P36084(2)'
# sbatch run_CombFold.sbatch 'Q12186(1),P36084(2),1:2' # ratio validated
#
# Single quoted argument: comma-separated NAME(COUNT) entries.
# Optional trailing ratio (e.g. 1:2) must match the parenthesized
# counts in the order the proteins were written.
#
# Pair PDB sources (ALL from pooled-PPI DB now):
# - homodimers   -> pooled-PPI DB (self-pairs: uniprot_id1 == uniprot_id2)
# - heterodimers -> pooled-PPI DB (cross-pairs: uniprot_id1 != uniprot_id2)
# ============================================================

set -euo pipefail

# --- Config -------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
SEQUENCES_PARQUET="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.04/proteins.parquet"
POOLED_PPI_DB="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.04"
COMBFOLD_SIF="/cluster/project/beltrao/shared/alphafold3/images/combfold_latest.sif"
OUTPUT_BASE="/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/first_setup/CombFold"
# ------------------------------------------------------------

source "${VENV}/bin/activate"

if [[ $# -ne 1 ]]; then
    echo "ERROR: expected exactly one argument (quoted)."
    echo "Usage: sbatch $(basename "$0") 'Q12186(1),P36084(2)'"
    echo " or: sbatch $(basename "$0") 'Q12186(1),P36084(2),1:2'"
    exit 1
fi

SPEC="$1"
echo "Spec: ${SPEC}"

mkdir -p logs

export SPEC OUTPUT_BASE SEQUENCES_PARQUET POOLED_PPI_DB

# --- Recompute complex name on bash side (no temp file) -----
COMPLEX_NAME=$(python -c "
import os, re
spec = os.environ['SPEC']
entry_re = re.compile(r'^([A-Za-z0-9_]+)\((\d+)\)\$')
counts = {}
for tok in [t.strip() for t in spec.split(',') if t.strip()]:
    m = entry_re.match(tok)
    if m:
        counts[m.group(1)] = int(m.group(2))
print('_'.join(f'{p}x{counts[p]}' for p in sorted(counts)))
")

if [[ -z "${COMPLEX_NAME}" ]]; then
    echo "ERROR: could not derive complex name from spec '${SPEC}'."
    exit 1
fi

INPUT_DIR="${OUTPUT_BASE}/${COMPLEX_NAME}_input"
PDBS_DIR="${OUTPUT_BASE}/${COMPLEX_NAME}_input/pdbs/"
OUTPUT_DIR="${OUTPUT_BASE}/${COMPLEX_NAME}_output/"
SUBUNITS_JSON="${INPUT_DIR}/subunits.json"

mkdir -p "${INPUT_DIR}/pdbs" "${OUTPUT_DIR}"

echo "Complex name: ${COMPLEX_NAME}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

# ---- Step 1: Validate spec, write subunits.json + pairwise PDBs ----
echo "[$(date)] Parsing spec, writing subunits.json and pairwise PDBs..."

python - <<'EOF'
import collections
import json
import os
import re
import sys
from functools import reduce
from math import gcd
from pathlib import Path

import pandas as pd
import af3io
import pooled_ppi

spec = os.environ["SPEC"]
output_base = os.environ["OUTPUT_BASE"]
sequences_path = os.environ["SEQUENCES_PARQUET"]
pooled_ppi_db = os.environ["POOLED_PPI_DB"]

# --- Parse spec ----------------------------------------------------------
tokens = [t.strip() for t in spec.split(",") if t.strip()]
entry_re = re.compile(r"^([A-Za-z0-9_]+)\((\d+)\)$")
ratio_re = re.compile(r"^\d+(:\d+)+$")

counts = collections.OrderedDict()
ratio_token = None
for tok in tokens:
    m = entry_re.match(tok)
    if m:
        name, n = m.group(1), int(m.group(2))
        if n < 1:
            print(f"ERROR: count must be >= 1 in token '{tok}'")
            sys.exit(1)
        if name in counts:
            print(f"ERROR: protein '{name}' listed twice in spec")
            sys.exit(1)
        counts[name] = n
    elif ratio_re.match(tok):
        if ratio_token is not None:
            print(f"ERROR: spec contains more than one ratio token "
                  f"('{ratio_token}' and '{tok}')")
            sys.exit(1)
        ratio_token = tok
    else:
        print(f"ERROR: cannot parse token '{tok}'. "
              f"Expected NAME(COUNT) or a ratio like 1:2.")
        sys.exit(1)

if not counts:
    print("ERROR: spec contains no protein entries.")
    sys.exit(1)

# --- Validate optional ratio against parenthesized counts ----------------
if ratio_token is not None:
    ratio_parts = [int(x) for x in ratio_token.split(":")]
    listed_counts = list(counts.values())  # order proteins were written

    if len(ratio_parts) != len(listed_counts):
        print(f"ERROR: ratio '{ratio_token}' has {len(ratio_parts)} parts "
              f"but spec has {len(listed_counts)} proteins.")
        sys.exit(1)

    def reduce_ratio(xs):
        g = reduce(gcd, xs)
        return tuple(x // g for x in xs)

    listed_reduced = reduce_ratio(listed_counts)
    ratio_reduced = reduce_ratio(ratio_parts)

    if listed_reduced != ratio_reduced:
        proteins_in_order = list(counts)
        counts_str = ":".join(str(c) for c in listed_counts)
        ratio_str = ":".join(str(r) for r in ratio_parts)
        print("ERROR: ratio mismatch.")
        print(f" Proteins (order given): {proteins_in_order}")
        print(f" Counts from parentheses: {counts_str} "
              f"(reduced {':'.join(map(str, listed_reduced))})")
        print(f" Ratio token: {ratio_str} "
              f"(reduced {':'.join(map(str, ratio_reduced))})")
        sys.exit(1)

    print(f"Ratio check OK: {ratio_token} matches counts "
          f"{':'.join(str(c) for c in listed_counts)}.")

# Sort by UniProt ID for stable output naming downstream
proteins = sorted(counts)
counts = collections.OrderedDict((p, counts[p]) for p in proteins)
complex_name = "_".join(f"{p}x{counts[p]}" for p in proteins)
print(f"Resolved complex: {complex_name}")
print(f"Stoichiometry: {dict(counts)}")

input_dir = Path(output_base) / f"{complex_name}_input"
pdbs_dir = input_dir / "pdbs"
input_dir.mkdir(parents=True, exist_ok=True)
pdbs_dir.mkdir(parents=True, exist_ok=True)

# --- Write subunits.json -------------------------------------------------
sequences = pd.read_parquet(sequences_path)

js = collections.OrderedDict()
chain_iter = af3io.input.enumerate_chains()
for protein in proteins:
    n = counts[protein]
    chain_names = [next(chain_iter) for _ in range(n)]
    seq = sequences.query("uniprot_id == @protein")["seq"].squeeze()
    if not isinstance(seq, str):
        print(f"ERROR: sequence not found (or ambiguous) for {protein}")
        sys.exit(1)
    js[protein] = {
        "name": protein,
        "chain_names": chain_names,
        "start_res": 1,
        "sequence": seq,
    }
    print(f"Subunit {protein}: {n} copies, chains {chain_names}")

subunits_path = input_dir / "subunits.json"
with open(subunits_path, "w") as fh:
    json.dump(js, fh, indent=2)
print(f"Written: {subunits_path}")

# --- Required pairwise PDBs: split into homo and hetero ------------------
required_homodimers = []  # list of uniprot_id with count >= 2
required_heteropairs = []  # list of tuple(id1, id2), sorted

for i, p1 in enumerate(proteins):
    for p2 in proteins[i:]:
        if p1 == p2:
            if counts[p1] >= 2:
                required_homodimers.append(p1)
        else:
            required_heteropairs.append(tuple(sorted([p1, p2])))

print(f"Required homodimers: {required_homodimers}")
print(f"Required heteropairs: {required_heteropairs}")

# --- Look up ALL pairs (homo + hetero) in pooled PPI DB ------------------
# The pooled-PPI DB stores UniProt IDs in lowercase; the spec uses uppercase.
# Build a lowercase lookup set for querying, keep original IDs for output.
proteins_lower = {p.lower() for p in proteins}

pp = pooled_ppi.PooledPredictionsDb(pooled_ppi_db)

candidates = (
    pp.pairs
    .query("uniprot_id1.str.lower() in @proteins_lower & "
           "uniprot_id2.str.lower() in @proteins_lower")
    .sort_values("af3_pair")
    .reset_index(drop=True)
)
# pair_key uses lowercase IDs (as stored in the DB).
# Use a list comprehension (NOT .apply) — returning a tuple from .apply
# makes pandas try to expand it into multiple columns, which raises
# "Cannot set a DataFrame with multiple columns to the single column pair_key".
candidates["pair_key"] = [
    tuple(sorted([a.lower(), b.lower()]))
    for a, b in zip(candidates["uniprot_id1"], candidates["uniprot_id2"])
]

available_homodimers = {
    p[0] for p in set(candidates["pair_key"]) if p[0] == p[1]
}
available_heteropairs = {
    p for p in set(candidates["pair_key"]) if p[0] != p[1]
}

# --- Check availability (compare in lowercase) ---------------------------
required_homodimer_set = {uid.lower() for uid in required_homodimers}
required_heteropair_set = {tuple(sorted([a.lower(), b.lower()]))
                           for a, b in required_heteropairs}

missing_homo = required_homodimer_set - available_homodimers
missing_hetero = required_heteropair_set - available_heteropairs

if missing_homo:
    print("ERROR: missing homodimer predictions in pooled-PPI DB:")
    for uid in sorted(missing_homo):
        print(f"  {uid} (self-pair {uid}-{uid} not found)")
    sys.exit(1)

if missing_hetero:
    print("ERROR: missing heterodimer predictions in pooled-PPI DB:")
    for m in sorted(missing_hetero):
        print(f"  {m[0]} - {m[1]}")
    print("Cannot run CombFold without these. "
          "Generate them with AF-Multimer / AF3 / Boltz first.")
    sys.exit(1)

# --- Write PDBs (all from pooled-PPI DB) ---------------------------------
# Map lowercase -> original uppercase for output filenames
lower_to_orig = {p.lower(): p for p in proteins}

print("Writing homodimers from pooled-PPI DB:")
for uid in sorted(required_homodimer_set):
    pair = (uid, uid)
    row = candidates.query("pair_key == @pair").iloc[0]
    db_id1 = f'{row["name"]}_{row.af3_id1}'
    db_id2 = f'{row["name"]}_{row.af3_id2}'
    orig = lower_to_orig[uid]
    path_pdb = pdbs_dir / f"AFM_{orig}_{orig}_unrelaxed_rank_1_model_1.pdb"
    pp.save_ids([db_id1, db_id2], str(path_pdb))
    print(f"  written: {path_pdb}")

print("Writing heterodimers from pooled-PPI DB:")
for pair in sorted(required_heteropair_set):
    row = candidates.query("pair_key == @pair").iloc[0]
    db_id1 = f'{row["name"]}_{row.af3_id1}'
    db_id2 = f'{row["name"]}_{row.af3_id2}'
    orig1 = lower_to_orig[pair[0]]
    orig2 = lower_to_orig[pair[1]]
    path_pdb = pdbs_dir / (
        f"AFM_{orig1}_{orig2}_"
        f"unrelaxed_rank_1_model_1.pdb"
    )
    pp.save_ids([db_id1, db_id2], str(path_pdb))
    print(f"  written: {path_pdb}")

print("Done writing PDBs.")
EOF

# ---- Step 2: Run CombFold via Singularity ------------------
echo "[$(date)] Running CombFold for ${COMPLEX_NAME}..."

singularity exec \
    --bind "${OUTPUT_BASE}" \
    "${COMBFOLD_SIF}" \
    sh -c "python /app/CombFold/scripts/run_on_pdbs.py \
    ${SUBUNITS_JSON} \
    ${PDBS_DIR} \
    ${OUTPUT_DIR}"

echo "[$(date)] CombFold finished. Output in: ${OUTPUT_DIR}"
