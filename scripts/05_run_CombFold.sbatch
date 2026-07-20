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
# sbatch run_CombFold.sbatch 'Q12186(1),P36084(2)' pool
# sbatch run_CombFold.sbatch 'Q12186(1),P36084(2),1:2' pair   # ratio validated
#
# Args:
#   1) Single quoted spec: comma-separated NAME(COUNT) entries,
#      optional trailing ratio (e.g. 1:2) matching the parenthesized
#      counts in the order the proteins were written.
#   2) SOURCE: 'pair' or 'pool' - which AF3 input_type to pull the
#      best-ranked (max ranking_score) pairwise model from.
#
# Works for any stoichiometry, including homomultimers (NAME(N>=2)):
# CombFold only needs ONE pairwise dimer structure per unique
# (protein, protein) contact - reused across however many copies the
# final assembly has.
#
# Pair PDB source: pooled-PPI summary_models.parquet, filtered to
# input_type == SOURCE, best model per pair by ranking_score.
# ============================================================

set -euo pipefail

# --- Config -------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# Homodimers: only available in the 26.07 snapshot, with a pair/pool distinction.
POOLED_PPI_DB_HOMO="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
# Heterodimers: only available in the older 26.04 snapshot (no pair/pool
# distinction there - every entry is pool-derived), 26.07 has far fewer.
POOLED_PPI_DB_HETERO="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.04"
SEQUENCES_PARQUET="${POOLED_PPI_DB_HOMO}/proteins.parquet"
COMBFOLD_SIF="/cluster/project/beltrao/shared/alphafold3/images/combfold_latest.sif"
OUTPUT_BASE="/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/first_setup/CombFold"
# ------------------------------------------------------------

source "${VENV}/bin/activate"

if [[ $# -ne 2 ]]; then
    echo "ERROR: expected exactly two arguments (spec, source)."
    echo "Usage: sbatch $(basename "$0") 'Q12186(1),P36084(2)' pool"
    echo " or: sbatch $(basename "$0") 'Q12186(1),P36084(2),1:2' pair"
    exit 1
fi

SPEC="$1"
SOURCE="$2"
if [[ "${SOURCE}" != "pair" && "${SOURCE}" != "pool" ]]; then
    echo "ERROR: source must be 'pair' or 'pool', got '${SOURCE}'."
    exit 1
fi
echo "Spec: ${SPEC}"
echo "Source: ${SOURCE}"

mkdir -p logs

export SPEC SOURCE OUTPUT_BASE SEQUENCES_PARQUET POOLED_PPI_DB_HOMO POOLED_PPI_DB_HETERO

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

# Source-suffixed dirs, so 'pair' and 'pool' runs never collide and can run concurrently
RUN_NAME="${COMPLEX_NAME}_${SOURCE}"
INPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_input"
PDBS_DIR="${OUTPUT_BASE}/${RUN_NAME}_input/pdbs/"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_output/"
SUBUNITS_JSON="${INPUT_DIR}/subunits.json"

mkdir -p "${INPUT_DIR}/pdbs" "${OUTPUT_DIR}"

echo "Complex name: ${COMPLEX_NAME}"
echo "Run name: ${RUN_NAME}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

# ---- Step 1: Validate spec, write subunits.json + pairwise PDBs ----
echo "[$(date)] Parsing spec, writing subunits.json and pairwise PDBs (source=${SOURCE})..."

python - <<'EOF'
import collections
import io
import itertools
import json
import os
import re
import sys
from functools import reduce
from math import gcd
from pathlib import Path

import polars as pl
import pandas as pd
import Bio.PDB
import foldcomp
import af3io

spec = os.environ["SPEC"]
source = os.environ["SOURCE"]
output_base = os.environ["OUTPUT_BASE"]
sequences_path = os.environ["SEQUENCES_PARQUET"]
pooled_ppi_db_homo = os.environ["POOLED_PPI_DB_HOMO"]
pooled_ppi_db_hetero = os.environ["POOLED_PPI_DB_HETERO"]

FC_DB_HOMO = os.path.join(pooled_ppi_db_homo, "predictions-db", "predictions-db")
FC_DB_HETERO = os.path.join(pooled_ppi_db_hetero, "predictions-db", "predictions-db")

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

input_dir = Path(output_base) / f"{complex_name}_{source}_input"
pdbs_dir = input_dir / "pdbs"
input_dir.mkdir(parents=True, exist_ok=True)
pdbs_dir.mkdir(parents=True, exist_ok=True)

# --- Write subunits.json (source-independent: sequence only) -------------
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

# --- Required pairwise dimers: homodimers (count>=2) + heteropairs -------
# Same list for any stoichiometry - a homomultimer NAME(N>=2) still only
# needs ONE NAME-NAME dimer structure, reused across all N copies by CombFold.
required_homo = []    # list of (lo, lo), lowercase - only 26.07 has these
required_hetero = []  # list of (lo, hi), lowercase - only 26.04 has these
for i, p1 in enumerate(proteins):
    for p2 in proteins[i:]:
        if p1 == p2:
            if counts[p1] >= 2:
                required_homo.append((p1.lower(), p1.lower()))
        else:
            required_hetero.append(tuple(sorted([p1.lower(), p2.lower()])))

print(f"Required homodimers ({len(required_homo)}): {required_homo}")
print(f"Required heteropairs ({len(required_hetero)}): {required_hetero}")

lower_to_orig = {p.lower(): p for p in proteins}

# --- save_ids equivalent: merge two foldcomp chain-structures into one PDB ---
def save_pair_pdb(fc_db: str, fc_key1: str, fc_key2: str, out_path: Path) -> None:
    parser = Bio.PDB.PDBParser(QUIET=True)
    struct0 = None
    with foldcomp.open(fc_db, ids=[fc_key1, fc_key2]) as db:
        for index, ((name, pdb), chain_id) in enumerate(
            itertools.islice(zip(db, af3io.input.enumerate_chains()), None)
        ):
            struct = parser.get_structure(index, io.StringIO(pdb))
            if index == 0:
                struct0 = struct
            else:
                chain0 = next(struct[0].get_chains())
                chain0.id = chain_id
                struct0[0].add(chain0)
    pdbio = Bio.PDB.PDBIO()
    pdbio.set_structure(struct0)
    pdbio.save(str(out_path))

missing_any = False

# --- Homodimers: 26.07 summary_models.parquet, pair/pool selectable via SOURCE ---
if required_homo:
    required_homo_keys = {f"{lo}__{hi}" for lo, hi in required_homo}
    best_homo = (
        pl.scan_parquet(os.path.join(pooled_ppi_db_homo, "summary_models.parquet"))
        .with_columns(
            (pl.min_horizontal("af3_id1", "af3_id2") + "__" +
             pl.max_horizontal("af3_id1", "af3_id2")).alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(required_homo_keys))
        .filter(pl.col("input_type") == source)
        .sort("ranking_score", descending=True)
        .unique(subset="pair_id", keep="first")
        .select("pair_id", "input_name", "sample", "chain_id1", "chain_id2",
                "ranking_score", "chain_pair_iptm")
        .collect()
    )
    found = set(best_homo["pair_id"].to_list())
    missing = required_homo_keys - found
    if missing:
        missing_any = True
        print(f"ERROR: missing homodimer '{source}' predictions in 26.07 DB:")
        for key in sorted(missing):
            lo, _ = key.split("__")
            print(f"  {lower_to_orig.get(lo, lo)} (self-pair not found)")
    else:
        print(f"Writing homodimer PDBs from '{source}' models (26.07):")
        for row in best_homo.iter_rows(named=True):
            lo, hi = row["pair_id"].split("__")
            orig1, orig2 = lower_to_orig[lo], lower_to_orig[hi]
            fc_key1 = f'{row["input_name"]}_{row["sample"]}_{row["chain_id1"]}'
            fc_key2 = f'{row["input_name"]}_{row["sample"]}_{row["chain_id2"]}'
            path_pdb = pdbs_dir / f"AFM_{orig1}_{orig2}_unrelaxed_rank_1_model_1.pdb"
            save_pair_pdb(FC_DB_HOMO, fc_key1, fc_key2, path_pdb)
            print(f"  written: {path_pdb}  (ranking_score={row['ranking_score']:.3f}, "
                  f"iptm={row['chain_pair_iptm']:.3f})")

# --- Heterodimers: 26.04 summary_pairs.parquet - pool-derived only, no  ---
# --- pair/pool distinction there, so SOURCE doesn't change this lookup. ---
if required_hetero:
    if source == "pair":
        print("NOTE: heterodimers have no direct pairwise ('pair') predictions "
              "in the 26.04 DB - using the pooled predictions there regardless "
              "of --paired-model.")
    required_hetero_keys = {f"{lo}__{hi}" for lo, hi in required_hetero}
    best_hetero = (
        pl.scan_parquet(os.path.join(pooled_ppi_db_hetero, "summary_pairs.parquet"))
        .with_columns(
            (pl.min_horizontal("af3_id1", "af3_id2") + "__" +
             pl.max_horizontal("af3_id1", "af3_id2")).alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(required_hetero_keys))
        .sort("chain_pair_iptm_best_corrected", descending=True)
        .unique(subset="pair_id", keep="first")
        .select("pair_id", "name", "af3_id1", "af3_id2", "chain_pair_iptm_best_corrected")
        .collect()
    )
    found = set(best_hetero["pair_id"].to_list())
    missing = required_hetero_keys - found
    if missing:
        missing_any = True
        print("ERROR: missing heterodimer predictions in 26.04 DB:")
        for key in sorted(missing):
            lo, hi = key.split("__")
            print(f"  {lower_to_orig.get(lo, lo)} - {lower_to_orig.get(hi, hi)}")
        print("Cannot run CombFold without these. "
              "Generate them with AF-Multimer / AF3 / Boltz first.")
    else:
        print("Writing heterodimer PDBs from pooled predictions (26.04):")
        for row in best_hetero.iter_rows(named=True):
            lo, hi = row["pair_id"].split("__")
            orig1, orig2 = lower_to_orig[lo], lower_to_orig[hi]
            fc_key1 = f'{row["name"]}_{row["af3_id1"]}'
            fc_key2 = f'{row["name"]}_{row["af3_id2"]}'
            path_pdb = pdbs_dir / f"AFM_{orig1}_{orig2}_unrelaxed_rank_1_model_1.pdb"
            save_pair_pdb(FC_DB_HETERO, fc_key1, fc_key2, path_pdb)
            print(f"  written: {path_pdb}  (iptm_best_corrected="
                  f"{row['chain_pair_iptm_best_corrected']:.3f})")

if missing_any:
    sys.exit(1)
EOF

# ---- Step 2: Run CombFold via Singularity ------------------
echo "[$(date)] Running CombFold for ${RUN_NAME}..."

singularity exec \
    --bind "${OUTPUT_BASE}" \
    "${COMBFOLD_SIF}" \
    sh -c "python /app/CombFold/scripts/run_on_pdbs.py \
    ${SUBUNITS_JSON} \
    ${PDBS_DIR} \
    ${OUTPUT_DIR}"

echo "[$(date)] CombFold finished. Output in: ${OUTPUT_DIR}"
