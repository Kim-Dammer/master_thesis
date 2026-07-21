#!/bin/bash
#SBATCH --account es_biol
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
# sbatch run_CombFold.sbatch 'Q12186(1),P36084(2),1:2' pair # ratio validated
#
# Args:
# 1) Single quoted spec: comma-separated NAME(COUNT) entries,
# optional trailing ratio (e.g. 1:2) matching the parenthesized
# counts in the order the proteins were written.
# 2) SOURCE: 'pair' or 'pool' - which AF3 input_type to pull the
# best-ranked (max ranking_score) pairwise model from.
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
echo "[$(date)] Parsing spec, writing subunits.json and pairwise PDBs (source=${SOURCE})...."

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
job_id = os.environ.get("SLURM_JOB_ID", "unknown")

FC_DB_HOMO = os.path.join(pooled_ppi_db_homo, "predictions-db", "predictions-db")
FC_DB_HETERO = os.path.join(pooled_ppi_db_hetero, "predictions-db", "predictions-db")

# One row per required pairwise dimer, recording exactly what was
# requested vs. what was actually used - written out as a per-job CSV
# and summarized at the end of this run.
source_log_rows = []

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
            print(f"ERROR: spec contains more than one ratio token ('{ratio_token}' and '{tok}')")
            sys.exit(1)
        ratio_token = tok
    else:
        print(f"ERROR: cannot parse token '{tok}'. Expected NAME(COUNT) or a ratio like 1:2.")
        sys.exit(1)

if not counts:
    print("ERROR: spec contains no protein entries.")
    sys.exit(1)

# --- Validate optional ratio against parenthesized counts ----------------
if ratio_token is not None:
    ratio_parts = [int(x) for x in ratio_token.split(":")]
    listed_counts = list(counts.values())  # order proteins were written

    if len(ratio_parts) != len(listed_counts):
        print(f"ERROR: ratio '{ratio_token}' has {len(ratio_parts)} parts but spec has {len(listed_counts)} proteins.")
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
        print(f" Counts from parentheses: {counts_str} (reduced {':'.join(map(str, listed_reduced))})")
        print(f" Ratio token: {ratio_str} (reduced {':'.join(map(str, ratio_reduced))})")
        sys.exit(1)

    print(f"Ratio check OK: {ratio_token} matches counts {':'.join(str(c) for c in listed_counts)}.")

# Sort by UniProt ID for stable output naming downstream
proteins = sorted(counts)
counts = collections.OrderedDict((p, counts[p]) for p in proteins)
complex_name = "_".join(f"{p}x{counts[p]}" for p in proteins)
run_name = f"{complex_name}_{source}"
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
# Returns False (instead of crashing) if either entry is missing from the
# foldcomp DB, so the caller can fall back to the next-best sample.
def save_pair_pdb(fc_db: str, fc_key1: str, fc_key2: str, out_path: Path) -> bool:
    parser = Bio.PDB.PDBParser(QUIET=True)
    struct0 = None
    n_found = 0
    with foldcomp.open(fc_db, ids=[fc_key1, fc_key2]) as db:
        for index, ((name, pdb), chain_id) in enumerate(
            itertools.islice(zip(db, af3io.input.enumerate_chains()), None)
        ):
            n_found += 1
            struct = parser.get_structure(index, io.StringIO(pdb))
            if index == 0:
                struct0 = struct
            else:
                chain0 = next(struct[0].get_chains())
                chain0.id = chain_id
                struct0[0].add(chain0)
    if n_found < 2 or struct0 is None:
        return False
    pdbio = Bio.PDB.PDBIO()
    pdbio.set_structure(struct0)
    pdbio.save(str(out_path))
    return True

missing_any = False

# --- Homodimers: 26.07 summary_models.parquet, pair/pool selectable via SOURCE ---
# --- Homodimers: 26.07 summary_models.parquet, pair/pool selectable via SOURCE ---
if required_homo:
    required_homo_keys = {f"{lo}__{hi}" for lo, hi in required_homo}
    fallback_source = "pool" if source == "pair" else "pair"
    # Keep ALL candidate samples for BOTH input_types (not just the requested
    # source), so we can fall back across sources if the requested one has
    # no structure in the foldcomp DB at all (common for 'pair': only ~5% of
    # homodimer pair-jobs have their structure ingested, even though scores
    # exist for all of them in summary_models.parquet).
    candidates_homo = (
        pl.scan_parquet(os.path.join(pooled_ppi_db_homo, "summary_models.parquet"))
        .with_columns(
            (pl.min_horizontal(["af3_id1", "af3_id2"]) + "__" +
             pl.max_horizontal(["af3_id1", "af3_id2"])).alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(required_homo_keys))
        .filter(pl.col("input_type").is_in([source, fallback_source]))
        # FIX: Pull valid tracking columns instead of non-existent db_ids
        .select("pair_id", "input_type", "input_name", "sample", "chain_id1", "chain_id2",
                "ranking_score", "chain_pair_iptm")
        .collect()
    )

    missing = set()
    for key in sorted(required_homo_keys):
        lo, hi = key.split("__")
        orig1, orig2 = lower_to_orig[lo], lower_to_orig[hi]
        path_pdb = pdbs_dir / f"AFM_{orig1}_{orig2}_unrelaxed_rank_1_model_1.pdb"
        written = False

        for try_type in [source, fallback_source]:
            ranked = (
                candidates_homo.filter(
                    (pl.col("pair_id") == key) & (pl.col("input_type") == try_type)
                ).sort("ranking_score", descending=True)
            )
            if ranked.height == 0:
                continue
            for row in ranked.iter_rows(named=True):
                # FIX: Dynamically build the keys from the correct input layout
                fc_key1 = f"{row['input_name']}_{row['sample']}_{row['chain_id1']}"
                fc_key2 = f"{row['input_name']}_{row['sample']}_{row['chain_id2']}"
                if save_pair_pdb(FC_DB_HOMO, fc_key1, fc_key2, path_pdb):
                    tag = "" if try_type == source else f" [FALLBACK: requested '{source}', used '{try_type}']"
                    print(f" written: {path_pdb} (type={try_type}, sample={row['sample']}, "
                          f"ranking_score={row['ranking_score']:.3f}, "
                          f"iptm={row['chain_pair_iptm']:.3f}){tag}")
                    written = True
                    source_log_rows.append({
                        "job_id": job_id, "run_name": run_name, "spec": spec,
                        "pair_type": "homo", "protein1": orig1, "protein2": orig2,
                        "requested_source": source, "used_source": try_type,
                        "sample": row["sample"], "ranking_score": row["ranking_score"],
                        "chain_pair_iptm": row["chain_pair_iptm"], "status": "written",
                    })
                    break
                print(f" WARNING: {try_type} sample {row['sample']} structure missing from "
                      f"foldcomp DB for {orig1} - trying next-best...")
            if written:
                break

        if not written:
            print(f"ERROR: no '{source}' or fallback '{fallback_source}' structure found in "
                  f"foldcomp DB for {orig1} (self-pair).")
            missing.add(key)
            source_log_rows.append({
                "job_id": job_id, "run_name": run_name, "spec": spec,
                "pair_type": "homo", "protein1": orig1, "protein2": orig2,
                "requested_source": source, "used_source": "MISSING",
                "sample": None, "ranking_score": None,
                "chain_pair_iptm": None, "status": "missing",
            })

    if missing:
        missing_any = True

# --- Heterodimers: 26.04 summary_pairs.parquet - pool-derived only, no ---
# --- pair/pool distinction there, so SOURCE doesn't change this lookup. ---
if required_hetero:
    if source == "pair":
        print("NOTE: heterodimers have no direct pairwise ('pair') predictions "
              "in the 26.04 DB - using the pooled predictions there regardless "
              "of --paired-model.")
    required_hetero_keys = {f"{lo}__{hi}" for lo, hi in required_hetero}
    candidates_hetero = (
        pl.scan_parquet(os.path.join(pooled_ppi_db_hetero, "summary_pairs.parquet"))
        .with_columns(
            (pl.min_horizontal(["af3_id1", "af3_id2"]) + "__" +
             pl.max_horizontal(["af3_id1", "af3_id2"])).alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(required_hetero_keys))
        .select("pair_id", "name", "af3_id1", "af3_id2", "chain_pair_iptm_best_corrected")
        .collect()
    )

    missing = set()
    for key in sorted(required_hetero_keys):
        lo, hi = key.split("__")
        orig1, orig2 = lower_to_orig[lo], lower_to_orig[hi]
        ranked = candidates_hetero.filter(pl.col("pair_id") == key).sort(
            "chain_pair_iptm_best_corrected", descending=True
        )
        if ranked.height == 0:
            print(f"ERROR: no heterodimer predictions in 26.04 DB for {orig1} - {orig2}")
            missing.add(key)
            source_log_rows.append({
                "job_id": job_id, "run_name": run_name, "spec": spec,
                "pair_type": "hetero", "protein1": orig1, "protein2": orig2,
                "requested_source": source, "used_source": "MISSING",
                "sample": None, "ranking_score": None,
                "chain_pair_iptm": None, "status": "missing",
            })
            continue
        path_pdb = pdbs_dir / f"AFM_{orig1}_{orig2}_unrelaxed_rank_1_model_1.pdb"
        written = False
        for row in ranked.iter_rows(named=True):
            fc_key1 = f'{row["name"]}_{row["af3_id1"]}'
            fc_key2 = f'{row["name"]}_{row["af3_id2"]}'
            if save_pair_pdb(FC_DB_HETERO, fc_key1, fc_key2, path_pdb):
                print(f" written: {path_pdb} (iptm_best_corrected="
                      f"{row['chain_pair_iptm_best_corrected']:.3f})")
                written = True
                source_log_rows.append({
                    "job_id": job_id, "run_name": run_name, "spec": spec,
                    "pair_type": "hetero", "protein1": orig1, "protein2": orig2,
                    "requested_source": source, "used_source": "pool",
                    "sample": None, "ranking_score": None,
                    "chain_pair_iptm": row["chain_pair_iptm_best_corrected"], "status": "written",
                })
                break
            print(f" WARNING: structure missing from foldcomp DB for {orig1} - {orig2} "
                  f"pool '{row['name']}' - trying next-best pool...")
        if not written:
            print(f"ERROR: all {ranked.height} candidate structures for {orig1} - {orig2} "
                  f"are missing from the foldcomp DB.")
            missing.add(key)
            source_log_rows.append({
                "job_id": job_id, "run_name": run_name, "spec": spec,
                "pair_type": "hetero", "protein1": orig1, "protein2": orig2,
                "requested_source": source, "used_source": "MISSING",
                "sample": None, "ranking_score": None,
                "chain_pair_iptm": None, "status": "missing",
            })

    if missing:
        missing_any = True
        print("Cannot run CombFold without these. "
              "Generate them with AF-Multimer / AF3 / Boltz first.")

# --- Per-job source log: exact record of what was requested vs used ------
log_dir = Path(output_base) / "pdb_source_logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"{run_name}_{job_id}.csv"
# FIX #3: only claim the log was written if we actually wrote it.
if source_log_rows:
    pl.DataFrame(source_log_rows).write_csv(log_path)
    print(f"Source log written: {log_path}")
else:
    print("No source log rows to write.")

# FIX #2: break down the summary by pair_type so homodimer fallbacks
# (genuine source substitutions) are distinguishable from heterodimers
# (which are always pool, regardless of --paired-model).
n_requested_source = sum(1 for r in source_log_rows if r["requested_source"] == source)
n_used_source = sum(1 for r in source_log_rows if r["used_source"] == source)
n_fallback_homo = sum(1 for r in source_log_rows
                      if r["pair_type"] == "homo" and r["requested_source"] == source
                      and r["used_source"] not in (source, "MISSING"))
n_hetero_pool = sum(1 for r in source_log_rows
                    if r["pair_type"] == "hetero" and r["used_source"] == "pool")
n_missing = sum(1 for r in source_log_rows if r["used_source"] == "MISSING")
print(f"SUMMARY (job {job_id}): {n_requested_source} pairs requested '{source}'; "
      f"{n_used_source} used '{source}', {n_fallback_homo} homodimer fell back, "
      f"{n_hetero_pool} heterodimer (always pool), {n_missing} missing.")

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