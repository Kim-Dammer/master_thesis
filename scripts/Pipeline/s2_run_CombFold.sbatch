#!/bin/bash
#SBATCH --account es_biol
#SBATCH --partition es_biol
#SBATCH --job-name=combfold
#SBATCH --output=logs/combfold_%j.out
#SBATCH --error=logs/combfold_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=7G
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
# best-ranked (max ranking_score) pairwise model from. Applies to
# HOMODIMERS ONLY; heteropairs are always taken from pool (see below).
#
# Works for any stoichiometry, including homomultimers (NAME(N>=2)):
# CombFold only needs ONE pairwise dimer structure per unique
# (protein, protein) contact - reused across however many copies the
# final assembly has.
#
# Pair PDB source (single 26.07 snapshot):
#   homodimers -> summary_models.parquet (per-sample), filtered to
#     input_type == SOURCE (pair/pool) with cross-source fallback,
#     best model per pair by ranking_score.
#   heteropairs -> summary_models.parquet (per-sample), filtered to
#     input_type == 'pool' ONLY (the SOURCE flag is ignored for heteropairs;
#     the small isolated two-body 'pair' heteropair slice is deliberately
#     excluded), best model per pair by ranking_score.
#   The loaded sample's own per-sample PAE (chain_pair_pae_min_{min,max,mean})
#   is recorded per contact in <run_name>_pairs.csv.
# ============================================================

set -euo pipefail

# --- Config -------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# Single unified snapshot (26.07) for BOTH homodimers and heteropairs.
# Both read summary_models.parquet (per-sample), ranked by ranking_score.
# Homodimers honor SOURCE (pair/pool) with cross-source fallback;
# heteropairs are pool-only (SOURCE ignored).
POOLED_PPI_DB="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
SEQUENCES_PARQUET="${POOLED_PPI_DB}/proteins.parquet"
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

export SPEC SOURCE OUTPUT_BASE SEQUENCES_PARQUET POOLED_PPI_DB

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
import string
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
pooled_ppi_db = os.environ["POOLED_PPI_DB"]
job_id = os.environ.get("SLURM_JOB_ID", "unknown")

# Single 26.07 foldcomp DB for both homodimers and heteropairs.
FC_DB = os.path.join(pooled_ppi_db, "predictions-db", "predictions-db")

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

# CombFold requires SINGLE-character chain IDs in subunits.json (its
# copy_pdb_rename_chain() asserts len(chain)==1). af3io.enumerate_chains()
# rolls over to two-char names ('AA', 'AB', ...) past 'Z', which crashes
# CombFold deep inside the container for any complex with >26 total chains.
# Use a fixed 62-char single-character alphabet instead, and fail early with a
# clear message if a complex needs more chains than that.
_CHAIN_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits  # 62 single chars

def _single_char_chains():
    for _c in _CHAIN_ALPHABET:
        yield _c

total_chains = sum(counts[p] for p in proteins)
if total_chains > len(_CHAIN_ALPHABET):
    print(
        f"ERROR: complex {complex_name} needs {total_chains} chains, but only "
        f"{len(_CHAIN_ALPHABET)} single-character chain IDs are supported by CombFold."
    )
    sys.exit(1)

js = collections.OrderedDict()
chain_iter = _single_char_chains()
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
required_hetero = []  # list of (lo, hi), lowercase - from 26.07 summary_models
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
        pl.scan_parquet(os.path.join(pooled_ppi_db, "summary_models.parquet"))
        .with_columns(
            (pl.min_horizontal(["af3_id1", "af3_id2"]) + "__" +
             pl.max_horizontal(["af3_id1", "af3_id2"])).alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(required_homo_keys))
        .filter(pl.col("input_type").is_in([source, fallback_source]))
        # FIX: Pull valid tracking columns instead of non-existent db_ids.
        # PAE columns are per-sample (this table is per-sample), so they describe
        # the exact loaded model - recorded in the per-complex pairs CSV.
        .select("pair_id", "input_type", "input_name", "sample", "chain_id1", "chain_id2",
                "ranking_score", "chain_pair_iptm",
                "chain_pair_pae_min_min", "chain_pair_pae_min_max", "chain_pair_pae_min_mean")
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
                if save_pair_pdb(FC_DB, fc_key1, fc_key2, path_pdb):
                    tag = "" if try_type == source else f" [FALLBACK: requested '{source}', used '{try_type}']"
                    print(f" written: {path_pdb} (type={try_type}, sample={row['sample']}, "
                          f"ranking_score={row['ranking_score']:.3f}, "
                          f"iptm={row['chain_pair_iptm']:.3f}){tag}")
                    written = True
                    source_log_rows.append({
                        "job_id": job_id, "run_name": run_name, "spec": spec,
                        "pair_type": "homo", "protein1": orig1, "protein2": orig2,
                        "requested_source": source, "used_source": try_type,
                        "input_type": row["input_type"], "input_name": row["input_name"],
                        "sample": row["sample"], "ranking_score": row["ranking_score"],
                        "chain_pair_iptm": row["chain_pair_iptm"],
                        "chain_pair_pae_min_min": row["chain_pair_pae_min_min"],
                        "chain_pair_pae_min_max": row["chain_pair_pae_min_max"],
                        "chain_pair_pae_min_mean": row["chain_pair_pae_min_mean"],
                        "status": "written",
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
                "input_type": None, "input_name": None,
                "sample": None, "ranking_score": None, "chain_pair_iptm": None,
                "chain_pair_pae_min_min": None, "chain_pair_pae_min_max": None,
                "chain_pair_pae_min_mean": None, "status": "missing",
            })

    if missing:
        missing_any = True

# --- Heteropairs: 26.07 summary_models.parquet, POOL ONLY ---------------------
# Heteropairs are drawn ONLY from pooled predictions (input_type == 'pool'),
# regardless of the SOURCE flag (which governs homodimers). Rationale: a
# heteropair extracted from the large pooled prediction is a consistent object
# for calibration, whereas the small isolated two-body 'pair' heteropair slice
# is a different prediction regime and is deliberately excluded here. We still
# read summary_models.parquet at the per-sample level, rank by ranking_score
# (same metric as homodimers), and load the exact selected sample via the
# foldcomp key <input_name>_<sample>_<chain_id>, so the recorded PAE describes
# the model actually assembled.
if required_hetero:
    required_hetero_keys = {f"{lo}__{hi}" for lo, hi in required_hetero}
    candidates_hetero = (
        pl.scan_parquet(os.path.join(pooled_ppi_db, "summary_models.parquet"))
        .with_columns(
            (pl.min_horizontal(["af3_id1", "af3_id2"]) + "__" +
             pl.max_horizontal(["af3_id1", "af3_id2"])).alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(required_hetero_keys))
        .filter(pl.col("input_type") == "pool")
        .select("pair_id", "input_type", "input_name", "sample", "chain_id1", "chain_id2",
                "ranking_score", "chain_pair_iptm",
                "chain_pair_pae_min_min", "chain_pair_pae_min_max", "chain_pair_pae_min_mean")
        .collect()
    )

    missing = set()
    for key in sorted(required_hetero_keys):
        lo, hi = key.split("__")
        orig1, orig2 = lower_to_orig[lo], lower_to_orig[hi]
        path_pdb = pdbs_dir / f"AFM_{orig1}_{orig2}_unrelaxed_rank_1_model_1.pdb"
        written = False

        ranked = (
            candidates_hetero.filter(pl.col("pair_id") == key)
            .sort("ranking_score", descending=True)
        )
        for row in ranked.iter_rows(named=True):
            fc_key1 = f"{row['input_name']}_{row['sample']}_{row['chain_id1']}"
            fc_key2 = f"{row['input_name']}_{row['sample']}_{row['chain_id2']}"
            if save_pair_pdb(FC_DB, fc_key1, fc_key2, path_pdb):
                print(f" written: {path_pdb} (type=pool, sample={row['sample']}, "
                      f"ranking_score={row['ranking_score']:.3f}, "
                      f"iptm={row['chain_pair_iptm']:.3f})")
                written = True
                source_log_rows.append({
                    "job_id": job_id, "run_name": run_name, "spec": spec,
                    "pair_type": "hetero", "protein1": orig1, "protein2": orig2,
                    "requested_source": "pool", "used_source": "pool",
                    "input_type": row["input_type"], "input_name": row["input_name"],
                    "sample": row["sample"], "ranking_score": row["ranking_score"],
                    "chain_pair_iptm": row["chain_pair_iptm"],
                    "chain_pair_pae_min_min": row["chain_pair_pae_min_min"],
                    "chain_pair_pae_min_max": row["chain_pair_pae_min_max"],
                    "chain_pair_pae_min_mean": row["chain_pair_pae_min_mean"],
                    "status": "written",
                })
                break
            print(f" WARNING: pool sample {row['sample']} structure missing from "
                  f"foldcomp DB for {orig1} - {orig2} - trying next-best...")

        if not written:
            print(f"ERROR: no 'pool' structure found in "
                  f"26.07 foldcomp DB for {orig1} - {orig2}.")
            missing.add(key)
            source_log_rows.append({
                "job_id": job_id, "run_name": run_name, "spec": spec,
                "pair_type": "hetero", "protein1": orig1, "protein2": orig2,
                "requested_source": "pool", "used_source": "MISSING",
                "input_type": None, "input_name": None,
                "sample": None, "ranking_score": None, "chain_pair_iptm": None,
                "chain_pair_pae_min_min": None, "chain_pair_pae_min_max": None,
                "chain_pair_pae_min_mean": None, "status": "missing",
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

# --- Per-complex pairs CSV: one row per REQUIRED contact (incl. missing) -----
# Records exactly which single model CombFold used for each pairwise contact and
# that model's own confidence metrics. Columns are a stable subset/reorder of
# source_log_rows; PAE values are per-sample (they describe the loaded model).
pairs_csv_cols = [
    "run_name", "spec", "pair_type", "protein1", "protein2",
    "requested_source", "used_source", "input_type", "input_name", "sample",
    "ranking_score", "chain_pair_iptm",
    "chain_pair_pae_min_min", "chain_pair_pae_min_max", "chain_pair_pae_min_mean",
    "status",
]
pairs_csv_path = log_dir / f"{run_name}_pairs.csv"
if source_log_rows:
    pl.DataFrame(source_log_rows).select(pairs_csv_cols).write_csv(pairs_csv_path)
    print(f"Per-complex pairs CSV written: {pairs_csv_path}")
else:
    print("No pairs to write to per-complex CSV.")

# Break down the summary by pair_type. Homodimers honor the requested SOURCE
# with cross-source fallback; heteropairs are pool-only (no source choice,
# no fallback), so they are reported separately.
n_homo = sum(1 for r in source_log_rows if r["pair_type"] == "homo")
n_homo_used_source = sum(1 for r in source_log_rows
                         if r["pair_type"] == "homo" and r["used_source"] == source)
n_fallback_homo = sum(1 for r in source_log_rows
                      if r["pair_type"] == "homo" and r["requested_source"] == source
                      and r["used_source"] not in (source, "MISSING"))
n_hetero = sum(1 for r in source_log_rows if r["pair_type"] == "hetero")
n_hetero_pool = sum(1 for r in source_log_rows
                    if r["pair_type"] == "hetero" and r["used_source"] == "pool")
n_missing = sum(1 for r in source_log_rows if r["used_source"] == "MISSING")
print(f"SUMMARY (job {job_id}): "
      f"homodimers={n_homo} (requested '{source}': {n_homo_used_source} used '{source}', "
      f"{n_fallback_homo} fell back); "
      f"heteropairs={n_hetero} (pool-only: {n_hetero_pool} used pool); "
      f"{n_missing} missing.")

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