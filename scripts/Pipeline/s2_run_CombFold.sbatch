#!/bin/bash
#SBATCH --account es_biol
#SBATCH --partition es_biol
#SBATCH --job-name=combfold
#SBATCH --output=logs/combfold_%j.out
#SBATCH --error=logs/combfold_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
# --time and --mem-per-cpu were increased vs. the single-model version.
# Step 1 now fetches/decompresses up to 5x as many SQLite blobs, and Step 2's
# CombFold hierarchical beam search has up to 5x as many candidate rigid-body
# transforms per required pairwise contact to score. Re-calibrate both with
# `seff <jobid>` on a handful of small/medium/large (by #chains) complexes
# before trusting these defaults at scale -- see TOP_N_MODELS below.

# ============================================================
# CombFold assembly script
#
# Usage (CLI is UNCHANGED -- same 2/3 positional args as before):
# sbatch run_CombFold.sbatch 'Q12186(1),P36084(2)' pool
# sbatch run_CombFold.sbatch 'Q12186(1),P36084(2),1:2' pair # ratio validated
# sbatch run_CombFold.sbatch 'Q12186(1),P36084(2)' pool 5   # explicit top-N
#
# METRIC is NOT a positional arg -- it's read from the environment (set via
# `sbatch --export=ALL,METRIC=<metric> ...`, which is what
# 01_combfold_with_stoic.py's --metric flag does). Falls back to 'iptm' if
# unset (e.g. a manual sbatch call with no --export). One of:
#     ranking_score  - AF3's own ranking_score column (higher = better)
#     pae            - chain_pair_pae_min (lower = better)
#     iptm           - chain_pair_iptm_corrected (higher = better)
#     ptm_avg        - mean of the two chains' chain_ptm (higher = better)
# Ties (equal metric value) are broken by the LOWEST `sample` index -- see
# rank_candidates() below.
#
# Args:
# 1) Single quoted spec: comma-separated NAME(COUNT) entries,
# optional trailing ratio (e.g. 1:2) matching the parenthesized
# counts in the order the proteins were written.
# 2) SOURCE: 'pair' or 'pool' - which AF3 input_type to pull the
# top-ranked pairwise model(s) from. Applies to BOTH homodimers and
# heteropairs, STRICTLY: whichever input_type is requested is the only one
# ever used, for every pair. There is no cross-source fallback -- if SOURCE
# has fewer than TOP_N_MODELS (or zero) models for a given pair, that pair
# gets fewer models (or is reported missing); it never pulls from the
# other input_type to fill the gap.
# 3) [optional] TOP_N_MODELS: how many top-METRIC AF3 samples
# to supply per required pairwise contact (default 1). Uses however many
# are actually available (from SOURCE only) if fewer than N exist for a
# given pair.
#
# RUN_NAME / output-dir naming does NOT include METRIC -- it stays exactly
# {complex_name}_{source}, same as before this change. Different --metric
# runs are kept apart because 01_combfold_with_stoic.py's --out-dir (and
# therefore OUTPUT_BASE) is metric-specific for each run you submit -- NOT
# because this script encodes the metric into any path. If you ever point
# two different --metric runs at the SAME --out-dir, they WILL collide and
# silently overwrite each other's input/output directories.
#
# Works for any stoichiometry, including homomultimers (NAME(N>=2)):
# CombFold only needs pairwise dimer structure(s) per unique
# (protein, protein) contact - reused across however many copies the
# final assembly has.
#
# Multiple models per pair:
#   Up to TOP_N_MODELS distinct AF3 samples per pair, all from the single
#   requested SOURCE, are written as separate PDBs
#   (AFM_{p1}_{p2}_unrelaxed_rank_{k}_model_{k}.pdb, k=1..written_count,
#   ordered best-to-worst by METRIC). This does NOT change CombFold's
#   scoring (still interface-pLDDT from B-factors -- METRIC here is ONLY
#   used to choose which AF3 samples to hand to CombFold). ranking_score,
#   chain_pair_iptm_corrected, chain_pair_pae_min and (when METRIC is
#   ptm_avg) ptm_avg are all still recorded per model in the pairs
#   CSV/source log for reference regardless of which one was used to
#   pick/order the models.
#
# Pair PDB source (single 26.08 snapshot):
#   homodimers -> summary_models.parquet (per-sample), filtered to
#     input_type == SOURCE (pair/pool) ONLY, no fallback, top-N models
#     per pair by METRIC.
#   heteropairs -> summary_models.parquet (per-sample), filtered to
#     input_type == SOURCE ONLY, top-N models per pair by METRIC.
#   Each loaded sample's own per-sample PAE (chain_pair_pae_min, chain_pair_pae_min_recap)
#   is recorded per model in <run_name>_pairs.csv (one row per model now,
#   not one row per contact -- see model_rank column).
#
#   Structures themselves are fetched from a single SQLite DB
#   (models.sqlite: table 'models', columns db_id/fcz, foldcomp-compressed)
#   for BOTH 'pair' and 'pool' input_type rows. 
# ============================================================

set -euo pipefail

# --- Config -------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# Single unified snapshot (26.08) for BOTH homodimers and heteropairs.
# Both strictly honor SOURCE (pair/pool) with NO cross-source fallback.
POOLED_PPI_DB="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08"
SEQUENCES_PARQUET="${POOLED_PPI_DB}/proteins.parquet"
COMBFOLD_SIF="/cluster/project/beltrao/shared/alphafold3/images/combfold_latest.sif"
OUTPUT_BASE="/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/first_setup/CombFold"

# METRIC: read from environment (set via `sbatch --export=ALL,METRIC=...`,
# which is what 01_combfold_with_stoic.py's --metric flag does). Defaults to
# 'iptm' if not exported at all (e.g. a manual sbatch call).
METRIC="${METRIC:-iptm}"

# PLDDT_TRIM_CUTOFF: trim low-confidence termini of every chain in every
# pairwise PDB before handing it to CombFold. From each end, residues are
# removed while their CA pLDDT (B-factor) is < cutoff; trimming stops at the
# first residue whose CA pLDDT >= cutoff. Interior low-pLDDT regions are kept.
# Residue numbering is NOT changed, so residues still line up with the
# full-length sequence in subunits.json. Set to 0 to disable trimming.
# Read from environment so it can be overridden via sbatch --export.
PLDDT_TRIM_CUTOFF="${PLDDT_TRIM_CUTOFF:-50}"
# 1 = also write the untrimmed pair PDBs to <run>_input/pdbs_untrimmed/
SAVE_UNTRIMMED_PDBS="${SAVE_UNTRIMMED_PDBS:-1}"
# INTERFACE_KEEP_DIST (Angstrom): second stop criterion for the terminal trim.
# Trimming from an end also stops at the first residue whose CA is within
# this distance of ANY CA of the partner chain, even if its pLDDT is
# below PLDDT_TRIM_CUTOFF. That residue and everything inward of it is kept.
# Distances are measured against the partner's UNTRIMMED chain, so the result
# does not depend on which chain is trimmed first. Set to 0 to disable
# (pure pLDDT trimming).
INTERFACE_KEEP_DIST="${INTERFACE_KEEP_DIST:-8}"
# CROP_SUBUNITS (1/0): when trimming is on, the sequence of each protein in
# subunits.json is cropped to the union of the residues kept across all of that
# protein's pair models, with start_res = first kept residue (original numbering
# preserved). CombFold then builds the assembly from the cropped representatives,
# so the output contains only the trimmed part of each protein.
# 0 = subunits.json keeps the full-length sequences.
CROP_SUBUNITS="${CROP_SUBUNITS:-1}"
# ------------------------------------------------------------

source "${VENV}/bin/activate"

VALID_METRICS=("ranking_score" "pae" "iptm" "ptm_avg")
metric_ok=false
for m in "${VALID_METRICS[@]}"; do
    if [[ "${METRIC}" == "${m}" ]]; then
        metric_ok=true
        break
    fi
done
if [[ "${metric_ok}" != true ]]; then
    echo "ERROR: METRIC must be one of: ${VALID_METRICS[*]}, got '${METRIC}'."
    exit 1
fi

if [[ $# -ne 2 && $# -ne 3 ]]; then
    echo "ERROR: expected two or three arguments (spec, source, [top_n_models])."
    echo "Usage: sbatch $(basename "$0") 'Q12186(1),P36084(2)' pool"
    echo " or: sbatch $(basename "$0") 'Q12186(1),P36084(2),1:2' pair"
    echo " or: sbatch $(basename "$0") 'Q12186(1),P36084(2)' pool 5"
    exit 1
fi

SPEC="$1"
SOURCE="$2"
TOP_N_MODELS="${3:-1}"

if [[ "${SOURCE}" != "pair" && "${SOURCE}" != "pool" ]]; then
    echo "ERROR: source must be 'pair' or 'pool', got '${SOURCE}'."
    exit 1
fi
if ! [[ "${TOP_N_MODELS}" =~ ^[0-9]+$ ]] || [[ "${TOP_N_MODELS}" -lt 1 ]]; then
    echo "ERROR: top_n_models must be a positive integer, got '${TOP_N_MODELS}'."
    exit 1
fi
echo "Spec: ${SPEC}"
echo "Source: ${SOURCE}"
echo "Metric: ${METRIC}"
echo "Top N models per pair: ${TOP_N_MODELS}"
echo "pLDDT terminal trim cutoff: ${PLDDT_TRIM_CUTOFF}"
echo "Interface keep distance: ${INTERFACE_KEEP_DIST} A"
echo "Crop subunits to trimmed region: ${CROP_SUBUNITS}"

mkdir -p logs

export SPEC SOURCE METRIC TOP_N_MODELS OUTPUT_BASE SEQUENCES_PARQUET POOLED_PPI_DB PLDDT_TRIM_CUTOFF SAVE_UNTRIMMED_PDBS INTERFACE_KEEP_DIST CROP_SUBUNITS

# --- Recompute complex name on bash side (no temp file) -----
COMPLEX_NAME=$(python -c "
import os, re, hashlib
spec = os.environ['SPEC']
entry_re = re.compile(r'^([A-Za-z0-9_]+)\((\d+)\)\$')
counts = {}
for tok in [t.strip() for t in spec.split(',') if t.strip()]:
    m = entry_re.match(tok)
    if m:
        counts[m.group(1)] = int(m.group(2))
name = '_'.join(f'{p}x{counts[p]}' for p in sorted(counts))
if len(name) > 200:
    name = name[:180] + '_' + hashlib.sha1(name.encode()).hexdigest()[:12]
print(name)
")

if [[ -z "${COMPLEX_NAME}" ]]; then
    echo "ERROR: could not derive complex name from spec '${SPEC}'."
    exit 1
fi

# RUN_NAME stays exactly {complex_name}_{source} -- NO metric suffix. This is
# what keeps 01_combfold_with_stoic.py's run_name derivation
# (f"{spec_to_complex_name(spec)}_{source}") correct with zero changes there.
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
echo "[$(date)] Parsing spec, writing subunits.json and pairwise PDBs (source=${SOURCE}, metric=${METRIC})...."

python - <<'EOF'
import collections
import io
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
import Bio.SeqUtils
import foldcomp
import sqlite_utils
from sqlite_utils.db import NotFoundError
import af3io

spec = os.environ["SPEC"]
source = os.environ["SOURCE"]
metric = os.environ["METRIC"]
top_n_models = int(os.environ["TOP_N_MODELS"])
output_base = os.environ["OUTPUT_BASE"]
sequences_path = os.environ["SEQUENCES_PARQUET"]
pooled_ppi_db = os.environ["POOLED_PPI_DB"]
job_id = os.environ.get("SLURM_JOB_ID", "unknown")
plddt_trim_cutoff = float(os.environ.get("PLDDT_TRIM_CUTOFF", "0"))
save_untrimmed = os.environ.get("SAVE_UNTRIMMED_PDBS", "1") == "1"
interface_keep_dist = float(os.environ.get("INTERFACE_KEEP_DIST", "0"))
crop_subunits = os.environ.get("CROP_SUBUNITS", "0") == "1"

# ============================================================
# MODEL SELECTION -- how "best" is defined for METRIC.
# SORT_COL:       the summary_models.parquet (or derived) column ranked on.
# SORT_DESCENDING: True -> higher is better, False -> lower is better.
# NEEDS_PTM_AVG:  whether ptm_avg must be computed first (extra join
#                 against summary_confidences.parquet).
# ============================================================
if metric == "ranking_score":
    SORT_COL, SORT_DESCENDING, NEEDS_PTM_AVG = "ranking_score", True, False
elif metric == "pae":
    SORT_COL, SORT_DESCENDING, NEEDS_PTM_AVG = "chain_pair_pae_min", False, False
elif metric == "iptm":
    SORT_COL, SORT_DESCENDING, NEEDS_PTM_AVG = "chain_pair_iptm_corrected", True, False
elif metric == "ptm_avg":
    SORT_COL, SORT_DESCENDING, NEEDS_PTM_AVG = "ptm_avg", True, True
else:
    assert False, f"METRIC must be one of ranking_score/pae/iptm/ptm_avg, got {metric!r}"
print(f"Selection metric: {metric} (column={SORT_COL}, descending={SORT_DESCENDING})")


def rank_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """
    Order candidate AF3 models for ONE pair from best to worst, per METRIC.

    Rule:
      1) Drop rows where the chosen metric column is null (can't be ranked).
      2) Sort by that column (direction set by SORT_DESCENDING above).
      3) TIE-BREAK: if two or more models have the EXACT SAME metric value,
         the one with the LOWEST `sample` index wins -- deterministic,
         independent of source-parquet row order.

    Result is ordered best-first; row 0 is what TOP_N_MODELS=1 would use.
    """
    return (
        df.filter(pl.col(SORT_COL).is_not_null())
        .sort([SORT_COL, "sample"], descending=[SORT_DESCENDING, False])
    )


# --- ptm_avg support: only computed when METRIC == "ptm_avg" ---------------
_CHAIN_INDEX = {label: i for i, label in zip(range(2000), af3io.input.enumerate_chains())}


def _chain_ptm_at(chain_ptm: list[float] | None, chain_id: str) -> float:
    assert chain_id in _CHAIN_INDEX, \
        f"chain_id={chain_id!r} not in _CHAIN_INDEX (only built up to 2000 chains)."
    idx = _CHAIN_INDEX[chain_id]
    assert chain_ptm is not None and idx < len(chain_ptm), \
        f"chain_ptm missing or too short for chain_id={chain_id!r} (idx={idx})"
    return chain_ptm[idx]


def add_ptm_avg(df: pl.DataFrame) -> pl.DataFrame:
    """Join chain_ptm (summary_confidences.parquet) onto each candidate row
    and compute ptm_avg = mean(chain_ptm[chain_id1], chain_ptm[chain_id2])."""
    n_before = df.height

    confidences = (
        pl.scan_parquet(os.path.join(pooled_ppi_db, "summary_confidences.parquet"))
        .select("input_name", "input_type", "seed", "sample", "chain_ptm")
        .collect()
    )
    dupe_counts = confidences.group_by(["input_name", "input_type", "seed", "sample"]).len()
    assert (dupe_counts["len"] == 1).all(), (
        "summary_confidences.parquet has duplicate (input_name, input_type, seed, sample) keys -- "
        "a join on this table would silently fan out candidate rows."
    )

    out = df.join(confidences, on=["input_name", "input_type", "seed", "sample"], how="left")
    assert out.height == n_before, (
        f"join with summary_confidences.parquet changed row count "
        f"({n_before} -> {out.height}) -- fan-out from a duplicate key."
    )
    assert out["chain_ptm"].null_count() == 0, \
        "no chain_ptm found for some candidate rows -- data-integrity bug, not a normal gap."

    ptm1 = out.select(pl.struct("chain_ptm", "chain_id1").map_elements(
        lambda s: _chain_ptm_at(s["chain_ptm"], s["chain_id1"]), return_dtype=pl.Float64
    ).alias("ptm1"))["ptm1"]
    ptm2 = out.select(pl.struct("chain_ptm", "chain_id2").map_elements(
        lambda s: _chain_ptm_at(s["chain_ptm"], s["chain_id2"]), return_dtype=pl.Float64
    ).alias("ptm2"))["ptm2"]

    return out.with_columns(
        ((ptm1 + ptm2) / 2).alias("ptm_avg")
    ).drop("chain_ptm")


# --- Everything below this line is unchanged pipeline logic ----------------

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
    listed_counts = list(counts.values())

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

proteins = sorted(counts)
counts = collections.OrderedDict((p, counts[p]) for p in proteins)
complex_name = "_".join(f"{p}x{counts[p]}" for p in proteins)
if len(complex_name) > 200:
    import hashlib
    complex_name = complex_name[:180] + "_" + hashlib.sha1(complex_name.encode()).hexdigest()[:12]
run_name = f"{complex_name}_{source}"  # unchanged: no metric in run_name
print(f"Resolved complex: {complex_name}")
print(f"Stoichiometry: {dict(counts)}")

input_dir = Path(output_base) / f"{complex_name}_{source}_input"  # unchanged
pdbs_dir = input_dir / "pdbs"
input_dir.mkdir(parents=True, exist_ok=True)
pdbs_dir.mkdir(parents=True, exist_ok=True)

# --- Write subunits.json (source-independent: sequence only) -------------
sequences = pd.read_parquet(sequences_path)

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

# subunits.json is written ONCE, after all pair models are fetched and trimmed
# (see "Write subunits.json" below), because the cropped ranges are only known then.

# --- Required pairwise dimers: homodimers (count>=2) + heteropairs -------
required_homo = []
required_hetero = []
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

SQLITE_DB = os.path.join(pooled_ppi_db, "models.sqlite")
_sqlite_db_handle = None


def _get_sqlite_db() -> sqlite_utils.Database:
    global _sqlite_db_handle
    if _sqlite_db_handle is None:
        _sqlite_db_handle = sqlite_utils.Database(SQLITE_DB)
    return _sqlite_db_handle


def _fetch_pdb_str(db: sqlite_utils.Database, id_: str) -> str | None:
    try:
        row = db["models"].get(id_)
    except NotFoundError:
        return None
    db_id, pdb_str = foldcomp.decompress(row["fcz"])
    if db_id != id_:
        print(f"  WARNING: SQLite id mismatch: requested {id_}, got {db_id}")
        return None
    return pdb_str


def trim_low_plddt_termini(chain, cutoff: float, label: str,
                           partner_ns=None, keep_dist: float = 0.0) -> dict:
    """Trim both termini of `chain` in place.

    From each end, residues are removed until the first residue that meets
    EITHER stop criterion:
      (1) its CA pLDDT (B-factor) >= cutoff, or
      (2) its CA is within keep_dist A of any CA of the partner chain
          (partner_ns: NeighborSearch over the partner's UNTRIMMED CA atoms;
          skipped if None or keep_dist <= 0).
    That residue and everything inward of it is kept, including interior
    low-pLDDT regions. Residue numbering is left untouched.

    Returns a dict with n removed per end and why each end stopped
    ("plddt", "interface", "plddt+interface", "no_trim" if cutoff<=0, or
    "none_kept" if no residue met either criterion -> chain left untrimmed).
    """
    info = {"n_n": 0, "n_c": 0, "stop_n": "no_trim", "stop_c": "no_trim"}
    if cutoff <= 0:
        return info
    residues = list(chain)
    if not residues:
        return info

    ca_b = [r["CA"].get_bfactor() for r in residues if "CA" in r]
    # Sanity check on pLDDT scale: AF3 writes 0-100. A 0-1 scale would make
    # every residue fall below 50 and silently trigger the 'keep all' path.
    if ca_b and max(ca_b) <= 1.0:
        print(f"ERROR: {label}: max CA B-factor is {max(ca_b):.3f} -- pLDDT looks "
              f"like it is on a 0-1 scale, but cutoff={cutoff} assumes 0-100.")
        sys.exit(1)

    use_iface = partner_ns is not None and keep_dist > 0

    def stop_reason(r):
        """'' if r should be trimmed, else which criterion keeps it."""
        good_plddt = ("CA" in r) and (r["CA"].get_bfactor() >= cutoff)
        near = use_iface and ("CA" in r) and bool(partner_ns.search(r["CA"].coord, keep_dist))
        return "+".join(x for x, ok in (("plddt", good_plddt), ("interface", near)) if ok)

    # Scan inward from each end; stop at the first residue that is kept.
    first = next((i for i, r in enumerate(residues) if stop_reason(r)), None)
    if first is None:
        print(f"  WARNING: {label}: no residue with pLDDT >= {cutoff}"
              f"{f' or within {keep_dist} A of partner' if use_iface else ''}; "
              f"keeping chain untrimmed.")
        info.update(stop_n="none_kept", stop_c="none_kept")
        return info
    last = next(i for i in range(len(residues) - 1, -1, -1) if stop_reason(residues[i]))

    for r in residues[:first] + residues[last + 1:]:
        chain.detach_child(r.id)
    info.update(n_n=first, n_c=len(residues) - 1 - last,
                stop_n=stop_reason(residues[first]), stop_c=stop_reason(residues[last]))
    if info["n_n"] or info["n_c"]:
        kept = residues[first:last + 1]
        print(f"  trimmed {label}: -{info['n_n']} N-term (stop: {info['stop_n']}), "
              f"-{info['n_c']} C-term (stop: {info['stop_c']}) "
              f"(kept res {kept[0].id[1]}-{kept[-1].id[1]}, {len(kept)}/{len(residues)})")
    return info


# --- Full-length representative subunits ------------------------------------
# CombFold's run_on_pdbs.py needs, for EVERY subunit, at least one PDB in the
# pdbs folder that contains the COMPLETE subunit sequence (it builds the final
# assembly from these "representative" subunits and asserts
# "missing rep subunits" otherwise). Trimmed pair PDBs only ever contain a
# partial subunit, so with trimming we add one single-chain, UNTRIMMED PDB per
# protein (REP_<protein>_full_length.pdb). A single-chain PDB has no subunit
# pair, so it yields no transformations -- it only serves as the representative.
# The chain used is the untrimmed chain with the highest mean CA pLDDT over all
# pair models written, which is the same rule CombFold applies itself.
rep_candidates: dict[str, tuple[float, str]] = {}
# protein -> (first, last) residue number kept in ANY of its trimmed pair chains
kept_ranges: dict[str, tuple[int, int]] = {}


def _register_kept_range(protein: str, chain) -> None:
    """Widen protein's kept range to include the residues left in `chain`."""
    ids = [r.id[1] for r in chain]
    assert ids, f"{protein}: chain {chain.id} has no residues left after trimming"
    lo, hi = min(ids), max(ids)
    if protein in kept_ranges:
        lo, hi = min(lo, kept_ranges[protein][0]), max(hi, kept_ranges[protein][1])
    kept_ranges[protein] = (lo, hi)


def _register_rep_candidate(protein: str, chain) -> None:
    """Remember `chain` (untrimmed) as representative for `protein` if it has
    the highest mean CA pLDDT seen so far (ties keep the first, like CombFold)."""
    b = [r["CA"].get_bfactor() for r in chain if "CA" in r]
    assert b, f"{protein}: chain {chain.id} has no CA atoms"
    mean_b = sum(b) / len(b)
    if protein in rep_candidates and rep_candidates[protein][0] >= mean_b:
        return
    c = chain.copy()
    c.id = "A"
    s_ = Bio.PDB.Structure.Structure("rep")
    m_ = Bio.PDB.Model.Model(0)
    s_.add(m_)
    m_.add(c)
    pio = Bio.PDB.PDBIO()
    pio.set_structure(s_)
    buf = io.StringIO()
    pio.save(buf)
    rep_candidates[protein] = (mean_b, buf.getvalue())


def _chain_seq_like_combfold(pdb_text: str) -> str:
    """Same construction as CombFold's get_chain_to_seq(use_seqres=False):
    residue ids 1..max, 'X' where a residue (with CA) is missing."""
    model = Bio.PDB.PDBParser(QUIET=True).get_structure("r", io.StringIO(pdb_text))[0]
    chain = next(model.get_chains())
    res = {r.id[1]: r for r in chain if "CA" in r}
    return "".join(
        Bio.SeqUtils.seq1(res[i].get_resname()) if i in res else "X"
        for i in range(1, max(res) + 1)
    )


def cropped_subunits() -> collections.OrderedDict:
    """Return subunits dict with, per protein, the sequence cropped to the union
    of kept residues and start_res = first kept residue number (so residue
    numbers in the assembled output equal the original full-length numbering).
    `js` itself is not modified (it keeps the full-length sequences)."""
    trimmed = collections.OrderedDict()
    for protein in proteins:
        assert protein in kept_ranges, f"no kept range recorded for {protein}"
        lo, hi = kept_ranges[protein]
        full_seq = js[protein]["sequence"]
        assert 1 <= lo <= hi <= len(full_seq), (
            f"{protein}: kept range {lo}-{hi} outside sequence length {len(full_seq)}"
        )
        rep_seq = _chain_seq_like_combfold(rep_candidates[protein][1])
        assert rep_seq[lo - 1:hi] == full_seq[lo - 1:hi], (
            f"{protein}: residues {lo}-{hi} of the structure do not match subunits.json "
            f"(numbering/sequence disagree), refusing to crop"
        )
        entry = dict(js[protein])
        entry["start_res"] = lo
        entry["sequence"] = full_seq[lo - 1:hi]
        trimmed[protein] = entry
        print(f"Cropped subunit {protein}: residues {lo}-{hi} of {len(full_seq)} "
              f"({len(full_seq) - (hi - lo + 1)} removed)")
    return trimmed


def write_rep_subunits() -> None:
    """Write REP_<protein>_full_length.pdb for every protein of the complex."""
    for protein in proteins:
        assert protein in rep_candidates, (
            f"no untrimmed chain was recorded for {protein}; cannot write its "
            f"full-length representative (was a pair model missing?)"
        )
        mean_b, pdb_text = rep_candidates[protein]
        chain_seq = _chain_seq_like_combfold(pdb_text)
        assert js[protein]["sequence"] in chain_seq, (
            f"{protein}: representative chain sequence does not contain the "
            f"subunits.json sequence -- CombFold would not recognise it as a complete subunit"
        )
        out = pdbs_dir / f"REP_{protein}_full_length.pdb"
        out.write_text(pdb_text)
        print(f"Representative (untrimmed) subunit {protein}: mean pLDDT {mean_b:.1f} -> {out}")


def save_pair_pdb(fc_key1: str, fc_key2: str, out_path: Path, proteins: tuple[str, str]) -> dict | None:
    """Fetch both chains, merge them into one 2-chain structure (unchanged
    coordinates), optionally save an untrimmed copy, then trim low-pLDDT
    termini of each chain independently and write the PDB CombFold reads.
    Returns per-chain trim counts, or None if a chain is missing from SQLite."""
    db = _get_sqlite_db()
    parser = Bio.PDB.PDBParser(QUIET=True)
    struct0 = None
    for index, (id_, chain_id) in enumerate(
        zip([fc_key1, fc_key2], af3io.input.enumerate_chains())
    ):
        pdb_str = _fetch_pdb_str(db, id_)
        if pdb_str is None:
            return None
        struct = parser.get_structure(index, io.StringIO(pdb_str))
        if index == 0:
            struct0 = struct
        else:
            chain0 = next(struct[0].get_chains())
            chain0.id = chain_id
            struct0[0].add(chain0)

    chains = list(struct0[0])
    assert len(chains) == 2, f"expected 2 chains in merged pair, got {len(chains)}"
    if plddt_trim_cutoff > 0:
        # Register BEFORE trimming: the representative must be full length.
        for protein, chain in zip(proteins, chains):
            _register_rep_candidate(protein, chain)

    pdbio = Bio.PDB.PDBIO()
    if save_untrimmed and plddt_trim_cutoff > 0:
        # MUST be outside pdbs/ -- CombFold reads every PDB in that folder.
        untrimmed_dir = out_path.parent.parent / "pdbs_untrimmed"
        untrimmed_dir.mkdir(exist_ok=True)
        pdbio.set_structure(struct0)
        pdbio.save(str(untrimmed_dir / out_path.name))

    # Build CA neighbour searches on BOTH chains BEFORE any trimming, so each
    # chain's interface check sees its partner's full, untrimmed CAs.
    if plddt_trim_cutoff > 0 and interface_keep_dist > 0:
        searches = [Bio.PDB.NeighborSearch([r["CA"] for r in c if "CA" in r]) for c in chains]
    else:
        searches = [None, None]

    trim_info = {}
    for index, (chain, id_) in enumerate(zip(chains, [fc_key1, fc_key2])):
        t = trim_low_plddt_termini(chain, plddt_trim_cutoff, id_,
                                   partner_ns=searches[1 - index],
                                   keep_dist=interface_keep_dist)
        k = index + 1
        trim_info[f"trim_nterm_chain{k}"] = t["n_n"]
        trim_info[f"trim_cterm_chain{k}"] = t["n_c"]
        trim_info[f"trim_nstop_chain{k}"] = t["stop_n"]
        trim_info[f"trim_cstop_chain{k}"] = t["stop_c"]

    if plddt_trim_cutoff > 0:
        for protein, chain in zip(proteins, chains):
            _register_kept_range(protein, chain)

    pdbio.set_structure(struct0)
    pdbio.save(str(out_path))
    return trim_info

missing_any = False

_SELECT_COLS = [
    "pair_id", "af3_id1", "af3_id2", "input_type", "input_name",
    "batch_id", "seed", "sample", "chain_id1", "chain_id2",
    "ranking_score", "chain_pair_iptm",
    "chain_pair_iptm_expected", "chain_pair_iptm_corrected",
    "chain_pair_pae_min", "chain_pair_pae_min_recap",
]

# --- Homodimers -----------------------------------------------------------
if required_homo:
    required_homo_keys = {f"{lo}__{hi}" for lo, hi in required_homo}
    candidates_homo = (
        pl.scan_parquet(os.path.join(pooled_ppi_db, "summary_models.parquet"))
        .with_columns(
            (pl.min_horizontal(["af3_id1", "af3_id2"]) + "__" +
             pl.max_horizontal(["af3_id1", "af3_id2"])).alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(required_homo_keys))
        .filter(pl.col("input_type") == source)
        .select(_SELECT_COLS)
        .collect()
    )
    if NEEDS_PTM_AVG:
        candidates_homo = add_ptm_avg(candidates_homo)

    missing = set()
    for key in sorted(required_homo_keys):
        lo, hi = key.split("__")
        orig1, orig2 = lower_to_orig[lo], lower_to_orig[hi]
        n_written = 0
        seen_rows = set()

        ranked = rank_candidates(candidates_homo.filter(pl.col("pair_id") == key))
        for row in ranked.iter_rows(named=True):
            if n_written >= top_n_models:
                break
            row_key = (row["input_name"], row["sample"])
            if row_key in seen_rows:
                continue
            fc_key1 = f"{row['input_name']}_{row['sample']}_{row['chain_id1']}"
            fc_key2 = f"{row['input_name']}_{row['sample']}_{row['chain_id2']}"
            n_written += 1
            path_pdb = pdbs_dir / f"AFM_{orig1}_{orig2}_unrelaxed_rank_{n_written}_model_{n_written}.pdb"
            trim_info = save_pair_pdb(fc_key1, fc_key2, path_pdb, proteins=(orig1, orig2))
            if trim_info is not None:
                seen_rows.add(row_key)
                print(f" written ({n_written}/{top_n_models}): {path_pdb} (type={source}, "
                      f"sample={row['sample']}, {metric}={row[SORT_COL]:.3f}, "
                      f"ranking_score={row['ranking_score']:.3f})")
                source_log_rows.append({
                    "job_id": job_id, "run_name": run_name, "spec": spec,
                    "pair_type": "homo", "protein1": orig1, "protein2": orig2,
                    "requested_source": source, "used_source": source,
                    "metric": metric, "model_rank": n_written,
                    "af3_id1": row["af3_id1"], "af3_id2": row["af3_id2"],
                    "chain_id1": row["chain_id1"], "chain_id2": row["chain_id2"],
                    "input_type": row["input_type"], "input_name": row["input_name"],
                    "batch_id": row["batch_id"], "seed": row["seed"],
                    "sample": row["sample"], "ranking_score": row["ranking_score"],
                    "chain_pair_iptm": row["chain_pair_iptm"],
                    "chain_pair_iptm_expected": row["chain_pair_iptm_expected"],
                    "chain_pair_iptm_corrected": row["chain_pair_iptm_corrected"],
                    "chain_pair_pae_min": row["chain_pair_pae_min"],
                    "chain_pair_pae_min_recap": row["chain_pair_pae_min_recap"],
                    "ptm_avg": row.get("ptm_avg"),
                    "plddt_trim_cutoff": plddt_trim_cutoff,
                    "interface_keep_dist": interface_keep_dist,
                    **trim_info,
                    "status": "written",
                })
            else:
                n_written -= 1
                print(f" WARNING: {source} sample {row['sample']} structure missing from "
                      f"SQLite predictions DB for {orig1} - trying next-best...")

        if n_written == 0:
            print(f"ERROR: no '{source}' structure found in "
                  f"SQLite predictions DB for {orig1} (self-pair). "
                  f"(Strict single-source mode: not falling back to the other input_type.)")
            missing.add(key)
            source_log_rows.append({
                "job_id": job_id, "run_name": run_name, "spec": spec,
                "pair_type": "homo", "protein1": orig1, "protein2": orig2,
                "requested_source": source, "used_source": "MISSING",
                "metric": metric, "model_rank": None,
                "af3_id1": None, "af3_id2": None,
                "chain_id1": None, "chain_id2": None,
                "input_type": None, "input_name": None,
                "batch_id": None, "seed": None,
                "sample": None, "ranking_score": None, "chain_pair_iptm": None,
                "chain_pair_iptm_expected": None, "chain_pair_iptm_corrected": None,
                "chain_pair_pae_min": None, "chain_pair_pae_min_recap": None,
                "ptm_avg": None,
                "plddt_trim_cutoff": plddt_trim_cutoff,
                "interface_keep_dist": interface_keep_dist,
                "trim_nterm_chain1": None, "trim_cterm_chain1": None,
                "trim_nterm_chain2": None, "trim_cterm_chain2": None,
                "trim_nstop_chain1": None, "trim_cstop_chain1": None,
                "trim_nstop_chain2": None, "trim_cstop_chain2": None,
                "status": "missing",
            })
        elif n_written < top_n_models:
            print(f" NOTE: only {n_written}/{top_n_models} models available for "
                  f"{orig1} (self-pair) under source='{source}'; proceeding with what's available.")

    if missing:
        missing_any = True

# --- Heteropairs -----------------------------------------------------------
if required_hetero:
    required_hetero_keys = {f"{lo}__{hi}" for lo, hi in required_hetero}
    candidates_hetero = (
        pl.scan_parquet(os.path.join(pooled_ppi_db, "summary_models.parquet"))
        .with_columns(
            (pl.min_horizontal(["af3_id1", "af3_id2"]) + "__" +
             pl.max_horizontal(["af3_id1", "af3_id2"])).alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(required_hetero_keys))
        .filter(pl.col("input_type") == source)
        .select(_SELECT_COLS)
        .collect()
    )
    if NEEDS_PTM_AVG:
        candidates_hetero = add_ptm_avg(candidates_hetero)

    missing = set()
    for key in sorted(required_hetero_keys):
        lo, hi = key.split("__")
        orig1, orig2 = lower_to_orig[lo], lower_to_orig[hi]
        n_written = 0

        ranked = rank_candidates(candidates_hetero.filter(pl.col("pair_id") == key))
        for row in ranked.iter_rows(named=True):
            if n_written >= top_n_models:
                break
            fc_key1 = f"{row['input_name']}_{row['sample']}_{row['chain_id1']}"
            fc_key2 = f"{row['input_name']}_{row['sample']}_{row['chain_id2']}"
            n_written += 1
            path_pdb = pdbs_dir / f"AFM_{orig1}_{orig2}_unrelaxed_rank_{n_written}_model_{n_written}.pdb"
            trim_info = save_pair_pdb(fc_key1, fc_key2, path_pdb, proteins=(orig1, orig2))
            if trim_info is not None:
                print(f" written ({n_written}/{top_n_models}): {path_pdb} (type={source}, "
                      f"sample={row['sample']}, {metric}={row[SORT_COL]:.3f}, "
                      f"ranking_score={row['ranking_score']:.3f})")
                source_log_rows.append({
                    "job_id": job_id, "run_name": run_name, "spec": spec,
                    "pair_type": "hetero", "protein1": orig1, "protein2": orig2,
                    "requested_source": source, "used_source": source,
                    "metric": metric, "model_rank": n_written,
                    "af3_id1": row["af3_id1"], "af3_id2": row["af3_id2"],
                    "chain_id1": row["chain_id1"], "chain_id2": row["chain_id2"],
                    "input_type": row["input_type"], "input_name": row["input_name"],
                    "batch_id": row["batch_id"], "seed": row["seed"],
                    "sample": row["sample"], "ranking_score": row["ranking_score"],
                    "chain_pair_iptm": row["chain_pair_iptm"],
                    "chain_pair_iptm_expected": row["chain_pair_iptm_expected"],
                    "chain_pair_iptm_corrected": row["chain_pair_iptm_corrected"],
                    "chain_pair_pae_min": row["chain_pair_pae_min"],
                    "chain_pair_pae_min_recap": row["chain_pair_pae_min_recap"],
                    "ptm_avg": row.get("ptm_avg"),
                    "plddt_trim_cutoff": plddt_trim_cutoff,
                    "interface_keep_dist": interface_keep_dist,
                    **trim_info,
                    "status": "written",
                })
            else:
                n_written -= 1
                print(f" WARNING: {source} sample {row['sample']} structure missing from "
                      f"SQLite predictions DB for {orig1} - {orig2} - trying next-best...")

        if n_written == 0:
            print(f"ERROR: no '{source}' structure found in "
                  f"26.08 SQLite predictions DB for {orig1} - {orig2}. "
                  f"(Strict single-source mode: not falling back to the other input_type.)")
            missing.add(key)
            source_log_rows.append({
                "job_id": job_id, "run_name": run_name, "spec": spec,
                "pair_type": "hetero", "protein1": orig1, "protein2": orig2,
                "requested_source": source, "used_source": "MISSING",
                "metric": metric, "model_rank": None,
                "af3_id1": None, "af3_id2": None,
                "chain_id1": None, "chain_id2": None,
                "input_type": None, "input_name": None,
                "batch_id": None, "seed": None,
                "sample": None, "ranking_score": None, "chain_pair_iptm": None,
                "chain_pair_iptm_expected": None, "chain_pair_iptm_corrected": None,
                "chain_pair_pae_min": None, "chain_pair_pae_min_recap": None,
                "ptm_avg": None,
                "plddt_trim_cutoff": plddt_trim_cutoff,
                "interface_keep_dist": interface_keep_dist,
                "trim_nterm_chain1": None, "trim_cterm_chain1": None,
                "trim_nterm_chain2": None, "trim_cterm_chain2": None,
                "trim_nstop_chain1": None, "trim_cstop_chain1": None,
                "trim_nstop_chain2": None, "trim_cstop_chain2": None,
                "status": "missing",
            })
        elif n_written < top_n_models:
            print(f" NOTE: only {n_written}/{top_n_models} models available for "
                  f"{orig1} - {orig2} under source='{source}'; proceeding with what's available.")

    if missing:
        missing_any = True
        print("Cannot run CombFold without these. "
              "Generate them with AF-Multimer / AF3 / Boltz first.")

# --- Full-length representatives (only needed when trimming) ---------------
if plddt_trim_cutoff > 0 and not missing_any:
    write_rep_subunits()

# --- Write subunits.json (once, after all pair models are fetched) ----------
# Cropped when trimming + CROP_SUBUNITS are on; full-length otherwise. If a
# pair is missing, the job exits below and CombFold is not run anyway.
if plddt_trim_cutoff > 0 and crop_subunits and not missing_any:
    subunits_out = cropped_subunits()
else:
    subunits_out = js
subunits_path = input_dir / "subunits.json"
with open(subunits_path, "w") as fh:
    json.dump(subunits_out, fh, indent=2)
print(f"Written: {subunits_path}")

# --- Per-job source log ----------------------------------------------------
log_dir = Path(output_base) / "pdb_source_logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"{run_name}_pairs.csv"
if source_log_rows:
    pl.DataFrame(source_log_rows).write_csv(log_path)
    print(f"Source log written: {log_path}")
else:
    print("No source log rows to write.")

pairs_csv_cols = [
    "run_name", "spec", "pair_type", "protein1", "protein2",
    "requested_source", "used_source", "metric", "model_rank",
    "af3_id1", "af3_id2", "chain_id1", "chain_id2",
    "input_type", "input_name", "batch_id", "seed", "sample",
    "ranking_score", "chain_pair_iptm",
    "chain_pair_iptm_expected", "chain_pair_iptm_corrected",
    "chain_pair_pae_min", "chain_pair_pae_min_recap", "ptm_avg",
    "plddt_trim_cutoff", "interface_keep_dist",
    "trim_nterm_chain1", "trim_cterm_chain1", "trim_nterm_chain2", "trim_cterm_chain2",
    "trim_nstop_chain1", "trim_cstop_chain1", "trim_nstop_chain2", "trim_cstop_chain2",
    "status",
]
pairs_csv_path = log_dir / f"{run_name}_pairs.csv"
if source_log_rows:
    pl.DataFrame(source_log_rows).select(pairs_csv_cols).write_csv(pairs_csv_path)
    print(f"Per-complex pairs CSV written: {pairs_csv_path}")
else:
    print("No pairs to write to per-complex CSV.")

def _contact_key(r):
    return (r["pair_type"], r["protein1"], r["protein2"])

homo_contacts = {_contact_key(r) for r in source_log_rows if r["pair_type"] == "homo"}
hetero_contacts = {_contact_key(r) for r in source_log_rows if r["pair_type"] == "hetero"}
missing_contacts = {_contact_key(r) for r in source_log_rows if r["used_source"] == "MISSING"}
written_rows = [r for r in source_log_rows if r["status"] == "written"]

n_homo = len(homo_contacts)
n_homo_written_rows = sum(1 for r in written_rows if r["pair_type"] == "homo")
n_hetero = len(hetero_contacts)
n_hetero_written_rows = sum(1 for r in written_rows if r["pair_type"] == "hetero")
n_missing = len(missing_contacts)
print(f"SUMMARY (job {job_id}, metric='{metric}', top_n_models={top_n_models}, source='{source}' strict, "
      f"no cross-source fallback): "
      f"homodimer contacts={n_homo} ({n_homo_written_rows} models written total); "
      f"heteropair contacts={n_hetero} ({n_hetero_written_rows} models written total); "
      f"{n_missing} contact(s) fully missing (0 models found).")

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