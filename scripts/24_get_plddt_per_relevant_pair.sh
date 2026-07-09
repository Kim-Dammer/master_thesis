#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_plddt_pairwise
#SBATCH --output=logs/lookup_plddt_pairwise_%j.out
#SBATCH --error=logs/lookup_plddt_pairwise_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=100G

# ===========================================================================
# lookup_plddt_pairwise.sh — Per-PAIR pLDDT + length + pLDDT distribution.
#
# For each pair, picks the best pool (max corrected ipTM) where both proteins
# were co-folded, then extracts each protein's pLDDT FROM THAT POOL.
#
# Output columns:
#   Protein_1, Protein_2,
#   plddt1, len1, plddt2, len2,
#   mean_plddt_simple, mean_plddt_weighted,
#   chain_pair_iptm_best_corrected,
#   frac1_<50, frac1_50_60, frac1_60_70, frac1_70_80, frac1_80_90, frac1_>=90,
#   frac2_<50, frac2_50_60, frac2_60_70, frac2_70_80, frac2_80_90, frac2_>=90
#
# CRASH RECOVERY: Checkpoints every 100 keys. Resubmit to resume.
# ===========================================================================

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting per-pair pLDDT extraction with distribution..."

python3 - <<'PYEOF'
import os, gc
from io import StringIO
from pathlib import Path
import polars as pl
import numpy as np
import foldcomp
from Bio.PDB import PDBParser
import pooled_ppi

# --- Paths -----------------------------------------------------------------
POOLED_PPI_DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.04"
PAIRS_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/pLDDT/Ym_and_complex_used_pairs.csv")
OUTPUT_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/pLDDT/Ym_and_complex_used_pairs_plddt_plddt_per_pair.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FC_DB = os.path.join(POOLED_PPI_DB, "predictions-db/predictions-db")
CHECKPOINT_DIR = OUTPUT_PATH.parent / "plddt_checkpoints_pairwise"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5000
CHECKPOINT_INTERVAL = 100

# pLDDT bins: <50, 50-60, 60-70, 70-80, 80-90, >=90
BIN_LABELS = ["<50", "50_60", "60_70", "70_80", "80_90", "ge90"]
BIN_EDGES = [0, 50, 60, 70, 80, 90, 100.01]  # right-open except last

def compute_plddt_stats(plddt_vals):
    """Compute mean pLDDT and bin counts from a list of B-factor values."""
    if not plddt_vals or len(plddt_vals) == 0:
        return None, [0]*6, 0
    arr = np.array(plddt_vals, dtype=float)
    mean_plddt = float(np.mean(arr))
    n_total = len(arr)
    counts = []
    for b in range(len(BIN_EDGES) - 1):
        lo = BIN_EDGES[b]
        hi = BIN_EDGES[b + 1]
        if b == len(BIN_EDGES) - 2:  # last bin: >=90 (inclusive of 100)
            counts.append(int(np.sum(arr >= lo)))
        else:
            counts.append(int(np.sum((arr >= lo) & (arr < hi))))
    return mean_plddt, counts, n_total

# --- Helper: write checkpoint ----------------------------------------------
def write_checkpoint(buffer, idx):
    part_path = CHECKPOINT_DIR / f"part_{idx:06d}.parquet"
    tmp_path = part_path.with_suffix(".parquet.tmp")
    data = {"fc_key": [b[0] for b in buffer], "mean_plddt": [b[1] for b in buffer]}
    for i, label in enumerate(BIN_LABELS):
        data["count_" + label] = [b[2][i] for b in buffer]
    data["n_residues"] = [b[3] for b in buffer]
    pl.DataFrame(data).write_parquet(tmp_path)
    tmp_path.rename(part_path)

# --- Load input pairs ------------------------------------------------------
print(f"Reading pairs from {PAIRS_PATH} ...")
pairs_df = pl.read_csv(PAIRS_PATH)
print(f"  {pairs_df.height} pairs, columns: {pairs_df.columns}")

if "Protein_1" in pairs_df.columns and "Protein_2" in pairs_df.columns:
    col_a, col_b = "Protein_1", "Protein_2"
elif "protein_A" in pairs_df.columns and "protein_B" in pairs_df.columns:
    col_a, col_b = "protein_A", "protein_B"
else:
    col_a, col_b = pairs_df.columns[0], pairs_df.columns[1]
    print(f"  WARNING: Assuming pair columns are '{col_a}' and '{col_b}'")

# Normalize to canonical order (uppercase)
target_pairs = (
    pairs_df.with_columns([
        pl.min_horizontal(col_a, col_b).alias("p1"),
        pl.max_horizontal(col_a, col_b).alias("p2"),
    ])
    .select(["p1", "p2"]).unique()
)
print(f"  {target_pairs.height} unique target pairs")

# --- Load proteins.parquet for lengths -------------------------------------
print("Loading proteins.parquet for lengths ...")
proteins_meta = pl.read_parquet(os.path.join(POOLED_PPI_DB, "proteins.parquet"))
length_lookup = proteins_meta.select(["uniprot_id", "seq_len"])

# --- Load pooled-PPI DB ----------------------------------------------------
print("Loading pooled-PPI DB ...")
pp = pooled_ppi.PooledPredictionsDb(POOLED_PPI_DB)
pp_df = pl.from_pandas(pp.pairs)
del pp; gc.collect()

# Normalize DB pairs to canonical order
pp_df = pp_df.with_columns([
    pl.min_horizontal("uniprot_id1", "uniprot_id2").alias("p1"),
    pl.max_horizontal("uniprot_id1", "uniprot_id2").alias("p2"),
])

# For each pair, pick the best-scoring pool (max chain_pair_iptm_best_corrected)
best_pairs_all = (
    pp_df.sort("chain_pair_iptm_best_corrected", descending=True)
    .unique(subset=["af3_id1", "af3_id2"], keep="first")
    .select([
        "af3_id1", "af3_id2", "uniprot_id1", "uniprot_id2",
        "name", "chain_id1", "chain_id2", "chain_pair_iptm_best_corrected",
        "p1", "p2",
    ])
)

# --- Filter to target pairs ------------------------------------------------
print("Filtering to target pairs ...")
best_pairs = best_pairs_all.join(target_pairs, on=["p1", "p2"], how="inner")
print(f"  {best_pairs.height} target pairs found in DB "
      f"(out of {target_pairs.height} requested)")
del best_pairs_all, pp_df
gc.collect()

# Build foldcomp keys: {name}_{af3_id} (name == pool_hash)
best_pairs = best_pairs.with_columns([
    (pl.col("name") + "_" + pl.col("af3_id1")).alias("fc_key1"),
    (pl.col("name") + "_" + pl.col("af3_id2")).alias("fc_key2"),
])

print(f"Pairs to process: {best_pairs.height}")

# --- Extract pLDDT from foldcomp -------------------------------------------
parser = PDBParser(QUIET=True)

all_keys = list(set(
    best_pairs["fc_key1"].to_list() + best_pairs["fc_key2"].to_list()
))
print(f"Unique foldcomp keys to fetch: {len(all_keys)}")
print(f"  Example keys: {all_keys[:3]}")

# --- Resume from checkpoints -----------------------------------------------
# Stores: fc_key -> (mean_plddt, [counts], n_residues)
key_stats = {}
processed_keys = set()
existing_parts = sorted(CHECKPOINT_DIR.glob("part_*.parquet"))
if existing_parts:
    print(f"Found {len(existing_parts)} checkpoint parts, loading ...")
    for part in existing_parts:
        try:
            part_df = pl.read_parquet(part)
            cols = part_df.columns
            has_bins = all("count_" + lbl in cols for lbl in BIN_LABELS)
            for row_idx in range(part_df.height):
                k = part_df["fc_key"][row_idx]
                mp = part_df["mean_plddt"][row_idx]
                if has_bins:
                    counts = [part_df["count_" + lbl][row_idx] for lbl in BIN_LABELS]
                    nr = part_df["n_residues"][row_idx]
                else:
                    # Old-format checkpoint (mean only) — re-extract this key
                    counts = None
                    nr = 0
                if counts is not None:
                    key_stats[k] = (mp, counts, nr)
                    processed_keys.add(k)
        except Exception as e:
            print(f"  WARNING: Could not load {part.name}: {e} — removing")
            part.unlink()
    print(f"  Resumed with {len(processed_keys)} already-processed keys")

# Keys that need (re-)extraction: not in cache, or old-format without bins
remaining_keys = [k for k in all_keys if k not in processed_keys]
print(f"  Remaining keys to process: {len(remaining_keys)}")

if len(remaining_keys) > 0:
    n_found = 0
    n_missing = 0
    n_not_in_db = 0
    buffer = []
    part_idx = len(existing_parts)

    for i in range(0, len(remaining_keys), BATCH_SIZE):
        batch_keys = remaining_keys[i:i + BATCH_SIZE]
        batch_set = set(batch_keys)
        returned_keys = set()

        with foldcomp.open(FC_DB, ids=batch_keys) as db:
            for name, pdb_str in db:
                returned_keys.add(name)
                try:
                    struct = parser.get_structure(name, StringIO(pdb_str))
                    chain = next(struct[0].get_chains())
                    residues = [r for r in chain.get_residues() if "CA" in r]
                    plddt_vals = [r["CA"].get_bfactor() for r in residues]
                    mean_plddt, counts, n_res = compute_plddt_stats(plddt_vals)
                    n_found += 1
                except Exception:
                    mean_plddt, counts, n_res = None, [0]*6, 0
                    n_missing += 1

                key_stats[name] = (mean_plddt, counts, n_res)
                buffer.append((name, mean_plddt, counts, n_res))

                if len(buffer) >= CHECKPOINT_INTERVAL:
                    write_checkpoint(buffer, part_idx)
                    part_idx += 1
                    buffer = []

        n_not_in_db += len(batch_set - returned_keys)
        print(f"  Processed {min(i + BATCH_SIZE, len(remaining_keys))}/{len(remaining_keys)} "
              f"(found: {n_found}, errors: {n_missing}, not_in_db: {n_not_in_db}, "
              f"checkpoints: {part_idx})")

    if buffer:
        write_checkpoint(buffer, part_idx)
        print(f"  Wrote final checkpoint (part_{part_idx:06d})")

# --- Build pLDDT stats table -----------------------------------------------
print("Assembling final output ...")

fc_keys = list(key_stats.keys())
stats_data = {"fc_key": fc_keys, "mean_plddt": [key_stats[k][0] for k in fc_keys]}
for i, label in enumerate(BIN_LABELS):
    stats_data["count_" + label] = [key_stats[k][1][i] for k in fc_keys]
stats_data["n_residues"] = [key_stats[k][2] for k in fc_keys]
plddt_df = pl.DataFrame(stats_data)

# Compute fractions per bin
for label in BIN_LABELS:
    plddt_df = plddt_df.with_columns(
        (pl.col("count_" + label) / pl.col("n_residues")).alias("frac_" + label)
    )

result_df = (
    best_pairs
    .join(
        plddt_df.rename({"fc_key": "fc_key1", "mean_plddt": "plddt1"}),
        on="fc_key1",
        how="left",
    )
    .join(
        plddt_df.rename({"fc_key": "fc_key2", "mean_plddt": "plddt2"}),
        on="fc_key2",
        how="left",
    )
    # Add lengths from proteins.parquet (per-protein, pool-independent)
    .join(
        length_lookup.rename({"uniprot_id": "uniprot_id1", "seq_len": "len1"}),
        on="uniprot_id1",
        how="left",
    )
    .join(
        length_lookup.rename({"uniprot_id": "uniprot_id2", "seq_len": "len2"}),
        on="uniprot_id2",
        how="left",
    )
    # Both simple and length-weighted averages
    .with_columns(
        ((pl.col("plddt1") + pl.col("plddt2")) / 2).alias("mean_plddt_simple"),
        ((pl.col("plddt1") * pl.col("len1") + pl.col("plddt2") * pl.col("len2"))
         / (pl.col("len1") + pl.col("len2"))).alias("mean_plddt_weighted"),
    )
    # Select final columns: pairs, plddt, lengths, averages, iptm, fractions
    .select(
        [pl.col("uniprot_id1").alias(col_a),
         pl.col("uniprot_id2").alias(col_b),
         "plddt1", "len1", "plddt2", "len2",
         "mean_plddt_simple", "mean_plddt_weighted",
         "chain_pair_iptm_best_corrected"]
        + [pl.col("frac_" + lbl).alias("frac1_" + lbl) for lbl in BIN_LABELS]
        + [pl.col("frac_" + lbl + "_right").alias("frac2_" + lbl) for lbl in BIN_LABELS]
    )
)

result_df.write_parquet(OUTPUT_PATH)

# Also CSV for inspection
csv_path = OUTPUT_PATH.with_suffix(".csv")
result_df.write_csv(csv_path)

# Report coverage
n_valid = result_df.filter(pl.col("mean_plddt_weighted").is_not_null()).height
print(f"\nDone! Output: {OUTPUT_PATH}")
print(f"  CSV:  {csv_path}")
print(f"  Shape: {result_df.shape}")
print(f"  Columns: {result_df.columns}")
print(f"  pLDDT coverage: {n_valid}/{result_df.height} non-null "
      f"({100*n_valid/result_df.height:.1f}%)")
print(f"\n  Sample (first 3 rows):")
print(result_df.head(3))

# --- Clean up checkpoints --------------------------------------------------
for p in CHECKPOINT_DIR.glob("part_*.parquet"):
    p.unlink()
print("Checkpoints cleaned up.")
PYEOF

echo "[$(date)] Script run complete."
