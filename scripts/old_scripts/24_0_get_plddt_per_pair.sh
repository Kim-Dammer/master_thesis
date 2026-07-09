#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_plddt
#SBATCH --output=logs/lookup_plddt_%j.out
#SBATCH --error=logs/lookup_plddt_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=100G

# ===========================================================================
# lookup_plddt.sh — Extract per-chain mean pLDDT from the pooled-PPI yeast
# foldcomp structure database.
#
# pLDDT is NOT in the parquet summary files — it lives in the structure
# B-factors, stored in the foldcomp database at predictions-db/predictions-db.
#
# The foldcomp key for each chain is: {pool_hash}_{af3_id}
# where pool_hash == the "name" column in pp.pairs (already a sha1 of pool_id)
#
# CRASH RECOVERY: Checkpoint parts are written every 100 keys. If the job
# crashes, just resubmit — it will load existing checkpoints and resume.
# Checkpoints are cleaned up automatically after the final output is written.
# ===========================================================================

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting pLDDT lookup from foldcomp structures..."

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
PAIRS_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/yeast_protein_pairs.parquet")
OUTPUT_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/yeast_pairs_plddt.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FC_DB = os.path.join(POOLED_PPI_DB, "predictions-db/predictions-db")
BATCH_SIZE = 5000          # keys per foldcomp.open() call (I/O efficiency)
CHECKPOINT_INTERVAL = 100  # keys per checkpoint part file (crash safety)
CHECKPOINT_DIR = OUTPUT_PATH.parent / "plddt_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper: write a checkpoint part ---------------------------------------
def write_checkpoint(buffer, idx):
    """Write a checkpoint part file (atomic via temp + rename)."""
    part_path = CHECKPOINT_DIR / f"part_{idx:06d}.parquet"
    tmp_path = part_path.with_suffix(".parquet.tmp")
    pl.DataFrame({
        "fc_key": [b[0] for b in buffer],
        "mean_plddt": [b[1] for b in buffer],
    }).write_parquet(tmp_path)
    tmp_path.rename(part_path)

# --- Load user's target pairs ----------------------------------------------
print(f"Reading target pairs from {PAIRS_PATH} ...")
pairs_df = pl.read_parquet(PAIRS_PATH)
print(f"  {pairs_df.height} rows, columns: {pairs_df.columns}")

# Normalize to canonical order (p1 <= p2) using uppercase uniprot IDs
# Adjust column names if your file uses different names
if "protein_A" in pairs_df.columns and "protein_B" in pairs_df.columns:
    col_a, col_b = "protein_A", "protein_B"
elif "uniprot_id1" in pairs_df.columns and "uniprot_id2" in pairs_df.columns:
    col_a, col_b = "uniprot_id1", "uniprot_id2"
else:
    # Assume first two columns are the pair
    col_a, col_b = pairs_df.columns[0], pairs_df.columns[1]
    print(f"  WARNING: Assuming pair columns are '{col_a}' and '{col_b}'")

target_pairs = (
    pairs_df.with_columns([
        pl.min_horizontal(col_a, col_b).alias("p1"),
        pl.max_horizontal(col_a, col_b).alias("p2"),
    ])
    .select(["p1", "p2"]).unique()
)
print(f"  {target_pairs.height} unique target pairs")

# --- Load DB ---------------------------------------------------------------
print("Loading pooled-PPI DB ...")
pp = pooled_ppi.PooledPredictionsDb(POOLED_PPI_DB)
pp_df = pl.from_pandas(pp.pairs)
del pp; gc.collect()

# Normalize DB pairs to canonical order (uppercase uniprot IDs)
pp_df = pp_df.with_columns([
    pl.min_horizontal("uniprot_id1", "uniprot_id2").alias("p1"),
    pl.max_horizontal("uniprot_id1", "uniprot_id2").alias("p2"),
])

# For each pair, pick the best-scoring pool (max chain_pair_iptm_best_corrected)
# The "name" column IS the pool_hash (sha1 of pool_id) — used as the foldcomp
# filename prefix. Do NOT re-hash it.
best_pairs_all = (
    pp_df.sort("chain_pair_iptm_best_corrected", descending=True)
    .unique(subset=["af3_id1", "af3_id2"], keep="first")
    .select([
        "af3_id1", "af3_id2", "uniprot_id1", "uniprot_id2",
        "name", "chain_id1", "chain_id2", "chain_pair_iptm_best_corrected",
        "p1", "p2",
    ])
)

# --- Filter to user's target pairs only ------------------------------------
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

print(f"Best pairs to process: {best_pairs.height}")

# --- Extract pLDDT from foldcomp structures --------------------------------
parser = PDBParser(QUIET=True)

all_keys = list(set(
    best_pairs["fc_key1"].to_list() + best_pairs["fc_key2"].to_list()
))
print(f"Unique foldcomp keys to fetch: {len(all_keys)}")
print(f"  Example keys: {all_keys[:3]}")

# --- Resume from existing checkpoints --------------------------------------
key_to_plddt = {}
processed_keys = set()
existing_parts = sorted(CHECKPOINT_DIR.glob("part_*.parquet"))
if existing_parts:
    print(f"Found {len(existing_parts)} checkpoint parts, loading ...")
    for part in existing_parts:
        try:
            part_df = pl.read_parquet(part)
            for k, v in zip(part_df["fc_key"].to_list(), part_df["mean_plddt"].to_list()):
                key_to_plddt[k] = v
                processed_keys.add(k)
        except Exception as e:
            print(f"  WARNING: Could not load {part.name}: {e} — removing corrupt part")
            part.unlink()
    print(f"  Resumed with {len(processed_keys)} already-processed keys")

# Filter to unprocessed keys only
remaining_keys = [k for k in all_keys if k not in processed_keys]
print(f"  Remaining keys to process: {len(remaining_keys)}")

if len(remaining_keys) == 0:
    print("All keys already processed — skipping to final output assembly.")
else:
    # --- Process with checkpointing ----------------------------------------
    n_found = 0
    n_missing = 0
    n_not_in_db = 0
    buffer = []
    part_idx = len(existing_parts)  # continue numbering from where we left off

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
                    plddt = float(np.mean(plddt_vals)) if plddt_vals else None
                    n_found += 1
                except Exception:
                    plddt = None
                    n_missing += 1

                key_to_plddt[name] = plddt
                buffer.append((name, plddt))

                # Checkpoint every CHECKPOINT_INTERVAL keys
                if len(buffer) >= CHECKPOINT_INTERVAL:
                    write_checkpoint(buffer, part_idx)
                    part_idx += 1
                    buffer = []

        # Count keys that were requested but not returned by foldcomp
        not_returned = batch_set - returned_keys
        n_not_in_db += len(not_returned)

        print(f"  Processed {min(i + BATCH_SIZE, len(remaining_keys))}/{len(remaining_keys)} keys "
              f"(found: {n_found}, parse_errors: {n_missing}, not_in_db: {n_not_in_db}, "
              f"checkpoints: {part_idx})")

    # Flush any remaining buffer
    if buffer:
        write_checkpoint(buffer, part_idx)
        print(f"  Wrote final checkpoint (part_{part_idx:06d})")

# --- Map back to pairs via join --------------------------------------------
print("Assembling final output ...")
plddt_df = pl.DataFrame({
    "fc_key": list(key_to_plddt.keys()),
    "mean_plddt": list(key_to_plddt.values()),
})

result_df = (
    best_pairs
    .join(
        plddt_df.rename({"fc_key": "fc_key1", "mean_plddt": "mean_plddt1"}),
        on="fc_key1",
        how="left",
    )
    .join(
        plddt_df.rename({"fc_key": "fc_key2", "mean_plddt": "mean_plddt2"}),
        on="fc_key2",
        how="left",
    )
    .with_columns(
        ((pl.col("mean_plddt1") + pl.col("mean_plddt2")) / 2).alias("mean_plddt_pair")
    )
    .select([
        "uniprot_id1", "uniprot_id2",
        "mean_plddt1", "mean_plddt2", "mean_plddt_pair",
        "chain_pair_iptm_best_corrected",
    ])
)

result_df.write_parquet(OUTPUT_PATH)

# Report coverage
n_valid = result_df.filter(pl.col("mean_plddt_pair").is_not_null()).height
print(f"\nDone! Output: {OUTPUT_PATH}")
print(f"  Shape: {result_df.shape}")
print(f"  pLDDT coverage: {n_valid}/{result_df.height} non-null "
      f"({100*n_valid/result_df.height:.1f}%)")

# --- Clean up checkpoints (only after successful output) -------------------
for p in CHECKPOINT_DIR.glob("part_*.parquet"):
    p.unlink()
print("Checkpoints cleaned up.")
PYEOF

echo "[$(date)] Script run complete."
