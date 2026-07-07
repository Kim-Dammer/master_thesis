#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_plddt_pairs
#SBATCH --output=logs/lookup_plddt_pairs_%j.out
#SBATCH --error=logs/lookup_plddt_pairs_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G

# ===========================================================================
# lookup_plddt_pairs.sh — Per-protein pLDDT + length, joined to pairs.
#
# For each unique protein in the input pairs:
#   1. Find a pool where it appears (from pp.pairs)
#   2. Extract mean pLDDT from foldcomp B-factors (one extraction per protein)
#   3. Get sequence length from proteins.parquet
#
# Output: one row per pair with plddt1, plddt2, len1, len2, mean_plddt_pair
#
# Reuses existing checkpoints in plddt_checkpoints/ if available.
# ===========================================================================

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting per-protein pLDDT extraction for pairs..."

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
PAIRS_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/Ym_and_complex_used_pairs.csv") #contains only pairs from complexes of yeastmap and complex, not all possible pairs
OUTPUT_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/Ym_and_complex_used_pairs_plddt.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FC_DB = os.path.join(POOLED_PPI_DB, "predictions-db/predictions-db")
CHECKPOINT_DIR = OUTPUT_PATH.parent / "plddt_checkpoints_used_pairs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load input pairs ------------------------------------------------------
print(f"Reading pairs from {PAIRS_PATH} ...")
pairs_df = pl.read_csv(PAIRS_PATH)
print(f"  {pairs_df.height} pairs, columns: {pairs_df.columns}")

# Detect column names
if "Protein_1" in pairs_df.columns and "Protein_2" in pairs_df.columns:
    col_a, col_b = "Protein_1", "Protein_2"
elif "protein_A" in pairs_df.columns and "protein_B" in pairs_df.columns:
    col_a, col_b = "protein_A", "protein_B"
else:
    col_a, col_b = pairs_df.columns[0], pairs_df.columns[1]
    print(f"  WARNING: Assuming pair columns are '{col_a}' and '{col_b}'")

# Collect unique proteins (uppercase uniprot IDs)
proteins_a = pairs_df[col_a].unique().to_list()
proteins_b = pairs_df[col_b].unique().to_list()
unique_proteins = sorted(set(proteins_a + proteins_b))
print(f"  {len(unique_proteins)} unique proteins to extract")

# --- Load proteins.parquet for sequence lengths ----------------------------
print("Loading proteins.parquet for lengths ...")
proteins_meta = pl.read_parquet(os.path.join(POOLED_PPI_DB, "proteins.parquet"))
# proteins.parquet has: af3_id, uniprot_id, uniprot_genes, ..., seq_len, seq
# uniprot_id is uppercase, af3_id is lowercase
print(f"  {proteins_meta.height} proteins in DB")

# --- Load pooled-PPI DB to find pools for each protein ---------------------
print("Loading pooled-PPI DB ...")
pp = pooled_ppi.PooledPredictionsDb(POOLED_PPI_DB)
pp_df = pl.from_pandas(pp.pairs)
del pp; gc.collect()

# For each unique protein, find one pool where it appears (best ipTM)
# Build a lookup: uniprot_id -> (pool_hash, af3_id)
print("Building protein-to-pool lookup ...")

# Protein appears as uniprot_id1 (af3_id1) or uniprot_id2 (af3_id2)
# Pick the pool with the best chain_pair_iptm_best_corrected for each protein
side1 = (
    pp_df.sort("chain_pair_iptm_best_corrected", descending=True)
    .unique(subset=["uniprot_id1"], keep="first")
    .select([
        pl.col("uniprot_id1").alias("uniprot_id"),
        pl.col("af3_id1").alias("af3_id"),
        pl.col("name").alias("pool_hash"),
    ])
)
side2 = (
    pp_df.sort("chain_pair_iptm_best_corrected", descending=True)
    .unique(subset=["uniprot_id2"], keep="first")
    .select([
        pl.col("uniprot_id2").alias("uniprot_id"),
        pl.col("af3_id2").alias("af3_id"),
        pl.col("name").alias("pool_hash"),
    ])
)

# Combine both sides, keep best pool per protein
protein_pools = (
    pl.concat([side1, side2])
    .sort("pool_hash")  # arbitrary tiebreak; could sort by iptm but we don't have it here
    .unique(subset=["uniprot_id"], keep="first")
)

# Filter to our unique proteins
target_proteins_df = pl.DataFrame({"uniprot_id": unique_proteins})
protein_pools = protein_pools.join(target_proteins_df, on="uniprot_id", how="inner")
print(f"  {protein_pools.height}/{len(unique_proteins)} proteins found in DB")

# Build foldcomp keys: {pool_hash}_{af3_id}
protein_pools = protein_pools.with_columns(
    (pl.col("pool_hash") + "_" + pl.col("af3_id")).alias("fc_key")
)

# Report missing proteins
missing = set(unique_proteins) - set(protein_pools["uniprot_id"].to_list())
if missing:
    print(f"  WARNING: {len(missing)} proteins not found in DB: {list(missing)[:5]}...")

# --- Load existing checkpoints (reuse from previous runs) ------------------
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
            print(f"  WARNING: Could not load {part.name}: {e}")
    print(f"  Loaded {len(processed_keys)} cached pLDDT values")

# --- Extract pLDDT for proteins not yet in cache ---------------------------
all_keys = protein_pools["fc_key"].to_list()
remaining_keys = [k for k in all_keys if k not in processed_keys]
print(f"  Keys to extract: {len(remaining_keys)} (cached: {len(all_keys) - len(remaining_keys)})")

if len(remaining_keys) > 0:
    parser = PDBParser(QUIET=True)
    n_found = 0
    n_missing = 0
    n_not_in_db = 0
    buffer = []
    part_idx = len(existing_parts)

    BATCH_SIZE = 5000
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

                if len(buffer) >= 100:
                    part_path = CHECKPOINT_DIR / f"part_{part_idx:06d}.parquet"
                    tmp_path = part_path.with_suffix(".parquet.tmp")
                    pl.DataFrame({
                        "fc_key": [b[0] for b in buffer],
                        "mean_plddt": [b[1] for b in buffer],
                    }).write_parquet(tmp_path)
                    tmp_path.rename(part_path)
                    part_idx += 1
                    buffer = []

        n_not_in_db += len(batch_set - returned_keys)
        print(f"  Processed {min(i + BATCH_SIZE, len(remaining_keys))}/{len(remaining_keys)} "
              f"(found: {n_found}, errors: {n_missing}, not_in_db: {n_not_in_db})")

    # Flush remaining buffer
    if buffer:
        part_path = CHECKPOINT_DIR / f"part_{part_idx:06d}.parquet"
        tmp_path = part_path.with_suffix(".parquet.tmp")
        pl.DataFrame({
            "fc_key": [b[0] for b in buffer],
            "mean_plddt": [b[1] for b in buffer],
        }).write_parquet(tmp_path)
        tmp_path.rename(part_path)

# --- Build per-protein pLDDT + length table --------------------------------
print("Building per-protein table ...")
plddt_df = pl.DataFrame({
    "fc_key": list(key_to_plddt.keys()),
    "mean_plddt": list(key_to_plddt.values()),
})

protein_table = (
    protein_pools
    .join(plddt_df, on="fc_key", how="left")
    .join(
        proteins_meta.select(["uniprot_id", "seq_len"]),
        on="uniprot_id",
        how="left",
    )
    .select(["uniprot_id", "mean_plddt", "seq_len"])
    .rename({"mean_plddt": "plddt", "seq_len": "length"})
)

print(f"  Per-protein table: {protein_table.height} proteins")
n_valid = protein_table.filter(pl.col("plddt").is_not_null()).height
print(f"  pLDDT coverage: {n_valid}/{protein_table.height} "
      f"({100*n_valid/protein_table.height:.1f}%)")

# --- Join back to pairs ----------------------------------------------------
print("Joining to pairs ...")
result_df = (
    pairs_df
    .join(
        protein_table.rename({"uniprot_id": col_a, "plddt": "plddt1", "length": "len1"}),
        on=col_a,
        how="left",
    )
    .join(
        protein_table.rename({"uniprot_id": col_b, "plddt": "plddt2", "length": "len2"}),
        on=col_b,
        how="left",
    )
    .with_columns(
        ((pl.col("plddt1") + pl.col("plddt2")) / 2).alias("mean_plddt_pair")
    )
)

result_df.write_parquet(OUTPUT_PATH)

# Also write CSV for easy inspection
csv_path = OUTPUT_PATH.with_suffix(".csv")
result_df.write_csv(csv_path)

print(f"\nDone! Output: {OUTPUT_PATH}")
print(f"  CSV:  {csv_path}")
print(f"  Shape: {result_df.shape}")
print(f"  Columns: {result_df.columns}")
print(f"\n  Sample (first 5 rows):")
print(result_df.head(5))
PYEOF

echo "[$(date)] Script run complete."
