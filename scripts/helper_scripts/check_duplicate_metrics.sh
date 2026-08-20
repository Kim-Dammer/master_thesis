#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=check_duplicate_metrics
#SBATCH --output=logs/check_duplicate_metrics_%j.out
#SBATCH --error=logs/check_duplicate_metrics_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

# check_duplicate_metrics.sh
#
# Diagnoses the 57 groups of (complex_ac, pair, seed, sample, batch_id,
# chain_id1, chain_id2) that had identical descriptor columns but different
# chain_pair_iptm/pae_min values in combined_pool_pairs_metrics.csv.
#
# Step 1: resolve the UniProt IDs to their actual af3_id values via
# proteins.parquet (rather than guessing case).
# Step 2: pull every summary_models.parquet row for that pair at
# seed=4, sample=1, to see if the duplication already exists upstream in
# Jurgen's snapshot or is introduced by our own join.
#
# Usage:
#   sbatch check_duplicate_metrics.sh
set -euo pipefail
mkdir -p logs

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

DATA_DIR="${POOLED_PPI_DATA_DIR:-/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08}"

python - "$DATA_DIR" <<'PY'
import sys
import polars as pl

data_dir = sys.argv[1]

print("=== Step 1: resolve af3_id for P02309 / P04911 ===")
proteins = (
    pl.scan_parquet(f"{data_dir}/proteins.parquet")
    .filter(pl.col("uniprot_id").str.to_uppercase().is_in(["P02309", "P04911"]))
    .select("af3_id", "uniprot_id")
    .collect()
)
print(proteins)

if proteins.height != 2:
    print(f"WARNING: expected 2 matching proteins, got {proteins.height} - "
          "check uniprot_id values / casing before trusting step 2.")
    sys.exit(1)

id_map = dict(zip(proteins["uniprot_id"].to_list(), proteins["af3_id"].to_list()))
id1 = id_map["P02309"]
id2 = id_map["P04911"]
print(f"af3_id for P02309: {id1!r}")
print(f"af3_id for P04911: {id2!r}")

print()
print("=== Step 2: raw summary_models.parquet rows at seed=4, sample=1 ===")
raw = (
    pl.scan_parquet(f"{data_dir}/summary_models.parquet")
    .filter(
        ((pl.col("af3_id1") == id1) & (pl.col("af3_id2") == id2))
        | ((pl.col("af3_id1") == id2) & (pl.col("af3_id2") == id1))
    )
    .filter(pl.col("seed") == 4, pl.col("sample") == 1)
    .collect()
)
print(f"n_rows: {raw.height}")
print(raw.select(
    "af3_id1", "af3_id2", "batch_id", "seed", "sample",
    "chain_id1", "chain_id2", "input_name",
    "chain_pair_iptm", "chain_pair_pae_min",
))

if raw.height > 1:
    n_unique_descriptors = raw.select(
        "af3_id1", "af3_id2", "batch_id", "seed", "sample", "chain_id1", "chain_id2"
    ).unique().height
    print()
    print(f"Distinct (af3_id1, af3_id2, batch_id, seed, sample, chain_id1, chain_id2) "
          f"combos among these {raw.height} rows: {n_unique_descriptors}")
    if n_unique_descriptors < raw.height:
        print("=> Duplication exists UPSTREAM in the raw parquet itself "
              "(same descriptors, multiple metric values). Flag to Jurgen.")
    else:
        print("=> Raw parquet rows are all distinct on descriptors - "
              "duplication must be introduced downstream, in our join/concat.")
PY
