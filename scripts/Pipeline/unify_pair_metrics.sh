#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=unify_pair_metrics
#SBATCH --output=logs/unify_pair_metrics_%j.out
#SBATCH --error=logs/unify_pair_metrics_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20G

# unify_pair_metrics.sh  <part1.csv>  <part2.csv>  <output_dir>
#
# Combines two benchmark-pair-metrics CSVs (built at different times against
# different pooled-ppi-yeast snapshots, so their metric columns differ) into
# one consistent table. All old metric columns (chain_pair_iptm,
# chain_pair_pae_min_min/_max/_mean, chain_pair_pae_min/_recap) are dropped
# and re-pulled fresh from the current (26.08) summary_models.parquet for
# BOTH parts, so both end up sourced from the same snapshot with the same
# four columns: chain_pair_iptm, chain_pair_iptm_corrected,
# chain_pair_pae_min, chain_pair_pae_min_recap.
#
# The parquet scan is filtered down to only the (af3_id1, af3_id2, batch_id,
# seed, sample, input_type) combos actually present in the input CSVs before
# collecting, so it doesn't pull the full ~70M-row table.
#
# Usage:
#   ./unify_pair_metrics.sh  part1.csv  part2.csv  ./out
#   POOLED_PPI_DATA_DIR=/other/dir ./unify_pair_metrics.sh  p1.csv p2.csv out/
set -euo pipefail
PART1="${1:?need part-one csv as arg 1}"
PART2="${2:?need part-two csv as arg 2}"
OUTDIR="${3:?need output dir as arg 3}"
mkdir -p "$OUTDIR"
mkdir -p logs

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

python - "$PART1" "$PART2" "$OUTDIR" <<'PY'
import sys, os
import polars as pl
from tqdm import tqdm

part1_path, part2_path, outdir = sys.argv[1], sys.argv[2], sys.argv[3]

DATA_DIR = os.environ.get(
    "POOLED_PPI_DATA_DIR",
    "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08")
MODELS_PARQUET = f"{DATA_DIR}/summary_models.parquet"
JOIN_KEYS = ["af3_id1", "af3_id2", "batch_id", "seed", "sample", "input_type"]
OLD_METRIC_COLS = {
    "chain_pair_iptm",
    "chain_pair_pae_min_min", "chain_pair_pae_min_max", "chain_pair_pae_min_mean",
    "chain_pair_pae_min", "chain_pair_pae_min_recap",
}

pbar = tqdm(total=7, desc="unify_pair_metrics")

pbar.set_description("reading csvs")
part1 = pl.read_csv(part1_path)
part2 = pl.read_csv(part2_path)
pbar.update(1)

pbar.set_description("dropping old metrics")
part1_base = part1.drop([c for c in OLD_METRIC_COLS if c in part1.columns])
part2_base = part2.drop([c for c in OLD_METRIC_COLS if c in part2.columns])
assert set(part1_base.columns) == set(part2_base.columns), (
    f"Column mismatch after dropping metrics: "
    f"{set(part1_base.columns) ^ set(part2_base.columns)}"
)
pbar.update(1)

pbar.set_description("concatenating parts")
combined_base = pl.concat([part1_base, part2_base], how="vertical")
pbar.update(1)

pbar.set_description("scanning 26.08 parquet (filtered)")
needed_keys = combined_base.select(JOIN_KEYS).unique()
models = (
    pl.scan_parquet(MODELS_PARQUET)
    .select(
        *JOIN_KEYS,
        "chain_pair_iptm", "chain_pair_iptm_corrected",
        "chain_pair_pae_min", "chain_pair_pae_min_recap",
    )
    .join(needed_keys.lazy(), on=JOIN_KEYS, how="inner")
    .collect()
)
pbar.update(1)

pbar.set_description("joining")
combined = combined_base.join(models, on=JOIN_KEYS, how="left")
pbar.update(1)

pbar.set_description("checking for unmatched rows")
n_missing = combined.filter(pl.col("chain_pair_iptm").is_null()).height
n_total = combined.height
tqdm.write(f"{n_missing} / {n_total} rows failed to match against {MODELS_PARQUET}")
if n_missing:
    tqdm.write("WARNING: some rows did not resolve against the snapshot - "
               "inspect these before trusting the combined dataset:")
    tqdm.write(str(combined.filter(pl.col("chain_pair_iptm").is_null()).select(*JOIN_KEYS)))
pbar.update(1)

pbar.set_description("writing output")
out_path = os.path.join(outdir, "combined_pool_pairs_metrics.csv")
combined.write_csv(out_path)
tqdm.write(f"Wrote unified metrics to {out_path} ({n_total} rows)")
pbar.update(1)
pbar.close()
PY
