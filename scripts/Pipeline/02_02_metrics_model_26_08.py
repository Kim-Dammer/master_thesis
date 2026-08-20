"""
Unify chain_pair_iptm / chain_pair_iptm_corrected / chain_pair_pae_min /
chain_pair_pae_min_recap across benchmark part one and part two by re-pulling
all four metrics fresh from the current (26.08) summary_models.parquet.

Part one was built against the old (26.07) snapshot and has
chain_pair_pae_min_min/_max/_mean instead of chain_pair_pae_min/_recap, and
neither part has chain_pair_iptm_corrected yet. Since 26.08 is a superset of
26.07, both parts' pairs should still resolve there, giving a single
consistent source for both.
"""

import polars as pl
from tqdm import tqdm

PART1_CSV = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/7_benchmark_part_one/7_benchmark_part_one_pool_pairs_metrics.csv"
PART2_CSV = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/8_benchmark_part_two/8_benchmark_part_two_pool_pairs_metrics.csv"

MODELS_PARQUET = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08/summary_models.parquet"
JOIN_KEYS = ["af3_id1", "af3_id2", "batch_id", "seed", "input_name", "sample", "input_type"]
OLD_METRIC_COLS = {
    "chain_pair_iptm",
    "chain_pair_pae_min_min", "chain_pair_pae_min_max", "chain_pair_pae_min_mean",
    "chain_pair_pae_min", "chain_pair_pae_min_recap",
}

pbar = tqdm(total=7, desc="unify_pair_metrics")

pbar.set_description("reading csvs")
part1 = pl.read_csv(PART1_CSV)
part2 = pl.read_csv(PART2_CSV)
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
# Build the set of exact join-key combos we actually need, so predicate
# pushdown only reads matching rows instead of the full ~70M-row table.
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
tqdm.write(f"{n_missing} / {n_total} rows failed to match against 26.08")
if n_missing:
    tqdm.write("WARNING: some rows did not resolve against the 26.08 snapshot - "
               "inspect these before trusting the combined dataset:")
    tqdm.write(str(combined.filter(pl.col("chain_pair_iptm").is_null()).select(*JOIN_KEYS)))
pbar.update(1)


pbar.set_description("writing output")
out_path = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/8_benchmark_part_two/combined_pool_pairs_metrics.csv"
combined.write_csv(out_path)
tqdm.write(f"Wrote unified metrics to {out_path} ({n_total} rows)")
pbar.update(1)
pbar.close()
