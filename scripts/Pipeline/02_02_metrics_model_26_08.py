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

INPUT_CSV = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/10_all_CP_complexes/10_all_CP_complexes_pool_pairs_metrics.csv"  
OUT_CSV   = INPUT_CSV.replace(".csv", "_26_08_metrics.csv")

MODELS_PARQUET = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08/summary_models.parquet"
JOIN_KEYS = ["af3_id1", "af3_id2", "batch_id", "seed", "input_name", "sample", "input_type"]
OLD_METRIC_COLS = {
    "chain_pair_iptm",
    "chain_pair_pae_min_min", "chain_pair_pae_min_max", "chain_pair_pae_min_mean",
    "chain_pair_pae_min", "chain_pair_pae_min_recap",
    "chain_pair_iptm_corrected",
}

pbar = tqdm(total=5, desc="unify_pair_metrics")

pbar.set_description("reading csv")
df = pl.read_csv(INPUT_CSV)
pbar.update(1)

pbar.set_description("dropping old metrics")
base = df.drop([c for c in OLD_METRIC_COLS if c in df.columns])
pbar.update(1)

pbar.set_description("scanning 26.08 parquet (filtered)")
needed_keys = base.select(JOIN_KEYS).unique()
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
combined = base.join(models, on=JOIN_KEYS, how="left")
pbar.update(1)

pbar.set_description("checking + writing")
n_missing = combined.filter(pl.col("chain_pair_iptm").is_null()).height
n_total = combined.height
tqdm.write(f"{n_missing} / {n_total} rows failed to match against 26.08")
if n_missing:
    tqdm.write(str(combined.filter(pl.col("chain_pair_iptm").is_null()).select(*JOIN_KEYS)))
combined.write_csv(OUT_CSV)
tqdm.write(f"Wrote {OUT_CSV} ({n_total} rows)")
pbar.update(1)
pbar.close()