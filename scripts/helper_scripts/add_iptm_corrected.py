import polars as pl

INPUT_CSV = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/10_all_CP_complexes/10_all_CP_complexes_pool_pairs_metrics_with_plddt.csv"
OUT_CSV   = INPUT_CSV.replace(".csv", "_corrected.csv")
MODELS_PARQUET = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08/summary_models.parquet"
JOIN_KEYS = ["af3_id1", "af3_id2", "batch_id", "seed", "input_name", "sample", "input_type"]

df = pl.read_csv(INPUT_CSV)
needed_keys = df.select(JOIN_KEYS).unique()

iptm_corrected = (
    pl.scan_parquet(MODELS_PARQUET)
    .select(*JOIN_KEYS, "chain_pair_iptm_corrected")
    .join(needed_keys.lazy(), on=JOIN_KEYS, how="inner")
    .collect()
)

combined = df.join(iptm_corrected, on=JOIN_KEYS, how="left")

n_missing = combined["chain_pair_iptm_corrected"].is_null().sum()
print(f"{n_missing} / {combined.height} rows missing chain_pair_iptm_corrected")
combined.write_csv(OUT_CSV)
print(f"Wrote {OUT_CSV}")