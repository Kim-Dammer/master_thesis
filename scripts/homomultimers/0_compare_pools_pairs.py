"""
Compare paired vs pooled AlphaFold3 interaction scores across every homodimer
in the pooled-ppi models table, and attach per-chain pLDDT for the
best-scoring pair model and best-scoring pool model.

Usage:
    uv run 0_compare_pools_pairs.py

Outputs:
    pair_vs_pool_summary.csv   - one row per homodimer:
                                  min/median/max iptm & pae for pair vs. best pool,
                                  plus plddt1/plddt2/mean_plddt for pair and pool
    pair_vs_pool_scatter.png   - pair-max-iptm vs pool-max-iptm scatter
"""
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import foldcomp
from Bio.PDB import PDBParser

# --- Paths -------------------------------------------------------------
DATA_DIRS = ["/data", "/workspace/data", "/contents/data",
             "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"]
DATA_DIR = next(Path(p) for p in DATA_DIRS if Path(p).is_dir())
OUT_DIR = Path("/cluster/project/beltrao/kdammer/master_thesis/data/Homomltimer")
FC_DB = DATA_DIR / "predictions-db" / "predictions-db"
BATCH_SIZE = 5000

# --- Step 1: job-level iptm/pae summary (as before) ---------------------
lf = pl.scan_parquet(DATA_DIR / "summary_models.parquet").filter(
    pl.col("af3_id1") == pl.col("af3_id2")  # homodimers only
).with_columns(
    (pl.min_horizontal("af3_id1", "af3_id2") + "__" + pl.max_horizontal("af3_id1", "af3_id2")).alias("pair_id")
)

job = lf.group_by(["pair_id", "input_type", "input_name"]).agg(
    n_models=pl.len(),
    iptm_min=pl.col("chain_pair_iptm").min(),
    iptm_median=pl.col("chain_pair_iptm").median(),
    iptm_max=pl.col("chain_pair_iptm").max(),
    pae_min=pl.col("chain_pair_pae_min").min(),
    pae_median=pl.col("chain_pair_pae_min").median(),
    pae_max=pl.col("chain_pair_pae_min").max(),
)

STAT_COLS = ["input_name", "n_models", "iptm_min", "iptm_median", "iptm_max",
             "pae_min", "pae_median", "pae_max"]

pair_lf = (
    job.filter(pl.col("input_type") == "pair").drop("input_type")
    .rename({c: f"{c}_pair" for c in STAT_COLS})
)
pool_lf = job.filter(pl.col("input_type") == "pool").drop("input_type")

n_pools = pool_lf.group_by("pair_id").agg(pl.col("input_name").n_unique().alias("n_pools"))
pool_best = (
    pool_lf.sort("iptm_max", descending=True).unique(subset="pair_id", keep="first")
    .rename({c: f"{c}_pool" for c in STAT_COLS})
)

summary = (
    pair_lf.join(pool_best, on="pair_id", how="full", coalesce=True)
    .join(n_pools, on="pair_id", how="left", coalesce=True)
    .with_columns(
        pl.col("n_pools").fill_null(0),
        (pl.col("iptm_max_pair") - pl.col("iptm_max_pool")).alias("delta_iptm_max"),
        (pl.col("pae_min_pair") - pl.col("pae_min_pool")).alias("delta_pae_min"),
    )
)

# --- Step 2: pick the single best-scoring MODEL row (not just stats) for
# pair and pool, so we know exactly which sample/chain to pull pLDDT from ---
def best_model_keys(lf_raw: pl.LazyFrame, input_type: str, suffix: str) -> pl.LazyFrame:
    return (
        lf_raw.filter(pl.col("input_type") == input_type)
        .sort("chain_pair_iptm", descending=True)
        .unique(subset="pair_id", keep="first")
        .with_columns(
            (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8) + "_" + pl.col("chain_id1"))
            .alias(f"fc_key1_{suffix}"),
            (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8) + "_" + pl.col("chain_id2"))
            .alias(f"fc_key2_{suffix}"),
        )
        .select("pair_id", f"fc_key1_{suffix}", f"fc_key2_{suffix}")
    )

fc_keys = (
    best_model_keys(lf, "pair", "pair")
    .join(best_model_keys(lf, "pool", "pool"), on="pair_id", how="full", coalesce=True)
    .collect()
)

# --- Step 3: fetch structures from foldcomp and compute mean pLDDT ------
all_keys = sorted(set(
    fc_keys["fc_key1_pair"].drop_nulls().to_list()
    + fc_keys["fc_key2_pair"].drop_nulls().to_list()
    + fc_keys["fc_key1_pool"].drop_nulls().to_list()
    + fc_keys["fc_key2_pool"].drop_nulls().to_list()
))
print(f"Unique foldcomp keys to fetch: {len(all_keys)}")

parser = PDBParser(QUIET=True)
plddt_by_key: dict[str, float] = {}
n_found = n_missing = 0

for i in range(0, len(all_keys), BATCH_SIZE):
    batch = all_keys[i:i + BATCH_SIZE]
    with foldcomp.open(str(FC_DB), ids=batch) as db:
        for name, pdb_str in db:
            try:
                struct = parser.get_structure(name, StringIO(pdb_str))
                chain = next(struct[0].get_chains())
                plddt_vals = [r["CA"].get_bfactor() for r in chain.get_residues() if "CA" in r]
                plddt_by_key[name] = float(np.mean(plddt_vals)) if plddt_vals else None
                n_found += 1
            except Exception:
                plddt_by_key[name] = None
                n_missing += 1
    print(f"  fetched {min(i + BATCH_SIZE, len(all_keys))}/{len(all_keys)} "
          f"(found: {n_found}, errors: {n_missing})")

plddt_map = pl.DataFrame({
    "fc_key": list(plddt_by_key.keys()),
    "plddt": list(plddt_by_key.values()),
})

def attach_plddt(df: pl.DataFrame, key_col: str, out_col: str) -> pl.DataFrame:
    return (
        df.join(plddt_map.rename({"fc_key": key_col, "plddt": out_col}), on=key_col, how="left")
    )

fc_keys = (
    fc_keys.pipe(attach_plddt, "fc_key1_pair", "plddt1_pair")
    .pipe(attach_plddt, "fc_key2_pair", "plddt2_pair")
    .pipe(attach_plddt, "fc_key1_pool", "plddt1_pool")
    .pipe(attach_plddt, "fc_key2_pool", "plddt2_pool")
    .with_columns(
        ((pl.col("plddt1_pair") + pl.col("plddt2_pair")) / 2).alias("mean_plddt_pair"),
        ((pl.col("plddt1_pool") + pl.col("plddt2_pool")) / 2).alias("mean_plddt_pool"),
    )
    .select("pair_id", "plddt1_pair", "plddt2_pair", "mean_plddt_pair",
            "plddt1_pool", "plddt2_pool", "mean_plddt_pool")
)

# --- Step 4: merge everything and write out ------------------------------
summary = (
    summary.collect(engine="streaming")
    .join(fc_keys, on="pair_id", how="left")
    .sort("iptm_max_pair", descending=True)
)

summary.write_csv(OUT_DIR / "pair_vs_pool_summary.csv")
print(f"{summary.height} unique homodimers -> pair_vs_pool_summary.csv")

both = summary.drop_nulls(["iptm_max_pair", "iptm_max_pool"])
print(f"{both.height} pairs tested in both pair and pool contexts")
print("correlation (pair max iptm vs pool max iptm):",
      round(both.select(pl.corr("iptm_max_pair", "iptm_max_pool")).item(), 3))
print("mean delta_iptm_max (pair - pool):", round(both["delta_iptm_max"].mean(), 3))
print("median delta_iptm_max (pair - pool):", round(both["delta_iptm_max"].median(), 3))

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(both["iptm_max_pool"], both["iptm_max_pair"], s=15, alpha=0.5)
lim = [0, 1]
ax.plot(lim, lim, ls="--", c="grey", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("pool iptm (max of samples)")
ax.set_ylabel("pair iptm (max of samples)")
ax.set_title(f"pair vs pool iptm, n={both.height} pairs tested in both contexts")
fig.tight_layout()
fig.savefig(OUT_DIR / "pair_vs_pool_scatter.png", dpi=150)
print("wrote pair_vs_pool_scatter.png")