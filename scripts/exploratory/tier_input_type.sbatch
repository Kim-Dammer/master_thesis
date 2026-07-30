#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=tier_input_type
#SBATCH --output=logs/tier_input_type_%j.out
#SBATCH --error=logs/tier_input_type_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=50G

# =============================================================================
# Test the hypothesis: LOW-tier proteins (few partners) entered the DB through
# dedicated PAIR runs (input_type='pair'), while HIGH-tier proteins (many
# partners) came from POOL runs (input_type='pool').
# aka the proteins that have only very few pairs (less than 4000) were not run by jürgen in the pools, but actually by paola and thus shoudl have input type pair
#
# Method:
#   1. Load proteins_in_heterodimers.csv (from the previous extraction).
#   2. Classify each protein HIGH (>=4000 partners) vs LOW (<4000).
#   3. Scan summary_models.parquet (lazy/streaming), project only the columns
#      we need: af3_id1, af3_id2, input_type. Never touch models.sqlite.
#   4. For each model row, uppercase the IDs, look up each protein's tier,
#      and tally: (tier1 x tier2) x input_type.
#   5. Print a 2x2x2 contingency table + per-protein input_type breakdown.
#
# Output:
#   tier_input_type_table.csv        (tier1 x tier2 x input_type -> count)
#   tier_protein_input_type.csv      (per-protein: n_pair, n_pool, tier)
#   logs/tier_input_type_<job>.out / .err
# =============================================================================

set -euo pipefail

# ---- config ----------------------------------------------------------------
DATA_DIR="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
MODELS_PARQUET="${DATA_DIR}/summary_models.parquet"
PROTEINS_CSV="proteins_in_heterodimers.csv"   # from inspect_summary_pairs.sbatch
TIER_THRESHOLD=4000

# Proteins to print full partner lists for. Either:
#   (a) a comma-separated list in CHECK_PROTEIN_PARTNERS, or
#   (b) a CSV file path in CHECK_PROTEINS_FILE (one ID per line, header "identifier").
#       Only UniProt-format IDs are kept ([OPQ]xxxxx); -PRO_ isoform suffixes
#       are stripped; EBI-/URS- IDs are skipped.
# If both are set, the file takes precedence. If neither, defaults to the 10
# missing-pair proteins.
CHECK_PROTEIN_PARTNERS="P00546,P32943,P02994,P32471,P36008,Q06410,Q12421,P07260,P10081,P39935"
CHECK_PROTEINS_FILE="/cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_CP_proteins.csv"

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

mkdir -p logs

if [ ! -f "${MODELS_PARQUET}" ]; then
    echo "ERROR: ${MODELS_PARQUET} not found." >&2; exit 1
fi
if [ ! -f "${PROTEINS_CSV}" ]; then
    echo "ERROR: ${PROTEINS_CSV} not found. Run inspect_summary_pairs.sbatch first." >&2; exit 1
fi

export MODELS_PARQUET PROTEINS_CSV TIER_THRESHOLD CHECK_PROTEIN_PARTNERS CHECK_PROTEINS_FILE

python3 - << 'PYEOF'
import os
import polars as pl

models_path = os.environ["MODELS_PARQUET"]
proteins_csv = os.environ["PROTEINS_CSV"]
threshold = int(os.environ["TIER_THRESHOLD"])
check_proteins_raw = os.environ.get("CHECK_PROTEIN_PARTNERS", "").strip()
check_proteins_file = os.environ.get("CHECK_PROTEINS_FILE", "").strip()

# ---- 1. Load protein tiers from the CSV ------------------------------------
prot = pl.read_csv(proteins_csv)
prot = prot.with_columns(
    pl.when(pl.col("n_partners") >= threshold)
      .then(pl.lit("HIGH"))
      .otherwise(pl.lit("LOW"))
      .alias("tier")
)
print(f"Loaded {prot.height:,} proteins from {proteins_csv}")
print(f"  HIGH (>= {threshold} partners): {prot.filter(pl.col('tier')=='HIGH').height:,}")
print(f"  LOW  (<  {threshold} partners): {prot.filter(pl.col('tier')=='LOW').height:,}")
print()

# tier lookup: uniprot_id -> tier
tier_map = dict(zip(prot["uniprot_id"], prot["tier"]))

# ---- 2. Inspect summary_models.parquet schema ------------------------------
lf = pl.scan_parquet(models_path)
schema = lf.collect_schema()
cols = schema.names()
print("=== summary_models.parquet columns ===")
for c in cols:
    print(f"  {c} - {schema[c]}")
print()

need = ["af3_id1", "af3_id2", "input_type"]
missing = [c for c in need if c not in cols]
if missing:
    print(f"ERROR: missing columns {missing} in summary_models.parquet")
    raise SystemExit(1)

print("=== input_type unique values ===")
print(lf.select("input_type").unique().collect())
print()

# ---- 3. Build the contingency table ----------------------------------------
tier_df = pl.DataFrame({
    "uniprot_id": list(tier_map.keys()),
    "tier": list(tier_map.values()),
}).lazy()

models_proj = (
    lf.select(
        pl.col("af3_id1").str.to_uppercase().alias("id1"),
        pl.col("af3_id2").str.to_uppercase().alias("id2"),
        pl.col("input_type"),
    )
)

with_t1 = models_proj.join(
    tier_df.rename({"uniprot_id": "id1", "tier": "tier1"}), on="id1", how="left"
)
with_t2 = with_t1.join(
    tier_df.rename({"uniprot_id": "id2", "tier": "tier2"}), on="id2", how="left"
)

result = (
    with_t2
    .with_columns(
        pl.when(pl.col("tier1") == "LOW")
          .then(pl.col("tier2") + "x" + pl.col("tier1"))
          .otherwise(pl.col("tier1") + "x" + pl.col("tier2"))
          .alias("pair_tier")
    )
    .group_by("pair_tier", "input_type")
    .agg(pl.len().alias("n_models"))
    .sort(["pair_tier", "input_type"])
    .collect()
)

print("=== CONTINGENCY TABLE: pair_tier x input_type (model counts) ===")
print(result)
print()

pivot = result.pivot(on="input_type", values="n_models", index="pair_tier").fill_null(0)
print("=== PIVOTED ===")
print(pivot)
print()

print("=== ROW PROPORTIONS (within each pair_tier) ===")
for row in pivot.iter_rows(named=True):
    pt = row["pair_tier"]
    total = sum(v for k, v in row.items() if k != "pair_tier")
    parts = [f"{k}={v} ({100*v/total:.1f}%)" for k, v in row.items() if k != "pair_tier" and v > 0]
    print(f"  {pt:>12}: total={total:>10,}  |  " + ", ".join(parts))
print()

# ---- 4. Per-protein input_type breakdown -----------------------------------
# Model-row counts: how many model rows per protein x input_type.
side1 = with_t2.select(pl.col("id1").alias("protein"), pl.col("input_type")).collect()
side2 = with_t2.select(pl.col("id2").alias("protein"), pl.col("input_type")).collect()
model_counts = (
    pl.concat([side1, side2])
      .group_by("protein", "input_type")
      .agg(pl.len().alias("n"))
      .pivot(on="input_type", values="n", index="protein")
      .fill_null(0)
)
for c in ["pair", "pool"]:
    if c not in model_counts.columns:
        model_counts = model_counts.with_columns(pl.lit(0).alias(c))
model_counts = model_counts.rename({"pool": "n_pool", "pair": "n_pair"}).with_columns(
    (pl.col("n_pool") + pl.col("n_pair")).alias("n_total_models")
)

# Distinct-partner counts: how many UNIQUE partners per protein x input_type.
# (n_pool/n_pair count model rows; a pair can have multiple seeds.)
pool_s1 = models_proj.filter(pl.col("input_type") == "pool").select(
    pl.col("id1").alias("protein"), pl.col("id2").alias("partner")).collect()
pool_s2 = models_proj.filter(pl.col("input_type") == "pool").select(
    pl.col("id2").alias("protein"), pl.col("id1").alias("partner")).collect()
partners_pool = (
    pl.concat([pool_s1, pool_s2])
      .group_by("protein")
      .agg(pl.col("partner").n_unique().alias("n_partners_pool"))
)

pair_s1 = models_proj.filter(pl.col("input_type") == "pair").select(
    pl.col("id1").alias("protein"), pl.col("id2").alias("partner")).collect()
pair_s2 = models_proj.filter(pl.col("input_type") == "pair").select(
    pl.col("id2").alias("protein"), pl.col("id1").alias("partner")).collect()
partners_pair = (
    pl.concat([pair_s1, pair_s2])
      .group_by("protein")
      .agg(pl.col("partner").n_unique().alias("n_partners_pair"))
)

# Join everything together
tier_eager = tier_df.collect().rename({"uniprot_id": "protein"})
per_prot = (
    model_counts
    .join(partners_pool, on="protein", how="left")
    .join(partners_pair, on="protein", how="left")
    .join(tier_eager, on="protein", how="inner")
    .with_columns([
        pl.col("n_partners_pool").fill_null(0),
        pl.col("n_partners_pair").fill_null(0),
    ])
    .with_columns(
        (pl.col("n_pair") / (pl.col("n_pair") + pl.col("n_pool"))).alias("frac_pair"),
    )
    .sort(["tier", "n_pool"], descending=[False, True])
)

out_pp = os.environ.get("OUT_PER_PROTEIN", "tier_protein_input_type.csv")
per_prot.write_csv(out_pp)
print(f"Wrote per-protein breakdown -> {out_pp}")
print(f"  columns: {per_prot.columns}")
print(f"  rows: {per_prot.height:,}")
print()

# ---- 5. Summary by tier ----------------------------------------------------
print("=== TIER SUMMARY ===")
for t in ["HIGH", "LOW"]:
    sub = per_prot.filter(pl.col("tier") == t)
    if sub.height == 0:
        continue
    fp = sub["frac_pair"]
    print(f"  {t} tier ({sub.height:,} proteins):")
    print(f"    model rows : n_pool  min={sub['n_pool'].min():,}  median={int(sub['n_pool'].median()):,}  max={sub['n_pool'].max():,}")
    print(f"    model rows : n_pair  min={sub['n_pair'].min():,}  median={int(sub['n_pair'].median()):,}  max={sub['n_pair'].max():,}")
    print(f"    frac_pair  : min={fp.min():.4f}  median={fp.median():.4f}  mean={fp.mean():.4f}  max={fp.max():.4f}")
    print(f"    partners   : n_partners_pool  min={sub['n_partners_pool'].min():,}  median={int(sub['n_partners_pool'].median()):,}  max={sub['n_partners_pool'].max():,}")
    print(f"    partners   : n_partners_pair  min={sub['n_partners_pair'].min():,}  median={int(sub['n_partners_pair'].median()):,}  max={sub['n_partners_pair'].max():,}")
print()

# ---- 6. The missing-pair proteins specifically -----------------------------
print("=== MISSING-PAIR PROTEINS (from the 6 blocked complexes) ===")
missing = ["P00546","P32943","P02994","P32471","P36008","Q06410","Q12421","P07260","P10081","P39935"]
sub = per_prot.filter(pl.col("protein").is_in(missing)).sort("tier")
print(sub.select("protein","tier","n_pool","n_pair","n_total_models","n_partners_pool","n_partners_pair","frac_pair"))
print()

# ---- 7. Per-protein partner lists (pool vs pair) ---------------------------
# For each protein in the check list, write the actual list of distinct
# partners reached via pool runs vs pair runs to protein_partners.csv.
# Protein list source (in priority order):
#   1. CHECK_PROTEINS_FILE  (CSV with "identifier" column; UniProt IDs only)
#   2. CHECK_PROTEIN_PARTNERS (comma-separated string)
import re

uniprot_re = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$")

if check_proteins_file and os.path.isfile(check_proteins_file):
    cp_df = pl.read_csv(check_proteins_file)
    # The column is "identifier"; take the first column if name differs.
    id_col = cp_df.columns[0]
    raw_ids = cp_df[id_col].to_list()
    # Strip -PRO_ isoform suffixes, uppercase, keep only UniProt-format IDs.
    check_proteins = []
    skipped = []
    for r in raw_ids:
        base = str(r).split("-PRO_")[0].strip().upper()
        if uniprot_re.match(base):
            check_proteins.append(base)
        else:
            skipped.append(r)
    print(f"=== CHECK PROTEINS from {check_proteins_file} ===")
    print(f"  total entries: {len(raw_ids)}, UniProt-format: {len(check_proteins)}, skipped: {len(skipped)}")
    if skipped:
        print(f"  skipped (non-UniProt): {skipped[:10]}{'...' if len(skipped)>10 else ''}")
    print()
elif check_proteins_raw:
    check_proteins = [p.strip().upper() for p in check_proteins_raw.split(",") if p.strip()]
else:
    check_proteins = []

if check_proteins:
    print(f"=== PARTNER LISTS for {len(check_proteins)} proteins ===")
    print()

    # Collect only rows where id1 or id2 is one of the check proteins.
    # The is_in filter pushes down to Parquet row groups -> fast.
    relevant = (
        models_proj
        .filter(pl.col("id1").is_in(check_proteins) | pl.col("id2").is_in(check_proteins))
        .collect()
    )

    # Vectorized: for each row, the "protein" is the one in the check list,
    # the "partner" is the other id. Handle both orientations in one pass.
    # Side A: id1 is the check protein, id2 is the partner
    side_a = relevant.filter(pl.col("id1").is_in(check_proteins)).select(
        pl.col("id1").alias("protein"),
        pl.col("id2").alias("partner"),
        pl.col("input_type"),
    )
    # Side B: id2 is the check protein, id1 is the partner
    side_b = relevant.filter(pl.col("id2").is_in(check_proteins)).select(
        pl.col("id2").alias("protein"),
        pl.col("id1").alias("partner"),
        pl.col("input_type"),
    )

    all_partners = (
        pl.concat([side_a, side_b])
          .group_by("protein", "input_type", "partner")
          .agg()  # deduplicate (multiple seeds -> one row per protein x type x partner)
          .sort(["protein", "input_type", "partner"])
    )

    # Write the full partner lists to CSV
    out_partners = os.environ.get("OUT_PARTNERS", "protein_partners.parquet")
    all_partners.write_parquet(out_partners)
    print(f"Wrote partner lists -> {out_partners} ({all_partners.height:,} rows)")
    print()

    # Print summary: per-protein counts
    summary = (
        all_partners
        .group_by("protein", "input_type")
        .agg(pl.len().alias("n_partners"))
        .pivot(on="input_type", values="n_partners", index="protein")
        .fill_null(0)
    )
    for c in ["pair", "pool"]:
        if c not in summary.columns:
            summary = summary.with_columns(pl.lit(0).alias(c))
    summary = summary.rename({"pool": "n_pool_partners", "pair": "n_pair_partners"})
    summary = summary.join(tier_eager, on="protein", how="left").sort("protein")

    print(f"Per-protein partner counts (first 20):")
    print(summary.head(20))
    print(f"... ({summary.height:,} proteins total)")
    print()

    # Print full lists for the 10 missing-pair proteins (if in the check set)
    missing = ["P00546","P32943","P02994","P32471","P36008","Q06410","Q12421","P07260","P10081","P39935"]
    missing_in_set = [p for p in missing if p in check_proteins]
    if missing_in_set:
        print(f"=== MISSING-PAIR PROTEIN PARTNER LISTS ({len(missing_in_set)} in check set) ===")
        for protein in missing_in_set:
            tier = tier_map.get(protein, "?")
            pool_p = sorted(all_partners.filter(
                (pl.col("protein") == protein) & (pl.col("input_type") == "pool")
            )["partner"].to_list())
            pair_p = sorted(all_partners.filter(
                (pl.col("protein") == protein) & (pl.col("input_type") == "pair")
            )["partner"].to_list())
            print(f"  {protein} (tier={tier}):")
            print(f"    pool partners ({len(pool_p)}): {pool_p}")
            print(f"    pair partners ({len(pair_p)}): {pair_p}")
            print()

# ---- 8. Write contingency table --------------------------------------------
out_tbl = os.environ.get("OUT_TABLE", "tier_input_type_table.csv")
result.write_csv(out_tbl)
print(f"Wrote contingency table -> {out_tbl}")
PYEOF

echo "[sbatch] done."
