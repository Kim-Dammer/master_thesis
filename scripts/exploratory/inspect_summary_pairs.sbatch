#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=inspect_summary_pairs
#SBATCH --output=logs/inspect_summary_pairs_%j.out
#SBATCH --error=logs/inspect_summary_pairs_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G

# =============================================================================
# Inspect summary_pairs.parquet AND extract the list of yeast proteins that
# appear in at least one COMPUTED HETERODIMER, with a per-protein partner count.
#
# Why this is fast: summary_pairs.parquet is the ~7.6M-row pair INDEX (one row
# per computed heterodimer). polars.scan_parquet is LAZY + streaming, and we
# project only the two ID columns for the extraction -- we never open the
# 124 GB models.sqlite (structure blobs). Runs in seconds, low RAM.
#
# Output:
#   proteins_in_heterodimers.csv   (uniprot_id, af3_id, n_partners)
#   logs/inspect_summary_pairs_<job>.out / .err
#
# Diagnostics (edit the CHECK_PROTEINS / CHECK_PAIR vars below):
#   CHECK_PROTEINS="P32943,P02994,..."   -> per-protein partner counts
#   CHECK_PAIR="P00546 P32943"           -> does this heterodimer pair exist?
# =============================================================================

set -euo pipefail

# ---- config ----------------------------------------------------------------
DATA_DIR="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
PAIRS_PARQUET="${DATA_DIR}/summary_pairs.parquet"
OUT_CSV="proteins_in_heterodimers.csv"

# Diagnostics (leave empty to skip). Comma-separated IDs / space-separated pair.
CHECK_PROTEINS="P32943,P02994,P07260,P10081,Q06410,Q12421"
#CHECK_PAIR="P00546 P32943"
CHECK_PAIR="P00546 P39935"

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

mkdir -p logs

# ---- sanity: file exists ---------------------------------------------------
if [ ! -f "${PAIRS_PARQUET}" ]; then
    echo "ERROR: ${PAIRS_PARQUET} not found. Check DATA_DIR." >&2
    exit 1
fi

# ---- run inspect + extract (single python process) -------------------------
# NOTE on input_type: it is a column of summary_MODELS.parquet (get_models),
# NOT of summary_PAIRS.parquet (get_pairs). We guard it so the script never
# aborts under `set -euo pipefail` if it is absent.
export PAIRS_PARQUET OUT_CSV CHECK_PROTEINS CHECK_PAIR

python3 - << 'PYEOF'
import os
import polars as pl

path = os.environ["PAIRS_PARQUET"]
out_csv = os.environ["OUT_CSV"]
check_proteins = os.environ.get("CHECK_PROTEINS", "").strip()
check_pair = os.environ.get("CHECK_PAIR", "").strip()

lf = pl.scan_parquet(path)
schema = lf.collect_schema()
cols = schema.names()

print("=== columns ===")
for c in cols:
    print(f"  {c} - {schema[c]}")

print()
print("=== iptm-related columns ===")
iptm_cols = [c for c in cols if "iptm" in c.lower()]
print("  " + ", ".join(iptm_cols) if iptm_cols else "  (none)")

print()
if "input_type" in cols:
    print("=== input_type unique values ===")
    print(lf.select("input_type").unique().collect())
else:
    print("=== input_type: NOT a column in summary_pairs.parquet ===")
    print("    (it lives in summary_models.parquet / get_models; expected)")

print()
print("=== sample rows (first 5) ===")
print(lf.limit(5).collect())

# ---- extraction: proteins in computed heterodimers + partner counts --------
# Heterodimers only (af3_id1 != af3_id2). Uppercase to match the CombFold
# pipeline convention (DB stores lowercase af3_id). De-duplicate to the unique
# unordered pair set (a pair can be co-pooled multiple times).
het = (
    lf.filter(pl.col("af3_id1") != pl.col("af3_id2"))
      .select(
          pl.min_horizontal("af3_id1", "af3_id2").str.to_uppercase().alias("a"),
          pl.max_horizontal("af3_id1", "af3_id2").str.to_uppercase().alias("b"),
      )
      .unique()
)

n_pairs = het.select(pl.len()).collect().item()

# Long-form a->b and b->a, then distinct partners per protein.
long = pl.concat([
    het.select(pl.col("a").alias("protein"), pl.col("b").alias("partner")),
    het.select(pl.col("b").alias("protein"), pl.col("a").alias("partner")),
])
cov = (
    long.group_by("protein")
        .agg(pl.col("partner").n_unique().alias("n_partners"))
        .with_columns(
            pl.col("protein").alias("af3_id").str.to_lowercase(),
        )
        .rename({"protein": "uniprot_id"})
        .select("uniprot_id", "af3_id", "n_partners")
        .sort(["n_partners", "uniprot_id"], descending=[True, False])
        .collect()
)

print()
print("=== SUMMARY ===")
print(f"  unique heterodimer pairs          : {n_pairs:,}")
print(f"  proteins in computed heterodimers : {cov.height:,}")
if cov.height:
    pc = cov["n_partners"]
    print(f"  partners per protein: min={pc.min()}, median={int(pc.median())}, "
          f"mean={pc.mean():.1f}, max={pc.max()}")

cov.write_csv(out_csv)
print(f"\nWrote {cov.height:,} proteins -> {out_csv}")

# ---- diagnostic: per-protein partner counts --------------------------------
if check_proteins:
    wanted = [w.strip().upper() for w in check_proteins.split(",") if w.strip()]
    lookup = {row["uniprot_id"]: row["n_partners"] for row in cov.iter_rows(named=True)}
    print("\n=== check-proteins ===")
    for w in wanted:
        if w in lookup:
            print(f"  {w}: {lookup[w]} partner(s)")
        else:
            print(f"  {w}: NOT in any computed heterodimer")

# ---- diagnostic: does a specific heterodimer pair exist? -------------------
if check_pair:
    parts = check_pair.split()
    if len(parts) == 2:
        a, b = parts[0].strip().upper(), parts[1].strip().upper()
        lo, hi = (a, b) if a <= b else (b, a)
        exists = (
            het.filter((pl.col("a") == lo) & (pl.col("b") == hi))
               .select(pl.len()).collect().item()
        )
        print(f"\n=== check-pair ===")
        print(f"  Pair {a}-{b}: {'EXISTS' if exists else 'NOT FOUND'} in computed heterodimers")
    else:
        print(f"\n=== check-pair ===\n  CHECK_PAIR must be two IDs separated by a space, got: {check_pair!r}")
PYEOF

echo "[sbatch] done. See ${OUT_CSV}"
