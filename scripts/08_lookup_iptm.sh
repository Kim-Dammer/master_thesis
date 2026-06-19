#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_iptm
#SBATCH --output=logs/lookup_iptm_%j.out
#SBATCH --error=logs/lookup_iptm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G

# ===========================================================================
# lookup_iptm.sbatch — High-speed vectorized ipTM lookup using Polars.
#
# Usage:
#    sbatch 08_lookup_iptm.sh
# ===========================================================================

set -euo pipefail

# --- Config ----------------------------------------------------------------
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
# ---------------------------------------------------------------------------

source "${VENV}/bin/activate"

mkdir -p logs

echo "[$(date)] Starting high-performance ipTM lookup..."

python - <<'EOF'
from pathlib import Path
import sys
import polars as pl
import pooled_ppi

# --- Hardcoded Dataset Paths ---
CSV_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/notebooks/Dataframes/Yeast_Map/yeastmap_complex_pairs_with_scores_incl_db.csv")
POOLED_PPI_DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.03"
OUTPUT_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/iptm_scores.csv")

PAIR_COL = "complex_pairs"

def parse_pairs(entry: str | None) -> list[list[str]] | None:
    """Parse a 'X-Y' or 'X-Y-Z' entry into sorted pairs."""
    if entry is None:
        return None
    parts = [p.strip() for p in str(entry).split("-") if p.strip()]
    pairs = []
    n = len(parts)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = parts[i], parts[j]
            if p1 > p2:
                p1, p2 = p2, p1
            pairs.append([p1, p2])
    return pairs

def main():
    print(f"Reading {CSV_PATH} ...")
    df = pl.read_csv(CSV_PATH)
    print(f"  {df.height} rows, columns: {df.columns}")

    if PAIR_COL not in df.columns:
        candidates = [c for c in df.columns if "complex_pair" in c.lower() or "pair" in c.lower()]
        if candidates:
            print(f"  Column '{PAIR_COL}' not found. Candidates with 'pair': {candidates}")
        sys.exit(f"Column '{PAIR_COL}' not found in CSV.")

    df = df.with_row_index("row_id")

    print("Parsing target pairs ...")
    pairs_df = (
        df.select(["row_id", PAIR_COL])
        .with_columns(
            pl.col(PAIR_COL)
            .map_elements(parse_pairs, return_dtype=pl.List(pl.List(pl.String)))
            .alias("pairs")
        )
        .explode("pairs")
        .drop_nulls("pairs")
        .with_columns([
            pl.col("pairs").list.get(0).alias("p1"),
            pl.col("pairs").list.get(1).alias("p2"),
        ])
    )

    unique_pairs = pairs_df.select(["p1", "p2"]).unique()
    print(f"  Found {unique_pairs.height} unique pairs across {df.height} rows")

    print(f"Loading pooled-PPI DB from {POOLED_PPI_DB} ...")
    pp = pooled_ppi.PooledPredictionsDb(POOLED_PPI_DB)
    
    # Efficient conversion of internal DB pandas frame into Polars
    pp_df = pl.from_pandas(pp.pairs)
    print(f"  pp_df shape: {pp_df.shape}")

    # Dynamically discover ipTM target column
    iptm_col = None
    for candidate in [
        "chain_pair_iptm_mean_corrected",
        "chain_pair_iptm",
        "iptm",
        "ipTM",
        "average_iptm",
    ]:
        if candidate in pp_df.columns:
            iptm_col = candidate
            break

    if iptm_col is None:
        iptm_candidates = [c for c in pp_df.columns if "iptm" in c.lower()]
        if iptm_candidates:
            iptm_col = iptm_candidates[0]
        else:
            sys.exit("Cannot proceed without an ipTM column.")

    print(f"  Using ipTM column: {iptm_col}")

    # Normalize DB layout horizontally (p1 <= p2) for infallible hashing
    print("Normalizing DB structures and index matching...")
    lookup_df = (
    pp_df.select(["uniprot_id1", "uniprot_id2", iptm_col])
    .with_columns([
        pl.min_horizontal("uniprot_id1", "uniprot_id2").alias("p1"),
        pl.max_horizontal("uniprot_id1", "uniprot_id2").alias("p2"),
    ])
    .select(["p1", "p2", iptm_col])
    .group_by(["p1", "p2"])
    .agg(pl.col(iptm_col).max())  # <-- take best score, not first
    )

    # Fast hash-join mapping execution
    lookup_results = unique_pairs.join(lookup_df, on=["p1", "p2"], how="left")
    found_count = lookup_results.filter(pl.col(iptm_col).is_not_null()).height
    missing_count = lookup_results.filter(pl.col(iptm_col).is_null()).height
    print(f"  Looked up {unique_pairs.height} pairs: {found_count} found, {missing_count} missing.")

    print("Mapping ipTM matrix keys back to root dataset rows ...")
    joined_pairs = pairs_df.join(lookup_df, on=["p1", "p2"], how="left")

    joined_pairs = joined_pairs.with_columns(
        pl.col(iptm_col)
        .map_elements(lambda x: f"{x:.4f}" if x is not None else "NaN", return_dtype=pl.String)
        .alias("formatted_score")
    )
    joined_pairs = joined_pairs.with_columns(
        pl.format("{}_{}:{}", pl.col("p1"), pl.col("p2"), pl.col("formatted_score")).alias("pair_score_str")
    )

    agg_df = joined_pairs.group_by("row_id").agg([
        pl.col(iptm_col).max().alias("iptm"),
        pl.col("pair_score_str").str.join(";").alias("iptm_all_pairs")  
    ])

    final_df = (
        df.join(agg_df, on="row_id", how="left")
        .with_columns(pl.col("iptm_all_pairs").fill_null(""))
        .sort("row_id")
        .drop("row_id")
    )

    n_with_iptm = final_df.filter(pl.col("iptm").is_not_null()).height
    print(f"  {n_with_iptm}/{final_df.height} rows processed with matching metrics.")

    final_df.write_csv(OUTPUT_PATH)
    print(f"\nSaved file output to: {OUTPUT_PATH}")
    print(f"  Shape: {final_df.shape}")

if __name__ == "__main__":
    main()
EOF

echo "[$(date)] Script run complete."