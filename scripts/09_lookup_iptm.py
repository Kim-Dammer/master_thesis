#!/usr/bin/env python
"""


lookup_iptm.py — Look up ipTM scores for all pairs in the yeastmap CSV.

Reads the CSV with a 'complex:complex_pairs' column (entries like "P25554-P38811"),
looks up ipTM from the pooled-PPI database, adds an 'iptm' column, and saves
the result as iptm_scores.csv.

Usage (on cluster login node):
    source /cluster/project/beltrao/kdammer/master_thesis/.venv/bin/activate
    python lookup_iptm.py
"""

from pathlib import Path
import sys
import polars as pl
import pooled_ppi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/notebooks/Dataframes/Yeast_Map/yeastmap_complex_pairs_with_scores_incl_db.csv")
POOLED_PPI_DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.03"
OUTPUT_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/iptm_scores.csv")

PAIR_COL = "complex_pairs"  # column with entries like "P25554-P38811"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            # Ensure pairs are lexicographically sorted (p1 <= p2)
            if p1 > p2:
                p1, p2 = p2, p1
            pairs.append([p1, p2])
    return pairs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading {CSV_PATH} ...")
    df = pl.read_csv(CSV_PATH)
    print(f"  {df.height} rows, columns: {df.columns}")

    if PAIR_COL not in df.columns:
        # Try to find the right column
        candidates = [c for c in df.columns if "complex_pair" in c.lower() or "pair" in c.lower()]
        if candidates:
            print(f"  Column '{PAIR_COL}' not found. Candidates with 'pair': {candidates}")
        sys.exit(f"Column '{PAIR_COL}' not found in CSV.")

    # Create a unique row identifier to cleanly map exploded pairs back later
    df = df.with_row_index("row_id")

    # --- Parse pairs --------------------------------------------------------
    print("Parsing pairs ...")
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

    # --- Load pooled-PPI DB -------------------------------------------------
    print(f"Loading pooled-PPI DB from {POOLED_PPI_DB} ...")
    pp = pooled_ppi.PooledPredictionsDb(POOLED_PPI_DB)
    
    # Convert the internal pandas DataFrame from the DB interface to Polars
    pp_df = pl.from_pandas(pp.pairs)
    print(f"  pp_df shape: {pp_df.shape}")
    print(f"  pp_df columns: {pp_df.columns}")

    # Find the ipTM column
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
        print("  [ERROR] No ipTM column found in pp.pairs!")
        print(f"  Available columns: {pp_df.columns}")
        iptm_candidates = [c for c in pp_df.columns if "iptm" in c.lower()]
        if iptm_candidates:
            iptm_col = iptm_candidates[0]
            print(f"  Falling back to: {iptm_col}")
        else:
            sys.exit("Cannot proceed without an ipTM column.")

    print(f"  Using ipTM column: {iptm_col}")

    # --- Look up ipTM for each pair -----------------------------------------
    # Normalize the DB layout order (p1 <= p2) using horizontal expressions
    lookup_df = (
        pp_df.select(["uniprot_id1", "uniprot_id2", iptm_col])
        .with_columns([
            pl.min_horizontal("uniprot_id1", "uniprot_id2").alias("p1"),
            pl.max_horizontal("uniprot_id1", "uniprot_id2").alias("p2"),
        ])
        .select(["p1", "p2", iptm_col])
        .unique(subset=["p1", "p2"])
    )

    # Performance logging (matches exact tracking from the original script)
    lookup_results = unique_pairs.join(lookup_df, on=["p1", "p2"], how="left")
    found_count = lookup_results.filter(pl.col(iptm_col).is_not_null()).height
    missing_count = lookup_results.filter(pl.col(iptm_col).is_null()).height
    print(f"  Looked up {unique_pairs.height} pairs: "
          f"{found_count} found, {missing_count} missing in pooled-PPI DB")

    # --- Map back to DataFrame rows -----------------------------------------
    print("Mapping ipTM back to DataFrame rows ...")
    
    # Vectorized left join instead of row-by-row filtering
    joined_pairs = pairs_df.join(lookup_df, on=["p1", "p2"], how="left")

    # Format output scores exactly matching the original `:.4f` constraints
    joined_pairs = joined_pairs.with_columns(
        pl.col(iptm_col)
        .map_elements(lambda x: f"{x:.4f}" if x is not None else "NaN", return_dtype=pl.String)
        .alias("formatted_score")
    )
    joined_pairs = joined_pairs.with_columns(
        pl.format("{}_{}:{}", pl.col("p1"), pl.col("p2"), pl.col("formatted_score")).alias("pair_score_str")
    )

    # Aggregate metric scores back up to the original row context
    agg_df = joined_pairs.group_by("row_id").agg([
        pl.col(iptm_col).max().alias("iptm"),
        pl.col("pair_score_str").str.join(";").alias("iptm_all_pairs")  
    ])

    # Recombine aggregated metrics with the source dataset
    final_df = (
        df.join(agg_df, on="row_id", how="left")
        .with_columns(pl.col("iptm_all_pairs").fill_null(""))
        .sort("row_id")
        .drop("row_id")
    )

    n_with_iptm = final_df.filter(pl.col("iptm").is_not_null()).height
    print(f"  {n_with_iptm}/{final_df.height} rows have at least one ipTM value")

    # --- Save ---------------------------------------------------------------
    final_df.write_csv(OUTPUT_PATH)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"  Shape: {final_df.shape}")
    print(f"  New columns: 'iptm', 'iptm_all_pairs'")


if __name__ == "__main__":
    main()