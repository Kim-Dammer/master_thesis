"""
Adding match class for best pdb match for each complex - script version of
07_preparing_Complex_to_pdb_mapping section "Find best partial / superset complexes"

Example:
uv run 5_add_match_class_to_best_pdb.py \
        --mmeseq_homogy_results /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/homology_pdb_mapping \
        --complex_portal_tsv /cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv \
        --exact_pdb_match_csv /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/exact_pdb_match/best_coverage_complex/best_coverage_download_log.csv \
        --output /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/all_pdb_matches.csv
"""

import argparse
from pathlib import Path

import polars as pl
from procompa import get_project_root
from procompa.helpers import clean_identifiers, count_elements

PRJ_ROOT = get_project_root()
DEFAULT_DATA_DIR = PRJ_ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Add match class to best pdb match for each complex")
    parser.add_argument(
        "--mmeseq_homogy_results",
        type=Path,
        required=True,
        help="Path to the directory containing the mmseq homology results "
             "(must contain best_match_per_complex.csv, exact_matches.csv, summary_per_complex.csv)",
    )
    parser.add_argument(
        "--complex_portal_tsv",
        type=Path,
        default=DEFAULT_DATA_DIR / "Complex_Portal" / "Saccharomyces_cerevisiae_ComplexTab.tsv",
        help="Path to the Complex Portal tsv file",
    )
    parser.add_argument(
        "--exact_pdb_match_csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "complete_complex_pdb_mapping_v2" / "exact_pdb_match"
                / "best_coverage_complex" / "best_coverage_download_log.csv",
        help="Path to the exact pdb match download log csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the final concatenated all_pdb_matches table to (csv)",
    )
    return parser.parse_args()


def classify_best_match_per_complex(mmeseq_homogy_results: Path) -> pl.DataFrame:
    """
    1. Classify complexes where only a partial match has been found as partial, superset,
    or not sufficient.

    Partial: All pdb proteins are in the CP complex, pdb_complex captures at least 50.01 %
    of the proteins in the CP complex. -> CP complex bigger
    Superset: All CP complex proteins are in the pdb complex, pdb_complex consists of at
    least 50.01 % of the proteins in the pdb complex. -> pdb complex bigger

    partial: extra_proteins = 0 (equivalent to n_distinct_proteins_pdb - n_matched),
        coverage_frac > 0.50 (equivalent to n_matched / n_prot > 0.50)
    superset: n_matched = n_proteins, coverage_frac > 0.50 (n_matched / n_prot > 0.5001)
    """
    best_match_per_complex = pl.read_csv(mmeseq_homogy_results / "best_match_per_complex.csv")

    best_match_per_complex = best_match_per_complex.with_columns(
        match_class=pl.when(
            (pl.col("extra_proteins") == 0) & (pl.col("coverage_frac") > 0.50)
        )
        .then(pl.lit("partial"))
        .when(
            (pl.col("n_matched") == pl.col("n_proteins")) & (pl.col("coverage_frac") > 0.50)
        )
        .then(pl.lit("superset"))
        .otherwise(pl.lit("no_sufficient_match"))
    )

    return best_match_per_complex.select(["complex_ac", "n_proteins", "pdb_id", "match_class"])


def get_exact_homology_matches(mmeseq_homogy_results: Path) -> pl.DataFrame:
    """2. Complexes where the mmseq pipeline found an exact match based on homology search."""
    exact_homology_matches = pl.read_csv(mmeseq_homogy_results / "exact_matches.csv")

    # keep first match for each cp complex
    exact_homology_matches = (
        exact_homology_matches.group_by("complex_ac", maintain_order=True)
        .first()
        .with_columns(match_class=pl.lit("exact_homology_match"))
    )

    return exact_homology_matches.select(["complex_ac", "n_proteins", "pdb_id", "match_class"])


def get_no_homology_match(mmeseq_homogy_results: Path) -> pl.DataFrame:
    """3. Complexes for which the homology search didn't find any match (not even one protein)."""
    no_homology_match = pl.read_csv(mmeseq_homogy_results / "summary_per_complex.csv").filter(
        ~pl.col("best_match_pdb").is_not_null()
    )

    no_homology_match = no_homology_match.with_columns(
        match_class=pl.lit("no_homology_match"),
        pdb_id=pl.lit("none"),
    )

    return no_homology_match.select(["complex_ac", "n_proteins", "pdb_id", "match_class"])


def get_exact_pdb_match(complex_portal_tsv: Path, exact_pdb_match_csv: Path) -> pl.DataFrame:
    """4. Complexes for which a match has been found without homology search, based on uniprot id."""
    # get amount of proteins for each complex / how many proteins are in each complex
    complex_portal_df = pl.read_csv(complex_portal_tsv, separator="\t")
    complex_portal_df = clean_identifiers(
        complex_portal_df.select(["#Complex ac", "Identifiers (and stoichiometry) of molecules in complex"])
    )
    complex_portal_df = count_elements(
        complex_portal_df, column="Cleaned Identifiers", separator=" ", new_column_name="n_proteins"
    )
    complex_portal_df = complex_portal_df.select(["#Complex ac", "n_proteins"])

    exact_pdb_match = pl.read_csv(exact_pdb_match_csv)

    # add information on complex size
    exact_pdb_match = (
        exact_pdb_match.join(
            complex_portal_df.select([pl.col("#Complex ac"), pl.col("n_proteins")]),
            left_on="Complex_ac",
            right_on="#Complex ac",
            how="left",
        ).with_columns(match_class=pl.lit("exact_pdb_match"))
    )

    exact_pdb_match = exact_pdb_match.select(["Complex_ac", "n_proteins", "pdb_id", "match_class"])
    return exact_pdb_match.rename({"Complex_ac": "complex_ac"}).select(
        ["complex_ac", "n_proteins", "pdb_id", "match_class"]
    )


def keep_only_exact_match_if_present(df: pl.DataFrame) -> pl.DataFrame:
    """For complexes that have found matches with and without homology, keep the exact match."""
    exact_complexes = (
        df.filter(pl.col("match_class") == "exact_pdb_match").get_column("complex_ac").unique().to_list()
    )

    return df.filter(
        ~pl.col("complex_ac").is_in(exact_complexes) | (pl.col("match_class") == "exact_pdb_match")
    )


def build_all_pdb_matches(
    mmeseq_homogy_results: Path, complex_portal_tsv: Path, exact_pdb_match_csv: Path
) -> pl.DataFrame:
    pdb_match_dataframes = [
        get_exact_pdb_match(complex_portal_tsv, exact_pdb_match_csv),
        get_exact_homology_matches(mmeseq_homogy_results),
        get_no_homology_match(mmeseq_homogy_results),
        classify_best_match_per_complex(mmeseq_homogy_results),
    ]

    pdb_match_dataframes = [
        pdb_matches.with_columns(pl.col("n_proteins").cast(pl.Int64)) for pdb_matches in pdb_match_dataframes
    ]

    all_pdb_matches = pl.concat(pdb_match_dataframes)

    return keep_only_exact_match_if_present(all_pdb_matches)


def main():
    args = parse_args()

    all_pdb_matches = build_all_pdb_matches(
        mmeseq_homogy_results=args.mmeseq_homogy_results,
        complex_portal_tsv=args.complex_portal_tsv,
        exact_pdb_match_csv=args.exact_pdb_match_csv,
    )

    # count entries for each match class
    all_pdb_matches_summary = all_pdb_matches.group_by(pl.col("match_class")).agg(
        pl.len().alias("n_complexes")
    ).sort("match_class")

    print(all_pdb_matches_summary)

    # how for complexes bigger 2, should be 374 in total
    bigger_complexes_pdb_matches_summary = (
        all_pdb_matches.filter(pl.col("n_proteins") > 2)
        .group_by(pl.col("match_class"))
        .agg(pl.len().alias("n_complexes"))
        .sort("match_class")
    )

    print(bigger_complexes_pdb_matches_summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_pdb_matches.write_csv(args.output)
    print(f"Wrote final concatenated table with {all_pdb_matches.height} rows to {args.output}")


if __name__ == "__main__":
    main()