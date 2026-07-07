from itertools import combinations, product
import polars as pl


def make_pairs(df: pl.DataFrame, column: str, sep: str = " ") -> pl.DataFrame:
    """
    Explode each row's `column` into all unique unordered protein pairs,
    one pair per output row. Keeps all original columns, plus `Protein_1`
    / `Protein_2` for the pair.

    Supports plain tokens ("P1 P2 P3") and paralogue groups
    ("[P1,P2] [P3,P4]") mixed freely -- proteins inside the same "[..]"
    group are treated as alternatives for one slot and are never paired
    with each other, only with proteins from other groups in the row.
    """

    def _row_pairs(s: str):
        groups = [t[1:-1].split(",") if t[0] == "[" else [t] for t in s.split(sep) if t]
        return sorted({
            tuple(sorted((a, b)))
            for g1, g2 in combinations(groups, 2)
            for a, b in product(g1, g2)
        })

    return (
        df.with_columns(
            pl.col(column)
            .map_elements(_row_pairs, return_dtype=pl.List(pl.Array(pl.Utf8, 2)))
            .alias("_p")
        )
        .explode("_p")
        .drop_nulls("_p")
        .with_columns(
            pl.col("_p").arr.get(0).alias("Protein_1"),
            pl.col("_p").arr.get(1).alias("Protein_2"),
        )
        .drop("_p")
    )



def clean_identifiers(
    df: pl.DataFrame,
    column: str = "Identifiers (and stoichiometry) of molecules in complex",
    new_column: str = "Cleaned Identifiers",
) -> pl.DataFrame:
    """
    Clean a pipe-separated identifier column, e.g.:
        "P32797(0)|P38960(0)|CHEBI:49601(0)|Q07921(0)"
    into a space-separated string, dropping CHEBI entries and (n) suffixes:
        "P32797 P38960 Q07921"
    """
    return df.with_columns(
        pl.col(column)
        .str.split("|")
        .list.eval(pl.element().str.replace(r"\(\d+\)$", ""))
        .list.eval(pl.element().filter(~pl.element().str.to_uppercase().str.starts_with("CHEBI")))
        .list.join(" ")
        .alias(new_column)
    )