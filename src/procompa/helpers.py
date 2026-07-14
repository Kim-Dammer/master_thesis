from itertools import combinations, product
import polars as pl
import re


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



def count_elements(
        df: pl.DataFrame, 
        column: str  = "Cleaned Identifiers", 
        separator: str = " ", 
        new_column_name: str = "n_proteins"
        ) -> pl.DataFrame:
    """
    Splits a string column by a separator and counts the number of elements,
    adding the result as a new column.
    """
    return df.with_columns(
        pl.col(column)
        .str.split(separator)
        .list.len()
        .alias(new_column_name)
    )




def add_interesting_stoichiometry(
    df: pl.DataFrame,
    column: str,
    check_stoichiometry_given: bool = False,
) -> pl.DataFrame:
    """
    Adds 'interesting_stoichiometry': True if any protein's stoichiometry is >= 2.
    Optionally adds 'stoichiometry_given': True if all proteins have stoichiometry != 0,
    False if any protein has stoichiometry 0, and null if no protein entries are present.
    Non-protein entries (CPX-..., CHEBI:...) are ignored for both checks.
    """
    uniprot_entry_re = re.compile(r"([A-Za-z0-9:_-]+)\((\d+)\)")

    def _is_uniprot_id(identifier: str) -> bool:
        return not (identifier.startswith("CPX-") or identifier.startswith("CHEBI:"))

    def protein_numbers(value: str | None) -> list[int]:
        if value is None:
            return []
        return [
            int(num)
            for ident, num in uniprot_entry_re.findall(value)
            if _is_uniprot_id(ident)
        ]

    def has_high_stoichiometry(value: str | None) -> bool:
        return any(n >= 2 for n in protein_numbers(value))

    df = df.with_columns(
        pl.col(column)
        .map_elements(has_high_stoichiometry, return_dtype=pl.Boolean)
        .alias("interesting_stoichiometry")
    )

    if check_stoichiometry_given:
        def stoichiometry_given(value: str | None) -> bool | None:
            nums = protein_numbers(value)
            if not nums:
                return None
            return all(n != 0 for n in nums)

        df = df.with_columns(
            pl.col(column)
            .map_elements(stoichiometry_given, return_dtype=pl.Boolean)
            .alias("stoichiometry_given")
        )

    return df


def get_unique_proteins(df: pl.DataFrame, column: str) -> list[str]:
    """
    Returns a sorted list of all unique protein (UniProt-style) identifiers
    found in the given column. Entries like CPX-... and CHEBI:... are excluded.
    """
    uniprot_entry_re = re.compile(r"([A-Za-z0-9:_-]+)\((\d+)\)")

    def _is_uniprot_id(identifier: str) -> bool:
        return not (identifier.startswith("CPX-") or identifier.startswith("CHEBI:"))

    unique_proteins: set[str] = set()

    for value in df[column]:
        if value is None:
            continue
        for ident, _num in uniprot_entry_re.findall(value):
            if _is_uniprot_id(ident):
                unique_proteins.add(ident)

    return sorted(unique_proteins)