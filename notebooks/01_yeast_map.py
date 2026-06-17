import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import marimo as mo

    return mo, pl


@app.cell
def _():
    from procompa import get_project_root

    PRJ_ROOT = get_project_root()
    data_dir = PRJ_ROOT / "data"
    return (data_dir,)


@app.cell
def _(data_dir, pl):
    stoic_data = pl.read_csv(data_dir / "Stoic/data_file_stoic.csv")
    return (stoic_data,)


@app.cell
def _(stoic_data):
    stoic_data["seq_name", "split", "sequence" ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    yeast complex db
    """)
    return


@app.cell
def _(data_dir, pl):
    sac_complex_tab = pl.read_csv(
        data_dir / "Complex_Portal/Saccharomyces cerevisiae_ComplexTab.tsv",
        separator="\t",
        truncate_ragged_lines=True,
    )
    return (sac_complex_tab,)


@app.cell
def _(sac_complex_tab):
    sac_complex_tab
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Human Complex database
    """)
    return


@app.cell
def _():
    # hu_complex_tab = pl.read_csv(
    #     data_dir / "Complex_Portal/Human_ComplexTab.tsv",
    #     separator="\t",
    #     ignore_errors=True,
    #     truncate_ragged_lines=True,
    # )

    # hu_complex_tab
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    filter for complexes where stoichiometry is known for all proteins
    """)
    return


@app.cell
def _(pl, sac_complex_tab):
    known_stoichiometry = sac_complex_tab.filter(~pl.col("Identifiers (and stoichiometry) of molecules in complex").str.contains(r"\(0\)"))
    known_all_stoichiometry = known_stoichiometry.filter(~pl.col("Expanded participant list").str.contains(r"\(0\)")) #doenst filter anything else out, just to check
    return (known_stoichiometry,)


@app.cell
def _(pl, sac_complex_tab):
    unknown_stoichiometry = sac_complex_tab.filter(pl.col("Identifiers (and stoichiometry) of molecules in complex").str.contains(r"\(0\)"))
    return (unknown_stoichiometry,)


@app.cell
def _(known_stoichiometry):
    known_stoichiometry.head(1)
    return


@app.cell
def _(unknown_stoichiometry):
    unknown_stoichiometry
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add confidence score
    """)
    return


@app.cell
def _(known_stoichiometry, pl):
    eco_map = {
        "ECO:0000353": 5,
        "ECO:0005543": 5,
        "ECO:0005610": 4,
        "ECO:0005544": 4,
        "ECO:0005546": 4,
        "ECO:0005547": 3,
        "ECO:0007653": 2,
        "ECO:0008004": 1,
    }

    known_stoichiometry_with_confidence = known_stoichiometry.with_columns(
        pl.col("Evidence Code")
        .str.extract_all(r"ECO:\d{7}")
        .map_elements(lambda xs: max((eco_map.get(x, 0) for x in xs), default=None))
        .alias("confidence")
    )
    return (known_stoichiometry_with_confidence,)


@app.cell
def _(known_stoichiometry_with_confidence):
    known_stoichiometry_with_confidence.group_by("confidence").len().sort("confidence", descending=True)
    return


if __name__ == "__main__":
    app.run()
