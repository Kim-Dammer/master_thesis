import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    from procompa import get_project_root

    PRJ_ROOT = get_project_root()
    data_dir = PRJ_ROOT / "data"
    return


if __name__ == "__main__":
    app.run()
