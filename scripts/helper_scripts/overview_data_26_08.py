import polars as pl

base = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08"
out = "/cluster/project/beltrao/kdammer/master_thesis/scripts/helper_scripts/logs/overview_data_26_08.txt"

files = ["summary_models.parquet", "summary_pairs.parquet", "summary_confidences.parquet", "pools.parquet", "proteins.parquet"]

CHUNK = 10  # columns per chunk

lines = []
for fname in files:
    df = pl.read_parquet(f"{base}/{fname}", n_rows=3)
    lines.append(f"\n=== {fname} ({df.shape[1]} cols) ===")
    cols = df.columns
    for i in range(0, len(cols), CHUNK):
        chunk_cols = cols[i:i+CHUNK]
        with pl.Config(tbl_cols=CHUNK, tbl_width_chars=200, tbl_rows=3):
            lines.append(f"--- cols {i+1}-{i+len(chunk_cols)} ---")
            lines.append(str(df.select(chunk_cols)))

output = "\n".join(lines)
print(output)
with open(out, "w") as f:
    f.write(output)
print(f"\nwritten to {out}")