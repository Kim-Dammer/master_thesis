#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=inspect_all_parquet
#SBATCH --output=logs/inspect_all_parquet_%j.out
#SBATCH --error=logs/inspect_all_parquet_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

mkdir -p logs

python3 << 'PYEOF'
import glob
import os
import polars as pl

# Directories to inspect, in order
DATA_DIRS = [
    "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.04",
    "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07",
]

# Show full column content, don't truncate the table
pl.Config.set_tbl_cols(-1)          # show all columns
pl.Config.set_tbl_rows(5)           # show 5 rows in head
pl.Config.set_fmt_str_lengths(100)  # don't cut long strings

def inspect_parquet(path):
    print(f"--- {os.path.basename(path)} ---")

    # Columns + dtypes (cheap, no data read)
    try:
        schema = pl.scan_parquet(path).collect_schema()
        print("columns:")
        for c in schema.names():
            print(f"    {c} - {schema[c]}")
    except Exception as e:
        print(f"    [could not read schema: {e}]")
        print()
        return

    # Head (first 5 rows)
    print("head:")
    try:
        print(pl.scan_parquet(path).limit(5).collect())
    except Exception as e:
        print(f"    [could not read head: {e}]")
    print()

for data_dir in DATA_DIRS:
    print("=" * 80)
    print(f"data from {os.path.basename(data_dir)}")
    print("=" * 80)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        print("  [no parquet files found]\n")
        continue

    for path in parquet_files:
        inspect_parquet(path)
PYEOF
