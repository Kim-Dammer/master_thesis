#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=inspect_summary_pairs
#SBATCH --output=logs/inspect_summary_pairs_%j.out
#SBATCH --error=logs/inspect_summary_pairs_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

python3 -c "
import polars as pl

path = '/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07/summary_pairs.parquet'

schema = pl.scan_parquet(path).collect_schema()
print('=== columns ===')
for c in schema.names():
    print(' ', c, '-', schema[c])

print()
print('=== iptm-related columns ===')
for c in schema.names():
    if 'iptm' in c.lower():
        print(' ', c)

print()
print('=== input_type unique values ===')
print(pl.scan_parquet(path).select('input_type').unique().collect(engine='streaming'))

print()
print('=== sample rows ===')
print(pl.scan_parquet(path).limit(5).collect())
"