#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=unique_input_type
#SBATCH --output=logs/unique_input_type_%j.out
#SBATCH --error=logs/unique_input_type_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

python3 -c "
import polars as pl

path = '/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07/summary_pairs.parquet'

unique_vals = (
    pl.scan_parquet(path)
    .select('input_type')
    .unique()
    .collect(engine='streaming')
    .sort('input_type')
)

print('=== Unique Input Types ===')
for v in unique_vals['input_type']:
    print(v)

print(f'\nTotal unique values: {unique_vals.height}')
"
