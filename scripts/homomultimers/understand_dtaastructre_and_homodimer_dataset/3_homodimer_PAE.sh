#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=lookup_iptm_homodimer
#SBATCH --output=logs/homodimer_PAE_iptm%j.out
#SBATCH --error=logs/homodimer_PAE_iptm%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"
python3 -c "
import polars as pl

path = '/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07/summary_model.parquet'
schema = pl.scan_parquet(path).collect_schema()
print('=== columns ===')
for c in schema.names():
    print(' ', c)

print()
print('=== iptm-related columns ===')
for c in schema.names():
    if 'iptm' in c.lower():
        print(' ', c)

print()
print('=== homodimer rows (af3_id1 == af3_id2) ===')
homo = (pl.scan_parquet(path)
        .filter(pl.col('af3_id1') == pl.col('af3_id2'))
        .collect())

print('count:', homo.height)
print(homo.head(10))

# Save the dataframe
out_path = '/cluster/project/beltrao/kdammer/master_thesis/data/Homomltimer/homodimers_iptm_PAE.parquet'
homo.write_parquet(out_path)
print(f'Successfully saved homodimers to: {out_path}')
"