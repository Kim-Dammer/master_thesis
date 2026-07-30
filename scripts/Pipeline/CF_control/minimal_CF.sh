#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=minimal_CF
#SBATCH --output=logs/minimal_CF_%j.out
#SBATCH --error=logs/minimal_CF_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15G
set -eo pipefail
VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
source "${VENV}/bin/activate"

mkdir -p logs
# -----------------------------------------------------------------------------
# 1. Target Proteins Definition
# -----------------------------------------------------------------------------
# Change or pass protein IDs as arguments (e.g., ./run_combfold.sh P00546 P20438 P20486)
if [ $# -gt 0 ]; then
  PROTEINS="$*"
else
  PROTEINS="P00546 P20438 P20486"
fi

echo "==> Preparing CombFold inputs for proteins: ${PROTEINS}"

# -----------------------------------------------------------------------------
# 2. Python Input Preparation (subunits.json & PDB extraction)
# -----------------------------------------------------------------------------
python - << EOF
import collections
import json
import sys
from pathlib import Path
import pandas as pd
import af3io
import pooled_ppi
from procompa import get_data_dir

# Set up list of target proteins
protein_list = "${PROTEINS}".split()
proteins = set(protein_list)
complex_name = '_'.join(sorted(proteins))

print(f"Complex Name: {complex_name}")

# Directories
base_dir = get_data_dir() / "25.12_pooled-ppi-yeast/data-26.03"
input_dir = Path(f'complex-assembly/{complex_name}_input')
pdbs_dir = input_dir / 'pdbs'
output_dir = Path(f'complex-assembly/{complex_name}_output')

input_dir.mkdir(parents=True, exist_ok=True)
pdbs_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Generate subunits.json
js = collections.OrderedDict()
sequences = pd.read_parquet(base_dir / "proteins.parquet")

for protein, chain_name in zip(sorted(proteins), af3io.input.enumerate_chains()):
    seq = sequences.query('uniprot_id == @protein')['seq'].squeeze()
    js[protein] = {
        'name': protein,
        'chain_names': [chain_name],
        'start_res': 1,
        'sequence': seq,
    }

subunits_path = input_dir / 'subunits.json'
with open(subunits_path, 'w') as fh:
    json.dump(js, fh, indent=2)

print(f"Saved JSON to: {subunits_path}")

# Extract and save PDB pair structures
pp = pooled_ppi.PooledPredictionsDb(base_dir)
pairs_ = (
    pp.pairs.query('uniprot_id1 in @proteins & uniprot_id2 in @proteins')
    .sort_values('af3_pair')
    .reset_index(drop=True)
)
pairs_['pml_id'] = pairs_['uniprot_id1'].astype(str) + '_' + pairs_['uniprot_id2'].astype(str)

for _, r in pairs_.iterrows():
    db_id1 = f'{r["name"]}_{r.af3_id1}'
    db_id2 = f'{r["name"]}_{r.af3_id2}'
    path_pdb = pdbs_dir / f'AFM_{r.uniprot_id1}_{r.uniprot_id2}_unrelaxed_rank_1_model_1.pdb'
    print(f"Writing PDB: {path_pdb}")
    pp.save_ids([db_id1, db_id2], str(path_pdb))

EOF

# -----------------------------------------------------------------------------
# 3. Compute Complex Directory Name & Execute CombFold
# -----------------------------------------------------------------------------
COMPLEX_NAME=$(echo "$PROTEINS" | tr ' ' '\n' | sort | tr '\n' '_' | sed 's/_$//')

SUBUNITS_FILE="complex-assembly/${COMPLEX_NAME}_input/subunits.json"
PDBS_DIR="complex-assembly/${COMPLEX_NAME}_input/pdbs/"
OUTPUT_DIR="complex-assembly/${COMPLEX_NAME}_output/"

echo "==> Executing CombFold container..."

singularity exec /cluster/project/beltrao/shared/alphafold3/images/combfold_latest.sif \
  python /app/CombFold/scripts/run_on_pdbs.py \
  "$SUBUNITS_FILE" \
  "$PDBS_DIR" \
  "$OUTPUT_DIR"

echo "==> Done! Output saved in ${OUTPUT_DIR}"