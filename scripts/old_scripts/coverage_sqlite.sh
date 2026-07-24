#!/bin/bash
#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=coverage_sqlite
#SBATCH --output=logs/coverage_sqlite%j.out
#SBATCH --error=logs/coverage_sqlite%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
# Population-level pair/pool homodimer structure coverage — pure shell, no Python.
# Reads the DB's .lookup index (plain text) and, for each protein in the list,
# counts how many PAIR structure keys exist. Pair keys look like:
#     {id}_{id}_{sample}_{chain}      e.g. o13297_o13297_0_A
# Runs light enough for the login node, or submit via sbatch (see bottom).
#
# Usage:  bash coverage.sh <protein_list.csv> [out.csv]
set -euo pipefail

DATA="/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
LOOKUP="$DATA/predictions-db/predictions-db.lookup"

LIST="${1:?usage: coverage.sh <protein_list.csv> [out.csv]}"
OUT="${2:-coverage.csv}"

# 1. Extract just the key column (col 2) from the lookup, lowercase, sort-unique.
#    This is the set of all db_ids that actually exist.
echo "Extracting keys from $LOOKUP ..."
awk '{print $2}' "$LOOKUP" | tr 'A-Z' 'a-z' | sort -u > /tmp/db_keys.txt
echo "  DB has $(wc -l < /tmp/db_keys.txt) unique keys."

# 2. Pull only the PAIR self-homodimer keys: pattern  id_id_sample_chain
#    (input_name for a homodimer pair is {id}_{id}; pool input_name is a hash,
#     so grepping the doubled-id prefix isolates pair keys.)
grep -E '^[a-z0-9]+_[a-z0-9]+_[0-9]+_[A-Za-z]$' /tmp/db_keys.txt \
  | awk -F'_' '$1==$2 {print $1}' | sort | uniq -c > /tmp/pair_counts.txt
#   -> lines like:  "10 o13297"  meaning 10 pair keys (5 samples x 2 chains)

# 3. For each protein in the list, look up its pair-key count.
echo "protein,pair_keys,verdict" > "$OUT"
n_pair=0; n_none=0; total=0
# normalize list: lowercase, strip header, dedup
tr 'A-Z' 'a-z' < "$LIST" | grep -viE '^uniprot' | sed '/^\s*$/d' | sort -u \
  > /tmp/prot_list.txt

# build an assoc lookup of counts
declare -A CNT
while read -r c p; do CNT["$p"]=$c; done < /tmp/pair_counts.txt

while read -r p; do
  total=$((total+1))
  k=${CNT[$p]:-0}
  if [ "$k" -gt 0 ]; then
    echo "$p,$k,pair_available" >> "$OUT"; n_pair=$((n_pair+1))
  else
    echo "$p,0,no_pair" >> "$OUT"; n_none=$((n_none+1))
  fi
done < /tmp/prot_list.txt

echo
echo "Total proteins:      $total"
echo "  pair_available:    $n_pair"
echo "  no_pair:           $n_none"
echo "Wrote $OUT"
