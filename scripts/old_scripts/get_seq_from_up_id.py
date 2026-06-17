import pandas as pd
import requests
import time
from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"

# ── Load IDs ──────────────────────────────────────────────────────────────
df_ids = pd.read_csv(data_dir / "Pipeline/uniprot_ids_first_setup.csv")
# Adjust column name as needed
id_col = df_ids.columns[0]
uniprot_ids = df_ids[id_col].dropna().unique().tolist()
print(f"Fetching sequences for {len(uniprot_ids)} IDs")

# ── Batch fetch from UniProt ──────────────────────────────────────────────
BATCH = 500
sequences = {}

for i in range(0, len(uniprot_ids), BATCH):
    batch = uniprot_ids[i:i+BATCH]
    print(f"  Batch {i//BATCH+1}: {len(batch)} IDs...")
    r = requests.get(
        "https://rest.uniprot.org/uniprotkb/stream",
        params={
            "query": " OR ".join(f"accession:{a}" for a in batch),
            "format": "fasta",
            "size": len(batch),
        },
        timeout=120,
    )
    if r.status_code == 429:
        time.sleep(int(r.headers.get("Retry-After", 10)))
        continue
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}, skipping batch")
        continue

    # Parse FASTA
    cur_id, cur_seq = None, []
    for line in r.text.splitlines():
        if line.startswith(">"):
            if cur_id:
                sequences[cur_id] = "".join(cur_seq)
            cur_id = line.split("|")[1] if "|" in line else line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id:
        sequences[cur_id] = "".join(cur_seq)
    time.sleep(1)

# ── Build output dataframe ────────────────────────────────────────────────
df_seq = pd.DataFrame(
    [{"uniprot_id": uid, "sequence": sequences.get(uid, "")} for uid in uniprot_ids]
)
df_seq.to_csv(data_dir / "Pipeline/uniprot_mapped_sequences.csv", index=False)
print(f"Done: {len(sequences)}/{len(uniprot_ids)} sequences retrieved → uniprot_sequences.csv")