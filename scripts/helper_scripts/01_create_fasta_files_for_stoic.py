import polars as pl
import requests
import time
from pathlib import Path
from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"

# ── Config ────────────────────────────────────────────────────────────────
COMPLEX_FILE = Path( data_dir / "Pipeline/first_setup_pipeline_complexes.csv" )
OUT_DIR = Path(data_dir / "Pipeline/first_setup_pipeline_complexes")
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load df and collect all unique UniProt IDs ─────────────────────────
df = pl.read_csv(COMPLEX_FILE)

all_ids = (
    df.select(pl.col("predicted_complex").str.split(" ").alias("ids"))
    .explode("ids")
    .filter(pl.col("ids") != "")
    .unique()["ids"]
    .sort()
    .to_list()
)
print(f"Total unique UniProt IDs: {len(all_ids)}")

# ── 2. Fetch sequences from UniProt ───────────────────────────────────────
BATCH = 500
sequences = {}

for i in range(0, len(all_ids), BATCH):
    batch = all_ids[i:i+BATCH]
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

print(f"Retrieved {len(sequences)}/{len(all_ids)} sequences")

# ── 3. Write one FASTA file per complex ───────────────────────────────────
rows = df.select(["#Complex ac", "predicted_complex"]).iter_rows()
missing_total = 0

for complex_ac, members_str in rows:
    member_ids = members_str.strip().split()
    fasta_path = OUT_DIR / f"{complex_ac}.fasta"
    missing = 0
    with open(fasta_path, "w") as f:
            for uid in member_ids:
                seq = sequences.get(uid, "")
                f.write(f">{uid}\n")
                if seq:
                    for j in range(0, len(seq), 80):
                        f.write(seq[j:j+80] + "\n")
                else:
                    missing += 1
                    total_missing = missing_total + 1

print(f"Wrote {df.height} FASTA files to {OUT_DIR}/")
if missing_total:
    print(f"Warning: {missing_total} protein(s) had no sequence retrieved")