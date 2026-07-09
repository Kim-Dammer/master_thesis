"""
Fetch protein sequences from UniProt for a list of identifiers.

Handles three root causes that broke the original script:
  1. UniProt limits queries to 100 OR conditions (batches of 500 -> HTTP 400).
  2. One invalid-format ID poisons the entire batch query (HTTP 400).
     Non-accession identifiers (ORF names like YBR118W, gene names like CDC19)
     must be validated and mapped separately.
  3. The /stream endpoint is unreliable; /search is used instead.

Strategy:
  - Validate every ID against the UniProt accession format regex.
  - Valid-format accessions are fetched in batches of 100 via accession: queries.
  - Invalid-format IDs are mapped to UniProt accessions via gene name search
    (organism_id:559292 AND gene:NAME), then their sequences are fetched.
  - Obsolete/deleted accessions (valid format, no sequence) are recorded as
    empty strings, not errors.
"""

import re
import time

import pandas as pd
import requests

from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"
YEAST_ORGANISM_ID = "559292"  # S. cerevisiae S288c

# ── UniProt accession format ──────────────────────────────────────────────
# Matches both 6-char (P00549) and 10-char (A0A1B2C3D4) accession patterns.
ACCESSION_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]([A-Z0-9]{3}[0-9])?$"  # P12345, Q1A2B3C4D5
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"      # A12345, A1B2C3D4
)

# ── Load IDs ──────────────────────────────────────────────────────────────
df_ids = pd.read_csv(
    data_dir / "iPTM_and_pLDDT/all_yeast_proteins.csv", header=None
)
uniprot_ids = df_ids[0].dropna().astype(str).str.strip().unique().tolist()
print(f"Loaded {len(uniprot_ids)} IDs")

# ── Partition into valid accessions vs. non-accession IDs ─────────────────
valid_accs = [uid for uid in uniprot_ids if ACCESSION_RE.match(uid)]
invalid_ids = [uid for uid in uniprot_ids if not ACCESSION_RE.match(uid)]
print(f"  {len(valid_accs)} valid-format accessions")
print(f"  {len(invalid_ids)} non-accession IDs (will map via gene name)")

if invalid_ids:
    print(f"  Examples of non-accession IDs: {invalid_ids[:10]}")

# ── Helpers ───────────────────────────────────────────────────────────────
BATCH = 100
MAX_RETRIES = 3
sequences = {}


def fetch_batch_fasta(acc_list):
    """Fetch FASTA for a list of valid-format accessions (max 100)."""
    query = " OR ".join(f"accession:{a}" for a in acc_list)
    for attempt in range(MAX_RETRIES):
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": query, "format": "fasta", "size": len(acc_list)},
            timeout=120,
        )
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            print(f"429, waiting {wait}s...", end=" ", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.text
        print(f"HTTP {r.status_code}", end=" ", flush=True)
        time.sleep(2)
    print(f"FAILED: {r.text[:150]}", end=" ")
    return ""


def parse_fasta(fasta_text):
    """Parse FASTA text into {accession: sequence}."""
    result = {}
    cur_id, cur_seq = None, []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            if cur_id:
                result[cur_id] = "".join(cur_seq)
            cur_id = line.split("|")[1] if "|" in line else line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id:
        result[cur_id] = "".join(cur_seq)
    return result


# ── Fetch valid accessions in batches of 100 ──────────────────────────────
print(f"\nFetching {len(valid_accs)} valid accessions in batches of {BATCH}...")
for i in range(0, len(valid_accs), BATCH):
    batch = valid_accs[i : i + BATCH]
    print(f" Batch {i // BATCH + 1}: {len(batch)} IDs...", end=" ", flush=True)
    fasta = fetch_batch_fasta(batch)
    if fasta:
        parsed = parse_fasta(fasta)
        sequences.update(parsed)
        got = sum(1 for a in batch if a in parsed)
        print(f"{got}/{len(batch)} ok")
    else:
        print()
    time.sleep(1)

# ── Map non-accession IDs via gene name search ────────────────────────────
if invalid_ids:
    print(f"\nMapping {len(invalid_ids)} non-accession IDs via gene name...")
    mapped = {}
    for j, gid in enumerate(invalid_ids):
        print(f"  [{j+1}/{len(invalid_ids)}] {gid}...", end=" ", flush=True)
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": f"organism_id:{YEAST_ORGANISM_ID} AND gene:{gid}",
                "format": "tsv",
                "fields": "accession",
                "size": 5,
            },
            timeout=60,
        )
        if r.status_code == 200:
            lines = [
                l.strip()
                for l in r.text.splitlines()
                if l.strip() and not l.startswith("Entry")
            ]
            if lines:
                acc = lines[0]
                mapped[gid] = acc
                print(f"-> {acc}")
            else:
                print("not found")
        else:
            print(f"HTTP {r.status_code}")
        time.sleep(0.5)

    # Fetch sequences for mapped accessions
    mapped_accs = list(set(mapped.values()))
    print(f"\nFetching sequences for {len(mapped_accs)} mapped accessions...")
    for i in range(0, len(mapped_accs), BATCH):
        batch = mapped_accs[i : i + BATCH]
        print(f" Batch {i // BATCH + 1}: {len(batch)} IDs...", end=" ", flush=True)
        fasta = fetch_batch_fasta(batch)
        if fasta:
            parsed = parse_fasta(fasta)
            sequences.update(parsed)
            got = sum(1 for a in batch if a in parsed)
            print(f"{got}/{len(batch)} ok")
        else:
            print()
        time.sleep(1)

    # Record mappings so the output includes both original ID and mapped accession
    df_map = pd.DataFrame(
        [{"original_id": gid, "mapped_accession": mapped.get(gid, "")} for gid in invalid_ids]
    )
    map_path = data_dir / "iPTM_and_pLDDT/unmapped_id_to_seq_proteins.csv"
    df_map.to_csv(map_path, index=False)
    print(f"  Mapping saved -> {map_path.name}")

# ── Build output dataframe ────────────────────────────────────────────────
df_seq = pd.DataFrame(
    [{"uniprot_id": uid, "sequence": sequences.get(uid, "")} for uid in uniprot_ids]
)
out_path = (
    data_dir / "iPTM_and_pLDDT/all_yeast_proteins_uniprot_mapped_sequences.csv"
)
df_seq.to_csv(out_path, index=False)
retrieved = (df_seq["sequence"] != "").sum()
missing = len(uniprot_ids) - retrieved
print(f"\nDone: {retrieved}/{len(uniprot_ids)} sequences retrieved -> {out_path.name}")
if missing:
    print(f"  {missing} IDs have no sequence (obsolete, deleted, or unmappable)")
