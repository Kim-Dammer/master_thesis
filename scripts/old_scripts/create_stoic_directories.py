#!/usr/bin/env python3
"""
Fetch FASTA sequences for each subunit of predicted complexes and write
one FASTA file per complex.

Expects a DataFrame (e.g. loaded from CSV/TSV) with columns:
- "Complex ac"       : complex accession, used as output filename
- "predicted_complex": space-separated UniProt IDs of the subunits

For each row, writes a file <output_dir>/<complex_ac>.fasta containing:
>UniProtID
SEQUENCE
... for each subunit.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"


def fetch_sequence(uniprot_id: str, session: requests.Session, retries: int = 3) -> str:
    """Fetch the raw sequence (no header) for a UniProt ID."""
    url = UNIPROT_FASTA_URL.format(uniprot_id)
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                lines = resp.text.strip().splitlines()
                seq_lines = [l for l in lines if not l.startswith(">")]
                return "".join(seq_lines)
            elif resp.status_code == 404:
                print(f"  WARNING: {uniprot_id} not found (404)", file=sys.stderr)
                return ""
            else:
                print(f"  WARNING: {uniprot_id} returned status {resp.status_code}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  WARNING: error fetching {uniprot_id} (attempt {attempt+1}): {e}", file=sys.stderr)
        time.sleep(1)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Fetch FASTA files per complex from UniProt IDs.")
    parser.add_argument("input", help="Path to input table (CSV or TSV)")
    parser.add_argument("output_dir", help="Directory to write FASTA files to")
    parser.add_argument("--sep", default=None, help="Column separator (default: auto-detect from extension)")
    parser.add_argument("--complex-col", default="Complex ac", help="Column name for complex accession")
    parser.add_argument("--pred-col", default="predicted_complex", help="Column name for predicted complex (space-separated UniProt IDs)")
    parser.add_argument("--cache", default=None, help="Optional path to a TSV file used as a local sequence cache (uniprot_id\\tsequence)")
    args = parser.parse_args()

    # Load dataframe
    if args.sep:
        sep = args.sep
    elif args.input.endswith(".tsv"):
        sep = "\t"
    else:
        sep = ","
    df = pd.read_csv(args.input, sep=sep)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional local cache to avoid refetching
    cache = {}
    if args.cache and Path(args.cache).exists():
        cache_df = pd.read_csv(args.cache, sep="\t", header=None, names=["uniprot_id", "sequence"])
        cache = dict(zip(cache_df["uniprot_id"], cache_df["sequence"]))
        print(f"Loaded {len(cache)} cached sequences from {args.cache}")

    session = requests.Session()
    fetched_this_run = {}

    for idx, row in df.iterrows():
        complex_ac = str(row[args.complex_col]).strip()
        uniprot_ids = str(row[args.pred_col]).split()

        if not complex_ac or complex_ac.lower() == "nan":
            print(f"Row {idx}: missing complex accession, skipping", file=sys.stderr)
            continue

        out_path = out_dir / f"{complex_ac}.fasta"
        records = []

        for uid in uniprot_ids:
            uid = uid.strip()
            if not uid:
                continue

            seq = cache.get(uid) or fetched_this_run.get(uid)
            if seq is None:
                print(f"Fetching {uid} for complex {complex_ac}...")
                seq = fetch_sequence(uid, session)
                fetched_this_run[uid] = seq

            if not seq:
                print(f"  No sequence found for {uid}, skipping in {out_path.name}", file=sys.stderr)
                continue

            records.append((uid, seq))

        if records:
            with open(out_path, "w") as f:
                for uid, seq in records:
                    f.write(f">{uid}\n{seq}\n")
            print(f"Wrote {out_path} ({len(records)} sequences)")
        else:
            print(f"Skipped {complex_ac}: no sequences retrieved", file=sys.stderr)

    # Save cache for future runs
    if args.cache:
        all_seqs = {**cache, **fetched_this_run}
        cache_df = pd.DataFrame(
            [(k, v) for k, v in all_seqs.items() if v],
            columns=["uniprot_id", "sequence"],
        )
        cache_df.to_csv(args.cache, sep="\t", header=False, index=False)
        print(f"Updated cache with {len(cache_df)} sequences -> {args.cache}")


if __name__ == "__main__":
    main()