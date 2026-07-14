#!/usr/bin/env python
"""
Fetch sequences for mixed identifier types and write a FASTA file.

Handles:
  P04037-PRO_0000006114   -> UniProt (strip -PRO_, fetch canonical)
  URS000006F31F_559292    -> RNAcentral
  EBI-16420196            -> Complex Portal export (needs --complex-tsv)

Usage:
  uv run 25_create_fasta_file.py \
      --csv /cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_CP_proteins.csv \
      --id-col identifier \
      --complex-tsv /cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv\
      --out /cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_CP_proteins_sequences.fasta
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][0-9A-Z]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)
PRO_RE = re.compile(
    r"^([OPQ][0-9][0-9A-Z]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})-PRO_\d+$"
)
URS_RE = re.compile(r"^(URS[0-9A-F]+)_\d+$")
EBI_RE = re.compile(r"^EBI-\d+$")


def _get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fetch_sequences/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sequence fetchers
# ---------------------------------------------------------------------------

def fetch_uniprot(acc: str) -> str | None:
    """Fetch canonical sequence from UniProt REST API."""
    data = _get_json(f"https://rest.uniprot.org/uniprotkb/{acc}.json")
    if data and "sequence" in data:
        return data["sequence"].get("value")
    return None


def fetch_rnacentral(urs_id: str) -> str | None:
    """Fetch sequence from RNAcentral API."""
    data = _get_json(f"https://rnacentral.org/api/v1/rna/{urs_id}")
    if data:
        return data.get("sequence")
    return None


def build_ebi_cache(complex_tsv: Path) -> dict[str, str]:
    """Parse Complex Portal TSV, find EBI- IDs and their parent complexes,
    then query Complex Portal export to get sequences.
    Returns {ebi_id: sequence}."""
    ebi_to_complex: dict[str, str] = {}
    with open(complex_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        mol_col = "Identifiers (and stoichiometry) of molecules in complex"
        for row in reader:
            cpx = row.get("#Complex ac", "").strip()
            raw = row.get(mol_col, "")
            if not isinstance(raw, str):
                continue
            for token in raw.split("|"):
                base = token.split("(")[0].strip()
                if EBI_RE.match(base):
                    ebi_to_complex[base] = cpx

    if not ebi_to_complex:
        return {}

    print(f"  Resolving {len(ebi_to_complex)} EBI- interactors from Complex Portal...",
          file=sys.stderr)
    cache: dict[str, str] = {}
    for ebi_id, cpx in ebi_to_complex.items():
        data = _get_json(f"https://www.ebi.ac.uk/intact/complex-ws/export/{cpx}")
        if data:
            for item in data.get("data", []):
                if item.get("object") != "interactor":
                    continue
                ident = item.get("identifier", {})
                if ident.get("db") == "intact" and ident.get("id") == ebi_id:
                    seq = item.get("sequence")
                    if seq:
                        cache[ebi_id] = seq
        time.sleep(0.2)
    return cache


def classify_and_fetch(raw_id: str, ebi_cache: dict[str, str]) -> tuple[str, str | None]:
    """Return (raw_id, sequence) for one identifier."""
    if PRO_RE.match(raw_id):
        base = PRO_RE.match(raw_id).group(1)
        return (raw_id, fetch_uniprot(base))
    elif UNIPROT_RE.match(raw_id):
        return (raw_id, fetch_uniprot(raw_id))
    elif URS_RE.match(raw_id):
        return (raw_id, fetch_rnacentral(raw_id))
    elif EBI_RE.match(raw_id):
        return (raw_id, ebi_cache.get(raw_id))
    return (raw_id, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path, help="CSV with identifiers")
    ap.add_argument("--id-col", default="identifier", help="Column name with IDs")
    ap.add_argument("--complex-tsv", type=Path, default=None,
                    help="Complex Portal TSV (required to resolve EBI- IDs)")
    ap.add_argument("--out", required=True, type=Path, help="Output FASTA file")
    ap.add_argument("--threads", type=int, default=16,
                    help="Number of parallel threads for API calls (default: 16)")
    args = ap.parse_args()

    # Read identifiers
    ids: list[str] = []
    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if args.id_col not in reader.fieldnames:
            sys.exit(f"Column '{args.id_col}' not found. Available: {reader.fieldnames}")
        for row in reader:
            val = row[args.id_col].strip()
            if val:
                ids.append(val)

    print(f"Loaded {len(ids)} identifiers from {args.csv}", file=sys.stderr)

    # Pre-fetch EBI- cache if needed
    ebi_cache: dict[str, str] = {}
    has_ebi = any(EBI_RE.match(i) for i in ids)
    if has_ebi:
        if not args.complex_tsv:
            print("WARNING: EBI- IDs found but --complex-tsv not provided. "
                  "These will be skipped.", file=sys.stderr)
        else:
            ebi_cache = build_ebi_cache(args.complex_tsv)

    # Fetch in parallel with progress
    results: dict[str, str | None] = {}
    n_ok = n_fail = 0
    total = len(ids)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(classify_and_fetch, raw_id, ebi_cache): raw_id
            for raw_id in ids
        }
        for i, future in enumerate(as_completed(futures), 1):
            raw_id, seq = future.result()
            results[raw_id] = seq
            if seq:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  FAIL (no sequence): {raw_id}", file=sys.stderr)
            if i % 200 == 0 or i == total:
                print(f"  [{i}/{total}] fetched ({n_ok} ok, {n_fail} fail)",
                      file=sys.stderr)

    # Write in original CSV order
    with open(args.out, "w") as out_fh:
        for raw_id in ids:
            seq = results.get(raw_id)
            if seq:
                out_fh.write(f">{raw_id}\n{seq}\n")

    print(f"\nDone: {n_ok} sequences written, {n_fail} failed/skipped", file=sys.stderr)
    print(f"Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
