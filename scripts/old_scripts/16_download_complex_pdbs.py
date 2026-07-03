#!/usr/bin/env python
"""
16_download_complex_pdbs.py — Download PDB files cross-referenced from
Complex Portal for all yeast complexes.

Reads Saccharomyces_cerevisiae_ComplexTab.tsv and extracts wwpdb references
from TWO columns:

  1. "Cross references" — tagged as (identity) or (subset)
  2. "Experimental evidence" — untagged (just `wwpdb:XXXX`)

Downloads each unique PDB once from RCSB, then organises per-CPX copies into
three folders:

  identity/                — PDB represents the complete complex
  subset/                  — Complex is part of a larger PDB assembly
  experimental_evidence/   — PDB cited as experimental evidence (untagged)

Files are named:  {CPX_ID}_{PDB_ID}.pdb   (e.g. CPX-21_6I52.pdb)
Already-existing files are skipped (resume-friendly).

Usage:
    python 16_download_complex_pdbs.py \
        --tsv data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv \
        --out-dir data/Complex_pdb_files

    # Or as an sbatch job (recommended):
    sbatch scripts/15_download_complex_pdbs.sbatch
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DOWNLOAD_DELAY_SEC = 0.3   # be nice to RCSB
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

# Three tag categories — output folder names match exactly
TAGS = ("identity", "subset", "experimental_evidence")

# wwpdb tagged tokens: wwpdb:6I52(identity) | wwpdb:4M77(subset)
WWPDB_TAGGED_RE = re.compile(r"^wwpdb:([A-Za-z0-9]{4})\((identity|subset)\)$")

# wwpdb bare token (used in the "Experimental evidence" column): wwpdb:5VSU
WWPDB_BARE_RE = re.compile(r"\bwwpdb:([A-Za-z0-9]{4})\b")

# Any wwpdb: token — used only to spot malformed entries we ultimately drop
WWPDB_ANY_RE = re.compile(r"^wwpdb:[^|]+$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_pdb_xrefs(cross_ref_str: str, malformed_log: list[tuple[str, str]] | None = None,
                    cpx_id_for_log: str = "") -> dict[str, list[str]]:
    """Parse the 'Cross references' column for wwpdb entries.

    Returns {"identity": [pdb_id, ...], "subset": [pdb_id, ...]}.
    Malformed wwpdb tokens (e.g. PubMed IDs misfiled as wwpdb) are recorded
    in malformed_log if provided.
    """
    result: dict[str, list[str]] = {"identity": [], "subset": []}
    if not isinstance(cross_ref_str, str) or not cross_ref_str:
        return result

    for token in cross_ref_str.split("|"):
        token = token.strip()
        m = WWPDB_TAGGED_RE.match(token)
        if m:
            pdb_id = m.group(1).upper()
            tag = m.group(2)
            if pdb_id not in result[tag]:
                result[tag].append(pdb_id)
        elif WWPDB_ANY_RE.match(token):
            # Looks like a wwpdb entry but doesn't match expected format
            # (e.g. CPX-1316: wwpdb:35022249(identity) — a PubMed ID misfile)
            if malformed_log is not None:
                malformed_log.append((cpx_id_for_log, token))

    return result


def parse_experimental_evidence(exp_evidence_str: str) -> list[str]:
    """Parse the 'Experimental evidence' column for untagged wwpdb entries.

    Returns a list of uppercase PDB IDs found. Uses a word-boundary regex so
    `wwpdb:5VSU` is captured even when surrounded by other tokens separated
    by `|`, `,`, `;`, or whitespace.
    """
    if not isinstance(exp_evidence_str, str) or not exp_evidence_str:
        return []
    pdbs: list[str] = []
    for match in WWPDB_BARE_RE.finditer(exp_evidence_str):
        pdb_id = match.group(1).upper()
        if pdb_id not in pdbs:
            pdbs.append(pdb_id)
    return pdbs


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, dest_path: Path) -> bool:
    """Download a single PDB file from RCSB. Returns True on success."""
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url, headers={"User-Agent": "ComplexPDBDownloader/1.0"})
            with urlopen(req, timeout=30) as resp:
                content = resp.read()
            if not content:
                print(f"  [WARN] Empty response for {pdb_id}")
                return False
            dest_path.write_bytes(content)
            return True
        except HTTPError as e:
            if e.code == 404:
                print(f"  [SKIP] {pdb_id} not found on RCSB (404)")
                return False
            print(f"  [RETRY] {pdb_id} HTTP {e.code} (attempt {attempt}/{MAX_RETRIES})")
        except URLError as e:
            print(f"  [RETRY] {pdb_id} URL error: {e.reason} (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            print(f"  [RETRY] {pdb_id} error: {e} (attempt {attempt}/{MAX_RETRIES})")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SEC)

    print(f"  [FAIL] {pdb_id} after {MAX_RETRIES} attempts")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True, type=Path,
                    help="Path to Saccharomyces_cerevisiae_ComplexTab.tsv")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory (will contain identity/, subset/, experimental_evidence/, _raw/)")
    ap.add_argument("--delay", type=float, default=DOWNLOAD_DELAY_SEC,
                    help=f"Seconds between downloads (default: {DOWNLOAD_DELAY_SEC})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and report what would be downloaded; do not download.")
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"TSV not found: {args.tsv}")

    # --- Create output directories ---
    tag_dirs: dict[str, Path] = {tag: args.out_dir / tag for tag in TAGS}
    raw_dir = args.out_dir / "_raw"
    for d in (*tag_dirs.values(), raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Parse TSV ---
    print(f"Reading {args.tsv}")
    cpx_to_pdbs: dict[str, dict[str, list[str]]] = {}   # cpx_id -> {tag: [pdb_id, ...]}
    malformed_log: list[tuple[str, str]] = []

    with open(args.tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cpx_id = row.get("#Complex ac", "").strip()
            if not cpx_id:
                continue

            # 1. Tagged wwpdb entries in 'Cross references'
            xref_str = row.get("Cross references", "")
            xref_pdbs = parse_pdb_xrefs(xref_str, malformed_log, cpx_id)

            # 2. Untagged wwpdb entries in 'Experimental evidence'
            exp_str = row.get("Experimental evidence", "")
            exp_pdbs = parse_experimental_evidence(exp_str)

            # Deduplicate exp_pdbs against tagged xref_pdbs (don't double-tag the same PDB)
            tagged_pdbs = set(xref_pdbs["identity"]) | set(xref_pdbs["subset"])
            exp_pdbs_unique = [p for p in exp_pdbs if p not in tagged_pdbs]

            entry: dict[str, list[str]] = {
                "identity": list(xref_pdbs["identity"]),
                "subset": list(xref_pdbs["subset"]),
                "experimental_evidence": exp_pdbs_unique,
            }

            if any(entry[tag] for tag in TAGS):
                cpx_to_pdbs[cpx_id] = entry

    # --- Log malformed entries ---
    if malformed_log:
        print(f"\n[SKIP_MALFORMED] Dropped {len(malformed_log)} wwpdb token(s) that don't look like 4-char PDB IDs:")
        for cpx_id, tok in malformed_log:
            print(f"  {cpx_id}: {tok}")

    # --- Coverage summary ---
    counts: dict[str, set[tuple[str, str]]] = {tag: set() for tag in TAGS}
    unique_pdbs_per_tag: dict[str, set[str]] = {tag: set() for tag in TAGS}
    for cpx, pdbs in cpx_to_pdbs.items():
        for tag in TAGS:
            for pdb_id in pdbs[tag]:
                counts[tag].add((cpx, pdb_id))
                unique_pdbs_per_tag[tag].add(pdb_id)

    all_pdbs: set[str] = set().union(*unique_pdbs_per_tag.values())

    print(f"\nFound {len(cpx_to_pdbs)} complexes with PDB cross-references.")
    for tag in TAGS:
        print(f"  {tag:24s}  {len(counts[tag]):4d} (CPX, PDB) pairs, "
              f"{len(unique_pdbs_per_tag[tag]):4d} unique PDB IDs")
    print(f"  {'TOTAL UNIQUE PDB IDs':24s}  {len(all_pdbs):4d}")

    # --- Build pdb_to_cpxs (each unique PDB -> which CPXs/tags reference it) ---
    pdb_to_cpxs: dict[str, dict[str, list[str]]] = {}
    for cpx, pdbs in cpx_to_pdbs.items():
        for tag in TAGS:
            for pdb_id in pdbs[tag]:
                pdb_to_cpxs.setdefault(pdb_id, {t: [] for t in TAGS})
                if cpx not in pdb_to_cpxs[pdb_id][tag]:
                    pdb_to_cpxs[pdb_id][tag].append(cpx)

    # --- Download each unique PDB once ---
    downloaded = 0
    skipped_existing = 0
    failed = 0

    print(f"\nDownloading {len(pdb_to_cpxs)} unique PDB files into {raw_dir}...")

    for i, (pdb_id, tags_for_pdb) in enumerate(sorted(pdb_to_cpxs.items()), start=1):
        raw_path = raw_dir / f"{pdb_id}.pdb"

        if raw_path.exists() and raw_path.stat().st_size > 0:
            skipped_existing += 1
            continue

        if args.dry_run:
            print(f"  [{i}/{len(pdb_to_cpxs)}] Would download {pdb_id}")
            continue

        print(f"  [{i}/{len(pdb_to_cpxs)}] Downloading {pdb_id}...", end=" ", flush=True)
        if download_pdb(pdb_id, raw_path):
            downloaded += 1
            print("OK")
        else:
            failed += 1
            continue
        time.sleep(args.delay)

    if args.dry_run:
        print(f"\n[DRY-RUN] Would download {len(pdb_to_cpxs)} unique PDB files.")
        # Still write the mapping CSV so the user can inspect the planned coverage
        _write_mapping_csv(args.out_dir, cpx_to_pdbs)
        return

    # --- Organise per-CPX copies into identity/, subset/, experimental_evidence/ ---
    print(f"\nOrganising files into identity/, subset/, experimental_evidence/...")
    copied_counts: dict[str, int] = {tag: 0 for tag in TAGS}

    for pdb_id, tags_for_pdb in sorted(pdb_to_cpxs.items()):
        raw_path = raw_dir / f"{pdb_id}.pdb"
        if not raw_path.exists():
            continue

        for tag in TAGS:
            for cpx in tags_for_pdb[tag]:
                dest = tag_dirs[tag] / f"{cpx}_{pdb_id}.pdb"
                if not dest.exists():
                    dest.write_bytes(raw_path.read_bytes())
                    copied_counts[tag] += 1

    # --- Write mapping CSV ---
    mapping_path = _write_mapping_csv(args.out_dir, cpx_to_pdbs)

    # --- Final summary ---
    print(f"\nDone.")
    print(f"  Downloaded:        {downloaded}")
    print(f"  Skipped existing:  {skipped_existing}")
    print(f"  Failed:            {failed}")
    for tag in TAGS:
        n_files = len(list(tag_dirs[tag].glob("*.pdb")))
        print(f"  {tag}/ files: {n_files} (copied this run: {copied_counts[tag]})")
    print(f"  Mapping CSV:       {mapping_path}")
    print(f"\nOutput structure:")
    print(f"  {args.out_dir}/")
    print(f"    identity/                — PDB represents complete complex")
    print(f"    subset/                  — Complex is part of larger PDB assembly")
    print(f"    experimental_evidence/   — PDB cited as evidence (untagged in TSV)")
    print(f"    _raw/                    — Unique downloads (one per PDB ID)")
    print(f"    complex_pdb_mapping.csv  — Per-(CPX, PDB, tag) row")


def _write_mapping_csv(out_dir: Path, cpx_to_pdbs: dict[str, dict[str, list[str]]]) -> Path:
    mapping_path = out_dir / "complex_pdb_mapping.csv"
    with open(mapping_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["#Complex ac", "pdb_id", "tag"])
        for cpx in sorted(cpx_to_pdbs.keys()):
            for tag in TAGS:
                for pdb_id in cpx_to_pdbs[cpx][tag]:
                    writer.writerow([cpx, pdb_id, tag])
    return mapping_path


if __name__ == "__main__":
    main()
