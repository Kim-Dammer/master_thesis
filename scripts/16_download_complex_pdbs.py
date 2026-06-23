#!/usr/bin/env python
"""
15_download_complex_pdbs.py — Download PDB files cross-referenced from
Complex Portal for all yeast complexes.

Reads Saccharomyces_cerevisiae_ComplexTab.tsv, extracts wwpdb cross-references
tagged as (identity) or (subset), downloads the corresponding PDB files from
RCSB, and organizes them into two folders:
  identity/  — PDBs where the structure represents the complete complex
  subset/    — PDBs where the complex is part of a larger assembly

Files are named: {CPX_ID}_{PDB_ID}.pdb  (e.g. CPX-21_6I52.pdb)
Already-existing files are skipped (resume-friendly).

Usage:
    python 15_download_complex_pdbs.py \
        --tsv data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv \
        --out-dir data/Complex_pdb_files

    # Or as an sbatch job (recommended for large downloads):
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
DOWNLOAD_DELAY_SEC = 0.3  # be nice to RCSB
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

# ---------------------------------------------------------------------------
# Parse cross-references
# ---------------------------------------------------------------------------

def parse_pdb_xrefs(cross_ref_str: str) -> dict[str, list[str]]:
    """Parse the 'Cross references' column for wwpdb entries.

    Returns {"identity": [pdb_id, ...], "subset": [pdb_id, ...]}.
    Untagged wwpdb entries are ignored.
    """
    result: dict[str, list[str]] = {"identity": [], "subset": []}
    if not isinstance(cross_ref_str, str) or not cross_ref_str:
        return result

    for token in cross_ref_str.split("|"):
        token = token.strip()
        # Match patterns like: wwpdb:6I52(identity) or wwpdb:4M77(subset)
        m = re.match(r"^wwpdb:([A-Za-z0-9]{4})\((identity|subset)\)$", token)
        if m:
            pdb_id = m.group(1).upper()
            tag = m.group(2)
            if pdb_id not in result[tag]:
                result[tag].append(pdb_id)

    return result


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True, type=Path,
                    help="Path to Saccharomyces_cerevisiae_ComplexTab.tsv")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory (will contain identity/ and subset/ subdirs)")
    ap.add_argument("--delay", type=float, default=DOWNLOAD_DELAY_SEC,
                    help=f"Seconds between downloads (default: {DOWNLOAD_DELAY_SEC})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and print what would be downloaded, but don't download")
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"TSV not found: {args.tsv}")

    # Create output directories
    identity_dir = args.out_dir / "identity"
    subset_dir = args.out_dir / "subset"
    identity_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse TSV ---
    print(f"Reading {args.tsv}")
    cpx_to_pdbs: dict[str, dict[str, list[str]]] = {}  # cpx_id -> {"identity": [...], "subset": [...]}

    with open(args.tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cpx_id = row.get("#Complex ac", "").strip()
            if not cpx_id:
                continue
            xref_str = row.get("Cross references", "")
            pdbs = parse_pdb_xrefs(xref_str)
            if pdbs["identity"] or pdbs["subset"]:
                cpx_to_pdbs[cpx_id] = pdbs

    # --- Summary ---
    all_identity = set()
    all_subset = set()
    for pdbs in cpx_to_pdbs.values():
        all_identity.update(pdbs["identity"])
        all_subset.update(pdbs["subset"])

    print(f"Found {len(cpx_to_pdbs)} complexes with PDB cross-references")
    print(f"  Identity PDBs: {len(all_identity)} unique")
    print(f"  Subset PDBs:   {len(all_subset)} unique")
    overlap = all_identity & all_subset
    if overlap:
        print(f"  (Overlap: {len(overlap)} PDBs appear as both identity and subset across complexes)")

    # --- Download ---
    total_downloads = len(all_identity) + len(all_subset)
    downloaded = 0
    skipped_existing = 0
    failed = 0

    # Collect all (pdb_id, tag, cpx_ids) tuples
    # A single PDB can be referenced by multiple CPX IDs — download once, symlink or copy
    # We'll download once and then create copies named per-CPX
    pdb_to_cpxs: dict[str, dict[str, list[str]]] = {}  # pdb_id -> {"identity": [cpx, ...], "subset": [cpx, ...]}
    for cpx_id, pdbs in cpx_to_pdbs.items():
        for pdb_id in pdbs["identity"]:
            pdb_to_cpxs.setdefault(pdb_id, {"identity": [], "subset": []})
            if cpx_id not in pdb_to_cpxs[pdb_id]["identity"]:
                pdb_to_cpxs[pdb_id]["identity"].append(cpx_id)
        for pdb_id in pdbs["subset"]:
            pdb_to_cpxs.setdefault(pdb_id, {"identity": [], "subset": []})
            if cpx_id not in pdb_to_cpxs[pdb_id]["subset"]:
                pdb_to_cpxs[pdb_id]["subset"].append(cpx_id)

    # Download each unique PDB once, then copy to per-CPX names
    # First pass: download raw PDB files to a temp location
    raw_dir = args.out_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(pdb_to_cpxs)} unique PDB files...")

    for i, (pdb_id, tags) in enumerate(sorted(pdb_to_cpxs.items()), start=1):
        raw_path = raw_dir / f"{pdb_id}.pdb"

        # Skip if already downloaded
        if raw_path.exists() and raw_path.stat().st_size > 0:
            skipped_existing += 1
        else:
            if args.dry_run:
                print(f"  [{i}/{len(pdb_to_cpxs)}] Would download {pdb_id}")
                continue
            print(f"  [{i}/{len(pdb_to_cpxs)}] Downloading {pdb_id}...", end=" ", flush=True)
            success = download_pdb(pdb_id, raw_path)
            if success:
                downloaded += 1
                print("OK")
            else:
                failed += 1
                continue
            time.sleep(args.delay)

    if args.dry_run:
        print(f"\n[DRY-RUN] Would download {len(pdb_to_cpxs)} PDB files")
        print(f"  Identity: {len(all_identity)}, Subset: {len(all_subset)}")
        return

    # Second pass: create per-CPX copies in identity/ and subset/ folders
    print(f"\nOrganizing files into identity/ and subset/...")

    for pdb_id, tags in sorted(pdb_to_cpxs.items()):
        raw_path = raw_dir / f"{pdb_id}.pdb"
        if not raw_path.exists():
            continue

        for cpx_id in tags["identity"]:
            dest = identity_dir / f"{cpx_id}_{pdb_id}.pdb"
            if not dest.exists():
                dest.write_bytes(raw_path.read_bytes())

        for cpx_id in tags["subset"]:
            dest = subset_dir / f"{cpx_id}_{pdb_id}.pdb"
            if not dest.exists():
                dest.write_bytes(raw_path.read_bytes())

    # --- Write mapping CSV ---
    mapping_path = args.out_dir / "complex_pdb_mapping.csv"
    with open(mapping_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["#Complex ac", "pdb_id", "tag"])
        for cpx_id, pdbs in sorted(cpx_to_pdbs.items()):
            for pdb_id in pdbs["identity"]:
                writer.writerow([cpx_id, pdb_id, "identity"])
            for pdb_id in pdbs["subset"]:
                writer.writerow([cpx_id, pdb_id, "subset"])

    # --- Final summary ---
    n_identity_files = len(list(identity_dir.glob("*.pdb")))
    n_subset_files = len(list(subset_dir.glob("*.pdb")))

    print(f"\nDone!")
    print(f"  Downloaded: {downloaded}, Skipped (existing): {skipped_existing}, Failed: {failed}")
    print(f"  identity/ files: {n_identity_files}")
    print(f"  subset/   files: {n_subset_files}")
    print(f"  Mapping CSV: {mapping_path}")
    print(f"\nOutput structure:")
    print(f"  {args.out_dir}/")
    print(f"    identity/   — PDBs where structure = complete complex")
    print(f"    subset/     — PDBs where complex is part of larger assembly")
    print(f"    _raw/       — Raw downloaded files (one per unique PDB ID)")
    print(f"    complex_pdb_mapping.csv")


if __name__ == "__main__":
    main()
