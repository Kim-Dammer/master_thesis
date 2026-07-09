#!/usr/bin/env python
"""
24_download_best_coverage_pdbs.py — For each complex, pick the exact-match
PDB with the highest Overall_coverage (tie-break: best resolution) and
download its chosen biological assembly into best_coverage_complex/.

Reads pdb_stats.csv from script 23. Uses polars. Standalone — no SIFTS/UniProt
queries, only RCSB file downloads.

Usage:
    uv run 21_2_download_best_coverage_pdbs.py \
        --pdb-stats /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/pdb_stats.csv \
        --out-dir  /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/best_coverage_complex

    # audit-only (show choices, no downloads):
    python 24_download_best_coverage_pdbs.py --pdb-stats ... --out-dir ... --dry-run
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb{asm_id}"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}-assembly{asm_id}.cif"
API_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5
RCSB_DELAY = 0.3
# Plain string (not compiled) — polars str.extract requires a string pattern.
ASM_RE = r"biological_assembly_(\d+)"


def log(msg: str) -> None:
    print(msg, flush=True)


def download_one(url: str, out_path: Path, pdb_id: str, asm_id: str) -> str:
    """Try one URL with retries. Returns 'ok' | '404' | 'fail'."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BestCovDL/1.0"})
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                content = resp.read()
            if content and (b"ATOM" in content or b"_atom_site." in content):
                out_path.write_bytes(content)
                return "ok"
            log(f"    [WARN] {pdb_id} asm{asm_id}: no atom records")
            return "fail"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "404"
            log(f"    [RETRY] {pdb_id} asm{asm_id} HTTP {e.code} (attempt {attempt})")
        except Exception as e:
            log(f"    [RETRY] {pdb_id} asm{asm_id} {e} (attempt {attempt})")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return "fail"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-stats", required=True, type=Path,
                    help="pdb_stats.csv from script 23")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory (best_coverage_complex/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show choices, no downloads.")
    args = ap.parse_args()

    if not args.pdb_stats.exists():
        sys.exit(f"pdb_stats.csv not found: {args.pdb_stats}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("24_download_best_coverage_pdbs.py")
    log("=" * 70)

    # --- Selection: best PDB per complex ---
    df = pl.read_csv(args.pdb_stats)
    log(f"  Loaded {df.height} rows from {args.pdb_stats}")

    # Script 23 writes empty strings for NaN coverage/resolution, so read_csv
    # may infer these as String. Cast to Float64 (empty -> null) so numeric
    # sort works instead of lexicographic.
    df = df.with_columns([
        pl.col("Overall_coverage").cast(pl.Float64, strict=False),
        pl.col("resolution").cast(pl.Float64, strict=False),
    ])

    # Filter out no_assembly_metadata rows; parse assembly number
    df = df.filter(pl.col("assembly") != "no_assembly_metadata")
    df = df.with_columns(
        pl.col("assembly").str.extract(ASM_RE, 1).alias("asm_id")
    )
    # Drop any rows where asm_id couldn't be parsed (shouldn't happen after filter)
    df = df.filter(pl.col("asm_id").is_not_null())

    # Sort: highest coverage first, then best (lowest) resolution, then CPX for determinism
    df = df.sort(
        ["Overall_coverage", "resolution", "Complex_ac"],
        descending=[True, False, False],
        nulls_last=[False, True, False],
    )
    best = df.group_by("Complex_ac", maintain_order=True).first()
    log(f"  {best.height} complexes with a chosen best-coverage PDB")

    if args.dry_run:
        log("\n[DRY-RUN] Choices (no downloads):")
        print(best.select(["Complex_ac", "pdb_id", "asm_id", "assembly",
                           "Overall_coverage", "resolution"]).head(20).to_pandas().to_string(index=False))
        return

    # --- Download ---
    log(f"\nDownloading {best.height} assembly files to {args.out_dir}...")
    log_rows = []
    downloaded = skipped = failed = 0

    for i, row in enumerate(best.iter_rows(named=True), 1):
        cpx = row["Complex_ac"]
        pdb_id = str(row["pdb_id"]).upper()
        asm_id = row["asm_id"]

        pdb_path = args.out_dir / f"{cpx}_{pdb_id}_asm{asm_id}.pdb"
        cif_path = args.out_dir / f"{cpx}_{pdb_id}_asm{asm_id}.cif"

        # Resume: skip if either format already exists
        if pdb_path.exists() and pdb_path.stat().st_size > 0:
            skipped += 1
            log_rows.append({**_log_row(row, pdb_path.name), "status": "skip"})
            continue
        if cif_path.exists() and cif_path.stat().st_size > 0:
            skipped += 1
            log_rows.append({**_log_row(row, cif_path.name), "status": "skip"})
            continue

        # Try PDB format
        status = download_one(RCSB_PDB_URL.format(pdb_id=pdb_id, asm_id=asm_id),
                              pdb_path, pdb_id, asm_id)
        if status == "ok":
            downloaded += 1
            log_rows.append({**_log_row(row, pdb_path.name), "status": "ok"})
            log(f"  [{i}/{best.height}] {cpx} {pdb_id} asm{asm_id}: downloaded (.pdb)")
            time.sleep(RCSB_DELAY)
            continue

        # Fall back to mmCIF
        log(f"    [CIF-fallback] {pdb_id} asm{asm_id}")
        status = download_one(RCSB_CIF_URL.format(pdb_id=pdb_id, asm_id=asm_id),
                              cif_path, pdb_id, asm_id)
        if status == "ok":
            downloaded += 1
            log_rows.append({**_log_row(row, cif_path.name), "status": "ok"})
            log(f"  [{i}/{best.height}] {cpx} {pdb_id} asm{asm_id}: downloaded (.cif)")
        else:
            failed += 1
            log_rows.append({**_log_row(row, ""), "status": "fail"})
            log(f"  [{i}/{best.height}] {cpx} {pdb_id} asm{asm_id}: FAILED")
        time.sleep(RCSB_DELAY)

    # Write log
    log_df = pl.DataFrame(log_rows)
    log_path = args.out_dir / "best_coverage_download_log.csv"
    log_df.write_csv(log_path)

    log("\n" + "=" * 70)
    log("DONE")
    log("=" * 70)
    log(f"  Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    log(f"  Log: {log_path}")
    log(f"  Files in: {args.out_dir}")


def _log_row(row: dict, filename: str) -> dict:
    return {
        "Complex_ac": row["Complex_ac"],
        "pdb_id": row["pdb_id"],
        "assembly": row["assembly"],
        "Overall_coverage": row["Overall_coverage"],
        "resolution": row["resolution"],
        "filename": filename,
    }


if __name__ == "__main__":
    main()
