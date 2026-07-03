#!/usr/bin/env python
"""
run_pipeline.py — End-to-end reproducible pipeline for preparing reference PDB
structures for Saccharomyces cerevisiae Complex Portal complexes.

Runs 4 steps in sequence:
  STEP 1: Download PDBs Complex Portal annotated (identity/subset/experimental_evidence).
  STEP 2: For every unique protein in the TSV, query SIFTS for all PDBs it appears in.
  STEP 3: For each complex, find PDBs whose protein set EXACTLY matches the complex's.
  STEP 4: Download every exact-match PDB (skip those already in _raw/ from step 1).

CombFold comparison is NOT included — run 19_compare_combfold_to_pdb.py separately.

Resume-friendly: sifts_cache.json and _raw/ file-existence checks skip already-done work.
To force a full re-run, delete the --out-dir first.

Usage:
    uv run 21_Complex_pdb_files_run_pipeline.py 
    --tsv    /cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv\
    --out-dir /cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files

    # audit-only (no downloads, no API calls):
    python run_pipeline.py --tsv ... --out-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
SIFTS_BEST_STRUCTURES_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{u}"
SIFTS_PDB_UNIPROT_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"

RCSB_DELAY = 0.3
SIFTS_DELAY = 0.1
API_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5

UNIPROT_RE = re.compile(r"^[OPQ][0-9][0-9A-Z]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
BRACKET_RE = re.compile(r"^\[([A-Z0-9_,\s]+)\]$")
PRO_ISOFORM_RE = re.compile(r"^([OPQ][0-9][0-9A-Z]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})-PRO_\d+$")
WWPDB_TAGGED_RE = re.compile(r"^wwpdb:([A-Za-z0-9]{4})\((identity|subset)\)$")
WWPDB_BARE_RE = re.compile(r"\bwwpdb:([A-Za-z0-9]{4})\b")
WWPDB_ANY_RE = re.compile(r"^wwpdb:[^|]+$")

TAGS = ("identity", "subset", "experimental_evidence")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def _get_json(url: str, user_agent: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        log(f"    HTTP {e.code}: {url}")
        return None
    except Exception as e:
        log(f"    ERR {e}: {url}")
        return None


def extract_proteins(member: str) -> set[str]:
    """Extract UniProt accessions from one molecule-list token.
    Handles plain UniProts, [bracketed paralog groups], and PRO_ isoforms."""
    member = member.strip()
    if UNIPROT_RE.match(member):
        return {member}
    m = BRACKET_RE.match(member)
    if m:
        return {s.strip() for s in m.group(1).split(",") if UNIPROT_RE.match(s.strip())}
    m = PRO_ISOFORM_RE.match(member)
    if m:
        return {m.group(1)}
    return set()


# ---------------------------------------------------------------------------
# TSV parsing (shared by steps 1, 2, 3)
# ---------------------------------------------------------------------------

def parse_tsv(tsv_path: Path) -> tuple[
    dict[str, dict],        # cpx -> {name, proteins:set, counts:dict, xrefs:dict}
    dict[str, list[str]],   # uniprot -> [cpx ids]
]:
    df = pd.read_csv(tsv_path, sep="\t")
    mol_col = "Identifiers (and stoichiometry) of molecules in complex"
    name_col = "Recommended name"
    xref_col = "Cross references"
    expev_col = "Experimental evidence"

    complexes: dict[str, dict] = {}
    uniprot_to_complexes: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        cpx = str(row["#Complex ac"]).strip()
        if not cpx:
            continue
        name = str(row.get(name_col, "")).strip()

        # Proteins + stoichiometry
        proteins: set[str] = set()
        counts: dict[str, int] = {}
        raw = row.get(mol_col, "")
        if isinstance(raw, str):
            for token in raw.split("|"):
                base = token.split("(")[0].strip()
                count_m = re.search(r"\((\d+)\)", token)
                count = int(count_m.group(1)) if count_m else 1
                for u in extract_proteins(base):
                    proteins.add(u)
                    counts[u] = counts.get(u, 0) + count

        # Complex Portal PDB xrefs (identity/subset from Cross references)
        xrefs = {"identity": [], "subset": [], "experimental_evidence": []}
        xref_raw = row.get(xref_col, "")
        if isinstance(xref_raw, str):
            for token in xref_raw.split("|"):
                token = token.strip()
                m = WWPDB_TAGGED_RE.match(token)
                if m:
                    pdb_id = m.group(1).upper()
                    tag = m.group(2)
                    if pdb_id not in xrefs[tag]:
                        xrefs[tag].append(pdb_id)
                elif WWPDB_ANY_RE.match(token):
                    log(f"  [SKIP_MALFORMED] {cpx}: {token}")

        # Untagged wwpdb in Experimental evidence
        expev_raw = row.get(expev_col, "")
        if isinstance(expev_raw, str):
            tagged = set(xrefs["identity"]) | set(xrefs["subset"])
            for match in WWPDB_BARE_RE.finditer(expev_raw):
                pdb_id = match.group(1).upper()
                if pdb_id not in tagged and pdb_id not in xrefs["experimental_evidence"]:
                    xrefs["experimental_evidence"].append(pdb_id)

        complexes[cpx] = {"name": name, "proteins": proteins, "counts": counts, "xrefs": xrefs}
        for u in proteins:
            uniprot_to_complexes.setdefault(u, [])
            if cpx not in uniprot_to_complexes[u]:
                uniprot_to_complexes[u].append(cpx)

    return complexes, uniprot_to_complexes


# ---------------------------------------------------------------------------
# RCSB download (shared by steps 1 and 4)
# ---------------------------------------------------------------------------

def download_pdb_or_cif(pdb_id: str, raw_dir: Path) -> tuple[str, str]:
    """Download one structure. Returns (status, format): ('ok'|'skip'|'fail', 'pdb'|'cif'|'')."""
    pdb_path = raw_dir / f"{pdb_id}.pdb"
    cif_path = raw_dir / f"{pdb_id}.cif"
    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        return ("skip", "pdb")
    if cif_path.exists() and cif_path.stat().st_size > 0:
        return ("skip", "cif")

    # Try .pdb
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(RCSB_PDB_URL.format(pdb_id=pdb_id),
                                         headers={"User-Agent": "PipelinePDBDownloader/1.0"})
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                content = resp.read()
            if content:
                pdb_path.write_bytes(content)
                return ("ok", "pdb")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                break
            log(f"  [RETRY] {pdb_id} HTTP {e.code} (attempt {attempt})")
        except Exception as e:
            log(f"  [RETRY] {pdb_id} {e} (attempt {attempt})")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    # Fall back to .cif
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(RCSB_CIF_URL.format(pdb_id=pdb_id),
                                         headers={"User-Agent": "PipelinePDBDownloader/1.0"})
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                content = resp.read()
            if content:
                cif_path.write_bytes(content)
                return ("ok", "cif")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log(f"  [FAIL] {pdb_id} not found as .pdb or .cif")
                return ("fail", "")
            log(f"  [RETRY] {pdb_id} CIF HTTP {e.code} (attempt {attempt})")
        except Exception as e:
            log(f"  [RETRY] {pdb_id} CIF {e} (attempt {attempt})")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return ("fail", "")


# ---------------------------------------------------------------------------
# SIFTS cache
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"best_structures": {}, "pdb_uniprots": {}}


def save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    tmp.replace(path)


def query_best_structures(uniprot: str, cache: dict) -> list[dict]:
    if uniprot not in cache["best_structures"]:
        data = _get_json(SIFTS_BEST_STRUCTURES_URL.format(u=uniprot), "PipelineSIFTS/1.0")
        cache["best_structures"][uniprot] = data.get(uniprot, []) if data else []
        time.sleep(SIFTS_DELAY)
    return cache["best_structures"][uniprot]


def query_pdb_uniprots(pdb_id: str, cache: dict) -> set[str]:
    key = pdb_id.lower()
    if key not in cache["pdb_uniprots"]:
        data = _get_json(SIFTS_PDB_UNIPROT_URL.format(pdb=key), "PipelineSIFTS/1.0")
        uniprots = set(data[key].get("UniProt", {}).keys()) if data and key in data else set()
        cache["pdb_uniprots"][key] = sorted(uniprots)
        time.sleep(SIFTS_DELAY)
    return set(cache["pdb_uniprots"][key])


# ---------------------------------------------------------------------------
# STEP 1 — download Complex Portal annotated PDBs
# ---------------------------------------------------------------------------

def step1_download_complex_portal_pdbs(complexes: dict, out_dir: Path) -> Path:
    log("\n" + "=" * 70)
    log("STEP 1: Download Complex Portal annotated PDBs")
    log("=" * 70)

    raw_dir = out_dir / "_raw"
    tag_dirs = {t: out_dir / t for t in TAGS}
    for d in (raw_dir, *tag_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    # Collect unique PDBs and their (cpx, tag) references
    pdb_to_cpxs: dict[str, dict[str, list[str]]] = {}
    for cpx, info in complexes.items():
        for tag in TAGS:
            for pdb_id in info["xrefs"][tag]:
                pdb_to_cpxs.setdefault(pdb_id, {t: [] for t in TAGS})
                if cpx not in pdb_to_cpxs[pdb_id][tag]:
                    pdb_to_cpxs[pdb_id][tag].append(cpx)

    log(f"  Unique Complex Portal PDBs to download: {len(pdb_to_cpxs)}")
    downloaded = skipped = failed = 0
    for i, pdb_id in enumerate(sorted(pdb_to_cpxs), 1):
        status, fmt = download_pdb_or_cif(pdb_id, raw_dir)
        if status == "ok":
            downloaded += 1
            log(f"  [{i}/{len(pdb_to_cpxs)}] {pdb_id}: downloaded ({fmt})")
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
        time.sleep(RCSB_DELAY)

    # Per-CPX copies
    log("  Organizing per-CPX copies into identity/, subset/, experimental_evidence/...")
    for pdb_id, tags in pdb_to_cpxs.items():
        for fmt_ext in (".pdb", ".cif"):
            src = raw_dir / f"{pdb_id}{fmt_ext}"
            if src.exists():
                break
        else:
            continue
        for tag in TAGS:
            for cpx in tags[tag]:
                dest = tag_dirs[tag] / f"{cpx}_{pdb_id}{src.suffix}"
                if not dest.exists():
                    dest.write_bytes(src.read_bytes())

    # Mapping CSV
    mapping_path = out_dir / "complex_pdb_mapping.csv"
    with open(mapping_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["#Complex ac", "pdb_id", "tag"])
        for cpx in sorted(complexes):
            for tag in TAGS:
                for pdb_id in complexes[cpx]["xrefs"][tag]:
                    w.writerow([cpx, pdb_id, tag])

    log(f"  Done. Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    log(f"  Mapping CSV: {mapping_path}")
    return mapping_path


# ---------------------------------------------------------------------------
# STEP 2 — SIFTS best_structures for all proteins → uniprot_pdb_mapping.csv
# ---------------------------------------------------------------------------

def step2_sifts_uniprot_to_pdb(complexes: dict, uniprot_to_complexes: dict,
                               cache: dict, out_dir: Path) -> Path:
    log("\n" + "=" * 70)
    log("STEP 2: Query SIFTS for all PDBs each protein appears in")
    log("=" * 70)

    all_uniprots = sorted(uniprot_to_complexes.keys())
    log(f"  Querying SIFTS best_structures for {len(all_uniprots)} unique proteins...")

    rows = []
    for i, u in enumerate(all_uniprots, 1):
        entries = query_best_structures(u, cache)
        if entries:
            cpxs = uniprot_to_complexes.get(u, [])
            cpx_str = ";".join(cpxs)
            name_str = ";".join(complexes[c]["name"] for c in cpxs)
            for e in entries:
                rows.append({
                    "uniprot_id": u,
                    "pdb_id": str(e.get("pdb_id", "")).upper(),
                    "chain_id": e.get("chain_id", ""),
                    "experimental_method": e.get("experimental_method", ""),
                    "resolution": e.get("resolution"),
                    "coverage": e.get("coverage"),
                    "unp_start": e.get("unp_start"),
                    "unp_end": e.get("unp_end"),
                    "tax_id": e.get("tax_id"),
                    "complex_accessions": cpx_str,
                    "complex_names": name_str,
                })
        if i % 200 == 0:
            log(f"    [{i}/{len(all_uniprots)}] processed")

    df1 = pd.DataFrame(rows)
    uniprot_pdb_dir = out_dir / "uniprot_pdb"
    uniprot_pdb_dir.mkdir(parents=True, exist_ok=True)
    df1_path = uniprot_pdb_dir / "uniprot_pdb_mapping.csv"
    df1.to_csv(df1_path, index=False)
    log(f"  Done. {len(df1)} rows, {df1['uniprot_id'].nunique()} proteins with >=1 PDB.")
    log(f"  CSV: {df1_path}")
    return df1_path


# ---------------------------------------------------------------------------
# STEP 3 — exact protein-set matches → complex_pdb_exact_match.csv
# ---------------------------------------------------------------------------

def step3_exact_matches(complexes: dict, cache: dict, out_dir: Path) -> Path:
    log("\n" + "=" * 70)
    log("STEP 3: Find PDBs with exact protein-set match for each complex")
    log("=" * 70)

    # Collect all PDBs seen in best_structures, get each one's full UniProt set
    all_pdbs = {str(e.get("pdb_id", "")).upper()
                for entries in cache["best_structures"].values()
                for e in entries if e.get("pdb_id")}
    log(f"  Querying SIFTS /mappings/uniprot for {len(all_pdbs)} unique PDBs...")

    pdb_proteins: dict[str, set[str]] = {}
    for i, pdb in enumerate(sorted(all_pdbs), 1):
        pdb_proteins[pdb] = query_pdb_uniprots(pdb, cache)
        if i % 200 == 0:
            log(f"    [{i}/{len(all_pdbs)}] PDBs processed")

    uniprots_with_pdb = {u for u, entries in cache["best_structures"].items() if entries}

    # Pre-index best resolution per PDB
    pdb_best_res: dict[str, float | None] = {}
    for entries in cache["best_structures"].values():
        for e in entries:
            pdb = str(e.get("pdb_id", "")).upper()
            if not pdb:
                continue
            res = e.get("resolution")
            if res is None:
                continue
            if pdb not in pdb_best_res or res < pdb_best_res[pdb]:
                pdb_best_res[pdb] = res

    def res_key(pdb: str) -> tuple:
        res = pdb_best_res.get(pdb)
        return (res is None, res if res is not None else 0.0, pdb)

    rows = []
    for cpx in sorted(complexes):
        cpx_prots = complexes[cpx]["proteins"]
        n_complex = len(cpx_prots)
        n_with_pdb = len(cpx_prots & uniprots_with_pdb)
        exact_pdbs = sorted([pdb for pdb, prots in pdb_proteins.items() if prots == cpx_prots])

        if exact_pdbs:
            best = min(exact_pdbs, key=res_key)
            best_res = pdb_best_res.get(best)
            match_class = "exact_match"
        else:
            best, best_res = "", None
            match_class = "no_match"

        rows.append({
            "complex_accession": cpx,
            "complex_name": complexes[cpx]["name"],
            "n_complex_proteins": n_complex,
            "n_exact_match_pdbs": len(exact_pdbs),
            "all_exact_pdbs": ";".join(exact_pdbs),
            "best_pdb_id": best,
            "best_pdb_resolution": best_res if best_res is not None else "",
            "match_class": match_class,
            "complex_proteins": ";".join(sorted(cpx_prots)),
            "n_proteins_with_pdb": n_with_pdb,
        })

    df2 = pd.DataFrame(rows)
    uniprot_pdb_dir = out_dir / "uniprot_pdb"
    df2_path = uniprot_pdb_dir / "complex_pdb_exact_match.csv"
    df2.to_csv(df2_path, index=False)
    n_exact = (df2["match_class"] == "exact_match").sum()
    log(f"  Done. {len(df2)} complexes; {n_exact} with >=1 exact-match PDB.")
    log(f"  CSV: {df2_path}")
    return df2_path


# ---------------------------------------------------------------------------
# STEP 4 — download all exact-match PDBs (skip those already in _raw/)
# ---------------------------------------------------------------------------

def step4_download_exact_match_pdbs(exact_csv: Path, raw_dir: Path, out_dir: Path) -> Path:
    log("\n" + "=" * 70)
    log("STEP 4: Download all exact-match PDBs")
    log("=" * 70)

    df = pd.read_csv(exact_csv)
    exact_pdbs = set()
    for val in df[df["match_class"] == "exact_match"]["all_exact_pdbs"].dropna():
        for p in str(val).split(";"):
            p = p.strip()
            if p:
                exact_pdbs.add(p.upper())

    log(f"  Exact-match PDBs to ensure downloaded: {len(exact_pdbs)}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = failed = 0
    log_rows = []
    for i, pdb_id in enumerate(sorted(exact_pdbs), 1):
        status, fmt = download_pdb_or_cif(pdb_id, raw_dir)
        log_rows.append({"pdb_id": pdb_id, "status": status, "format": fmt})
        if status == "ok":
            downloaded += 1
            log(f"  [{i}/{len(exact_pdbs)}] {pdb_id}: downloaded ({fmt})")
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
        time.sleep(RCSB_DELAY)

    log_path = out_dir / "exact_match_download_log.csv"
    with open(log_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id", "status", "format"])
        w.writeheader()
        w.writerows(log_rows)

    log(f"  Done. Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    log(f"  Log: {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True, type=Path,
                    help="Saccharomyces_cerevisiae_ComplexTab.tsv")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory (all results land here)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse TSV and report counts; no downloads, no API calls.")
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"TSV not found: {args.tsv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Reading {args.tsv}")
    complexes, uniprot_to_complexes = parse_tsv(args.tsv)
    all_uniprots = sorted(uniprot_to_complexes.keys())

    n_with_xref = sum(1 for c in complexes if any(complexes[c]["xrefs"][t] for t in TAGS))
    unique_xref_pdbs = set()
    for c in complexes:
        for t in TAGS:
            unique_xref_pdbs.update(complexes[c]["xrefs"][t])

    log(f"  Complexes: {len(complexes)}")
    log(f"  Unique UniProt accessions: {len(all_uniprots)}")
    log(f"  Complexes with Complex Portal PDB xref: {n_with_xref}")
    log(f"  Unique Complex Portal PDB IDs (to download in step 1): {len(unique_xref_pdbs)}")

    if args.dry_run:
        log("\n[DRY-RUN] No downloads, no API calls. No files written.")
        for cpx in sorted(complexes)[:5]:
            info = complexes[cpx]
            log(f"  {cpx} ({info['name']}): proteins={sorted(info['proteins'])}, "
                f"xrefs={info['xrefs']}")
        return

    cache_path = args.out_dir / "sifts_cache.json"
    cache = load_cache(cache_path)
    log(f"  SIFTS cache: {len(cache['best_structures'])} proteins, "
        f"{len(cache['pdb_uniprots'])} PDBs already queried")

    # STEP 1
    step1_download_complex_portal_pdbs(complexes, args.out_dir)
    save_cache(cache_path, cache)

    # STEP 2
    step2_sifts_uniprot_to_pdb(complexes, uniprot_to_complexes, cache, args.out_dir)
    save_cache(cache_path, cache)

    # STEP 3
    exact_csv = step3_exact_matches(complexes, cache, args.out_dir)
    save_cache(cache_path, cache)

    # STEP 4
    raw_dir = args.out_dir / "_raw"
    step4_download_exact_match_pdbs(exact_csv, raw_dir, args.out_dir)

    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE")
    log("=" * 70)
    log(f"  Output dir: {args.out_dir}")
    log(f"  Reference PDBs in _raw/: {len(list(raw_dir.glob('*.pdb'))) + len(list(raw_dir.glob('*.cif')))}")
    log(f"  Next: run 19_compare_combfold_to_pdb.py to score CombFold predictions.")


if __name__ == "__main__":
    main()
