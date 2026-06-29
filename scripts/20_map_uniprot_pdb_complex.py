#!/usr/bin/env python
"""
17_map_uniprot_pdb_complex.py — Map Complex Portal yeast complexes to PDB
structures via EBI SIFTS, producing two dataframes.

DF1 uniprot_pdb_mapping.csv  — one row per (UniProt, PDB, chain) from SIFTS.
DF2 complex_pdb_exact_match.csv — one row per Complex Portal complex (all 634);
   reports PDBs whose protein set EXACTLY equals the complex's protein set.

Exact match = set equality of UniProt accessions (stoichiometry ignored).
Non-protein members (CHEBI/URS/CPX-/EBI-) are dropped. Bracketed paralog
groups like [P0CX46,P0CX45] are expanded. PRO_ isoforms (P00410-PRO_xxx)
are reduced to their base UniProt.

Usage:
    python 17_map_uniprot_pdb_complex.py \
        --tsv  data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv \
        --out-dir data/Complex_pdb_files/uniprot_pdb

    # audit-only (no SIFTS calls):
    python 17_map_uniprot_pdb_complex.py --tsv ... --out-dir ... --dry-run

Resume: sifts_cache.json in out-dir stores all SIFTS responses.
"""

from __future__ import annotations

import argparse
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
BEST_STRUCTURES_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{u}"
PDB_UNIPROT_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
API_DELAY = 0.1
API_TIMEOUT = 30

UNIPROT_RE = re.compile(r"^[OPQ][0-9][0-9A-Z]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
BRACKET_RE = re.compile(r"^\[([A-Z0-9_,\s]+)\]$")
PRO_ISOFORM_RE = re.compile(r"^([OPQ][0-9][0-9A-Z]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})-PRO_\d+$")


# ---------------------------------------------------------------------------
# Step 1 — extract protein IDs from the TSV
# ---------------------------------------------------------------------------

def extract_proteins(member: str) -> set[str]:
    """Return the set of UniProt accessions in one molecule-list token.

    Handles plain UniProts, [bracketed paralog groups], and PRO_ isoforms.
    Returns empty set for CHEBI/URS/CPX-/EBI- and anything non-protein.
    """
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


def parse_tsv(tsv_path: Path) -> tuple[
    dict[str, set[str]],   # CPX -> UniProt set
    dict[str, str],        # CPX -> recommended name
    dict[str, list[str]],  # UniProt -> CPX IDs it belongs to
]:
    df = pd.read_csv(tsv_path, sep="\t")
    mol_col = "Identifiers (and stoichiometry) of molecules in complex"
    name_col = "Recommended name"

    complex_proteins: dict[str, set[str]] = {}
    complex_names: dict[str, str] = {}
    uniprot_to_complexes: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        cpx = str(row["#Complex ac"]).strip()
        if not cpx:
            continue
        complex_names[cpx] = str(row.get(name_col, "")).strip()

        raw = row.get(mol_col, "")
        proteins: set[str] = set()
        if isinstance(raw, str):
            for token in raw.split("|"):
                proteins |= extract_proteins(token.split("(")[0])

        complex_proteins[cpx] = proteins
        for u in proteins:
            uniprot_to_complexes.setdefault(u, [])
            if cpx not in uniprot_to_complexes[u]:
                uniprot_to_complexes[u].append(cpx)

    return complex_proteins, complex_names, uniprot_to_complexes


# ---------------------------------------------------------------------------
# Step 2 — SIFTS API (cached for resume)
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


def _get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SIFTSComplexMapper/2.0"})
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"    HTTP {e.code}: {url}")
        return None
    except Exception as e:
        print(f"    ERR {e}: {url}")
        return None


def query_best_structures(uniprot: str, cache: dict) -> list[dict]:
    if uniprot not in cache["best_structures"]:
        data = _get_json(BEST_STRUCTURES_URL.format(u=uniprot))
        cache["best_structures"][uniprot] = data.get(uniprot, []) if data else []
        time.sleep(API_DELAY)
    return cache["best_structures"][uniprot]


def query_pdb_uniprots(pdb_id: str, cache: dict) -> set[str]:
    key = pdb_id.lower()
    if key not in cache["pdb_uniprots"]:
        data = _get_json(PDB_UNIPROT_URL.format(pdb=key))
        uniprots = set(data[key].get("UniProt", {}).keys()) if data and key in data else set()
        cache["pdb_uniprots"][key] = sorted(uniprots)
        time.sleep(API_DELAY)
    return set(cache["pdb_uniprots"][key])


# ---------------------------------------------------------------------------
# Step 3 — build DF1 (UniProt -> PDB, long format)
# ---------------------------------------------------------------------------

def build_df1(uniprot_to_complexes, complex_names, cache, all_uniprots) -> pd.DataFrame:
    rows = []
    for u in all_uniprots:
        entries = query_best_structures(u, cache)
        if not entries:
            continue
        cpxs = uniprot_to_complexes.get(u, [])
        cpx_str = ";".join(cpxs)
        name_str = ";".join(complex_names.get(c, "") for c in cpxs)
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
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4 — build DF2 (complex -> exact-match PDBs, all 634 rows)
# ---------------------------------------------------------------------------

def build_df2(complex_proteins, complex_names, cache) -> pd.DataFrame:
    # Collect every PDB seen in best_structures, get its full UniProt set.
    all_pdbs = {str(e.get("pdb_id", "")).upper()
                for entries in cache["best_structures"].values()
                for e in entries if e.get("pdb_id")}
    pdb_proteins: dict[str, set[str]] = {}
    for pdb in all_pdbs:
        pdb_proteins[pdb] = query_pdb_uniprots(pdb, cache)

    # UniProts that have at least one PDB structure (diagnostic column).
    uniprots_with_pdb = {u for u, entries in cache["best_structures"].items() if entries}

    # Pre-index best_structures by PDB ID so best-PDB lookup is O(1), not O(n).
    # For each PDB, keep the lowest non-null resolution across all its chains.
    pdb_best_resolution: dict[str, float | None] = {}
    for entries in cache["best_structures"].values():
        for e in entries:
            pdb = str(e.get("pdb_id", "")).upper()
            if not pdb:
                continue
            res = e.get("resolution")
            if res is None:
                continue
            if pdb not in pdb_best_resolution or res < pdb_best_resolution[pdb]:
                pdb_best_resolution[pdb] = res

    def res_key(pdb: str) -> tuple:
        """Sort key: lowest resolution first; None (NMR) sorts last; tiebreak PDB ID."""
        res = pdb_best_resolution.get(pdb)
        return (res is None, res if res is not None else 0.0, pdb)

    rows = []
    for cpx in sorted(complex_proteins):
        cpx_prots = complex_proteins[cpx]
        n_complex = len(cpx_prots)
        n_with_pdb = len(cpx_prots & uniprots_with_pdb)

        # Exact match: PDB's UniProt set == complex's UniProt set.
        exact_pdbs = sorted([pdb for pdb, prots in pdb_proteins.items() if prots == cpx_prots])

        if exact_pdbs:
            best = min(exact_pdbs, key=res_key)
            best_res = pdb_best_resolution.get(best)
            match_class = "exact_match"
        else:
            best, best_res = "", None
            match_class = "no_match"

        rows.append({
            "complex_accession": cpx,
            "complex_name": complex_names.get(cpx, ""),
            "n_complex_proteins": n_complex,
            "n_exact_match_pdbs": len(exact_pdbs),
            "all_exact_pdbs": ";".join(exact_pdbs),
            "best_pdb_id": best,
            "best_pdb_resolution": best_res if best_res is not None else "",
            "match_class": match_class,
            "complex_proteins": ";".join(sorted(cpx_prots)),
            "n_proteins_with_pdb": n_with_pdb,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse TSV and report counts; no SIFTS calls, no CSVs.")
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"TSV not found: {args.tsv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.tsv}")
    complex_proteins, complex_names, uniprot_to_complexes = parse_tsv(args.tsv)
    all_uniprots = sorted(uniprot_to_complexes.keys())

    print(f"  Complexes: {len(complex_proteins)}")
    print(f"  Unique UniProt accessions: {len(all_uniprots)}")

    if args.dry_run:
        print("\n[DRY-RUN] No SIFTS calls. No CSVs written.")
        for cpx in sorted(complex_proteins)[:5]:
            print(f"  {cpx} ({complex_names.get(cpx,'')}): {sorted(complex_proteins[cpx])}")
        return

    cache_path = args.out_dir / "sifts_cache.json"
    cache = load_cache(cache_path)
    print(f"  Cache: {len(cache['best_structures'])} UniProts, "
          f"{len(cache['pdb_uniprots'])} PDBs already queried")

    print(f"\nStep 1/3: SIFTS best_structures for {len(all_uniprots)} UniProts...")
    df1 = build_df1(uniprot_to_complexes, complex_names, cache, all_uniprots)
    save_cache(cache_path, cache)
    df1_path = args.out_dir / "uniprot_pdb_mapping.csv"
    df1.to_csv(df1_path, index=False)
    print(f"  DF1: {df1_path}  ({len(df1)} rows, {df1['uniprot_id'].nunique()} UniProts)")

    print(f"\nStep 2/3: SIFTS /mappings/uniprot for unique PDBs...")
    df2 = build_df2(complex_proteins, complex_names, cache)
    save_cache(cache_path, cache)
    df2_path = args.out_dir / "complex_pdb_exact_match.csv"
    df2.to_csv(df2_path, index=False)
    print(f"  DF2: {df2_path}  ({len(df2)} rows)")

    print("\n=== Summary ===")
    print(f"  DF1 rows (UniProt-PDB-chain): {len(df1)}")
    vc = df2["match_class"].value_counts()
    for cls, n in vc.items():
        print(f"  DF2 {cls}: {n}")
    n_with_exact = (df2["n_exact_match_pdbs"] > 0).sum()
    print(f"  Complexes with >=1 exact-match PDB: {n_with_exact} / {len(df2)}")


if __name__ == "__main__":
    main()
