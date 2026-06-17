#!/usr/bin/env python3
"""
Find PDB structures for yeast protein complexes.

Supports two input formats:
  1. Complex Portal TSV (Saccharomyces cerevisiae_ComplexTab.tsv)
  2. Yeast.MAP CSV (yeast.MAP_complexes_wConfidenceScores_wGenenames_total779_20251214.csv)

For each complex, finds PDB entries that contain ALL protein subunits
(complete match), which can then be used as ground truth for evaluating
CombFold predictions.

Usage:
    # Complex Portal format (auto-detected by .tsv extension or --format complex-portal):
    python find_pdb_structures_for_yeast_complexes.py \
        --complexes Saccharomyces\ cerevisiae_ComplexTab.tsv \
        --output complex_portal_pdb_mapping.csv \
        --pdb-mirror /scicore/data/managed/PDB/latest

    # Yeast.MAP format (auto-detected by .csv extension or --format yeast-map):
    python find_pdb_structures_for_yeast_complexes.py \
        --complexes yeast.MAP_complexes_wConfidenceScores_wGenenames_total779_20251214.csv \
        --output yeast_map_pdb_mapping.csv \
        --pdb-mirror /scicore/data/managed/PDB/latest

Output CSV columns:
    - complex_id: Complex Portal ID (CPX-xxxx) or yeastMAP_ID
    - recommended_name: complex name (Complex Portal only, empty for yeast.MAP)
    - n_protein_subunits: number of UniProt protein subunits
    - stoichiometry: JSON {UniProt: copy_number} (Complex Portal only)
    - UniProt_ACCs: space-separated UniProt accessions
    - has_complete_match: whether any PDB entry contains all protein subunits
    - n_complete_pdbs: number of PDB entries with all protein subunits
    - best_pdb_id: PDB entry with best resolution among complete matches
    - best_pdb_resolution, best_pdb_method: resolution and method of best PDB
    - best_pdb_path: local file path on the PDB mirror (if --pdb-mirror given)
    - best_pdb_url: RCSB download URL for the PDB file
    - all_complete_pdb_ids: semicolon-separated list of all matching PDB IDs
    - all_complete_pdb_paths: semicolon-separated local paths (if --pdb-mirror)
    - pdb_chain_details: JSON with per-UniProt chain counts for best PDB
    - complex_portal_pdb_xrefs: PDB IDs from Complex Portal cross-references (if available)
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# 1. Download / cache SIFTS UniProt→PDB mapping
# ---------------------------------------------------------------------------

SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_pdb.tsv.gz"


def download_sifts(cache_path):
    """Download and parse the SIFTS uniprot_pdb.tsv.gz file."""
    gz_path = cache_path + ".gz"

    if not os.path.exists(cache_path):
        print(f"Downloading SIFTS mapping from {SIFTS_URL} ...")
        urllib.request.urlretrieve(SIFTS_URL, gz_path)
        print(f"  Downloaded {os.path.getsize(gz_path)/1e6:.1f} MB, decompressing ...")
        with gzip.open(gz_path, "rb") as fin, open(cache_path, "wb") as fout:
            fout.write(fin.read())
        os.remove(gz_path)
        print(f"  Saved to {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)")
    else:
        print(f"Using cached SIFTS file: {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)")

    print("Parsing SIFTS mapping ...")
    sifts = pd.read_csv(cache_path, sep="\t", comment="#")
    sifts["PDB_list"] = sifts["PDB"].str.split(";")
    exploded = sifts.explode("PDB_list")
    exploded = exploded.rename(columns={"SP_PRIMARY": "uniprot_id", "PDB_list": "pdb_id"})
    exploded["pdb_id"] = exploded["pdb_id"].str.strip().str.lower()
    exploded = exploded[["uniprot_id", "pdb_id"]].dropna()
    exploded = exploded[exploded["pdb_id"] != ""]

    uniprot_to_pdb = exploded.groupby("uniprot_id")["pdb_id"].apply(set).to_dict()
    print(f"  {len(uniprot_to_pdb)} UniProt IDs mapped to PDB entries")
    return uniprot_to_pdb


# ---------------------------------------------------------------------------
# 2. Query RCSB Data API for PDB entry metadata
# ---------------------------------------------------------------------------

def _get_pdb_entry_info(pdb_id):
    """Return (resolution, method) for a single PDB entry via RCSB Data API."""
    try:
        resp = requests.get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}",
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            info = data.get("rcsb_entry_info", {})
            res = info.get("resolution_combined")
            if isinstance(res, list) and res:
                res = res[0]
            method = info.get("experimental_method")
            if isinstance(method, list) and method:
                method = method[0]
            return pdb_id.lower(), res, method, "ok"
        return pdb_id.lower(), None, None, f"http_{resp.status_code}"
    except Exception as exc:
        return pdb_id.lower(), None, None, f"error:{exc}"


def batch_pdb_info(pdb_ids, workers=20):
    """Fetch resolution + method for many PDB entries concurrently."""
    results = {}
    print(f"Querying RCSB Data API for {len(pdb_ids)} PDB entries ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_get_pdb_entry_info, pid): pid for pid in pdb_ids}
        done = 0
        for fut in as_completed(futures):
            pid, res, method, status = fut.result()
            if status == "ok":
                results[pid] = {"resolution": res, "method": method}
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(pdb_ids)}")
    print(f"  Got info for {len(results)} entries")
    return results


# ---------------------------------------------------------------------------
# 3. Query SIFTS REST API for chain-level UniProt mapping (stoichiometry)
# ---------------------------------------------------------------------------

def _get_sifts_chains(pdb_id):
    """Return {uniprot_id: [chain_ids]} for a single PDB entry via SIFTS REST API."""
    try:
        resp = requests.get(
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}",
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            pdb_data = data.get(pdb_id.lower(), {})
            uniprot_data = pdb_data.get("UniProt", {})
            mapping = {}
            for uid, info in uniprot_data.items():
                chains = set()
                for m in info.get("mappings", []):
                    cid = m.get("chain_id")
                    if cid:
                        chains.add(cid)
                mapping[uid] = sorted(chains)
            return pdb_id.lower(), mapping, "ok"
        return pdb_id.lower(), {}, f"http_{resp.status_code}"
    except Exception as exc:
        return pdb_id.lower(), {}, f"error:{exc}"


def batch_sifts_chains(pdb_ids, workers=10):
    """Fetch chain-level UniProt mapping for many PDB entries concurrently."""
    results = {}
    print(f"Querying SIFTS REST API for {len(pdb_ids)} PDB entries ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_get_sifts_chains, pid): pid for pid in pdb_ids}
        done = 0
        for fut in as_completed(futures):
            pid, mapping, status = fut.result()
            if status == "ok":
                results[pid] = mapping
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(pdb_ids)}")
    print(f"  Got chain info for {len(results)} entries")
    return results


# ---------------------------------------------------------------------------
# 4. PDB file path helpers
# ---------------------------------------------------------------------------

def pdb_mirror_path(pdb_id, mirror_root):
    """Construct the local mmCIF path on the PDB mirror."""
    mid2 = pdb_id[1:3]
    return f"{mirror_root}/data/structures/divided/mmCIF/{mid2}/{pdb_id}.cif.gz"


def pdb_download_url(pdb_id):
    """RCSB download URL for the PDB format file."""
    return f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"


# ---------------------------------------------------------------------------
# 5. Input parsing
# ---------------------------------------------------------------------------

UNIPROT_RE = re.compile(r"^[A-NR-Z][0-9][A-Z][A-Z0-9]{4}|[OPQ][0-9][A-Z0-9]{3}[0-9]$")


def parse_complex_portal(filepath):
    """Parse Complex Portal TSV into standardized complex records."""
    df = pd.read_csv(filepath, sep="\t")
    print(f"  Loaded {len(df)} complexes from Complex Portal TSV")

    records = []
    for _, row in df.iterrows():
        # Parse identifiers with stoichiometry
        id_col = "Identifiers (and stoichiometry) of molecules in complex"
        id_string = row.get(id_col, "")
        if pd.isna(id_string) or id_string == "-":
            continue

        parts = str(id_string).split("|")
        all_ids = []
        stoichiometry = {}
        for part in parts:
            match = re.match(r"^([A-Za-z0-9_]+)\((\d+)\)$", part.strip())
            if match:
                uid, copies = match.groups()
                all_ids.append(uid)
                stoichiometry[uid] = int(copies)
            else:
                all_ids.append(part.strip())

        # Filter to UniProt protein IDs only
        protein_ids = [uid for uid in all_ids if UNIPROT_RE.match(uid)]

        # Extract PDB cross-references from Complex Portal
        xref_string = str(row.get("Cross references", ""))
        pdb_xrefs = []
        if not pd.isna(xref_string) and xref_string != "-":
            for xpart in xref_string.split("|"):
                xpart = xpart.strip()
                if xpart.startswith("wwpdb:"):
                    pdb_id = xpart.split(":")[1].split("(")[0]
                    pdb_xrefs.append(pdb_id.lower())

        records.append({
            "complex_id": row["#Complex ac"],
            "recommended_name": row.get("Recommended name", ""),
            "UniProt_ACCs": " ".join(protein_ids),
            "n_protein_subunits": len(protein_ids),
            "n_total_subunits": len(all_ids),
            "stoichiometry": json.dumps(stoichiometry) if stoichiometry else "",
            "complex_portal_pdb_xrefs": ";".join(sorted(pdb_xrefs)),
            "_uniprot_set": set(protein_ids),
        })

    return pd.DataFrame(records)


def parse_yeast_map(filepath):
    """Parse yeast.MAP CSV into standardized complex records."""
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} complexes from yeast.MAP CSV")

    records = []
    for _, row in df.iterrows():
        uniprots = [u.strip() for u in str(row["UniProt_ACCs"]).split() if u.strip()]
        protein_ids = [u for u in uniprots if UNIPROT_RE.match(u)]

        records.append({
            "complex_id": row["yeastMAP_ID"],
            "recommended_name": "",
            "UniProt_ACCs": " ".join(protein_ids),
            "n_protein_subunits": len(protein_ids),
            "n_total_subunits": len(uniprots),
            "stoichiometry": "",
            "complex_portal_pdb_xrefs": "",
            "_uniprot_set": set(protein_ids),
        })

    return pd.DataFrame(records)


def load_complexes(filepath, fmt=None):
    """Load complexes from file, auto-detecting format if needed."""
    if fmt is None:
        ext = os.path.splitext(filepath)[1].lower()
        fmt = "complex-portal" if ext == ".tsv" else "yeast-map"

    if fmt == "complex-portal":
        return parse_complex_portal(filepath)
    else:
        return parse_yeast_map(filepath)


# ---------------------------------------------------------------------------
# 6. Match complexes to PDB structures
# ---------------------------------------------------------------------------

def match_complexes_to_pdb(complexes_df, uniprot_to_pdb):
    """For each complex, find PDB entries that contain ALL its protein UniProt IDs."""
    all_complete_pdb_ids = set()
    n_complete_pdbs_list = []
    all_complete_ids_list = []
    has_match_list = []

    for _, row in complexes_df.iterrows():
        uniprot_set = row["_uniprot_set"]

        if len(uniprot_set) < 2:
            # Skip single-protein "complexes"
            n_complete_pdbs_list.append(0)
            all_complete_ids_list.append("")
            has_match_list.append(False)
            continue

        subunit_pdbs = {u: uniprot_to_pdb.get(u, set()) for u in uniprot_set}

        if all(len(p) > 0 for p in subunit_pdbs.values()):
            common = set.intersection(*subunit_pdbs.values())
        else:
            common = set()

        all_complete_pdb_ids.update(common)
        n_complete_pdbs_list.append(len(common))
        all_complete_ids_list.append(";".join(sorted(common)) if common else "")
        has_match_list.append(len(common) > 0)

    complexes_df = complexes_df.copy()
    complexes_df["has_complete_match"] = has_match_list
    complexes_df["n_complete_pdbs"] = n_complete_pdbs_list
    complexes_df["all_complete_pdb_ids"] = all_complete_ids_list

    return complexes_df, all_complete_pdb_ids


def annotate_best_pdb(complexes_df, pdb_info, sifts_chains, pdb_mirror=None):
    """Add best-PDB columns (resolution, method, chain details, file paths)."""
    best_pdb_ids = []
    best_resolutions = []
    best_methods = []
    best_pdb_paths = []
    best_pdb_urls = []
    all_pdb_paths = []
    chain_details_list = []

    for _, row in complexes_df.iterrows():
        common_pdbs_str = row.get("all_complete_pdb_ids", "")
        uniprot_set = row["_uniprot_set"]

        if not common_pdbs_str:
            best_pdb_ids.append("")
            best_resolutions.append(None)
            best_methods.append("")
            best_pdb_paths.append("")
            best_pdb_urls.append("")
            all_pdb_paths.append("")
            chain_details_list.append("")
            continue

        common_pdbs = set(common_pdbs_str.split(";"))

        # Pick best PDB: prefer entries with resolution (X-ray/EM) over NMR,
        # then pick best resolution
        best_pdb = None
        best_res = None
        best_method = None
        best_has_res = False

        for pid in sorted(common_pdbs):
            info = pdb_info.get(pid, {})
            res = info.get("resolution")
            method = info.get("method", "")

            if res is not None:
                if not best_has_res:
                    best_pdb = pid
                    best_res = res
                    best_method = method
                    best_has_res = True
                elif res < best_res:
                    best_pdb = pid
                    best_res = res
                    best_method = method
            else:
                if best_pdb is None:
                    best_pdb = pid
                    best_res = None
                    best_method = method

        if best_pdb is None:
            best_pdb = sorted(common_pdbs)[0]

        best_pdb_ids.append(best_pdb)
        best_resolutions.append(best_res)
        best_methods.append(best_method or pdb_info.get(best_pdb, {}).get("method", ""))

        # File paths
        if pdb_mirror:
            best_pdb_paths.append(pdb_mirror_path(best_pdb, pdb_mirror))
            all_p = ";".join(pdb_mirror_path(p, pdb_mirror) for p in sorted(common_pdbs))
            all_pdb_paths.append(all_p)
        else:
            best_pdb_paths.append("")
            all_pdb_paths.append("")
        best_pdb_urls.append(pdb_download_url(best_pdb))

        # Chain-level stoichiometry details for best PDB
        chains = sifts_chains.get(best_pdb, {})
        detail = {}
        for u in uniprot_set:
            if u in chains:
                detail[u] = {"n_chains": len(chains[u]), "chains": chains[u]}
            else:
                detail[u] = {"n_chains": 0, "chains": []}
        chain_details_list.append(json.dumps(detail))

    complexes_df["best_pdb_id"] = best_pdb_ids
    complexes_df["best_pdb_resolution"] = best_resolutions
    complexes_df["best_pdb_method"] = best_methods
    complexes_df["best_pdb_path"] = best_pdb_paths
    complexes_df["best_pdb_url"] = best_pdb_urls
    complexes_df["all_complete_pdb_paths"] = all_pdb_paths
    complexes_df["pdb_chain_details"] = chain_details_list

    # Drop temporary column
    complexes_df = complexes_df.drop(columns=["_uniprot_set"])
    return complexes_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find PDB structures for yeast protein complexes (complete match only)."
    )
    parser.add_argument(
        "--complexes",
        required=True,
        help="Path to complexes file (Complex Portal TSV or yeast.MAP CSV)",
    )
    parser.add_argument(
        "--format",
        choices=["complex-portal", "yeast-map", "auto"],
        default="auto",
        help="Input format (default: auto-detect from file extension)",
    )
    parser.add_argument(
        "--output",
        default="yeast_complexes_pdb_mapping.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of concurrent API requests",
    )
    parser.add_argument(
        "--sifts-cache",
        default="sifts_uniprot_pdb.tsv",
        help="Local cache path for SIFTS mapping file",
    )
    parser.add_argument(
        "--pdb-mirror",
        default=None,
        help="Root path of local PDB mirror (e.g. /scicore/data/managed/PDB/latest). "
             "If given, adds local mmCIF file paths to the output.",
    )
    parser.add_argument(
        "--min-subunits",
        type=int,
        default=2,
        help="Minimum number of protein subunits to consider a complex (default: 2)",
    )
    args = parser.parse_args()

    # 1. Load complexes
    fmt = None if args.format == "auto" else args.format
    print(f"Loading complexes from {args.complexes} ...")
    complexes_df = load_complexes(args.complexes, fmt=fmt)

    # Filter by minimum subunits
    before = len(complexes_df)
    complexes_df = complexes_df[complexes_df["n_protein_subunits"] >= args.min_subunits].reset_index(drop=True)
    print(f"  {before} total, {len(complexes_df)} with >= {args.min_subunits} protein subunits")

    # 2. Download / cache SIFTS mapping
    uniprot_to_pdb = download_sifts(args.sifts_cache)

    # 3. Match complexes to PDB
    print("Matching complexes to PDB entries ...")
    complexes_df, all_complete_pdb_ids = match_complexes_to_pdb(complexes_df, uniprot_to_pdb)
    n_complete = complexes_df["has_complete_match"].sum()
    print(f"  {n_complete} / {len(complexes_df)} complexes have at least one complete PDB match")
    print(f"  {len(all_complete_pdb_ids)} unique PDB IDs across all complete matches")

    if not all_complete_pdb_ids:
        print("No complete PDB matches found. Exiting.")
        complexes_df.to_csv(args.output, index=False)
        return

    # 4. Get PDB entry metadata (resolution, method)
    pdb_info = batch_pdb_info(sorted(all_complete_pdb_ids), workers=args.workers)

    # 5. Get chain-level stoichiometry from SIFTS REST API
    sifts_chains = batch_sifts_chains(sorted(all_complete_pdb_ids), workers=args.workers)

    # 6. Annotate best PDB per complex
    print("Annotating best PDB per complex ...")
    complexes_df = annotate_best_pdb(complexes_df, pdb_info, sifts_chains, pdb_mirror=args.pdb_mirror)

    # 7. Save output
    complexes_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"  {len(complexes_df)} rows, {len(complexes_df.columns)} columns")

    # 8. Print summary
    complete = complexes_df[complexes_df["has_complete_match"]]
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total complexes (>= {args.min_subunits} protein subunits): {len(complexes_df)}")
    print(f"With complete PDB match:            {n_complete}")
    print(f"Without complete PDB match:         {(~complexes_df['has_complete_match']).sum()}")
    print(f"\nComplete matches by complex size:")
    for size in sorted(complete["n_protein_subunits"].unique()):
        n = (complete["n_protein_subunits"] == size).sum()
        print(f"  {size}-subunit complexes: {n}")
    print(f"\nExperimental methods (best PDB):")
    print(complete["best_pdb_method"].value_counts().to_string())
    print(f"\nResolution stats (best PDB):")
    res = complete["best_pdb_resolution"].dropna()
    if len(res) > 0:
        print(f"  Median: {res.median():.2f} A, Mean: {res.mean():.2f} A")
        print(f"  Range:  {res.min():.2f} - {res.max():.2f} A")
    print(f"\nTop 10 highest-resolution complete matches:")
    top_cols = ["complex_id", "n_protein_subunits", "best_pdb_id", "best_pdb_resolution", "best_pdb_method"]
    top = complete.nsmallest(10, "best_pdb_resolution")[top_cols]
    print(top.to_string(index=False))

    # Complex Portal xref comparison (if available)
    if "complex_portal_pdb_xrefs" in complete.columns:
        has_xref = complete["complex_portal_pdb_xrefs"].apply(lambda x: bool(x and x.strip()))
        if has_xref.any():
            print(f"\nComplex Portal PDB cross-references:")
            print(f"  Complexes with CP xrefs: {has_xref.sum()}")
            # Check how many of our SIFTS-found PDBs overlap with CP xrefs
            overlap_count = 0
            novel_count = 0
            for _, row in complete[has_xref].iterrows():
                cp_pdbs = set(row["complex_portal_pdb_xrefs"].split(";"))
                found_pdbs = set(row["all_complete_pdb_ids"].split(";"))
                if cp_pdbs & found_pdbs:
                    overlap_count += 1
                if found_pdbs - cp_pdbs:
                    novel_count += 1
            print(f"  SIFTS matches overlap with CP xrefs: {overlap_count}")
            print(f"  SIFTS found additional PDBs beyond CP: {novel_count}")


if __name__ == "__main__":
    main()
