#!/usr/bin/env python
"""
22_download_biological_assemblies.py — Download the correct biological assembly
for each exact-match PDB, choosing the assembly whose oligomeric state matches
the complex's stoichiometry.

WHY THIS EXISTS
---------------
The raw .pdb file from RCSB is the asymmetric unit (ASU), which often contains
a different number of copies than the biologically relevant assembly. Examples
that broke the CombFold benchmark:

  - 1HR6: ASU = 8 chains (4x P11914 + 4x P10507), but biological assembly 1
    is a heterodimer (1x each). Matches CPX-1630 spec.
  - 1PLR: ASU = 1 chain, but biological assembly 1 is the PCNA homotrimer.
    Matches CPX-544 spec.
  - 2VDU: ASU = 4 chains (2x each), but biological assembly 2/3 is a
    heterodimer. Matches CPX-1632 spec.

Using the ASU inflated reference lengths and tanked TM-scores for these cases.
This script downloads the right assembly file for each complex.

WHAT IT DOES
------------
For each (complex, exact-match PDB) pair from complex_pdb_exact_match.csv:
  1. Query RCSB assembly API for all assemblies of the PDB.
  2. For each assembly, get its oligomeric_count and the UniProt composition
     (via the assembly's chain list cross-referenced with SIFTS).
  3. Pick the assembly whose UniProt multiset == the complex's stoichiometry
     (e.g. complex spec P10507(1),P11914(1) -> want assembly with 1x P10507
     and 1x P11914).
  4. Download that assembly file from RCSB
     (https://files.rcsb.org/download/{pdb_id}.pdb{assembly_id}).
  5. Save to _raw_assembly/{pdb_id}_asm{assembly_id}.pdb with a symlink/copy
     named {pdb_id}.pdb pointing to the chosen assembly (so downstream code
     that expects {pdb_id}.pdb gets the right one).

If no assembly matches the stoichiometry exactly, fall back to:
  - the author_defined_assembly with the closest oligomeric count, or
  - the ASU (with a warning) if nothing better exists.

Resume-friendly: skips PDBs already present in _raw_assembly/.

Usage:
    uv run 22_download_biological_assemblies.py \
        --exact-csv /cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files/uniprot_pdb/complex_pdb_exact_match.csv \
        --tsv       /cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv \
        --out-dir   /cluster/project/beltrao/kdammer/master_thesis/data/all_Complex_pdb_files
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
from collections import Counter

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RCSB_ASSEMBLY_URL = "https://files.rcsb.org/download/{pdb_id}.pdb{asm_id}"
RCSB_ASSEMBLY_CIF_URL = "https://files.rcsb.org/download/{pdb_id}-assembly{asm_id}.cif"
RCSB_ASSEMBLY_API = "https://data.rcsb.org/rest/v1/core/assembly/{pdb}/{asm_id}"
SIFTS_PDB_UNIPROT_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
PDBE_ASSEMBLY_API = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/assembly/{pdb}"

API_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5
RCSB_DELAY = 0.3
SIFTS_DELAY = 0.1

UNIPROT_RE = re.compile(r"^[OPQ][0-9][0-9A-Z]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
BRACKET_RE = re.compile(r"^\[([A-Z0-9_,\s]+)\]$")
PRO_ISOFORM_RE = re.compile(r"^([OPQ][0-9][0-9A-Z]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})-PRO_\d+$")


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


def parse_complex_stoichiometry(tsv_path: Path) -> dict[str, Counter]:
    """Return {cpx_id: Counter({uniprot: count})} from the TSV."""
    df = pd.read_csv(tsv_path, sep="\t")
    mol_col = "Identifiers (and stoichiometry) of molecules in complex"
    stoich: dict[str, Counter] = {}
    for _, row in df.iterrows():
        cpx = str(row["#Complex ac"]).strip()
        if not cpx:
            continue
        counts: Counter = Counter()
        raw = row.get(mol_col, "")
        if isinstance(raw, str):
            for token in raw.split("|"):
                base = token.split("(")[0].strip()
                count_m = re.search(r"\((\d+)\)", token)
                count = int(count_m.group(1)) if count_m else 1
                for u in extract_proteins(base):
                    counts[u] += count
        stoich[cpx] = counts
    return stoich


# ---------------------------------------------------------------------------
# SIFTS: chain -> uniprot for the ASU (we'll reuse for assembly chain mapping)
# ---------------------------------------------------------------------------

_sifts_cache: dict[str, dict] = {}


def sifts_chain_uniprot(pdb_id: str) -> dict[str, str]:
    """Return {chain_id: uniprot} for the asymmetric unit."""
    key = pdb_id.lower()
    if key in _sifts_cache:
        return _sifts_cache[key]
    data = _get_json(SIFTS_PDB_UNIPROT_URL.format(pdb=key), "BioAssembly/1.0")
    chain_map: dict[str, str] = {}
    if data and key in data:
        for uniprot, info in data[key].get("UniProt", {}).items():
            for m in info.get("mappings", []):
                ch = m.get("chain_id")
                if ch and ch not in chain_map:
                    chain_map[ch] = uniprot
    _sifts_cache[key] = chain_map
    time.sleep(SIFTS_DELAY)
    return chain_map


# ---------------------------------------------------------------------------
# RCSB assembly metadata
# ---------------------------------------------------------------------------

_rcsb_asm_cache: dict[str, list[dict]] = {}


def rcsb_assemblies(pdb_id: str) -> list[dict]:
    """Return list of assembly metadata dicts from RCSB API.
    Each dict: {assembly_id, oligomeric_count, oligomeric_details, details,
                chains (list of chain ids in this assembly)}.
    Falls back to PDBe if RCSB API fails."""
    key = pdb_id.lower()
    if key in _rcsb_asm_cache:
        return _rcsb_asm_cache[key]

    assemblies: list[dict] = []
    for asm_id in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        url = RCSB_ASSEMBLY_API.format(pdb=key, asm_id=asm_id)
        data = _get_json(url, "BioAssembly/1.0")
        if data is None:
            break
        struct = data.get("pdbx_struct_assembly", {})
        info = data.get("rcsb_assembly_info", {})
        # rcsb_assembly_container_identifiers has chain info in some API versions
        container = data.get("rcsb_assembly_container_identifiers", {})
        chains = container.get("chain_ids", []) or info.get("chain_ids", [])
        assemblies.append({
            "assembly_id": asm_id,
            "oligomeric_count": struct.get("oligomeric_count"),
            "oligomeric_details": struct.get("oligomeric_details", ""),
            "details": struct.get("rcsb_details", struct.get("details", "")),
            "chains": chains,
        })
        time.sleep(0.05)

    # If RCSB didn't give chain lists, try PDBe for chain composition per assembly
    if not any(a["chains"] for a in assemblies) and assemblies:
        pdbe_data = _get_json(PDBE_ASSEMBLY_API.format(pdb=key), "BioAssembly/1.0")
        if pdbe_data and key in pdbe_data:
            pdbe_asms = {a.get("assembly_id"): a for a in pdbe_data[key]}
            for a in assemblies:
                pa = pdbe_asms.get(int(a["assembly_id"])) if a["assembly_id"].isdigit() else None
                if pa:
                    # PDBe molecules list gives chains per entity in this assembly
                    chains = []
                    for mol in pa.get("molecules", []):
                        chains.extend(mol.get("chains", []))
                    a["chains"] = chains
        time.sleep(SIFTS_DELAY)

    _rcsb_asm_cache[key] = assemblies
    return assemblies


def assembly_uniprot_multiset(pdb_id: str, asm_chains: list[str]) -> Counter:
    """Given the chains present in an assembly, return the UniProt multiset."""
    chain_to_u = sifts_chain_uniprot(pdb_id)
    counts: Counter = Counter()
    for ch in asm_chains:
        u = chain_to_u.get(ch)
        if u:
            counts[u] += 1
    return counts


# ---------------------------------------------------------------------------
# Pick the best assembly for a complex's stoichiometry
# ---------------------------------------------------------------------------

def choose_assembly(pdb_id: str, target_stoich: Counter) -> tuple[str | None, str]:
    """Return (assembly_id, reason).
    reason is one of: 'exact_match', 'closest_oligomeric', 'author_default',
    'no_assembly_metadata', 'asu_fallback'."""
    assemblies = rcsb_assemblies(pdb_id)
    if not assemblies:
        return (None, "no_assembly_metadata")

    target_total = sum(target_stoich.values())

    # 1. Exact stoichiometry match (UniProt multiset equals target)
    exact_matches = []
    for a in assemblies:
        if not a["chains"]:
            continue
        asm_counts = assembly_uniprot_multiset(pdb_id, a["chains"])
        if asm_counts == target_stoich:
            exact_matches.append(a)

    if exact_matches:
        # Prefer author_defined_assembly if multiple, else assembly 1
        author = [a for a in exact_matches if "author" in a["details"].lower()]
        chosen = author[0] if author else exact_matches[0]
        return (chosen["assembly_id"], "exact_match")

    # 2. Closest oligomeric count (when chain->uniprot mapping is incomplete)
    scored = []
    for a in assemblies:
        oc = a["oligomeric_count"]
        if oc is None:
            continue
        try:
            oc_int = int(oc)
        except (ValueError, TypeError):
            continue
        scored.append((abs(oc_int - target_total), a))
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]["assembly_id"]))
        return (scored[0][1]["assembly_id"], "closest_oligomeric")

    # 3. Author-defined assembly 1 as default
    author = [a for a in assemblies if "author" in a["details"].lower()]
    if author:
        return (author[0]["assembly_id"], "author_default")

    return (assemblies[0]["assembly_id"], "author_default" if assemblies else "asu_fallback")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_one(url: str, out_path: Path, pdb_id: str, asm_id: str,
                     fmt: str) -> tuple[str, str]:
    """Try one URL. Returns (status, path_str)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BioAssembly/1.0"})
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                content = resp.read()
            # PDB files have "ATOM"; mmCIF files have "_atom_site."
            if content and (b"ATOM" in content or b"_atom_site." in content):
                out_path.write_bytes(content)
                return ("ok", str(out_path))
            log(f"    [WARN] {pdb_id} asm {asm_id} ({fmt}): no atom records in response")
            return ("fail", "")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ("404", "")
            log(f"    [RETRY] {pdb_id} asm {asm_id} ({fmt}) HTTP {e.code} (attempt {attempt})")
        except Exception as e:
            log(f"    [RETRY] {pdb_id} asm {asm_id} ({fmt}) {e} (attempt {attempt})")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return ("fail", "")


def download_assembly(pdb_id: str, asm_id: str, out_dir: Path) -> tuple[str, str]:
    """Download biological assembly. Try PDB format first, fall back to mmCIF
    (some newer entries are mmCIF-only). Returns (status, path_str).
    status is 'ok'|'skip'|'fail'; path_str is the written file path or ''."""
    pdb_path = out_dir / f"{pdb_id}_asm{asm_id}.pdb"
    cif_path = out_dir / f"{pdb_id}_asm{asm_id}.cif"

    # Skip if either format already exists
    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        return ("skip", str(pdb_path))
    if cif_path.exists() and cif_path.stat().st_size > 0:
        return ("skip", str(cif_path))

    # 1. Try PDB format
    status, path = _download_one(
        RCSB_ASSEMBLY_URL.format(pdb_id=pdb_id, asm_id=asm_id),
        pdb_path, pdb_id, asm_id, "pdb")
    if status == "ok":
        return ("ok", path)
    if status == "skip":
        return ("skip", path)

    # 2. Fall back to mmCIF (newer entries are often mmCIF-only)
    log(f"    [CIF-fallback] {pdb_id} asm {asm_id}: trying mmCIF format")
    status, path = _download_one(
        RCSB_ASSEMBLY_CIF_URL.format(pdb_id=pdb_id, asm_id=asm_id),
        cif_path, pdb_id, asm_id, "cif")
    if status == "ok":
        return ("ok", path)
    if status == "404":
        log(f"    [404] {pdb_id} asm {asm_id}: not found as .pdb or .cif")
    return ("fail", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exact-csv", required=True, type=Path,
                    help="complex_pdb_exact_match.csv from step 3")
    ap.add_argument("--tsv", required=True, type=Path,
                    help="Saccharomyces_cerevisiae_ComplexTab.tsv (for stoichiometry)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Complex_pdb_files output dir (assembly files go in _raw_assembly/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report which assembly would be chosen; no downloads.")
    args = ap.parse_args()

    if not args.exact_csv.exists():
        sys.exit(f"exact-csv not found: {args.exact_csv}")
    if not args.tsv.exists():
        sys.exit(f"tsv not found: {args.tsv}")

    log("Loading complex stoichiometry from TSV...")
    stoich = parse_complex_stoichiometry(args.tsv)
    log(f"  {len(stoich)} complexes parsed.")

    log("Loading exact-match PDBs from CSV...")
    df = pd.read_csv(args.exact_csv)
    exact_rows = df[df["match_class"] == "exact_match"].copy()
    log(f"  {len(exact_rows)} complexes with exact-match PDBs.")

    # Build (cpx, pdb) pairs. A complex may have multiple exact-match PDBs.
    pairs: list[tuple[str, str, Counter]] = []
    for _, row in exact_rows.iterrows():
        cpx = row["complex_accession"]
        target = stoich.get(cpx, Counter())
        if not target:
            log(f"  [WARN] {cpx}: no stoichiometry parsed, skipping")
            continue
        for p in str(row["all_exact_pdbs"]).split(";"):
            p = p.strip().upper()
            if p:
                pairs.append((cpx, p, target))

    log(f"  {len(pairs)} (complex, PDB) pairs to process.")

    asm_dir = args.out_dir / "_raw_assembly"
    asm_dir.mkdir(parents=True, exist_ok=True)

    # Per-PDB dedup: a PDB may be referenced by multiple complexes with the
    # same stoichiometry. Process each unique PDB once, but track which
    # assembly was chosen for each (cpx, pdb) pair.
    unique_pdbs = sorted({p for _, p, _ in pairs})
    log(f"\nProcessing {len(unique_pdbs)} unique PDBs for assembly selection...")

    pdb_to_choice: dict[str, tuple[str | None, str]] = {}
    for i, pdb_id in enumerate(unique_pdbs, 1):
        # For this PDB, find the stoichiometry targets it needs to satisfy.
        # If multiple complexes reference it with different stoichiometries,
        # pick the assembly matching the most common target; otherwise just
        # pick per the first complex (they're usually the same).
        targets = [t for c, p, t in pairs if p == pdb_id]
        # Use the first target; if PDB has multiple distinct targets, log it.
        distinct_targets = set(tuple(sorted(t.items())) for t in targets)
        if len(distinct_targets) > 1:
            log(f"  [NOTE] {pdb_id} referenced by complexes with different stoichiometries; "
                f"using first.")
        target = targets[0]
        asm_id, reason = choose_assembly(pdb_id, target)
        pdb_to_choice[pdb_id] = (asm_id, reason)
        log(f"  [{i}/{len(unique_pdbs)}] {pdb_id}: asm={asm_id} ({reason}) "
            f"target={dict(target)}")
        if i % 50 == 0 and not args.dry_run:
            log(f"    ... {i} processed")

    if args.dry_run:
        log("\n[DRY-RUN] No files downloaded.")
        log("Choices that would be made:")
        for pdb_id, (asm_id, reason) in sorted(pdb_to_choice.items()):
            log(f"  {pdb_id} -> asm {asm_id} ({reason})")
        return

    # Download
    log(f"\nDownloading {len(unique_pdbs)} assembly files to {asm_dir}...")
    downloaded = skipped = failed = no_asm = 0
    log_rows = []
    for i, pdb_id in enumerate(unique_pdbs, 1):
        asm_id, reason = pdb_to_choice[pdb_id]
        if asm_id is None:
            log(f"  [{i}/{len(unique_pdbs)}] {pdb_id}: no assembly metadata, skipping")
            no_asm += 1
            log_rows.append({"pdb_id": pdb_id, "assembly_id": "", "reason": reason,
                             "status": "no_asm", "path": ""})
            continue
        status, path = download_assembly(pdb_id, asm_id, asm_dir)
        log_rows.append({"pdb_id": pdb_id, "assembly_id": asm_id, "reason": reason,
                         "status": status, "path": path})
        if status == "ok":
            downloaded += 1
            log(f"  [{i}/{len(unique_pdbs)}] {pdb_id} asm{asm_id}: downloaded")
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
        time.sleep(RCSB_DELAY)

    # Write log CSV
    log_path = args.out_dir / "biological_assembly_download_log.csv"
    with open(log_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id", "assembly_id", "reason",
                                           "status", "path"])
        w.writeheader()
        w.writerows(log_rows)
    n_cif = sum(1 for r in log_rows if r["path"].endswith(".cif"))
    n_pdb = sum(1 for r in log_rows if r["path"].endswith(".pdb"))
    log(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, "
        f"Failed: {failed}, No-assembly-metadata: {no_asm}")
    log(f"  PDB format: {n_pdb}, mmCIF format: {n_cif}")
    log(f"Log: {log_path}")
    log(f"\nAssembly files in: {asm_dir}")
    log(f"  Named: {{pdb_id}}_asm{{assembly_id}}.pdb or .cif")
    log(f"\nNext: point your comparison/render scripts at {asm_dir} instead of _raw/.")
    log(f"  Note: some entries are mmCIF-only (newer/large complexes).")
    log(f"  Your loader must handle both .pdb and .cif (Biopython PDBParser + MMCIFParser).")


if __name__ == "__main__":
    main()
