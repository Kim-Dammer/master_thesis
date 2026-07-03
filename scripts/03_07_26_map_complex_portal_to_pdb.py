#!/usr/bin/env python
"""
map_complex_portal_to_pdb.py — End-to-end pipeline that maps Complex Portal
yeast complexes to PDB structures and downloads the correct structure file
(asymmetric unit or biological assembly) whose stoichiometry matches each
complex.

Takes only the Complex Portal TSV as input and produces:
  - uniprot_pdb_mapping.csv         (UniProt -> PDB-chain, long format)
  - complex_pdb_exact_match.csv     (complex -> exact-match PDBs, with has_paralogue)
  - complex_pdb_mapping.csv         (Complex Portal cross-ref PDBs: cpx, pdb_id, tag)
  - structure_download_log.csv      (download log: pdb_id, assembly_id, source, reason, ...)
  - _raw_assembly/                  (downloaded structure files: ASU or bio assembly)
  - identity/ subset/ experimental_evidence/  (per-complex copies of cross-ref PDBs)
  - sifts_cache.json                (resume cache for SIFTS API responses)

Key features vs the original 21_1 + 21_2 scripts:
  1. Paralogue-flexible matching: a PDB matches a complex if it contains
     exactly one paralog from each bracketed paralog group (not both, not
     zero), plus all fixed (non-paralog) proteins, and no extra proteins.
  2. Stoichiometry-aware structure selection: the ASU and biological
     assemblies 1-10 are all candidates; the one whose UniProt multiset
     exactly equals the complex's (adapted) stoichiometry is chosen.
  3. ASU as a candidate: when no biological assembly matches the
     stoichiometry but the ASU does, the ASU is downloaded.
  4. has_paralogue column in complex_pdb_exact_match.csv.

Resume-friendly: sifts_cache.json and file-existence checks skip already-done
work. To force a full re-run, delete the --out-dir first.

Usage:
    uv run 03_07_26_map_complex_portal_to_pdb.py \
        --tsv    /cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv \
        --out-dir /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping

    # audit-only (no downloads, no API calls):
    python map_complex_portal_to_pdb.py --tsv ... --out-dir ... --dry-run
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
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_ASSEMBLY_URL = "https://files.rcsb.org/download/{pdb_id}.pdb{asm_id}"
RCSB_ASSEMBLY_CIF_URL = "https://files.rcsb.org/download/{pdb_id}-assembly{asm_id}.cif"
RCSB_ASSEMBLY_API = "https://data.rcsb.org/rest/v1/core/assembly/{pdb}/{asm_id}"
SIFTS_BEST_STRUCTURES_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{u}"
SIFTS_PDB_UNIPROT_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
PDBe_ASSEMBLY_API = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/assembly/{pdb}"

RCSB_DELAY = 0.3
SIFTS_DELAY = 0.1
RCSB_ASM_API_DELAY = 0.05
API_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5

UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][0-9A-Z]{3}[0-9]$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)
BRACKET_RE = re.compile(r"^\[([A-Z0-9_,\s]+)\]$")
PRO_ISOFORM_RE = re.compile(
    r"^([OPQ][0-9][0-9A-Z]{3}[0-9]"
    r"|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})-PRO_\d+$"
)
WWPDB_TAGGED_RE = re.compile(r"^wwpdb:([A-Za-z0-9]{4})\((identity|subset)\)$")
WWPDB_BARE_RE = re.compile(r"\bwwpdb:([A-Za-z0-9]{4})\b")
WWPDB_ANY_RE = re.compile(r"^wwpdb:[^|]+$")

TAGS = ("identity", "subset", "experimental_evidence")
ASU_ID = "0"  # pseudo assembly id for the asymmetric unit


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Protein / paralog extraction
# ---------------------------------------------------------------------------

def _extract_proteins_from_token(member: str) -> set[str]:
    """Extract UniProt accessions from a single molecule token.

    Handles plain UniProts, [bracketed paralog groups], and PRO_ isoforms.
    Returns empty set for CHEBI/URS/CPX-/EBI- and anything non-protein.
    """
    member = member.strip()
    if UNIPROT_RE.match(member):
        return {member}
    m = BRACKET_RE.match(member)
    if m:
        return {
            s.strip()
            for s in m.group(1).split(",")
            if UNIPROT_RE.match(s.strip())
        }
    m = PRO_ISOFORM_RE.match(member)
    if m:
        return {m.group(1)}
    return set()


def _is_bracketed(token: str) -> bool:
    """True if the token is a bracketed paralog group like [P0CX46,P0CX45]."""
    return token.strip().startswith("[")


# ---------------------------------------------------------------------------
# TSV parsing — produces per-complex: fixed_proteins, paralog_groups,
# stoichiometry, has_paralogue, xrefs, name
# ---------------------------------------------------------------------------

def parse_tsv(
    tsv_path: Path,
) -> tuple[
    dict[str, dict],        # cpx -> {name, fixed_proteins, paralog_groups,
                            #        stoichiometry, has_paralogue, xrefs}
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
        if not cpx or cpx == "nan":
            continue
        name = str(row.get(name_col, "")).strip()

        fixed_proteins: set[str] = set()
        paralog_groups: list[set[str]] = []
        stoichiometry: Counter = Counter()

        raw = row.get(mol_col, "")
        if isinstance(raw, str):
            for token in raw.split("|"):
                base = token.split("(")[0].strip()
                count_m = re.search(r"\((\d+)\)", token)
                count = int(count_m.group(1)) if count_m else 1

                if _is_bracketed(base):
                    prots = _extract_proteins_from_token(base)
                    if prots:
                        paralog_groups.append(prots)
                        for u in prots:
                            stoichiometry[u] += count
                else:
                    prots = _extract_proteins_from_token(base)
                    fixed_proteins |= prots
                    for u in prots:
                        stoichiometry[u] += count

        has_paralogue = len(paralog_groups) > 0

        # Complex Portal PDB xrefs (identity/subset from Cross references)
        xrefs: dict[str, list[str]] = {
            "identity": [],
            "subset": [],
            "experimental_evidence": [],
        }
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
                if (
                    pdb_id not in tagged
                    and pdb_id not in xrefs["experimental_evidence"]
                ):
                    xrefs["experimental_evidence"].append(pdb_id)

        complexes[cpx] = {
            "name": name,
            "fixed_proteins": fixed_proteins,
            "paralog_groups": paralog_groups,
            "stoichiometry": stoichiometry,
            "has_paralogue": has_paralogue,
            "xrefs": xrefs,
        }

        all_proteins = fixed_proteins | {
            u for grp in paralog_groups for u in grp
        }
        for u in all_proteins:
            uniprot_to_complexes.setdefault(u, [])
            if cpx not in uniprot_to_complexes[u]:
                uniprot_to_complexes[u].append(cpx)

    return complexes, uniprot_to_complexes


# ---------------------------------------------------------------------------
# Paralogue-flexible matching
# ---------------------------------------------------------------------------

def complex_matches_pdb(
    fixed_proteins: set[str],
    paralog_groups: list[set[str]],
    pdb_uniprots: set[str],
) -> bool:
    """Check whether a PDB's UniProt set matches a complex spec.

    Match conditions:
      1. PDB contains every fixed protein.
      2. PDB contains exactly one protein from each paralog group.
      3. PDB contains no proteins outside fixed_proteins ∪ paralog members.
    """
    # Condition 1: all fixed proteins present
    if not fixed_proteins <= pdb_uniprots:
        return False

    # Collect all paralog-group members
    all_paralog_members: set[str] = set()
    for grp in paralog_groups:
        all_paralog_members |= grp

    # Condition 3: no extra proteins
    allowed = fixed_proteins | all_paralog_members
    if not pdb_uniprots <= allowed:
        return False

    # Condition 2: exactly one from each paralog group
    for grp in paralog_groups:
        present = pdb_uniprots & grp
        if len(present) != 1:
            return False

    return True


def adapt_stoichiometry(
    stoichiometry: Counter,
    paralog_groups: list[set[str]],
    pdb_uniprots: set[str],
) -> Counter:
    """Build the target stoichiometry using only the paralogs present in the PDB.

    For each paralog group, keep only the one paralog that is in the PDB.
    Fixed proteins keep their original stoichiometry.
    """
    adapted = Counter()
    # Fixed proteins
    for u, c in stoichiometry.items():
        if any(u in grp for grp in paralog_groups):
            continue  # handle paralogs below
        adapted[u] = c

    # Paralogs: keep only the one present in the PDB
    for grp in paralog_groups:
        present = pdb_uniprots & grp
        if len(present) == 1:
            u = next(iter(present))
            adapted[u] = stoichiometry.get(u, 1)
        # If 0 or >1 present, this shouldn't happen (matching already filtered),
        # but we leave the adapted stoich without this group.

    return adapted


# ---------------------------------------------------------------------------
# SIFTS cache (file-based, for resume)
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
        data = _get_json(
            SIFTS_BEST_STRUCTURES_URL.format(u=uniprot), "ComplexPDBMapper/1.0"
        )
        cache["best_structures"][uniprot] = (
            data.get(uniprot, []) if data else []
        )
        time.sleep(SIFTS_DELAY)
    return cache["best_structures"][uniprot]


def query_pdb_uniprots(pdb_id: str, cache: dict) -> set[str]:
    key = pdb_id.lower()
    if key not in cache["pdb_uniprots"]:
        data = _get_json(
            SIFTS_PDB_UNIPROT_URL.format(pdb=key), "ComplexPDBMapper/1.0"
        )
        uniprots = (
            set(data[key].get("UniProt", {}).keys())
            if data and key in data
            else set()
        )
        cache["pdb_uniprots"][key] = sorted(uniprots)
        time.sleep(SIFTS_DELAY)
    return set(cache["pdb_uniprots"][key])


# ---------------------------------------------------------------------------
# SIFTS chain -> UniProt for ASU (in-memory cache)
# ---------------------------------------------------------------------------

_sifts_chain_cache: dict[str, dict[str, str]] = {}


def sifts_chain_uniprot(pdb_id: str) -> dict[str, str]:
    """Return {chain_id: uniprot} for the asymmetric unit."""
    key = pdb_id.lower()
    if key in _sifts_chain_cache:
        return _sifts_chain_cache[key]
    data = _get_json(
        SIFTS_PDB_UNIPROT_URL.format(pdb=key), "ComplexPDBMapper/1.0"
    )
    chain_map: dict[str, str] = {}
    if data and key in data:
        for uniprot, info in data[key].get("UniProt", {}).items():
            for m in info.get("mappings", []):
                ch = m.get("chain_id")
                if ch and ch not in chain_map:
                    chain_map[ch] = uniprot
    _sifts_chain_cache[key] = chain_map
    time.sleep(SIFTS_DELAY)
    return chain_map


# ---------------------------------------------------------------------------
# RCSB / PDBe assembly metadata (in-memory cache)
# ---------------------------------------------------------------------------

_rcsb_asm_cache: dict[str, list[dict]] = {}


def rcsb_assemblies(pdb_id: str) -> list[dict]:
    """Return list of assembly metadata dicts from RCSB API (with PDBe fallback
    for chain lists).

    Each dict: {assembly_id, oligomeric_count, oligomeric_details, details,
                chains (list of chain ids in this assembly)}.
    """
    key = pdb_id.lower()
    if key in _rcsb_asm_cache:
        return _rcsb_asm_cache[key]

    assemblies: list[dict] = []
    for asm_id in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        url = RCSB_ASSEMBLY_API.format(pdb=key, asm_id=asm_id)
        data = _get_json(url, "ComplexPDBMapper/1.0")
        if data is None:
            break
        struct = data.get("pdbx_struct_assembly", {})
        info = data.get("rcsb_assembly_info", {})
        container = data.get("rcsb_assembly_container_identifiers", {})
        chains = (
            container.get("chain_ids", []) or info.get("chain_ids", [])
        )
        assemblies.append(
            {
                "assembly_id": asm_id,
                "oligomeric_count": struct.get("oligomeric_count"),
                "oligomeric_details": struct.get("oligomeric_details", ""),
                "details": struct.get(
                    "rcsb_details", struct.get("details", "")
                ),
                "chains": chains,
            }
        )
        time.sleep(RCSB_ASM_API_DELAY)

    # If RCSB didn't give chain lists, try PDBe
    if not any(a["chains"] for a in assemblies) and assemblies:
        pdbe_data = _get_json(
            PDBe_ASSEMBLY_API.format(pdb=key), "ComplexPDBMapper/1.0"
        )
        if pdbe_data and key in pdbe_data:
            pdbe_asms = {a.get("assembly_id"): a for a in pdbe_data[key]}
            for a in assemblies:
                pa = (
                    pdbe_asms.get(int(a["assembly_id"]))
                    if a["assembly_id"].isdigit()
                    else None
                )
                if pa:
                    chains = []
                    for mol in pa.get("molecules", []):
                        chains.extend(mol.get("chains", []))
                    a["chains"] = chains
        time.sleep(SIFTS_DELAY)

    _rcsb_asm_cache[key] = assemblies
    return assemblies


def assembly_uniprot_multiset(
    pdb_id: str, asm_chains: list[str]
) -> Counter:
    """Given the chains present in an assembly, return the UniProt multiset."""
    chain_to_u = sifts_chain_uniprot(pdb_id)
    counts: Counter = Counter()
    for ch in asm_chains:
        u = chain_to_u.get(ch)
        if u:
            counts[u] += 1
    return counts


def asu_uniprot_multiset(pdb_id: str) -> Counter:
    """Return the UniProt multiset for the asymmetric unit."""
    chain_to_u = sifts_chain_uniprot(pdb_id)
    counts: Counter = Counter()
    for u in chain_to_u.values():
        counts[u] += 1
    return counts


# ---------------------------------------------------------------------------
# Structure selection: ASU + biological assemblies as candidates
# ---------------------------------------------------------------------------

def choose_structure(
    pdb_id: str, target_stoich: Counter
) -> tuple[str | None, str]:
    """Choose the best structure file for a complex's stoichiometry.

    Candidates:
      - ASU (assembly_id = "0"): stoichiometry from SIFTS chain mapping.
      - Biological assemblies 1-10: stoichiometry from RCSB/PDBe chain lists.

    Returns (assembly_id, reason) where assembly_id is "0" for ASU or "1".."10"
    for a biological assembly, or None if no metadata at all.

    reason is one of:
      'exact_match_asu', 'exact_match_assembly', 'closest_oligomeric',
      'author_default', 'asu_fallback', 'no_assembly_metadata'
    """
    assemblies = rcsb_assemblies(pdb_id)

    target_total = sum(target_stoich.values())

    # --- Tier 1: exact stoichiometry match across all candidates ---
    exact_asu = False
    exact_assemblies: list[dict] = []

    # Check ASU
    asu_counts = asu_uniprot_multiset(pdb_id)
    if asu_counts and asu_counts == target_stoich:
        exact_asu = True

    # Check biological assemblies
    for a in assemblies:
        if not a["chains"]:
            continue
        asm_counts = assembly_uniprot_multiset(pdb_id, a["chains"])
        if asm_counts == target_stoich:
            exact_assemblies.append(a)

    if exact_assemblies:
        # Prefer biological assembly over ASU on ties
        author = [
            a for a in exact_assemblies if "author" in a["details"].lower()
        ]
        chosen = author[0] if author else exact_assemblies[0]
        return (chosen["assembly_id"], "exact_match_assembly")

    if exact_asu:
        return (ASU_ID, "exact_match_asu")

    # --- Tier 2: closest oligomeric count (biological assemblies only) ---
    scored: list[tuple[int, dict]] = []
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

    # --- Tier 3: author-defined assembly 1 as default ---
    author = [a for a in assemblies if "author" in a["details"].lower()]
    if author:
        return (author[0]["assembly_id"], "author_default")

    # --- Tier 4: ASU fallback if no assembly metadata ---
    if assemblies:
        return (assemblies[0]["assembly_id"], "author_default")
    return (ASU_ID, "asu_fallback")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_one(
    url: str, out_path: Path, pdb_id: str, label: str, fmt: str
) -> tuple[str, str]:
    """Try one URL. Returns (status, path_str).
    status: 'ok' | '404' | 'fail'"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ComplexPDBMapper/1.0"}
            )
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                content = resp.read()
            if content and (b"ATOM" in content or b"_atom_site." in content):
                out_path.write_bytes(content)
                return ("ok", str(out_path))
            log(
                f"    [WARN] {pdb_id} {label} ({fmt}): "
                f"no atom records in response"
            )
            return ("fail", "")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ("404", "")
            log(
                f"    [RETRY] {pdb_id} {label} ({fmt}) HTTP {e.code} "
                f"(attempt {attempt})"
            )
        except Exception as e:
            log(f"    [RETRY] {pdb_id} {label} ({fmt}) {e} (attempt {attempt})")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return ("fail", "")


def download_structure(
    pdb_id: str, asm_id: str, out_dir: Path
) -> tuple[str, str]:
    """Download a structure file (ASU or biological assembly).

    For ASU (asm_id == "0"): downloads {pdb_id}.pdb or .cif
    For assembly: downloads {pdb_id}.pdb{asm_id} or {pdb_id}-assembly{asm_id}.cif

    Returns (status, path_str).
    status: 'ok' | 'skip' | 'fail'
    """
    if asm_id == ASU_ID:
        stem = f"{pdb_id}_asu"
        pdb_path = out_dir / f"{stem}.pdb"
        cif_path = out_dir / f"{stem}.cif"
        pdb_url = RCSB_PDB_URL.format(pdb_id=pdb_id)
        cif_url = RCSB_CIF_URL.format(pdb_id=pdb_id)
        label = "ASU"
    else:
        stem = f"{pdb_id}_asm{asm_id}"
        pdb_path = out_dir / f"{stem}.pdb"
        cif_path = out_dir / f"{stem}.cif"
        pdb_url = RCSB_ASSEMBLY_URL.format(pdb_id=pdb_id, asm_id=asm_id)
        cif_url = RCSB_ASSEMBLY_CIF_URL.format(
            pdb_id=pdb_id, asm_id=asm_id
        )
        label = f"asm{asm_id}"

    # Skip if already downloaded
    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        return ("skip", str(pdb_path))
    if cif_path.exists() and cif_path.stat().st_size > 0:
        return ("skip", str(cif_path))

    # Try PDB format
    status, path = _download_one(pdb_url, pdb_path, pdb_id, label, "pdb")
    if status == "ok":
        return ("ok", path)

    # Fall back to mmCIF
    log(f"    [CIF-fallback] {pdb_id} {label}: trying mmCIF format")
    status, path = _download_one(cif_url, cif_path, pdb_id, label, "cif")
    if status == "ok":
        return ("ok", path)
    if status == "404":
        log(f"    [FAIL] {pdb_id} {label}: not found as .pdb or .cif")
    return ("fail", "")


# ---------------------------------------------------------------------------
# STEP 1: Download Complex Portal cross-referenced PDBs
# ---------------------------------------------------------------------------

def step1_download_xref_pdbs(
    complexes: dict[str, dict], out_dir: Path
) -> None:
    log("\n" + "=" * 70)
    log("STEP 1: Download Complex Portal cross-referenced PDBs")
    log("=" * 70)

    raw_dir = out_dir / "_raw_assembly"
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

    # For each PDB, choose the best structure using the first complex's
    # stoichiometry as target.
    pdb_choices: dict[str, tuple[str | None, str]] = {}
    for i, pdb_id in enumerate(sorted(pdb_to_cpxs), 1):
        # Find the first complex that references this PDB (any tag)
        first_cpx = None
        for tag in TAGS:
            if pdb_to_cpxs[pdb_id][tag]:
                first_cpx = pdb_to_cpxs[pdb_id][tag][0]
                break
        if first_cpx is None:
            continue
        target = complexes[first_cpx]["stoichiometry"]
        asm_id, reason = choose_structure(pdb_id, target)
        pdb_choices[pdb_id] = (asm_id, reason)
        log(
            f"  [{i}/{len(pdb_to_cpxs)}] {pdb_id}: "
            f"asm={asm_id} ({reason})"
        )

    # Download
    downloaded = skipped = failed = no_asm = 0
    for i, pdb_id in enumerate(sorted(pdb_to_cpxs), 1):
        asm_id, reason = pdb_choices[pdb_id]
        if asm_id is None:
            log(f"  [{i}/{len(pdb_to_cpxs)}] {pdb_id}: no metadata, skipping")
            no_asm += 1
            continue
        status, path = download_structure(pdb_id, asm_id, raw_dir)
        if status == "ok":
            downloaded += 1
            log(f"  [{i}/{len(pdb_to_cpxs)}] {pdb_id}: downloaded")
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
        time.sleep(RCSB_DELAY)

    # Per-CPX copies
    log("  Organizing per-CPX copies into identity/, subset/, "
        "experimental_evidence/...")
    for pdb_id, tags in pdb_to_cpxs.items():
        asm_id, _ = pdb_choices.get(pdb_id, (None, ""))
        if asm_id is None:
            continue
        if asm_id == ASU_ID:
            stem = f"{pdb_id}_asu"
        else:
            stem = f"{pdb_id}_asm{asm_id}"
        src = None
        for ext in (".pdb", ".cif"):
            candidate = raw_dir / f"{stem}{ext}"
            if candidate.exists():
                src = candidate
                break
        if src is None:
            continue
        for tag in TAGS:
            for cpx in tags[tag]:
                dest = tag_dirs[tag] / f"{cpx}_{src.name}"
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

    log(f"  Done. Downloaded: {downloaded}, Skipped: {skipped}, "
        f"Failed: {failed}, No-metadata: {no_asm}")
    log(f"  Mapping CSV: {mapping_path}")


# ---------------------------------------------------------------------------
# STEP 2: SIFTS best_structures for all proteins -> uniprot_pdb_mapping.csv
# ---------------------------------------------------------------------------

def step2_sifts_uniprot_to_pdb(
    complexes: dict[str, dict],
    uniprot_to_complexes: dict[str, list[str]],
    cache: dict,
    out_dir: Path,
) -> None:
    log("\n" + "=" * 70)
    log("STEP 2: Query SIFTS for all PDBs each protein appears in")
    log("=" * 70)

    all_uniprots = sorted(uniprot_to_complexes.keys())
    log(
        f"  Querying SIFTS best_structures for "
        f"{len(all_uniprots)} unique proteins..."
    )

    rows = []
    for i, u in enumerate(all_uniprots, 1):
        entries = query_best_structures(u, cache)
        if entries:
            cpxs = uniprot_to_complexes.get(u, [])
            cpx_str = ";".join(cpxs)
            name_str = ";".join(complexes[c]["name"] for c in cpxs)
            for e in entries:
                rows.append(
                    {
                        "uniprot_id": u,
                        "pdb_id": str(e.get("pdb_id", "")).upper(),
                        "chain_id": e.get("chain_id", ""),
                        "experimental_method": e.get(
                            "experimental_method", ""
                        ),
                        "resolution": e.get("resolution"),
                        "coverage": e.get("coverage"),
                        "unp_start": e.get("unp_start"),
                        "unp_end": e.get("unp_end"),
                        "tax_id": e.get("tax_id"),
                        "complex_accessions": cpx_str,
                        "complex_names": name_str,
                    }
                )
        if i % 200 == 0:
            log(f"    [{i}/{len(all_uniprots)}] processed")

    df1 = pd.DataFrame(rows)
    uniprot_pdb_dir = out_dir / "uniprot_pdb"
    uniprot_pdb_dir.mkdir(parents=True, exist_ok=True)
    df1_path = uniprot_pdb_dir / "uniprot_pdb_mapping.csv"
    df1.to_csv(df1_path, index=False)
    log(
        f"  Done. {len(df1)} rows, "
        f"{df1['uniprot_id'].nunique() if len(df1) else 0} proteins with >=1 PDB."
    )
    log(f"  CSV: {df1_path}")


# ---------------------------------------------------------------------------
# STEP 3: Exact-match PDBs with paralogue flexibility
# ---------------------------------------------------------------------------

def step3_exact_matches(
    complexes: dict[str, dict], cache: dict, out_dir: Path
) -> Path:
    log("\n" + "=" * 70)
    log("STEP 3: Find PDBs with exact protein-set match for each complex")
    log("=" * 70)

    # Collect all PDBs seen in best_structures
    all_pdbs = {
        str(e.get("pdb_id", "")).upper()
        for entries in cache["best_structures"].values()
        for e in entries
        if e.get("pdb_id")
    }
    log(
        f"  Querying SIFTS /mappings/uniprot for "
        f"{len(all_pdbs)} unique PDBs..."
    )

    pdb_proteins: dict[str, set[str]] = {}
    for i, pdb in enumerate(sorted(all_pdbs), 1):
        pdb_proteins[pdb] = query_pdb_uniprots(pdb, cache)
        if i % 200 == 0:
            log(f"    [{i}/{len(all_pdbs)}] PDBs processed")

    uniprots_with_pdb = {
        u for u, entries in cache["best_structures"].items() if entries
    }

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
        info = complexes[cpx]
        fixed = info["fixed_proteins"]
        groups = info["paralog_groups"]
        all_proteins = fixed | {u for grp in groups for u in grp}
        n_complex = len(all_proteins)
        n_with_pdb = len(all_proteins & uniprots_with_pdb)

        # Paralogue-flexible matching
        exact_pdbs = sorted(
            [
                pdb
                for pdb, prots in pdb_proteins.items()
                if complex_matches_pdb(fixed, groups, prots)
            ]
        )

        if exact_pdbs:
            best = min(exact_pdbs, key=res_key)
            best_res = pdb_best_res.get(best)
            match_class = "exact_match"
        else:
            best, best_res = "", None
            match_class = "no_match"

        rows.append(
            {
                "complex_accession": cpx,
                "complex_name": info["name"],
                "n_complex_proteins": n_complex,
                "n_exact_match_pdbs": len(exact_pdbs),
                "all_exact_pdbs": ";".join(exact_pdbs),
                "best_pdb_id": best,
                "best_pdb_resolution": (
                    best_res if best_res is not None else ""
                ),
                "match_class": match_class,
                "complex_proteins": ";".join(sorted(all_proteins)),
                "n_proteins_with_pdb": n_with_pdb,
                "has_paralogue": info["has_paralogue"],
            }
        )

    df2 = pd.DataFrame(rows)
    uniprot_pdb_dir = out_dir / "uniprot_pdb"
    df2_path = uniprot_pdb_dir / "complex_pdb_exact_match.csv"
    df2.to_csv(df2_path, index=False)
    n_exact = (df2["match_class"] == "exact_match").sum()
    n_para = df2["has_paralogue"].sum()
    log(
        f"  Done. {len(df2)} complexes; {n_exact} with >=1 exact-match PDB; "
        f"{n_para} complexes have paralog groups."
    )
    log(f"  CSV: {df2_path}")
    return df2_path


# ---------------------------------------------------------------------------
# STEP 4: Download exact-match PDBs (ASU or bio assembly)
# ---------------------------------------------------------------------------

def step4_download_exact_match_pdbs(
    complexes: dict[str, dict],
    exact_csv: Path,
    out_dir: Path,
) -> None:
    log("\n" + "=" * 70)
    log("STEP 4: Download all exact-match PDBs")
    log("=" * 70)

    df = pd.read_csv(exact_csv)
    exact_rows = df[df["match_class"] == "exact_match"].copy()

    # Build (cpx, pdb) pairs with adapted stoichiometry
    pairs: list[tuple[str, str, Counter]] = []
    for _, row in exact_rows.iterrows():
        cpx = row["complex_accession"]
        info = complexes.get(cpx)
        if info is None:
            continue
        for p in str(row["all_exact_pdbs"]).split(";"):
            p = p.strip().upper()
            if not p:
                continue
            # We need the PDB's UniProt set to adapt stoichiometry.
            # We'll query it inside the loop below; for now store the
            # full stoichiometry and adapt later.
            pairs.append((cpx, p, info))

    log(f"  {len(pairs)} (complex, PDB) pairs to process.")

    raw_dir = out_dir / "_raw_assembly"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate by PDB; track which (cpx, pdb) pairs share a PDB
    unique_pdbs = sorted({p for _, p, _ in pairs})
    log(
        f"\nProcessing {len(unique_pdbs)} unique PDBs for "
        f"structure selection..."
    )

    pdb_to_choice: dict[str, tuple[str | None, str]] = {}
    for i, pdb_id in enumerate(unique_pdbs, 1):
        # Find all complexes referencing this PDB
        cpx_infos = [(c, info) for c, p, info in pairs if p == pdb_id]

        # Get the PDB's UniProt set to adapt stoichiometry
        pdb_uniprots = query_pdb_uniprots(pdb_id, _get_runtime_cache())

        # Use the first complex's adapted stoichiometry
        first_cpx, first_info = cpx_infos[0]
        adapted = adapt_stoichiometry(
            first_info["stoichiometry"],
            first_info["paralog_groups"],
            pdb_uniprots,
        )

        # Check if multiple complexes have different adapted stoichiometries
        if len(cpx_infos) > 1:
            distinct = set()
            for cpx, info in cpx_infos:
                a = adapt_stoichiometry(
                    info["stoichiometry"],
                    info["paralog_groups"],
                    pdb_uniprots,
                )
                distinct.add(tuple(sorted(a.items())))
            if len(distinct) > 1:
                log(
                    f"  [NOTE] {pdb_id} referenced by complexes with "
                    f"different stoichiometries; using first."
                )

        asm_id, reason = choose_structure(pdb_id, adapted)
        pdb_to_choice[pdb_id] = (asm_id, reason)
        log(
            f"  [{i}/{len(unique_pdbs)}] {pdb_id}: asm={asm_id} ({reason}) "
            f"target={dict(adapted)}"
        )

    # Download
    log(f"\nDownloading {len(unique_pdbs)} structure files to {raw_dir}...")
    downloaded = skipped = failed = no_asm = 0
    log_rows = []
    for i, pdb_id in enumerate(unique_pdbs, 1):
        asm_id, reason = pdb_to_choice[pdb_id]
        source = "asu" if asm_id == ASU_ID else "assembly"
        if asm_id is None:
            log(
                f"  [{i}/{len(unique_pdbs)}] {pdb_id}: no metadata, skipping"
            )
            no_asm += 1
            log_rows.append(
                {
                    "pdb_id": pdb_id,
                    "assembly_id": "",
                    "source": "",
                    "reason": reason,
                    "status": "no_asm",
                    "path": "",
                }
            )
            continue
        status, path = download_structure(pdb_id, asm_id, raw_dir)
        log_rows.append(
            {
                "pdb_id": pdb_id,
                "assembly_id": asm_id,
                "source": source,
                "reason": reason,
                "status": status,
                "path": path,
            }
        )
        if status == "ok":
            downloaded += 1
            log(f"  [{i}/{len(unique_pdbs)}] {pdb_id}: downloaded ({source})")
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
        time.sleep(RCSB_DELAY)

    log_path = out_dir / "structure_download_log.csv"
    with open(log_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "pdb_id",
                "assembly_id",
                "source",
                "reason",
                "status",
                "path",
            ],
        )
        w.writeheader()
        w.writerows(log_rows)

    n_cif = sum(1 for r in log_rows if r["path"].endswith(".cif"))
    n_pdb = sum(1 for r in log_rows if r["path"].endswith(".pdb"))
    log(
        f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, "
        f"Failed: {failed}, No-metadata: {no_asm}"
    )
    log(f"  PDB format: {n_pdb}, mmCIF format: {n_cif}")
    log(f"Log: {log_path}")


# ---------------------------------------------------------------------------
# Runtime cache accessor (for step4 to access the file cache)
# ---------------------------------------------------------------------------

_runtime_cache: dict = {"best_structures": {}, "pdb_uniprots": {}}


def set_runtime_cache(cache: dict) -> None:
    global _runtime_cache
    _runtime_cache = cache


def _get_runtime_cache() -> dict:
    return _runtime_cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tsv",
        required=True,
        type=Path,
        help="Saccharomyces_cerevisiae_ComplexTab.tsv",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory (all results land here)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse TSV and report counts; no downloads, no API calls.",
    )
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"TSV not found: {args.tsv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Reading {args.tsv}")
    complexes, uniprot_to_complexes = parse_tsv(args.tsv)
    all_uniprots = sorted(uniprot_to_complexes.keys())

    n_with_xref = sum(
        1 for c in complexes
        if any(complexes[c]["xrefs"][t] for t in TAGS)
    )
    unique_xref_pdbs: set[str] = set()
    for c in complexes:
        for t in TAGS:
            unique_xref_pdbs.update(complexes[c]["xrefs"][t])

    n_with_paralogue = sum(
        1 for c in complexes if complexes[c]["has_paralogue"]
    )

    log(f"  Complexes: {len(complexes)}")
    log(f"  Unique UniProt accessions: {len(all_uniprots)}")
    log(f"  Complexes with Complex Portal PDB xref: {n_with_xref}")
    log(
        f"  Unique Complex Portal PDB IDs (to download in step 1): "
        f"{len(unique_xref_pdbs)}"
    )
    log(f"  Complexes with paralog groups: {n_with_paralogue}")

    if args.dry_run:
        log("\n[DRY-RUN] No downloads, no API calls. No files written.")
        for cpx in sorted(complexes)[:5]:
            info = complexes[cpx]
            log(
                f"  {cpx} ({info['name']}): "
                f"fixed={sorted(info['fixed_proteins'])}, "
                f"paralog_groups={[sorted(g) for g in info['paralog_groups']]}, "
                f"has_paralogue={info['has_paralogue']}, "
                f"xrefs={info['xrefs']}"
            )
        return

    cache_path = args.out_dir / "sifts_cache.json"
    cache = load_cache(cache_path)
    set_runtime_cache(cache)
    log(
        f"  SIFTS cache: {len(cache['best_structures'])} proteins, "
        f"{len(cache['pdb_uniprots'])} PDBs already queried"
    )

    # STEP 1
    step1_download_xref_pdbs(complexes, args.out_dir)
    save_cache(cache_path, cache)

    # STEP 2
    step2_sifts_uniprot_to_pdb(
        complexes, uniprot_to_complexes, cache, args.out_dir
    )
    save_cache(cache_path, cache)

    # STEP 3
    exact_csv = step3_exact_matches(complexes, cache, args.out_dir)
    save_cache(cache_path, cache)

    # STEP 4
    step4_download_exact_match_pdbs(complexes, exact_csv, args.out_dir)
    save_cache(cache_path, cache)

    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE")
    log("=" * 70)
    raw_dir = args.out_dir / "_raw_assembly"
    n_files = (
        len(list(raw_dir.glob("*.pdb"))) + len(list(raw_dir.glob("*.cif")))
    )
    log(f"  Output dir: {args.out_dir}")
    log(f"  Structure files in _raw_assembly/: {n_files}")


if __name__ == "__main__":
    main()
