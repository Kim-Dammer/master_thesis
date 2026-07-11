#!/usr/bin/env python
"""
Maps Complex Portal yeast complexes to exact-match PDB structures and reports per-protein
coverage, length-weighted overall coverage, best biological assembly, and
resolution. No PDB files are downloaded in this script

Does not yet deal well with paralogues (requires both for perfect macth with the same stoichiometry as one of them would have)
Coverage calcualtion is not correct yet, instead of https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/P14284, it would need to get coverage from 
https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/6PSX
Fully self-contained: queries SIFTS, UniProt, and RCSB assembly APIs itself
and keeps all caches inside the output folder.


Outputs:
  complex_all_possible_pdbs.csv  — one row per complex (all 634)
  pdb_stats.csv                  — one row per (exact-match PDB, complex) pair
  sifts_best_structures_cache.json
  sifts_pdb_uniprots_cache.json
  uniprot_lengths_cache.json
  rcsb_assembly_cache.json

Usage:
    uv run 21_1_complex_pdb_mapping_v2.py \
        --tsv    /cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv \
        --out-dir /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/excact_pdb_match
"""
from __future__ import annotations

import argparse
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
SIFTS_BEST_STRUCTURES_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{u}"
SIFTS_PDB_UNIPROT_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/{u}.json"
RCSB_ASSEMBLY_API = "https://data.rcsb.org/rest/v1/core/assembly/{pdb}/{asm_id}"

API_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5
SIFTS_DELAY = 0.1
UNIPROT_DELAY = 0.1
RCSB_ASM_DELAY = 0.05

UNIPROT_RE = re.compile(r"^[OPQ][0-9][0-9A-Z]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
BRACKET_RE = re.compile(r"^\[([A-Z0-9_,\s]+)\]$")
PRO_ISOFORM_RE = re.compile(
    r"^([OPQ][0-9][0-9A-Z]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})-PRO_\d+$"
)


# ---------------------------------------------------------------------------
# Helpers
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


def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

def parse_tsv(tsv_path: Path) -> dict[str, dict]:
    """Return {cpx: {name, proteins:set, counts:Counter}}."""
    df = pd.read_csv(tsv_path, sep="\t")
    mol_col = "Identifiers (and stoichiometry) of molecules in complex"
    name_col = "Recommended name"

    complexes: dict[str, dict] = {}
    for _, row in df.iterrows():
        cpx = str(row["#Complex ac"]).strip()
        if not cpx:
            continue
        name = str(row.get(name_col, "")).strip()

        proteins: set[str] = set()
        counts: Counter = Counter()
        raw = row.get(mol_col, "")
        if isinstance(raw, str):
            for token in raw.split("|"):
                base = token.split("(")[0].strip()
                count_m = re.search(r"\((\d+)\)", token)
                count = int(count_m.group(1)) if count_m else 1
                for u in extract_proteins(base):
                    proteins.add(u)
                    counts[u] += count

        complexes[cpx] = {"name": name, "proteins": proteins, "counts": counts}
    return complexes


# ---------------------------------------------------------------------------
# SIFTS: best_structures per UniProt
# ---------------------------------------------------------------------------

def query_sifts_best_structures(
    uniprots: list[str], cache_path: Path
) -> dict[str, list[dict]]:
    """Return {uniprot: [entry, ...]} from SIFTS best_structures."""
    cache = _load_cache(cache_path)
    total = len(uniprots)
    log(f"  Querying SIFTS best_structures for {total} unique proteins...")
    for i, u in enumerate(uniprots, 1):
        if u in cache:
            continue
        data = _get_json(SIFTS_BEST_STRUCTURES_URL.format(u=u), "MappingV2/1.0")
        cache[u] = data.get(u, []) if data else []
        time.sleep(SIFTS_DELAY)
        if i % 200 == 0:
            log(f"    [{i}/{total}] processed")
            _save_cache(cache_path, cache)
    _save_cache(cache_path, cache)
    return cache


# ---------------------------------------------------------------------------
# SIFTS: full UniProt set per PDB
# ---------------------------------------------------------------------------

def query_sifts_pdb_uniprots(
    pdb_ids: list[str], cache_path: Path
) -> dict[str, list[str]]:
    """Return {pdb_id: [uniprot, ...]} (full UniProt set of each PDB)."""
    cache = _load_cache(cache_path)
    total = len(pdb_ids)
    log(f"  Querying SIFTS /mappings/uniprot for {total} unique PDBs...")
    for i, pdb_id in enumerate(pdb_ids, 1):
        key = pdb_id.lower()
        if key in cache:
            continue
        data = _get_json(SIFTS_PDB_UNIPROT_URL.format(pdb=key), "MappingV2/1.0")
        uniprots = (
            sorted(data[key].get("UniProt", {}).keys())
            if data and key in data
            else []
        )
        cache[key] = uniprots
        time.sleep(SIFTS_DELAY)
        if i % 200 == 0:
            log(f"    [{i}/{total}] processed")
            _save_cache(cache_path, cache)
    _save_cache(cache_path, cache)
    return cache


# ---------------------------------------------------------------------------
# SIFTS: chain -> uniprot mapping per PDB (for assembly stoichiometry)
# ---------------------------------------------------------------------------

def sifts_chain_uniprot(pdb_id: str, pdb_uniprots_cache: dict) -> dict[str, str]:
    """Return {chain_id: uniprot} for the asymmetric unit, from a fresh SIFTS
    /mappings/uniprot call (cached in pdb_uniprots_cache by reference)."""
    key = pdb_id.lower()
    data = _get_json(SIFTS_PDB_UNIPROT_URL.format(pdb=key), "MappingV2/1.0")
    chain_map: dict[str, str] = {}
    if data and key in data:
        for uniprot, info in data[key].get("UniProt", {}).items():
            for m in info.get("mappings", []):
                ch = m.get("chain_id")
                if ch and ch not in chain_map:
                    chain_map[ch] = uniprot
    time.sleep(SIFTS_DELAY)
    return chain_map


# ---------------------------------------------------------------------------
# UniProt: canonical sequence length
# ---------------------------------------------------------------------------

def query_uniprot_lengths(
    uniprots: list[str], cache_path: Path
) -> dict[str, int]:
    """Return {uniprot: canonical_sequence_length}."""
    cache = _load_cache(cache_path)
    total = len(uniprots)
    log(f"  Querying UniProt REST API for canonical lengths of {total} proteins...")
    for i, u in enumerate(uniprots, 1):
        if u in cache:
            continue
        data = _get_json(UNIPROT_API_URL.format(u=u), "MappingV2/1.0")
        if data and "sequence" in data:
            cache[u] = int(data["sequence"].get("length", 0))
        else:
            cache[u] = None  # 404 / obsolete accession
            log(f"    [WARN] {u}: no UniProt record (obsolete/deleted?)")
        time.sleep(UNIPROT_DELAY)
        if i % 200 == 0:
            log(f"    [{i}/{total}] processed")
            _save_cache(cache_path, cache)
    _save_cache(cache_path, cache)
    return cache


# ---------------------------------------------------------------------------
# RCSB assembly metadata + stoichiometry matching
# ---------------------------------------------------------------------------

def rcsb_assemblies(pdb_id: str, cache_path: Path) -> list[dict]:
    """Return list of assembly metadata dicts from RCSB API.
    Each: {assembly_id, oligomeric_count, oligomeric_details, details, chains}."""
    cache = _load_cache(cache_path)
    key = pdb_id.lower()
    if key in cache:
        return cache[key]

    assemblies: list[dict] = []
    for asm_id in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        url = RCSB_ASSEMBLY_API.format(pdb=key, asm_id=asm_id)
        data = _get_json(url, "MappingV2/1.0")
        if data is None:
            break
        struct = data.get("pdbx_struct_assembly", {})
        info = data.get("rcsb_assembly_info", {})
        container = data.get("rcsb_assembly_container_identifiers", {})
        chains = container.get("chain_ids", []) or info.get("chain_ids", [])
        assemblies.append({
            "assembly_id": asm_id,
            "oligomeric_count": struct.get("oligomeric_count"),
            "oligomeric_details": struct.get("oligomeric_details", ""),
            "details": struct.get("rcsb_details", struct.get("details", "")),
            "chains": chains,
        })
        time.sleep(RCSB_ASM_DELAY)

    cache[key] = assemblies
    _save_cache(cache_path, cache)
    return assemblies


def choose_assembly(
    pdb_id: str, target_stoich: Counter, asm_cache_path: Path
) -> tuple[str | None, str]:
    """Return (assembly_id, reason).
    reason: 'exact_match' | 'closest_oligomeric' | 'author_default' |
            'no_assembly_metadata'."""
    assemblies = rcsb_assemblies(pdb_id, asm_cache_path)
    if not assemblies:
        return (None, "no_assembly_metadata")

    target_total = sum(target_stoich.values())

    # 1. Exact stoichiometry match (UniProt multiset equals target)
    chain_to_u = sifts_chain_uniprot(pdb_id, {})
    exact_matches = []
    for a in assemblies:
        if not a["chains"]:
            continue
        asm_counts: Counter = Counter()
        for ch in a["chains"]:
            u = chain_to_u.get(ch)
            if u:
                asm_counts[u] += 1
        if asm_counts == target_stoich:
            exact_matches.append(a)

    if exact_matches:
        author = [a for a in exact_matches if "author" in a["details"].lower()]
        chosen = author[0] if author else exact_matches[0]
        return (chosen["assembly_id"], "exact_match")

    # 2. Closest oligomeric count
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
    return (assemblies[0]["assembly_id"], "author_default")


# ---------------------------------------------------------------------------
# Build the two dataframes
# ---------------------------------------------------------------------------

def build_dataframes(
    complexes: dict[str, dict],
    sifts_best: dict[str, list[dict]],
    pdb_uniprots: dict[str, list[str]],
    uniprot_lengths: dict[str, int | None],
    asm_cache_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Index: per-protein max coverage per PDB, and per-PDB min resolution
    # from best_structures entries.
    # {uniprot: {pdb_id: max_coverage}}
    prot_pdb_cov: dict[str, dict[str, float]] = {}
    # {pdb_id: min_resolution}
    pdb_min_res: dict[str, float | None] = {}
    for u, entries in sifts_best.items():
        for e in entries:
            pdb = str(e.get("pdb_id", "")).upper()
            if not pdb:
                continue
            cov = e.get("coverage")
            if cov is not None:
                prot_pdb_cov.setdefault(u, {})
                if pdb not in prot_pdb_cov[u] or cov > prot_pdb_cov[u][pdb]:
                    prot_pdb_cov[u][pdb] = float(cov)
            res = e.get("resolution")
            if res is not None:
                if pdb not in pdb_min_res or res < pdb_min_res[pdb]:
                    pdb_min_res[pdb] = float(res)

    # pdb -> full UniProt set (uppercase keys)
    pdb_proteins: dict[str, set[str]] = {
        pdb.upper(): set(us) for pdb, us in pdb_uniprots.items()
    }

    # --- DF1: complex_all_possible_pdbs ---
    df1_rows = []
    for cpx in sorted(complexes):
        cpx_prots = complexes[cpx]["proteins"]
        exact_pdbs = sorted(
            pdb for pdb, prots in pdb_proteins.items() if prots == cpx_prots
        )
        df1_rows.append({
            "Complex_ac": cpx,
            "cleaned_identifiers": ";".join(sorted(cpx_prots)),
            "identity_match_pdb": ";".join(exact_pdbs),
        })
    df1 = pd.DataFrame(df1_rows)

    # --- DF2: pdb_stats ---
    df2_rows = []
    for cpx in sorted(complexes):
        cpx_prots = complexes[cpx]["proteins"]
        target_stoich = complexes[cpx]["counts"]
        exact_pdbs = sorted(
            pdb for pdb, prots in pdb_proteins.items() if prots == cpx_prots
        )
        for pdb_id in exact_pdbs:
            # Per-protein coverage
            per_prot = []
            weighted_sum = 0.0
            length_sum = 0
            for u in sorted(cpx_prots):
                cov = prot_pdb_cov.get(u, {}).get(pdb_id, 0.0)
                length = uniprot_lengths.get(u)
                per_prot.append(f"{u}:{cov:.4f}")
                if length is not None and length > 0:
                    weighted_sum += cov * length
                    length_sum += length
            overall = weighted_sum / length_sum if length_sum > 0 else float("nan")

            # Assembly
            asm_id, reason = choose_assembly(pdb_id, target_stoich, asm_cache_path)
            if asm_id is None:
                assembly_str = "no_assembly_metadata"
            elif reason == "exact_match":
                assembly_str = f"biological_assembly_{asm_id}"
            elif reason == "closest_oligomeric":
                assembly_str = f"biological_assembly_{asm_id}_closest_oligomeric"
            else:
                assembly_str = f"biological_assembly_{asm_id}_{reason}"

            df2_rows.append({
                "pdb_id": pdb_id,
                "Complex_ac": cpx,
                "assembly": assembly_str,
                "per_protein_coverage": ",".join(per_prot),
                "Overall_coverage": round(overall, 4) if overall == overall else "",
                "resolution": pdb_min_res.get(pdb_id, ""),
            })
    df2 = pd.DataFrame(df2_rows)
    return df1, df2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tsv", required=True, type=Path,
                    help="Saccharomyces_cerevisiae_ComplexTab.tsv")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory (complete_complex_pdb_mapping_v2/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse TSV and report counts; no API calls, no CSVs written.")
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"TSV not found: {args.tsv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("23_complex_pdb_mapping_v2.py — standalone PDB mapping + stats")
    log("=" * 70)

    log("\nParsing TSV...")
    complexes = parse_tsv(args.tsv)
    all_uniprots = sorted({u for info in complexes.values() for u in info["proteins"]})
    n_with_proteins = sum(1 for c in complexes.values() if c["proteins"])
    log(f"  {len(complexes)} complexes, {n_with_proteins} with >=1 protein")
    log(f"  {len(all_uniprots)} unique UniProt accessions")

    if args.dry_run:
        log("\n[DRY-RUN] No API calls, no CSVs written.")
        for cpx in sorted(complexes)[:5]:
            info = complexes[cpx]
            log(f"  {cpx} ({info['name']}): proteins={sorted(info['proteins'])}, "
                f"counts={dict(info['counts'])}")
        return

    # Caches live inside the output folder (fully standalone)
    bs_cache = args.out_dir / "sifts_best_structures_cache.json"
    pu_cache = args.out_dir / "sifts_pdb_uniprots_cache.json"
    ul_cache = args.out_dir / "uniprot_lengths_cache.json"
    asm_cache = args.out_dir / "rcsb_assembly_cache.json"

    # 1. SIFTS best_structures per UniProt
    log("\nSTEP 1: SIFTS best_structures per protein")
    sifts_best = query_sifts_best_structures(all_uniprots, bs_cache)
    n_with_pdb = sum(1 for v in sifts_best.values() if v)
    log(f"  {n_with_pdb} proteins have >=1 PDB structure")

    # 2. SIFTS full UniProt set per PDB
    log("\nSTEP 2: SIFTS full UniProt set per PDB")
    all_pdbs = sorted({
        str(e.get("pdb_id", "")).upper()
        for entries in sifts_best.values()
        for e in entries if e.get("pdb_id")
    })
    log(f"  {len(all_pdbs)} unique PDBs to query")
    pdb_uniprots = query_sifts_pdb_uniprots(all_pdbs, pu_cache)

    # 3. UniProt canonical lengths
    log("\nSTEP 3: UniProt canonical sequence lengths")
    uniprot_lengths = query_uniprot_lengths(all_uniprots, ul_cache)
    n_lengths = sum(1 for v in uniprot_lengths.values() if v is not None)
    log(f"  {n_lengths} lengths retrieved")

    # 4. Build dataframes (queries RCSB assembly API on demand)
    log("\nSTEP 4: Building dataframes (assembly selection per exact-match PDB)")
    df1, df2 = build_dataframes(
        complexes, sifts_best, pdb_uniprots, uniprot_lengths, asm_cache
    )

    # 5. Save
    df1_path = args.out_dir / "complex_all_possible_pdbs.csv"
    df2_path = args.out_dir / "pdb_stats.csv"
    df1.to_csv(df1_path, index=False)
    df2.to_csv(df2_path, index=False)

    n_exact = (df1["identity_match_pdb"] != "").sum()
    log("\n" + "=" * 70)
    log("DONE")
    log("=" * 70)
    log(f"  complex_all_possible_pdbs.csv: {len(df1)} rows, {n_exact} with exact-match PDB")
    log(f"  pdb_stats.csv:                  {len(df2)} (PDB, complex) pairs")
    log(f"  Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
