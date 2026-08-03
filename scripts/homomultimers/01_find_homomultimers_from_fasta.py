#!/usr/bin/env python3
"""
find_homomultimers_from_fasta.py

Find PDB homomultimers (>=2 protein chains, all identical sequence) for a set
of UniProt proteins. Input is a FASTA-like file (UniProt ID header + sequence).

Pipeline:
  1. Parse input FASTA -> {uniprot_id: sequence}
  2. Query SIFTS best_structures for each UniProt -> {pdb_id: set(uniprot_id)}
  3. Parse pdb_seqres.txt (only matched PDBs) -> {pdb_id: {chain: {mol, seq}}}
  4. Keep PDBs with >=2 protein chains, all identical -> homomultimers
  5. Write homomultimers.csv

Usage:
  uv run 01_find_homomultimers_from_fasta.py \
      --fasta    /cluster/project/beltrao/kdammer/master_thesis/data/Homomltimer/Pipeline_prep/prot_id_mapping.csv \
      --pdb /cluster/project/beltrao/kdammer/master_thesis/data/pdb/pdb_seqres.txt \
      --out-dir  /cluster/project/beltrao/kdammer/master_thesis/data/Homomltimer/Pipeline_prep
"""
import argparse
import json
import re
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SIFTS_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{u}"
HEADER_RE = re.compile(r"^>([A-Za-z0-9]{4})_([A-Za-z0-9]+)\s+mol:(\S+)\s+length:(\d+)")


# -- 1. Parse input FASTA ----------------------------------------------------
def parse_input(path):
    """FASTA reader -> {uniprot_id: sequence}."""
    seqs, uid, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if uid:
                seqs[uid] = "".join(buf)
            uid, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if uid:
        seqs[uid] = "".join(buf)
    return seqs


# -- 2. SIFTS query (cached, parallel) ---------------------------------------
def _fetch_one(uid):
    """Query SIFTS for one UniProt -> list of entries (empty on 404/no data)."""
    try:
        url = SIFTS_URL.format(u=uid)
        req = urllib.request.Request(url, headers={"User-Agent": "HomomultimerFinder/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return uid, json.loads(resp.read()).get(uid, [])
    except Exception:
        return uid, []  # 404 = no PDB structures for this protein


def query_sifts(uniprots, cache_path, workers=16):
    """Query SIFTS in parallel -> {pdb_id: set(uniprot_id)}. Cached for resume."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [u for u in uniprots if u not in cache]
    print(f"  {len(todo)} to query, {len(cache)} cached")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (uid, entries) in enumerate(
            pool.map(_fetch_one, todo), 1
        ):
            cache[uid] = entries
            if i % 200 == 0:
                cache_path.write_text(json.dumps(cache))
                print(f"  ...{i}/{len(todo)} queried")

    cache_path.write_text(json.dumps(cache))

    pdb_to_uniprot = defaultdict(set)
    for uid, entries in cache.items():
        for e in entries:
            pdb = e.get("pdb_id", "").lower()
            if pdb:
                pdb_to_uniprot[pdb].add(uid)
    return pdb_to_uniprot


# -- 3. Parse pdb_seqres.txt (filtered) --------------------------------------
def parse_pdb_seqres(path, target_pdbs):
    """Stream-parse pdb_seqres.txt, keep only target PDBs.
    -> {pdb_id: {chain_id: {mol, seq}}}"""
    data, pdb, chain, mol, buf = defaultdict(dict), None, None, None, []

    def flush():
        if pdb and chain and pdb in target_pdbs:
            data[pdb][chain] = {"mol": mol, "seq": "".join(buf)}

    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            flush()
            m = HEADER_RE.match(line)
            if m and m.group(1).lower() in target_pdbs:
                pdb, chain, mol = m.group(1).lower(), m.group(2), m.group(3)
            else:
                pdb = chain = mol = None
            buf = []
        elif pdb:
            buf.append(line.strip())
    flush()
    return data


# -- 4. Find homomultimers ---------------------------------------------------
def find_homomultimers(fasta_data, pdb_to_uniprot, input_seqs):
    """Keep PDBs with >=2 protein chains, all identical sequence."""
    out = []
    for pdb, chains in fasta_data.items():
        prot_seqs = [c["seq"] for c in chains.values()
                     if c["mol"] == "protein" and c["seq"]]
        if len(prot_seqs) >= 2 and len(set(prot_seqs)) == 1:
            uniprots = sorted(pdb_to_uniprot.get(pdb, set()))
            uid_str = ";".join(uniprots)
            uid_len = len(input_seqs[uniprots[0]]) if len(uniprots) == 1 and uniprots[0] in input_seqs else ""
            out.append({
                "pdb_id": pdb,
                "seq_len": len(prot_seqs[0]),
                "n_chains": len(prot_seqs),
                "has_non_protein_part": any(c["mol"] != "protein" for c in chains.values()),
                "uniprot_id": uid_str,
                "uniprot_seq_len": uid_len,
            })
    return sorted(out, key=lambda r: r["pdb_id"])


# -- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, type=Path, help="FASTA-like input (UniProt ID + sequence)")
    ap.add_argument("--pdb", required=True, type=Path, help="pdb_seqres.txt")
    ap.add_argument("--out-dir", default=Path("."), type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse input
    input_seqs = parse_input(args.fasta)
    print(f"Loaded {len(input_seqs)} proteins from {args.fasta.name}")

    # 2. Query SIFTS
    print(f"Querying SIFTS for {len(input_seqs)} UniProt IDs...")
    cache_path = args.out_dir / "sifts_cache.json"
    pdb_to_uniprot = query_sifts(list(input_seqs.keys()), cache_path)
    n_with_pdb = sum(1 for v in json.loads(cache_path.read_text()).values() if v)
    print(f"  {len(pdb_to_uniprot)} PDB IDs found ({n_with_pdb} proteins have structures)")

    # 3. Parse pdb_seqres.txt (filtered)
    print(f"Parsing {args.pdb.name} (filtering to {len(pdb_to_uniprot)} PDBs)...")
    fasta_data = parse_pdb_seqres(args.pdb, set(pdb_to_uniprot.keys()))
    print(f"  {len(fasta_data)} PDBs found in FASTA")

    # 4. Find homomultimers
    homo = find_homomultimers(fasta_data, pdb_to_uniprot, input_seqs)
    print(f"  {len(homo)} homomultimers found")

    # 5. Write output
    out_path = args.out_dir / "homomultimers.csv"
    cols = ["pdb_id", "seq_len", "n_chains", "has_non_protein_part", "uniprot_id", "uniprot_seq_len"]
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in homo:
            f.write(f"{r['pdb_id']},{r['seq_len']},{r['n_chains']},"
                    f"{r['has_non_protein_part']},{r['uniprot_id']},{r['uniprot_seq_len']}\n")
    print(f"Wrote {out_path} ({len(homo)} rows)")


if __name__ == "__main__":
    main()
