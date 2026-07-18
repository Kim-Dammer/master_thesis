#!/usr/bin/env python3
"""
find_homomultimers.py — Protein-first homomultimer identification.

1. go through the pdb file, if sequences of all chains match keep in csv called homomultimers.csv with pdb_id seq_len, and n_chains
2. for these pdb file get the corresponding prot id

Starts from a yeast protein list, uses an existing UniProt→PDB mapping CSV to
get yeast-associated PDB IDs, scans ONLY those PDBs in the PDB seqres FASTA
file, and identifies homomultimers (>=2 protein chains, all identical sequence).
UniProt IDs come from the mapping CSV — zero API calls.

Usage:
    uv run 1_find_homomultimers.py \
        --proteins /cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_CP_proteins.csv\
        --mapping /cluster/project/beltrao/kdammer/master_thesis/data/Homomltimer/uniprot_pdb_mapping.csv\
        --fasta /cluster/project/beltrao/kdammer/master_thesis/data/pdb/pdb_seqres.txt \
        --sequences /cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_yeast_proteins_uniprot_mapped_sequences.csv \
        --out-dir /cluster/project/beltrao/kdammer/master_thesis/data/Homomltimer
"""
import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

HEADER_RE = re.compile(r"^>([A-Za-z0-9]{4})_([A-Za-z0-9]+)\s+mol:(\S+)\s+length:(\d+)")
UNIPROT_RE = re.compile(r"^[OPQ][0-9][0-9A-Z]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")


def load_proteins(path):
    """Read protein list CSV (single column 'identifier'), return set of valid UniProt IDs."""
    ids = set()
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            uid = row[0].strip()
            # strip PRO_ isoform suffix: P00410-PRO_0000006048 -> P00410
            uid = uid.split("-PRO_")[0]
            if UNIPROT_RE.match(uid):
                ids.add(uid)
    return ids


def load_uniprot_seq_lengths(path):
    """Read all_yeast_proteins_uniprot_mapped_sequences.csv (columns: uniprot_id,sequence).
    Return {uniprot_id: len(sequence)}."""
    lengths = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            uid = row["uniprot_id"].strip()
            seq = row.get("sequence", "")
            if uid and seq:
                lengths[uid] = len(seq.strip())
    return lengths


def load_mapping(path, protein_ids):
    """Read uniprot_pdb_mapping.csv, filter to yeast proteins.
    Return {pdb_id: set(uniprot_id)} and set of PDB IDs to scan."""
    pdb_to_uniprots = defaultdict(set)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            uid = row["uniprot_id"].strip()
            if uid in protein_ids:
                pdb_to_uniprots[row["pdb_id"].strip().lower()].add(uid)
    return pdb_to_uniprots


def parse_fasta_filtered(path, target_pdbs):
    """Parse FASTA but only keep chains for PDB IDs in target_pdbs.
    Return {pdb_id: {chain_id: {mol, seq}}}."""
    data = defaultdict(dict)
    pdb = chain = None; seq = []
    target = target_pdbs  # set, for O(1) lookup

    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if pdb and chain and pdb in target:
                data[pdb][chain]["seq"] = "".join(seq)
            m = HEADER_RE.match(line)
            if m and m.group(1).lower() in target:
                pdb, chain = m.group(1).lower(), m.group(2)
                data[pdb][chain] = {"mol": m.group(3), "seq": ""}
                seq = []
            else:
                pdb = chain = None
        elif pdb and chain:
            seq.append(line.strip())
    if pdb and chain and pdb in target:
        data[pdb][chain]["seq"] = "".join(seq)
    return dict(data)


def find_homomultimers(fasta_data, pdb_to_uniprots, uniprot_seq_lengths):
    """Identify homomultimers: >=2 protein chains, all identical sequence."""
    out = []
    for pdb, chains in fasta_data.items():
        prot = [c["seq"] for c in chains.values() if c["mol"] == "protein" and c["seq"]]
        if len(prot) >= 2 and len(set(prot)) == 1:
            has_nonprot = any(c["mol"] != "protein" for c in chains.values())
            uniprots = sorted(pdb_to_uniprots.get(pdb, set()))
            uniprot_str = ";".join(uniprots)
            # if single uniprot, look up its full sequence length
            uniprot_seq_len = uniprot_seq_lengths.get(uniprots[0], "") if len(uniprots) == 1 else ""
            out.append({
                "pdb_id": pdb,
                "seq_len": len(prot[0]),
                "n_chains": len(prot),
                "has_non_protein_part": has_nonprot,
                "uniprot_id": uniprot_str,
                "uniprot_seq_len": uniprot_seq_len,
            })
    return sorted(out, key=lambda r: r["pdb_id"])


def main():
    ap = argparse.ArgumentParser(description="Find homomultimers for yeast proteins using existing CSV + FASTA.")
    ap.add_argument("--proteins", type=Path, required=True, help="Protein list CSV (single column: identifier)")
    ap.add_argument("--mapping", type=Path, required=True, help="uniprot_pdb_mapping.csv")
    ap.add_argument("--fasta", type=Path, required=True, help="PDB seqres FASTA file (.txt)")
    ap.add_argument("--sequences", type=Path, required=False, help="CSV with UniProt sequences (columns: uniprot_id,sequence)")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load yeast proteins
    proteins = load_proteins(args.proteins)
    print(f"Loaded {len(proteins)} yeast UniProt IDs from {args.proteins.name}")

    # 1b. Load UniProt sequence lengths (optional)
    uniprot_seq_lengths = {}
    if args.sequences:
        uniprot_seq_lengths = load_uniprot_seq_lengths(args.sequences)
        print(f"Loaded {len(uniprot_seq_lengths)} UniProt sequence lengths from {args.sequences.name}")

    # 2. Load mapping, filter to yeast proteins
    pdb_to_uniprots = load_mapping(args.mapping, proteins)
    target_pdbs = set(pdb_to_uniprots.keys())
    print(f"Found {len(target_pdbs)} yeast-associated PDB IDs in {args.mapping.name}")

    # 3. Parse FASTA (only for target PDBs)
    print(f"Parsing FASTA (filtering to {len(target_pdbs)} PDBs)...")
    fasta_data = parse_fasta_filtered(args.fasta, target_pdbs)
    print(f"  {len(fasta_data)} PDBs found in FASTA")

    # 4. Identify homomultimers
    homo = find_homomultimers(fasta_data, pdb_to_uniprots, uniprot_seq_lengths)
    print(f"  {len(homo)} homomultimers found (>=2 identical protein chains)")

    # 5. Write output
    out_path = args.out_dir / "homomultimers.csv"
    fieldnames = ["pdb_id", "seq_len", "n_chains", "has_non_protein_part", "uniprot_id"]
    if uniprot_seq_lengths:
        fieldnames.append("uniprot_seq_len")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(homo)
    print(f"Wrote {out_path} ({len(homo)} rows)")


if __name__ == "__main__":
    sys.exit(main())
