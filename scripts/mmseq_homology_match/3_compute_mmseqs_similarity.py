#!/usr/bin/env python3
"""
Post-process mmseqs2 easy-search output (with qaln/taln columns) to compute,
for each query protein x PDB hit pair:
  - identity_percent   : exact-match residues / full query length
  - similarity_percent : (exact matches + BLOSUM62-positive substitutions) / full query length

Both metrics are defined the same way as run_single_true_identity.py's
true_identity_percent / true_similarity_percent, so mmseqs2 results are
directly comparable to your existing HMMER-based true_identity.csv files.

Since mmseqs2 (with -a 1) already produces the real alignment (qaln/taln),
no re-alignment is needed here - this is pure counting, so it's fast even
for thousands of hits.

Usage:
uv run 3_compute_mmseqs_similarity.py \
    --input /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run/mmseqs_results.tsv \
    --output /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run/results/mmseqs_identity_similarity.csv \
    --summary /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run/results/mmseqs_per_protein_summary.csv

"""

import argparse
import csv
from pathlib import Path
 
from Bio.Align import substitution_matrices
 
BLOSUM62 = substitution_matrices.load("BLOSUM62")
 
 
def score_pair(a: str, b: str) -> int:
    """BLOSUM62 score for a single aligned residue pair (both uppercase, no gaps)."""
    try:
        return BLOSUM62[a, b]
    except (KeyError, IndexError):
        try:
            return BLOSUM62[b, a]
        except (KeyError, IndexError):
            return -99  # unknown residue code (e.g. 'U','O','X','*') - never counts as similar # unknown residue code (e.g. 'X') - never counts as similar
 
 
def compute_identity_similarity(qaln: str, taln: str, qlen: int) -> tuple[float, float]:
    """
    qaln/taln are already-aligned (same length, '-' = gap) strings from mmseqs2.
    Returns (identity_percent, similarity_percent), both as % of full query length.
    """
    matches = 0
    similar = 0
    for a, b in zip(qaln.upper(), taln.upper()):
        if a == "-" or b == "-":
            continue
        if a == b:
            matches += 1
            similar += 1
        elif score_pair(a, b) > 0:
            similar += 1
 
    identity = (matches / qlen) * 100 if qlen else 0.0
    similarity = (similar / qlen) * 100 if qlen else 0.0
    return identity, similarity
 
 
def main():
    parser = argparse.ArgumentParser(description="Compute identity/similarity from mmseqs2 alignments")
    parser.add_argument("--input", required=True, help="mmseqs easy-search TSV output")
    parser.add_argument("--output", required=True, help="Output CSV: one row per (protein, hit) pair")
    parser.add_argument("--summary", required=True, help="Output CSV: one row per protein (best hit)")
    parser.add_argument("--identity-threshold", type=float, default=30.0,
                         help="Identity %% threshold for is_high_homology flag (default: 30.0)")
    args = parser.parse_args()
 
    rows_out = []
    best_per_protein = {}  # protein_id -> best row (by identity)
 
    with open(args.input) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            protein_id = row["query"]
            hit_pdb_id = row["target"]
            qlen = int(row["qlen"])
            qaln = row["qaln"]
            taln = row["taln"]
 
            identity, similarity = compute_identity_similarity(qaln, taln, qlen)
 
            out_row = {
                "protein_id": protein_id,
                "hit_pdb_id": hit_pdb_id,
                "identity_percent": round(identity, 2),
                "similarity_percent": round(similarity, 2),
                "evalue": row["evalue"],
                "alnlen": row["alnlen"],
                "qlen": qlen,
                "tlen": row["tlen"],
                "is_high_homology": "YES" if identity > args.identity_threshold else "NO",
            }
            rows_out.append(out_row)
 
            current_best = best_per_protein.get(protein_id)
            if current_best is None or identity > current_best["identity_percent"]:
                best_per_protein[protein_id] = out_row
 
    # Write full per-hit CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "protein_id", "hit_pdb_id", "identity_percent", "similarity_percent",
            "evalue", "alnlen", "qlen", "tlen", "is_high_homology"
        ])
        writer.writeheader()
        writer.writerows(rows_out)
 
    # Write one-row-per-protein summary (best hit only)
    with open(args.summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "protein_id", "hit_pdb_id", "identity_percent", "similarity_percent",
            "evalue", "alnlen", "qlen", "tlen", "is_high_homology"
        ])
        writer.writeheader()
        for protein_id in sorted(best_per_protein):
            writer.writerow(best_per_protein[protein_id])
 
    n_proteins = len(best_per_protein)
    n_hits = len(rows_out)
    n_with_homology = sum(1 for r in best_per_protein.values() if r["is_high_homology"] == "YES")
    print(f"Processed {n_hits} hits across {n_proteins} proteins.")
    print(f"{n_with_homology}/{n_proteins} proteins have a best hit above {args.identity_threshold}% identity.")
    print(f"Per-hit results: {args.output}")
    print(f"Per-protein summary: {args.summary}")
 
 
if __name__ == "__main__":
    main()
 