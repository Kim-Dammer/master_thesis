#!/usr/bin/env python3
"""
Post-process mmseqs2 easy-search output (with qaln/taln columns) to compute,
for each query protein x PDB hit pair:
 - identity_percent : exact-match residues / full query length
 - similarity_percent : (exact matches + BLOSUM62-positive substitutions) / full query length

Both metrics are defined the same way as run_single_true_identity.py's
true_identity_percent / true_similarity_percent, so mmseqs2 results are
directly comparable to your existing HMMER-based true_identity.csv files.

Reads a Parquet file (mmseqs2 raw output) in batches via PyArrow, computes
identity/similarity with numpy-vectorized BLOSUM62 lookups, and writes the
result incrementally as Parquet. Handles 5M+ rows without stalling.

Usage:
uv run 3_compute_mmseqs_similarity.py \
  --input /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_e_value_100/mmseqs_results.parquet \
  --output /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_e_value_100/results/mmseqs_identity_similarity_e_value_100.parquet \
  --summary /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_e_value_100/results/mmseqs_per_protein_summary_e_value_100.parquet
"""

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from Bio.Align import substitution_matrices

# ── Pre-build a 128×128 ASCII-indexed BLOSUM62 score matrix ────────────
# This replaces the per-pair try/except lookup with a single numpy
# fancy-indexing operation: _SCORE[q_bytes, t_bytes] gives all scores
# at once for the entire alignment.
_BLOSUM62 = substitution_matrices.load("BLOSUM62")
_SCORE = np.full((128, 128), -99, dtype=np.int16)
for _a in "ACDEFGHIKLMNPQRSTVWY":
    for _b in "ACDEFGHIKLMNPQRSTVWY":
        s = int(_BLOSUM62[_a, _b])
        _SCORE[ord(_a), ord(_b)] = s
        _SCORE[ord(_b), ord(_a)] = s

_DASH = ord("-")

# ── Output Parquet schema ──────────────────────────────────────────────
SCHEMA = pa.schema([
    ("protein_id",         pa.string()),
    ("hit_pdb_id",         pa.string()),
    ("identity_percent",   pa.float64()),
    ("similarity_percent", pa.float64()),
    ("evalue",             pa.float64()),
    ("alnlen",             pa.int32()),
    ("qlen",               pa.int32()),
    ("tlen",               pa.int32()),
    ("is_high_homology",   pa.bool_()),
])

BATCH_SIZE = 100_000


def compute_identity_similarity(qaln: str, taln: str, qlen: int) -> tuple[float, float]:
    """
    qaln/taln are already-aligned (same length, '-' = gap) strings from mmseqs2.
    Returns (identity_percent, similarity_percent), both as % of full query length.

    Uses numpy vectorized operations instead of a per-character Python loop:
    - np.frombuffer gives an O(1) view of the ASCII bytes
    - _SCORE[q, t] does the BLOSUM62 lookup for ALL positions at once
    """
    q = np.frombuffer(qaln.upper().encode("ascii"), dtype=np.uint8)
    t = np.frombuffer(taln.upper().encode("ascii"), dtype=np.uint8)

    non_gap = (q != _DASH) & (t != _DASH)
    matches = int(np.count_nonzero(non_gap & (q == t)))
    similar = int(np.count_nonzero(non_gap & (_SCORE[q, t] > 0)))

    identity = (matches / qlen) * 100 if qlen else 0.0
    similarity = (similar / qlen) * 100 if qlen else 0.0
    return identity, similarity


def main():
    parser = argparse.ArgumentParser(
        description="Compute identity/similarity from mmseqs2 alignments (Parquet in → Parquet out)"
    )
    parser.add_argument("--input", required=True, help="mmseqs2 output Parquet file (with qaln/taln)")
    parser.add_argument("--output", required=True, help="Output Parquet: one row per (protein, hit) pair")
    parser.add_argument("--summary", required=True, help="Output Parquet: one row per protein (best hit)")
    parser.add_argument("--identity-threshold", type=float, default=30.0,
                        help="Identity %% threshold for is_high_homology flag (default: 30.0)")
    args = parser.parse_args()

    threshold = args.identity_threshold

    # ── Stream the input Parquet in batches ────────────────────────────
    pf = pq.ParquetFile(args.input)
    n_total = pf.metadata.num_rows
    print(f"Input: {args.input} ({n_total:,} rows)", flush=True)

    writer = pq.ParquetWriter(args.output, SCHEMA, compression="snappy")

    batch = []
    total_rows = 0
    best_per_protein = {}  # protein_id -> best row dict (by identity)

    for arrow_batch in pf.iter_batches(batch_size=BATCH_SIZE):
        # Convert batch to Python dicts (only the columns we need)
        cols = {name: arrow_batch.column(name).to_pylist()
                for name in ("query", "target", "evalue", "alnlen",
                             "qlen", "tlen", "qaln", "taln")}

        for i in range(len(cols["query"])):
            protein_id = cols["query"][i]
            hit_pdb_id = cols["target"][i]
            qlen = cols["qlen"][i]
            qaln = cols["qaln"][i]
            taln = cols["taln"][i]

            identity, similarity = compute_identity_similarity(qaln, taln, qlen)

            out_row = {
                "protein_id":         protein_id,
                "hit_pdb_id":         hit_pdb_id,
                "identity_percent":   round(identity, 2),
                "similarity_percent": round(similarity, 2),
                "evalue":             float(cols["evalue"][i]),
                "alnlen":             int(cols["alnlen"][i]),
                "qlen":               int(qlen),
                "tlen":               int(cols["tlen"][i]),
                "is_high_homology":   identity > threshold,
            }
            batch.append(out_row)
            total_rows += 1

            # Track best hit per protein
            current_best = best_per_protein.get(protein_id)
            if current_best is None or identity > current_best["identity_percent"]:
                best_per_protein[protein_id] = out_row

        # Flush batch to Parquet
        table = pa.Table.from_pylist(batch, schema=SCHEMA)
        writer.write_table(table)
        batch.clear()
        print(f" ...processed {total_rows:,} / {n_total:,} hits", flush=True)

    writer.close()
    print(f" ...done ({total_rows:,} hits)", flush=True)

    # ── Summary: best hit per protein → Parquet via polars ─────────────
    df_summary = pl.DataFrame([best_per_protein[k] for k in sorted(best_per_protein)])
    df_summary.write_parquet(args.summary)

    n_proteins = len(best_per_protein)
    n_with_homology = int(df_summary["is_high_homology"].sum())
    print(f"Processed {total_rows:,} hits across {n_proteins} proteins.")
    print(f"{n_with_homology}/{n_proteins} proteins have a best hit above {threshold}% identity.")
    print(f"Per-hit results:   {args.output}")
    print(f"Per-protein summary: {args.summary}")


if __name__ == "__main__":
    main()
