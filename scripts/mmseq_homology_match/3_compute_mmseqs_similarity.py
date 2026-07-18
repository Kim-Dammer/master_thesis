#!/usr/bin/env python3
"""
Post-process mmseqs2 easy-search output (with qaln/taln columns) to compute,
for each query protein x PDB hit pair:
 - identity_percent                : exact-match residues / full query length (original metric)
 - similarity_percent              : (matches + BLOSUM62-positive subs) / full query length
 - blast_identity_percent          : exact-match residues / aligned length, gaps included (standard/BLAST convention)
 - blast_similarity_percent        : (matches + BLOSUM62-positive subs) / aligned length, gaps included
 - alnlen, qlen, tlen              : aligned length (incl. gaps), query length, target length
 - is_high_homology                : identity_percent > threshold (original qlen-based metric)
 - high_homology_blast             : blast_identity_percent > threshold (standard alnlen-based metric)
 - high_homology_blast_filtered    : high_homology_blast AND alnlen >= min-aln-len
                                      (guards against short, coincidental high-identity matches)

uv run 3_compute_mmseqs_similarity.py \
  --input /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_max_sensitivity/mmseqs_results.parquet \
  --output /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_max_sensitivity/results/mmseqs_new_identity_similarity_max_sensitivity.parquet \
  --summary /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_max_sensitivity/results/mmseqs_new_per_protein_summary_max_sensitivity.parquet
"""

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from Bio.Align import substitution_matrices

# ── Pre-build a 128×128 ASCII-indexed BLOSUM62 score matrix ────────────
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
    ("protein_id",                    pa.string()),
    ("hit_pdb_id",                     pa.string()),
    ("identity_percent",               pa.float64()),  # matches / qlen (original metric)
    ("similarity_percent",             pa.float64()),  # matches+similar / qlen (original metric)
    ("blast_identity_percent",         pa.float64()),  # matches / alnlen, gaps included (standard/BLAST)
    ("blast_similarity_percent",       pa.float64()),  # matches+similar / alnlen, gaps included (standard/BLAST)
    ("evalue",                         pa.float64()),
    ("alnlen",                         pa.int32()),
    ("qlen",                           pa.int32()),
    ("tlen",                           pa.int32()),
    ("is_high_homology",               pa.bool_()),    # original qlen-based metric
    ("high_homology_blast",            pa.bool_()),    # standard alnlen-based metric
    ("high_homology_blast_filtered",   pa.bool_()),    # + minimum alignment length cutoff
])

BATCH_SIZE = 100_000


def compute_identity_similarity(qaln: str, taln: str, qlen: int) -> tuple[float, float, float, float, int]:
    """
    qaln/taln are already-aligned (same length, '-' = gap) strings from mmseqs2.
    Returns (identity_percent, similarity_percent, blast_identity_percent,
              blast_similarity_percent, total_aligned_len).

    - identity_percent / similarity_percent: normalized by full query length (qlen)
    - blast_identity_percent / blast_similarity_percent: normalized by total aligned
      length INCLUDING gap columns (len(qaln)), following BLAST/mmseqs' own alnlen convention
    """
    q = np.frombuffer(qaln.upper().encode("ascii"), dtype=np.uint8)
    t = np.frombuffer(taln.upper().encode("ascii"), dtype=np.uint8)

    non_gap = (q != _DASH) & (t != _DASH)
    matches = int(np.count_nonzero(non_gap & (q == t)))
    similar = int(np.count_nonzero(non_gap & (_SCORE[q, t] > 0)))

    total_aligned_len = len(q)  # includes gap columns — standard BLAST/mmseqs alnlen convention

    identity = (matches / qlen) * 100 if qlen else 0.0
    similarity = (similar / qlen) * 100 if qlen else 0.0

    blast_identity = (matches / total_aligned_len) * 100 if total_aligned_len else 0.0
    blast_similarity = (similar / total_aligned_len) * 100 if total_aligned_len else 0.0

    return identity, similarity, blast_identity, blast_similarity, total_aligned_len


def main():
    parser = argparse.ArgumentParser(
        description="Compute identity/similarity from mmseqs2 alignments (Parquet in → Parquet out)"
    )
    parser.add_argument("--input", required=True, help="mmseqs2 output Parquet file (with qaln/taln)")
    parser.add_argument("--output", required=True, help="Output Parquet: one row per (protein, hit) pair")
    parser.add_argument("--summary", required=True, help="Output Parquet: one row per protein (best hit)")
    parser.add_argument("--identity-threshold", type=float, default=30.0,
                        help="Identity %% threshold for homology flags (default: 30.0)")
    parser.add_argument("--min-aln-len", type=int, default=30,
                        help="Minimum alignment length (aa, incl. gaps) for high_homology_blast_filtered "
                             "(default: 30, guards against short coincidental matches)")
    args = parser.parse_args()

    threshold = args.identity_threshold
    min_aln_len = args.min_aln_len

    # ── Stream the input Parquet in batches ────────────────────────────
    pf = pq.ParquetFile(args.input)
    n_total = pf.metadata.num_rows
    print(f"Input: {args.input} ({n_total:,} rows)", flush=True)

    writer = pq.ParquetWriter(args.output, SCHEMA, compression="snappy")

    batch = []
    total_rows = 0
    best_per_protein = {}  # protein_id -> best row dict (by blast_identity_percent)

    for arrow_batch in pf.iter_batches(batch_size=BATCH_SIZE):
        cols = {name: arrow_batch.column(name).to_pylist()
                for name in ("query", "target", "evalue", "alnlen",
                             "qlen", "tlen", "qaln", "taln")}

        for i in range(len(cols["query"])):
            protein_id = cols["query"][i]
            hit_pdb_id = cols["target"][i]
            qlen = cols["qlen"][i]
            qaln = cols["qaln"][i]
            taln = cols["taln"][i]

            identity, similarity, blast_identity, blast_similarity, total_aligned_len = (
                compute_identity_similarity(qaln, taln, qlen)
            )

            is_blast_homology = blast_identity > threshold

            out_row = {
                "protein_id":                    protein_id,
                "hit_pdb_id":                     hit_pdb_id,
                "identity_percent":               round(identity, 2),
                "similarity_percent":             round(similarity, 2),
                "blast_identity_percent":         round(blast_identity, 2),
                "blast_similarity_percent":       round(blast_similarity, 2),
                "evalue":                         float(cols["evalue"][i]),
                "alnlen":                         int(cols["alnlen"][i]),
                "qlen":                           int(qlen),
                "tlen":                           int(cols["tlen"][i]),
                "is_high_homology":               identity > threshold,
                "high_homology_blast":            is_blast_homology,
                "high_homology_blast_filtered":   is_blast_homology and total_aligned_len >= min_aln_len,
            }
            batch.append(out_row)
            total_rows += 1

            # Track best hit per protein (ranked by blast_identity_percent)
            current_best = best_per_protein.get(protein_id)
            if current_best is None or blast_identity > current_best["blast_identity_percent"]:
                best_per_protein[protein_id] = out_row

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
    n_with_homology_blast = int(df_summary["high_homology_blast"].sum())
    n_with_homology_blast_filtered = int(df_summary["high_homology_blast_filtered"].sum())
    print(f"Processed {total_rows:,} hits across {n_proteins} proteins.")
    print(f"{n_with_homology}/{n_proteins} proteins have a best hit above {threshold}% identity (qlen-based).")
    print(f"{n_with_homology_blast}/{n_proteins} proteins have a best hit above {threshold}% BLAST identity (alnlen-based).")
    print(f"{n_with_homology_blast_filtered}/{n_proteins} proteins have a best hit above {threshold}% BLAST identity "
          f"AND alignment length >= {min_aln_len} aa (filtered).")
    print(f"Per-hit results:   {args.output}")
    print(f"Per-protein summary: {args.summary}")


if __name__ == "__main__":
    main()