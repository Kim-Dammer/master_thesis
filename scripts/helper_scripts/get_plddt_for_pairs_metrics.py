#!/usr/bin/env python
"""
get_plddt_for_pairs_metrics.py — Add per-chain pLDDT to a pairs-metrics CSV.

AF3's summary_confidences.json (source of ranking_score, chain_pair_iptm,
chain_pair_pae_min_*, already in *_pairs_metrics.csv) has no per-residue
confidence. pLDDT lives as the B-factor column of the actual predicted PDB
structures.


KEY FORMAT CHANGED. It's now:
    db_id = f"{input_name}_{sample}_{chain_id}"
(input_name, sample, chain_id1/chain_id2 -- all already columns in
*_pairs_metrics.csv, so db_id1/db_id2 can be built directly, no need to
re-query yp.get_models()).

Usage:
    uv run get_plddt_for_pairs_metrics.py \
        --pairs-csv /path/to/7_benchmark_part_one_pool_pairs_metrics.csv \
        --out-csv   /path/to/7_benchmark_part_one_pool_pairs_metrics_with_plddt.csv
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import foldcomp
from Bio.PDB import PDBParser

# Single source of truth for where models.sqlite lives -- reuse the library's
# own path resolution instead of re-deriving/hardcoding it.
import pooled_ppi.yeast_pools as yp
import sqlite_utils

MODELS_TABLE = "models"
MODELS_DB_FILENAME = "models.sqlite"

BIN_LABELS = ["<50", "50_60", "60_70", "70_80", "80_90", "ge90"]
BIN_EDGES = [0, 50, 60, 70, 80, 90, 100.01]

CHUNK_SIZE = 500  # ids per SQL "IN (...)" batch


def compute_plddt_stats(plddt_vals: list[float]):
    if not plddt_vals:
        return None, [0] * 6, 0
    arr = np.array(plddt_vals, dtype=float)
    counts = []
    for b in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        if b == len(BIN_EDGES) - 2:
            counts.append(int(np.sum(arr >= lo)))
        else:
            counts.append(int(np.sum((arr >= lo) & (arr < hi))))
    return float(np.mean(arr)), counts, len(arr)


def open_models_db() -> sqlite_utils.Database:
    path = yp.get_path(MODELS_DB_FILENAME)
    if not path.is_file():
        sys.exit(f"[get-plddt] {MODELS_DB_FILENAME} not found at {path}")
    print(f"[get-plddt] using models db: {path}")
    return sqlite_utils.Database(path)


def fetch_plddt_for_ids(db: sqlite_utils.Database, ids: list[str]) -> dict[str, tuple]:
    """Batched primary-key lookup + foldcomp decompress + CA B-factor extraction.

    Uses chunked "WHERE id IN (...)" queries rather than one .get() call per id
    (the convenience method used in the library's own examples) -- meaningfully
    faster once you're doing thousands of lookups against a 124GB db.
    """
    parser = PDBParser(QUIET=True)
    out: dict[str, tuple] = {}
    table = db[MODELS_TABLE]
    pk_col = table.pks[0] if table.pks else "id"

    for i in range(0, len(ids), CHUNK_SIZE):
        chunk = ids[i:i + CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        rows = db.execute(
            f"SELECT {pk_col}, fcz FROM {MODELS_TABLE} WHERE {pk_col} IN ({placeholders})",
            chunk,
        ).fetchall()
        found_this_chunk = set()
        for id_, blob in rows:
            found_this_chunk.add(id_)
            try:
                db_id, pdb_str = foldcomp.decompress(blob)
                assert db_id == id_, f"key mismatch: requested {id_}, got {db_id}"
                struct = parser.get_structure(id_, io.StringIO(pdb_str))
                chain = next(struct[0].get_chains())
                plddt_vals = [r["CA"].get_bfactor() for r in chain.get_residues() if "CA" in r]
                out[id_] = compute_plddt_stats(plddt_vals)
            except Exception as exc:
                print(f"  [warn] failed to parse {id_}: {exc}", file=sys.stderr)
                out[id_] = (None, [0] * 6, 0)
        missing = set(chunk) - found_this_chunk
        if missing:
            print(f"  [warn] {len(missing)} id(s) not found in {MODELS_TABLE}: "
                  f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}",
                  file=sys.stderr)
        if (i // CHUNK_SIZE) % 20 == 0:
            print(f"  ... {min(i + CHUNK_SIZE, len(ids))}/{len(ids)} ids processed")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    required = ("input_name", "sample", "chain_id1", "chain_id2")
    for col in required:
        if col not in df.columns:
            sys.exit(f"[get-plddt] required column '{col}' missing from {args.pairs_csv}")

    df["db_id1"] = (
        df["input_name"].astype(str) + "_" + df["sample"].astype(str) + "_" + df["chain_id1"].astype(str)
    )
    df["db_id2"] = (
        df["input_name"].astype(str) + "_" + df["sample"].astype(str) + "_" + df["chain_id2"].astype(str)
    )

    db = open_models_db()
    all_ids = sorted(set(df["db_id1"]) | set(df["db_id2"]))
    print(f"[get-plddt] {len(df)} rows -> {len(all_ids)} unique db_id(s) to fetch")

    stats = fetch_plddt_for_ids(db, all_ids)

    def _lookup(key: str, idx: int):
        s = stats.get(key)
        return s[idx] if s is not None else None

    df["plddt1"] = df["db_id1"].apply(lambda k: _lookup(k, 0))
    df["plddt2"] = df["db_id2"].apply(lambda k: _lookup(k, 0))
    df["n_residues1"] = df["db_id1"].apply(lambda k: _lookup(k, 2))
    df["n_residues2"] = df["db_id2"].apply(lambda k: _lookup(k, 2))
    df["mean_plddt_simple"] = (df["plddt1"] + df["plddt2"]) / 2
    denom = (df["n_residues1"] + df["n_residues2"]).replace(0, np.nan)
    df["mean_plddt_weighted"] = (
        df["plddt1"] * df["n_residues1"] + df["plddt2"] * df["n_residues2"]
    ) / denom

    df = df.drop(columns=["db_id1", "db_id2"])
    df.to_csv(args.out_csv, index=False)

    n_valid = df["mean_plddt_weighted"].notna().sum()
    print(f"[get-plddt] wrote {args.out_csv} ({len(df)} rows); "
          f"pLDDT coverage {n_valid}/{len(df)} ({100 * n_valid / len(df):.1f}%)")


if __name__ == "__main__":
    main()
