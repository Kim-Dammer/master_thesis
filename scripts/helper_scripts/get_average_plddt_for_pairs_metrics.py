#!/usr/bin/env python
"""
get_plddt_for_pairs_metrics.py

Adds per-row, per-model pLDDT (plddt1, plddt2) to *_pairs_metrics.csv.

pLDDT = mean B-factor of CA atoms in the predicted structure for that
protein, in that specific pool/pair prediction (input_name + sample +
chain_id -- the exact model your CSV's ranking_score/iptm/pae already refer
to). Structures are blobs in models.sqlite (table 'models', col 'fcz',
foldcomp-compressed), keyed as f"{input_name}_{sample}_{chain_id}"
(see pooled_ppi.yeast_pools.get_models / load_predictions_db).

Also asserts that batch_id + sample in the input CSV still match the current
summary_models.parquet, i.e. that the model being scored is the same one the
CSV's other metrics were computed from.

Requires: pip install -U "pooled-ppi @ git+https://github.com/jurgjn/pooled-ppi@develop"

Usage:
    uv run get_average_plddt_for_pairs_metrics.py --pairs-csv /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/8_benchmark_part_two/combined_pool_pairs_metrics.csv --out-csv /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/8_benchmark_part_two/combined_pool_pairs_metrics_with_plddt.csv
"""

from __future__ import annotations
 
import argparse
import io
import os
from pathlib import Path
 
import numpy as np
import pandas as pd
import foldcomp
from Bio.PDB import PDBParser
from tqdm import tqdm
import sqlite_utils

# pooled_ppi.yeast_pools.get_data() hardcodes a fallback to data-26.07 when
# none of /data, /workspace/data, /contents/data exist (true on Euler), so
# yp.get_path()/yp.get_models() silently resolve to the WRONG snapshot here.
# Bypass pooled_ppi entirely for path resolution; only the DATA_DIR below
# needs updating if the snapshot changes again.
DATA_DIR = Path(os.environ.get(
    "POOLED_PPI_DATA_DIR",
    "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08"))

CHUNK = 500
JOIN_KEYS = ["input_name", "sample", "chain_id1", "chain_id2"]


def verify_metadata(df: pd.DataFrame) -> None:
    """Assert df's batch_id/sample still match the current summary_models.parquet."""
    input_names = sorted(df["input_name"].astype(str).unique())
    print(f"[verify] reading summary_models.parquet, filtering to {len(input_names)} input_name(s) ...")
    current = pd.read_parquet(
        DATA_DIR / "summary_models.parquet",
        filters=[("input_name", "in", input_names)],
    )
    print(f"[verify] got {len(current)} matching row(s), checking batch_id ...")
    assert "batch_id" in current.columns, "current summary_models.parquet has no batch_id column"
 
    merged = df.merge(current[JOIN_KEYS + ["batch_id"]], on=JOIN_KEYS,
                       how="left", suffixes=("", "_current"))
    n_missing = merged["batch_id_current"].isna().sum()
    assert n_missing == 0, f"{n_missing} row(s) not found in current metadata"
    n_mismatch = (merged["batch_id_current"] != merged["batch_id"]).sum()
    assert n_mismatch == 0, f"{n_mismatch} row(s) have a batch_id mismatch vs current metadata"
    print("[verify] OK, all rows match current metadata.")
 
 
def mean_plddt(pdb_str: str) -> float:
    struct = PDBParser(QUIET=True).get_structure("x", io.StringIO(pdb_str))
    chain = next(struct[0].get_chains())
    vals = [r["CA"].get_bfactor() for r in chain.get_residues() if "CA" in r]
    assert vals, "no CA atoms found"
    return float(np.mean(vals))
 
 
def fetch_plddt(db: sqlite_utils.Database, ids: list[str]) -> dict[str, float]:
    pk = db["models"].pks[0]
    out: dict[str, float] = {}
    for i in tqdm(range(0, len(ids), CHUNK), desc="fetching pLDDT", unit="chunk"):
        chunk = ids[i:i + CHUNK]
        rows = db.execute(
            f"SELECT {pk}, fcz FROM models WHERE {pk} IN ({','.join('?' * len(chunk))})", chunk
        ).fetchall()
        assert len(rows) == len(chunk), f"{len(chunk) - len(rows)} id(s) not found in models table"
        for id_, blob in rows:
            db_id, pdb_str = foldcomp.decompress(blob)
            assert db_id == id_, f"key mismatch: requested {id_}, got {db_id}"
            out[id_] = mean_plddt(pdb_str)
    return out
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--skip-verify", action="store_true",
                     help="Skip the batch_id/sample cross-check against summary_models.parquet "
                          "(that full-table read can be slow; skip once you've confirmed it's clean).")
    args = ap.parse_args()
 
    df = pd.read_csv(args.pairs_csv)
    for col in JOIN_KEYS + ["batch_id"]:
        assert col in df.columns, f"required column '{col}' missing from {args.pairs_csv}"
 
    if not args.skip_verify:
        verify_metadata(df)
 
    df["db_id1"] = df["input_name"].astype(str) + "_" + df["sample"].astype(str) + "_" + df["chain_id1"].astype(str)
    df["db_id2"] = df["input_name"].astype(str) + "_" + df["sample"].astype(str) + "_" + df["chain_id2"].astype(str)
 
    models_path = DATA_DIR / "models.sqlite"
    assert models_path.is_file(), f"models.sqlite not found at {models_path}"
    db = sqlite_utils.Database(models_path)
 
    all_ids = sorted(set(df["db_id1"]) | set(df["db_id2"]))
    stats = fetch_plddt(db, all_ids)
 
    df["plddt1"] = df["db_id1"].map(stats)
    df["plddt2"] = df["db_id2"].map(stats)
    df.drop(columns=["db_id1", "db_id2"]).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} rows)")
 
 
if __name__ == "__main__":
    main()
 