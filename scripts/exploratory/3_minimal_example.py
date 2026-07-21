#!/usr/bin/env python
"""
diag_pair_ingestion.py — Check if a protein's pair structures are in foldcomp.

Usage: python diag_pair_ingestion.py P00445
       python diag_pair_ingestion.py O13297 P00445 Q01217
"""
import sys
import polars as pl
import foldcomp

DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
FC = f"{DB}/predictions-db/predictions-db"
PQ = f"{DB}/summary_models.parquet"


def probe(keys):
    found = set()
    with foldcomp.open(FC, ids=list(keys)) as db:
        for name, _ in db:
            found.add(name)
    return found


def get_ids(protein, input_type, sample=0):
    df = (
        pl.scan_parquet(PQ)
        .filter((pl.col("af3_id1") == protein) & (pl.col("af3_id2") == protein)
                & (pl.col("input_type") == input_type) & (pl.col("sample") == sample))
        .with_columns(
            (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8)
             + "_" + pl.col("chain_id1")).alias("db_id1"),
            (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8)
             + "_" + pl.col("chain_id2")).alias("db_id2"),
        )
        .select("db_id1", "db_id2", "ranking_score", "chain_pair_iptm")
        .collect()
    )
    return df.row(0, named=True) if df.height else None


def show(protein):
    p = get_ids(protein, "pair")
    pool = get_ids(protein, "pool")
    if not p:
        print(f"{protein}: no pair row in parquet")
        return
    # Probe all case variants of pair keys to rule out capitalization
    variants = set()
    for k in (p["db_id1"], p["db_id2"]):
        variants |= {k, k.upper(), k.lower(), k.capitalize()}
    pair_found = bool(probe(variants))
    pool_found = bool(probe([pool["db_id1"], pool["db_id2"]])) if pool else False
    print(f"{protein}: pair score={p['ranking_score']:.3f} | "
          f"pair structure: {'FOUND' if pair_found else 'MISSING'} | "
          f"pool structure: {'FOUND' if pool_found else 'missing'}")


if __name__ == "__main__":
    if not sys.argv[1:]:
        sys.exit("Usage: python diag_pair_ingestion.py P00445 [P00446 ...]")
    for p in sys.argv[1:]:
        show(p.lower())
