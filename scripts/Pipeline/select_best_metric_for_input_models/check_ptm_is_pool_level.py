#!/usr/bin/env python3
"""
Quick sanity check: for a 'pool'-type AF3 sample that yields several
different pairwise contacts, does ptm collapse to ONE value shared by all
of them, while chain_pair_iptm_corrected / chain_pair_pae_min vary per
contact?

Uses DuckDB directly on the parquet files (no full load, pushdown filters) --
should run in seconds even against the full 26.08 snapshot.

Usage: uv run python check_ptm_is_pool_level.py
"""
from pathlib import Path

import duckdb

POOLED_PPI_DB = Path("/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08")
MODELS = POOLED_PPI_DB / "summary_models.parquet"
CONFIDENCES = POOLED_PPI_DB / "summary_confidences.parquet"

con = duckdb.connect()

# 1) Find a 'pool' sample that produced several distinct pairwise contacts.
top_pool = con.execute(f"""
    SELECT input_name, input_type, seed, sample, count(*) AS n_contacts
    FROM read_parquet('{MODELS}')
    WHERE input_type = 'pool'
    GROUP BY input_name, input_type, seed, sample
    ORDER BY n_contacts DESC
    LIMIT 1
""").fetchone()
input_name, input_type, seed, sample, n_contacts = top_pool
print(f"Picked pool sample: input_name={input_name!r} seed={seed} sample={sample} "
      f"-> {n_contacts} distinct pairwise contacts extracted from it.\n")

# 2) Pull every contact from that one sample, plus its (single) ptm joined in.
rows = con.execute(f"""
    SELECT m.af3_id1, m.af3_id2,
           m.chain_pair_iptm_corrected, m.chain_pair_pae_min,
           c.ptm
    FROM read_parquet('{MODELS}') m
    JOIN read_parquet('{CONFIDENCES}') c
      ON m.input_name = c.input_name AND m.input_type = c.input_type
     AND m.seed = c.seed AND m.sample = c.sample
    WHERE m.input_name = ? AND m.input_type = ? AND m.seed = ? AND m.sample = ?
    ORDER BY m.chain_pair_iptm_corrected DESC
""", [input_name, input_type, seed, sample]).fetchdf()

print(rows.to_string(index=False))
print(f"\nDistinct ptm values across these {len(rows)} contacts: {rows['ptm'].nunique()}")
print(f"Distinct chain_pair_iptm_corrected values: {rows['chain_pair_iptm_corrected'].nunique()}")
print(f"Distinct chain_pair_pae_min values: {rows['chain_pair_pae_min'].nunique()}")
