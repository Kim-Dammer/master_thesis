#!/usr/bin/env python
"""
coverage_sqlite.py — Robustly + efficiently test whether pair/pool PDBs are
retrievable from the new models.sqlite backend for all proteins.

Two layers of robustness:
  1. MEMBERSHIP: fast point-lookups (db_id is indexed -> ~0.1s each, batched
     in one connection) for every needed key. No full-table scan, no blobs.
  2. RETRIEVAL PROOF: for a few sample proteins, actually call
     yp.save_predictions_db and confirm a valid multi-chain PDB is written.
     Membership can lie if a blob is corrupt; this proves real retrieval.

Usage:
    uv run coverage_sqlite.py --from-file /cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_CP_proteins.csv --out coverage.csv
"""
import argparse, sqlite3, sys, time, os
import polars as pl

DATA = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07"
PARQUET = f"{DATA}/summary_models.parquet"
SQLITE = f"{DATA}/models.sqlite"

ap = argparse.ArgumentParser()
ap.add_argument("--from-file", required=True)
ap.add_argument("--out", default="coverage.csv")
ap.add_argument("--proof-n", type=int, default=5, help="proteins to fully retrieve as proof")
args = ap.parse_args()

prots = [l.strip().lower() for l in open(args.from_file)
         if l.strip() and not l.lower().startswith("uniprot")]
prots = list(dict.fromkeys(prots))
print(f"Checking {len(prots)} proteins...")

# --- self-pair rows from parquet; synthesize db_id keys as the package does ---
rows = (pl.scan_parquet(PARQUET)
    .filter(pl.col("af3_id1").is_in(prots) & (pl.col("af3_id1") == pl.col("af3_id2")))
    .with_columns(
        (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8) + "_" + pl.col("chain_id1")).alias("k1"),
        (pl.col("input_name") + "_" + pl.col("sample").cast(pl.Utf8) + "_" + pl.col("chain_id2")).alias("k2"))
    .select("af3_id1", "input_name", "input_type", "sample", "k1", "k2")
    .collect())
if rows.height == 0:
    sys.exit("No self-pair rows found for requested proteins.")

needed = sorted(set(rows["k1"].to_list() + rows["k2"].to_list()))
print(f"Probing {len(needed):,} db_id keys via indexed point-lookups...")

# --- LAYER 1: fast membership via batched point-lookups (db_id is indexed) ---
t = time.time()
con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
found = set()
CH = 900  # SQLite max variables ~999; stay under
for i in range(0, len(needed), CH):
    chunk = needed[i:i+CH]
    q = f"SELECT db_id FROM models WHERE db_id IN ({','.join('?'*len(chunk))})"
    found.update(r[0] for r in con.execute(q, chunk))
    if (i // CH) % 10 == 0:
        print(f"  {i+len(chunk):,}/{len(needed):,} probed, {len(found):,} found ({time.time()-t:.0f}s)")
con.close()
print(f"Found {len(found):,} of {len(needed):,} keys ({time.time()-t:.0f}s)")

# --- aggregate: pair/pool/missing per protein ---
kl = list(found)
rows = rows.with_columns((pl.col("k1").is_in(kl) & pl.col("k2").is_in(kl)).alias("ok"))
agg = rows.group_by(["af3_id1", "input_type"]).agg(pl.col("ok").sum().alias("n_ok"))
pair = agg.filter(pl.col("input_type") == "pair").select("af3_id1", pl.col("n_ok").alias("pair_ok"))
pool = agg.filter(pl.col("input_type") == "pool").select("af3_id1", pl.col("n_ok").alias("pool_ok"))
summ = (pair.join(pool, on="af3_id1", how="full", coalesce=True)
        .with_columns(pl.col("pair_ok").fill_null(0), pl.col("pool_ok").fill_null(0)))
summ = summ.with_columns(
    pl.when(pl.col("pair_ok") > 0).then(pl.lit("pair_available"))
     .when(pl.col("pool_ok") > 0).then(pl.lit("pool_fallback"))
     .otherwise(pl.lit("missing")).alias("verdict")).sort("af3_id1")
summ.write_csv(args.out)

n = summ.height
print(f"\n=== MEMBERSHIP RESULT (all {n} proteins) ===")
for v in ("pair_available", "pool_fallback", "missing"):
    c = summ.filter(pl.col("verdict") == v).height
    print(f"  {v:16s}: {c:5d} ({100*c/n:.1f}%)")
print(f"Wrote {args.out}")

# --- LAYER 2: retrieval proof on a few pair_available proteins ---
proof = summ.filter(pl.col("pair_ok") > 0)["af3_id1"].to_list()[:args.proof_n]
if proof:
    print(f"\n=== RETRIEVAL PROOF (actually writing PDBs for {len(proof)} proteins) ===")
    import pooled_ppi.yeast_pools as yp
    os.makedirs("proof_pdbs", exist_ok=True)
    for pid in proof:
        r = rows.filter((pl.col("af3_id1") == pid) & (pl.col("input_type") == "pair")
                        & (pl.col("sample") == 0)).row(0, named=True)
        out = f"proof_pdbs/{pid}_pair_0.pdb"
        try:
            yp.save_predictions_db([r["k1"], r["k2"]], out)
            sz = os.path.getsize(out)
            nchains = len(set(l[21] for l in open(out) if l.startswith("ATOM")))
            print(f"  {pid}: OK  {sz:,} bytes, {nchains} chains")
        except Exception as e:
            print(f"  {pid}: FAILED — {type(e).__name__}: {e}")
