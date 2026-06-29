import pooled_ppi

DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.04"
print("Loading DB from " + DB + " ...")
pp = pooled_ppi.PooledPredictionsDb(DB)

print("=== ALL columns in pp.pairs ===")
for col in pp.pairs.columns:
    print("  " + col)

print("=== iptm-related columns ===")
iptm_cols = [c for c in pp.pairs.columns if "iptm" in c.lower()]
for c in iptm_cols:
    print("  " + c)

print("=== length-related columns ===")
len_cols = [c for c in pp.pairs.columns if "length" in c.lower() or "len" in c.lower()]
for c in len_cols:
    print("  " + c)

print("=== corrected-related columns ===")
corr_cols = [c for c in pp.pairs.columns if "correct" in c.lower()]
for c in corr_cols:
    print("  " + c)

print("Done.")