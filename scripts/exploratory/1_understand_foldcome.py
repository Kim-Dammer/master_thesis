import polars as pl
import foldcomp
from pathlib import Path

import foldcomp
import pooled_ppi.yeast_pools as yp


FC_DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.07/predictions-db/predictions-db"
ids = [f"p00445_p00445_{s}_{c}" for s in range(5) for c in ["A", "B"]]

found = []
with foldcomp.open(FC_DB, ids=ids) as db:
    for name, pdb in db:
        found.append(name)

print(f"{len(found)}/{len(ids)} found:", found)
