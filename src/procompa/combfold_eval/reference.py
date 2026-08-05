"""Reference structure acquisition and reference-form selection.

For each reference PDB id we obtain BOTH the asymmetric unit and every
biological assembly from RCSB, then choose which form to treat as primary by a
rule fixed BEFORE any metric is computed (so we never cherry-pick the form that
happens to score best):

  1. best UniProt-composition match to the model  (fewest missing, then fewest
     extra subunit copies);
  2. tie -> prefer a biological assembly over the asymmetric unit;
  3. tie -> the form with more resolved residues.

All forms are still scored and reported; only `is_primary` differs. Restricting
to a single biological unit is what makes the reference-normalized TM-score
meaningful (scoring a 3-chain model against a 6-chain asymmetric unit deflates
the reference-normalized number for no biological reason).
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Config
from .structure_utils import load_and_clean, CleanStructure
from . import mapping as M

# Per-pdb_id locks so concurrent batch workers scoring different complexes that
# happen to share a reference PDB id don't race on the same ref_cache/<pdb_id>/
# files (check-exists-then-download-then-write is not otherwise atomic). The
# meta-lock only guards creation of a new per-id lock, not the download itself.
_ref_locks: Dict[str, threading.Lock] = {}
_ref_locks_meta = threading.Lock()


def _lock_for(pdb_id: str) -> threading.Lock:
    with _ref_locks_meta:
        lock = _ref_locks.get(pdb_id)
        if lock is None:
            lock = threading.Lock()
            _ref_locks[pdb_id] = lock
        return lock


# Thread-safe cache-hit/download counters, purely for rerun visibility (does not
# affect any decision -- the exists+non-empty check above is unchanged). Reset at
# the start of each run_batch() call and reported as one summary line at the end,
# so you can visually confirm a rerun is reusing the persistent cache rather than
# re-fetching.
_cache_stats = {"hits": 0, "downloads": 0, "failed": 0}
_cache_stats_lock = threading.Lock()


def reset_cache_stats() -> None:
    with _cache_stats_lock:
        _cache_stats["hits"] = 0
        _cache_stats["downloads"] = 0
        _cache_stats["failed"] = 0


def get_cache_stats() -> Dict[str, int]:
    with _cache_stats_lock:
        return dict(_cache_stats)


def _record(outcome: str) -> None:
    with _cache_stats_lock:
        _cache_stats[outcome] += 1


@dataclass
class RefForm:
    form_id: str                 # "assembly_1" | "asymmetric_unit" | ...
    is_assembly: bool
    path: str
    clean: Optional[CleanStructure] = None
    assign: Dict = field(default_factory=dict)
    comp: Dict[str, int] = field(default_factory=dict)
    n_chains: int = 0
    n_shared: int = 0
    n_missing: int = 0
    n_extra: int = 0
    total_resolved: int = 0
    is_primary: bool = False
    select_reason: str = ""


def _download(url: str, dest: str, timeout: int) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        if not data:
            return False
        with open(dest, "wb") as fh:
            fh.write(data)
        return True
    except Exception:
        return False


def _assembly_ids(pdb_id: str, cfg: Config) -> List[str]:
    url = f"{cfg.rcsb_data_api}/{pdb_id.upper()}"
    try:
        with urllib.request.urlopen(url, timeout=cfg.download_timeout_s) as r:
            data = json.load(r)
        ids = data.get("rcsb_entry_container_identifiers", {}).get("assembly_ids") or []
        return [str(a) for a in ids]
    except Exception:
        return []


def acquire_reference_files(pdb_id: str, cfg: Config,
                            local_au: Optional[str] = None) -> Dict[str, str]:
    """Return {form_id: file_path} for the AU and each biological assembly.

    Files are cached under `ref_cache/<pdb_id>/`. `local_au`, if given, is used
    for the asymmetric-unit form instead of downloading it.
    """
    pdb_id = pdb_id.lower()
    cache = os.path.join(cfg.ref_cache, pdb_id)
    forms: Dict[str, str] = {}

    with _lock_for(pdb_id):
        os.makedirs(cache, exist_ok=True)

        # asymmetric unit
        au_path = os.path.join(cache, f"{pdb_id}.cif")
        if local_au and os.path.exists(local_au):
            forms["asymmetric_unit"] = local_au  # local override, not a cache hit/download
        else:
            if os.path.exists(au_path) and os.path.getsize(au_path) > 0:
                forms["asymmetric_unit"] = au_path
                _record("hits")
            elif _download(f"{cfg.rcsb_files}/{pdb_id}.cif", au_path, cfg.download_timeout_s):
                forms["asymmetric_unit"] = au_path
                _record("downloads")
            else:
                _record("failed")

        # biological assemblies
        for aid in _assembly_ids(pdb_id, cfg):
            ap = os.path.join(cache, f"{pdb_id}-assembly{aid}.cif")
            if os.path.exists(ap) and os.path.getsize(ap) > 0:
                forms[f"assembly_{aid}"] = ap
                _record("hits")
            elif _download(f"{cfg.rcsb_files}/{pdb_id}-assembly{aid}.cif", ap, cfg.download_timeout_s):
                forms[f"assembly_{aid}"] = ap
                _record("downloads")
            else:
                _record("failed")
    return forms


def build_and_select(pdb_id: str, model_comp: Dict[str, int], candidates: Dict[str, str],
                     cfg: Config, local_au: Optional[str] = None) -> List[RefForm]:
    """Acquire, clean, assign and rank all reference forms; mark one primary."""
    files = acquire_reference_files(pdb_id, cfg, local_au=local_au)
    forms: List[RefForm] = []
    for form_id, path in files.items():
        clean = load_and_clean(path, cfg)
        assign = M.assign_all(clean, candidates, cfg)
        comp = M.composition(assign)
        shared, missing, extra = M.composition_match_score(model_comp, comp)
        total_resolved = sum(len(clean.chains[c].seq)
                             for c, a in assign.items() if a.uniprot)
        forms.append(RefForm(
            form_id=form_id,
            is_assembly=form_id.startswith("assembly"),
            path=path, clean=clean, assign=assign, comp=comp,
            n_chains=len(clean.protein_chains()),
            n_shared=shared, n_missing=missing, n_extra=extra,
            total_resolved=total_resolved,
        ))
    if not forms:
        return forms
    # rank: fewest missing, fewest extra, assemblies first, most resolved
    forms.sort(key=lambda f: (f.n_missing, f.n_extra, 0 if f.is_assembly else 1,
                              -f.total_resolved))
    best = forms[0]
    best.is_primary = True
    best.select_reason = (
        f"best composition match (missing={best.n_missing}, extra={best.n_extra}); "
        + ("biological assembly" if best.is_assembly else "asymmetric unit")
        + f"; resolved={best.total_resolved}"
    )
    return forms
