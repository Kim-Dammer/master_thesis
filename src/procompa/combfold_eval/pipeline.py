"""Batch driver: run the CombFold-vs-reference comparison over a mapping file.

Input mapping file (CSV) contract
---------------------------------
Required columns:
  complex_ac   complex identifier (e.g. CPX-2800); used to locate CombFold output
  pdb_id       best-match reference PDB id (e.g. 2qlv). Use "SELF" or leave the
               `local_ref` column set to score against a local file only.
Optional columns (all improve robustness; none are mandatory):
  matching_hits   "uniprot:pdb_chain,..." — used only as a fallback UniProt source
  folder          explicit CombFold output folder (abs, or relative to combfold_base)
  model           explicit model PDB path or glob (overrides folder discovery)
  uniprots        explicit UniProt list ("P1;P2" or "P1x2,P3") for this complex
  local_ref       local reference structure file (used as the asymmetric unit)

How each complex is resolved
-----------------------------
  * CombFold models: `model` glob if given, else
    `<combfold_base>/<folder-or-auto>/assembled_results/output_clustered_*.pdb`
    (auto = first directory whose name contains the complex_ac), else a flat
    `<combfold_base>/<complex_ac>*.pdb` search. All clusters scored by default.
  * CombFold confidence: `assembled_results/confidence.txt` (both known formats).
  * Candidate UniProts (for sequence-based chain assignment): `uniprots` column,
    else parsed from the folder/model name (`..._P06782x1_P12904x1_...`), else
    from `matching_hits`. Full sequences come from the offline CSV / UniProt REST.

Outputs (written to out_dir)
----------------------------
  complex_summary.csv   one row per (complex, cluster, reference form)
  per_chain.csv         one row per matched subunit
  per_interface.csv     one row per native interface
  json/<complex>_c<k>.json  chain map + reference-form provenance + run log
  run_log.txt           per-complex status lines

Every complex is wrapped so one failure never aborts the batch; the failure is
recorded as an explicit status instead.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

from .config import Config
from .sequences import UniProtSequences
from .compare import compare_one
from . import reference as _reference
from .manifest import (
    load_manifest,
    candidate_stoichiometries,
    resolve_folders,
)

# UniProt accession pattern (official) + optional xN stoichiometry
_UNIPROT = r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
_UNIPROT_STOIC = re.compile(_UNIPROT + r"(?:[xX](\d+))?")
_ACC_ONLY = re.compile(_UNIPROT)
_CLUST = re.compile(r"output_clustered_(\d+)\.pdb")

SUMMARY_COLS = [
    "complex_ac", "pdb_id", "stoich_source", "cf_folder_type",
    "cf_cluster", "cf_rank", "cf_confidence",
    "ref_form", "ref_assembly_id", "is_primary_ref",
    "n_shared_subunits", "missing_uniprots", "extra_uniprots",
    "complex_TM_ref", "complex_TM_mod",
    "global_rmsd_nocyc", "global_rmsd_wcyc", "n_res_global", "coverage_global",
    "mean_dockq", "pairing_source", "candidate_source", "ref_select_reason",
    "status", "flags",
]
CHAIN_COLS = [
    "complex_ac", "pdb_id", "cf_cluster", "ref_form", "is_primary_ref",
    "uniprot", "chain_mod", "chain_ref", "slot",
    "tm_ref", "tm_mod", "rmsd_nocyc", "n_res_nocyc", "rmsd_wcyc", "n_res_wcyc",
    "n_res", "coverage_resolved", "coverage_fulllen", "seq_identity",
]
IFACE_COLS = [
    "complex_ac", "pdb_id", "cf_cluster", "ref_form", "is_primary_ref",
    "interface_uniprots", "iface_native_key", "chain1_mod", "chain2_mod",
    "dockq", "fnat", "irmsd", "lrmsd", "capri_class",
]


def parse_uniprots_from_name(name: str) -> "OrderedDict[str,int]":
    """Extract UniProt accessions + copy counts from a folder/file/list string."""
    out: "OrderedDict[str,int]" = OrderedDict()
    if not name:
        return out
    for m in _UNIPROT_STOIC.finditer(name.upper()):
        acc = m.group(0).split("X")[0] if m.group(1) else m.group(0)
        acc = _ACC_ONLY.match(acc).group(0) if _ACC_ONLY.match(acc) else acc
        cnt = int(m.group(1)) if m.group(1) else 1
        out[acc] = out.get(acc, 0) + cnt
    return out


def cluster_idx_from_filename(path: str) -> int:
    m = _CLUST.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def parse_confidence(path: str) -> Dict[int, Tuple[float, int]]:
    """Return {cluster_idx: (confidence, rank)}; robust to both known formats."""
    conf: Dict[int, Tuple[float, int]] = {}
    if not path or not os.path.exists(path):
        return conf
    lines = [ln.strip() for ln in open(path) if ln.strip()]
    # format A: "<...>/output_clustered_N.pdb  <score>" (one per line, rank=order)
    rank = 0
    for ln in lines:
        if "output_clustered" in ln and (" " in ln or "\t" in ln):
            toks = ln.split()
            idx = cluster_idx_from_filename(toks[0])
            try:
                score = float(toks[-1])
            except ValueError:
                score = float("nan")
            if idx >= 0:
                conf[idx] = (score, rank)
                rank += 1
    if conf:
        return conf
    # format B: "0:81.89;1:81.46"
    rank = 0
    for ln in lines:
        for part in ln.replace(",", ";").split(";"):
            if ":" in part:
                a, b = part.split(":", 1)
                try:
                    conf[int(a.strip())] = (float(b.strip()), rank)
                    rank += 1
                except ValueError:
                    continue
    return conf


def find_complex_dirs(base: str, complex_ac: str, folder_hint: Optional[str]) -> List[str]:
    """Return candidate CombFold directories for a complex, most-likely first.

    Multiple names can match (e.g. `<ac>_input` and `<ac>_output`); we return
    them all so the caller can pick the one that actually holds models. Explicit
    hints come first, then name matches, with `_output`-style folders preferred.
    """
    cands: List[str] = []
    if folder_hint and str(folder_hint) != "nan":
        for cand in (str(folder_hint), os.path.join(base, str(folder_hint))):
            if os.path.isdir(cand) and cand not in cands:
                cands.append(cand)
    norm = complex_ac.replace("-", "_").upper()
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return cands
    dirs = [e for e in entries if os.path.isdir(os.path.join(base, e))]
    starts = [e for e in dirs if e.replace("-", "_").upper().startswith(norm)]
    contains = [e for e in dirs if norm in e.replace("-", "_").upper() and e not in starts]
    # prefer output/result folders over input folders when names tie
    rank = lambda e: (0 if any(k in e.lower() for k in ("output", "result", "assembl")) else 1)
    for e in sorted(starts, key=rank) + sorted(contains, key=rank):
        full = os.path.join(base, e)
        if full not in cands:
            cands.append(full)
    return cands


def find_model_files(search_dir: str, globs) -> List[str]:
    for g in globs:
        hits = sorted(glob.glob(os.path.join(search_dir, g)))
        if hits:
            return hits
    return []


def locate_models(base: str, complex_ac: str, folder_hint: Optional[str],
                  globs) -> Tuple[Optional[str], List[str]]:
    """Find (complex_dir, model_files). Canonical CombFold `assembled_results/`
    layout is preferred across ALL candidate directories before falling back to
    looser globs, so an `<ac>_input` folder never shadows `<ac>_output`."""
    cands = find_complex_dirs(base, complex_ac, folder_hint)
    canonical = os.path.join("assembled_results", "output_clustered_*.pdb")
    for cand in cands:
        hits = sorted(glob.glob(os.path.join(cand, canonical)))
        if hits:
            return cand, hits
    for cand in cands:
        hits = find_model_files(cand, globs)
        if hits:
            return cand, hits
    return (cands[0] if cands else None), []


def _matching_hits_uniprots(val: str) -> "OrderedDict[str,int]":
    out: "OrderedDict[str,int]" = OrderedDict()
    if not val or str(val) == "nan":
        return out
    for hit in str(val).replace(";", ",").split(","):
        acc = hit.split(":")[0].strip().upper()
        if _ACC_ONLY.fullmatch(acc):
            out[acc] = out.get(acc, 0) + 1
    return out


def resolve_candidates(row: pd.Series, complex_dir: Optional[str], model_path: str,
                       seqs: UniProtSequences) -> Tuple[Dict[str, str], str, Dict[str, int]]:
    """Return (candidates {uniprot:seq}, source_label, expected_stoich)."""
    sources = []
    if "uniprots" in row and str(row.get("uniprots")) != "nan" and row.get("uniprots"):
        sources.append(("uniprots_col", str(row["uniprots"])))
    if complex_dir:
        sources.append(("folder_name", os.path.basename(complex_dir)))
    sources.append(("model_name", os.path.basename(model_path)))
    if "matching_hits" in row:
        sources.append(("matching_hits", None))

    for label, text in sources:
        stoich = (_matching_hits_uniprots(row.get("matching_hits"))
                  if label == "matching_hits" else parse_uniprots_from_name(text))
        if stoich:
            candidates = {}
            for acc in stoich:
                s = seqs.get(acc)
                if s:
                    candidates[acc] = s
            if candidates:
                return candidates, label, dict(stoich)
    return {}, "none", {}


def _row_status(cac, pdb, status, flags="", **extra):
    d = {c: "" for c in SUMMARY_COLS}
    d.update({"complex_ac": cac, "pdb_id": pdb, "status": status, "flags": flags})
    d.update(extra)
    return d


def _process_row(idx: int, row: "pd.Series", cfg: Config, seqs: UniProtSequences,
                 json_dir: str, save_json: bool) -> Tuple[int, List[Dict], List[Dict], List[Dict], List[str]]:
    """Score one mapping-file row (one complex_ac). Fully self-contained: every
    resource it touches (compare_one's temp work dir, per-complex JSON path,
    reference/sequence caches guarded internally) is either unique to this call
    or lock-protected, so this function is safe to run concurrently for
    different rows. Returns (idx, summary_rows, chain_rows, iface_rows, log_lines)
    so the caller can put results back in original row order regardless of
    which order the workers finish in.
    """
    summary_rows: List[Dict] = []
    chain_rows: List[Dict] = []
    iface_rows: List[Dict] = []
    log: List[str] = []

    cac = str(row["complex_ac"]).strip()
    pdb = str(row["pdb_id"]).strip()
    # Manifest-driven provenance (blank on the legacy complex_ac name-matching
    # path). Injected by run_batch when a manifest resolves this row to a
    # specific composition folder: `cf_folder_type` is the folder suffix
    # (pool_output / pair_output) and `stoich_source` is which manifest
    # composition it came from (identifiers / pred_1 / pred_2 / pred_3). Stamped
    # onto every emitted row so multiple folders for one complex stay
    # distinguishable, and used to keep per-complex JSON filenames unique.
    def _tag(col: str) -> str:
        return (str(row[col]).strip()
                if col in row and str(row.get(col)) not in ("nan", "None", "")
                else "")
    stoich_source = _tag("stoich_source")
    cf_folder_type = _tag("cf_folder_type")
    try:
        # ---- locate model(s) ----
        complex_dir = None
        models: List[str] = []
        model_col = row.get("model") if "model" in row else None
        if model_col and str(model_col) != "nan":
            mc = str(model_col)
            models = sorted(glob.glob(mc)) if any(c in mc for c in "*?[") else (
                [mc] if os.path.exists(mc) else [])
        else:
            complex_dir, models = locate_models(
                cfg.combfold_base, cac,
                row.get("folder") if "folder" in row else None,
                cfg.cf_model_globs)
            if not models:
                for g in (f"{cac.replace('-', '_')}*.pdb", f"{cac}*.pdb"):
                    hits = sorted(glob.glob(os.path.join(cfg.combfold_base, g)))
                    if hits:
                        models = hits
                        break
        if not models:
            summary_rows.append(_row_status(cac, pdb, "no_model_found",
                                             stoich_source=stoich_source,
                                             cf_folder_type=cf_folder_type))
            log.append(f"{cac}: no CombFold model found")
            return idx, summary_rows, chain_rows, iface_rows, log

        # ---- confidence ----
        conf: Dict[int, Tuple[float, int]] = {}
        if complex_dir:
            conf = parse_confidence(os.path.join(complex_dir, "assembled_results",
                                                 cfg.cf_confidence_name))
            if not conf:
                conf = parse_confidence(os.path.join(complex_dir, cfg.cf_confidence_name))

        # ---- candidate UniProts ----
        candidates, csource, stoich = resolve_candidates(row, complex_dir, models[0], seqs)
        if not candidates:
            summary_rows.append(_row_status(cac, pdb, "no_uniprots_resolved",
                                             stoich_source=stoich_source,
                                             cf_folder_type=cf_folder_type))
            log.append(f"{cac}: could not resolve candidate UniProts")
            return idx, summary_rows, chain_rows, iface_rows, log

        # ---- local reference override ----
        local_ref = None
        if "local_ref" in row and str(row.get("local_ref")) != "nan" and row.get("local_ref"):
            lr = str(row["local_ref"])
            local_ref = lr if os.path.exists(lr) else None

        model_list = models if cfg.score_all_clusters else models[:1]
        for i, mp in enumerate(model_list):
            cidx = cluster_idx_from_filename(mp)
            if cidx < 0:
                cidx = i
            score, rank = conf.get(cidx, (float("nan"), cidx))
            meta = {"complex_ac": cac, "pdb_id": pdb, "cf_cluster": cidx,
                    "cf_confidence": score, "cf_rank": rank,
                    "candidate_source": csource, "expected_stoich": stoich}
            res = compare_one(mp, pdb, candidates, cfg, meta=meta, local_au=local_ref)
            for s in res["summary"]:
                s.setdefault("candidate_source", csource)
                s["stoich_source"] = stoich_source
                s["cf_folder_type"] = cf_folder_type
            for r in res["per_chain"]:
                r.setdefault("stoich_source", stoich_source)
                r.setdefault("cf_folder_type", cf_folder_type)
            for r in res["per_interface"]:
                r.setdefault("stoich_source", stoich_source)
                r.setdefault("cf_folder_type", cf_folder_type)
            summary_rows.extend(res["summary"])
            chain_rows.extend(res["per_chain"])
            iface_rows.extend(res["per_interface"])
            if save_json:
                json_stem = (f"{cac}_{stoich_source or 'na'}_{cf_folder_type or 'na'}_c{cidx}"
                             if (stoich_source or cf_folder_type) else f"{cac}_c{cidx}")
                with open(os.path.join(json_dir, f"{json_stem}.json"), "w") as fh:
                    json.dump(res["json"], fh, indent=2, default=str)
            log.append(f"{cac} cluster {cidx}: {len(res['summary'])} form(s) scored "
                       f"(candidates from {csource})")
    except Exception as e:  # never abort the whole batch
        summary_rows.append(_row_status(cac, pdb, "error", flags=str(e)[:200],
                                         stoich_source=stoich_source,
                                         cf_folder_type=cf_folder_type))
        log.append(f"{cac}: ERROR {type(e).__name__}: {e}")
    return idx, summary_rows, chain_rows, iface_rows, log


def run_batch(cfg: Config, mapping_path: str, seqs: UniProtSequences,
              complexes: Optional[List[str]] = None, save_json: bool = True,
              n_workers: Optional[int] = None):
    """Execute the batch and write the three CSVs + per-complex JSON + log.

    n_workers: number of rows (complexes) to score concurrently. Each row is
    independent (own temp dir, own JSON file; shared sequence/reference caches
    are lock-protected -- see UniProtSequences.get and reference.acquire_reference_files),
    so this is a pure orchestration-level speedup with no effect on any metric.
    Default: min(8, cpu_count). Pass n_workers=1 to reproduce the original
    strictly-serial code path (e.g. for debugging).
    """
    cfg = cfg.resolved()
    os.makedirs(cfg.out_dir, exist_ok=True)
    json_dir = os.path.join(cfg.out_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    _reference.reset_cache_stats()

    df = pd.read_csv(mapping_path)
    df.columns = [c.strip() for c in df.columns]
    if "complex_ac" not in df.columns or "pdb_id" not in df.columns:
        raise ValueError("mapping file must have columns 'complex_ac' and 'pdb_id'")

    if n_workers is None:
        n_workers = min(8, os.cpu_count() or 4)
    n_workers = max(1, int(n_workers))

    # ---- Optional manifest-driven folder resolution ---------------------
    # CombFold output folders can be named by UniProt *composition*
    # (e.g. P06782x1_P12904x1_P34164x1_pool_output) instead of by complex_ac.
    # When that happens the complex_ac name-matching in find_complex_dirs()
    # cannot locate them and every complex reports no_model_found. If a manifest
    # is configured we translate each complex_ac into its candidate composition
    # folder(s) -- identifiers + pred_1/2/3, each x cf_folder_suffixes -- and
    # score every folder that exists on disk as its own work item. Complexes
    # with no manifest entry (or whose composition folders are absent) fall back
    # to the original name-matching path.
    manifest_by_ac: Dict[str, "pd.Series"] = {}
    if cfg.manifest_csv:
        manifest_by_ac = load_manifest(cfg.manifest_csv)

    # One complex_ac can now expand into several work items (one per resolved
    # composition folder), so work items carry their own unique key -- the df
    # row index is no longer unique across items.
    work_items: List["pd.Series"] = []
    for _idx, row in df.iterrows():
        cac = str(row["complex_ac"]).strip()
        if complexes and cac not in complexes:
            continue
        mrow = manifest_by_ac.get(cac) if manifest_by_ac else None
        resolved: List[Tuple[str, str, str]] = []  # (stoich_source, suffix, abs_folder)
        if mrow is not None:
            for label, stoich in candidate_stoichiometries(mrow):
                for suffix, path in resolve_folders(cfg.combfold_base, stoich,
                                                     cfg.cf_folder_suffixes):
                    resolved.append((label, suffix, path))
        if resolved:
            for label, suffix, path in resolved:
                r = row.copy()
                r["folder"] = path            # consumed by find_complex_dirs()
                r["stoich_source"] = label    # provenance -> output rows + JSON
                r["cf_folder_type"] = suffix
                work_items.append(r)
        else:
            work_items.append(row)            # legacy name-matching fallback

    results: Dict[int, Tuple[List[Dict], List[Dict], List[Dict], List[str]]] = {}
    if n_workers == 1:
        for wid, row in enumerate(work_items):
            _, s, c, i, l = _process_row(wid, row, cfg, seqs, json_dir, save_json)
            results[wid] = (s, c, i, l)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_row, wid, row, cfg, seqs, json_dir, save_json)
                       for wid, row in enumerate(work_items)]
            for fut in as_completed(futures):
                wid, s, c, i, l = fut.result()
                results[wid] = (s, c, i, l)

    # Reassemble in deterministic dispatch order (mapping-file row order, then
    # resolved-folder order), regardless of which worker finished first.
    summary_rows: List[Dict] = []
    chain_rows: List[Dict] = []
    iface_rows: List[Dict] = []
    log: List[str] = []
    for wid in range(len(work_items)):
        s, c, i, l = results[wid]
        summary_rows.extend(s)
        chain_rows.extend(c)
        iface_rows.extend(i)
        log.extend(l)

    summary_df = pd.DataFrame(summary_rows)
    chain_df = pd.DataFrame(chain_rows)
    iface_df = pd.DataFrame(iface_rows)
    summary_df = _order_cols(summary_df, SUMMARY_COLS)
    chain_df = _order_cols(chain_df, CHAIN_COLS)
    iface_df = _order_cols(iface_df, IFACE_COLS)

    summary_df.to_csv(os.path.join(cfg.out_dir, "complex_summary.csv"), index=False)
    chain_df.to_csv(os.path.join(cfg.out_dir, "per_chain.csv"), index=False)
    iface_df.to_csv(os.path.join(cfg.out_dir, "per_interface.csv"), index=False)

    stats = _reference.get_cache_stats()
    cache_line = (f"[ref cache] {stats['hits']} file(s) reused from {cfg.ref_cache}, "
                  f"{stats['downloads']} newly downloaded, {stats['failed']} failed")
    print(cache_line)
    log.append(cache_line)

    with open(os.path.join(cfg.out_dir, "run_log.txt"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    return summary_df, chain_df, iface_df


def _order_cols(df: pd.DataFrame, preferred: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=preferred)
    cols = [c for c in preferred if c in df.columns] + \
           [c for c in df.columns if c not in preferred]
    return df[cols]