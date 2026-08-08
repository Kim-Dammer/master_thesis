"""Orchestration: compare ONE CombFold model against ONE reference PDB id.

This ties the library together for a single (complex, cluster) model:

  1. load + clean the model; assign every model chain to a UniProt by sequence.
  2. acquire ALL reference forms (biological assemblies + asymmetric unit),
     assign their chains, and pick the primary form by composition (pre-metric).
  3. for EACH reference form, restrict both structures to the shared subunits
     (min copies per UniProt), give them a shared single-character chain
     alphabet, and:
       - run DockQ; use its optimal chain mapping as the authoritative,
         copy-resolved model<->reference pairing (this is what resolves
         homo-oligomer copy ambiguity that sequence alone cannot);
       - cross-check that DockQ's pairing stays within the same UniProt and
         flag any disagreement (the user's sequence assignment stays the
         reference of truth);
       - per paired chain: single-chain US-align (per-subunit TM, both norms)
         and fixed-correspondence Kabsch RMSD (cycles 0 and N) on co-observed CA;
       - one whole-complex Kabsch over ALL paired CA -> global assembly RMSD;
       - whole-complex US-align (-mm) -> complex TM (both norms);
       - per-interface DockQ (Fnat/iRMSD/LRMSD/CAPRI) + mean over native ifaces.

Every metric is reported with coverage and atom counts, and every non-trivial
situation raises an explicit flag rather than failing silently.

DockQ mapping orientation (verified empirically, do not change without re-test):
  best_mapping = {NATIVE_chain_id: MODEL_chain_id};
  each interface's chain1/chain2 are MODEL chain ids; interface key = native ids.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .structure_utils import load_and_clean, write_subset, write_single_chain, CleanStructure
from . import mapping as M
from .reference import build_and_select, RefForm
from .metrics import run_usalign, run_dockq, rmsd_fixed, capri_class

# 62 usable single-character PDB chain ids
_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def _fmt_list(xs: List[str]) -> str:
    return ";".join(xs)


def _ref_assembly_id(form_id: str) -> str:
    if form_id == "asymmetric_unit":
        return "AU"
    if form_id.startswith("assembly_"):
        return form_id.split("_", 1)[1]
    return form_id


def _shared_counts(model_comp: Dict[str, int], ref_comp: Dict[str, int],
                   model_assign: Dict[str, M.ChainAssignment]) -> "OrderedDict[str,int]":
    """min(copies) per shared UniProt, ordered by model chain order."""
    seen = OrderedDict()
    for cid, a in model_assign.items():
        if a.uniprot and a.uniprot in ref_comp and a.uniprot not in seen:
            seen[a.uniprot] = min(model_comp[a.uniprot], ref_comp[a.uniprot])
    return seen


def _build_slots(model_assign: Dict[str, M.ChainAssignment],
                 ref_assign: Dict[str, M.ChainAssignment],
                 shared: "OrderedDict[str,int]") -> List[Tuple[str, str, str, str]]:
    """Return slots [(letter, uniprot, model_orig_chain, ref_orig_chain)].

    Copies of the same UniProt are paired by chain order only as a *starting*
    labelling; DockQ later decides the true copy correspondence. model and ref
    share the same letter alphabet so single-copy subunits map letter->letter.
    """
    model_by_uni: Dict[str, List[str]] = defaultdict(list)
    ref_by_uni: Dict[str, List[str]] = defaultdict(list)
    for cid, a in model_assign.items():
        if a.uniprot:
            model_by_uni[a.uniprot].append(cid)
    for cid, a in ref_assign.items():
        if a.uniprot:
            ref_by_uni[a.uniprot].append(cid)
    slots: List[Tuple[str, str, str, str]] = []
    li = 0
    for u, k in shared.items():
        for i in range(k):
            if li >= len(_ALPHABET):
                return slots  # caller flags "too_many_chains"
            slots.append((_ALPHABET[li], u, model_by_uni[u][i], ref_by_uni[u][i]))
            li += 1
    return slots


@dataclass
class FormResult:
    summary: Dict
    per_chain: List[Dict] = field(default_factory=list)
    per_interface: List[Dict] = field(default_factory=list)


def _compare_against_form(model_clean: CleanStructure,
                          model_assign: Dict[str, M.ChainAssignment],
                          model_comp: Dict[str, int],
                          candidates: Dict[str, str],
                          form: RefForm, cfg: Config, meta: Dict,
                          work_dir: str) -> FormResult:
    flags: List[str] = []
    ref_clean = form.clean
    ref_assign = form.assign
    ref_comp = form.comp

    base = {
        "complex_ac": meta.get("complex_ac", ""),
        "pdb_id": meta.get("pdb_id", ""),
        "cf_cluster": meta.get("cf_cluster", -1),
        "cf_confidence": meta.get("cf_confidence", float("nan")),
        "cf_rank": meta.get("cf_rank", -1),
        "ref_form": form.form_id,
        "ref_assembly_id": _ref_assembly_id(form.form_id),
        "is_primary_ref": form.is_primary,
        "ref_select_reason": form.select_reason,
        "stoich_source": meta.get("stoich_source", ""),
        "cf_folder_type": meta.get("cf_folder_type", ""),
    }

    missing = [u for u in model_comp if model_comp[u] > ref_comp.get(u, 0)]
    extra = [u for u in ref_comp if ref_comp[u] > model_comp.get(u, 0)]
    if any(d.kind == "nucleic" for d in ref_clean.chains.values()):
        flags.append("reference_has_nucleic")

    shared = _shared_counts(model_comp, ref_comp, model_assign)
    n_shared = int(sum(shared.values()))
    summary = dict(base)
    summary.update({
        "n_shared_subunits": n_shared,
        "missing_uniprots": _fmt_list(missing),
        "extra_uniprots": _fmt_list(extra),
        "complex_TM_ref": float("nan"), "complex_TM_mod": float("nan"),
        "global_rmsd_nocyc": float("nan"), "global_rmsd_wcyc": float("nan"),
        "n_res_global": 0, "coverage_global": float("nan"),
        "mean_dockq": float("nan"),
        "pairing_source": "", "status": "", "flags": "",
    })
    if n_shared == 0:
        summary["status"] = "no_shared_subunits"
        summary["flags"] = _fmt_list(sorted(set(flags)))
        return FormResult(summary=summary)

    slots = _build_slots(model_assign, ref_assign, shared)
    if len(slots) < n_shared:
        flags.append("too_many_chains_truncated")

    model_map = OrderedDict((mo, letter) for (letter, u, mo, ro) in slots)
    ref_map = OrderedDict((ro, letter) for (letter, u, mo, ro) in slots)
    out2uni = {letter: u for (letter, u, mo, ro) in slots}
    model_out2orig = {letter: mo for (letter, u, mo, ro) in slots}
    ref_out2orig = {letter: ro for (letter, u, mo, ro) in slots}

    model_pdb = os.path.join(work_dir, "model_shared.pdb")
    ref_pdb = os.path.join(work_dir, "ref_shared.pdb")
    write_subset(model_clean, model_map, model_pdb)
    write_subset(ref_clean, ref_map, ref_pdb)

    # ---- DockQ: authoritative copy-resolved pairing ---------------------
    dq = run_dockq(model_pdb, ref_pdb, cfg.dockq_bin,
                    allowed_mismatches=cfg.dockq_allowed_mismatches)
    pairs: List[Tuple[str, str]] = []       # (model_letter, ref_letter)
    paired_ref = set()
    if dq.ok and dq.mapping:
        for ref_out, model_out in dq.mapping.items():   # {native: model}
            if ref_out in ref_out2orig and model_out in model_out2orig:
                pairs.append((model_out, ref_out))
                paired_ref.add(ref_out)
        pairing_source = "dockq"
    else:
        pairing_source = "sequence"
        flags.append("dockq_no_interface_or_failed")
    for (letter, u, mo, ro) in slots:            # fill any unmapped chain
        if letter not in paired_ref:
            pairs.append((letter, letter))
            if pairing_source == "dockq":
                flags.append("chain_unmapped_by_dockq")
    if pairing_source == "sequence" and any(k > 1 for k in shared.values()):
        flags.append("homo_oligomer_copy_pairing_arbitrary")
    for (m_out, r_out) in pairs:
        if out2uni.get(m_out) != out2uni.get(r_out):
            flags.append("mapping_disagreement")

    # ---- per-chain metrics + accumulate global correspondence -----------
    per_chain: List[Dict] = []
    all_m, all_r = [], []
    total_ref_resolved = 0
    for (m_out, r_out) in pairs:
        uni = out2uni.get(m_out, out2uni.get(r_out, ""))
        m_chain = model_clean.chains[model_out2orig[m_out]]
        r_chain = ref_clean.chains[ref_out2orig[r_out]]
        corr = M.residue_correspondence(m_chain, r_chain)
        total_ref_resolved += len(r_chain.seq)
        if corr.n_aligned:
            all_m.append(corr.model_ca)
            all_r.append(corr.ref_ca)
        r0 = rmsd_fixed(corr.model_ca, corr.ref_ca, cycles=0,
                        min_atoms=cfg.outlier_min_atoms)
        r5 = rmsd_fixed(corr.model_ca, corr.ref_ca, cycles=cfg.outlier_cycles,
                        cutoff=cfg.outlier_cutoff, min_atoms=cfg.outlier_min_atoms,
                        min_frac=cfg.min_frac_atoms_kept)
        m1 = os.path.join(work_dir, f"m_{m_out}.pdb")
        r1 = os.path.join(work_dir, f"r_{r_out}.pdb")
        write_single_chain(model_clean, model_out2orig[m_out], m1)
        write_single_chain(ref_clean, ref_out2orig[r_out], r1)
        tm = run_usalign(m1, r1, cfg.usalign_bin, mm=0)
        full = candidates.get(uni)
        cov_full = round(corr.n_aligned / len(full), 4) if full else float("nan")
        per_chain.append({
            **base,
            "uniprot": uni,
            "chain_mod": model_out2orig[m_out],
            "chain_ref": ref_out2orig[r_out],
            "slot": m_out,
            "tm_ref": tm.tm_ref, "tm_mod": tm.tm_mod,
            "rmsd_nocyc": r0["rmsd"], "n_res_nocyc": r0["n_used"],
            "rmsd_wcyc": r5["rmsd"], "n_res_wcyc": r5["n_used"],
            "n_res": corr.n_aligned,
            "coverage_resolved": corr.cov_ref_resolved,
            "coverage_fulllen": cov_full,
            "seq_identity": corr.pct_identity,
        })

    # ---- global assembly RMSD (one whole-complex Kabsch) ----------------
    if all_m:
        P = np.vstack(all_m)
        Q = np.vstack(all_r)
        g0 = rmsd_fixed(P, Q, cycles=0, min_atoms=cfg.outlier_min_atoms)
        g5 = rmsd_fixed(P, Q, cycles=cfg.outlier_cycles, cutoff=cfg.outlier_cutoff,
                        min_atoms=cfg.outlier_min_atoms, min_frac=cfg.min_frac_atoms_kept)
        n_res_global = int(P.shape[0])
        coverage_global = round(n_res_global / total_ref_resolved, 4) if total_ref_resolved else float("nan")
        summary["global_rmsd_nocyc"] = g0["rmsd"]
        summary["global_rmsd_wcyc"] = g5["rmsd"]
        summary["n_res_global"] = n_res_global
        summary["coverage_global"] = coverage_global

    # ---- complex TM (whole-complex US-align) ----------------------------
    mm = 1 if len(slots) >= 2 else 0
    ctm = run_usalign(model_pdb, ref_pdb, cfg.usalign_bin, mm=mm)
    if ctm.ok:
        summary["complex_TM_ref"] = ctm.tm_ref
        summary["complex_TM_mod"] = ctm.tm_mod
    else:
        flags.append("complex_usalign_failed")

    # ---- per-interface DockQ (chain1/chain2 are MODEL ids) --------------
    per_interface: List[Dict] = []
    if dq.ok:
        for key, iface in dq.interfaces.items():
            u1 = out2uni.get(iface.chain1)
            u2 = out2uni.get(iface.chain2)
            label = "|".join(sorted([x for x in (u1, u2) if x])) or key
            per_interface.append({
                **base,
                "interface_uniprots": label,
                "iface_native_key": key,
                "chain1_mod": iface.chain1, "chain2_mod": iface.chain2,
                "dockq": iface.dockq, "fnat": iface.fnat,
                "irmsd": iface.irmsd, "lrmsd": iface.lrmsd,
                "capri_class": iface.capri,
            })
        summary["mean_dockq"] = dq.global_dockq

    summary["pairing_source"] = pairing_source
    summary["status"] = "scored"
    summary["flags"] = _fmt_list(sorted(set(flags)))
    return FormResult(summary=summary, per_chain=per_chain, per_interface=per_interface)


def compare_one(model_path: str, pdb_id: str, candidates: Dict[str, str],
                cfg: Config, meta: Optional[Dict] = None,
                local_au: Optional[str] = None,
                work_root: Optional[str] = None) -> Dict:
    """Compare one CombFold model against all reference forms of one PDB id.

    Returns {"summary": [...], "per_chain": [...], "per_interface": [...],
             "json": {...}} where each list has one entry per reference form.
    """
    meta = dict(meta or {})
    meta.setdefault("pdb_id", pdb_id)
    log: List[str] = []
    made_tmp = False
    if work_root is None:
        work_root = tempfile.mkdtemp(prefix="cfeval_")
        made_tmp = True

    out = {"summary": [], "per_chain": [], "per_interface": [],
           "json": {"meta": meta, "log": log}}
    try:
        if not os.path.exists(model_path):
            log.append(f"model missing: {model_path}")
            out["summary"].append({**meta, "ref_form": "", "status": "model_missing",
                                   "flags": ""})
            return out
        model_clean = load_and_clean(model_path, cfg)
        model_assign = M.assign_all(model_clean, candidates, cfg)
        model_comp = M.composition(model_assign)
        out["json"]["model_chains"] = {
            cid: {"uniprot": a.uniprot, "identity": a.identity,
                  "coverage": a.coverage, "kind": a.kind,
                  "is_same_protein": a.is_same_protein, "is_homolog": a.is_homolog}
            for cid, a in model_assign.items()}
        out["json"]["model_composition"] = model_comp
        if not model_comp:
            log.append("no model chain assigned to any candidate UniProt")
            out["summary"].append({**meta, "ref_form": "",
                                   "status": "model_unassigned", "flags": ""})
            return out

        forms = build_and_select(pdb_id, model_comp, candidates, cfg, local_au=local_au)
        if not forms:
            log.append(f"reference acquisition failed for {pdb_id}")
            out["summary"].append({**meta, "ref_form": "",
                                   "status": "ref_download_failed", "flags": ""})
            return out
        out["json"]["reference_forms"] = [
            {"form_id": f.form_id, "path": f.path, "n_chains": f.n_chains,
             "composition": f.comp, "n_shared": f.n_shared, "n_missing": f.n_missing,
             "n_extra": f.n_extra, "total_resolved": f.total_resolved,
             "is_primary": f.is_primary, "select_reason": f.select_reason,
             "ref_chains": {cid: a.uniprot for cid, a in f.assign.items() if a.uniprot}}
            for f in forms]

        for f in forms:
            wd = os.path.join(work_root, f"{meta.get('complex_ac','x')}_"
                              f"c{meta.get('cf_cluster','x')}_{f.form_id}")
            os.makedirs(wd, exist_ok=True)
            fr = _compare_against_form(model_clean, model_assign, model_comp,
                                       candidates, f, cfg, meta, wd)
            out["summary"].append(fr.summary)
            out["per_chain"].extend(fr.per_chain)
            out["per_interface"].extend(fr.per_interface)
            if cfg_save := getattr(cfg, "save_primary_structures", False):
                if f.is_primary:
                    dst = os.path.join(cfg.out_dir, "structures",
                                       f"{meta.get('complex_ac','x')}_c{meta.get('cf_cluster','x')}_{f.form_id}")
                    os.makedirs(dst, exist_ok=True)
                    for fn in ("model_shared.pdb", "ref_shared.pdb"):
                        src = os.path.join(wd, fn)
                        if os.path.exists(src):
                            shutil.copy(src, os.path.join(dst, fn))
        return out
    finally:
        if made_tmp:
            shutil.rmtree(work_root, ignore_errors=True)
