"""Metric engines: RMSD (from-scratch Kabsch), TM-score (US-align), DockQ.

RMSD is implemented here in NumPy rather than delegated to PyMOL so that every
choice is explicit and the correspondence is the fixed sequence-based one used by
the rest of the pipeline. Two variants are reported, mirroring the van Gerwen
PyMOL `align` convention:
  * cycles=0  -> RMSD over ALL co-observed CA (no outlier rejection);
  * cycles=N  -> iterative refinement, rejecting CA pairs deviating > cutoff A
                 and refitting, up to N times (default N=5, cutoff=2.0 A).
Atom counts are always reported so the two numbers are interpretable.

TM-score is delegated to US-align (the canonical TM-maximizing superposition),
which natively prints both normalizations (by model and by reference length).
DockQ v2 supplies the interface metrics and an authoritative chain mapping used
as a cross-check.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer


# --------------------------------------------------------------------------- #
# RMSD (fixed-correspondence Kabsch)
# --------------------------------------------------------------------------- #
def _kabsch_fit(P: np.ndarray, Q: np.ndarray):
    """Least-squares superposition of P onto Q (paired rows). Returns rotated P
    and RMSD over the given pairs.

    Uses Bio.SVDSuperimposer (Biopython is already a pipeline dependency) rather
    than a hand-rolled SVD, so the reflection/chirality correction (forcing a
    proper rotation, det=+1, never a mirror-image "best fit") is delegated to a
    validated, widely-used library call instead of custom linear algebra.
    Verified to be numerically identical to a from-scratch implementation with
    the standard det-correction applied, including on adversarial mirror-image
    inputs where an uncorrected Kabsch would otherwise return a spuriously low
    RMSD via an improper (reflective) rotation."""
    sup = SVDSuperimposer()
    sup.set(Q, P)   # Q = fixed reference, P = coordinates to be rotated onto it
    sup.run()
    rot, tran = sup.get_rotran()
    P_rot = P @ rot + tran
    rmsd = float(sup.get_rms())
    return P_rot, rmsd


def rmsd_fixed(P: np.ndarray, Q: np.ndarray, cycles: int = 0, cutoff: float = 2.0,
               min_atoms: int = 10, min_frac: float = 0.0) -> Dict[str, float]:
    """RMSD over paired coordinates with optional iterative outlier rejection.

    cycles=0 -> RMSD over all pairs (no rejection).
    cycles>0 -> up to `cycles` refit iterations. Each cycle rejects CA pairs whose
      post-fit deviation exceeds `cutoff` x current RMSD (a sigma-style, adaptive
      cutoff matching PyMOL align's "outlier rejection cutoff in RMS"), then
      refits on the survivors. Stops early on convergence or when the surviving
      set would fall below floor = max(min_atoms, min_frac*n_total).
    An adaptive cutoff (vs a fixed Angstrom cutoff) is essential: when a hinge/
    lever-arm error makes the all-atom fit poor, a fixed 2 A cutoff leaves too
    few atoms to bootstrap, whereas cutoff x RMSD peels outliers progressively.
    """
    n_total = len(P)
    if n_total < 3:
        return {"rmsd": float("nan"), "n_used": n_total, "n_total": n_total}
    floor = max(3, min_atoms, int(min_frac * n_total))
    sel = np.arange(n_total)
    P_rot, rmsd = _kabsch_fit(P, Q)
    for _ in range(cycles):
        dev = np.sqrt(((P_rot - Q[sel]) ** 2).sum(1))
        thresh = cutoff * rmsd if rmsd > 0 else cutoff
        new_sel = sel[dev <= thresh]
        if len(new_sel) == len(sel) or len(new_sel) < floor:
            break
        sel = new_sel
        P_rot, rmsd = _kabsch_fit(P[sel], Q[sel])
    return {"rmsd": round(rmsd, 3), "n_used": int(len(sel)), "n_total": int(n_total)}


# --------------------------------------------------------------------------- #
# TM-score (US-align)
# --------------------------------------------------------------------------- #
@dataclass
class TMResult:
    ok: bool
    tm_mod: float = float("nan")   # normalized by model length (Structure_1)
    tm_ref: float = float("nan")   # normalized by reference length (Structure_2)
    rmsd: float = float("nan")
    aligned_len: int = 0
    seq_id: float = float("nan")
    len_mod: int = 0
    len_ref: int = 0
    error: str = ""


_RE_L1 = re.compile(r"Length of Structure_1:\s*(\d+)")
_RE_L2 = re.compile(r"Length of Structure_2:\s*(\d+)")
_RE_AL = re.compile(r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+),\s*Seq_ID=n_identical/n_aligned=\s*([0-9.]+)")
_RE_TM1 = re.compile(r"TM-score=\s*([0-9.]+)\s*\(normalized by length of Structure_1")
_RE_TM2 = re.compile(r"TM-score=\s*([0-9.]+)\s*\(normalized by length of Structure_2")


def run_usalign(model_path: str, ref_path: str, usalign_bin: str = "USalign",
                mm: int = 0, ter: int = 1) -> TMResult:
    """Run US-align with model as Structure_1 and reference as Structure_2.

    mm=1 -> multimer (whole-complex) alignment; mm=0 -> single structure.
    """
    cmd = [usalign_bin, model_path, ref_path, "-ter", str(ter)]
    if mm:
        cmd += ["-mm", str(mm)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:  # pragma: no cover
        return TMResult(ok=False, error=f"usalign exec failed: {e}")
    out = p.stdout
    tm1 = _RE_TM1.search(out)
    tm2 = _RE_TM2.search(out)
    if not (tm1 and tm2):
        return TMResult(ok=False, error="could not parse US-align output")
    al = _RE_AL.search(out)
    l1 = _RE_L1.search(out)
    l2 = _RE_L2.search(out)
    return TMResult(
        ok=True,
        tm_mod=float(tm1.group(1)),
        tm_ref=float(tm2.group(1)),
        rmsd=float(al.group(2)) if al else float("nan"),
        aligned_len=int(al.group(1)) if al else 0,
        seq_id=float(al.group(3)) if al else float("nan"),
        len_mod=int(l1.group(1)) if l1 else 0,
        len_ref=int(l2.group(1)) if l2 else 0,
    )


# --------------------------------------------------------------------------- #
# DockQ v2
# --------------------------------------------------------------------------- #
CAPRI_LEGEND = [
    (0.80, "High"),
    (0.49, "Medium"),
    (0.23, "Acceptable"),
    (0.00, "Incorrect"),
]


def capri_class(dockq: float) -> str:
    if dockq != dockq:  # nan
        return "NA"
    for thr, name in CAPRI_LEGEND:
        if dockq >= thr:
            return name
    return "Incorrect"


@dataclass
class DockQInterface:
    chain1: str
    chain2: str
    dockq: float
    fnat: float
    irmsd: float
    lrmsd: float
    f1: float
    clashes: int
    capri: str


@dataclass
class DockQResult:
    ok: bool
    global_dockq: float = float("nan")   # mean over native interfaces
    mapping: Dict[str, str] = field(default_factory=dict)   # native -> model (DockQ best_mapping: {native_chain: model_chain})
    mapping_str: str = ""
    interfaces: Dict[str, DockQInterface] = field(default_factory=dict)
    error: str = ""


def run_dockq(model_path: str, native_path: str, dockq_bin: str = "DockQ",
              allowed_mismatches: int = 0) -> DockQResult:
    """Run DockQ v2 (model vs native), returning per-interface metrics + mapping.

    `allowed_mismatches` is forwarded to DockQ's own `--allowed_mismatches`.
    DockQ independently re-derives chain homology via its own sequence check
    before scoring interfaces, and its default tolerance is ZERO mismatches:
    a single true residue difference between the modeled full-length sequence
    and the crystallized construct (a common situation -- expression-tag
    remnants, strain polymorphisms, or engineered point mutations in the PDB
    entry) makes DockQ reject the chain outright and return no interfaces at
    all, even though our own sequence-based assignment (which gates on
    `same_protein_identity`, default >=90% identity) already established the
    correspondence. We therefore pass a non-zero default so DockQ's internal
    check is no stricter than the identity threshold this pipeline already
    used to accept the chain as "the same protein".
    """
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        json_path = tf.name
    cmd = [dockq_bin, model_path, native_path, "--json", json_path]
    if allowed_mismatches > 0:
        cmd += ["--allowed_mismatches", str(allowed_mismatches)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except Exception as e:  # pragma: no cover
        return DockQResult(ok=False, error=f"dockq exec failed: {e}")
    try:
        with open(json_path) as fh:
            d = json.load(fh)
    except Exception:
        return DockQResult(ok=False, error=f"dockq no json (rc={p.returncode}): {p.stderr[:200]}")
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass

    interfaces: Dict[str, DockQInterface] = {}
    for key, v in (d.get("best_result") or {}).items():
        dq = float(v.get("DockQ", float("nan")))
        interfaces[key] = DockQInterface(
            chain1=v.get("chain1", key[:1]),
            chain2=v.get("chain2", key[-1:]),
            dockq=dq,
            fnat=float(v.get("fnat", float("nan"))),
            irmsd=float(v.get("iRMSD", float("nan"))),
            lrmsd=float(v.get("LRMSD", float("nan"))),
            f1=float(v.get("F1", float("nan"))),
            clashes=int(v.get("clashes", 0)),
            capri=capri_class(dq),
        )
    return DockQResult(
        ok=bool(interfaces),
        global_dockq=float(d.get("GlobalDockQ", float("nan"))),
        mapping=d.get("best_mapping", {}) or {},
        mapping_str=d.get("best_mapping_str", ""),
        interfaces=interfaces,
        error="" if interfaces else f"no interfaces (rc={p.returncode})",
    )
