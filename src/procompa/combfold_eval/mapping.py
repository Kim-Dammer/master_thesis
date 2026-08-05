"""Sequence-based chain mapping and residue correspondence.

Two jobs, both done from sequence (the user's explicit choice), never from chain
IDs or chain lengths:

1. Assign each structure chain (model AND reference) to a UniProt accession by
   local BLOSUM62 alignment against the complex's candidate UniProt sequences.
   This is what lets us match a model chain to the *right* reference chain even
   when chain letters are scrambled between the two files.

2. For a matched (model, reference) chain pair, align the two OBSERVED sequences
   to get a 1:1 residue correspondence over co-observed positions. This fixed
   correspondence is what the RMSD (Kabsch) uses, and it is where coverage is
   measured. Residue *numbers* are never trusted (crystal cores are renumbered
   and have gaps).

Identical-copy permutations in homo-oligomers cannot be resolved by sequence
alone; that is handled structurally in compare.py (US-align / DockQ), and this
module only provides the UniProt-level grouping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

from .config import Config
from .structure_utils import ChainData, CleanStructure

_BLOSUM62 = substitution_matrices.load("BLOSUM62")


def _aligner(mode: str = "local") -> PairwiseAligner:
    a = PairwiseAligner()
    a.substitution_matrix = _BLOSUM62
    a.open_gap_score = -11.0
    a.extend_gap_score = -1.0
    a.mode = mode
    return a


def pairwise_align(seq_a: str, seq_b: str, mode: str = "local"
                   ) -> Tuple[List[Tuple[int, int]], int, int, float]:
    """Align two sequences; return (index_pairs, n_identical, aligned_len, score).

    index_pairs are 0-based (i_in_a, j_in_b) for ungapped aligned columns, in
    order, so they can index directly into the two chains' residue lists.
    """
    if not seq_a or not seq_b:
        return [], 0, 0, 0.0
    aln = _aligner(mode).align(seq_a, seq_b)[0]
    pairs: List[Tuple[int, int]] = []
    n_id = 0
    a_blocks, b_blocks = aln.aligned
    for (a0, a1), (b0, b1) in zip(a_blocks, b_blocks):
        for k in range(a1 - a0):
            i, j = a0 + k, b0 + k
            pairs.append((i, j))
            if seq_a[i] == seq_b[j]:
                n_id += 1
    return pairs, n_id, len(pairs), float(aln.score)


@dataclass
class ChainAssignment:
    chain_id: str
    kind: str
    uniprot: Optional[str] = None
    identity: float = 0.0        # % identical over aligned columns
    coverage: float = 0.0        # aligned columns / observed chain length
    is_same_protein: bool = False
    is_homolog: bool = False
    runner_up: Optional[str] = None
    runner_up_identity: float = 0.0


def assign_chain(chain: ChainData, candidates: Dict[str, str], cfg: Config) -> ChainAssignment:
    """Assign one chain to its best-matching candidate UniProt."""
    ca = ChainAssignment(chain_id=chain.chain_id, kind=chain.kind)
    if chain.kind != "protein" or len(chain.seq) < 1:
        return ca
    scored = []
    for acc, full in candidates.items():
        if not full:
            continue
        pairs, n_id, aln_len, _ = pairwise_align(chain.seq, full, mode="local")
        if aln_len == 0:
            continue
        identity = 100.0 * n_id / aln_len
        coverage = aln_len / max(1, len(chain.seq))
        scored.append((identity * coverage, identity, coverage, acc))
    if not scored:
        return ca
    scored.sort(reverse=True)
    _, identity, coverage, acc = scored[0]
    ca.uniprot = acc
    ca.identity = round(identity, 2)
    ca.coverage = round(coverage, 4)
    ca.is_same_protein = identity >= cfg.same_protein_identity and coverage >= 0.5
    ca.is_homolog = (not ca.is_same_protein) and identity >= cfg.homolog_identity
    if len(scored) > 1:
        ca.runner_up = scored[1][3]
        ca.runner_up_identity = round(scored[1][1], 2)
    if not (ca.is_same_protein or ca.is_homolog):
        ca.uniprot = None
    return ca


def assign_all(clean: CleanStructure, candidates: Dict[str, str], cfg: Config
               ) -> Dict[str, ChainAssignment]:
    return {cid: assign_chain(ch, candidates, cfg) for cid, ch in clean.chains.items()}


def composition(assign: Dict[str, ChainAssignment]) -> Dict[str, int]:
    """UniProt multiset over protein chains that received an assignment."""
    comp: Dict[str, int] = {}
    for a in assign.values():
        if a.uniprot:
            comp[a.uniprot] = comp.get(a.uniprot, 0) + 1
    return comp


def composition_match_score(model_comp: Dict[str, int], ref_comp: Dict[str, int]
                            ) -> Tuple[int, int, int]:
    """Compare two UniProt multisets.

    Returns (n_shared_chains, n_missing_from_ref, n_extra_in_ref) where
    n_shared_chains sums min(copies) over shared UniProts (the number of subunit
    copies that can actually be compared).
    """
    shared = 0
    missing = 0
    for acc, n in model_comp.items():
        r = ref_comp.get(acc, 0)
        shared += min(n, r)
        if r < n:
            missing += (n - r)
    extra = 0
    for acc, n in ref_comp.items():
        m = model_comp.get(acc, 0)
        if m < n:
            extra += (n - m)
    return shared, missing, extra


@dataclass
class ResidueCorrespondence:
    model_ca: np.ndarray      # (N,3) model CA for co-observed positions
    ref_ca: np.ndarray        # (N,3) reference CA for co-observed positions
    n_aligned: int
    n_identical: int
    pct_identity: float       # over aligned columns
    cov_ref_resolved: float   # aligned / reference observed length
    cov_model_resolved: float # aligned / model observed length


def residue_correspondence(model_chain: ChainData, ref_chain: ChainData
                           ) -> ResidueCorrespondence:
    """Fixed 1:1 CA correspondence for a matched chain pair (sequence alignment)."""
    pairs, n_id, aln_len, _ = pairwise_align(model_chain.seq, ref_chain.seq, mode="local")
    if aln_len == 0:
        z = np.zeros((0, 3))
        return ResidueCorrespondence(z, z, 0, 0, 0.0, 0.0, 0.0)
    m_ca = model_chain.ca_coords
    r_ca = ref_chain.ca_coords
    mi = np.array([i for i, _ in pairs])
    rj = np.array([j for _, j in pairs])
    return ResidueCorrespondence(
        model_ca=m_ca[mi],
        ref_ca=r_ca[rj],
        n_aligned=aln_len,
        n_identical=n_id,
        pct_identity=round(100.0 * n_id / aln_len, 2),
        cov_ref_resolved=round(aln_len / max(1, len(ref_chain.seq)), 4),
        cov_model_resolved=round(aln_len / max(1, len(model_chain.seq)), 4),
    )
