"""Structure I/O and clean-up.

Responsibilities (all explicit, so the user can see exactly what is done to a
structure before any metric is computed):
  * read PDB or mmCIF (also gzipped), keep only the first model
  * remove hydrogens, waters, alternative conformations (keep primary/altloc-A),
    ligands, ions and other non-amino-acid heteroatoms
  * normalize modified residues (MSE->MET etc.) so they are KEPT as protein
  * classify each chain as protein / nucleic / other, so nucleic-acid chains are
    flagged (and excluded from protein metrics) instead of silently dropped
  * extract, per protein chain, the observed one-letter sequence and the ordered
    CA coordinates used by TM-score and RMSD
Nothing here superimposes or scores; it only produces clean, labelled inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import gemmi

from .config import Config, AA3TO1, MODRES_RENAME

# DNA/RNA residue names (for flagging nucleic-acid chains, not for scoring)
NUC = {"DA", "DC", "DG", "DT", "DU", "DI", "A", "C", "G", "U", "I", "N"}


@dataclass
class ResidueRec:
    seqnum: int
    icode: str
    resname: str
    aa: str                 # one-letter code
    ca: Tuple[float, float, float]


@dataclass
class ChainData:
    chain_id: str
    kind: str               # "protein" | "nucleic" | "other"
    seq: str                # observed one-letter sequence (CA-bearing residues)
    residues: List[ResidueRec] = field(default_factory=list)
    n_protein: int = 0
    n_nucleic: int = 0
    n_other: int = 0

    @property
    def ca_coords(self) -> np.ndarray:
        if not self.residues:
            return np.zeros((0, 3))
        return np.array([r.ca for r in self.residues], dtype=float)


@dataclass
class CleanStructure:
    source: str
    st: gemmi.Structure                 # cleaned gemmi structure (model 0 only)
    chains: Dict[str, ChainData]        # chain_id -> ChainData
    entry_id: str = ""

    def protein_chains(self) -> Dict[str, ChainData]:
        return {c: d for c, d in self.chains.items() if d.kind == "protein"}


def _classify(prot: int, nuc: int, other: int) -> str:
    if prot == 0 and nuc == 0:
        return "other"
    if prot >= nuc:
        return "protein"
    return "nucleic"


def load_and_clean(path: str, cfg: Config) -> CleanStructure:
    """Read a structure file and return a cleaned, labelled CleanStructure."""
    st = gemmi.read_structure(path)
    entry_id = st.name or ""
    st.setup_entities()

    # keep only the first model (NMR ensembles / multi-model files)
    while len(st) > 1:
        del st[len(st) - 1]

    st.remove_hydrogens()
    st.remove_waters()
    # keep the primary conformer (altloc 'A'); by PDB convention this is the
    # highest-occupancy alternate, and it removes altloc ambiguity for all tools.
    st.remove_alternative_conformations()

    model = st[0]
    new_model = gemmi.Model("1")
    chains: Dict[str, ChainData] = {}

    for chain in model:
        n_prot = n_nuc = n_other = 0
        new_chain = gemmi.Chain(chain.name)
        residues: List[ResidueRec] = []
        for res in chain:
            name = res.name
            if name in AA3TO1:
                # normalize modified-residue atom naming so tools treat it as parent
                if name in MODRES_RENAME:
                    newname, atom_renames = MODRES_RENAME[name]
                    res.name = newname
                    for a in res:
                        if a.name in atom_renames:
                            a.name = atom_renames[a.name]
                            a.element = gemmi.Element("S")
                n_prot += 1
                new_chain.add_residue(res)
                ca = None
                for a in res:
                    if a.name == "CA" and (ca is None or a.occ > ca.occ):
                        ca = a
                if ca is not None:
                    residues.append(ResidueRec(
                        seqnum=res.seqid.num,
                        icode=(res.seqid.icode.strip() or ""),
                        resname=res.name,
                        aa=AA3TO1[name],
                        ca=(ca.pos.x, ca.pos.y, ca.pos.z),
                    ))
            elif name in NUC:
                n_nuc += 1
            else:
                n_other += 1

        kind = _classify(n_prot, n_nuc, n_other)
        # sequence built only from CA-bearing (observed) protein residues
        seq = "".join(r.aa for r in residues)
        chains[chain.name] = ChainData(
            chain_id=chain.name, kind=kind, seq=seq, residues=residues,
            n_protein=n_prot, n_nucleic=n_nuc, n_other=n_other,
        )
        if len(new_chain) > 0:
            new_model.add_chain(new_chain)

    clean_st = gemmi.Structure()
    clean_st.name = st.name
    try:
        clean_st.cell = st.cell
        clean_st.spacegroup_hm = st.spacegroup_hm
    except Exception:
        pass
    clean_st.add_model(new_model)

    return CleanStructure(source=path, st=clean_st, chains=chains, entry_id=entry_id)


def write_subset(clean: CleanStructure, chain_map: Dict[str, str], out_path: str,
                 keep_only: Optional[List[str]] = None) -> List[str]:
    """Write selected chains to a PDB, optionally renaming them.

    chain_map: {source_chain_id -> output_chain_id}. Order of output follows the
    insertion order of chain_map. Only protein chains present in the cleaned
    structure are written. Returns the list of output chain ids actually written.
    """
    src_model = clean.st[0]
    out_model = gemmi.Model("1")
    written: List[str] = []
    for src_id, out_id in chain_map.items():
        if keep_only is not None and src_id not in keep_only:
            continue
        if src_id not in [c.name for c in src_model]:
            continue
        src_chain = src_model[src_id]
        new_chain = gemmi.Chain(out_id)
        for res in src_chain:
            new_chain.add_residue(res)
        if len(new_chain) > 0:
            out_model.add_chain(new_chain)
            written.append(out_id)
    out_st = gemmi.Structure()
    out_st.name = clean.st.name
    out_st.add_model(out_model)
    out_st.write_pdb(out_path)
    return written


def write_single_chain(clean: CleanStructure, src_chain_id: str, out_path: str,
                       out_chain_id: str = "A") -> bool:
    return len(write_subset(clean, {src_chain_id: out_chain_id}, out_path)) == 1
