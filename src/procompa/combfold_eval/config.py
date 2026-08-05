"""Configuration for the CombFold-vs-reference comparison pipeline.

All non-obvious choices live here so a user can see and change them in one place.
Every threshold is documented with the reason it exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    # ---- Locations -------------------------------------------------------
    # Root that contains one folder per complex with CombFold assembled models.
    combfold_base: str = "."
    # Where downloaded/parsed reference structures are cached (RCSB mmCIF).
    ref_cache: str = "./ref_cache"
    # Where all outputs (CSVs, per-complex JSON, log) are written.
    out_dir: str = "./combfold_eval_out"
    # Offline UniProt full-length sequences: CSV with columns uniprot_id,sequence.
    # If a UniProt id is missing here we fall back to the UniProt REST API.
    uniprot_seq_csv: Optional[str] = None

    # ---- External tools --------------------------------------------------
    usalign_bin: str = "USalign"      # US-align executable (TM-score)
    dockq_bin: str = "DockQ"          # DockQ v2 CLI (interface metric)
    # DockQ independently re-derives chain homology via its own sequence check
    # (default tolerance: 0 mismatches) before it will score any interface. A
    # single true residue difference between the CombFold model (built from the
    # full-length UniProt sequence) and the crystallized construct (expression
    # tags, strain polymorphisms, engineered point mutations -- all common in
    # real PDB entries) makes DockQ reject the chain and silently return zero
    # interfaces, even though our own sequence-based assignment already passed
    # `same_protein_identity`. We therefore tell DockQ to tolerate the same
    # small number of mismatches so its internal check is not stricter than the
    # gate this pipeline already applied. 10 residues covers typical construct
    # differences without being large enough to risk matching a genuinely
    # different (paralogous) chain -- the shared-subunit PDBs DockQ receives
    # already contain only chains our own alignment accepted as the same protein.
    dockq_allowed_mismatches: int = 10

    # ---- CombFold model discovery ---------------------------------------
    # Glob patterns (relative to a complex folder) used to find assembled models,
    # tried in order. The first pattern that yields files wins.
    cf_model_globs: tuple = (
        "assembled_results/output_clustered_*.pdb",
        "*_output.pdb",
        "output_clustered_*.pdb",
        "*.pdb",
    )
    cf_confidence_name: str = "confidence.txt"  # cluster_idx:score;... (sorted desc)

    # ---- Chain / sequence logic -----------------------------------------
    # Minimum polymer length (residues) for a chain to be treated as a real
    # subunit rather than a peptide/tag fragment.
    min_chain_len: int = 20
    # %identity (local alignment, over the shorter observed sequence) at/above
    # which a chain is assigned to a complex UniProt. Matches at or above
    # `same_protein_identity` are "same protein"; between homolog_identity and
    # same_protein_identity are accepted but flagged as homolog/paralog.
    same_protein_identity: float = 90.0
    homolog_identity: float = 30.0

    # ---- RMSD outlier rejection (mirrors PyMOL align cycles) -------------
    # cycles=0 -> RMSD over ALL co-observed CA (no rejection).
    # cycles=N -> up to N refit iterations, each rejecting CA pairs whose
    # post-fit deviation exceeds `outlier_cutoff` Angstrom, refitting on the rest.
    outlier_cycles: int = 5
    outlier_cutoff: float = 2.0  # Angstrom; PyMOL align default cutoff
    # Refinement floor: stop rejecting once the surviving set would drop below
    # max(outlier_min_atoms, min_frac_atoms_kept * n). Kept small so the refined
    # "core RMSD" is revealed (mirrors PyMOL align's aggressive rejection); the
    # surviving atom count (n_used) is always reported for interpretability.
    outlier_min_atoms: int = 10
    min_frac_atoms_kept: float = 0.0

    # ---- Reference acquisition ------------------------------------------
    rcsb_data_api: str = "https://data.rcsb.org/rest/v1/core/entry"
    rcsb_files: str = "https://files.rcsb.org/download"
    download_timeout_s: int = 60
    # If True, treat a user-supplied local file as the AU reference and still
    # try to download biological assemblies for form comparison.
    allow_local_reference: bool = True

    # ---- Behaviour flags -------------------------------------------------
    score_all_clusters: bool = True   # user choice: score every CombFold cluster
    report_both_ref_forms: bool = True  # score bio-assembly AND asymmetric unit
    # If True, copy the shared-subunit model/reference PDBs of the PRIMARY form
    # into out_dir/structures/ so the user can open the exact scored inputs.
    save_primary_structures: bool = False

    def resolved(self) -> "Config":
        """Return a copy with paths expanded to absolute."""
        d = asdict(self)
        for k in ("combfold_base", "ref_cache", "out_dir", "uniprot_seq_csv"):
            if d.get(k):
                d[k] = os.path.abspath(os.path.expanduser(d[k]))
        return Config(**d)


# Canonical 3-letter -> 1-letter amino-acid table, incl. common modified residues
# that we normalize to their parent (so they are kept, not stripped as ligands).
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # modified residues normalized to parent for sequence purposes
    "MSE": "M",  # selenomethionine -> Met (very common in crystal structures)
    "SEC": "C",  # selenocysteine
    "PYL": "K",  # pyrrolysine
    "HYP": "P",  # hydroxyproline
    "SEP": "S", "TPO": "T", "PTR": "Y",  # phospho- Ser/Thr/Tyr
    "MLY": "K", "M3L": "K", "ALY": "K",  # methyl/acetyl Lys
    "CSO": "C", "CME": "C", "CSD": "C",  # oxidized/modified Cys
    "KCX": "K", "LLP": "K",
}

# residues we rename so the heavy-atom naming matches the parent amino acid
MODRES_RENAME = {
    "MSE": ("MET", {"SE": "SD"}),
    "SEC": ("CYS", {"SE": "SG"}),
}
