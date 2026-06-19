#!/usr/bin/env python
"""
Diagnose why CombFold succeeds for spec P00937(1),P00899(2) but fails for P00937(2),P00899(1).

Both runs use the SAME heterodimer PDB. The only difference is which homodimer
(P00937 or P00899) is paired with it.

This script tests two competing hypotheses for why direction matters:

  H1 (representative-structure mismatch):
      For the protein that appears as the homodimer, CombFold's representative
      is taken from the AFDB file, but the transformation between it and the
      partner has to be derived from the heterodimer file. If the protein's
      conformation differs between the two files (induced fit, domain motion),
      the cross-file alignment is geometrically bad and the resulting transform
      is wrong.

  H2 (interface collision):
      The AFDB homodimer might place its second copy in a position that
      sterically clashes with where the partner protein would dock based
      on the heterodimer's interface. The clash filter then discards every
      candidate assembly.

The script produces:

  - Cα RMSD (full chain) of the heterodimer's P00937 onto each chain of the
    AFDB P00937 homodimer, and same for P00899 (control).
  - Cα RMSD restricted to plDDT > 80 residues (matches CombFold's alignment
    policy from the methods section).
  - For each (homodimer, partner) combination, builds the predicted 3-chain
    assembly by overlay and reports the minimum heavy-atom distance between
    the duplicated chain and the partner — this is the steric-clash test.

USAGE:

  python diagnose_combfold_directionality.py \\
      --homodimer-p00937 /cluster/.../merged_pdbs/P00937.pdb \\
      --homodimer-p00899 /cluster/.../merged_pdbs/P00899.pdb \\
      --heterodimer     /cluster/.../first_setup/CombFold/P00899x1_P00937x2_input/pdbs/AFM_P00899_P00937_unrelaxed_rank_1_model_1.pdb

  (The heterodimer path can come from EITHER of the two run input folders;
   they're the same file.)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from Bio.PDB import PDBParser, Superimposer
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure
except ImportError:
    raise SystemExit(
        "biopython is required. In your venv:\n"
        "    pip install biopython\n"
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_chain_ca_coords(
    pdb_path: Path,
    chain_id: str,
    plddt_cutoff: float | None = None,
) -> tuple[list[Atom], np.ndarray]:
    """Return CA atoms and their coordinates from one chain.

    If plddt_cutoff is set, only residues whose CA B-factor exceeds the cutoff
    are returned (AF/AFM write plDDT into the B-factor column).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    chain = structure[0][chain_id]
    atoms = []
    for residue in chain:
        if residue.id[0] != " ":  # skip hetero residues
            continue
        if "CA" not in residue:
            continue
        ca = residue["CA"]
        if plddt_cutoff is not None and ca.get_bfactor() < plddt_cutoff:
            continue
        atoms.append(ca)
    coords = np.array([a.get_coord() for a in atoms])
    return atoms, coords


def match_by_residue_id(
    atoms_a: list[Atom], atoms_b: list[Atom]
) -> tuple[list[Atom], list[Atom]]:
    """Pair atoms by residue sequence ID. Drops residues missing from either."""
    by_id_a = {a.get_parent().id[1]: a for a in atoms_a}
    by_id_b = {a.get_parent().id[1]: a for a in atoms_b}
    common = sorted(set(by_id_a) & set(by_id_b))
    return [by_id_a[i] for i in common], [by_id_b[i] for i in common]


def superimpose_rmsd(
    fixed: list[Atom], moving: list[Atom]
) -> tuple[float, np.ndarray, np.ndarray]:
    """RMSD after optimal rigid-body superposition.

    Returns (rmsd, rotation_matrix, translation_vector).
    """
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    rot, trans = sup.rotran
    return sup.rms, rot, trans


def apply_transform(
    coords: np.ndarray, rot: np.ndarray, trans: np.ndarray
) -> np.ndarray:
    """Apply (rot, trans) to a coordinate array."""
    return coords @ rot + trans


def all_heavy_atom_coords(pdb_path: Path, chain_id: str) -> np.ndarray:
    """All non-hydrogen atom coordinates of one chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    chain = structure[0][chain_id]
    coords = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        for atom in residue:
            if atom.element != "H":
                coords.append(atom.get_coord())
    return np.array(coords)


def min_interchain_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Minimum pairwise distance between two coordinate arrays."""
    # broadcast to (n_a, n_b)
    # for n_a*n_b up to ~30M elements this is fine in memory
    if coords_a.size == 0 or coords_b.size == 0:
        return float("inf")
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return float(np.sqrt(d2.min()))


# --------------------------------------------------------------------------
# Conformational comparison (H1)
# --------------------------------------------------------------------------

def conformational_rmsd(
    homodimer_pdb: Path,
    heterodimer_pdb: Path,
    heterodimer_chain_for_this_protein: str,
    protein_label: str,
) -> None:
    """Print RMSDs of one protein's CA in heterodimer vs. each homodimer chain.

    Tests H1: does the protein's conformation differ between the two contexts?
    """
    print(f"\n--- Conformational RMSD: {protein_label} ---")

    # heterodimer has the protein in one chain (A or B depending on order in file)
    het_atoms_all, _ = load_chain_ca_coords(heterodimer_pdb, heterodimer_chain_for_this_protein)
    het_atoms_hi, _ = load_chain_ca_coords(
        heterodimer_pdb, heterodimer_chain_for_this_protein, plddt_cutoff=80.0
    )
    print(f"  heterodimer CA atoms: {len(het_atoms_all)} total, "
          f"{len(het_atoms_hi)} with plDDT>80")

    for homo_chain in ("A", "B"):
        try:
            hom_atoms_all, _ = load_chain_ca_coords(homodimer_pdb, homo_chain)
            hom_atoms_hi, _ = load_chain_ca_coords(homodimer_pdb, homo_chain, plddt_cutoff=80.0)
        except KeyError:
            print(f"  homodimer chain {homo_chain}: not present, skipping")
            continue

        # Match by residue ID and superimpose
        het_m, hom_m = match_by_residue_id(het_atoms_all, hom_atoms_all)
        if len(het_m) < 10:
            print(f"  homodimer chain {homo_chain}: only {len(het_m)} matched residues, skipping")
            continue
        rmsd_all, _, _ = superimpose_rmsd(het_m, hom_m)

        het_m_hi, hom_m_hi = match_by_residue_id(het_atoms_hi, hom_atoms_hi)
        rmsd_hi = None
        if len(het_m_hi) >= 10:
            rmsd_hi, _, _ = superimpose_rmsd(het_m_hi, hom_m_hi)

        msg = (
            f"  heterodimer.{heterodimer_chain_for_this_protein} "
            f"vs homodimer.{homo_chain}: "
            f"all-CA RMSD = {rmsd_all:.2f} A (n={len(het_m)})"
        )
        if rmsd_hi is not None:
            msg += f"  |  plDDT>80 RMSD = {rmsd_hi:.2f} A (n={len(het_m_hi)})"
        print(msg)


# --------------------------------------------------------------------------
# Interface / clash check (H2)
# --------------------------------------------------------------------------

def assembly_clash_check(
    homodimer_pdb: Path,
    heterodimer_pdb: Path,
    duplicated_protein_chain_in_homodimer_anchor: str,  # 'A'
    duplicated_protein_chain_in_homodimer_partner: str,  # 'B'
    duplicated_protein_chain_in_heterodimer: str,
    partner_protein_chain_in_heterodimer: str,
    duplicated_protein_label: str,
    partner_protein_label: str,
) -> None:
    """Build the predicted 3-chain assembly and report min interchain distance.

    Logic:
      - The homodimer fixes the relative orientation of the two duplicated
        chains: 'anchor' (chain A in the homodimer file) and 'partner' (chain B).
      - The heterodimer fixes the relative orientation of the duplicated protein
        to the partner protein.
      - Align the heterodimer's duplicated chain onto the homodimer's anchor chain.
        This rigid-body transform also moves the partner protein into its
        predicted position relative to the anchor.
      - Now check the steric distance between the homodimer's partner chain
        (the second duplicated protein) and the moved partner protein. If this
        is < 2.0 A on heavy atoms anywhere, the candidate assembly has a clash
        and CombFold will reject it.
    """
    print(f"\n--- Clash check: {duplicated_protein_label}(x2) + {partner_protein_label}(x1) ---")

    # Step 1: align heterodimer.duplicated -> homodimer.anchor by CA, plDDT>80
    het_dup_hi, _ = load_chain_ca_coords(
        heterodimer_pdb, duplicated_protein_chain_in_heterodimer, plddt_cutoff=80.0
    )
    hom_anchor_hi, _ = load_chain_ca_coords(
        homodimer_pdb, duplicated_protein_chain_in_homodimer_anchor, plddt_cutoff=80.0
    )
    het_m, hom_m = match_by_residue_id(het_dup_hi, hom_anchor_hi)
    if len(het_m) < 10:
        # fall back to all-CA
        het_dup_all, _ = load_chain_ca_coords(
            heterodimer_pdb, duplicated_protein_chain_in_heterodimer
        )
        hom_anchor_all, _ = load_chain_ca_coords(
            homodimer_pdb, duplicated_protein_chain_in_homodimer_anchor
        )
        het_m, hom_m = match_by_residue_id(het_dup_all, hom_anchor_all)

    if len(het_m) < 10:
        print(f"  too few matched residues to align, skipping")
        return

    rmsd, rot, trans = superimpose_rmsd(hom_m, het_m)
    print(f"  aligned heterodimer.{duplicated_protein_chain_in_heterodimer} "
          f"onto homodimer.{duplicated_protein_chain_in_homodimer_anchor}: "
          f"RMSD = {rmsd:.2f} A (n={len(het_m)})")

    # Step 2: apply same transform to the partner protein in the heterodimer
    partner_coords = all_heavy_atom_coords(
        heterodimer_pdb, partner_protein_chain_in_heterodimer
    )
    partner_moved = apply_transform(partner_coords, rot, trans)

    # Step 3: get the second copy of the duplicated protein from the homodimer
    second_dup_coords = all_heavy_atom_coords(
        homodimer_pdb, duplicated_protein_chain_in_homodimer_partner
    )

    # Step 4: compute minimum heavy-atom distance
    min_dist = min_interchain_distance(partner_moved, second_dup_coords)
    print(f"  min heavy-atom distance: "
          f"moved {partner_protein_label} vs. homodimer.{duplicated_protein_chain_in_homodimer_partner} "
          f"({duplicated_protein_label} copy 2) = {min_dist:.2f} A")
    if min_dist < 2.0:
        print(f"  >>> CLASH (< 2.0 A) — this assembly would be rejected by CombFold's clash filter")
    elif min_dist < 3.5:
        print(f"  >>> borderline (2-3.5 A) — assembly may or may not survive clash filter")
    else:
        print(f"  no clash detected (>= 3.5 A)")


# --------------------------------------------------------------------------
# Heterodimer chain inspection
# --------------------------------------------------------------------------

def report_heterodimer_chains(heterodimer_pdb: Path) -> dict[str, int]:
    """Return chain_id -> residue count, so we can identify which chain is which."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("het", str(heterodimer_pdb))
    counts = {}
    for chain in structure[0]:
        nres = sum(1 for r in chain if r.id[0] == " " and "CA" in r)
        counts[chain.id] = nres
    print(f"Heterodimer ({heterodimer_pdb.name}) chains: {counts}")
    return counts


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--homodimer-p00937", type=Path, required=True)
    ap.add_argument("--homodimer-p00899", type=Path, required=True)
    ap.add_argument(
        "--heterodimer",
        type=Path,
        required=True,
        help="AFM heterodimer PDB (same in both runs). "
             "Should contain one chain of P00899 and one of P00937.",
    )
    args = ap.parse_args()

    # Identify which chain in the heterodimer is which protein by residue count.
    # P00937 = 484 aa, P00899 = 507 aa.
    counts = report_heterodimer_chains(args.heterodimer)
    chain_p00937 = next((c for c, n in counts.items() if abs(n - 484) <= 5), None)
    chain_p00899 = next((c for c, n in counts.items() if abs(n - 507) <= 5), None)
    if chain_p00937 is None or chain_p00899 is None:
        raise SystemExit(
            f"Could not identify P00937/P00899 chains in heterodimer. "
            f"Expected 484 and 507 residues; got {counts}"
        )
    print(f"Heterodimer chain assignment: P00937={chain_p00937}, P00899={chain_p00899}")

    # ----- H1: conformational comparison -----
    conformational_rmsd(
        homodimer_pdb=args.homodimer_p00937,
        heterodimer_pdb=args.heterodimer,
        heterodimer_chain_for_this_protein=chain_p00937,
        protein_label="P00937",
    )
    conformational_rmsd(
        homodimer_pdb=args.homodimer_p00899,
        heterodimer_pdb=args.heterodimer,
        heterodimer_chain_for_this_protein=chain_p00899,
        protein_label="P00899",
    )

    # ----- H2: clash test for both directions -----
    # Direction A: P00937(x2) + P00899(x1)  [the FAILED case]
    assembly_clash_check(
        homodimer_pdb=args.homodimer_p00937,
        heterodimer_pdb=args.heterodimer,
        duplicated_protein_chain_in_homodimer_anchor="A",
        duplicated_protein_chain_in_homodimer_partner="B",
        duplicated_protein_chain_in_heterodimer=chain_p00937,
        partner_protein_chain_in_heterodimer=chain_p00899,
        duplicated_protein_label="P00937",
        partner_protein_label="P00899",
    )
    # Direction B: P00899(x2) + P00937(x1)  [the SUCCEEDED case]
    assembly_clash_check(
        homodimer_pdb=args.homodimer_p00899,
        heterodimer_pdb=args.heterodimer,
        duplicated_protein_chain_in_homodimer_anchor="A",
        duplicated_protein_chain_in_homodimer_partner="B",
        duplicated_protein_chain_in_heterodimer=chain_p00899,
        partner_protein_chain_in_heterodimer=chain_p00937,
        duplicated_protein_label="P00899",
        partner_protein_label="P00937",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
