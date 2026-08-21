#!/usr/bin/env python3
"""
chain_min_distance.py
=====================

For a protein complex structure (PDB or mmCIF), report — for every unordered
PAIR of chains — the MINIMUM distance between them, i.e. how close the two
chains come to each other ("closest contact").

    small value  -> the two chains touch / share an interface (or clash)
    large value  -> the two chains are far apart

By default the distance is the minimum Cα–Cα distance (backbone resolution:
fast, robust, and the standard coarse contact metric). Use --all-atom for the
true closest atomic approach (all heavy protein atoms).

Only PROTEIN residues are used. Waters, ions (including the classic calcium-ion
"CA" trap), ligands and nucleic-acid chains are ignored. Modified residues that
are still part of the protein chain (e.g. selenomethionine MSE) are kept.

Output: one tidy CSV per input structure, one row per unordered chain pair.
The row also records WHICH residue pair realizes the minimum — the putative
closest-contact site.

Usage
-----
    uv run 03_chain_min_distance.py INPUT [INPUT ...] \
        [--outdir DIR] [--out FILE] [--all-atom] [--matrix] [--model N]

    INPUT      one or more .pdb/.ent/.cif files, or a directory of them
    --outdir   output directory for the per-input CSV(s)   (default: .)
    --out      single output CSV path (only valid with ONE input)
    --all-atom use all heavy protein atoms instead of Cα only
    --matrix   also write a symmetric N×N distance-matrix CSV
    --model    model index to use                          (default: 0)

Dependencies: biopython, numpy, pandas.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser, is_aa
from Bio.PDB.PDBExceptions import PDBConstructionWarning

STRUCTURE_EXTS = {".pdb", ".ent", ".cif", ".mmcif"}


# ----------------------------------------------------------------------
# 1. Parsing and protein-atom selection
# ----------------------------------------------------------------------
def load_structure(path: Path):
    """Parse a .pdb/.ent or .cif/.mmcif file into a Biopython Structure."""
    if path.suffix.lower() in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        return parser.get_structure(path.stem, str(path))


def chain_protein_atoms(chain, all_atom: bool):
    """
    Collect protein-atom coordinates for one chain.

    Returns (coords, labels, n_res):
        coords : (N, 3) float64 array of atom coordinates
        labels : list of (resid, resname) aligned to `coords` rows
                 (in all-atom mode the residue label repeats for each atom)
        n_res  : number of protein residues used (== Cα count)

    A residue is kept iff it is an amino acid (Bio.PDB.is_aa, standard=False so
    modified residues like MSE count) AND it has a Cα atom. This automatically
    rejects waters, ions/metals (e.g. the calcium ion, whose residue/atom name
    is "CA" but is_aa("CA") is False), ligands and nucleic acids.
    """
    coords, labels, n_res = [], [], 0
    for res in chain.get_residues():
        if not is_aa(res, standard=False):
            continue
        if "CA" not in res:  # require a backbone Cα -> genuine protein residue
            continue
        n_res += 1
        resseq, icode = res.id[1], res.id[2].strip()
        resid = f"{resseq}{icode}"
        resname = res.resname.strip()
        if all_atom:
            for atom in res.get_atoms():
                if (atom.element or "").strip().upper() == "H":
                    continue  # heavy atoms only
                coords.append(atom.coord)
                labels.append((resid, resname))
        else:
            coords.append(res["CA"].coord)
            labels.append((resid, resname))
    arr = np.asarray(coords, dtype=float) if coords else np.zeros((0, 3), float)
    return arr, labels, n_res


# ----------------------------------------------------------------------
# 2. Minimum distance between two coordinate sets
# ----------------------------------------------------------------------
def min_pairwise_distance(A: np.ndarray, B: np.ndarray):
    """
    Minimum Euclidean distance between coordinate sets A (Na,3) and B (Nb,3).

    Returns (min_dist, ia, ib): the distance in Å and the row indices of the
    closest pair (ia in A, ib in B). Chunked over A so the temporary distance
    block stays bounded (~<50 MB) even for large all-atom chains.
    """
    if len(A) == 0 or len(B) == 0:
        return float("nan"), -1, -1
    block = max(1, 2_000_000 // max(1, len(B)))  # keep block*Nb bounded
    best2, best_ia, best_ib = np.inf, -1, -1
    for s in range(0, len(A), block):
        a = A[s:s + block]
        d2 = ((a[:, None, :] - B[None, :, :]) ** 2).sum(-1)  # (block, Nb)
        ia, ib = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if d2[ia, ib] < best2:
            best2, best_ia, best_ib = d2[ia, ib], s + int(ia), int(ib)
    return float(np.sqrt(best2)), best_ia, best_ib


# ----------------------------------------------------------------------
# 3. Per-structure analysis
# ----------------------------------------------------------------------
def analyze_structure(path: Path, all_atom: bool = False, model_idx: int = 0):
    """Return (dataframe, chain_ids, dist_col) for one structure file."""
    structure = load_structure(path)
    models = list(structure)
    if not models:
        raise ValueError(f"{path.name}: no models found")
    if model_idx >= len(models):
        raise IndexError(f"{path.name}: model {model_idx} absent (has {len(models)})")
    model = models[model_idx]

    # Keep only chains that contain >= 1 usable protein atom.
    chain_data = {}
    for chain in model:
        coords, labels, n_res = chain_protein_atoms(chain, all_atom)
        if len(coords) == 0:
            print(f"  [skip] {path.name} chain {chain.id!r}: no protein Cα",
                  file=sys.stderr)
            continue
        chain_data[chain.id] = (coords, labels, n_res)

    dist_col = "min_atom_dist_A" if all_atom else "min_ca_dist_A"
    chain_ids = list(chain_data.keys())
    rows = []
    for ci, cj in combinations(chain_ids, 2):
        Ai, Li, ni = chain_data[ci]
        Aj, Lj, nj = chain_data[cj]
        dmin, ia, ib = min_pairwise_distance(Ai, Aj)
        resid_i, resname_i = Li[ia] if ia >= 0 else ("", "")
        resid_j, resname_j = Lj[ib] if ib >= 0 else ("", "")
        rows.append({
            "structure": path.stem,
            "chain_i": ci, "chain_j": cj,
            dist_col: round(dmin, 3),
            "resid_i": resid_i, "resname_i": resname_i,
            "resid_j": resid_j, "resname_j": resname_j,
            "n_ca_i": ni, "n_ca_j": nj,
        })

    cols = ["structure", "chain_i", "chain_j", dist_col,
            "resid_i", "resname_i", "resid_j", "resname_j", "n_ca_i", "n_ca_j"]
    return pd.DataFrame(rows, columns=cols), chain_ids, dist_col


def write_matrix(df, chain_ids, dist_col, out_path):
    """Write a symmetric N×N min-distance matrix CSV (diagonal = 0)."""
    mat = pd.DataFrame(np.nan, index=chain_ids, columns=chain_ids, dtype=float)
    for _, r in df.iterrows():
        mat.loc[r["chain_i"], r["chain_j"]] = r[dist_col]
        mat.loc[r["chain_j"], r["chain_i"]] = r[dist_col]
    for c in chain_ids:
        mat.loc[c, c] = 0.0
    mat.to_csv(out_path)


# ----------------------------------------------------------------------
# 4. CLI
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Pairwise minimum inter-chain distance (closest contact) "
                    "for PDB/mmCIF complexes.")
    ap.add_argument("inputs", nargs="+", help=".pdb/.ent/.cif files or directories")
    ap.add_argument("--outdir", default=".", help="output dir for CSVs (default: .)")
    ap.add_argument("--out", default=None,
                    help="single output CSV path (only with one input)")
    ap.add_argument("--all-atom", action="store_true",
                    help="use all heavy protein atoms instead of Cα")
    ap.add_argument("--matrix", action="store_true",
                    help="also write a symmetric N×N distance-matrix CSV")
    ap.add_argument("--model", type=int, default=0,
                    help="model index to use (default: 0)")
    args = ap.parse_args(argv)

    # Expand inputs (accept files and directories).
    paths = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir()
                                if q.suffix.lower() in STRUCTURE_EXTS))
        else:
            paths.append(p)
    if not paths:
        sys.exit("No input structure files found.")
    if args.out and len(paths) > 1:
        sys.exit("--out works only with a single input; use --outdir for many.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "min_atom_dist" if args.all_atom else "min_ca_dist"

    for path in paths:
        if not path.exists():
            print(f"[warn] missing file: {path}", file=sys.stderr)
            continue
        df, chain_ids, dist_col = analyze_structure(
            path, all_atom=args.all_atom, model_idx=args.model)
        out_csv = Path(args.out) if args.out else outdir / f"{path.stem}_chain_{tag}.csv"
        df.to_csv(out_csv, index=False)
        if len(chain_ids) < 2:
            print(f"[warn] {path.name}: <2 protein chains -> header-only CSV "
                  f"({out_csv})", file=sys.stderr)
        elif args.matrix:
            write_matrix(df, chain_ids, dist_col,
                         outdir / f"{path.stem}_chain_{tag}_matrix.csv")


if __name__ == "__main__":
    main()
