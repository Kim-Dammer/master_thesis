#!/usr/bin/env python
"""
23_export_combined_pymol_session.py — Export ONE combined PyMOL session (.pse)
containing ALL filtered complexes, each in a collapsible group on a grid.

---------------
22_render_pdf_summary.py produces a pdf with all complexes, this creates a  single PyMOL
session file with all complexes that are not just pairs

WHAT IT DOES
------------
For each scored complex with >= MIN_SUBUNITS subunits (sum of stoichiometry):
  1. Load the biological assembly reference (same as render script).
  2. Superpose CombFold cluster-0 onto the reference (same Kabsch + Hungarian
     chain matching as render script and script 23).
  3. Collect all successfully-superposed complexes.
  4. Build ONE PyMOL script that:
     - creates a PyMOL group per complex (collapsible in the object panel)
     - loads reference + CombFold (transformed) into each group
     - colors CombFold chains vividly by UniProt
     - colors reference chains in pastel by UniProt, 40% transparent
     - shows non-protein molecules (ligands as sticks, ions as spheres,
       nucleic acids as cartoon)
     - translates each group to a unique grid position so all are visible
     - saves a single .pse session file
  5. Run PyMOL headless once to produce the .pse.

OBJECT NAMING (per complex)
---------------------------
PyMOL object/group names allow only [A-Za-z0-9_], no hyphens. CPX IDs like
"CPX-2896" become "CPX_2896".

  Group:              CPX_2896
  Reference object:   reference_1JCV_CPX_2896   (PDB + CPX for uniqueness)
  CombFold object:    CF_CPX_2896


GRID LAYOUT
-----------
Complexes are placed on a grid in the XY plane:
  - columns = ceil(sqrt(n_complexes))
  - spacing = 300 A (adjustable via --grid-spacing)
  - position(i) = (col * spacing, -row * spacing, 0)

NON-PROTEIN DISPLAY
-------------------
  - Ligands / cofactors (organic, non-polymer): sticks
  - Metal ions (Zn, Mg, Ca, Fe, Mn, Cu, Na, K, Cl, ...): spheres, scale 0.5
  - Nucleic acids (DNA/RNA): cartoon
  - Waters (HOH): hidden (clutter without insight)

FILTER
------
Default: complexes with >= 3 subunits (sum of stoichiometry).
  P00445(3)   = 3 subunits (homotrimer)  -> PASSES
  P15873(3)   = 3 subunits (homotrimer)  -> PASSES
  P00445(2)   = 2 subunits (homodimer)   -> excluded
  P10507(1),P11914(1) = 2 subunits (heterodimer) -> excluded
Adjust with --min-subunits.

Usage:
    uv run 23_export_pymol_sessions.py
    uv run 23_export_pymol_sessions.py --min-subunits 3
    uv run 23_export_pymol_sessions.py --grid-spacing 400
    uv run 23_export_pymol_sessions.py --keep-na          # include nucleic-acid refs
    uv run 23_export_pymol_sessions.py --pml-only         # skip PyMOL, just write .pml
    uv run 23_export_pymol_sessions.py --test-n 3         # first 3 only
"""
import os, re, sys, csv, json, time, urllib.request, subprocess, argparse, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Superimposer
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project paths (same convention as 22/23)
# ---------------------------------------------------------------------------
try:
    from procompa import get_project_root, get_data_dir
    PRJ_ROOT = get_project_root()
    data_dir = PRJ_ROOT / "data"
except ImportError:
    PRJ_ROOT = Path("/cluster/project/beltrao/kdammer/master_thesis")
    data_dir = PRJ_ROOT / "data"

# === CONFIG ===
CSV_PATH      = data_dir / "Pipeline/third_setup/third_setup_pipeline_complexes_combfold_results_with_exact_match_comparison_corrected_stoi_and_offset.csv"
COMBFOLD_DIR  = data_dir / "Pipeline/third_setup/CombFold"
ASSEMBLY_DIR  = data_dir / "all_Complex_pdb_files/_raw_assembly"
ASSEMBLY_LOG  = data_dir / "all_Complex_pdb_files/biological_assembly_download_log.csv"
PSE_DIR       = data_dir / "Pipeline/third_setup/pymol_sessions"
TMP_DIR       = "/tmp/pymol_combined_session"
PSE_PATH      = PSE_DIR / "all_complexes_combined.pse"
SUMMARY_CSV   = PSE_DIR / "combined_session_summary.csv"

# Distinct PyMOL colors for up to 14 proteins (vivid — for CombFold)
PROTEIN_COLORS = ["red", "blue", "green", "yellow", "violet", "cyan", "magenta",
                  "orange", "salmon", "lime", "pink", "wheat", "teal", "purple"]
# Pastel equivalents for the reference (so CombFold stands out in the overlay)
PASTEL_RGB = {
    "red":     [1.00, 0.60, 0.60],
    "blue":    [0.55, 0.70, 1.00],
    "green":   [0.60, 0.90, 0.60],
    "yellow":  [1.00, 0.95, 0.55],
    "violet":  [0.75, 0.65, 1.00],
    "cyan":    [0.60, 0.90, 0.95],
    "magenta": [1.00, 0.65, 0.90],
    "orange":  [1.00, 0.75, 0.50],
    "salmon":  [1.00, 0.70, 0.65],
    "lime":    [0.75, 0.95, 0.55],
    "pink":    [1.00, 0.80, 0.85],
    "wheat":   [0.95, 0.88, 0.70],
    "teal":    [0.55, 0.85, 0.85],
    "purple":  [0.80, 0.65, 0.90],
}
# Metal ions to show as spheres (common biological metals + halides)
ION_RESN = "ZN+MG+CA+FE+FE2+MN+CU+NA+K+CL+BR+I+F+CO+NI+CD+HG+SR+BA+RB+CS+LI"
# =============

_pdb_parser = PDBParser(QUIET=True)
_cif_parser = MMCIFParser(QUIET=True)


# ---------------------------------------------------------------------------
# Structure loading + chain helpers  (identical to script 23)
# ---------------------------------------------------------------------------

def load_structure(path):
    p = Path(path)
    if p.suffix.lower() == ".cif":
        return _cif_parser.get_structure(p.stem, str(p))
    return _pdb_parser.get_structure(p.stem, str(p))

def chain_ca_atoms(chain):
    return [res["CA"] for res in chain if res.id[0] == " " and "CA" in res]

def chain_ca_map(chain):
    """Return {resid_number: np.array(x,y,z)} for standard residues with CA."""
    out = {}
    for res in chain:
        if res.id[0] == " " and "CA" in res:
            out[res.id[1]] = res["CA"].get_coord()
    return out

def chain_length(chain):
    return len(chain_ca_atoms(chain))


# ---------------------------------------------------------------------------
# Assembly file discovery  (identical to script 23)
# ---------------------------------------------------------------------------

def load_assembly_log(log_path):
    """Load biological_assembly_download_log.csv into {pdb_id: {assembly_id, path, status}}."""
    log = {}
    if not log_path.exists():
        print(f"  [WARN] Assembly log not found: {log_path}")
        return log
    with open(log_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["pdb_id"]:
                log[row["pdb_id"].upper()] = {
                    "assembly_id": row["assembly_id"],
                    "path": row["path"],
                    "status": row["status"],
                }
    return log

def find_assembly_file(pdb_id, asm_log, asm_dir):
    """Find the biological assembly file for a PDB ID.
    Returns (Path, assembly_id, format) or (None, None, None)."""
    pdb_id = pdb_id.upper()
    entry = asm_log.get(pdb_id)
    if entry and entry["status"] in ("ok", "skip") and entry["path"]:
        p = Path(entry["path"])
        if p.exists():
            fmt = "cif" if p.suffix.lower() == ".cif" else "pdb"
            return (p, entry["assembly_id"], fmt)
    for pattern in [f"{pdb_id}_asm*.pdb", f"{pdb_id}_asm*.cif"]:
        matches = sorted(asm_dir.glob(pattern))
        if matches:
            p = matches[0]
            asm_id = p.stem.split("_asm")[-1] if "_asm" in p.stem else "1"
            fmt = "cif" if p.suffix.lower() == ".cif" else "pdb"
            return (p, asm_id, fmt)
    return (None, None, None)


# ---------------------------------------------------------------------------
# SIFTS + UniProt  (identical to script 23)
# ---------------------------------------------------------------------------

_uniprot_len_cache = {}
def get_uniprot_length(uniprot):
    if uniprot in _uniprot_len_cache: return _uniprot_len_cache[uniprot]
    try:
        with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta", timeout=15) as r:
            fasta = r.read().decode()
        _uniprot_len_cache[uniprot] = len("".join(fasta.split("\n")[1:]))
    except Exception:
        _uniprot_len_cache[uniprot] = None
    time.sleep(0.05)
    return _uniprot_len_cache[uniprot]

_sifts_cache = {}
_residue_offset_cache = {}

def get_chain_offsets(pdb_id):
    """Query PDBe residue_listing to get {struct_asym_id: offset} where
    offset = author_residue_number - residue_number (label) for the first
    residue of each chain."""
    pdb_id = pdb_id.lower()
    if pdb_id in _residue_offset_cache: return _residue_offset_cache[pdb_id]
    offsets = {}
    url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/residue_listing/{pdb_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
        for mol in data.get(pdb_id, {}).get("molecules", []):
            for chain_info in mol.get("chains", []):
                asym = chain_info.get("struct_asym_id")
                residues = chain_info.get("residues", [])
                if residues:
                    first = residues[0]
                    label = first.get("residue_number")
                    author = first.get("author_residue_number")
                    if author is not None and label is not None:
                        offsets[asym] = author - label
    except Exception: pass
    _residue_offset_cache[pdb_id] = offsets
    time.sleep(0.1)
    return offsets

def sifts_chain_mapping(pdb_id):
    """Query SIFTS /mappings/uniprot. Returns {chain_id: {uniprot, pdb_to_uniprot}}.
    Shifts label_seq_id to author_residue_number (what Biopython uses)."""
    pdb_id = pdb_id.lower()
    if pdb_id in _sifts_cache: return _sifts_cache[pdb_id]
    rl_offsets = get_chain_offsets(pdb_id)
    chain_map = {}
    try:
        with urllib.request.urlopen(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}", timeout=20) as r:
            data = json.loads(r.read())
        for uniprot, info in data.get(pdb_id, {}).get("UniProt", {}).items():
            for m in info.get("mappings", []):
                ch = m.get("chain_id")
                struct_asym = m.get("struct_asym_id")
                if not ch: continue
                us, ue = m.get("unp_start"), m.get("unp_end")
                ps = m.get("start", {}).get("residue_number")
                pe = m.get("end", {}).get("residue_number")
                if None in (us, ue, ps, pe): continue
                author_start = m.get("start", {}).get("author_residue_number")
                if author_start is not None:
                    offset = author_start - ps
                else:
                    offset = rl_offsets.get(struct_asym, 0)
                p2u = {ps + i + offset: us + i for i in range(min(ue - us + 1, pe - ps + 1))}
                if ch in chain_map:
                    chain_map[ch]["pdb_to_uniprot"].update(p2u)
                else:
                    chain_map[ch] = {"uniprot": uniprot, "pdb_to_uniprot": p2u}
    except Exception: pass
    _sifts_cache[pdb_id] = chain_map
    time.sleep(0.1)
    return chain_map

def has_nucleic_acid_chains(pdb_id):
    """Check if a PDB entry contains DNA/RNA polymer chains."""
    pdb_id = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        for mol in data.get(pdb_id, []):
            mt = mol.get("molecule_type", "").lower()
            if "nucleotide" in mt:
                return True
        return False
    except Exception:
        return False

def parse_spec(spec):
    """'P05453(1),P12385(1)' -> {'P05453': 1, 'P12385': 1}"""
    counts = {}
    for m in re.finditer(r"([A-Za-z0-9_]+)\((\d+)\)", spec):
        counts[m.group(1)] = int(m.group(2))
    return counts


# ---------------------------------------------------------------------------
# Multi-MODEL assembly chain collection + Hungarian matching  (identical to 23)
# ---------------------------------------------------------------------------

def collect_assembly_chains(ref_struct, sifts_map):
    """Collect all protein chains from ALL models in the assembly structure.
    Filters to chains that have a SIFTS UniProt mapping.
    Handles PDB (multi-MODEL) and mmCIF (symmetry copies as A-2, A-3) formats."""
    chains = []
    for model in ref_struct.get_models():
        model_id = model.id
        for chain in model.get_chains():
            ch_id = chain.id
            sifts_entry = sifts_map.get(ch_id)
            if sifts_entry is None and "-" in ch_id:
                base_id = ch_id.split("-")[0]
                sifts_entry = sifts_map.get(base_id)
            if sifts_entry is None:
                continue
            ca_map = chain_ca_map(chain)
            if not ca_map:
                continue
            chains.append({
                "model_id": model_id,
                "chain_id": ch_id,
                "chain_obj": chain,
                "uniprot": sifts_entry["uniprot"],
                "sifts": sifts_entry,
                "ca_map": ca_map,
                "length": len(ca_map),
            })
    return chains

def kabsch_rmsd(P, Q):
    if len(P) == 0:
        return float("nan")
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((R @ Pc.T).T - Qc) ** 2, axis=1))))

def match_chains_optimal(cf_struct, ref_chain_list, uniprot_lengths):
    """Match CombFold chains to reference assembly chains using global Hungarian
    assignment to minimise total RMSD. Returns {cf_chain_id: (uniprot, ref_chain)}."""
    BIG = 1e6
    cf_chains = []
    for cf_chain in cf_struct.get_chains():
        ca_map = chain_ca_map(cf_chain)
        if not ca_map:
            continue
        cf_chains.append({"chain_id": cf_chain.id, "chain_obj": cf_chain, "ca_map": ca_map})
    if not cf_chains or not ref_chain_list:
        return {}
    n_cf = len(cf_chains)
    n_ref = len(ref_chain_list)
    rmsd_matrix = np.full((n_cf, n_ref), BIG)
    for i, cf_c in enumerate(cf_chains):
        for j, ref_c in enumerate(ref_chain_list):
            p2u = ref_c["sifts"]["pdb_to_uniprot"]
            ref_uniprot_ca = {p2u[r]: c for r, c in ref_c["ca_map"].items() if r in p2u}
            common = sorted(set(cf_c["ca_map"].keys()) & set(ref_uniprot_ca.keys()))
            if not common:
                continue
            cf_coords = np.array([cf_c["ca_map"][r] for r in common])
            ref_coords = np.array([ref_uniprot_ca[r] for r in common])
            rmsd_matrix[i, j] = kabsch_rmsd(cf_coords, ref_coords)
    size = max(n_cf, n_ref)
    padded = np.full((size, size), BIG)
    padded[:n_cf, :n_ref] = rmsd_matrix
    row_ind, col_ind = linear_sum_assignment(padded)
    matches = {}
    for ri, ci in zip(row_ind, col_ind):
        if ri >= n_cf or ci >= n_ref:
            continue
        if rmsd_matrix[ri, ci] >= BIG:
            continue
        uniprot = ref_chain_list[ci]["uniprot"]
        matches[cf_chains[ri]["chain_id"]] = (uniprot, ref_chain_list[ci])
    return matches


# ---------------------------------------------------------------------------
# Superposition (multi-MODEL aware)  (identical to script 23)
# ---------------------------------------------------------------------------

def superpose_combfold_on_reference(cf_path, ref_path, spec_uniprots, out_path):
    """Superpose CombFold onto reference assembly using optimal chain matching.
    Returns (rmsd, n_atoms)."""
    cf = load_structure(cf_path)
    ref = load_structure(ref_path)
    uniprot_lengths = {u: get_uniprot_length(u) for u in spec_uniprots}
    pdb_id = Path(ref_path).stem.split("_asm")[0]
    sifts_map = sifts_chain_mapping(pdb_id)
    ref_chain_list = collect_assembly_chains(ref, sifts_map)
    if not ref_chain_list:
        raise RuntimeError("No SIFTS-mapped chains found in assembly")
    matches = match_chains_optimal(cf, ref_chain_list, uniprot_lengths)
    if not matches:
        raise RuntimeError("No chains matched between CombFold and reference")
    cf_atoms, ref_atoms = [], []
    for cf_chain_id, (uniprot, ref_chain) in matches.items():
        cf_chain = cf[0][cf_chain_id]
        cf_ca = chain_ca_map(cf_chain)
        p2u = ref_chain["sifts"]["pdb_to_uniprot"]
        u2p = {u: p for p, u in p2u.items()}
        ref_uniprot_ca = {p2u[r]: c for r, c in ref_chain["ca_map"].items() if r in p2u}
        common = sorted(set(cf_ca.keys()) & set(ref_uniprot_ca.keys()))
        ref_model = ref[ref_chain["model_id"]]
        ref_ch = ref_model[ref_chain["chain_id"]]
        for pos in common:
            cf_res = cf_chain[pos]
            if cf_res.id[0] == " " and "CA" in cf_res:
                cf_atoms.append(cf_res["CA"])
            pdb_resnum = u2p.get(pos)
            if pdb_resnum is not None and pdb_resnum in ref_ch:
                ref_res = ref_ch[pdb_resnum]
                if ref_res.id[0] == " " and "CA" in ref_res:
                    ref_atoms.append(ref_res["CA"])
    if not cf_atoms:
        raise RuntimeError("No common CA atoms found after matching")
    sup = Superimposer()
    sup.set_atoms(ref_atoms, cf_atoms)
    sup.apply(list(cf.get_atoms()))
    io = PDBIO(); io.set_structure(cf); io.save(str(out_path))
    return sup.rms, len(cf_atoms)


# ---------------------------------------------------------------------------
# Chain-to-UniProt mapping for coloring  (identical to script 23)
# ---------------------------------------------------------------------------

def match_combfold_chains_to_uniprots(cf_path, spec_uniprots):
    """Returns {chain_id: uniprot} for CombFold chain coloring.
    Homomers handled: all N chains of a homo-N-mer get the SAME UniProt/color."""
    cf = load_structure(cf_path)
    uniprot_lengths = {u: get_uniprot_length(u) for u in spec_uniprots}
    chain_to_uniprot = {}
    for chain in cf.get_chains():
        cl = chain_length(chain)
        if cl == 0:
            continue
        best_u = None
        for u, ulen in uniprot_lengths.items():
            if ulen is not None and ulen == cl:
                best_u = u
                break
        if best_u is None:
            valid = {u: ul for u, ul in uniprot_lengths.items() if ul is not None}
            if valid:
                best_u = min(valid, key=lambda u: abs(valid[u] - cl))
        if best_u:
            chain_to_uniprot[chain.id] = best_u
    return chain_to_uniprot

def match_ref_chains_to_uniprots(ref_path, spec_uniprots):
    """Returns {chain_id: uniprot} for the reference assembly.
    Handles multi-MODEL: collects chains from all models."""
    ref = load_structure(ref_path)
    pdb_id = Path(ref_path).stem.split("_asm")[0]
    sifts_map = sifts_chain_mapping(pdb_id)
    ref_chain_list = collect_assembly_chains(ref, sifts_map)
    chain_to_uniprot = {}
    for rc in ref_chain_list:
        chain_to_uniprot[rc["chain_id"]] = rc["uniprot"]
    return chain_to_uniprot


# ---------------------------------------------------------------------------
# Name sanitisation  (NEW — PyMOL names allow only [A-Za-z0-9_])
# ---------------------------------------------------------------------------

def pymol_name(s):
    """Sanitise a name for PyMOL: replace hyphens and other invalid chars with _."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(s))


# ---------------------------------------------------------------------------
# Grid layout  (NEW)
# ---------------------------------------------------------------------------

def grid_position(i, n_cols, spacing):
    """Return (x, y, z) for complex index i (0-based) on a grid."""
    col = i % n_cols
    row = i // n_cols
    return (col * spacing, -row * spacing, 0.0)

def grid_columns(n_complexes):
    """Number of columns for a roughly square grid."""
    return max(1, math.ceil(math.sqrt(n_complexes)))


# ---------------------------------------------------------------------------
# Combined PML builder  (NEW — the core of this script)
# ---------------------------------------------------------------------------

def build_combined_pml(complexes_data, pse_path, spacing=300):
    """Build ONE PyMOL script that loads all complexes into a single session.

    complexes_data: list of dicts, each with keys:
        cpx, ref_pdb, ref_path, cf_transformed_path, tm,
        cf_chain_to_uniprot, ref_chain_to_uniprot, uniprot_colors
    pse_path: where to save the .pse
    spacing: grid spacing in Angstroms

    Returns the PML text.
    """
    n = len(complexes_data)
    n_cols = grid_columns(n)

    lines = []
    lines.append("# Auto-generated by 24_export_combined_pymol_session.py")
    lines.append(f"# {n} complexes, ALL AT ORIGIN (toggle visibility in object panel)")
    lines.append(f"# Only the first complex ({pymol_name(complexes_data[0]['cpx'])}) is visible initially.")
    lines.append("# Each complex is a collapsible PyMOL group with two objects:")
    lines.append("#   reference_{PDB}_{CPX}  (pastel, 40% transparent)")
    lines.append("#   CF_{CPX}               (vivid, superposed onto reference)")
    lines.append("")
    lines.append("bg_color white")
    lines.append("set ray_shadows, 0")
    lines.append("set antialias, 0")
    # Multi-MODEL fix: must be GLOBAL (per-object 'set all_states, on, <obj>' doesn't work)
    lines.append("set all_states, on")
    lines.append("")

    for i, cd in enumerate(complexes_data):
        cpx_raw = cd["cpx"]
        ref_pdb = cd["ref_pdb"]
        ref_path = cd["ref_path"]
        cf_path = cd["cf_transformed_path"]
        tm = cd["tm"]
        cf_chain_to_uniprot = cd["cf_chain_to_uniprot"]
        ref_chain_to_uniprot = cd["ref_chain_to_uniprot"]
        uniprot_colors = cd["uniprot_colors"]

        # Sanitise names (no hyphens allowed in PyMOL)
        cpx = pymol_name(cpx_raw)                 # CPX-2896 -> CPX_2896
        group_name = cpx                           # CPX_2896
        ref_obj = f"reference_{ref_pdb}_{cpx}"     # reference_1JCV_CPX_2896
        cf_obj = f"CF_{cpx}"                       # CF_CPX_2896

        x, y, z = grid_position(i, n_cols, spacing)

        lines.append(f"# === Complex {cpx_raw} (TM={tm:.4f}, ref={ref_pdb}) "
                     f"grid=({x:.0f},{y:.0f}) ===")
        lines.append(f"group {group_name}")
        lines.append(f"load {ref_path}, {ref_obj}")
        lines.append(f"load {cf_path}, {cf_obj}")
        lines.append(f"group {group_name}, {ref_obj}")
        lines.append(f"group {group_name}, {cf_obj}")

        # Multi-MODEL fix: all_states is a GLOBAL setting (per-object doesn't work)
        # Set once globally at the top of the PML, not here per-object.

        # Hide everything first, then show selectively
        lines.append(f"hide everything, {group_name}")

        # Protein + nucleic acid: cartoon on all polymer
        # NOTE: 'protein' is not a valid PyMOL selection keyword in all versions.
        # Using 'polymer' alone covers both protein and nucleic acid polymers.
        lines.append(f"show cartoon, {group_name} and polymer")

        # Non-protein: smart-by-type
        # Ligands / cofactors (organic, non-polymer, excluding waters) as sticks
        lines.append(f"show sticks, {group_name} and not polymer and not resn HOH")
        # Metal ions as spheres
        lines.append(f"show spheres, {group_name} and resn {ION_RESN}")
        lines.append(f"set sphere_scale, 0.5, {group_name} and resn {ION_RESN}")
        # Nucleic acids already shown as cartoon above (polymer includes NA).
        # If you want NA in a different color, color it here:
        # lines.append(f"color magenta, {group_name} and polymer and not name CA")

        # Color reference chains by UniProt (pastel)
        for chain_id, uniprot in ref_chain_to_uniprot.items():
            color = uniprot_colors.get(uniprot, "gray")
            pastel = PASTEL_RGB.get(color, [0.80, 0.80, 0.80])
            pastel_name = f"pastel_{color}"
            lines.append(f"set_color {pastel_name}, [{pastel[0]:.2f}, {pastel[1]:.2f}, {pastel[2]:.2f}]")
            lines.append(f"color {pastel_name}, {ref_obj} and chain {chain_id}")

        # Color CombFold chains by UniProt (vivid)
        for chain_id, uniprot in cf_chain_to_uniprot.items():
            color = uniprot_colors.get(uniprot, "gray")
            lines.append(f"color {color}, {cf_obj} and chain {chain_id}")

        # Reference semi-transparent so CombFold is visible inside it
        lines.append(f"set cartoon_transparency, 0.40, {ref_obj}")

        # All complexes at origin (no grid translation) — toggling between
        # complexes in the object panel keeps the same view position.
        lines.append("")

    # Finalise: hide all but first complex, zoom to first, save session
    # All complexes are at origin — toggle visibility in the object panel
    # to switch between them without losing your view position.
    lines.append("# === Finalise: only first complex visible ===")
    first_group = pymol_name(complexes_data[0]["cpx"])
    for cd in complexes_data[1:]:
        g = pymol_name(cd["cpx"])
        lines.append(f"disable {g}")
    lines.append(f"orient {first_group}")
    lines.append(f"zoom {first_group}")
    lines.append(f"save {pse_path}")
    lines.append("quit")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-subunits", type=int, default=3,
                    help="Minimum total number of subunits (sum of stoichiometry) in the "
                         "complex spec (default 3 = 'more than 2 subunits'). "
                         "P00445(3) = 3 subunits -> passes. P00445(2) = 2 subunits -> excluded.")
    ap.add_argument("--grid-spacing", type=int, default=300,
                    help="Grid spacing in Angstroms between complexes (default 300). "
                         "Increase if large complexes overlap.")
    ap.add_argument("--keep-na", action="store_true",
                    help="Include complexes whose reference PDB contains nucleic acid chains "
                         "(default: exclude them). With --keep-na, DNA/RNA is shown as cartoon.")
    ap.add_argument("--max-complexes", type=int, default=50,
                    help="Maximum number of complexes to include in one session (default 50). "
                         "Above this, PyMOL may struggle with memory. Top-N by TM-score are kept.")
    ap.add_argument("--pml-only", action="store_true",
                    help="Write the combined .pml but don't run PyMOL (for debugging).")
    ap.add_argument("--test-n", type=int, default=None,
                    help="Process only the first N complexes that pass the filter (for testing).")
    args = ap.parse_args()

    os.makedirs(PSE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Load assembly log
    print("Loading assembly download log...")
    asm_log = load_assembly_log(ASSEMBLY_LOG)
    print(f"  {len(asm_log)} entries in log.")

    # Load CSV
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    scored = df[df["comparison_status"] == "scored"].sort_values(
        "best_tm_score", ascending=False).reset_index(drop=True)
    print(f"Found {len(scored)} scored complexes in CSV.")

    # Filter: >= min_subunits subunits (sum of stoichiometry)
    filtered = []
    skipped_few_subunits = []
    for _, row in scored.iterrows():
        spec = row["comb_fold_submission"]
        spec_counts = parse_spec(spec)
        n_subunits = sum(spec_counts.values())
        if n_subunits < args.min_subunits:
            skipped_few_subunits.append((row["#Complex ac"], n_subunits, spec))
            continue
        filtered.append(row)

    print(f"\nFilter: >= {args.min_subunits} subunits -> {len(filtered)} complexes pass "
          f"({len(skipped_few_subunits)} excluded).")

    if skipped_few_subunits:
        print(f"  Excluded (fewer than {args.min_subunits} subunits):")
        for cpx, n, spec in skipped_few_subunits[:10]:
            print(f"    {cpx}: {n} subunits ({spec})")
        if len(skipped_few_subunits) > 10:
            print(f"    ... and {len(skipped_few_subunits) - 10} more")

    # Nucleic acid exclusion (unless --keep-na)
    if args.keep_na:
        clean_rows = filtered
        print(f"  --keep-na: keeping nucleic-acid references (DNA/RNA will show as cartoon).")
    else:
        clean_rows = []
        skipped_na = []
        for row in filtered:
            ref_pdb = row["best_tm_pdb"]
            if has_nucleic_acid_chains(ref_pdb):
                skipped_na.append((row["#Complex ac"], ref_pdb))
                continue
            clean_rows.append(row)
        print(f"  After nucleic acid exclusion: {len(clean_rows)} remain "
              f"({len(skipped_na)} excluded).")
        if skipped_na:
            print(f"  Excluded (nucleic acid in reference): " +
                  ", ".join(f"{c}({p})" for c, p in skipped_na))
            print(f"  (Use --keep-na to include them.)")

    # Cap at max_complexes
    if len(clean_rows) > args.max_complexes:
        print(f"\n  [WARN] {len(clean_rows)} complexes exceeds --max-complexes ({args.max_complexes}). "
              f"Keeping top {args.max_complexes} by TM-score.")
        print(f"  For larger sets, use 23_export_pymol_sessions.py (per-complex .pse files).")
        clean_rows = clean_rows[:args.max_complexes]

    if args.test_n:
        clean_rows = clean_rows[:args.test_n]
        print(f"\n[TEST MODE] Processing only first {args.test_n} complex(es).")

    print(f"\n{len(clean_rows)} complexes to include in the combined session.")
    print()

    # Superpose each complex and collect data for the combined PML
    complexes_data = []
    summary_rows = []
    success = fail = 0
    t0 = time.time()

    for i, row in enumerate(clean_rows, 1):
        cpx = row["#Complex ac"]
        tm = row["best_tm_score"]
        rmsd = row["best_rmsd"]
        ref_pdb = row["best_tm_pdb"]
        spec = row["comb_fold_submission"]

        # Completeness columns (from compare v3, may be absent in older CSVs)
        ref_complete_raw = row.get("ref_complete", "")
        if isinstance(ref_complete_raw, str):
            ref_complete = ref_complete_raw.lower() in ("true", "1", "yes")
        else:
            ref_complete = bool(ref_complete_raw)
        cf_stoic_str  = row.get("cf_stoichiometry", "")  or ""
        ref_stoic_str = row.get("ref_stoichiometry", "") or ""
        missing_str   = row.get("missing_uniprots", "")  or ""
        coverage_ratio = row.get("coverage_ratio", "")   or ""

        complete_tag = "" if ref_complete else " [MISMATCH]"
        spec_counts = parse_spec(spec)
        n_unique = len(spec_counts)
        n_subunits = sum(spec_counts.values())

        print(f"[{i}/{len(clean_rows)}] {cpx} ({n_unique} prots, {n_subunits} subunits, "
              f"TM={tm:.3f}, RMSD={rmsd:.2f}, ref={ref_pdb}{complete_tag})")

        # Find CombFold output
        spec_folder = "_".join(f"{u}x{spec_counts[u]}" for u in sorted(spec_counts))
        cf_path = Path(COMBFOLD_DIR) / f"{spec_folder}_output" / "assembled_results" / "output_clustered_0.pdb"
        if not cf_path.exists():
            print(f"  SKIP: CombFold file missing: {cf_path}")
            summary_rows.append({"complex": cpx, "ref_pdb": ref_pdb, "tm_score": tm,
                                 "status": "no_combfold", "grid_col": "", "grid_row": ""})
            fail += 1
            continue

        # Find assembly file
        ref_path, asm_id, asm_fmt = find_assembly_file(ref_pdb, asm_log, ASSEMBLY_DIR)
        if ref_path is None:
            print(f"  SKIP: assembly file not found for {ref_pdb}")
            summary_rows.append({"complex": cpx, "ref_pdb": ref_pdb, "tm_score": tm,
                                 "status": "no_assembly", "grid_col": "", "grid_row": ""})
            fail += 1
            continue

        # Superpose CombFold onto reference
        cf_transformed = Path(TMP_DIR) / f"{cpx}_combfold_transformed.pdb"
        try:
            rms, n_atoms = superpose_combfold_on_reference(cf_path, ref_path,
                                                           list(spec_counts.keys()),
                                                           cf_transformed)
        except Exception as e:
            print(f"  SUPERPOSE FAIL: {e}")
            summary_rows.append({"complex": cpx, "ref_pdb": ref_pdb, "tm_score": tm,
                                 "status": "superpose_failed", "grid_col": "", "grid_row": ""})
            fail += 1
            continue

        # Get chain-to-UniProt mappings for coloring
        spec_uniprots = list(dict.fromkeys(spec_counts.keys()))
        cf_chain_to_uniprot = match_combfold_chains_to_uniprots(cf_path, spec_uniprots)
        ref_chain_to_uniprot = match_ref_chains_to_uniprots(ref_path, spec_uniprots)

        # Assign colors per UniProt
        uniprot_colors = {}
        for idx, u in enumerate(sorted(spec_uniprots)):
            uniprot_colors[u] = PROTEIN_COLORS[idx % len(PROTEIN_COLORS)]

        # Collect for combined PML
        complexes_data.append({
            "cpx": cpx,
            "ref_pdb": ref_pdb,
            "ref_path": ref_path,
            "cf_transformed_path": cf_transformed,
            "tm": tm,
            "cf_chain_to_uniprot": cf_chain_to_uniprot,
            "ref_chain_to_uniprot": ref_chain_to_uniprot,
            "uniprot_colors": uniprot_colors,
        })
        success += 1
        elapsed = time.time() - t0
        print(f"  OK ({elapsed:.0f}s total, {success} superposed)")

        # Summary row (grid position assigned after we know the final count)
        summary_rows.append({
            "complex": cpx,
            "n_unique_proteins": n_unique,
            "n_subunits": n_subunits,
            "spec": spec,
            "ref_pdb": ref_pdb,
            "tm_score": round(tm, 4),
            "rmsd": round(rmsd, 2) if not pd.isna(rmsd) else "",
            "ref_complete": ref_complete,
            "cf_stoichiometry": cf_stoic_str,
            "ref_stoichiometry": ref_stoic_str,
            "missing_uniprots": missing_str,
            "coverage_ratio": coverage_ratio,
            "assembly_id": asm_id,
            "assembly_format": asm_fmt,
            "status": "ok",
            "grid_col": "",   # filled in below
            "grid_row": "",   # filled in below
        })

    if not complexes_data:
        print(f"\nNo complexes superposed successfully. Nothing to export.")
        sys.exit(1)

    # Assign grid positions now that we know the final count
    n_final = len(complexes_data)
    n_cols = grid_columns(n_final)
    for i, sd in enumerate(summary_rows):
        if sd["status"] == "ok":
            sd["grid_col"] = i % n_cols
            sd["grid_row"] = i // n_cols

    print(f"\n=== Superposition done: {success} ok, {fail} failed in {time.time()-t0:.0f}s ===")
    print(f"\nBuilding combined PML for {n_final} complexes "
          f"(grid: {n_cols} cols x {math.ceil(n_final/n_cols)} rows, "
          f"spacing {args.grid_spacing} A)...")

    # Build the combined PML
    pml = build_combined_pml(complexes_data, PSE_PATH, spacing=args.grid_spacing)
    pml_path = Path(TMP_DIR) / "combined_session.pml"
    pml_path.write_text(pml)
    print(f"  PML written: {pml_path} ({len(pml)} chars, {pml.count(chr(10))} lines)")

    if args.pml_only:
        print(f"\n[pml-only] Skipping PyMOL. PML at: {pml_path}")
    else:
        print(f"\nRunning PyMOL to build {PSE_PATH.name}...")
        try:
            result = subprocess.run(["pymol", "-c", "-q", str(pml_path)],
                                    capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"  [PyMOL err] returncode={result.returncode}")
                print(f"  stderr: {result.stderr[:500]}")
            elif PSE_PATH.exists() and PSE_PATH.stat().st_size > 0:
                size_mb = PSE_PATH.stat().st_size / (1024 * 1024)
                print(f"  OK -> {PSE_PATH} ({size_mb:.1f} MB)")
            else:
                print(f"  [Missing .pse] {PSE_PATH} not created or empty")
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] PyMOL hung (>600s) — try fewer complexes or --pml-only")

    # Write summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"\nSummary CSV: {SUMMARY_CSV}")

    print(f"\n=== Done in {time.time()-t0:.0f}s ===")
    print(f"\nCombined session: {PSE_PATH}")
    print(f"  Open in PyMOL: pymol {PSE_PATH.name}")
    print(f"  Each complex is a collapsible group in the object panel.")
    print(f"  Toggle reference_{'{PDB}'}_{'{CPX}'} visibility to switch between")
    print(f"  'CF alone' and 'superposed' views.")
    if not args.pml_only and PSE_PATH.exists():
        print(f"\n Made .pse file (and the summary CSV).")


if __name__ == "__main__":
    main()
