#!/usr/bin/env python
"""
22_render_pdf_summary.py — Render a PDF summary comparing CombFold-predicted
protein complex structures against their best-matching experimental PDB
reference structures.

For each scored complex (sorted by TM-score, highest first), the script:
  1. Loads the CombFold prediction and the stoichiometry-matched PDB reference
     (biological assembly or asymmetric unit, as selected by the mapping pipeline).
  2. Matches CombFold chains to reference chains via the Hungarian algorithm
     on per-chain RMSD, using SIFTS residue mappings to align residues.
  3. Superimposes the CombFold prediction onto the reference (Kabsch).
  4. Renders three PyMOL views at a shared orientation: CombFold alone,
     reference alone, and superposed (reference in pastel colors).
  5. Composes a one-page-per-complex PDF with the three views plus a text
     panel showing TM-score, RMSD, reference PDB, assembly ID, stoichiometry,
     and a per-protein color legend.

Proteins are colored consistently across all panels (same UniProt = same color).
PDBs containing nucleic acid chains are excluded by default.

Usage:
    uv run 22_render_pdf_summary.py
"""
import os, re, sys, csv, json, time, urllib.request, subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Superimposer
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")

try:
    from procompa import get_project_root, get_data_dir
    PRJ_ROOT = get_project_root()
    data_dir = PRJ_ROOT / "data"
except Exception:
    data_dir = Path("/cluster/project/beltrao/kdammer/master_thesis/data")

# === CONFIG ===
CSV_PATH     = data_dir / "Pipeline/third_setup/final_third_setup_pipeline_rmsd_tm.csv"
COMBFOLD_DIR = data_dir / "Pipeline/third_setup/CombFold"
ASSEMBLY_DIR = data_dir / "complete_complex_pdb_mapping/_raw_assembly"
ASSEMBLY_LOG = data_dir / "complete_complex_pdb_mapping/structure_download_log.csv"
OUT_DIR      = data_dir / "Pipeline/third_setup/final_superposition_figures"
TMP_DIR      = "/tmp/superpose_pdf_v3"
PDF_PATH     = data_dir / "Pipeline/third_setup/summary_combfold_third_setup.pdf"
IMG_W, IMG_H = 800, 600

PROTEIN_COLORS = ["red", "blue", "green", "yellow", "violet", "cyan", "magenta",
                  "orange", "salmon", "lime", "pink", "wheat", "teal", "purple"]
PASTEL_RGB = {
    "red": [1.00, 0.60, 0.60], "blue": [0.55, 0.70, 1.00],
    "green": [0.60, 0.90, 0.60], "yellow": [1.00, 0.95, 0.55],
    "violet": [0.75, 0.65, 1.00], "cyan": [0.60, 0.90, 0.95],
    "magenta": [1.00, 0.65, 0.90], "orange": [1.00, 0.75, 0.50],
    "salmon": [1.00, 0.70, 0.65], "lime": [0.75, 0.95, 0.55],
    "pink": [1.00, 0.80, 0.85], "wheat": [0.95, 0.88, 0.70],
    "teal": [0.55, 0.85, 0.85], "purple": [0.80, 0.65, 0.90],
}
COLOR_HEX = {
    "red": "#FF0000", "blue": "#0000FF", "green": "#00AA00",
    "yellow": "#D4AA00", "violet": "#8B00FF", "cyan": "#00AAAA",
    "magenta": "#FF00FF", "orange": "#FFA500", "salmon": "#FA8072",
    "lime": "#7CFC00", "pink": "#FFC0CB", "wheat": "#D2B48C",
    "teal": "#008080", "purple": "#800080",
}
# =============

_pdb_parser = PDBParser(QUIET=True)
_cif_parser = MMCIFParser(QUIET=True)

# --- Helpers ---

def load_structure(path):
    p = Path(path)
    return _cif_parser.get_structure(p.stem, str(p)) if p.suffix.lower() == ".cif" \
           else _pdb_parser.get_structure(p.stem, str(p))

def chain_ca_map(chain):
    return {r.id[1]: r["CA"].get_coord() for r in chain if r.id[0] == " " and "CA" in r}

def chain_length(chain):
    return sum(1 for r in chain if r.id[0] == " " and "CA" in r)

def kabsch_rmsd(P, Q):
    if len(P) == 0: return float("nan")
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((R @ Pc.T).T - Qc) ** 2, axis=1))))

def parse_spec(spec):
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"([A-Za-z0-9_]+)\((\d+)\)", spec)}

# --- Assembly file discovery ---

def load_assembly_log(log_path):
    if not log_path.exists():
        print(f"  [WARN] Assembly log not found: {log_path}"); return {}
    return {r["pdb_id"].upper(): {"assembly_id": r["assembly_id"], "path": r["path"], "status": r["status"]}
            for r in csv.DictReader(open(log_path, newline="")) if r["pdb_id"]}

def find_assembly_file(pdb_id, asm_log, asm_dir):
    """Find the biological assembly or ASU file for a PDB ID.
    Returns (Path, assembly_id, format) or (None, None, None).
    File naming: {pdb_id}_asm{N}.pdb (assembly) or {pdb_id}_asu.pdb (ASU)."""
    pdb_id = pdb_id.upper()
    entry = asm_log.get(pdb_id)
    if entry and entry["status"] in ("ok", "skip") and entry["path"]:
        p = Path(entry["path"])
        if p.exists(): return (p, entry["assembly_id"], "cif" if p.suffix.lower() == ".cif" else "pdb")
    for pat in [f"{pdb_id}_asm*.pdb", f"{pdb_id}_asm*.cif", f"{pdb_id}_asu.pdb", f"{pdb_id}_asu.cif"]:
        matches = sorted(asm_dir.glob(pat))
        if matches:
            p = matches[0]
            asm_id = p.stem.split("_asm")[-1] if "_asm" in p.stem else "0"
            return (p, asm_id, "cif" if p.suffix.lower() == ".cif" else "pdb")
    return (None, None, None)

# --- SIFTS + UniProt ---

_sifts_cache, _offset_cache, _len_cache = {}, {}, {}

def get_chain_offsets(pdb_id):
    """PDBe residue_listing -> {struct_asym_id: offset} where offset = author - label.
    Needed because SIFTS author_residue_number is null for ~62% of chains."""
    pdb_id = pdb_id.lower()
    if pdb_id in _offset_cache: return _offset_cache[pdb_id]
    offsets = {}
    try:
        with urllib.request.urlopen(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/residue_listing/{pdb_id}", timeout=30) as r:
            for mol in json.loads(r.read()).get(pdb_id, {}).get("molecules", []):
                for ch in mol.get("chains", []):
                    res = ch.get("residues", [])
                    if res:
                        a, l = res[0].get("author_residue_number"), res[0].get("residue_number")
                        if a is not None and l is not None: offsets[ch.get("struct_asym_id")] = a - l
    except Exception: pass
    _offset_cache[pdb_id] = offsets; time.sleep(0.1); return offsets

def sifts_chain_mapping(pdb_id):
    """SIFTS /mappings/uniprot -> {chain_id: {uniprot, pdb_to_uniprot}}.
    Shifts label_seq_id to author_residue_number (what Biopython uses)."""
    key = pdb_id.lower()
    if key in _sifts_cache: return _sifts_cache[key]
    rl_offsets = get_chain_offsets(pdb_id)
    chain_map = {}
    try:
        with urllib.request.urlopen(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{key}", timeout=20) as r:
            data = json.loads(r.read())
        for uniprot, info in data.get(key, {}).get("UniProt", {}).items():
            for m in info.get("mappings", []):
                ch = m.get("chain_id")
                if not ch: continue
                us, ue = m.get("unp_start"), m.get("unp_end")
                ps, pe = m.get("start", {}).get("residue_number"), m.get("end", {}).get("residue_number")
                if None in (us, ue, ps, pe): continue
                author_start = m.get("start", {}).get("author_residue_number")
                offset = author_start - ps if author_start is not None else rl_offsets.get(m.get("struct_asym_id"), 0)
                p2u = {ps + i + offset: us + i for i in range(min(ue - us + 1, pe - ps + 1))}
                if ch in chain_map:
                    chain_map[ch]["pdb_to_uniprot"].update(p2u)
                else:
                    chain_map[ch] = {"uniprot": uniprot, "pdb_to_uniprot": p2u}
    except Exception: pass
    _sifts_cache[key] = chain_map; time.sleep(0.1); return chain_map

def get_uniprot_length(uniprot):
    if uniprot in _len_cache: return _len_cache[uniprot]
    try:
        with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta", timeout=15) as r:
            _len_cache[uniprot] = len("".join(r.read().decode().split("\n")[1:]))
    except Exception:
        _len_cache[uniprot] = None
    time.sleep(0.05); return _len_cache[uniprot]

def has_nucleic_acid_chains(pdb_id):
    """Check if a PDB entry contains DNA/RNA polymer chains."""
    pdb_id = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        return any("nucleotide" in mol.get("molecule_type", "").lower() for mol in data.get(pdb_id, []))
    except Exception:
        return False

# --- Assembly chain collection (multi-MODEL + mmCIF symmetry aware) ---

def collect_assembly_chains(ref_struct, sifts_map):
    """Collect protein chains from ALL models. Handles PDB multi-MODEL symmetry
    copies and mmCIF -N chain IDs (e.g. 'A-2' -> base 'A' for SIFTS lookup)."""
    chains = []
    for model in ref_struct.get_models():
        for chain in model.get_chains():
            entry = sifts_map.get(chain.id)
            if entry is None and "-" in chain.id:
                entry = sifts_map.get(chain.id.split("-")[0])
            if entry is None: continue
            ca_map = chain_ca_map(chain)
            if ca_map:
                chains.append({"model_id": model.id, "chain_id": chain.id, "uniprot": entry["uniprot"],
                               "sifts": entry, "ca_map": ca_map, "length": len(ca_map)})
    return chains

# --- Chain matching (Hungarian algorithm) ---

def match_chains_optimal(cf_struct, ref_chain_list):
    """Global RMSD matrix -> linear_sum_assignment. Fixes same-length UniProt
    collision bug by finding optimal pairing across all chains at once.
    Returns {cf_chain_id: (uniprot, ref_chain_dict)} or {} if no matches."""
    cf_chains = [{"id": c.id, "ca": chain_ca_map(c)} for c in cf_struct.get_chains() if chain_ca_map(c)]
    if not cf_chains or not ref_chain_list: return {}
    n_cf, n_ref = len(cf_chains), len(ref_chain_list)
    BIG = 1e6
    rmsd_mat = np.full((n_cf, n_ref), BIG)
    for i, cf in enumerate(cf_chains):
        for j, ref in enumerate(ref_chain_list):
            p2u = ref["sifts"]["pdb_to_uniprot"]
            ref_u_ca = {p2u[r]: c for r, c in ref["ca_map"].items() if r in p2u}
            common = sorted(set(cf["ca"]) & set(ref_u_ca))
            if common:
                rmsd_mat[i, j] = kabsch_rmsd(np.array([cf["ca"][r] for r in common]),
                                             np.array([ref_u_ca[r] for r in common]))
    size = max(n_cf, n_ref)
    padded = np.full((size, size), BIG); padded[:n_cf, :n_ref] = rmsd_mat
    row_ind, col_ind = linear_sum_assignment(padded)
    matches = {}
    for ri, ci in zip(row_ind, col_ind):
        if ri < n_cf and ci < n_ref and rmsd_mat[ri, ci] < BIG:
            matches[cf_chains[ri]["id"]] = (ref_chain_list[ci]["uniprot"], ref_chain_list[ci])
    return matches

# --- Superposition + coloring (single load, single SIFTS fetch) ---

def superpose_and_color(cf_path, ref_path, spec_uniprots, out_path):
    """Superpose CombFold onto reference, save transformed PDB, and return
    chain-to-UniProt color mappings for both CF and reference.

    Returns (rmsd, n_atoms, cf_chain_to_uniprot, ref_chain_to_uniprot).
    Loads each structure once, fetches SIFTS once, collects chains once,
    matches once — then reuses the matching for coloring (zero extra API calls)."""
    cf = load_structure(cf_path)
    ref = load_structure(ref_path)

    pdb_id = Path(ref_path).stem.split("_asm")[0].split("_asu")[0]
    sifts_map = sifts_chain_mapping(pdb_id)
    ref_chain_list = collect_assembly_chains(ref, sifts_map)
    if not ref_chain_list:
        raise RuntimeError("No SIFTS-mapped chains found in assembly")

    matches = match_chains_optimal(cf, ref_chain_list)
    if not matches:
        raise RuntimeError("No chains matched between CombFold and reference")

    # Build paired atom lists for Superimposer
    cf_atoms, ref_atoms = [], []
    for cf_chain_id, (uniprot, ref_chain) in matches.items():
        cf_chain = cf[0][cf_chain_id]
        cf_ca = chain_ca_map(cf_chain)
        p2u = ref_chain["sifts"]["pdb_to_uniprot"]
        u2p = {u: p for p, u in p2u.items()}  # inverse: UniProt -> PDB resnum
        ref_uniprot_ca = {p2u[r]: c for r, c in ref_chain["ca_map"].items() if r in p2u}
        common = sorted(set(cf_ca) & set(ref_uniprot_ca))
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

    # Coloring: extract directly from matches and ref_chain_list
    cf_chain_to_uniprot = {cf_id: uniprot for cf_id, (uniprot, _) in matches.items()}
    ref_chain_to_uniprot = {rc["chain_id"]: rc["uniprot"] for rc in ref_chain_list}

    return sup.rms, len(cf_atoms), cf_chain_to_uniprot, ref_chain_to_uniprot

# --- PyMOL rendering ---

def render_three_views_pymol(ref_path, cf_transformed_path, out_dir, cpx_id,
                              cf_chain_to_uniprot, ref_chain_to_uniprot, uniprot_colors):
    """ONE PyMOL session -> 3 PNGs at 800x600, same orientation, per-protein coloring.
    In the superposed view, the reference is rendered in pastel versions of the
    same per-protein colors so CombFold (vivid) and reference (pastel) are
    distinguishable while the same protein stays recognisable."""

    # Build all color commands in one pass
    cf_cmds, ref_cmds, ref_pastel_cmds = [], [], []
    for chain_id, uniprot in cf_chain_to_uniprot.items():
        color = uniprot_colors.get(uniprot, "gray")
        cf_cmds.append(f"color {color}, combfold_sup and chain {chain_id}")
    for chain_id, uniprot in ref_chain_to_uniprot.items():
        color = uniprot_colors.get(uniprot, "gray")
        ref_cmds.append(f"color {color}, reference and chain {chain_id}")
        pastel = PASTEL_RGB.get(color, [0.80, 0.80, 0.80])
        pname = f"pastel_{color}"
        ref_pastel_cmds.append(f"set_color {pname}, [{pastel[0]:.2f}, {pastel[1]:.2f}, {pastel[2]:.2f}]")
        ref_pastel_cmds.append(f"color {pname}, reference and chain {chain_id}")

    pml = f"""
load {ref_path}, reference
load {cf_transformed_path}, combfold_sup
set all_states, on
hide everything
show cartoon
set ray_shadows, 0
bg_color white
set antialias, 0
hide everything, not polymer

{chr(10).join(ref_cmds)}
{chr(10).join(cf_cmds)}

orient reference
get_view tmp_view

disable reference
enable combfold_sup
set_view tmp_view
ray {IMG_W}, {IMG_H}
png {out_dir}/{cpx_id}_combfold.png

disable combfold_sup
enable reference
set_view tmp_view
ray {IMG_W}, {IMG_H}
png {out_dir}/{cpx_id}_reference.png

{chr(10).join(ref_pastel_cmds)}
enable combfold_sup
enable reference
set_view tmp_view
ray {IMG_W}, {IMG_H}
png {out_dir}/{cpx_id}_superposed.png
quit
"""
    pml_path = Path(TMP_DIR) / f"{cpx_id}.pml"
    pml_path.write_text(pml)
    try:
        result = subprocess.run(["pymol", "-c", "-q", str(pml_path)],
                                capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            print(f"  [PyMOL err] {result.stderr[:200]}"); return None
        paths = [out_dir / f"{cpx_id}_{v}.png" for v in ["combfold", "reference", "superposed"]]
        if all(p.exists() for p in paths): return paths
        print(f"  [Missing PNGs] expected 3, got {sum(p.exists() for p in paths)}"); return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] PyMOL hung (>240s)"); return None

# --- PDF composition ---

def compose_pdf(pages, pdf_path):
    """pages: list of dicts with keys: cpx, tm, rmsd, ref, asm_id, asm_fmt,
    spec_str, img_*, uniprot_colors, spec_counts, ref_complete, stoichiometry."""
    import matplotlib
    matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        for p in pages:
            fig, axes = plt.subplots(1, 4, figsize=(15, 5.5),
                                     gridspec_kw={"width_ratios": [1, 1, 1, 0.45]})
            for ax in axes[:3]: ax.axis("off")
            for ax, img_key, label in [(axes[0], "img_combfold", "CombFold"),
                                       (axes[1], "img_reference", "Reference"),
                                       (axes[2], "img_superposed", "Superposed")]:
                try: ax.imshow(imread(str(p[img_key])))
                except Exception: ax.text(0.5, 0.5, f"{label}\n(render failed)", ha="center", va="center")
                ax.set_title(label, fontsize=11, fontweight="bold")

            # Text panel
            axes[3].axis("off")
            tm, rmsd = p["tm"], p["rmsd"]
            tm_color = "#75A025" if tm > 0.9 else ("#FF9400" if tm > 0.5 else "#FD9BED")
            complete = p.get("ref_complete", True)
            cf_stoic = p.get("cf_stoic_str", "") or "?"
            ref_stoic = p.get("ref_stoic_str", "") or "?"
            missing = p.get("missing_str", "") or ""
            if complete:
                stoic_line = f"\nStoichiometry:\n  matched\n  CF:  {cf_stoic}\n  ref: {ref_stoic}"
            else:
                stoic_line = (f"\nStoichiometry:\n  MISMATCH\n  CF:  {cf_stoic}\n  "
                              f"ref: {ref_stoic}\n  missing: {missing or 'none'}")

            text = (f"{p['cpx']}\n\nTM-score:\n  {tm:.3f}\n\nRMSD:\n  "
                    f"{rmsd:.2f} A\n\nRef PDB:\n  {p['ref']}\n\nAssembly:\n  "
                    f"asm {p.get('asm_id', '?')} ({p.get('asm_fmt', '?')})"
                    f"{stoic_line}\n\nProteins:")
            axes[3].text(0.0, 0.95, text, fontsize=10, fontweight="bold",
                         va="top", ha="left", family="monospace",
                         color=tm_color if tm > 0.5 else "black")

            # MISMATCH badge
            if not complete:
                axes[3].text(0.90, 1.02, "MISMATCH", fontsize=10, fontweight="bold",
                             va="bottom", ha="right", family="monospace", color="#D62728",
                             bbox=dict(facecolor="#FFF0F0", edgecolor="#D62728",
                                       boxstyle="round,pad=0.3", linewidth=1.2),
                             transform=axes[3].transAxes, clip_on=False)

            # Protein list with color squares
            n_caption_lines = text.count("\n") + 1
            y_start = max(0.06, 0.95 - n_caption_lines * 0.045)
            for i, (u, color_name) in enumerate(p["uniprot_colors"].items()):
                count = p["spec_counts"].get(u, 1)
                color_hex = COLOR_HEX.get(color_name, "#808080")
                axes[3].text(0.05, y_start - i * 0.05, f"  {u}({count})", fontsize=8,
                             va="top", ha="left", family="monospace", color=color_hex)

            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)
    return pdf_path

# === Main ===

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

print("Loading assembly download log...")
asm_log = load_assembly_log(ASSEMBLY_LOG)
print(f"  {len(asm_log)} entries in log.")

df = pd.read_csv(CSV_PATH)
scored = df[df["comparison_status"] == "scored"].sort_values("best_tm_score", ascending=False).reset_index(drop=True)
print(f"Found {len(scored)} scored complexes.")

# Filter out nucleic-acid-containing PDBs
clean_rows, skipped_na = [], []
for _, row in scored.iterrows():
    if has_nucleic_acid_chains(row["best_tm_pdb"]):
        skipped_na.append((row["#Complex ac"], row["best_tm_pdb"]))
        print(f"  SKIP {row['#Complex ac']} (reference {row['best_tm_pdb']} contains DNA/RNA): TM={row['best_tm_score']:.3f}")
        continue
    clean_rows.append(row)

# === TEST MODE: uncomment to render only the first N complexes ===
# TEST_N = 3
# clean_rows = clean_rows[:TEST_N]
# print(f"[TEST MODE] Rendering only first {TEST_N} complex(es)")
# =================================================================

print(f"\n{len(clean_rows)} protein-only complexes to render.")
if skipped_na:
    print(f"Excluded {len(skipped_na)} with nucleic acid chains: " +
          ", ".join(f"{c}({p})" for c, p in skipped_na))
print()

pages, success, fail, t0 = [], 0, 0, time.time()

for i, row in enumerate(clean_rows, 1):
    cpx = row["#Complex ac"]; tm = row["best_tm_score"]; rmsd = row["best_rmsd"]; ref_pdb = row["best_tm_pdb"]
    spec = row["comb_fold_submission"]
    ref_complete_raw = row.get("ref_complete", "")
    ref_complete = ref_complete_raw.lower() in ("true", "1", "yes") if isinstance(ref_complete_raw, str) else bool(ref_complete_raw)
    cf_stoic_str = row.get("cf_stoichiometry", "") or ""
    ref_stoic_str = row.get("ref_stoichiometry", "") or ""
    missing_str = row.get("missing_uniprots", "") or ""
    complete_tag = "" if ref_complete else " [MISMATCH]"
    print(f"[{i}/{len(clean_rows)}] {cpx} (TM={tm:.3f}, RMSD={rmsd:.2f}, ref={ref_pdb}{complete_tag})")

    spec_counts = parse_spec(spec)
    spec_uniprots = list(dict.fromkeys(spec_counts.keys()))
    spec_folder = "_".join(f"{u}x{spec_counts[u]}" for u in sorted(spec_counts))
    cf_path = Path(COMBFOLD_DIR) / f"{spec_folder}_output" / "assembled_results" / "output_clustered_0.pdb"

    ref_path, asm_id, asm_fmt = find_assembly_file(ref_pdb, asm_log, ASSEMBLY_DIR)
    if ref_path is None:
        print(f"  SKIP: assembly file not found for {ref_pdb}"); fail += 1; continue
    if not cf_path.exists():
        print(f"  SKIP: CombFold file missing"); fail += 1; continue

    # Superpose + get coloring in one call (single load, single SIFTS fetch)
    cf_transformed = Path(TMP_DIR) / f"{cpx}_combfold_transformed.pdb"
    try:
        rms, n_atoms, cf_chain_to_uniprot, ref_chain_to_uniprot = \
            superpose_and_color(cf_path, ref_path, spec_uniprots, cf_transformed)
    except Exception as e:
        print(f"  SUPERPOSE FAIL: {e}"); fail += 1; continue

    # Assign colors per UniProt
    uniprot_colors = {u: PROTEIN_COLORS[idx % len(PROTEIN_COLORS)]
                      for idx, u in enumerate(sorted(spec_uniprots))}

    # Render
    paths = render_three_views_pymol(ref_path, cf_transformed, Path(OUT_DIR), cpx,
                                      cf_chain_to_uniprot, ref_chain_to_uniprot, uniprot_colors)
    if paths:
        pages.append({"cpx": cpx, "tm": tm, "rmsd": rmsd, "ref": ref_pdb,
                      "asm_id": asm_id, "asm_fmt": asm_fmt,
                      "spec_str": spec, "spec_counts": spec_counts,
                      "img_combfold": paths[0], "img_reference": paths[1], "img_superposed": paths[2],
                      "uniprot_colors": uniprot_colors,
                      "ref_complete": ref_complete,
                      "cf_stoic_str": cf_stoic_str, "ref_stoic_str": ref_stoic_str,
                      "missing_str": missing_str})
        success += 1
        print(f"  OK ({time.time()-t0:.0f}s total, {success} done)")
    else:
        fail += 1

print(f"\n=== Rendering done: {success} ok, {fail} failed in {time.time()-t0:.0f}s ===")
print(f"Composing PDF with {len(pages)} pages...")
pdf_path = compose_pdf(pages, PDF_PATH)

print(f"\nPDF saved: {PDF_PATH}")
print(f"  {len(pages)} pages, one protein-only complex per page")
print(f"  Layout: [CombFold] [Reference] [Superposed] [CPX + TM + RMSD + Ref + Assembly + Proteins]")
if skipped_na:
    print(f"\nExcluded (nucleic acid in reference): " + ", ".join(f"{c}({p})" for c, p in skipped_na))
