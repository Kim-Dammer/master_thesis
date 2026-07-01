"""
render_pdf_summary_v2.py — Case-A-only PDF with per-protein coloring and matched orientation.

Changes from v1:
  - Filters to Case A (stoichiometry matches between CombFold spec and reference PDB)
  - Each protein (UniProt) gets a distinct color, same in CombFold and reference
  - All 3 panels share the same orientation (uses superposed CombFold for all panels)
  - Right panel shows protein composition (e.g. P02557(1), P09733(1))
  - Non-protein chains (DNA/RNA/ligands) hidden

PDF layout per page:
  [CombFold] [Reference] [Superposed]   CPX-XXXX
                                         TM-score: 0.98
                                         RMSD: 1.06 Å
                                         Ref: 5W3H
                                         Proteins:
                                         P02557(1) - gray
                                         P09733(1) - blue

Usage:
    uv run render_pdf_summary_v2.py
"""
import os, re, sys, json, time, urllib.request, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Superimposer
import warnings
warnings.filterwarnings("ignore")
from procompa import get_project_root, get_data_dir

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"

# === CONFIG ===
CSV_PATH    = data_dir/"Pipeline/third_setup/third_setup_pipeline_complexes_combfold_results_with_exact_match_comparison.csv"
COMBFOLD_DIR = data_dir/"Pipeline/third_setup/CombFold"
RAW_DIR = data_dir/"all_Complex_pdb_files/_raw"
OUT_DIR     = data_dir/"Pipeline/third_setup/superposition_figures"
TMP_DIR     = "/tmp/superpose_pdf_v2"
PDF_PATH    = data_dir/"Pipeline/third_setup/combfold_benchmark_caseA.pdf"
IMG_W, IMG_H = 800, 600
# Distinct PyMOL colors for up to 14 proteins (used for CombFold and reference-alone)
PROTEIN_COLORS = ["red", "blue", "green", "yellow", "violet", "cyan", "magenta",
                  "orange", "salmon", "lime", "pink", "wheat", "teal", "purple"]
# Pastel equivalents for the reference in the SUPERPOSED view only.
# PyMOL accepts RGB triplets [r, g, b] in 0-1 range. These are ~60% lightness pastels
# of the vivid colors above, so the same protein is recognisable but the reference
# reads as the lighter/paler partner in the overlay.
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
# =============

_pdb_parser = PDBParser(QUIET=True)
_cif_parser = MMCIFParser(QUIET=True)

def load_structure(path):
    p = Path(path)
    if p.suffix.lower() == ".cif":
        return _cif_parser.get_structure(p.stem, str(p))
    return _pdb_parser.get_structure(p.stem, str(p))

def chain_ca_atoms(chain):
    return [res["CA"] for res in chain if res.id[0] == " " and "CA" in res]

def chain_length(chain):
    return len(chain_ca_atoms(chain))

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
def sifts_chain_mapping(pdb_id):
    pdb_id = pdb_id.lower()
    if pdb_id in _sifts_cache: return _sifts_cache[pdb_id]
    chain_map = {}
    try:
        with urllib.request.urlopen(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}", timeout=20) as r:
            data = json.loads(r.read())
        for uniprot, info in data.get(pdb_id, {}).get("UniProt", {}).items():
            for m in info.get("mappings", []):
                ch = m.get("chain_id")
                if not ch: continue
                us, ue = m.get("unp_start"), m.get("unp_end")
                ps = m.get("start", {}).get("residue_number")
                pe = m.get("end", {}).get("residue_number")
                if None in (us, ue, ps, pe): continue
                p2u = {ps + i: us + i for i in range(min(ue - us + 1, pe - ps + 1))}
                if ch in chain_map:
                    chain_map[ch]["pdb_to_uniprot"].update(p2u)
                else:
                    chain_map[ch] = {"uniprot": uniprot, "pdb_to_uniprot": p2u}
    except Exception: pass
    _sifts_cache[pdb_id] = chain_map
    time.sleep(0.1)
    return chain_map

def sifts_chain_counts(pdb_id):
    """Returns {uniprot: n_chains} for the asymmetric unit."""
    cm = sifts_chain_mapping(pdb_id)
    counts = {}
    for ch, info in cm.items():
        u = info["uniprot"]
        counts[u] = counts.get(u, 0) + 1
    return counts

def has_nucleic_acid_chains(pdb_id):
    """Check if a PDB entry contains DNA/RNA polymer chains (not just ligands/water).
    Uses PDBe molecules API. Returns True if any polyribonucleotide or
    polydeoxyribonucleotide molecule is present."""
    pdb_id = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        for mol in data.get(pdb_id, []):
            mt = mol.get("molecule_type", "").lower()
            if "nucleotide" in mt:  # polyribonucleotide or polydeoxyribonucleotide
                return True
        return False
    except Exception:
        return False  # if API fails, don't exclude (let it through)

def parse_spec(spec):
    counts = {}
    for m in re.finditer(r"([A-Za-z0-9_]+)\((\d+)\)", spec):
        counts[m.group(1)] = int(m.group(2))
    return counts

def classify_case(spec_counts, ref_counts):
    """A = stoichiometry matches, B = ref has more, C = ref has fewer, X = UniProt missing."""
    case = "A"
    for u, spec_n in spec_counts.items():
        ref_n = ref_counts.get(u, 0)
        if ref_n == 0: return "X"
        if spec_n < ref_n: case = "B" if case == "A" else case
        elif spec_n > ref_n: case = "C" if case == "A" else case
    return case

def match_combfold_chains_to_uniprots(cf_path, spec_uniprots):
    """Returns {chain_id: uniprot} by matching chain length to UniProt canonical length."""
    cf = load_structure(cf_path)
    uniprot_lengths = {u: get_uniprot_length(u) for u in spec_uniprots}
    chain_to_uniprot = {}
    used_uniprots = set()
    # Sort chains by length descending to match longest first (more unique)
    chains_by_len = sorted(cf.get_chains(), key=lambda c: chain_length(c), reverse=True)
    for chain in chains_by_len:
        cl = chain_length(chain)
        if cl == 0: continue
        best_u = None
        for u, ulen in uniprot_lengths.items():
            if u in used_uniprots: continue
            if ulen == cl:
                best_u = u; break
        if best_u is None:
            # Fallback: closest length
            avail = {u: ul for u, ul in uniprot_lengths.items() if u not in used_uniprots}
            if avail:
                best_u = min(avail, key=lambda u: abs(avail[u] - cl))
        if best_u:
            chain_to_uniprot[chain.id] = best_u
            used_uniprots.add(best_u)
    return chain_to_uniprot

def superpose_combfold_on_reference(cf_path, ref_path, spec_uniprots, out_path):
    cf = load_structure(cf_path)
    ref = load_structure(ref_path)
    uniprot_lengths = {u: get_uniprot_length(u) for u in spec_uniprots}
    ref_chains = sifts_chain_mapping(Path(ref_path).stem)
    uniprot_to_ref_chain = {}
    for rc, info in ref_chains.items():
        u = info["uniprot"]
        if u not in uniprot_to_ref_chain: uniprot_to_ref_chain[u] = rc
    cf_atoms, ref_atoms = [], []
    for cf_chain in cf.get_chains():
        cf_len = chain_length(cf_chain)
        if cf_len == 0: continue
        matched_u = None
        for u, ulen in uniprot_lengths.items():
            if ulen == cf_len and u in uniprot_to_ref_chain:
                matched_u = u; break
        if matched_u is None: continue
        ref_chain = ref[0][uniprot_to_ref_chain[matched_u]]
        cf_ca = chain_ca_atoms(cf_chain)
        ref_ca = chain_ca_atoms(ref_chain)
        p2u = ref_chains[uniprot_to_ref_chain[matched_u]]["pdb_to_uniprot"]
        cf_by_pos = {a.get_parent().get_id()[1]: a for a in cf_ca}
        ref_by_pos = {p2u[r]: a for a in ref_ca for r in [a.get_parent().get_id()[1]] if r in p2u}
        common = sorted(set(cf_by_pos.keys()) & set(ref_by_pos.keys()))
        for pos in common:
            cf_atoms.append(cf_by_pos[pos]); ref_atoms.append(ref_by_pos[pos])
    if not cf_atoms: raise RuntimeError("No chains matched")
    sup = Superimposer()
    sup.set_atoms(ref_atoms, cf_atoms)
    sup.apply(list(cf.get_atoms()))
    io = PDBIO(); io.set_structure(cf); io.save(str(out_path))
    return sup.rms, len(cf_atoms)

def render_three_views_pymol(ref_path, cf_transformed_path, out_dir, cpx_id,
                              cf_chain_to_uniprot, ref_chain_to_uniprot, uniprot_colors):
    """ONE PyMOL session -> 3 PNGs at 800x600, same orientation, per-protein coloring.
    In the superposed view, the reference is rendered in pastel versions of the
    same per-protein colors so CombFold (vivid) and reference (pastel) are
    distinguishable while the same protein stays recognisable."""
    
    # Build PyMOL color commands for CombFold (uses transformed PDB, same chain IDs)
    cf_color_cmds = []
    for chain_id, uniprot in cf_chain_to_uniprot.items():
        color = uniprot_colors.get(uniprot, "gray")
        cf_color_cmds.append(f"color {color}, combfold_sup and chain {chain_id}")
    
    # Build color commands for reference
    ref_color_cmds = []
    for chain_id, uniprot in ref_chain_to_uniprot.items():
        color = uniprot_colors.get(uniprot, "gray")
        ref_color_cmds.append(f"color {color}, reference and chain {chain_id}")

    # Build color commands for reference in SUPERPOSED view (pastel versions)
    # Set custom RGB colors via set_color, then apply
    ref_pastel_cmds = []
    for chain_id, uniprot in ref_chain_to_uniprot.items():
        color = uniprot_colors.get(uniprot, "gray")
        pastel = PASTEL_RGB.get(color, [0.80, 0.80, 0.80])
        pastel_name = f"pastel_{color}"
        ref_pastel_cmds.append(f"set_color {pastel_name}, [{pastel[0]:.2f}, {pastel[1]:.2f}, {pastel[2]:.2f}]")
        ref_pastel_cmds.append(f"color {pastel_name}, reference and chain {chain_id}")
    
    pml = f"""
load {ref_path}, reference
load {cf_transformed_path}, combfold_sup

hide everything
show cartoon
set ray_shadows, 0
bg_color white
set antialias, 0
# Hide non-protein (DNA, RNA, ligands)
hide everything, not polymer

# Color reference chains by UniProt
{chr(10).join(ref_color_cmds)}

# Color CombFold chains by UniProt
{chr(10).join(cf_color_cmds)}

# Orient on reference and save view
orient reference
get_view tmp_view

# 1. CombFold alone (same view as reference)
disable reference
enable combfold_sup
set_view tmp_view
ray {IMG_W}, {IMG_H}
png {out_dir}/{cpx_id}_combfold.png

# 2. Reference alone (vivid colors)
disable combfold_sup
enable reference
set_view tmp_view
ray {IMG_W}, {IMG_H}
png {out_dir}/{cpx_id}_reference.png

# 3. Superposed: recolor reference to pastel so CombFold (vivid) stands out
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
            print(f"  [PyMOL err] {result.stderr[:200]}")
            return None
        paths = [out_dir / f"{cpx_id}_{v}.png" for v in ["combfold", "reference", "superposed"]]
        if all(p.exists() for p in paths):
            return paths
        print(f"  [Missing PNGs] expected 3, got {sum(p.exists() for p in paths)}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] PyMOL hung (>240s)")
        return None

def compose_pdf(pages, pdf_path):
    """pages: list of dicts with keys: cpx, tm, rmsd, ref, spec_str, img_*, uniprot_colors."""
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.backends.backend_pdf import PdfPages

    fig_size = (15, 5.5)
    with PdfPages(pdf_path) as pdf:
        for p in pages:
            fig, axes = plt.subplots(1, 4, figsize=fig_size,
                                     gridspec_kw={"width_ratios": [1, 1, 1, 0.45]})
            for ax in axes[:3]:
                ax.axis("off")
            try: axes[0].imshow(imread(str(p["img_combfold"])))
            except Exception: axes[0].text(0.5, 0.5, "CombFold\n(render failed)", ha="center", va="center")
            try: axes[1].imshow(imread(str(p["img_reference"])))
            except Exception: axes[1].text(0.5, 0.5, "Reference\n(render failed)", ha="center", va="center")
            try: axes[2].imshow(imread(str(p["img_superposed"])))
            except Exception: axes[2].text(0.5, 0.5, "Superposed\n(render failed)", ha="center", va="center")
            axes[0].set_title("CombFold", fontsize=11, fontweight="bold")
            axes[1].set_title("Reference", fontsize=11, fontweight="bold")
            axes[2].set_title("Superposed", fontsize=11, fontweight="bold")
            
            # Text column
            axes[3].axis("off")
            tm = p["tm"]; rmsd = p["rmsd"]
            tm_color = "#75A025" if tm > 0.9 else ("#0279EE" if tm > 0.5 else "#FD9BED")
            
            # Build protein list with colors
            protein_lines = []
            for u, color_name in p["uniprot_colors"].items():
                count = p["spec_counts"].get(u, 1)
                # Map PyMOL color names to hex for matplotlib
                color_hex = {
                    "red": "#FF0000", "blue": "#0000FF", "green": "#00FF00",
                    "yellow": "#FFFF00", "violet": "#8B00FF", "cyan": "#00FFFF",
                    "magenta": "#FF00FF", "orange": "#FFA500", "salmon": "#FA8072",
                    "lime": "#00FF00", "pink": "#FFC0CB", "wheat": "#F5DEB3",
                    "teal": "#008080", "purple": "#800080"
                }.get(color_name, "#808080")
                protein_lines.append(f"  {u}({count})")
            
            text = f"{p['cpx']}\n\nTM-score:\n  {tm:.3f}\n\nRMSD:\n  {rmsd:.2f} Å\n\nRef PDB:\n  {p['ref']}\n\nProteins:"
            axes[3].text(0.0, 0.95, text, fontsize=10, fontweight="bold",
                         va="top", ha="left", family="monospace",
                         color=tm_color if tm > 0.5 else "black")
            
            # Add protein list with color squares
            y_start = 0.30
            for i, (u, color_name) in enumerate(p["uniprot_colors"].items()):
                count = p["spec_counts"].get(u, 1)
                color_hex = {
                    "red": "#FF0000", "blue": "#0000FF", "green": "#00AA00",
                    "yellow": "#D4AA00", "violet": "#8B00FF", "cyan": "#00AAAA",
                    "magenta": "#FF00FF", "orange": "#FFA500", "salmon": "#FA8072",
                    "lime": "#7CFC00", "pink": "#FFC0CB", "wheat": "#D2B48C",
                    "teal": "#008080", "purple": "#800080"
                }.get(color_name, "#808080")
                axes[3].text(0.05, y_start - i*0.05, f"■ {u}({count})", fontsize=8,
                            va="top", ha="left", family="monospace", color=color_hex)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)
    return pdf_path

# === Main ===
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
scored = df[df["comparison_status"] == "scored"].sort_values("best_tm_score", ascending=False).reset_index(drop=True)
print(f"Found {len(scored)} scored complexes. Classifying stoichiometry match...")

# Classify and filter to Case A, then exclude nucleic-acid-containing PDBs
case_a_rows = []
skipped_na = []
for _, row in scored.iterrows():
    spec_counts = parse_spec(row["comb_fold_submission"])
    ref_counts = sifts_chain_counts(row["best_tm_pdb"])
    case = classify_case(spec_counts, ref_counts)
    if case != "A":
        print(f"  SKIP {row['#Complex ac']} (case {case}, stoichiometry mismatch): TM={row['best_tm_score']:.3f}")
        continue
    # Check for DNA/RNA chains in the reference
    if has_nucleic_acid_chains(row["best_tm_pdb"]):
        skipped_na.append((row["#Complex ac"], row["best_tm_pdb"]))
        print(f"  SKIP {row['#Complex ac']} (reference {row['best_tm_pdb']} contains DNA/RNA): TM={row['best_tm_score']:.3f}")
        continue
    case_a_rows.append(row)

# === TEST MODE: uncomment to render only the first N complexes ===
# TEST_N = 1
# case_a_rows = case_a_rows[:TEST_N]
# print(f"[TEST MODE] Rendering only first {TEST_N} complex(es)")
# =================================================================

print(f"\n{len(case_a_rows)} Case A, protein-only complexes (fair comparisons) to render.")
if skipped_na:
    print(f"Excluded {len(skipped_na)} with nucleic acid chains: " +
          ", ".join(f"{c}({p})" for c, p in skipped_na))
print()

pages = []
success = fail = 0
t0 = time.time()

for i, row in enumerate(case_a_rows, 1):
    cpx = row["#Complex ac"]; tm = row["best_tm_score"]; rmsd = row["best_rmsd"]; ref_pdb = row["best_tm_pdb"]
    spec = row["comb_fold_submission"]
    print(f"[{i}/{len(case_a_rows)}] {cpx} (TM={tm:.3f}, RMSD={rmsd:.2f}, ref={ref_pdb})")

    spec_counts = parse_spec(spec)
    spec_uniprots = list(dict.fromkeys(spec_counts.keys()))
    spec_folder = "_".join(f"{u}x{spec_counts[u]}" for u in sorted(spec_counts))
    cf_path = Path(COMBFOLD_DIR) / f"{spec_folder}_output" / "assembled_results" / "output_clustered_0.pdb"
    ref_path = Path(RAW_DIR) / f"{ref_pdb}.pdb"
    if not ref_path.exists(): ref_path = Path(RAW_DIR) / f"{ref_pdb}.cif"

    if not cf_path.exists() or not ref_path.exists():
        print(f"  SKIP: missing files"); fail += 1; continue

    # Superpose
    cf_transformed = Path(TMP_DIR) / f"{cpx}_combfold_transformed.pdb"
    try:
        rms, n_atoms = superpose_combfold_on_reference(cf_path, ref_path, spec_uniprots, cf_transformed)
    except Exception as e:
        print(f"  SUPERPOSE FAIL: {e}"); fail += 1; continue

    # Get chain-to-UniProt mappings for coloring
    cf_chain_to_uniprot = match_combfold_chains_to_uniprots(cf_path, spec_uniprots)
    ref_chains = sifts_chain_mapping(ref_pdb)
    ref_chain_to_uniprot = {ch: info["uniprot"] for ch, info in ref_chains.items()}

    # Assign colors per UniProt
    uniprot_colors = {}
    for idx, u in enumerate(sorted(spec_uniprots)):
        uniprot_colors[u] = PROTEIN_COLORS[idx % len(PROTEIN_COLORS)]

    # Render
    paths = render_three_views_pymol(ref_path, cf_transformed, Path(OUT_DIR), cpx,
                                      cf_chain_to_uniprot, ref_chain_to_uniprot, uniprot_colors)
    if paths:
        pages.append({"cpx": cpx, "tm": tm, "rmsd": rmsd, "ref": ref_pdb,
                      "spec_str": spec, "spec_counts": spec_counts,
                      "img_combfold": paths[0], "img_reference": paths[1], "img_superposed": paths[2],
                      "uniprot_colors": uniprot_colors})
        success += 1
        elapsed = time.time() - t0
        print(f"  OK ({elapsed:.0f}s total, {success} done)")
    else:
        fail += 1

print(f"\n=== Rendering done: {success} ok, {fail} failed in {time.time()-t0:.0f}s ===")
print(f"Composing PDF with {len(pages)} pages...")
pdf_path = compose_pdf(pages, PDF_PATH)
print(f"\nPDF saved: {PDF_PATH}")
print(f"  {len(pages)} pages, one Case-A protein-only complex per page")
print(f"  Layout: [CombFold] [Reference] [Superposed] [CPX + TM + RMSD + Ref + Proteins]")
print(f"\nExcluded complexes:")
print(f"  Stoichiometry mismatch (Case B - ref has more copies): CPX-1688, CPX-1632, CPX-1742, CPX-1630")
print(f"  Stoichiometry mismatch (Case C - ref has fewer copies): CPX-2896, CPX-544")
print(f"  Nucleic acid chains in reference: " + ", ".join(f"{c}({p})" for c, p in skipped_na))
