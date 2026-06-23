# ============================================================
# RMSD Analysis: CombFold assemblies vs identity reference PDBs
# ============================================================
# Computes C-alpha RMSD after Kabsch superposition, matching
# chains by UniProt ID. Reports best RMSD per complex when
# multiple identity reference PDBs exist.
#
# Designed for the third-setup pipeline where each complex has
# exactly one row (true stoichiometry only, no Stoic predictions).

import json, re, os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from collections import defaultdict

# ---- Paths (adjust to your cluster setup) ----
OUTPUT_BASE = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/third_setup/CombFold"
COMPLEX_PDB_DIR = "/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files/identity"
RESULTS_CSV = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/third_setup/pdb_present_for_stoi_gr_two_third_setup_pipeline_complexes_combfold_results.csv"
PDB_MAPPING_CSV = "/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files/complex_pdb_mapping.csv"

# ---- Helper: parse stoichiometry spec to complex name ----
entry_re = re.compile(r"^([A-Za-z0-9_]+)\((\d+)\)$")

def spec_to_complex_name(spec):
    counts = {}
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        m = entry_re.match(tok)
        if m:
            counts[m.group(1)] = int(m.group(2))
    return "_".join(f"{p}x{counts[p]}" for p in sorted(counts))

# ---- Helper: parse spec into {UniProt: count} ----
def parse_spec(spec):
    counts = {}
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        m = entry_re.match(tok)
        if m:
            counts[m.group(1)] = int(m.group(2))
    return counts

# ---- Helper: parse max confidence from CombFold string ----
def parse_max_conf(conf_str):
    if not isinstance(conf_str, str) or not conf_str:
        return None
    scores = []
    for part in conf_str.split(";"):
        try:
            scores.append(float(part.split(":")[1]))
        except:
            pass
    return max(scores) if scores else None

# ---- Helper: get C-alpha coords for a chain ----
def get_ca_coords(chain):
    coords = []
    for residue in chain:
        if residue.has_id("CA"):
            coords.append(residue["CA"].get_coord())
    return np.array(coords)

# ---- Helper: extract UniProt -> chain mapping from CombFold subunits.json ----
def get_combfold_uniprot_map(complex_name):
    """Parse subunits.json written by the CombFold sbatch.

    The sbatch writes subunits.json as a dict keyed by UniProt ID:
      {
        "P00899": {
          "name": "P00899",
          "chain_names": ["A"],
          "start_res": 1,
          "sequence": "MST..."
        },
        ...
      }

    Returns: {UniProt -> [chain_id, ...]}
    """
    subunits_path = os.path.join(OUTPUT_BASE, f"{complex_name}_input", "subunits.json")
    if not os.path.exists(subunits_path):
        return {}
    with open(subunits_path) as f:
        subunits = json.load(f)

    mapping = {}  # UniProt -> list of chain IDs

    # Handle dict format (keyed by UniProt ID)
    if isinstance(subunits, dict):
        for key, sub in subunits.items():
            uniprot = sub.get("name", key)
            chain_names = sub.get("chain_names", [])
            if uniprot and chain_names:
                mapping[uniprot] = chain_names
    # Handle list format (fallback)
    elif isinstance(subunits, list):
        for sub in subunits:
            uniprot = sub.get("name", "")
            chain_id = sub.get("chain", "")
            if uniprot and chain_id:
                mapping.setdefault(uniprot, []).append(chain_id)

    return mapping

# ---- Helper: extract UniProt -> chain mapping from reference PDB DBREF lines ----
def get_ref_uniprot_map(pdb_path):
    """Parse DBREF/DBREF1 lines from a PDB file using fixed-column format.

    PDB DBREF format (columns are 1-indexed):
      1-6   DBREF or DBREF1
      8-11  PDB ID code (e.g. 4WWU)
      12    Chain ID (single character)
      14-18 SeqRes start (integer)
      20-24 SeqRes end (integer)
      26    Database name start
      27-30 Database name (e.g. UNP)
      32-41 Database accession code (e.g. Q99257)

    DBREF1/DBREF2 format (for longer sequences):
      DBREF1: same positions for chain, but database accession in 33-41
      DBREF2: chain in col 12, accession in 33-41

    Returns: {UniProt -> [chain_id, ...]}
    """
    mapping = {}  # UniProt -> list of chain IDs
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("DBREF "):
                # Fixed-column parsing
                # Chain ID is at column 12 (0-indexed: position 11)
                chain_id = line[11] if len(line) > 11 else ""
                # Database name is at columns 27-30 (0-indexed: 26-29)
                db_name = line[26:30].strip() if len(line) > 29 else ""
                # Database accession is at columns 33-41 (0-indexed: 32-40)
                db_acc = line[32:41].strip() if len(line) > 40 else ""

                if db_name == "UNP" and db_acc and chain_id.strip():
                    uniprot = db_acc
                    if not uniprot.startswith("PDB"):
                        mapping.setdefault(uniprot, []).append(chain_id)

            elif line.startswith("DBREF1"):
                # DBREF1: chain at col 12, accession at col 33-41
                chain_id = line[11] if len(line) > 11 else ""
                db_acc = line[32:41].strip() if len(line) > 40 else ""
                if db_acc and chain_id.strip() and not db_acc.startswith("PDB"):
                    mapping.setdefault(db_acc, []).append(chain_id)

    return mapping

# ---- Helper: fallback chain matching by sequence length ----
def get_ref_chain_by_length(ref_struct, target_lengths):
    """Fallback: match reference chains to UniProt IDs by sequence length.

    target_lengths: {UniProt -> residue_count}
    Returns: {UniProt -> [chain_id, ...]} or {} if ambiguous.
    """
    ref_chains = list(ref_struct[0])
    # Build {chain_id: n_residues}
    chain_lengths = {}
    for c in ref_chains:
        n_res = sum(1 for r in c if r.has_id("CA"))
        chain_lengths[c.id] = n_res

    # Invert: {n_residues -> [chain_ids]}
    len_to_chains = defaultdict(list)
    for cid, n in chain_lengths.items():
        len_to_chains[n].append(cid)

    # Match each UniProt by its expected length
    mapping = {}
    for uniprot, expected_len in target_lengths.items():
        candidates = len_to_chains.get(expected_len, [])
        if len(candidates) >= 1:
            mapping[uniprot] = candidates
        # else: no chain with that length

    return mapping

# ---- Helper: Kabsch RMSD between two coordinate arrays ----
def compute_rmsd(coords1, coords2):
    """Compute RMSD after optimal superposition (Kabsch algorithm)."""
    if len(coords1) != len(coords2) or len(coords1) < 3:
        return None
    # Center both coordinate sets
    centroid1 = coords1.mean(axis=0)
    centroid2 = coords2.mean(axis=0)
    c1 = coords1 - centroid1
    c2 = coords2 - centroid2
    # Cross-covariance matrix
    H = c1.T @ c2
    # SVD
    U, S, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    sign = np.array([1, 1, np.sign(d)])
    R = Vt.T @ np.diag(sign) @ U.T
    # Apply rotation and compute RMSD
    c1_rot = c1 @ R
    diff = c1_rot - c2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

# ---- Helper: compute per-chain RMSD for a complex ----
def compute_complex_rmsd(complex_name, ref_pdb_path, combfold_pdb_path,
                         spec_counts=None):
    """
    Match chains by UniProt ID, compute per-chain C-alpha RMSD,
    return average RMSD across matched chains and number of chains matched.

    Chain matching strategy:
    1. Try DBREF lines from reference PDB (UniProt -> chain)
    2. If no DBREF mapping found, fall back to sequence-length matching
       using the known stoichiometry from spec_counts.
    """
    parser = PDBParser(QUIET=True)

    # Load structures
    try:
        comb_struct = parser.get_structure("comb", combfold_pdb_path)
    except Exception as e:
        return None, 0, f"CombFold PDB load error: {e}"

    try:
        ref_struct = parser.get_structure("ref", ref_pdb_path)
    except Exception as e:
        return None, 0, f"Reference PDB load error: {e}"

    # Get UniProt -> chain mappings
    comb_map = get_combfold_uniprot_map(complex_name)  # UniProt -> [chain_ids]
    ref_map = get_ref_uniprot_map(ref_pdb_path)         # UniProt -> [chain_ids]

    # Fallback: if no DBREF mapping, try sequence-length matching
    if not ref_map and spec_counts is not None:
        # Build expected residue counts from CombFold chains
        comb_chains_obj = {c.id: c for c in comb_struct[0]}
        target_lengths = {}
        for uniprot, chain_ids in comb_map.items():
            if chain_ids and chain_ids[0] in comb_chains_obj:
                n_ca = len(get_ca_coords(comb_chains_obj[chain_ids[0]]))
                target_lengths[uniprot] = n_ca
        ref_map = get_ref_chain_by_length(ref_struct, target_lengths)
        if ref_map:
            print(f"    [fallback] Matched {len(ref_map)} UniProt(s) by sequence length")

    if not comb_map:
        return None, 0, "Missing CombFold UniProt mapping (subunits.json)"
    if not ref_map:
        return None, 0, "Missing reference UniProt mapping (no DBREF, length fallback failed)"

    # Get chain objects from first model
    comb_chains = {c.id: c for c in comb_struct[0]}
    ref_chains = {c.id: c for c in ref_struct[0]}

    # Match by UniProt and compute per-chain RMSD
    chain_rmsds = []
    matched_uniprots = set(comb_map.keys()) & set(ref_map.keys())

    if not matched_uniprots:
        comb_keys = set(comb_map.keys())
        ref_keys = set(ref_map.keys())
        return None, 0, f"No shared UniProt IDs (comb={sorted(comb_keys)[:5]}, ref={sorted(ref_keys)[:5]})"

    for uniprot in sorted(matched_uniprots):
        comb_chain_ids = comb_map[uniprot]
        ref_chain_ids = ref_map[uniprot]

        # Match up to min(len(comb), len(ref)) chains for this UniProt
        n_match = min(len(comb_chain_ids), len(ref_chain_ids))

        for i in range(n_match):
            cc_id = comb_chain_ids[i]
            rc_id = ref_chain_ids[i]

            if cc_id not in comb_chains or rc_id not in ref_chains:
                continue

            ca_comb = get_ca_coords(comb_chains[cc_id])
            ca_ref = get_ca_coords(ref_chains[rc_id])

            # Align by common residue count (take min length)
            n_res = min(len(ca_comb), len(ca_ref))
            if n_res < 3:
                continue

            rmsd = compute_rmsd(ca_comb[:n_res], ca_ref[:n_res])
            if rmsd is not None:
                chain_rmsds.append(rmsd)

    if not chain_rmsds:
        return None, 0, f"No chain pairs could be aligned (matched {len(matched_uniprots)} UniProt IDs but 0 valid chain pairs)"

    avg_rmsd = np.mean(chain_rmsds)
    return avg_rmsd, len(chain_rmsds), "OK"

# ============================================================
# Main analysis
# ============================================================

# Load data
df = pd.read_csv(RESULTS_CSV)
pdb_map = pd.read_csv(PDB_MAPPING_CSV)

# Filter to identity PDBs only
identity_pdbs = pdb_map[pdb_map["tag"] == "identity"].copy()

# Filter to assembled complexes (column is combfold_successfully, not "assembled")
assembled = df[df["combfold_successfully"] == True].copy()

# Get complexes that are both assembled AND have identity PDBs
assembled_complexes = set(assembled["#Complex ac"].unique())
identity_complexes = set(identity_pdbs["#Complex ac"].unique())
target_complexes = assembled_complexes & identity_complexes

print(f"Assembled complexes: {len(assembled_complexes)}")
print(f"Identity PDB complexes: {len(identity_complexes)}")
print(f"Overlap (target): {len(target_complexes)}")

# Parse confidence for assembled complexes
assembled["max_confidence"] = assembled["confidence_scores"].apply(parse_max_conf)

# For each target complex, get the confidence.
# Third-setup: one row per complex (true stoichiometry only), no Stoic predictions.
# So every assembled row IS the correct stoichiometry.
complex_conf = {}
complex_spec = {}
for cpx in target_complexes:
    cpx_rows = assembled[assembled["#Complex ac"] == cpx]
    best_row = cpx_rows.sort_values("max_confidence", ascending=False).iloc[0]
    complex_conf[cpx] = best_row["max_confidence"]
    complex_spec[cpx] = best_row["comb_fold_submission"]

# ---- Diagnostic: check one subunits.json and one reference PDB ----
sample_cpx = sorted(target_complexes)[0]
sample_spec = complex_spec[sample_cpx]
sample_name = spec_to_complex_name(sample_spec)
sample_subunits = os.path.join(OUTPUT_BASE, f"{sample_name}_input", "subunits.json")
sample_ref_pdbs = identity_pdbs[identity_pdbs["#Complex ac"] == sample_cpx]["pdb_id"].unique()

print(f"\n--- Diagnostic for {sample_cpx} ---")
print(f"  Spec: {sample_spec}")
print(f"  Complex name: {sample_name}")
print(f"  subunits.json exists: {os.path.exists(sample_subunits)}")
if os.path.exists(sample_subunits):
    with open(sample_subunits) as f:
        sub = json.load(f)
    print(f"  subunits.json type: {type(sub).__name__}")
    if isinstance(sub, dict):
        print(f"  subunits.json keys: {list(sub.keys())[:5]}")
        first_key = list(sub.keys())[0]
        print(f"  First entry ({first_key}): {sub[first_key]}")
    comb_map = get_combfold_uniprot_map(sample_name)
    print(f"  CombFold UniProt map: {comb_map}")

if len(sample_ref_pdbs) > 0:
    sample_ref_path = os.path.join(COMPLEX_PDB_DIR, f"{sample_cpx}_{sample_ref_pdbs[0]}.pdb")
    print(f"  Reference PDB exists: {os.path.exists(sample_ref_path)}")
    if os.path.exists(sample_ref_path):
        ref_map = get_ref_uniprot_map(sample_ref_path)
        print(f"  Reference DBREF map: {ref_map}")
        if not ref_map:
            # Show first few lines of the PDB to understand format
            with open(sample_ref_path) as f:
                lines = f.readlines()
            dbref_lines = [l.rstrip() for l in lines if l.startswith("DBREF")]
            print(f"  DBREF lines ({len(dbref_lines)}):")
            for l in dbref_lines[:5]:
                print(f"    {l}")
            if not dbref_lines:
                header_lines = [l.rstrip() for l in lines[:30]]
                print(f"  First 30 lines (no DBREF found):")
                for l in header_lines:
                    print(f"    {l}")
print("--- End diagnostic ---\n")

# Compute RMSD for each target complex
results = []
fail_reasons = defaultdict(int)

for cpx in sorted(target_complexes):
    # Get complex name from spec
    spec = complex_spec[cpx]
    complex_name = spec_to_complex_name(spec)
    spec_counts = parse_spec(spec)

    # CombFold output PDB
    combfold_pdb = os.path.join(
        OUTPUT_BASE, f"{complex_name}_output", "assembled_results", "output_clustered_0.pdb"
    )

    if not os.path.exists(combfold_pdb):
        print(f"  {cpx}: CombFold PDB not found at {combfold_pdb}")
        fail_reasons["CombFold PDB not found"] += 1
        continue

    # Get identity reference PDBs for this complex
    ref_pdbs = identity_pdbs[identity_pdbs["#Complex ac"] == cpx]["pdb_id"].unique()

    best_rmsd = None
    best_ref = None
    best_n_chains = 0
    last_status = "no ref PDBs found"

    for pdb_id in ref_pdbs:
        ref_path = os.path.join(COMPLEX_PDB_DIR, f"{cpx}_{pdb_id}.pdb")
        if not os.path.exists(ref_path):
            last_status = f"ref PDB file missing ({cpx}_{pdb_id}.pdb)"
            continue

        rmsd, n_chains, status = compute_complex_rmsd(
            complex_name, ref_path, combfold_pdb, spec_counts=spec_counts
        )
        last_status = status

        if rmsd is not None and (best_rmsd is None or rmsd < best_rmsd):
            best_rmsd = rmsd
            best_ref = pdb_id
            best_n_chains = n_chains

    if best_rmsd is not None:
        results.append({
            "Complex": cpx,
            "Best_ref_PDB": best_ref,
            "RMSD_A": round(best_rmsd, 2),
            "Chains_matched": best_n_chains,
            "CombFold_confidence": round(complex_conf.get(cpx, np.nan), 2),
            "N_ref_PDBs": len(ref_pdbs)
        })
        print(f"  {cpx}: RMSD={best_rmsd:.2f} A (ref={best_ref}, chains={best_n_chains})")
    else:
        print(f"  {cpx}: Could not compute RMSD -- {last_status}")
        fail_reasons[last_status] += 1

# Print failure summary
if fail_reasons:
    print(f"\n=== Failure reasons ===")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count}x: {reason}")

# Summary DataFrame
if not results:
    print("\nNo RMSD values computed. See diagnostic output above.")
else:
    rmsd_df = pd.DataFrame(results)
    print("\n=== RMSD Summary ===")
    print(rmsd_df.to_string(index=False))
    print(f"\nMedian RMSD: {rmsd_df['RMSD_A'].median():.2f} A")
    print(f"Mean RMSD:   {rmsd_df['RMSD_A'].mean():.2f} A")
    print(f"Range:       {rmsd_df['RMSD_A'].min():.2f} - {rmsd_df['RMSD_A'].max():.2f} A")

    # ---- Scatter plot: RMSD vs CombFold confidence ----
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rmsd_df["CombFold_confidence"], rmsd_df["RMSD_A"],
               s=60, color="#0279EE", edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_xlabel("CombFold Confidence", fontsize=12)
    ax.set_ylabel("RMSD (A)", fontsize=12)
    ax.set_title("CombFold Confidence vs Structural Accuracy (RMSD)", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Annotate outliers (RMSD > Q3 + 1.5*IQR)
    q1 = rmsd_df["RMSD_A"].quantile(0.25)
    q3 = rmsd_df["RMSD_A"].quantile(0.75)
    iqr = q3 - q1
    outlier_thresh = q3 + 1.5 * iqr
    for _, row in rmsd_df.iterrows():
        if row["RMSD_A"] > outlier_thresh:
            ax.annotate(row["Complex"], (row["CombFold_confidence"], row["RMSD_A"]),
                        fontsize=8, ha="left", va="bottom")
plt.tight_layout()
plt.savefig("/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/third_setup/rmsd_vs_confidence.png", dpi=400, bbox_inches="tight")
plt.savefig("/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/third_setup/rmsd_vs_confidence.svg", bbox_inches="tight")
plt.show()
