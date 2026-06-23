#!/bin/bash
#SBATCH --account=es_biol
#SBATCH --partition=es_biol
#SBATCH --job-name=download_pdbs
#SBATCH --output=/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files/logs/download_pdbs_%j.out
#SBATCH --error=/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files/logs/download_pdbs_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# ===========================================================================
# Download PDB files cross-referenced from Complex Portal for all yeast
# complexes. Organizes into identity/ and subset/ folders.
#
# Just run:  sbatch 15_download_complex_pdbs.sbatch
# ===========================================================================

set -euo pipefail

VENV="/cluster/project/beltrao/kdammer/master_thesis/.venv"
OUT_DIR="/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files"
TSV="/cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv"

mkdir -p "${OUT_DIR}/logs"

source "${VENV}/bin/activate"

echo "[$(date)] Starting PDB download..."

python - <<'PYEOF'
import csv, re, sys, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# --- Config ---
TSV = "/cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv"
OUT_DIR = Path("/cluster/project/beltrao/kdammer/master_thesis/data/Complex_pdb_files")
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DELAY = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 5

# --- Parse cross-references ---
def parse_pdb_xrefs(cross_ref_str):
    result = {"identity": [], "subset": []}
    if not isinstance(cross_ref_str, str) or not cross_ref_str:
        return result
    for token in cross_ref_str.split("|"):
        token = token.strip()
        m = re.match(r"^wwpdb:([A-Za-z0-9]{4})\((identity|subset)\)$", token)
        if m:
            pdb_id = m.group(1).upper()
            tag = m.group(2)
            if pdb_id not in result[tag]:
                result[tag].append(pdb_id)
    return result

# --- Download ---
def download_pdb(pdb_id, dest_path):
    url = RCSB_URL.format(pdb_id=pdb_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url, headers={"User-Agent": "ComplexPDBDownloader/1.0"})
            with urlopen(req, timeout=30) as resp:
                content = resp.read()
            if not content:
                print(f"  [WARN] Empty response for {pdb_id}")
                return False
            dest_path.write_bytes(content)
            return True
        except HTTPError as e:
            if e.code == 404:
                print(f"  [SKIP] {pdb_id} not found on RCSB (404)")
                return False
            print(f"  [RETRY] {pdb_id} HTTP {e.code} (attempt {attempt}/{MAX_RETRIES})")
        except URLError as e:
            print(f"  [RETRY] {pdb_id} URL error: {e.reason} (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            print(f"  [RETRY] {pdb_id} error: {e} (attempt {attempt}/{MAX_RETRIES})")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    print(f"  [FAIL] {pdb_id} after {MAX_RETRIES} attempts")
    return False

# --- Main ---
identity_dir = OUT_DIR / "identity"
subset_dir = OUT_DIR / "subset"
raw_dir = OUT_DIR / "_raw"
identity_dir.mkdir(parents=True, exist_ok=True)
subset_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)

# Parse TSV
print(f"Reading {TSV}")
cpx_to_pdbs = {}
with open(TSV, newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        cpx_id = row.get("#Complex ac", "").strip()
        if not cpx_id:
            continue
        pdbs = parse_pdb_xrefs(row.get("Cross references", ""))
        if pdbs["identity"] or pdbs["subset"]:
            cpx_to_pdbs[cpx_id] = pdbs

# Collect unique PDB IDs and which CPX IDs reference them
pdb_to_cpxs = {}
for cpx_id, pdbs in cpx_to_pdbs.items():
    for pdb_id in pdbs["identity"]:
        pdb_to_cpxs.setdefault(pdb_id, {"identity": [], "subset": []})
        if cpx_id not in pdb_to_cpxs[pdb_id]["identity"]:
            pdb_to_cpxs[pdb_id]["identity"].append(cpx_id)
    for pdb_id in pdbs["subset"]:
        pdb_to_cpxs.setdefault(pdb_id, {"identity": [], "subset": []})
        if cpx_id not in pdb_to_cpxs[pdb_id]["subset"]:
            pdb_to_cpxs[pdb_id]["subset"].append(cpx_id)

all_identity = {p for p, t in pdb_to_cpxs.items() if t["identity"]}
all_subset = {p for p, t in pdb_to_cpxs.items() if t["subset"]}

print(f"Found {len(cpx_to_pdbs)} complexes with PDB cross-references")
print(f"  Identity PDBs: {len(all_identity)} unique")
print(f"  Subset PDBs:   {len(all_subset)} unique")

# Download each unique PDB once
downloaded = 0
skipped = 0
failed = 0

print(f"\nDownloading {len(pdb_to_cpxs)} unique PDB files...")
for i, (pdb_id, tags) in enumerate(sorted(pdb_to_cpxs.items()), start=1):
    raw_path = raw_dir / f"{pdb_id}.pdb"
    if raw_path.exists() and raw_path.stat().st_size > 0:
        skipped += 1
        continue
    print(f"  [{i}/{len(pdb_to_cpxs)}] {pdb_id}...", end=" ", flush=True)
    if download_pdb(pdb_id, raw_path):
        downloaded += 1
        print("OK")
    else:
        failed += 1
        continue
    time.sleep(DELAY)

# Organize into per-CPX files
print(f"\nOrganizing into identity/ and subset/...")
for pdb_id, tags in sorted(pdb_to_cpxs.items()):
    raw_path = raw_dir / f"{pdb_id}.pdb"
    if not raw_path.exists():
        continue
    for cpx_id in tags["identity"]:
        dest = identity_dir / f"{cpx_id}_{pdb_id}.pdb"
        if not dest.exists():
            dest.write_bytes(raw_path.read_bytes())
    for cpx_id in tags["subset"]:
        dest = subset_dir / f"{cpx_id}_{pdb_id}.pdb"
        if not dest.exists():
            dest.write_bytes(raw_path.read_bytes())

# Write mapping CSV
mapping_path = OUT_DIR / "complex_pdb_mapping.csv"
with open(mapping_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["#Complex ac", "pdb_id", "tag"])
    for cpx_id, pdbs in sorted(cpx_to_pdbs.items()):
        for pdb_id in pdbs["identity"]:
            writer.writerow([cpx_id, pdb_id, "identity"])
        for pdb_id in pdbs["subset"]:
            writer.writerow([cpx_id, pdb_id, "subset"])

n_id = len(list(identity_dir.glob("*.pdb")))
n_sub = len(list(subset_dir.glob("*.pdb")))
print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
print(f"  identity/ files: {n_id}")
print(f"  subset/   files: {n_sub}")
print(f"  Mapping: {mapping_path}")
PYEOF

echo "[$(date)] Download complete."
