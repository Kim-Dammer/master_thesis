# Batch export for sharing with a supervisor:
#   one folder per complex, containing
#     <complex_ac>.pdb   — the CF prediction, FULL content (waters/ligands/nucleic kept)
#     <pdb_id>.pdb       — the reference structure, named by its PDB id, FULL content
#     <complex_ac>.pse   — a PyMOL session with BOTH loaded and colored; opens showing
#                           protein cartoons only, but nothing was deleted — waters,
#                           ligands, nucleic chains are still in the object, just hidden.
#                           e.g. `show sticks, hetatm` in PyMOL reveals ligands again.
# Then zips the whole output folder into one .zip.
#
# Usage:
#   pymol -cq export_for_supervisor.pml -- supervisor_export.csv supervisor_models/

python
import sys, os, csv, shutil
from pymol import cmd

args = sys.argv[1:]
csv_path, out_dir = args[0], args[1]
os.makedirs(out_dir, exist_ok=True)

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

print(f"exporting {len(rows)} complexes -> {out_dir}")

for i, r in enumerate(rows, 1):
    complex_ac = r["complex_ac"]
    pdb_id = r["pdb_id"]
    folder = os.path.join(out_dir, complex_ac)
    os.makedirs(folder, exist_ok=True)

    cmd.delete("all")
    cmd.load(r["reference_pdb_path"], "ref")
    cmd.load(r["usalign_pdb_path"], "pred")

    # NOTHING is deleted — waters, ligands, nucleic chains all stay in the object
    # and in the saved files. We only HIDE them by default so the session opens
    # looking clean; the supervisor can `show` them back with one click if wanted.
    cmd.hide("everything")
    cmd.show("cartoon", "polymer.protein")
    cmd.color("0x0072B2", "ref and polymer.protein")    # blue = reference
    cmd.color("0xE69F00", "pred and polymer.protein")   # orange = prediction
    cmd.bg_color("white")
    cmd.set("ray_shadows", 0)
    cmd.orient("ref")

    # named PDB exports — full, unmodified content (reference = PDB id, prediction = complex id)
    cmd.save(os.path.join(folder, f"{pdb_id}.pdb"), "ref")
    cmd.save(os.path.join(folder, f"{complex_ac}.pdb"), "pred")

    # ready-to-open session: opens clean (protein cartoons only), but waters/
    # ligands/nucleic are still in there — e.g. `show sticks, hetatm` reveals them
    cmd.save(os.path.join(folder, f"{complex_ac}.pse"))

    print(f"  {i}/{len(rows)}  {complex_ac}  (ref={pdb_id})")

# zip the whole thing up for easy sharing
zip_base = out_dir.rstrip("/")
shutil.make_archive(zip_base, "zip", out_dir)
print(f"wrote {zip_base}.zip")
python end
