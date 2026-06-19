#!/usr/bin/env python
"""
Rename AFDB PDB files based on their DBREF UniProt IDs.

Handles isoform-specific UniProt IDs (e.g. P06208-2, P07263-2).
By default, isoform suffixes are encoded as _isoN in the filename
(e.g. P06208-2 -> P06208_iso2), so both the canonical and isoform
structures get distinct filenames without collision:
  P06208.pdb       (canonical)
  P06208_iso2.pdb  (isoform 2)

Use --strip-isoform to drop the suffix entirely (P06208-2 -> P06208),
which will cause collisions if both canonical and isoform exist.

For a homodimer of P47177, the file is renamed to P47177.pdb (or
P47177_P47177.pdb with --homodimer-long).

For a heterodimer A-B, the file is renamed to <sorted A_B>.pdb.

Files that already match the target name are skipped. Files where no
DBREF lines are found are reported and left alone.
"""
import argparse
import re
from pathlib import Path

# DBREF line format (columns are fixed in the PDB spec):
# DBREF  XXXX A    1   404  UNP    P47177   2NDP_YEAST       1    404
# DBREF  XXXX A    1   404  UNP    P06208-2 YNL031C          1    404
DBREF_RE = re.compile(r"^DBREF\s+\S+\s+\S+\s+\d+\s+\d+\s+UNP\s+(\S+)")

# Isoform suffix pattern: -N at the end of a UniProt accession
ISOFORM_RE = re.compile(r"^(.+)-(\d+)$")


def encode_isoform(uniprot_id: str) -> str:
    """Encode isoform suffix as _isoN (e.g. P06208-2 -> P06208_iso2).
    Non-isoform IDs pass through unchanged."""
    m = ISOFORM_RE.match(uniprot_id)
    if m:
        return f"{m.group(1)}_iso{m.group(2)}"
    return uniprot_id


def strip_isoform(uniprot_id: str) -> str:
    """Remove isoform suffix entirely (e.g. P06208-2 -> P06208)."""
    m = ISOFORM_RE.match(uniprot_id)
    if m:
        return m.group(1)
    return uniprot_id


def extract_uniprot_ids(pdb_path: Path, mode: str = "encode") -> list[str]:
    """Return the ordered list of UniProt IDs from the DBREF lines.

    mode:
      "encode"  - P06208-2 -> P06208_iso2 (default, avoids collisions)
      "strip"   - P06208-2 -> P06208      (may collide with canonical)
      "raw"     - P06208-2 -> P06208-2    (keep as-is, dash in filename)
    """
    ids = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("DBREF"):
                # Once we hit ATOM/HETATM, DBREF lines are done
                if line.startswith(("ATOM", "HETATM", "MODEL")):
                    break
                continue
            m = DBREF_RE.match(line)
            if m:
                raw_id = m.group(1)
                if mode == "encode":
                    raw_id = encode_isoform(raw_id)
                elif mode == "strip":
                    raw_id = strip_isoform(raw_id)
                # mode == "raw": keep as-is
                ids.append(raw_id)
    return ids


def target_name(ids: list[str], homodimer_short: bool) -> str:
    """
    Build the target filename (without extension) from the UniProt IDs.
    """
    unique = sorted(set(ids))
    if len(unique) == 1 and homodimer_short:
        # Homodimer: just the protein ID
        return unique[0]
    # Heterodimer (or any other case): sorted, underscore-joined
    return "_".join(unique)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path, help="Folder containing AFDB *.pdb files")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be renamed but don't actually rename"
    )
    ap.add_argument(
        "--homodimer-long", action="store_true",
        help="Name homodimers P47177_P47177.pdb instead of P47177.pdb"
    )
    ap.add_argument(
        "--isoform-mode", choices=["encode", "strip", "raw"], default="encode",
        help="How to handle isoform IDs like P06208-2: "
             "'encode' -> P06208_iso2 (default, no collision), "
             "'strip' -> P06208 (may collide with canonical), "
             "'raw' -> P06208-2 (dash in filename)"
    )
    ap.add_argument(
        "--on-collision", choices=["skip", "overwrite", "suffix"], default="skip",
        help="What to do if the target name already exists "
             "(default: skip, leave original file with its AF-... name)"
    )
    args = ap.parse_args()

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    pdbs = sorted(folder.glob("*.pdb"))
    print(f"Found {len(pdbs)} PDB files in {folder}")

    stats = {"renamed": 0, "already": 0, "skipped_collision": 0,
             "no_dbref": 0, "isoform_encoded": 0, "errors": 0}

    for pdb in pdbs:
        try:
            ids = extract_uniprot_ids(pdb, mode=args.isoform_mode)
        except Exception as e:
            print(f"ERROR reading {pdb.name}: {e}")
            stats["errors"] += 1
            continue

        if not ids:
            print(f"NO DBREF: {pdb.name}")
            stats["no_dbref"] += 1
            continue

        # Track isoform encoding
        raw_ids = extract_uniprot_ids(pdb, mode="raw")
        if args.isoform_mode != "raw":
            for raw, processed in zip(raw_ids, ids):
                if raw != processed:
                    stats["isoform_encoded"] += 1

        new_stem = target_name(ids, homodimer_short=not args.homodimer_long)
        new_path = pdb.with_name(f"{new_stem}.pdb")

        if new_path == pdb:
            stats["already"] += 1
            continue

        if new_path.exists():
            if args.on_collision == "skip":
                print(f"COLLISION (skip): {pdb.name} -> {new_path.name} "
                      f"(target exists)")
                stats["skipped_collision"] += 1
                continue
            elif args.on_collision == "suffix":
                # find a free suffix
                i = 2
                while pdb.with_name(f"{new_stem}_v{i}.pdb").exists():
                    i += 1
                new_path = pdb.with_name(f"{new_stem}_v{i}.pdb")
                print(f"COLLISION (suffix): {pdb.name} -> {new_path.name}")
            elif args.on_collision == "overwrite":
                print(f"COLLISION (overwrite): {pdb.name} -> {new_path.name}")

        if args.dry_run:
            # Show isoform info in dry-run
            extra = ""
            if args.isoform_mode != "raw":
                changed_pairs = [(r, p) for r, p in zip(raw_ids, ids) if r != p]
                if changed_pairs:
                    extra = f"  [isoform: {' '.join(f'{r}->{p}' for r, p in changed_pairs)}]"
            print(f"DRY: {pdb.name} -> {new_path.name}{extra}")
        else:
            pdb.rename(new_path)
            print(f"OK:  {pdb.name} -> {new_path.name}")
        stats["renamed"] += 1

    print()
    print("Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
