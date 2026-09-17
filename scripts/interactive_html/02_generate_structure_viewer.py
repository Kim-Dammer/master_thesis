#!/usr/bin/env python3
"""
Generate self-contained HTML viewer for CP complexes with no PDB structure coverage.

For each complex (CF confidence > 80, no/partial homology PDB match):
  - Left panel : CombFold assembled model; hover shows UniProt ID per chain
  - Right panel: PDB reference structures (greedy set-cover); hover shows SIFTS
                 UniProt + whether chain is in the CP complex

Color modes:  Single (yellow) | By chain | By mapping
  "By mapping": CF chains colored by CP protein; PDB chains colored if their
  SIFTS UniProt is in the complex (same palette), gray if not.

Clicking a protein name in the "Mapped/Novel/No hit" summary lines opens a
popup with the raw SIFTS + MMseq evidence rows for that protein against the
currently selected reference PDB.

Output: data/CP_complexes_no_struct_coverage/structure_viewer.html
"""
import base64, gzip, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm
from procompa import get_project_root

PRJ_ROOT = get_project_root()
DATA     = PRJ_ROOT / "data"
CF_BASE  = DATA / "Pipeline/10_all_CP_complexes/CombFold"
OUT      = DATA / "Pipeline/viewer/02_structure_viewer_visualize_plddt_80.html"
MMSEQ    = DATA / "CP_complexes_no_struct_coverage/sanity_checks/mmseq_no_Strcut_filtered.parquet"
ANNOT    = DATA / "CP_complexes_no_struct_coverage/complex_pdb_annotations_map.csv"

MAX_PDB_REFS    = 20   # safeguard cap on reference PDBs shown per complex
MAX_INPUT_PAIRS = 15   # safeguard cap on embedded pairwise AF models per complex
MMSEQ_RAW = PRJ_ROOT / "scripts/mmseq_homology_match/mmseqs/mmseqs_run_max_sensitivity/results/mmseqs_new_identity_similarity_max_sensitivity.parquet"
SET_COVER_PARQUET = DATA / "CP_complexes_no_struct_coverage/minimal_complex_pdb_set_cover_max_identity.parquet"  #greddy pdb approach, but with max identity sort


# ── helpers ───────────────────────────────────────────────────────────────────

def uniprot_proteins(identifiers: str) -> set[str]:
    out = set()
    for part in identifiers.split("|"):
        part = part.strip()
        if part.startswith(("CHEBI", "URS")):
            continue
        m = re.match(r"^([A-Z][A-Z0-9]{5,})", part)
        if m:
            out.add(m.group(1))
    return out


def folder_proteins(name: str) -> set[str]:
    base = name.replace("_pool_output", "").replace("_pool_input", "")
    return set(re.findall(r"([A-Z][A-Z0-9]{5,})x\d+", base))


def folder_stoic_label(name: str) -> str:
    base  = name.replace("_pool_output", "")
    parts = re.findall(r"([A-Z][A-Z0-9]{5,})x(\d+)", base)
    return "  |  ".join(f"{p}x{n}" for p, n in parts)


def parse_confidence(conf_path: Path) -> list[dict]:
    models = []
    try:
        for line in conf_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    models.append({"path": Path(parts[0]), "score": float(parts[1])})
                except ValueError:
                    pass
    except Exception as e:
        print(f"  WARNING {conf_path}: {e}", file=sys.stderr)
    return sorted(models, key=lambda x: x["score"], reverse=True)


def compress(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode())).decode()


def strip_pdb(text: str) -> str:
    """Keep only coordinate records, discarding REMARK/HEADER/SEQRES etc.

    AlphaFold2 unrelaxed PDBs embed the full PAE matrix in REMARK 3, which
    can be 3–10 MB for large proteins. Stripping to ATOM/HETATM/TER/END
    reduces each pairwise input file to a few hundred KB before gzip, which
    then compresses to ~50 KB -- essential for keeping the HTML file small."""
    _KEEP = frozenset({"ATOM", "HETATM", "MODEL", "ENDMDL", "TER", "END"})
    return "\n".join(
        line for line in text.splitlines()
        if line[:6].strip() in _KEEP
    )



def read_chain_list(folder_name: str) -> dict[str, str] | None:
    """Ground-truth chain -> UniProt map from CombFold's own output manifest.
    Format: one 'UNIPROT_CHAIN.pdb' line per chain, e.g. 'P41735_E.pdb'.
    Returns None if the file is missing (caller falls back to the
    positional folder-name-parsing heuristic)."""
    path = CF_BASE / folder_name / "_unified_representation" / "assembly_output" / "chain.list"
    if not path.exists():
        return None
    mapping: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pdb$", line)
            if not m:
                print(f"  WARNING unparseable chain.list line in {folder_name}: {line!r}",
                      file=sys.stderr)
                continue
            uniprot, chain = m.group(1), m.group(2)
            mapping[chain] = uniprot
    except Exception as e:
        print(f"  WARNING failed to read chain.list for {folder_name}: {e}", file=sys.stderr)
        return None
    return mapping or None


def build_cf_chain_map(pdb_text: str, folder_name: str) -> dict[str, str]:
    """Map CombFold output chain letter -> UniProt accession.

    Primary source: CombFold's own chain.list manifest
    (_unified_representation/assembly_output/chain.list), which states the
    UNIPROT_CHAIN assignment directly -- no positional guessing.

    Fallback (only when chain.list is missing): the previous positional
    heuristic, which zips PDB-file chain-appearance-order against
    folder-name-parse-order. This has no independent ground truth and is
    known to break for large complexes with truncated folder names -- kept
    only so older/incomplete CombFold runs still get a best-effort mapping,
    with a warning so affected complexes are identifiable in the build log.
    """
    manifest = read_chain_list(folder_name)

    chains_in_model: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and len(line) > 21:
            ch = line[21]
            if ch not in chains_in_model:
                chains_in_model.append(ch)

    if manifest is not None:
        missing = [ch for ch in chains_in_model if ch not in manifest]
        if missing:
            print(f"  WARNING chain.list for {folder_name} has no entry for "
                  f"model chains {missing} -- these will render as 'unknown'",
                  file=sys.stderr)
        return manifest

    print(f"  WARNING no chain.list found for {folder_name} -- "
          f"falling back to positional folder-name matching (less reliable)",
          file=sys.stderr)
    base = folder_name.replace("_pool_output", "")
    uniprot_list = [
        acc
        for acc, n in re.findall(r"([A-Z][A-Z0-9]{5,})x(\d+)", base)
        for _ in range(int(n))
    ]
    if len(chains_in_model) != len(uniprot_list):
        print(
            f"  WARNING chain/uniprot count mismatch in {folder_name}: "
            f"{len(chains_in_model)} chains {chains_in_model} vs "
            f"{len(uniprot_list)} parsed from folder name {uniprot_list} -- "
            f"unmapped chains will render as 'unknown', not as uncovered",
            file=sys.stderr,
        )
    return {ch: uniprot_list[i] for i, ch in enumerate(chains_in_model) if i < len(uniprot_list)}


def build_pdb_chain_map(pdb_id: str, sifts_lookup: dict, complex_proteins: set) -> dict:
    """Map PDB chain letter -> {u: UniProt, cp: bool} via SIFTS."""
    return {
        chain: {"u": uni, "cp": uni in complex_proteins}
        for chain, uni in sifts_lookup.get(pdb_id.lower(), {}).items()
    }


def build_pdb_homology_map(pdb_id: str, homology_lookup: dict, complex_proteins: set) -> dict:
    """Map PDB chain letter -> list of CP proteins with homology hit to that chain."""
    return {
        chain: [p for p in prots if p in complex_proteins]
        for chain, prots in homology_lookup.get(pdb_id.lower(), {}).items()
    }


def read_input_models(input_folder_name: str, max_pairs: int | None = None) -> list[dict]:
    """Read, strip, and compress pairwise AlphaFold input PDBs.

    Expects files matching AFM_PROT1_PROT2_unrelaxed_*.pdb under
    <CF_BASE>/<input_folder_name>/pdbs/.  REMARK blocks are stripped before
    compression so PAE data (potentially several MB per file) is not embedded.
    Returns list of dicts: {filename, label, proteins: [p1, p2], pdb_gz}.

    max_pairs: if set, stop after reading this many files (sorted alphabetically)
    so that large complexes don't bloat the output HTML."""
    pdbs_dir = CF_BASE / input_folder_name / "pdbs"
    if not pdbs_dir.exists():
        print(f"  WARNING no input pdbs/ dir at {pdbs_dir} -- "
              f"Input Pairs tab will be empty for this complex", file=sys.stderr)
        return []
    all_pdb_files = sorted(pdbs_dir.glob("*.pdb"))
    models: list[dict] = []
    unmatched = 0
    for pdb_file in all_pdb_files:
        if max_pairs is not None and len(models) >= max_pairs:
            print(f"  WARNING input pairs capped at {max_pairs} "
                  f"(found {len(all_pdb_files)} files in {pdbs_dir})", file=sys.stderr)
            break
        m = re.match(r"AFM_([A-Z][A-Z0-9]{5,})_([A-Z][A-Z0-9]{5,})_", pdb_file.name)
        if not m:
            unmatched += 1
            continue
        p1, p2   = m.group(1), m.group(2)
        proteins = [p1, p2]
        label    = f"{p1} – {p2}" if p1 != p2 else f"{p1} (homo)"
        try:
            text     = pdb_file.read_text()
            stripped = strip_pdb(text)
            models.append({
                "filename": pdb_file.name,
                "label"   : label,
                "proteins": proteins,
                "pdb_gz"  : compress(stripped),
            })
        except Exception as e:
            print(f"  WARNING read input {pdb_file}: {e}", file=sys.stderr)
    if unmatched:
        print(f"  WARNING {unmatched}/{len(all_pdb_files)} files in {pdbs_dir} "
              f"did not match the AFM_<P1>_<P2>_ naming pattern -- "
              f"check for a different filename convention on large complexes",
              file=sys.stderr)
    return models


def sniff_delimiter(path: Path, default: str = ",") -> str:
    """Detect comma- vs tab-separated CSV from the first non-comment line."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            return "\t" if line.count("\t") > line.count(",") else ","
    return default


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading CSV / parquets...")
info_df = pl.read_csv(
    DATA / "CP_complexes_no_struct_coverage/confident_CF_complexes_with_pdb_match_info.csv"
)
complexes_df = (
    info_df
    .select(["complex_ac", "identifiers", "CF_confidence_max", "match_class"])
    .unique("complex_ac")
    .sort("CF_confidence_max", descending=True)
)
complexes_df = complexes_df.filter(pl.col("complex_ac") != "CPX-1602")
print(f"  {len(complexes_df)} complexes (CF > 80)")

pdb_hits = pl.read_parquet(
    DATA / "CP_complexes_no_struct_coverage/complex_pdb_hits.parquet"
)

print("Loading precomputed PDB set cover...")
assert SET_COVER_PARQUET.exists(), f"Set cover parquet not found: {SET_COVER_PARQUET}"
_sc = pl.read_parquet(SET_COVER_PARQUET)
assert {"complex_ac", "cover_100_pdbs", "cover_75_pdbs"}.issubset(set(_sc.columns)), (
    f"Set cover parquet missing expected columns; got {_sc.columns}"
)
cover_lookup: dict[str, dict[str, list[str]]] = {
    r["complex_ac"]: {
        "cover_100": r["cover_100_pdbs"] or [],
        "cover_75":  r["cover_75_pdbs"]  or [],
    }
    for r in _sc.iter_rows(named=True)
}
print(f"  {len(cover_lookup)} complexes with precomputed cover")

# Complex Portal annotations: complex-level name + per-PDB description.
# Coverage is partial by design (not every reference PDB has an entry) --
# lookups below must return None cleanly for missing pairs, not raise.
# The browser fills the gap for missing per-PDB rows with RCSB's own
# entry title, fetched alongside the structure at view time.
print("Loading complex/PDB annotations...")
assert ANNOT.exists(), f"Annotation file not found: {ANNOT}"
annot_df = pl.read_csv(ANNOT)
required_cols = {"complex_ac", "pdb_id", "cp_annotation", "pdb_annotation"}
assert required_cols.issubset(set(annot_df.columns)), (
    f"Annotation file missing columns: {required_cols - set(annot_df.columns)}"
)

cp_annotation_lookup: dict[str, str] = {}
pdb_annotation_lookup: dict[tuple[str, str], str] = {}
for r in annot_df.iter_rows(named=True):
    ac_  = r["complex_ac"]
    pid_ = (r["pdb_id"] or "").lower()
    cp_a  = r["cp_annotation"]
    pdb_a = r["pdb_annotation"]
    if ac_ and cp_a and ac_ not in cp_annotation_lookup:
        cp_annotation_lookup[ac_] = cp_a
    if ac_ and pid_ and pdb_a:
        pdb_annotation_lookup[(ac_, pid_)] = pdb_a
print(f"  {len(cp_annotation_lookup)} complex annotations, "
      f"{len(pdb_annotation_lookup)} complex/PDB pair annotations")

# SIFTS: PDB chain -> UniProt
print("Loading SIFTS...")
sifts_path = DATA / "pdb" / "pdb_chain_uniprot.csv"
assert sifts_path.exists(), f"SIFTS file not found: {sifts_path}"

sifts_sep = sniff_delimiter(sifts_path)
print(f"  detected delimiter: {'TAB' if sifts_sep == chr(9) else 'COMMA'}")
sifts_df = pl.read_csv(sifts_path, comment_prefix="#", infer_schema_length=0,
                        separator=sifts_sep)
assert "CHAIN" in sifts_df.columns and "SP_PRIMARY" in sifts_df.columns, (
    f"SIFTS columns mismatch — expected CHAIN/SP_PRIMARY, got {sifts_df.columns}. "
    f"Detected separator was {sifts_sep!r} -- check the file's actual format."
)

sifts_lookup: dict[str, dict[str, str]] = {}
pdb_col = sifts_df.columns[0]
for pdb, chain, uni in sifts_df.select([pdb_col, "CHAIN", "SP_PRIMARY"]).iter_rows():
    if pdb and chain and uni:
        sifts_lookup.setdefault(pdb.lower(), {}).setdefault(chain, uni)
assert len(sifts_lookup) > 0, "SIFTS lookup is empty after parsing"
print(f"  {len(sifts_lookup)} PDB entries loaded")

# MMseq homology (pre-filtered): PDB chain -> CP protein (cross-species)
print("Loading MMseq homology hits...")
assert MMSEQ.exists(), f"MMseq file not found: {MMSEQ}"
mmseq_df = pl.read_parquet(MMSEQ)
homology_lookup: dict[str, dict[str, list[str]]] = {}
for protein, hit in mmseq_df.select(["protein_id", "hit_pdb_id"]).iter_rows():
    parts = hit.split("_")
    if len(parts) >= 2:
        pid, chain = parts[0].lower(), parts[1]
        homology_lookup.setdefault(pid, {}).setdefault(chain, []).append(protein)
assert len(homology_lookup) > 0, "MMseq homology lookup is empty after parsing"
print(f"  {len(homology_lookup)} PDB entries with homology hits")

# Pre-scan ALL pool_output folders once
print("Scanning CombFold output folders...")
all_folders: dict[str, tuple[Path, list]] = {}
for d in CF_BASE.iterdir():
    if not d.name.endswith("_pool_output"):
        continue
    conf = d / "assembled_results" / "confidence.txt"
    if not conf.exists():
        continue
    models = parse_confidence(conf)
    if models:
        all_folders[d.name] = (d / "assembled_results", models)
print(f"  {len(all_folders)} folders with assembled results")

# Index folders by protein so per-complex matching is O(candidates) instead
# of O(all_folders). Previously folder_proteins() (a regex scan) ran once
# per complex PER FOLDER -- for many complexes against many folders this
# dominated runtime. Now it's computed once per folder, and per-complex
# lookup only has to check folders that actually share a protein with the
# complex, via an inverted index -- same result set as the full scan
# (still verified with the `fp <= target` subset check below), just far
# fewer candidates to check.
print("Indexing CombFold folders by protein (for fast complex matching)...")
folder_proteins_cache: dict[str, frozenset[str]] = {
    fname: frozenset(folder_proteins(fname)) for fname in all_folders
}
protein_to_folders: dict[str, list[str]] = {}
for fname, fp in folder_proteins_cache.items():
    for p in fp:
        protein_to_folders.setdefault(p, []).append(fname)
print(f"  indexed {len(protein_to_folders)} distinct proteins across folders")


# ── process each complex ──────────────────────────────────────────────────────

EMBED: dict[str, dict] = {}

def process_complex(row: dict) -> tuple[str, dict]:
    ac          = row["complex_ac"]
    identifiers = row["identifiers"]
    cf_max      = float(row["CF_confidence_max"])
    match_class = row["match_class"] or "unknown"
    target      = uniprot_proteins(identifiers)

    log = [f"\n{ac}  CF={cf_max:.1f}  {match_class}  ({len(target)} proteins)"]

    best_models: list[dict] = []
    best_label  = ""
    best_delta  = float("inf")

    def read_models(fname, models):
        candidate = []
        for m in models[:1]:
            if m["path"].exists():
                try:
                    text = m["path"].read_text()
                    candidate.append({
                        "name"     : m["path"].name,
                        "score"    : m["score"],
                        "pdb_gz"   : compress(text),
                        "chain_map": build_cf_chain_map(text, fname),
                        "folder"   : fname,
                        "path"     : str(m["path"]),
                    })
                except Exception as e:
                    log.append(f"  WARNING read {m['path']}: {e}")
        return candidate

    # Pass 1: protein-set match (exact or folder subset of target).
    # Only folders sharing at least one protein with `target` can possibly
    # satisfy `fp <= target` (fp non-empty), so the inverted index narrows
    # the scan to just those instead of walking every folder.
    candidate_fnames = set()
    for p in target:
        candidate_fnames.update(protein_to_folders.get(p, []))
    for fname in candidate_fnames:
        fp = folder_proteins_cache[fname]
        assembled, models = all_folders[fname]
        delta = abs(models[0]["score"] - cf_max)
        if fp and fp <= target and delta < best_delta:
            candidate = read_models(fname, models)
            if candidate:
                best_delta, best_label, best_models = delta, folder_stoic_label(fname), candidate

    # Pass 2: score-only fallback (large complexes with truncated folder names)
    if not best_models:
        log.append(f"  no protein-set match -- trying score-only fallback")
        for fname, (assembled, models) in all_folders.items():
            delta = abs(models[0]["score"] - cf_max)
            if delta < best_delta:
                candidate = read_models(fname, models)
                if candidate:
                    best_delta  = delta
                    best_label  = folder_stoic_label(fname) + "  (score-match only)"
                    best_models = candidate

    if best_models:
        log.append(f"  CF models: {len(best_models)}  delta={best_delta:.3f}  [{best_label[:60]}]")
    else:
        log.append(f"  WARNING: no CF models found")

    # PDB set cover -- precomputed; IDs only, browser fetches on demand
    _ac_cover = cover_lookup.get(ac, {})
    cover = _ac_cover.get("cover_100") or _ac_cover.get("cover_75") or []

    # pdb_proteins: still needed for the embed (maps each cover PDB -> proteins)
    pdb_proteins: dict[str, list[str]] = {}
    hits = pdb_hits.filter(pl.col("complex_ac") == ac)
    for h in hits.iter_rows(named=True):
        if h["pdb_id"] and h["proteins"]:
            pdb_proteins[h["pdb_id"]] = list(h["proteins"])

    pdb_cap_hit = len(cover) > MAX_PDB_REFS
    if pdb_cap_hit:
        log.append(f"  WARNING {ac}: PDB cover has {len(cover)} entries, "
                   f"capping at {MAX_PDB_REFS}")
    cover = cover[:MAX_PDB_REFS]
    log.append(f"  PDB cover: {cover}")

    cf_chain_map    = best_models[0].get("chain_map", {}) if best_models else {}
    if best_models:
        assert cf_chain_map, (
            f"CF chain map is empty for {ac} — check folder name parsing "
            f"(folder: {best_label})"
        )
    cf_models_clean = [{"name": m["name"], "score": m["score"], "pdb_gz": m["pdb_gz"],
                        "folder": m["folder"], "path": m["path"]}
                       for m in best_models]

    # Input pairwise AF models: stripped of REMARK/PAE before compression.
    # Capped defensively -- large complexes (many chains) can have a large
    # number of required pairwise predictions, and each embedded pair adds
    # to the output HTML's size, so an uncapped read risks bloating/slowing
    # the page precisely for the large complexes where this was reported broken.
    input_folder      = best_models[0]["folder"].replace("_pool_output", "_pool_input") if best_models else ""
    input_models_list = read_input_models(input_folder, max_pairs=MAX_INPUT_PAIRS) if input_folder else []
    if input_folder and not (CF_BASE / input_folder).exists():
        log.append(f"  WARNING expected input folder does not exist: {CF_BASE / input_folder} "
                   f"(derived from CF folder {best_models[0]['folder']!r} by suffix swap -- "
                   f"if the real folder is named differently for large complexes, this is why "
                   f"Input Pairs is empty)")
    if input_models_list:
        kb = sum(len(m["pdb_gz"]) for m in input_models_list) // 1024
        log.append(f"  Input pairs: {len(input_models_list)} ({kb} KB compressed)")
    else:
        log.append(f"  Input pairs: none found")

    entry = {
        "complex_ac"     : ac,
        "identifiers"    : identifiers,
        "cf_confidence"  : cf_max,
        "match_class"    : match_class,
        "stoic_label"    : best_label,
        "cf_models"      : cf_models_clean,
        "pdb_ids"        : cover,
        "pdb_cap_hit"    : pdb_cap_hit,
        "cp_proteins"    : sorted(target),
        "cf_chain_map"   : cf_chain_map,
        "cp_annotation"  : cp_annotation_lookup.get(ac),
        "pdb_proteins"   : {pid: pdb_proteins.get(pid, []) for pid in cover},
        "pdb_annotation" : {pid: pdb_annotation_lookup.get((ac, pid.lower())) for pid in cover},
        "pdb_chain_map": {
            pid: build_pdb_chain_map(pid, sifts_lookup, target)
            for pid in cover
        },
        "pdb_homology_map": {
            pid: build_pdb_homology_map(pid, homology_lookup, target)
            for pid in cover
        },
        "input_models": input_models_list,
    }

    # Verify that PDB chain maps are actually populated from SIFTS
    if cover:
        mapped = [pid for pid in cover if entry["pdb_chain_map"].get(pid)]
        assert mapped, (
            f"No PDB chain maps populated for {ac} (PDB IDs: {cover}). "
            f"Check that SIFTS covers these PDB entries."
        )
    if entry["cp_annotation"] is None:
        log.append(f"  NOTE: no Complex Portal annotation found for {ac}")

    tqdm.write("\n".join(log))
    return ac, entry


# I/O-bound (file reads, gzip compression releases the GIL) and each complex
# is independent, so this parallelizes cleanly -- mirrors the
# ThreadPoolExecutor(max_workers=16) pattern used elsewhere in this pipeline.
# executor.map preserves input order, so EMBED still ends up sorted by
# CF_confidence_max descending (same dropdown order as before), regardless
# of which complex's thread happens to finish first.
_rows = list(complexes_df.iter_rows(named=True))
with ThreadPoolExecutor(max_workers=16) as executor:
    for ac, entry in tqdm(executor.map(process_complex, _rows), total=len(_rows),
                           desc="Assembling complexes"):
        EMBED[ac] = entry


# ── detailed per-protein evidence (for click-to-inspect in the viewer) ─────
print("\nBuilding per-protein evidence (SIFTS + MMseq detail) for covered PDBs...")

used_pdbs_lower: set[str] = set()
all_target_proteins: set[str] = set()
for entry in EMBED.values():
    used_pdbs_lower.update(p.lower() for p in entry["pdb_ids"])
    all_target_proteins.update(entry["cp_proteins"])

# SIFTS detail: pdb_lower -> accession -> list of raw row dicts. All columns
# are passed through as-is (only CHAIN/SP_PRIMARY are known for certain) so
# whatever range/extra columns the real SIFTS file has are never silently
# dropped or mislabeled.
sifts_detail: dict[str, dict[str, list[dict]]] = {}
sifts_detail_df = sifts_df.filter(
    pl.col(pdb_col).str.to_lowercase().is_in(list(used_pdbs_lower))
)
for r in sifts_detail_df.iter_rows(named=True):
    pdb_l = (r[pdb_col] or "").lower()
    acc   = r["SP_PRIMARY"]
    if not pdb_l or not acc:
        continue
    sifts_detail.setdefault(pdb_l, {}).setdefault(acc, []).append(r)
n_sifts_detail = sum(len(v2) for v in sifts_detail.values() for v2 in v.values())
print(f"  SIFTS detail rows kept: {n_sifts_detail}")

# MMseq detail: reproduces the filtering given verbatim (identity>30 OR
# blast_identity>30, alnlen>30), restricted to proteins appearing in at
# least one target complex, from the RAW (unfiltered) mmseq parquet.
assert MMSEQ_RAW.exists(), f"MMseq raw parquet not found: {MMSEQ_RAW}"
mmseqs_raw = pl.read_parquet(MMSEQ_RAW)
mmseqs_detail_df = (
    mmseqs_raw
    .filter(pl.col("protein_id").is_in(list(all_target_proteins)))
    .filter((pl.col("identity_percent") > 30) | (pl.col("blast_identity_percent") > 30))
    .filter(pl.col("alnlen") > 30)
)
mmseqs_detail: dict[tuple[str, str], list[dict]] = {}
for r in mmseqs_detail_df.iter_rows(named=True):
    hit = r.get("hit_pdb_id") or ""
    parts = hit.split("_", 1)
    if len(parts) < 2:
        continue
    pdb_l, chain = parts[0].lower(), parts[1]
    if pdb_l not in used_pdbs_lower:
        continue
    mmseqs_detail.setdefault((r["protein_id"], pdb_l), []).append(r)
print(f"  MMseq detail hit-groups kept: {len(mmseqs_detail)}")

# Attach, per complex/pdb/protein, whatever evidence exists. Entries with
# no evidence at all are omitted rather than padding the JSON with
# empty lists.
for entry in EMBED.values():
    evidence: dict[str, dict[str, dict]] = {}
    for pid in entry["pdb_ids"]:
        pid_l = pid.lower()
        per_pdb = {}
        for protein in entry["cp_proteins"]:
            direct   = sifts_detail.get(pid_l, {}).get(protein, [])
            homology = mmseqs_detail.get((protein, pid_l), [])
            if direct or homology:
                per_pdb[protein] = {"direct": direct, "homology": homology}
        evidence[pid] = per_pdb
    entry["pdb_protein_evidence"] = evidence


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CP Complex Structure Viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body {
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:13px; color:rgb(36,36,36);
  background:#f0f0f0; padding:10px;
  height:100vh; display:flex; flex-direction:column; gap:8px;
}
.topbar {
  background:white; border:1px solid #ddd; border-radius:6px;
  padding:9px 14px; display:flex; align-items:center; gap:12px;
  flex-shrink:0; flex-wrap:wrap;
}
.topbar label { font-weight:600; white-space:nowrap; }
#complex-select {
  flex:1; min-width:220px; max-width:560px;
  padding:4px 8px; border:1px solid #ccc; border-radius:4px; font-size:13px;
}
.badge {
  display:inline-block; padding:2px 10px; border-radius:10px;
  font-size:11px; font-weight:600; white-space:nowrap;
}
.badge-cf    { background:#ddeeff; color:#004a80; border:1px solid #aaccee; }
.badge-match { background:#fff0d5; color:#6b3d00; border:1px solid #e8c97a; }
#stoic-label {
  font-size:11px; color:#aaa; font-family:monospace;
  max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.path-label {
  position:absolute; left:8px; bottom:6px; z-index:5; pointer-events:none;
  font-size:10px; color:#777; font-family:monospace;
  background:rgba(255,255,255,.85); padding:2px 6px; border-radius:4px;
  max-width:75%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.mapped-summary {
  flex-basis:100%; font-size:11px; font-family:monospace;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.mapped-summary.right { text-align:right; }
.mapped-summary .m-blue   { color:#0072B2; font-weight:600; }
.mapped-summary .m-off    { color:#999;    font-weight:600; }
.mapped-summary .m-purple { color:#8E5FBF; font-weight:600; }
.cp-annotation {
  flex-basis:100%; font-size:12px; font-weight:600; color:#333;
  white-space:normal; line-height:1.35;
}
.pdb-annotation {
  flex-basis:100%; font-size:11px; color:#666; font-style:italic;
  white-space:normal; line-height:1.3;
}
.cap-warning {
  font-size:11px; font-weight:600; color:#a83232;
  background:#fde3e3; border:1px solid #f0b3b3;
  border-radius:4px; padding:1px 8px; display:none;
}
.cap-warning.on { display:inline-block; }
.color-ctrl {
  margin-left:auto; display:flex; align-items:center;
  gap:6px; font-weight:normal; white-space:nowrap; font-size:12px;
}
.color-ctrl select {
  padding:2px 6px; border:1px solid #ccc; border-radius:4px; font-size:12px;
}
.plddt-ctrl {
  display:flex; align-items:center; gap:6px; font-weight:normal;
  white-space:nowrap; font-size:12px;
}
.plddt-ctrl input[type=range] { width:100px; vertical-align:middle; }
.plddt-ctrl input[type=number] {
  width:46px; font-family:monospace; padding:1px 3px;
  border:1px solid #ccc; border-radius:3px; font-size:12px;
}
.plddt-ctrl input:disabled { opacity:.4; }
.plddt-mode-label {
  display:flex; align-items:center; gap:3px; cursor:pointer;
}
.plddt-mode-label input[type=radio] { margin:0; cursor:pointer; }
.plddt-legend {
  display:flex; align-items:center; gap:4px; font-size:11px; color:#666;
}
.plddt-legend .sw {
  display:inline-block; width:9px; height:9px; border-radius:2px;
}
.panels {
  display:grid; grid-template-columns:1fr 1fr;
  gap:8px; flex:1; min-height:0;
}
.panel {
  background:white; border:1px solid #ddd; border-radius:6px;
  display:flex; flex-direction:column; overflow:hidden;
}
.panel-hdr {
  padding:7px 12px; border-bottom:1px solid #eee;
  display:flex; align-items:center; gap:8px;
  flex-shrink:0; flex-wrap:wrap; min-height:40px;
}
.panel-hdr h3 { font-size:13px; font-weight:600; }
.panel-body { flex:1; position:relative; min-height:0; }
.viewer3d   { width:100%; height:100%; }
.overlay {
  position:absolute; inset:0;
  background:rgba(255,255,255,.82);
  display:none; align-items:center; justify-content:center;
  font-size:13px; color:#555; text-align:center; padding:20px;
}
.overlay.on { display:flex; }
.navbtn {
  background:#f2f2f2; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:1px 9px; font-size:13px; line-height:1.6;
}
.navbtn:hover:not(:disabled) { background:#e4e4e4; }
.navbtn:disabled { opacity:.35; cursor:default; }
#scr-page {
  position:fixed; top:10px; right:14px; z-index:100;
  background:#0072B2; color:white; border:none; border-radius:5px;
  padding:4px 12px; font-size:12px; font-weight:600; cursor:pointer;
  box-shadow:0 1px 4px rgba(0,0,0,.25);
}
#scr-page:hover { background:#005a8e; }
#scr-page.busy  { opacity:.6; cursor:wait; }
#model-info { font-size:12px; color:#555; white-space:nowrap; }
.pdb-list   { display:flex; gap:5px; flex-wrap:wrap; }
.pdb-btn {
  background:#f2f2f2; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:2px 10px;
  font-size:12px; font-family:monospace; font-weight:600; letter-spacing:.5px;
}
.pdb-btn:hover { background:#e4e4e4; }
.pdb-btn.active { background:#0072B2; color:white; border-color:#0072B2; }
.pdb-btn.cached { border-color:#009E73; }

/* right-panel mode toggle */
.right-mode-tabs { display:flex; gap:3px; align-items:center; flex-shrink:0; }
.mode-tab {
  background:#f0f0f0; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:2px 9px; font-size:12px; font-weight:600; line-height:1.6;
  white-space:nowrap;
}
.mode-tab:hover:not(.active):not(:disabled) { background:#e4e4e4; }
.mode-tab.active  { background:#0072B2; color:white; border-color:#0072B2; }
.mode-tab:disabled { opacity:.38; cursor:not-allowed; }

/* input pair buttons */
.input-list { display:flex; gap:5px; flex-wrap:wrap; }
.input-btn {
  background:#f2f2f2; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:2px 10px;
  font-size:12px; font-family:monospace; font-weight:600;
  max-width:240px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.input-btn:hover { background:#e4e4e4; }
.input-btn.active { background:#009E73; color:white; border-color:#009E73; }

.protein-link { cursor:pointer; text-decoration:underline dotted; }
.protein-link:hover { opacity:.7; }
.evidence-backdrop {
  position:fixed; inset:0; background:rgba(0,0,0,.35); display:none; z-index:50;
}
.evidence-backdrop.on { display:block; }
.evidence-popup {
  position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
  background:white; border-radius:8px; box-shadow:0 4px 24px rgba(0,0,0,.25);
  width:min(640px,90vw); max-height:80vh; overflow:auto; z-index:51; display:none;
}
.evidence-popup.on { display:block; }
.evidence-popup-hdr {
  display:flex; justify-content:space-between; align-items:center;
  padding:10px 14px; border-bottom:1px solid #eee; font-weight:600; font-size:14px;
  position:sticky; top:0; background:white;
}
.evidence-popup-body { padding:12px 14px; font-size:12px; }
.evidence-section { margin-bottom:14px; }
.evidence-section h4 { font-size:12px; margin-bottom:6px; color:#555; }
.evidence-table { width:100%; border-collapse:collapse; font-family:monospace; font-size:11px; }
.evidence-table th, .evidence-table td { text-align:left; padding:3px 6px; border-bottom:1px solid #f0f0f0; }
.evidence-empty { color:#999; font-style:italic; }
</style>
</head>
<body>

<div class="topbar">
  <label>Complex:</label>
  <select id="complex-select"></select>
  <span class="badge badge-cf"    id="badge-cf"></span>
  <span class="badge badge-match" id="badge-match"></span>
  <span id="stoic-label" title=""></span>
  <div class="color-ctrl">
    Color:
    <select id="color-mode">
      <option value="single"    >Single (yellow)</option>
      <option value="by-chain"  >By chain</option>
      <option value="by-mapping">By mapping</option>
      <option value="plddt"     >By pLDDT</option>
    </select>
  </div>
  <div class="plddt-legend" id="plddt-legend" style="display:none">
    <span class="sw" style="background:#0053D6"></span>&gt;90
    <span class="sw" style="background:#65CBF3"></span>70-90
    <span class="sw" style="background:#FFDB13"></span>50-70
    <span class="sw" style="background:#FF7D45"></span>&lt;50
  </div>
  <div class="plddt-ctrl">
    <label class="plddt-mode-label">
      <input type="radio" name="plddt-mode" value="threshold" id="plddt-mode-threshold" checked>
      pLDDT &ge;
    </label>
    <input type="range" id="plddt-threshold" min="0" max="100" step="1" value="0">
    <input type="number" id="plddt-threshold-num" min="0" max="100" step="1" value="0">
    <label class="plddt-mode-label">
      <input type="radio" name="plddt-mode" value="top50" id="plddt-mode-top50">
      Top 50% per chain
    </label>
    <span style="color:#999">(AF models only)</span>
  </div>
</div>

<div class="panels">

  <div class="panel">
    <div class="panel-hdr">
      <h3>CombFold Assembly</h3>
      <button class="navbtn" id="prev-m">&#9664;</button>
      <span id="model-info">-</span>
      <button class="navbtn" id="next-m">&#9654;</button>
      <span id="cf-mapped-summary" class="mapped-summary"></span>
      <span id="cf-annotation" class="cp-annotation"></span>
    </div>
    <div class="panel-body">
      <div id="cf-viewer"  class="viewer3d"></div>
      <div id="cf-overlay" class="overlay">Loading...</div>
      <span id="cf-path" class="path-label" title=""></span>
    </div>
  </div>

  <div class="panel">
    <div class="panel-hdr">
      <div class="right-mode-tabs">
        <button class="mode-tab active" id="tab-pdb"
                onclick="setRightMode('pdb')">PDB Ref</button>
        <button class="mode-tab" id="tab-input"
                onclick="setRightMode('input')">Input Pairs</button>
      </div>
      <div class="pdb-list"   id="pdb-list"></div>
      <div class="input-list" id="input-list" style="display:none"></div>
      <span id="pdb-cap-warning" class="cap-warning"></span>
      <span id="pdb-mapped-summary" class="mapped-summary right"></span>
      <span id="pdb-annotation" class="pdb-annotation"></span>
    </div>
    <div class="panel-body">
      <div id="pdb-viewer"  class="viewer3d"></div>
      <div id="pdb-overlay" class="overlay">Loading...</div>
    </div>
  </div>

</div>

<button id="scr-page" title="Screenshot entire page">&#8595; PNG</button>
<div id="evidence-popup-backdrop" class="evidence-backdrop"></div>
<div id="evidence-popup" class="evidence-popup">
  <div class="evidence-popup-hdr">
    <span id="evidence-popup-title"></span>
    <button id="evidence-popup-close" class="navbtn">&times;</button>
  </div>
  <div id="evidence-popup-body" class="evidence-popup-body"></div>
</div>

<script>
const COMPLEXES = %%JSON%%;

/* Okabe-Ito palette */
const CC = ['#0072B2','#E69F00','#009E73','#CC79A7',
            '#56B4E9','#D55E00','#F0E442','#999999'];
const YELLOW = '#E69F00';
const GRAY   = '#bbbbbb';
const PURPLE = '#B39DDB'; /* "identity unknown" -- distinct from both YELLOW
                              (mapped, not covered) and GRAY (mapped, no hit).
                              Never means "confirmed uncovered". */

const LABEL_STYLE = {
  backgroundColor: 'black', backgroundOpacity: 0.75,
  fontColor: 'white', fontSize: 12, padding: 4, inFront: true,
};

/* ── gzip decompress ──────────────────────────────────────────────────── */
async function ungzip(b64) {
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const ds  = new DecompressionStream('gzip');
  const w   = ds.writable.getWriter();
  w.write(raw); w.close();
  const chunks = [];
  const r = ds.readable.getReader();
  for (;;) { const {done,value} = await r.read(); if (done) break; chunks.push(value); }
  let len=0, off=0;
  chunks.forEach(c => len+=c.length);
  const buf = new Uint8Array(len);
  chunks.forEach(c => { buf.set(c,off); off+=c.length; });
  return new TextDecoder().decode(buf);
}

/* ── state ────────────────────────────────────────────────────────────── */
let cfV=null, pdbV=null, cur=null, mIdx=0;
let colorMode      = 'single';
let plddtThreshold = 0;          // hide atoms with pLDDT (B-factor) below this, AF models only
let plddtMode      = 'threshold'; // 'threshold' (fixed value, slider or typed) | 'top50' (per-chain median split)
let cfRaw        = null, pdbRaw = null, pdbFmtCur = 'pdb';
let currentPdbId = null;

/* ── pLDDT (AlphaFold DB standard 4-bin palette) ─────────────────────────
   Applies to B-factor columns of AF3 models (CombFold assembly, input
   pairwise predictions) where B-factor was overloaded to store pLDDT.
   NOT meaningful for real crystallographic reference PDBs -- never
   applied there. */
function plddtColor(atom) {
  const b = atom.b;
  if (b == null)  return GRAY;
  if (b > 90)     return '#0053D6';
  if (b > 70)     return '#65CBF3';
  if (b > 50)     return '#FFDB13';
  return '#FF7D45';
}

/* Hide (clear style on) atoms whose B-factor/pLDDT falls below the active
   cutoff, on top of whatever cartoon style/coloring was just applied.
   Must be called AFTER setStyle for the visible atoms, since setStyle
   with an empty style object here overrides them for the hidden subset.
   Dispatches on the global plddtMode -- see applyThresholdFilter (fixed
   cutoff, from slider or typed number) and applyTop50Filter (per-chain
   median split) below. */
function applyPlddtFilter(v) {
  if (plddtMode === 'top50') applyTop50Filter(v);
  else                       applyThresholdFilter(v, plddtThreshold);
}

/* Fixed cutoff: hide any atom with pLDDT < threshold. No-op at 0. */
function applyThresholdFilter(v, threshold) {
  if (!threshold) return;
  const below = v.selectedAtoms({}).filter(a => a.b != null && a.b < threshold);
  if (below.length) {
    v.setStyle({serial: below.map(a => a.serial)}, {});
  }
}

/* Per-chain "top 50%": for each chain independently, compute the median
   pLDDT (B-factor) across that chain's atoms and hide whichever atoms
   fall below it -- i.e. keep only the upper half of each chain's own
   pLDDT distribution. Computed per chain (not globally) so a uniformly
   confident chain doesn't get needlessly thinned out just because
   another chain in the same model is worse, and vice versa. */
function applyTop50Filter(v) {
  const atoms = v.selectedAtoms({}).filter(a => a.b != null);
  if (!atoms.length) return;

  const byChain = {};
  atoms.forEach(a => (byChain[a.chain] ||= []).push(a.b));

  const chainMedian = {};
  for (const [ch, vals] of Object.entries(byChain)) {
    vals.sort((x, y) => x - y);
    const mid = vals.length >> 1;
    chainMedian[ch] = (vals.length % 2)
      ? vals[mid]
      : (vals[mid - 1] + vals[mid]) / 2;
  }

  const below = atoms.filter(a => a.b < chainMedian[a.chain]);
  if (below.length) {
    v.setStyle({serial: below.map(a => a.serial)}, {});
  }
}

/* right-panel input-pair mode */
let rightMode       = 'pdb';   // 'pdb' | 'input'
let currentInputIdx = -1;
let inputRaw        = null;
let inputChainMap   = {};       // chain letter → UniProt for displayed input model

/* pdbCache: { pid: {text, fmt, title} } -- holds all reference PDBs for the
   CURRENTLY SELECTED complex only. pdbCacheOwner tags which complex_ac
   the cache belongs to; selectComplex() resets both on every switch, and
   any in-flight fetch checks pdbCacheOwner before writing to the cache so
   a stale background fetch from a since-abandoned complex can't pollute
   the new one's cache. 'title' is the RCSB entry title, fetched alongside
   the structure and used as a fallback when no Complex Portal annotation
   exists for that PDB. */
let pdbCache      = {};
let pdbCacheOwner = null;

/* ── right-panel mode: PDB Ref ↔ Input Pairs ─────────────────────────── */
function setRightMode(mode) {
  rightMode = mode;
  document.getElementById('tab-pdb').classList.toggle('active',   mode === 'pdb');
  document.getElementById('tab-input').classList.toggle('active',  mode === 'input');
  document.getElementById('pdb-list').style.display    = mode === 'pdb'   ? '' : 'none';
  document.getElementById('input-list').style.display  = mode === 'input' ? '' : 'none';
  document.getElementById('pdb-cap-warning').style.display = mode === 'pdb' ? '' : 'none';
  document.getElementById('pdb-mapped-summary').innerHTML = '';

  if (mode === 'pdb') {
    // restore CF coloring to the current color-mode
    if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
    // restore current reference PDB (use cache if available)
    if (currentPdbId && pdbCache[currentPdbId]) {
      const {text, fmt} = pdbCache[currentPdbId];
      styleViewer(pdbV, text, fmt, 'pdb');
    } else if (currentPdbId) {
      loadPDB(currentPdbId);
    }
    updateAnnotations();
    updateMappedSummaries();
  } else {
    const models = cur?.input_models || [];
    buildInputList(models);
    if (models.length > 0) {
      loadInputModel(0);
    } else {
      pdbV.removeAllModels(); pdbV.render();
      spin('pdb-overlay', true, 'No input models embedded for this complex.');
      setTimeout(() => spin('pdb-overlay', false), 4000);
    }
    updateAnnotations();
  }
}

function buildInputList(models) {
  const list = document.getElementById('input-list');
  list.innerHTML = '';
  models.forEach((m, i) => {
    const b = document.createElement('button');
    b.className   = 'input-btn';
    b.textContent = m.label;
    b.title       = m.filename;
    b.onclick     = () => loadInputModel(i);
    list.appendChild(b);
  });
}

async function loadInputModel(idx) {
  const models = cur?.input_models || [];
  if (!cur || idx < 0 || idx >= models.length) return;
  currentInputIdx = idx;

  document.querySelectorAll('.input-btn')
    .forEach((b, i) => b.classList.toggle('active', i === idx));

  const m = models[idx];
  spin('pdb-overlay', true, 'Loading input pair...');
  try {
    const text = await ungzip(m.pdb_gz);
    inputRaw   = text;

    /* chain-letter → UniProt: chains in appearance order → proteins[0], proteins[1] */
    const chains = [...new Set(
      text.split('\n')
        .filter(l => l.startsWith('ATOM') && l.length > 21)
        .map(l => l[21])
    )];
    inputChainMap = {};
    chains.forEach((ch, i) => {
      inputChainMap[ch] = m.proteins[Math.min(i, m.proteins.length - 1)];
    });

    styleInputViewer(text, m.proteins);
    spin('pdb-overlay', false);
    if (cfRaw) highlightCFPair(m.proteins);
    updateAnnotations();
    updateMappedSummaries();
  } catch (e) {
    console.error(e);
    spin('pdb-overlay', true, 'Failed to load: ' + e.message);
    setTimeout(() => spin('pdb-overlay', false), 3500);
  }
}

/* Color the input-pair viewer: chain A → blue, chain B → green (both blue
   for homodimers).  These colors are intentionally never yellow so the
   pair is always distinguishable from the rest of the CF assembly. */
function styleInputViewer(text, proteins) {
  pdbV.removeAllModels();
  const model  = pdbV.addModel(text, 'pdb');
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();
  const homo   = proteins[0] === proteins[1];

  chains.forEach((ch, i) => {
    if (colorMode === 'plddt') {
      pdbV.setStyle({chain: ch}, {cartoon: {colorfunc: plddtColor}});
      return;
    }
    const color = homo ? CC[0] : (i === 0 ? CC[0] : CC[2]); // blue / green
    pdbV.setStyle({chain: ch}, {cartoon: {color}});
  });

  /* Input pairs are AF3 predictions too -- B-factor is pLDDT, filter applies. */
  applyPlddtFilter(pdbV);

  pdbV.setHoverable({}, true,
    (atom, v) => {
      const u = inputChainMap[atom.chain];
      const plddt = atom.b != null ? '  |  pLDDT ' + atom.b.toFixed(1) : '';
      v.removeAllLabels();
      v.addLabel(
        (u ? 'Chain ' + atom.chain + ': ' + u : 'Chain ' + atom.chain) + plddt,
        {...LABEL_STYLE, position: atom}
      );
      v.render();
    },
    (atom, v) => { v.removeAllLabels(); v.render(); }
  );

  pdbV.zoomTo(); pdbV.render();
}

/* Highlight CF assembly chains that belong to the selected input pair,
   using the SAME colors as the right-panel input viewer so chains can be
   matched by color across both panels:
     proteins[0] chains → blue  (CC[0])
     proteins[1] chains → green (CC[2])   (or blue too if homodimer)
     all other chains   → yellow (YELLOW) — fully visible, just distinguished */
function highlightCFPair(proteins) {
  if (!cfRaw) return;
  const homo = proteins[0] === proteins[1];

  cfV.removeAllModels();
  const model  = cfV.addModel(cfRaw, 'pdb');
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();

  chains.forEach(ch => {
    const u = cur?.cf_chain_map?.[ch];
    let color;
    if (homo && u === proteins[0]) {
      color = CC[0];    // homodimer: all copies blue
    } else if (u === proteins[0]) {
      color = CC[0];    // blue — matches right-panel chain A
    } else if (u === proteins[1]) {
      color = CC[2];    // green — matches right-panel chain B
    } else {
      color = YELLOW;   // rest of complex: visible but clearly distinct
    }
    cfV.setStyle({chain: ch}, {cartoon: {color}});
  });

  applyPlddtFilter(cfV);

  cfV.setHoverable({}, true, cfHoverCB, cfUnhoverCB);
  cfV.zoomTo(); cfV.render();
  updateMappedSummaries();
}

/* ── viewers + hover ──────────────────────────────────────────────────── */
/* Hover callbacks are named + module-level so styleViewer() can re-register
   them on every new model. setHoverable() only flags the atoms that exist
   in the viewer AT THE MOMENT IT IS CALLED -- it does nothing for atoms
   added afterwards. Previously this was called once here in initViewers(),
   before any model had been loaded (getAtomsFromSel({}) matched zero
   atoms), so no atom ever got hoverable=true and hovering silently did
   nothing. Fix: call setHoverable again after every addModel() (see
   styleViewer below). */
function cfHoverCB(atom, viewer) {
  const u = cur?.cf_chain_map?.[atom.chain];
  const plddt = atom.b != null ? '  |  pLDDT ' + atom.b.toFixed(1) : '';
  viewer.removeAllLabels();
  viewer.addLabel(
    (u ? 'Chain ' + atom.chain + ': ' + u : 'Chain ' + atom.chain + ' (unmapped)') + plddt,
    {...LABEL_STYLE, position: atom}
  );
  viewer.render();
}
function cfUnhoverCB(atom, viewer) { viewer.removeAllLabels(); viewer.render(); }

function pdbHoverCB(atom, viewer) {
  const info     = cur?.pdb_chain_map?.[currentPdbId]?.[atom.chain];
  const homology = cur?.pdb_homology_map?.[currentPdbId]?.[atom.chain];
  let text = 'Chain ' + atom.chain;
  if (info) text += ' | SIFTS: ' + info.u + (info.cp ? ' (direct match)' : ' (not in complex)');
  if (homology && homology.length > 0) text += ' | homology: ' + homology.join(', ');
  if (!info && (!homology || homology.length === 0)) text += ' (no mapping)';
  /* This is a crystallographic reference structure, so B-factor here is a
     real B-factor, not pLDDT -- labeled accordingly to avoid confusion. */
  if (atom.b != null) text += '  |  B-factor ' + atom.b.toFixed(1);
  viewer.removeAllLabels();
  viewer.addLabel(text, {...LABEL_STYLE, position: atom});
  viewer.render();
}
function pdbUnhoverCB(atom, viewer) { viewer.removeAllLabels(); viewer.render(); }

function initViewers() {
  const opts = {backgroundColor:'white', antialias:true};
  cfV  = $3Dmol.createViewer(document.getElementById('cf-viewer'),  opts);
  pdbV = $3Dmol.createViewer(document.getElementById('pdb-viewer'), opts);
}

/* ── coloring ─────────────────────────────────────────────────────────── */
function styleViewer(v, text, fmt, storeAs) {
  if (storeAs === 'cf')  cfRaw = text;
  if (storeAs === 'pdb') { pdbRaw = text; pdbFmtCur = fmt; }

  v.removeAllModels();
  const model      = v.addModel(text, fmt);
  const chains     = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();
  const cpProteins = cur?.cp_proteins || [];

  /* re-register hoverable on THIS model's atoms -- see note in initViewers() */
  if (storeAs === 'cf') v.setHoverable({}, true, cfHoverCB, cfUnhoverCB);
  else                  v.setHoverable({}, true, pdbHoverCB, pdbUnhoverCB);

  chains.forEach((ch, i) => {
    let color;
    if (colorMode === 'plddt') {
      v.setStyle({chain:ch}, {cartoon:{colorfunc: plddtColor}});
      return;
    } else if (colorMode === 'by-chain') {
      color = CC[i % CC.length];
    } else if (colorMode === 'single') {
      color = YELLOW;
    } else {
      /* by-mapping:
         CF chain  = blue if its UniProt (via cf_chain_map) is among the
                      currently shown PDB's proteins, yellow if it's known
                      but not covered, PURPLE if the chain has no identity
                      at all (cf_chain_map has no entry for it -- an
                      "unknown", never treated as "confirmed uncovered").
         PDB chain = blue if EITHER a direct SIFTS match (chain's own
                      UniProt is a CP protein) OR a cross-species homology
                      hit says so; gray otherwise. Both are independent
                      pieces of positive evidence and must be combined,
                      not just the homology one. */
      if (storeAs === 'cf') {
        const uniprot = cur?.cf_chain_map?.[ch];
        if (!uniprot) {
          color = PURPLE;
        } else {
          const pdbProts = cur?.pdb_proteins?.[currentPdbId] || [];
          color = pdbProts.includes(uniprot) ? '#0072B2' : YELLOW;
        }
      } else {
        const chainInfo   = cur?.pdb_chain_map?.[currentPdbId]?.[ch];
        const homologyHit = cur?.pdb_homology_map?.[currentPdbId]?.[ch];
        const mapped = (chainInfo && chainInfo.cp) || (homologyHit && homologyHit.length > 0);
        color = mapped ? '#0072B2' : GRAY;
      }
    }
    v.setStyle({chain:ch}, {cartoon:{color}});
  });

  /* pLDDT threshold filter only makes sense for AF3-derived models; the
     CF assembly panel always is one, the reference PDB panel (crystal
     structures, real B-factors) never is. */
  if (storeAs === 'cf') applyPlddtFilter(v);

  v.zoomTo(); v.render();
  updateMappedSummaries();
}

/* ── protein click-to-inspect popup ──────────────────────────────────────
   renderGroup builds the Mapped/Novel/No-hit summary lines with each
   protein as a clickable span; clicking one opens a popup showing the
   raw SIFTS + MMseq evidence rows for that protein against the currently
   selected reference PDB. */
function renderGroup(el, groups) {
  el.innerHTML = '';
  let wroteAny = false;
  groups.forEach(([cls, label, list]) => {
    if (!list.length) return;
    if (wroteAny) el.appendChild(document.createTextNode('\u00A0\u00A0|\u00A0\u00A0'));
    const labelSpan = document.createElement('span');
    labelSpan.className = cls;
    labelSpan.textContent = label + ': ';
    el.appendChild(labelSpan);
    list.forEach((protein, i) => {
      if (i > 0) el.appendChild(document.createTextNode(', '));
      const a = document.createElement('span');
      a.className = cls + ' protein-link';
      a.textContent = protein;
      a.dataset.protein = protein;
      el.appendChild(a);
    });
    wroteAny = true;
  });
  if (!wroteAny) el.textContent = 'No proteins';
}

function buildEvidenceSection(title, rows, preferredCols) {
  const wrap = document.createElement('div');
  wrap.className = 'evidence-section';
  const h = document.createElement('h4');
  h.textContent = title + ' (' + rows.length + ')';
  wrap.appendChild(h);
  if (!rows.length) {
    const p = document.createElement('div');
    p.className = 'evidence-empty';
    p.textContent = 'No rows for this protein / PDB.';
    wrap.appendChild(p);
    return wrap;
  }
  /* preferred columns first (if present), then anything else the row
     actually has -- so real data is never silently hidden just because
     we didn't anticipate the column name */
  const allCols = new Set();
  rows.forEach(r => Object.keys(r).forEach(k => allCols.add(k)));
  const cols = [
    ...preferredCols.filter(c => allCols.has(c)),
    ...[...allCols].filter(c => !preferredCols.includes(c)),
  ];
  const table = document.createElement('table');
  table.className = 'evidence-table';
  const thead = document.createElement('tr');
  cols.forEach(c => { const th = document.createElement('th'); th.textContent = c; thead.appendChild(th); });
  table.appendChild(thead);
  rows.forEach(r => {
    const tr = document.createElement('tr');
    cols.forEach(c => {
      const td = document.createElement('td');
      const v = r[c];
      td.textContent = (v === null || v === undefined) ? '' : String(v);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  wrap.appendChild(table);
  return wrap;
}

function showEvidencePopup(protein) {
  const ev = cur?.pdb_protein_evidence?.[currentPdbId]?.[protein];
  document.getElementById('evidence-popup-title').textContent =
    protein + '  vs  ' + (currentPdbId ? currentPdbId.toUpperCase() : '?');
  const body = document.getElementById('evidence-popup-body');
  body.innerHTML = '';
  body.appendChild(buildEvidenceSection('Direct SIFTS match', ev?.direct || [],
    ['CHAIN', 'SP_PRIMARY']));
  body.appendChild(buildEvidenceSection('MMseq homology hits', ev?.homology || [],
    ['hit_pdb_id', 'identity_percent', 'blast_identity_percent', 'alnlen']));
  document.getElementById('evidence-popup-backdrop').classList.add('on');
  document.getElementById('evidence-popup').classList.add('on');
}

function hideEvidencePopup() {
  document.getElementById('evidence-popup-backdrop').classList.remove('on');
  document.getElementById('evidence-popup').classList.remove('on');
}

document.addEventListener('click', (e) => {
  const t = e.target;
  if (t.classList?.contains('protein-link')) {
    showEvidencePopup(t.dataset.protein);
  } else if (t.id === 'evidence-popup-close' || t.id === 'evidence-popup-backdrop') {
    hideEvidencePopup();
  }
});

/* ── mapped-proteins summary (top-left CF / top-right PDB) ──────────────
   Only populated in "by mapping" color mode; cleared otherwise. */
function updateMappedSummaries() {
  const cfEl  = document.getElementById('cf-mapped-summary');
  const pdbEl = document.getElementById('pdb-mapped-summary');
  cfEl.innerHTML = ''; pdbEl.innerHTML = '';

  if (rightMode === 'input') {
    /* In input mode the CF panel is in "pair highlight" mode; show which
       proteins are highlighted rather than the full by-mapping breakdown. */
    if (currentInputIdx >= 0) {
      const m = (cur?.input_models || [])[currentInputIdx];
      if (m) {
        const unique = [...new Set(m.proteins)];
        cfEl.innerHTML =
          '<span class="m-blue">Highlighted: ' + unique.join(' \u2013 ') + '</span>';
      }
    }
    return;
  }

  if (!cur || colorMode !== 'by-mapping') return;

  const pdbProts      = cur.pdb_proteins?.[currentPdbId] || [];
  const chainEntries  = Object.entries(cur.cf_chain_map || {});
  const knownProteins = [...new Set(chainEntries.map(([, u]) => u))];
  const mappedProteins = knownProteins.filter(p => pdbProts.includes(p)).sort();
  const novelProteins  = knownProteins.filter(p => !pdbProts.includes(p)).sort();

  const allCfChains    = cur.cf_all_chains || Object.keys(cur.cf_chain_map || {});
  const unmappedChains = allCfChains.filter(ch => !(ch in (cur.cf_chain_map || {}))).sort();

  renderGroup(cfEl, [
    ['m-blue',   'Mapped',   mappedProteins],
    ['m-off',    'Novel',    novelProteins],
    ['m-purple', 'Unknown chains', unmappedChains],
  ]);

  const chainMap = cur.pdb_chain_map?.[currentPdbId] || {};
  const homMap   = cur.pdb_homology_map?.[currentPdbId] || {};
  const homSet = new Set();
  Object.values(chainMap).forEach(info => { if (info.cp) homSet.add(info.u); });
  Object.values(homMap).forEach(arr => arr.forEach(p => homSet.add(p)));
  const cpProteins = cur.cp_proteins || [];
  renderGroup(pdbEl, [
    ['m-blue', 'Mapped', cpProteins.filter(p => homSet.has(p))],
    ['m-off',  'No hit', cpProteins.filter(p => !homSet.has(p))],
  ]);
}

/* ── annotations (Complex Portal, with RCSB-title fallback) ──────────────
   CF panel: Complex Portal name for the whole complex, constant across
   PDB switches.
   PDB panel: Complex Portal per-PDB annotation if present; otherwise the
   RCSB entry title fetched alongside the structure (labeled as such, since
   it's a different source with different curation than Complex Portal). */
function updateAnnotations() {
  const cfEl  = document.getElementById('cf-annotation');
  const pdbEl = document.getElementById('pdb-annotation');
  cfEl.textContent = cur?.cp_annotation ? cur.cp_annotation : '(no Complex Portal annotation)';

  if (rightMode === 'input') {
    const m = (cur?.input_models || [])[currentInputIdx];
    pdbEl.textContent = m
      ? 'Input pair: ' + m.label + '  [' + m.filename + ']'
      : '(no input pair selected)';
    return;
  }

  const cpAnnot   = cur?.pdb_annotation?.[currentPdbId];
  const rcsbTitle = pdbCache[currentPdbId]?.title;
  if (cpAnnot) {
    pdbEl.textContent = cpAnnot;
  } else if (rcsbTitle) {
    pdbEl.textContent = rcsbTitle;
  } else {
    pdbEl.textContent = '(no annotation available)';
  }
}

/* ── overlay helper ───────────────────────────────────────────────────── */
function spin(id, on, msg) {
  const el = document.getElementById(id);
  el.classList.toggle('on', on);
  if (msg !== undefined) el.textContent = msg;
}

/* ── load CF model ────────────────────────────────────────────────────── */
async function loadCF(idx) {
  if (!cur) return;
  const models = cur.cf_models;
  if (!models || models.length === 0) {
    cfV.removeAllModels(); cfV.render();
    document.getElementById('model-info').textContent = 'No models';
    document.getElementById('prev-m').disabled = true;
    document.getElementById('next-m').disabled = true;
    return;
  }
  idx = Math.max(0, Math.min(idx, models.length-1));
  mIdx = idx;
  spin('cf-overlay', true, 'Loading...');
  try {
    const pdb = await ungzip(models[idx].pdb_gz);
    cur.cf_all_chains = [...new Set(
      pdb.split('\n')
        .filter(l => l.startsWith('ATOM') && l.length > 21)
        .map(l => l[21])
    )];
    styleViewer(cfV, pdb, 'pdb', 'cf');
    document.getElementById('model-info').textContent =
      (idx+1) + ' / ' + models.length + '  CF ' + models[idx].score.toFixed(1);
  } catch(e) {
    console.error(e);
    document.getElementById('model-info').textContent = 'Load error';
  } finally { spin('cf-overlay', false); }
  document.getElementById('prev-m').disabled = idx <= 0;
  document.getElementById('next-m').disabled = idx >= models.length-1;
}

/* ── RCSB fetches: structure + entry title (used as annotation fallback) ─ */
async function fetchPDBTitle(pid) {
  try {
    const r = await fetch('https://data.rcsb.org/rest/v1/core/entry/' + pid.toUpperCase());
    if (!r.ok) return null;
    const json = await r.json();
    return json?.struct?.title || null;
  } catch (e) {
    console.error('RCSB title fetch failed for', pid, e);
    return null;
  }
}

async function fetchPDBText(pid) {
  const structPromise = (async () => {
    const r1 = await fetch('https://files.rcsb.org/download/' + pid.toUpperCase() + '.pdb');
    if (r1.ok) return { text: await r1.text(), fmt: 'pdb' };
    const r2 = await fetch('https://files.rcsb.org/download/' + pid.toUpperCase() + '.cif');
    if (!r2.ok) throw new Error('RCSB 404 for ' + pid + ' (.pdb and .cif)');
    return { text: await r2.text(), fmt: 'mmcif' };
  })();
  const [struct, title] = await Promise.all([structPromise, fetchPDBTitle(pid)]);
  return { ...struct, title };
}

/* ── load PDB reference (cache-first) ─────────────────────────────────── */
async function loadPDB(pid) {
  if (!pid) return;
  currentPdbId = pid;
  document.querySelectorAll('.pdb-btn')
    .forEach(b => b.classList.toggle('active', b.dataset.pid === pid));
  updateAnnotations(); // shows whatever's cached so far (instant on cache hit)

  const owner = pdbCacheOwner;

  if (pdbCache[pid]) {
    const { text, fmt } = pdbCache[pid];
    styleViewer(pdbV, text, fmt, 'pdb');
    if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
    return;
  }

  spin('pdb-overlay', true, 'Fetching from RCSB...');
  try {
    const result = await fetchPDBText(pid);
    if (owner === pdbCacheOwner) pdbCache[pid] = result; // still same complex
    if (currentPdbId !== pid) return; // user switched to a different PDB meanwhile
    styleViewer(pdbV, result.text, result.fmt, 'pdb');
    if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
    updateAnnotations(); // title (if any) has now arrived
    spin('pdb-overlay', false);
  } catch(e) {
    console.error(e);
    spin('pdb-overlay', true, 'Could not load ' + pid.toUpperCase() + ' - ' + e.message);
    setTimeout(() => spin('pdb-overlay', false), 3000);
  }
}

/* ── background prefetch of the remaining reference PDBs ─────────────────
   Fetches sequentially (gentle on RCSB) and bails out immediately if the
   user has since switched to a different complex, so a slow prefetch
   never contaminates a new complex's cache. */
async function prefetchOtherPDBs(owner, pids) {
  for (const pid of pids) {
    if (owner !== pdbCacheOwner) return;
    if (pdbCache[pid]) continue;
    try {
      const result = await fetchPDBText(pid);
      if (owner !== pdbCacheOwner) return;
      pdbCache[pid] = result;
      const btn = document.querySelector('.pdb-btn[data-pid="' + pid + '"]');
      if (btn) btn.classList.add('cached');
      if (currentPdbId === pid) updateAnnotations(); // in case title arrived after view
    } catch (e) {
      console.error('prefetch failed for', pid, e);
    }
  }
}

/* ── select complex ───────────────────────────────────────────────────── */
function selectComplex(ac) {
  cur = COMPLEXES[ac];
  if (!cur) return;

  /* reset right-panel mode every time we switch complexes */
  rightMode       = 'pdb';
  currentInputIdx = -1;
  inputRaw        = null;
  inputChainMap   = {};
  document.getElementById('tab-pdb').classList.add('active');
  document.getElementById('tab-input').classList.remove('active');
  document.getElementById('pdb-list').style.display    = '';
  document.getElementById('input-list').style.display  = 'none';
  document.getElementById('pdb-cap-warning').style.display = '';
  /* enable/disable Input Pairs tab based on availability */
  const hasInputs = (cur.input_models || []).length > 0;
  document.getElementById('tab-input').disabled = !hasInputs;
  document.getElementById('tab-input').title    =
    hasInputs ? '' : 'No input models embedded for this complex';

  pdbCache      = {};   // drop previous complex's cache
  pdbCacheOwner = ac;

  document.getElementById('badge-cf').textContent =
    'CF ' + cur.cf_confidence.toFixed(1);
  document.getElementById('badge-match').textContent =
    cur.match_class.replace(/_/g,' ');
  const sl = document.getElementById('stoic-label');
  sl.textContent = cur.stoic_label;
  sl.title = cur.stoic_label;

  const cfModel0 = (cur.cf_models && cur.cf_models[0]) || null;
  const cfPathEl = document.getElementById('cf-path');
  cfPathEl.textContent = cfModel0 ? cfModel0.folder : '(no CF model found)';
  cfPathEl.title       = cfModel0 ? cfModel0.path   : '';

  const capEl = document.getElementById('pdb-cap-warning');
  if (cur.pdb_cap_hit) {
    capEl.textContent = 'capped at ' + (cur.pdb_ids?.length ?? 0) + ' PDBs';
    capEl.classList.add('on');
  } else {
    capEl.classList.remove('on');
  }

  const list = document.getElementById('pdb-list');
  list.innerHTML = '';
  const pids = cur.pdb_ids || [];
  if (pids.length === 0) {
    list.textContent = 'no PDB';
  } else {
    pids.forEach(pid => {
      const b = document.createElement('button');
      b.className   = 'pdb-btn';
      b.dataset.pid = pid;
      b.textContent = pid.toUpperCase();
      b.onclick = () => loadPDB(pid);
      list.appendChild(b);
    });
  }

  mIdx = 0;
  updateAnnotations();
  loadCF(0);
  if (pids.length) {
    loadPDB(pids[0]);
    prefetchOtherPDBs(ac, pids.slice(1));
  } else {
    currentPdbId = null; pdbV.removeAllModels(); pdbV.render();
    updateMappedSummaries(); updateAnnotations();
  }
}

/* ── dropdown ─────────────────────────────────────────────────────────── */
function buildDropdown() {
  const sel = document.getElementById('complex-select');
  Object.values(COMPLEXES).forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.complex_ac;
    const id  = c.identifiers.length > 55
                ? c.identifiers.slice(0,52)+'...'
                : c.identifiers;
    opt.textContent = c.complex_ac + '  -  ' + id;
    sel.appendChild(opt);
  });
  sel.onchange = () => selectComplex(sel.value);
}

/* ── event wiring ─────────────────────────────────────────────────────── */
document.getElementById('prev-m').onclick = () => loadCF(mIdx-1);
document.getElementById('next-m').onclick = () => loadCF(mIdx+1);

document.getElementById('scr-page').onclick = function() {
  const btn = this;
  btn.classList.add('busy');
  btn.textContent = '...';
  html2canvas(document.documentElement, {
    useCORS: true,
    allowTaint: true,
    scale: 3,
    width:        window.innerWidth,
    height:       window.innerHeight,
    windowWidth:  window.innerWidth,
    windowHeight: window.innerHeight,
    ignoreElements: el => el === btn,
  }).then(canvas => {
    const a = document.createElement('a');
    a.href     = canvas.toDataURL('image/png');
    a.download = (cur?.complex_ac || 'viewer') + '_screenshot.png';
    a.click();
  }).finally(() => {
    btn.classList.remove('busy');
    btn.textContent = '\u2193 PNG';
  });
};

document.getElementById('color-mode').onchange = function() {
  colorMode = this.value;
  document.getElementById('plddt-legend').style.display =
    colorMode === 'plddt' ? '' : 'none';
  if (rightMode === 'input') {
    /* In input mode the right panel is showing an input pair (already
       colorable by pLDDT via loadInputModel/styleInputViewer below), and
       the CF viewer is in pair-highlight mode which never follows the
       color dropdown. Re-render just the input pair so a switch to/from
       'plddt' takes effect immediately. */
    if (currentInputIdx >= 0) loadInputModel(currentInputIdx);
    return;
  }
  if (cfRaw)  styleViewer(cfV,  cfRaw,  'pdb',     'cf');
  if (pdbRaw) styleViewer(pdbV, pdbRaw, pdbFmtCur, 'pdb');
};

/* Re-render whatever pLDDT-filterable panel(s) are currently visible,
   after either the cutoff value or the filter mode changes. Reference
   PDB (crystal) panel intentionally NOT refiltered -- see applyPlddtFilter
   note; it never carries pLDDT data in the first place. */
function refilterAfterPlddtChange() {
  if (rightMode === 'input') {
    if (currentInputIdx >= 0) loadInputModel(currentInputIdx); // re-decompress + refilter
    else if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
  } else {
    if (cfRaw) styleViewer(cfV, cfRaw, 'pdb', 'cf');
  }
}

/* Slider and number box are two views onto the same plddtThreshold value
   -- keep them in sync so the user can drag OR type, whichever's handy. */
function setThresholdValue(raw) {
  plddtThreshold = Math.max(0, Math.min(100, Number(raw) || 0));
  document.getElementById('plddt-threshold').value     = plddtThreshold;
  document.getElementById('plddt-threshold-num').value = plddtThreshold;
}

document.getElementById('plddt-threshold').oninput = function() {
  setThresholdValue(this.value);
  refilterAfterPlddtChange();
};
document.getElementById('plddt-threshold-num').oninput = function() {
  setThresholdValue(this.value);
  refilterAfterPlddtChange();
};

/* Fixed-threshold vs. top-50%-per-chain mode toggle. The slider/number
   box are only meaningful in 'threshold' mode, so disable (not hide --
   keeps the layout stable) whichever control isn't active. */
document.querySelectorAll('input[name="plddt-mode"]').forEach(radio => {
  radio.onchange = function() {
    plddtMode = this.value;
    const isThreshold = plddtMode === 'threshold';
    document.getElementById('plddt-threshold').disabled     = !isThreshold;
    document.getElementById('plddt-threshold-num').disabled = !isThreshold;
    refilterAfterPlddtChange();
  };
});

/* ── init ─────────────────────────────────────────────────────────────── */
initViewers();
buildDropdown();
const first = Object.keys(COMPLEXES)[0];
if (first) {
  document.getElementById('complex-select').value = first;
  selectComplex(first);
}
</script>
</body>
</html>
"""

# ── write output ──────────────────────────────────────────────────────────────

json_blob = json.dumps(EMBED, ensure_ascii=True)
html_out  = HTML.replace("%%JSON%%", json_blob)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html_out, encoding="utf-8")

size_kb = OUT.stat().st_size / 1024
print(f"\n  Written: {OUT}")
print(f"   Size:    {size_kb:.0f} KB  (~{size_kb/1024:.1f} MB)")