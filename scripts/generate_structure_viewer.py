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
from math import ceil
from pathlib import Path

import polars as pl
from procompa import get_project_root

PRJ_ROOT = get_project_root()
DATA     = PRJ_ROOT / "data"
CF_BASE  = DATA / "Pipeline/10_all_CP_complexes/CombFold"
OUT      = DATA / "CP_complexes_no_struct_coverage/structure_viewer.html"
MMSEQ    = DATA / "CP_complexes_no_struct_coverage/sanity_checks/mmseq_no_Strcut_filtered.parquet"
ANNOT    = DATA / "CP_complexes_no_struct_coverage/complex_pdb_annotations_map.csv"

MAX_PDB_REFS = 20  # safeguard cap on reference PDBs shown per complex
MMSEQ_RAW = PRJ_ROOT / "scripts/mmseq_homology_match/mmseqs/mmseqs_run_max_sensitivity/results/mmseqs_new_identity_similarity_max_sensitivity.parquet"


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


def greedy_set_cover(candidates: list, target: frozenset, min_frac: float = 1.0) -> list[str]:
    n_req = ceil(min_frac * len(target)) if target else 0
    covered, selected, pool = frozenset(), [], list(candidates)
    while len(covered) < n_req and pool:
        i = max(range(len(pool)), key=lambda j: len(pool[j][1] - covered))
        pdb, prots = pool.pop(i)
        if not (new := prots - covered):
            break
        covered |= new
        selected.append(pdb)
    return selected


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


# ── process each complex ──────────────────────────────────────────────────────

EMBED: dict[str, dict] = {}

for row in complexes_df.iter_rows(named=True):
    ac          = row["complex_ac"]
    identifiers = row["identifiers"]
    cf_max      = float(row["CF_confidence_max"])
    match_class = row["match_class"] or "unknown"
    target      = uniprot_proteins(identifiers)

    print(f"\n{ac}  CF={cf_max:.1f}  {match_class}  ({len(target)} proteins)")

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
                    print(f"  WARNING read {m['path']}: {e}", file=sys.stderr)
        return candidate

    # Pass 1: protein-set match (exact or folder subset of target)
    for fname, (assembled, models) in all_folders.items():
        fp    = folder_proteins(fname)
        delta = abs(models[0]["score"] - cf_max)
        if (fp == target or (fp and fp <= target)) and delta < best_delta:
            candidate = read_models(fname, models)
            if candidate:
                best_delta, best_label, best_models = delta, folder_stoic_label(fname), candidate

    # Pass 2: score-only fallback (large complexes with truncated folder names)
    if not best_models:
        print(f"  no protein-set match -- trying score-only fallback")
        for fname, (assembled, models) in all_folders.items():
            delta = abs(models[0]["score"] - cf_max)
            if delta < best_delta:
                candidate = read_models(fname, models)
                if candidate:
                    best_delta  = delta
                    best_label  = folder_stoic_label(fname) + "  (score-match only)"
                    best_models = candidate

    if best_models:
        print(f"  CF models: {len(best_models)}  delta={best_delta:.3f}  [{best_label[:60]}]")
    else:
        print(f"  WARNING: no CF models found", file=sys.stderr)

    # PDB set cover -- IDs only, browser fetches on demand
    hits = pdb_hits.filter(pl.col("complex_ac") == ac)
    candidates = []
    pdb_proteins: dict[str, list[str]] = {}
    for h in hits.iter_rows(named=True):
        if h["pdb_id"] and h["proteins"]:
            candidates.append((h["pdb_id"], frozenset(h["proteins"])))
            pdb_proteins[h["pdb_id"]] = list(h["proteins"])

    cover = greedy_set_cover(candidates, frozenset(target), min_frac=1.0)
    if not cover:
        cover = greedy_set_cover(candidates, frozenset(target), min_frac=0.75)

    pdb_cap_hit = len(cover) > MAX_PDB_REFS
    if pdb_cap_hit:
        print(f"  WARNING {ac}: PDB cover has {len(cover)} entries, "
              f"capping at {MAX_PDB_REFS}", file=sys.stderr)
    cover = cover[:MAX_PDB_REFS]
    print(f"  PDB cover: {cover}")

    cf_chain_map    = best_models[0].get("chain_map", {}) if best_models else {}
    if best_models:
        assert cf_chain_map, (
            f"CF chain map is empty for {ac} — check folder name parsing "
            f"(folder: {best_label})"
        )
    cf_models_clean = [{"name": m["name"], "score": m["score"], "pdb_gz": m["pdb_gz"],
                        "folder": m["folder"], "path": m["path"]}
                       for m in best_models]

    EMBED[ac] = {
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
    }

    # Verify that PDB chain maps are actually populated from SIFTS
    if cover:
        mapped = [pid for pid in cover if EMBED[ac]["pdb_chain_map"].get(pid)]
        assert mapped, (
            f"No PDB chain maps populated for {ac} (PDB IDs: {cover}). "
            f"Check that SIFTS covers these PDB entries."
        )
    if EMBED[ac]["cp_annotation"] is None:
        print(f"  NOTE: no Complex Portal annotation found for {ac}", file=sys.stderr)


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
    </select>
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
      <h3>PDB Reference</h3>
      <div class="pdb-list" id="pdb-list"></div>
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
let colorMode    = 'single';
let cfRaw        = null, pdbRaw = null, pdbFmtCur = 'pdb';
let currentPdbId = null;

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
  viewer.removeAllLabels();
  viewer.addLabel(
    u ? 'Chain ' + atom.chain + ': ' + u : 'Chain ' + atom.chain + ' (unmapped)',
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
    if (colorMode === 'by-chain') {
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
    scale: window.devicePixelRatio || 1,
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
  if (cfRaw)  styleViewer(cfV,  cfRaw,  'pdb',     'cf');
  if (pdbRaw) styleViewer(pdbV, pdbRaw, pdbFmtCur, 'pdb');
};

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