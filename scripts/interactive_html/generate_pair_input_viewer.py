#!/usr/bin/env python3
"""
Generate self-contained HTML viewer for CombFold assemblies with their
input AF3 pairwise structures shown side-by-side.

For each complex (best CF output by CF_confidence, n_proteins > 2):
  Left  panel : CombFold assembled model — chain hover shows UniProt ID
  Right panel : Input AF3 pair PDB for the selected foldstep pair,
                blue (chain A = prot 1) / green (chain B = prot 2)
  Pair  bar   : click any foldstep-pair button to load its input PDB
                and highlight the corresponding chains in the CF assembly;
                dashed border = input PDB not found on disk

Usage:
    uv run python scripts/generate_pair_input_viewer.py

Output:
    data/Pipeline/10_all_CP_complexes/pair_input_viewer/pair_input_viewer.html
"""

import base64, gzip, json, re, sys
from pathlib import Path

import polars as pl
from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"

BENCH_PARQ  = data_dir / "Pipeline/10_all_CP_complexes/cf_pdb_structure_similarity/cf_pdb_eval_all_metrics_benchmark_bigger_complexes.parquet"
METRICS_CSV = data_dir / "Pipeline/10_all_CP_complexes/10_all_CP_complexes_pool_pairs_metrics_with_plddt_corrected.csv"
# Fallback pool_input search: pipeline-13 used plddt_70 threshold but has the same AF3 inputs
PIPELINE_13 = data_dir / "Pipeline/13_all_CP_complexes_plddt_70_threshold"
OUT         = data_dir / "Pipeline/viewerpair_utilized_input_viewer.html"


# ── helpers ───────────────────────────────────────────────────────────────────

def compress(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode())).decode()


def strip_pdb(text: str) -> str:
    _KEEP = frozenset({"ATOM", "HETATM", "MODEL", "ENDMDL", "TER", "END"})
    return "\n".join(l for l in text.splitlines() if l[:6].strip() in _KEEP)


def nan_to_none(v):
    return None if (v is not None and v != v) else v


def get_foldstep_pairs(combfold_output_path: str) -> list[str] | None:
    """Extract sorted UniProt pair names that CombFold used for this assembly."""
    pdb_path = Path(combfold_output_path)
    m = re.search(r"output_clustered_(\d+)\.pdb$", pdb_path.name)
    if not m:
        return None
    line_idx = int(m.group(1))

    assembly_output_dir = (
        pdb_path.parent.parent / "_unified_representation" / "assembly_output"
    )
    res_path        = assembly_output_dir / "output_clustered.res"
    chain_list_path = assembly_output_dir / "chain.list"

    if not res_path.exists() or not chain_list_path.exists():
        return None

    idx_to_uniprot = {
        i: Path(l.strip()).stem.split("_")[0]
        for i, l in enumerate(chain_list_path.read_text().strip().splitlines())
    }

    lines = [l for l in res_path.read_text().splitlines() if l.strip()]
    if line_idx >= len(lines):
        return None
    m2 = re.search(r"foldSteps: (.+)$", lines[line_idx])
    if not m2:
        return None

    return sorted({
        "_".join(sorted([idx_to_uniprot[int(a)], idx_to_uniprot[int(b)]]))
        for a, b in re.findall(r"\((\d+), (\d+)\)", m2.group(1))
        if int(a) in idx_to_uniprot and int(b) in idx_to_uniprot
    })


def find_pair_pdb(combfold_output_path: str, pair: str) -> tuple[Path | None, list[str]]:
    """
    Find the input AF3 pairwise PDB for a foldstep pair.
    Returns (path, [chainA_protein, chainB_protein]).
    Tries both protein orderings; falls back to pipeline-13 pool_input.
    chain_order is the actual A→B ordering in the file found.
    """
    p = Path(combfold_output_path)
    pool_output_dir = p.parent.parent           # .../pool_output/
    pool_dir_name   = pool_output_dir.name      # e.g. O13297x1_Q01159x1_pool_output
    pool_input_name = pool_dir_name.replace("pool_output", "pool_input")

    bases = [
        pool_output_dir.parent,                 # same-pipeline CombFold dir
        PIPELINE_13 / "CombFold",               # fallback: pipeline 13
    ]
    proteins = pair.split("_")

    for base in bases:
        pool_input = base / pool_input_name
        for order in [proteins, proteins[::-1]]:
            pdb = pool_input / "pdbs" / f"AFM_{'_'.join(order)}_unrelaxed_rank_1_model_1.pdb"
            if pdb.exists():
                return pdb, list(order)

    return None, list(proteins)  # default to sorted pair order when not found


def read_chain_list(cf_output_path: Path) -> dict[str, str] | None:
    """Chain letter → UniProt from CombFold's chain.list manifest."""
    folder     = cf_output_path.parent.parent
    chain_list = folder / "_unified_representation" / "assembly_output" / "chain.list"
    if not chain_list.exists():
        return None
    mapping: dict[str, str] = {}
    for line in chain_list.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pdb$", line)
        if m:
            mapping[m.group(2)] = m.group(1)   # chain letter → UniProt
        else:
            print(f"  WARNING unparseable chain.list line: {line!r}", file=sys.stderr)
    return mapping or None


def build_cf_chain_map(pdb_text: str, cf_output_path: Path) -> dict[str, str]:
    """Chain letter → UniProt, preferring chain.list; falls back to folder-name parsing."""
    manifest = read_chain_list(cf_output_path)

    chains_in_model: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and len(line) > 21:
            ch = line[21]
            if ch not in chains_in_model:
                chains_in_model.append(ch)

    if manifest is not None:
        missing = [ch for ch in chains_in_model if ch not in manifest]
        if missing:
            print(f"  WARNING chain.list missing model chains {missing}", file=sys.stderr)
        return manifest

    folder_name  = cf_output_path.parent.parent.name
    print(f"  WARNING no chain.list for {folder_name} — folder-name fallback", file=sys.stderr)
    base         = folder_name.replace("_pool_output", "")
    uniprot_list = [
        acc
        for acc, n in re.findall(r"([A-Z][A-Z0-9]{5,})x(\d+)", base)
        for _ in range(int(n))
    ]
    return {ch: uniprot_list[i] for i, ch in enumerate(chains_in_model) if i < len(uniprot_list)}


# ── load benchmark eval ───────────────────────────────────────────────────────

print("Loading benchmark parquet...")
bench = pl.read_parquet(BENCH_PARQ)

bench = (
    bench
    .filter(pl.col("n_proteins") > 2)
    .filter(
        pl.col("combfold_output_path").map_elements(
            lambda p: Path(p).exists(), return_dtype=pl.Boolean
        )
    )
)
print(f"  {bench.height} rows with CF output on disk (n_proteins > 2)")

bench = (
    bench
    .sort("CF_confidence", descending=True, nulls_last=True)
    .unique("complex_ac", keep="first")
    .sort("usalign_cpx_tm_score", descending=True, nulls_last=True)
)
print(f"  {bench.height} unique complexes (best CF output each)")

# ── foldstep pairs ────────────────────────────────────────────────────────────

print("Computing foldstep pairs from .res files...")
bench = bench.with_columns(
    pl.col("combfold_output_path")
    .map_elements(get_foldstep_pairs, return_dtype=pl.List(pl.String))
    .alias("foldstep_pairs")
)
n_ok = bench.filter(pl.col("foldstep_pairs").is_not_null()).height
print(f"  {n_ok}/{bench.height} complexes with foldstep pairs resolved")

# ── pair metrics ──────────────────────────────────────────────────────────────

print("Loading pair metrics...")
metrics = pl.read_csv(METRICS_CSV)

METRIC_COLS = [c for c in [
    "chain_pair_iptm",
    "chain_pair_iptm_corrected",
    "chain_pair_pae_min",
    "chain_pair_pae_min_recap",
    "plddt1",
    "plddt2",
] if c in metrics.columns]
print(f"  Metric columns available: {METRIC_COLS}")

bench = bench.with_row_index("row_id")

foldstep_long = (
    bench
    .select("row_id", "complex_ac", "foldstep_pairs")
    .explode("foldstep_pairs")
    .rename({"foldstep_pairs": "pair"})
    .drop_nulls("pair")
)

metrics = metrics.join(
    foldstep_long.select("complex_ac", "pair").unique(),
    on=["complex_ac", "pair"],
    how="semi",
)

dup_pairs = (
    metrics
    .group_by("complex_ac", "pair")
    .agg(pl.len().alias("n"))
    .filter(pl.col("n") > 1)
)
if dup_pairs.height:
    print(f"  {dup_pairs.height} ambiguous (complex_ac, pair) combos dropped ({dup_pairs['n'].sum()} rows)")
    metrics = metrics.join(
        dup_pairs.select("complex_ac", "pair"),
        on=["complex_ac", "pair"],
        how="anti",
    )

if "pair_type" in metrics.columns and "protein1" in metrics.columns:
    stoich = bench.select("complex_ac", "CP_stochiometry").unique("complex_ac")

    def is_real_homodimer(row: dict) -> bool:
        s = row.get("CP_stochiometry")
        if not s:
            return False
        try:
            return json.loads(s).get(row["protein1"], 0) >= 2
        except Exception:
            return False

    metrics = (
        metrics
        .join(stoich, on="complex_ac", how="left")
        .filter(
            (pl.col("pair_type") != "homo")
            | pl.struct("protein1", "CP_stochiometry")
              .map_elements(is_real_homodimer, return_dtype=pl.Boolean)
        )
    )

metrics_by_row = (
    foldstep_long
    .join(metrics, on=["complex_ac", "pair"], how="inner")
    .group_by("row_id")
    .agg(
        pl.col("pair").alias("af_pairs"),
        *[pl.col(c) for c in METRIC_COLS],
    )
)
bench = bench.join(metrics_by_row, on="row_id", how="left")
n_with = bench.filter(pl.col("af_pairs").is_not_null()).height
print(f"  {n_with}/{bench.height} complexes with pair metrics")

# ── build embed dict ──────────────────────────────────────────────────────────

print("\nBuilding embed data (reading PDB files)...")
EMBED: dict[str, dict] = {}

for row in bench.iter_rows(named=True):
    ac        = row["complex_ac"]
    out_path  = Path(row["combfold_output_path"])
    cf_conf   = row.get("CF_confidence")
    n_prot    = row["n_proteins"]
    tm_score  = row.get("usalign_cpx_tm_score")

    foldstep_pairs = row.get("foldstep_pairs") or []
    af_pairs       = row.get("af_pairs")       or []

    tm_str = f"{tm_score:.3f}" if (tm_score is not None and tm_score == tm_score) else "N/A"
    print(f"  {ac}  TM={tm_str}  CF={cf_conf:.1f}  n={n_prot}  foldstep_pairs={len(foldstep_pairs)}")

    # ── CF structure ──
    try:
        cf_text      = out_path.read_text()
        cf_chain_map = build_cf_chain_map(cf_text, out_path)
        cf_gz        = compress(strip_pdb(cf_text))
    except Exception as e:
        print(f"    WARNING: skipping {ac}: {e}", file=sys.stderr)
        continue

    # ── metrics lookup keyed by pair string ──
    metrics_lookup: dict[str, dict] = {}
    for j, pair_str in enumerate(af_pairs):
        entry = {}
        for col in METRIC_COLS:
            vals = row.get(col)
            v = vals[j] if (vals is not None and j < len(vals)) else None
            entry[col] = nan_to_none(v)
        metrics_lookup[pair_str] = entry

    # ── one entry per foldstep pair ──
    pairs: list[dict] = []
    n_found = 0
    for pair_str in foldstep_pairs:
        if "_" not in pair_str:
            continue

        pdb_path, chain_order = find_pair_pdb(row["combfold_output_path"], pair_str)
        pair_gz = None
        if pdb_path is not None:
            try:
                pair_text = pdb_path.read_text(errors="replace")
                pair_gz   = compress(strip_pdb(pair_text))
                n_found  += 1
            except Exception as e:
                print(f"    WARNING: could not read {pdb_path.name}: {e}", file=sys.stderr)

        m = metrics_lookup.get(pair_str, {})
        p1, p2 = chain_order[0], chain_order[1] if len(chain_order) > 1 else chain_order[0]
        pairs.append({
            "pair"    : pair_str,
            "proteins": chain_order,   # [chainA_protein, chainB_protein] — actual AF3 PDB order
            "label"   : f"{p1} – {p2}" if p1 != p2 else f"{p1} (homo)",
            "pair_gz" : pair_gz,
            **{col: nan_to_none(m.get(col)) for col in METRIC_COLS},
        })

    # Sort by ipTM_corrected descending (None last)
    pairs.sort(key=lambda p: (
        p.get("chain_pair_iptm_corrected") is None,
        -(p.get("chain_pair_iptm_corrected") or 0),
    ))

    print(f"    {n_found}/{len(pairs)} pair PDBs found on disk")

    EMBED[ac] = {
        "complex_ac"   : ac,
        "cf_confidence": nan_to_none(round(cf_conf, 2) if cf_conf is not None else None),
        "n_proteins"   : int(n_prot),
        "tm_score"     : nan_to_none(tm_score),
        "cf_gz"        : cf_gz,
        "cf_chain_map" : cf_chain_map,
        "pairs"        : pairs,
    }

print(f"\n  Total: {len(EMBED)} complexes embedded")
if not EMBED:
    print("ERROR: nothing to embed — check filters and paths", file=sys.stderr)
    sys.exit(1)


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CombFold Pair Input Viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 13px; color: rgb(36,36,36);
  background: #f0f0f0; padding: 10px;
  height: 100vh; display: flex; flex-direction: column; gap: 8px;
  overflow: hidden;
}

/* ── topbar ── */
.topbar {
  background: white; border: 1px solid #ddd; border-radius: 6px;
  padding: 8px 14px; display: flex; align-items: center; gap: 10px;
  flex-shrink: 0; flex-wrap: wrap;
}
.topbar label { font-weight: 600; white-space: nowrap; }
#complex-select {
  flex: 1; min-width: 240px; max-width: 600px;
  padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px;
}
.badge {
  display: inline-block; padding: 2px 10px; border-radius: 10px;
  font-size: 11px; font-weight: 600; white-space: nowrap;
}
.badge-cf  { background: #ddeeff; color: #004a80; border: 1px solid #aaccee; }
.badge-n   { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
.badge-tm  { background: #f3e5f5; color: #4a148c; border: 1px solid #ce93d8; }

/* ── panels ── */
.panels {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 8px; flex: 1; min-height: 0;
}
.panel {
  background: white; border: 1px solid #ddd; border-radius: 6px;
  display: flex; flex-direction: column; overflow: hidden;
}
.panel-hdr {
  padding: 7px 12px; border-bottom: 1px solid #eee;
  display: flex; align-items: center; gap: 8px;
  flex-shrink: 0; min-height: 38px;
}
.panel-hdr h3 { font-size: 13px; font-weight: 600; }
.panel-body  { flex: 1; position: relative; min-height: 0; }
.viewer3d    { width: 100%; height: 100%; }
.overlay {
  position: absolute; inset: 0;
  background: rgba(255,255,255,.82);
  display: none; align-items: center; justify-content: center;
  font-size: 13px; color: #555; text-align: center; padding: 20px;
}
.overlay.on { display: flex; }
.path-label {
  position: absolute; left: 8px; bottom: 6px; z-index: 5; pointer-events: none;
  font-size: 10px; color: #777; font-family: monospace;
  background: rgba(255,255,255,.85); padding: 2px 6px; border-radius: 4px;
  max-width: 80%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* ── pair bar ── */
.pair-bar {
  background: white; border: 1px solid #ddd; border-radius: 6px;
  padding: 7px 12px; display: flex; align-items: center; gap: 10px;
  flex-shrink: 0;
}
.pair-bar-label { font-weight: 600; font-size: 12px; color: #555; white-space: nowrap; }
.navbtn {
  background: #f2f2f2; border: 1px solid #ccc; border-radius: 4px;
  cursor: pointer; padding: 1px 8px; font-size: 13px; line-height: 1.6;
  flex-shrink: 0;
}
.navbtn:hover:not(:disabled) { background: #e4e4e4; }
.navbtn:disabled { opacity: .35; cursor: default; }
.navbtn.active-toggle { background: #0072B2; color: white; border-color: #005a8e; }
.pair-list-wrap {
  flex: 1; overflow-x: auto; display: flex; gap: 5px;
  padding-bottom: 3px; min-width: 0;
}
.pair-btn {
  white-space: nowrap; flex-shrink: 0;
  padding: 2px 10px; font-size: 11px; font-family: monospace; font-weight: 600;
  border: 1px solid #ccc; border-radius: 4px;
  cursor: pointer; background: #f2f2f2;
}
.pair-btn:hover:not(.active) { background: #e4e4e4; }
.pair-btn.active { background: #0072B2; color: white; border-color: #0072B2; }
/* dashed = input PDB not found on disk */
.pair-btn.no-pdb { border-style: dashed; opacity: 0.55; }
.pair-count { font-size: 11px; color: #999; white-space: nowrap; flex-shrink: 0; }

/* ── metrics ── */
.pair-metrics {
  font-size: 12px; font-family: monospace;
  white-space: nowrap; color: #444;
  border-left: 1px solid #eee; padding-left: 10px;
  display: flex; gap: 8px; align-items: center; flex-shrink: 0;
}
.m-label { color: #888; font-size: 11px; }
.m-val   { font-weight: 600; color: #111; }
.m-na    { color: #bbb; }
.m-sep   { color: #ddd; }

/* ── legend ── */
.legend {
  display: flex; gap: 10px; align-items: center;
  font-size: 11px; color: #666; white-space: nowrap;
  border-left: 1px solid #eee; padding-left: 10px; flex-shrink: 0;
}
.leg-sw {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 2px; margin-right: 3px; vertical-align: middle;
}
</style>
</head>
<body>

<div class="topbar">
  <label>Complex:</label>
  <select id="complex-select"></select>
  <span class="badge badge-cf" id="badge-cf"></span>
  <span class="badge badge-n"  id="badge-n"></span>
  <span class="badge badge-tm" id="badge-tm">TM ?</span>
</div>

<div class="panels">

  <div class="panel">
    <div class="panel-hdr">
      <h3>CombFold Assembly</h3>
    </div>
    <div class="panel-body">
      <div id="cf-viewer"  class="viewer3d"></div>
      <div id="cf-overlay" class="overlay">Loading...</div>
      <span id="cf-path" class="path-label"></span>
    </div>
  </div>

  <div class="panel">
    <div class="panel-hdr">
      <h3>Input Pair: <span id="pair-title" style="color:#0072B2;font-weight:700">—</span></h3>
    </div>
    <div class="panel-body">
      <div id="pair-viewer"  class="viewer3d"></div>
      <div id="pair-overlay" class="overlay on">Select a pair below</div>
    </div>
  </div>

</div>

<!-- pair navigation bar -->
<div class="pair-bar">
  <span class="pair-bar-label">Foldstep pairs:</span>
  <button class="navbtn" id="prev-pair" title="Previous pair (←)">&#9664;</button>
  <div class="pair-list-wrap" id="pair-list"></div>
  <button class="navbtn" id="next-pair" title="Next pair (→)">&#9654;</button>
  <span class="pair-count" id="pair-count"></span>
  <button class="navbtn" id="toggle-others" title="Hide/show non-pair chains (H)">Hide others</button>
  <div class="pair-metrics" id="pair-metrics">
    <span class="m-na">Select a pair to highlight it</span>
  </div>
  <div class="legend">
    <span><span class="leg-sw" style="background:#0072B2"></span>Prot 1 (chain A)</span>
    <span><span class="leg-sw" style="background:#009E73"></span>Prot 2 (chain B)</span>
    <span><span class="leg-sw" style="background:#E69F00"></span>Other (CF)</span>
  </div>
</div>

<script>
const COMPLEXES = %%JSON%%;

/* Okabe-Ito palette */
const CC     = ['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#F0E442','#999999'];
const YELLOW = '#E69F00';
const LABEL_STYLE = {
  backgroundColor: 'black', backgroundOpacity: 0.75,
  fontColor: 'white', fontSize: 12, padding: 4, inFront: true,
};

/* ── state ── */
let cfV = null, pairV = null;
let cur = null;
let cfRaw = null;
let currentPairIdx = -1;
let hideOthers = false;

/* ── gzip decompress ── */
async function ungzip(b64) {
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const ds  = new DecompressionStream('gzip');
  const w   = ds.writable.getWriter();
  w.write(raw); w.close();
  const chunks = [], r = ds.readable.getReader();
  for (;;) { const {done, value} = await r.read(); if (done) break; chunks.push(value); }
  let len = 0, off = 0;
  chunks.forEach(c => len += c.length);
  const buf = new Uint8Array(len);
  chunks.forEach(c => { buf.set(c, off); off += c.length; });
  return new TextDecoder().decode(buf);
}

/* ── overlay ── */
function spin(id, on, msg) {
  const el = document.getElementById(id);
  el.classList.toggle('on', on);
  if (msg !== undefined) el.textContent = msg;
}

/* ── init viewers ── */
function initViewers() {
  const opts = {backgroundColor: 'white', antialias: true};
  cfV   = $3Dmol.createViewer(document.getElementById('cf-viewer'),   opts);
  pairV = $3Dmol.createViewer(document.getElementById('pair-viewer'), opts);
}

/* ── hover callbacks ── */
function cfHoverCB(atom, viewer) {
  const u = cur?.cf_chain_map?.[atom.chain];
  viewer.removeAllLabels();
  viewer.addLabel(
    u ? `Chain ${atom.chain}: ${u}` : `Chain ${atom.chain} (unmapped)`,
    {...LABEL_STYLE, position: atom}
  );
  viewer.render();
}
function cfUnhoverCB(atom, viewer) { viewer.removeAllLabels(); viewer.render(); }

function pairHoverCB(atom, viewer) {
  /* proteins[] = [chainA_protein, chainB_protein] as stored in the pair embed */
  const pair = cur?.pairs?.[currentPairIdx];
  const chainOrder = 'ABCDEFGHIJKLMNOP'.split('');
  const idx  = chainOrder.indexOf(atom.chain);
  const u    = (pair && idx >= 0 && idx < pair.proteins.length)
                 ? pair.proteins[idx]
                 : atom.chain;
  viewer.removeAllLabels();
  viewer.addLabel(`Chain ${atom.chain}: ${u}`, {...LABEL_STYLE, position: atom});
  viewer.render();
}
function pairUnhoverCB(atom, viewer) { viewer.removeAllLabels(); viewer.render(); }

/* ── default by-chain style ── */
function styleByChain(v, text, fmt, hoverOn, hoverOff) {
  v.removeAllModels();
  const model  = v.addModel(text, fmt);
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();
  chains.forEach((ch, i) => v.setStyle({chain: ch}, {cartoon: {color: CC[i % CC.length]}}));
  v.setHoverable({}, true, hoverOn, hoverOff);
  v.zoomTo(); v.render();
}

/* ── highlight CF assembly for a selected pair ── */
/* proteins[0] (= chain A in the AF3 pair PDB) → blue
   proteins[1] (= chain B in the AF3 pair PDB) → green
   others                                       → yellow (or hidden)        */
function highlightCFPair(proteins) {
  if (!cfRaw || !cfV) return;
  const [p1, p2] = proteins;
  const homo     = p1 === p2;

  cfV.removeAllModels();
  const model  = cfV.addModel(cfRaw, 'pdb');
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();

  chains.forEach(ch => {
    const u = cur?.cf_chain_map?.[ch];
    if      (u === p1)          cfV.setStyle({chain: ch}, {cartoon: {color: CC[0]}});   // blue
    else if (!homo && u === p2) cfV.setStyle({chain: ch}, {cartoon: {color: CC[2]}});   // green
    else if (hideOthers)        cfV.setStyle({chain: ch}, {});                           // hidden
    else                        cfV.setStyle({chain: ch}, {cartoon: {color: YELLOW}});  // yellow
  });
  cfV.setHoverable({}, true, cfHoverCB, cfUnhoverCB);
  cfV.zoomTo(); cfV.render();
}

/* ── load pair PDB into right panel ── */
async function loadPairPDB(pair) {
  document.getElementById('pair-title').textContent = pair ? pair.label : '—';

  if (!pair || !pair.pair_gz) {
    pairV.removeAllModels(); pairV.render();
    spin('pair-overlay', true,
      pair ? `No input PDB found on disk for ${pair.label}` : 'Select a pair below');
    return;
  }

  spin('pair-overlay', true, `Loading ${pair.label}…`);
  try {
    const text = await ungzip(pair.pair_gz);
    pairV.removeAllModels();
    const model  = pairV.addModel(text, 'pdb');
    const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();
    /* Chain A → blue (CC[0]), chain B → green (CC[2]), extras follow palette */
    chains.forEach((ch, i) => {
      const color = i === 0 ? CC[0] : i === 1 ? CC[2] : CC[i % CC.length];
      pairV.setStyle({chain: ch}, {cartoon: {color}});
    });
    pairV.setHoverable({}, true, pairHoverCB, pairUnhoverCB);
    pairV.zoomTo(); pairV.render();
    spin('pair-overlay', false);
  } catch(e) {
    console.error(e);
    spin('pair-overlay', true, 'Load error: ' + e.message);
    setTimeout(() => spin('pair-overlay', false), 4000);
  }
}

/* ── pair bar ── */
function buildPairButtons() {
  const list  = document.getElementById('pair-list');
  list.innerHTML = '';
  const pairs = cur?.pairs || [];
  pairs.forEach((pair, i) => {
    const b     = document.createElement('button');
    b.className = 'pair-btn' + (pair.pair_gz ? '' : ' no-pdb');
    b.textContent = pair.label;
    const iptm  = pair.chain_pair_iptm_corrected;
    b.title     = [
      pair.pair,
      iptm != null ? `ipTM corr: ${iptm.toFixed(3)}` : null,
      pair.pair_gz ? null : '⚠ input PDB not found on disk',
    ].filter(Boolean).join('  |  ');
    b.onclick   = () => selectPair(i);
    list.appendChild(b);
  });
  document.getElementById('pair-count').textContent =
    pairs.length ? `${pairs.length} pair${pairs.length > 1 ? 's' : ''}` : 'no pairs';
  updateNavButtons();
  updateMetricsDisplay(-1);
}

function selectPair(idx) {
  const pairs = cur?.pairs || [];
  if (idx < 0 || idx >= pairs.length) return;
  currentPairIdx = idx;
  const pair     = pairs[idx];

  document.querySelectorAll('.pair-btn')
    .forEach((b, i) => b.classList.toggle('active', i === idx));
  document.querySelectorAll('.pair-btn')[idx]
    ?.scrollIntoView({block: 'nearest', inline: 'nearest'});

  updateNavButtons();
  updateMetricsDisplay(idx);

  if (cfRaw)  highlightCFPair(pair.proteins);
  loadPairPDB(pair);
}

function updateNavButtons() {
  const n = cur?.pairs?.length ?? 0;
  document.getElementById('prev-pair').disabled = currentPairIdx <= 0;
  document.getElementById('next-pair').disabled = currentPairIdx >= n - 1;
}

function updateMetricsDisplay(idx) {
  const el   = document.getElementById('pair-metrics');
  const pair = cur?.pairs?.[idx];
  if (!pair) {
    el.innerHTML = '<span class="m-na">Select a pair to highlight it</span>';
    return;
  }

  const fmt = (v, d = 3) =>
    v == null ? '<span class="m-na">n/a</span>'
              : `<span class="m-val">${Number(v).toFixed(d)}</span>`;

  const plddt1 = pair.plddt1, plddt2 = pair.plddt2;
  const plddtStr = (plddt1 == null && plddt2 == null)
    ? '<span class="m-na">n/a</span>'
    : `${fmt(plddt1, 1)} / ${fmt(plddt2, 1)}`;

  const items = [
    `<span class="m-label">ipTM corr:</span> ${fmt(pair.chain_pair_iptm_corrected)}`,
    `<span class="m-sep">|</span>`,
    `<span class="m-label">ipTM:</span> ${fmt(pair.chain_pair_iptm)}`,
    `<span class="m-sep">|</span>`,
    `<span class="m-label">PAE min:</span> ${fmt(pair.chain_pair_pae_min, 1)} Å`,
  ];
  // PAE recap only present in data-26.08 snapshot
  if (pair.chain_pair_pae_min_recap !== undefined) {
    items.push(`<span class="m-sep">|</span>`);
    items.push(`<span class="m-label">PAE recap:</span> ${fmt(pair.chain_pair_pae_min_recap, 1)} Å`);
  }
  items.push(`<span class="m-sep">|</span>`);
  items.push(`<span class="m-label">pLDDT:</span> ${plddtStr}`);

  el.innerHTML = items.join(' ');
}

/* ── load CF (from embedded gzip) ── */
async function loadCF() {
  if (!cur) return;
  cfRaw = null;
  spin('cf-overlay', true, 'Loading CF model…');
  try {
    const text = await ungzip(cur.cf_gz);
    cfRaw = text;
    document.getElementById('cf-path').textContent = cur.complex_ac;

    if (currentPairIdx >= 0 && cur.pairs[currentPairIdx]) {
      highlightCFPair(cur.pairs[currentPairIdx].proteins);
    } else {
      styleByChain(cfV, cfRaw, 'pdb', cfHoverCB, cfUnhoverCB);
      if (cur.pairs.length > 0) selectPair(0);   // auto-select first pair
    }
    spin('cf-overlay', false);
  } catch(e) {
    console.error(e);
    spin('cf-overlay', true, 'CF load error: ' + e.message);
    setTimeout(() => spin('cf-overlay', false), 4000);
  }
}

/* ── select complex ── */
function selectComplex(ac) {
  cur = COMPLEXES[ac];
  if (!cur) return;

  cfRaw = null;
  currentPairIdx = -1;

  document.getElementById('badge-cf').textContent =
    `CF ${cur.cf_confidence?.toFixed(1) ?? '?'}`;
  document.getElementById('badge-n').textContent  = `n = ${cur.n_proteins}`;
  document.getElementById('badge-tm').textContent =
    cur.tm_score != null ? `TM ${cur.tm_score.toFixed(3)}` : 'TM ?';

  /* reset right panel */
  pairV.removeAllModels(); pairV.render();
  spin('pair-overlay', true, 'Select a pair below');
  document.getElementById('pair-title').textContent = '—';

  buildPairButtons();
  loadCF();
}

/* ── sort: TM desc → n_proteins desc → CF desc ── */
function sortByTM(a, b) {
  const tmA = a.tm_score ?? -1, tmB = b.tm_score ?? -1;
  return tmB - tmA || b.n_proteins - a.n_proteins || (b.cf_confidence ?? 0) - (a.cf_confidence ?? 0);
}

/* ── dropdown ── */
function buildDropdown() {
  const sel = document.getElementById('complex-select');
  Object.values(COMPLEXES).sort(sortByTM).forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.complex_ac;
    opt.textContent = [
      c.complex_ac,
      `n=${c.n_proteins}`,
      `CF: ${c.cf_confidence?.toFixed(1) ?? '?'}`,
      `pairs: ${c.pairs.length}`,
      `TM: ${c.tm_score != null ? c.tm_score.toFixed(3) : '?'}`,
    ].join('  |  ');
    sel.appendChild(opt);
  });
  sel.onchange = () => selectComplex(sel.value);
}

/* ── toggle-others button ── */
function applyHideToggle() {
  const btn = document.getElementById('toggle-others');
  btn.textContent = hideOthers ? 'Show others' : 'Hide others';
  btn.classList.toggle('active-toggle', hideOthers);
  if (currentPairIdx >= 0 && cur?.pairs[currentPairIdx]) {
    if (cfRaw) highlightCFPair(cur.pairs[currentPairIdx].proteins);
  }
}
document.getElementById('toggle-others').addEventListener('click', () => {
  hideOthers = !hideOthers;
  applyHideToggle();
});

/* ── keyboard shortcuts ── */
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const n = cur?.pairs?.length ?? 0;
  if (e.key === 'ArrowLeft'  && currentPairIdx > 0)     selectPair(currentPairIdx - 1);
  if (e.key === 'ArrowRight' && currentPairIdx < n - 1) selectPair(currentPairIdx + 1);
  if (e.key === 'h' || e.key === 'H') { hideOthers = !hideOthers; applyHideToggle(); }
});

document.getElementById('prev-pair').onclick = () => selectPair(currentPairIdx - 1);
document.getElementById('next-pair').onclick = () => selectPair(currentPairIdx + 1);

/* ── init ── */
initViewers();
buildDropdown();
const first = Object.values(COMPLEXES).sort(sortByTM)[0]?.complex_ac;
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
print(f"  Size:    {size_kb:.0f} KB  (~{size_kb/1024:.1f} MB)")
