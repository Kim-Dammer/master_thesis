#!/usr/bin/env python3
"""
Generate self-contained HTML viewer for homomultimer pool vs pair comparison.

Left panel  : scrollable table of all complexes with TM-score & RMSD for
              pool and pair, colour-coded by TM-score.  Click a row to load.
Right panel : 3Dmol viewer overlaying three structures:
                • PDB reference     (blue,   #0072B2)
                • Pool aligned pred (orange, #E69F00)
                • Pair aligned pred (green,  #009E73)
              Each layer has a toggle checkbox + opacity slider.

The aligned PDB files are the US-align rotated mobile structures written by
  -o <output_dir>/usalign
i.e.  <cf_output_dir>/us_align/usalign.pdb
"""

import base64, gzip, json
from pathlib import Path

import polars as pl
from procompa import get_project_root

PRJ_ROOT = get_project_root()
DATA     = PRJ_ROOT / "data"

PARQUET  = PRJ_ROOT / "scripts/homomultimer_poll_pair_comparison/homomultimers_pool_pair_usalign.parquet"
OUT      = DATA / "Pipeline/viewer/05_homomultimer_pool_pair_viewer.html"


# ── helpers ───────────────────────────────────────────────────────────────────
 
def compress(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode())).decode()
 
def strip_pdb(text: str) -> str:
    _KEEP = frozenset({"ATOM", "HETATM", "MODEL", "ENDMDL", "TER", "END"})
    return "\n".join(l for l in text.splitlines() if l[:6].strip() in _KEEP)
 
def read_compress(path: Path, label: str, strip: bool = True) -> str | None:
    if not path.exists():
        print(f"  WARNING not found: {path}  [{label}]")
        return None
    try:
        text = path.read_text()
        return compress(strip_pdb(text) if strip else text)
    except Exception as e:
        print(f"  WARNING failed to read {label} at {path}: {e}")
        return None
 
def find_input_pair(cf_dir: Path, uniprot_id: str, n_chains: int, condition: str) -> Path | None:
    """Find the self-pair AF3 input PDB (homodimer) for pool or pair condition."""
    pdbs_dir = cf_dir / f"{uniprot_id}x{n_chains}_{condition}_input" / "pdbs"
    if not pdbs_dir.exists():
        print(f"  WARNING input pdbs dir not found: {pdbs_dir}")
        return None
    # Self-pair: AFM_UNIPROT_UNIPROT_*.pdb — take best rank (sorted alphabetically)
    matches = sorted(pdbs_dir.glob(f"AFM_{uniprot_id}_{uniprot_id}_*.pdb"))
    if not matches:
        print(f"  WARNING no self-pair PDB in {pdbs_dir}")
        return None
    return matches[0]
 
 
# ── load data & embed structures ──────────────────────────────────────────────
 
df = pl.read_parquet(PARQUET)
print(f"Loaded {df.height} rows from {PARQUET.name}")
 
EMBED: dict[str, dict] = {}
 
for row in df.iter_rows(named=True):
    ac        = row["complex_ac"]
    uniprot   = row["uniprot_id"]
    n_chains  = row["n_chains"]
    pool_pred = Path(row["cf_max_conf_path_pool"])
    pair_pred = Path(row["cf_max_conf_path_pair"])
    ref_path  = Path(row["pdb_path"])
 
    # CombFold root dir (…/CombFold/)
    cf_dir = pool_pred.parent.parent.parent
 
    pool_aligned = pool_pred.parent.parent / "us_align" / "usalign.pdb"
    pair_aligned = pair_pred.parent.parent / "us_align" / "usalign.pdb"
    pool_input   = find_input_pair(cf_dir, uniprot, n_chains, "pool")
    pair_input   = find_input_pair(cf_dir, uniprot, n_chains, "pair")
 
    ref_fmt = "cif" if ref_path.suffix == ".cif" else "pdb"
    ref_gz  = read_compress(ref_path, f"{ac} ref", strip=(ref_fmt == "pdb"))
 
    print(f"  {ac}: ref={'✓' if ref_path.exists() else '✗'}  "
          f"pool={'✓' if pool_aligned.exists() else '✗'}  "
          f"pair={'✓' if pair_aligned.exists() else '✗'}  "
          f"pool_in={'✓' if pool_input else '✗'}  "
          f"pair_in={'✓' if pair_input else '✗'}")
 
    EMBED[ac] = {
        "complex_ac"     : ac,
        "uniprot_id"     : uniprot,
        "pdb_id"         : row["pdb_id"],
        "n_chains"       : n_chains,
        "tm_pool"        : row.get("tm_score_cpx_pool"),
        "rmsd_pool"      : row.get("rmsd_cpx_pool"),
        "tm_pair"        : row.get("tm_score_cpx_pair"),
        "rmsd_pair"      : row.get("rmsd_cpx_pair"),
        "ref_gz"         : ref_gz,
        "ref_fmt"        : ref_fmt,
        "pool_gz"        : read_compress(pool_aligned, f"{ac} pool aligned"),
        "pair_gz"        : read_compress(pair_aligned, f"{ac} pair aligned"),
        "pool_input_gz"  : read_compress(pool_input,   f"{ac} pool input") if pool_input else None,
        "pair_input_gz"  : read_compress(pair_input,   f"{ac} pair input") if pair_input else None,
    }
 
print(f"\nEmbedded {len(EMBED)} complexes")
 
 
# ── HTML template ─────────────────────────────────────────────────────────────
 
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Homomultimer Pool vs Pair</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body {
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:13px; color:rgb(36,36,36);
  background:#f0f0f0;
  height:100vh; display:flex; flex-direction:column; gap:8px; padding:10px;
}
.topbar {
  background:white; border:1px solid #ddd; border-radius:6px;
  padding:9px 14px; display:flex; align-items:center; gap:12px; flex-shrink:0;
}
.topbar h2   { font-size:14px; font-weight:700; color:#333; }
.topbar .hint { font-size:12px; color:#999; }
.main { display:flex; gap:8px; flex:1; min-height:0; }
 
/* ── left table ───────────────────────────────────────────────── */
.table-panel {
  width:430px; flex-shrink:0;
  background:white; border:1px solid #ddd; border-radius:6px;
  display:flex; flex-direction:column; overflow:hidden;
}
.table-panel-hdr {
  padding:7px 12px; border-bottom:1px solid #eee;
  font-size:11px; font-weight:600; color:#666; flex-shrink:0;
}
.table-wrap { overflow-y:auto; flex:1; }
table { width:100%; border-collapse:collapse; font-size:12px; }
thead th {
  position:sticky; top:0; background:#f7f7f7; z-index:1;
  padding:6px 8px; font-weight:600; font-size:11px; color:#444;
  border-bottom:1px solid #ddd; white-space:nowrap;
}
thead th.r { text-align:right; }
tbody tr { cursor:pointer; transition:background .1s; }
tbody tr:hover    { background:#f0f5ff; }
tbody tr.selected { background:#ddeeff; }
tbody td { padding:5px 8px; border-bottom:1px solid #f2f2f2; white-space:nowrap; }
.mono { font-family:monospace; }
.r    { text-align:right; }
.tm-val { font-family:monospace; font-weight:700; text-align:right; }
 
/* ── right viewer panel ───────────────────────────────────────── */
.viewer-panel {
  flex:1; min-width:0;
  background:white; border:1px solid #ddd; border-radius:6px;
  display:flex; flex-direction:column; overflow:hidden;
}
.viewer-hdr {
  padding:8px 14px; border-bottom:1px solid #eee;
  display:flex; align-items:center; gap:10px; flex-shrink:0; flex-wrap:wrap;
}
.viewer-hdr h3 { font-size:13px; font-weight:700; }
.badge {
  display:inline-block; padding:2px 9px; border-radius:10px;
  font-size:11px; font-weight:600; white-space:nowrap;
  background:#eee; color:#333; border:1px solid #ccc; font-family:monospace;
}
 
/* ── mode tabs ────────────────────────────────────────────────── */
.mode-tabs { display:flex; gap:3px; flex-shrink:0; }
.mode-tab {
  background:#f0f0f0; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:2px 10px; font-size:12px; font-weight:600;
  line-height:1.6; white-space:nowrap;
}
.mode-tab:hover:not(.active) { background:#e4e4e4; }
.mode-tab.active { background:#0072B2; color:white; border-color:#0072B2; }
 
/* ── viewer body ──────────────────────────────────────────────── */
.viewer-body { flex:1; display:flex; flex-direction:column; min-height:0; }
#overlay-view { flex:1; position:relative; min-height:0; }
#viewer-overlay { width:100%; height:100%; }
 
/* shared: SBS + Input Pairs use the same two-panel layout */
.dual-view {
  flex:1; min-height:0;
  display:none;       /* JS sets to 'flex' when active */
  flex-direction:row;
}
.sbs-panel {
  flex:1; min-width:0;
  display:flex; flex-direction:column; overflow:hidden;
}
.sbs-panel + .sbs-panel { border-left:2px solid #e8e8e8; }
.sbs-hdr {
  padding:6px 10px; border-bottom:1px solid #eeeeee;
  background:#fafafa; flex-shrink:0;
  display:flex; align-items:center; gap:8px; flex-wrap:wrap; min-height:36px;
}
.sbs-label { font-size:12px; font-weight:700; white-space:nowrap; flex-shrink:0; }
.sbs-body  { flex:1; position:relative; min-height:0; }
.sbs-body > div:first-child { width:100%; height:100%; }
 
/* ctrl groups inside sbs-hdr — one per mode */
.ctrl-grp { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
 
/* loading spinner */
.spin {
  position:absolute; inset:0; background:rgba(255,255,255,.85);
  display:none; align-items:center; justify-content:center;
  font-size:13px; color:#666; text-align:center; padding:20px;
}
.spin.on { display:flex; }
 
/* layer controls */
.layer-controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
.layer-ctrl     { display:flex; align-items:center; gap:5px; }
.swatch {
  width:10px; height:10px; border-radius:2px; flex-shrink:0;
  border:1px solid rgba(0,0,0,.15);
}
.layer-ctrl label { font-size:12px; white-space:nowrap; cursor:pointer; user-select:none; }
.layer-ctrl input[type=checkbox] { cursor:pointer; }
.op-slider { width:65px; cursor:pointer; accent-color:#555; }
.metric-chip {
  font-size:11px; font-family:monospace; color:#444;
  background:#f5f5f5; border:1px solid #e2e2e2;
  border-radius:4px; padding:1px 7px; white-space:nowrap;
}
.metric-chip b { font-weight:700; }
.chain-legend {
  display:flex; gap:6px; align-items:center; font-size:11px; color:#555;
}
.chain-swatch {
  width:10px; height:10px; border-radius:2px; flex-shrink:0;
  border:1px solid rgba(0,0,0,.15);
}
</style>
</head>
<body>
 
<div class="topbar">
  <h2>Homomultimer — Pool vs Pair</h2>
  <span class="hint">Click a row to load structures</span>
</div>
 
<div class="main">
 
  <!-- ── left: complex table ─────────────────────────────────── -->
  <div class="table-panel">
    <div class="table-panel-hdr">Complexes — sorted by Pool TM-score ▼</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>UniProt</th><th>PDB</th><th class="r">N</th>
            <th class="r">TM pool</th><th class="r">TM pair</th>
            <th class="r">RMSD pool</th><th class="r">RMSD pair</th>
          </tr>
        </thead>
        <tbody id="cpx-tbody"></tbody>
      </table>
    </div>
  </div>
 
  <!-- ── right: viewer panel ─────────────────────────────────── -->
  <div class="viewer-panel">
    <div class="viewer-hdr">
 
      <div class="mode-tabs">
        <button class="mode-tab active" id="tab-overlay" onclick="setMode('overlay')">Overlay</button>
        <button class="mode-tab"        id="tab-sbs"     onclick="setMode('sbs')">Side by side</button>
        <button class="mode-tab"        id="tab-ip"      onclick="setMode('ip')">Input pairs</button>
      </div>
 
      <h3 id="vwr-title">—</h3>
      <span class="badge" id="vwr-pdb">—</span>
 
      <!-- overlay-mode layer controls -->
      <div class="layer-controls" id="overlay-controls">
        <div class="layer-ctrl">
          <input type="checkbox" id="chk-ref" checked onchange="updateOverlayLayers()">
          <div class="swatch" style="background:#0072B2"></div>
          <label for="chk-ref">Reference</label>
          <input type="range" class="op-slider" id="op-ref"
                 min="0" max="1" step="0.05" value="1" oninput="updateOverlayLayers()">
        </div>
        <div class="layer-ctrl">
          <input type="checkbox" id="chk-pool" checked onchange="updateOverlayLayers()">
          <div class="swatch" style="background:#E69F00"></div>
          <label for="chk-pool">Pool</label>
          <input type="range" class="op-slider" id="op-pool"
                 min="0" max="1" step="0.05" value="1" oninput="updateOverlayLayers()">
          <span class="metric-chip" id="chip-pool">—</span>
        </div>
        <div class="layer-ctrl">
          <input type="checkbox" id="chk-pair" checked onchange="updateOverlayLayers()">
          <div class="swatch" style="background:#009E73"></div>
          <label for="chk-pair">Pair</label>
          <input type="range" class="op-slider" id="op-pair"
                 min="0" max="1" step="0.05" value="1" oninput="updateOverlayLayers()">
          <span class="metric-chip" id="chip-pair">—</span>
        </div>
      </div>
 
    </div><!-- .viewer-hdr -->
 
    <div class="viewer-body">
 
      <!-- overlay view -->
      <div id="overlay-view">
        <div id="viewer-overlay"></div>
        <div class="spin on" id="ov-main">Select a complex from the table</div>
      </div>
 
      <!-- shared dual-panel container (used by both SBS and Input Pairs) -->
      <div class="dual-view" id="dual-view">
 
        <!-- left panel -->
        <div class="sbs-panel">
          <div class="sbs-hdr" id="sbs-hdr-l">
            <!-- SBS controls -->
            <div class="ctrl-grp" id="sbs-ctrl-l">
              <span class="sbs-label" style="color:#E69F00">Pool</span>
              <div class="layer-ctrl">
                <input type="checkbox" id="sbs-chk-ref-l" checked onchange="updateSBSLayers()">
                <div class="swatch" style="background:#0072B2"></div>
                <label for="sbs-chk-ref-l">Ref</label>
                <input type="range" class="op-slider" id="sbs-op-ref-l"
                       min="0" max="1" step="0.05" value="1" oninput="updateSBSLayers()">
              </div>
              <div class="layer-ctrl">
                <input type="checkbox" id="sbs-chk-pool" checked onchange="updateSBSLayers()">
                <div class="swatch" style="background:#E69F00"></div>
                <label for="sbs-chk-pool">Pool</label>
                <input type="range" class="op-slider" id="sbs-op-pool"
                       min="0" max="1" step="0.05" value="1" oninput="updateSBSLayers()">
              </div>
              <span class="metric-chip" id="chip-pool-sbs">—</span>
            </div>
            <!-- Input Pairs controls -->
            <div class="ctrl-grp" id="ip-ctrl-l" style="display:none">
              <span class="sbs-label" style="color:#E69F00">Pool input</span>
              <div class="layer-ctrl">
                <label style="font-size:12px">Opacity</label>
                <input type="range" class="op-slider" id="ip-op-l"
                       min="0" max="1" step="0.05" value="1" oninput="updateIPLayers()">
              </div>
              <div class="chain-legend" id="ip-legend-l"></div>
            </div>
          </div>
          <div class="sbs-body">
            <div id="viewer-left"></div>
            <div class="spin" id="ov-left"></div>
          </div>
        </div>
 
        <!-- right panel -->
        <div class="sbs-panel">
          <div class="sbs-hdr" id="sbs-hdr-r">
            <!-- SBS controls -->
            <div class="ctrl-grp" id="sbs-ctrl-r">
              <span class="sbs-label" style="color:#009E73">Pair</span>
              <div class="layer-ctrl">
                <input type="checkbox" id="sbs-chk-ref-r" checked onchange="updateSBSLayers()">
                <div class="swatch" style="background:#0072B2"></div>
                <label for="sbs-chk-ref-r">Ref</label>
                <input type="range" class="op-slider" id="sbs-op-ref-r"
                       min="0" max="1" step="0.05" value="1" oninput="updateSBSLayers()">
              </div>
              <div class="layer-ctrl">
                <input type="checkbox" id="sbs-chk-pair" checked onchange="updateSBSLayers()">
                <div class="swatch" style="background:#009E73"></div>
                <label for="sbs-chk-pair">Pair</label>
                <input type="range" class="op-slider" id="sbs-op-pair"
                       min="0" max="1" step="0.05" value="1" oninput="updateSBSLayers()">
              </div>
              <span class="metric-chip" id="chip-pair-sbs">—</span>
            </div>
            <!-- Input Pairs controls -->
            <div class="ctrl-grp" id="ip-ctrl-r" style="display:none">
              <span class="sbs-label" style="color:#009E73">Pair input</span>
              <div class="layer-ctrl">
                <label style="font-size:12px">Opacity</label>
                <input type="range" class="op-slider" id="ip-op-r"
                       min="0" max="1" step="0.05" value="1" oninput="updateIPLayers()">
              </div>
              <div class="chain-legend" id="ip-legend-r"></div>
            </div>
          </div>
          <div class="sbs-body">
            <div id="viewer-right"></div>
            <div class="spin" id="ov-right"></div>
          </div>
        </div>
 
      </div><!-- #dual-view -->
 
    </div><!-- .viewer-body -->
  </div><!-- .viewer-panel -->
 
</div><!-- .main -->
 
<script>
const COMPLEXES = %%JSON%%;
 
const COL_REF  = '#0072B2';
const COL_POOL = '#E69F00';
const COL_PAIR = '#009E73';
/* Okabe-Ito chain colours */
const CC = ['#E69F00','#0072B2','#009E73','#CC79A7','#56B4E9','#D55E00','#F0E442','#999999'];
 
/* ── gzip decompress ──────────────────────────────────────────── */
async function ungzip(b64) {
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const ds  = new DecompressionStream('gzip');
  const w   = ds.writable.getWriter();
  w.write(raw); w.close();
  const chunks = [], r = ds.readable.getReader();
  for(;;){ const {done,value} = await r.read(); if(done) break; chunks.push(value); }
  let len=0, off=0;
  chunks.forEach(c => len += c.length);
  const buf = new Uint8Array(len);
  chunks.forEach(c => { buf.set(c,off); off+=c.length; });
  return new TextDecoder().decode(buf);
}
 
function tmColor(v) {
  if (v == null) return '#aaa';
  const t = Math.min(1, Math.max(0, v));
  if (t < 0.5) return `rgb(210,${Math.round(t*2*180)},40)`;
  return `rgb(${Math.round((1-(t-0.5)*2)*210)},170,40)`;
}
function fmt(v, d=3) { return (v == null) ? '—' : v.toFixed(d); }
 
/* ── build table ──────────────────────────────────────────────── */
const tbody = document.getElementById('cpx-tbody');
const acs   = Object.keys(COMPLEXES).sort(
  (a,b) => (COMPLEXES[b].tm_pool ?? -1) - (COMPLEXES[a].tm_pool ?? -1)
);
acs.forEach(ac => {
  const c = COMPLEXES[ac];
  const tr = document.createElement('tr');
  tr.dataset.ac = ac;
  tr.innerHTML = `
    <td class="mono" style="font-size:11px">${c.uniprot_id}</td>
    <td class="mono" style="font-size:11px">${(c.pdb_id||'').toUpperCase()}</td>
    <td class="r">${c.n_chains}</td>
    <td class="tm-val" style="color:${tmColor(c.tm_pool)}">${fmt(c.tm_pool)}</td>
    <td class="tm-val" style="color:${tmColor(c.tm_pair)}">${fmt(c.tm_pair)}</td>
    <td class="r mono">${fmt(c.rmsd_pool,2)}</td>
    <td class="r mono">${fmt(c.rmsd_pair,2)}</td>
  `;
  tr.onclick = () => selectComplex(ac);
  tbody.appendChild(tr);
});
 
/* ── viewer state ─────────────────────────────────────────────── */
let viewMode   = 'overlay';
let viewerO    = null;          // overlay
let viewerL    = null;          // left dual-panel  (SBS pool / IP pool input)
let viewerR    = null;          // right dual-panel (SBS pair / IP pair input)
let dualInited = false;
 
let rawRef=null, rawPool=null, rawPair=null, rawFmt='pdb';
let rawPoolInput=null, rawPairInput=null;
 
let modORef=null, modOPool=null, modOPair=null;
let modL=null, modR=null;   // current left / right models (depend on mode)
 
let curAc = null;
 
/* ── viewer init ──────────────────────────────────────────────── */
function initOverlay() {
  viewerO = $3Dmol.createViewer('viewer-overlay', { backgroundColor:'white' });
}
function initDual() {
  if (dualInited) return;
  viewerL   = $3Dmol.createViewer('viewer-left',  { backgroundColor:'white' });
  viewerR   = $3Dmol.createViewer('viewer-right', { backgroundColor:'white' });
  dualInited = true;
}
 
/* ── chain colours for IP mode ────────────────────────────────── */
function chainColorsFromText(pdbText) {
  /* collect unique chain letters in order of first appearance */
  const chains = [];
  for (const line of pdbText.split('\n')) {
    if ((line.startsWith('ATOM') || line.startsWith('HETATM')) && line.length > 21) {
      const ch = line[21];
      if (!chains.includes(ch)) chains.push(ch);
    }
  }
  return chains;
}
 
function buildChainLegend(elId, chains) {
  const el = document.getElementById(elId);
  el.innerHTML = chains.map((ch, i) =>
    `<div class="chain-swatch" style="background:${CC[i % CC.length]}"></div>
     <span>Chain ${ch}</span>`
  ).join('');
}
 
function applyChainStyle(viewer, model, chains, opacity) {
  chains.forEach((ch, i) => {
    viewer.setStyle(
      { model, chain: ch },
      { cartoon: { color: CC[i % CC.length], opacity } }
    );
  });
}
 
/* ── layer updates ────────────────────────────────────────────── */
function updateOverlayLayers() {
  if (!viewerO) return;
  const showRef  = document.getElementById('chk-ref').checked;
  const showPool = document.getElementById('chk-pool').checked;
  const showPair = document.getElementById('chk-pair').checked;
  const opRef    = +document.getElementById('op-ref').value;
  const opPool   = +document.getElementById('op-pool').value;
  const opPair   = +document.getElementById('op-pair').value;
  if (modORef)  viewerO.setStyle({model:modORef},  showRef  ? {cartoon:{color:COL_REF,  opacity:opRef}}  : {});
  if (modOPool) viewerO.setStyle({model:modOPool}, showPool ? {cartoon:{color:COL_POOL, opacity:opPool}} : {});
  if (modOPair) viewerO.setStyle({model:modOPair}, showPair ? {cartoon:{color:COL_PAIR, opacity:opPair}} : {});
  viewerO.render();
}
 
function updateSBSLayers() {
  if (!viewerL || !viewerR) return;
  const opRefL = +document.getElementById('sbs-op-ref-l').value;
  const opPool = +document.getElementById('sbs-op-pool').value;
  const opRefR = +document.getElementById('sbs-op-ref-r').value;
  const opPair = +document.getElementById('sbs-op-pair').value;
  const showRefL = document.getElementById('sbs-chk-ref-l').checked;
  const showPool = document.getElementById('sbs-chk-pool').checked;
  const showRefR = document.getElementById('sbs-chk-ref-r').checked;
  const showPair = document.getElementById('sbs-chk-pair').checked;
  if (modL?.ref)  viewerL.setStyle({model:modL.ref},  showRefL ? {cartoon:{color:COL_REF,  opacity:opRefL}} : {});
  if (modL?.pred) viewerL.setStyle({model:modL.pred}, showPool ? {cartoon:{color:COL_POOL, opacity:opPool}} : {});
  viewerL.render();
  if (modR?.ref)  viewerR.setStyle({model:modR.ref},  showRefR ? {cartoon:{color:COL_REF,  opacity:opRefR}} : {});
  if (modR?.pred) viewerR.setStyle({model:modR.pred}, showPair ? {cartoon:{color:COL_PAIR, opacity:opPair}} : {});
  viewerR.render();
}
 
function updateIPLayers() {
  if (!viewerL || !viewerR) return;
  const opL = +document.getElementById('ip-op-l').value;
  const opR = +document.getElementById('ip-op-r').value;
  if (modL?.chains && modL?.pred) applyChainStyle(viewerL, modL.pred, modL.chains, opL);
  viewerL.render();
  if (modR?.chains && modR?.pred) applyChainStyle(viewerR, modR.pred, modR.chains, opR);
  viewerR.render();
}
 
/* ── load into viewers ────────────────────────────────────────── */
function loadOverlay() {
  if (!viewerO) return;
  viewerO.removeAllModels();
  modORef = modOPool = modOPair = null;
  if (rawRef)  modORef  = viewerO.addModel(rawRef,  rawFmt);
  if (rawPool) modOPool = viewerO.addModel(rawPool, 'pdb');
  if (rawPair) modOPair = viewerO.addModel(rawPair, 'pdb');
  updateOverlayLayers();
  viewerO.zoomTo();
}
 
function loadSBS() {
  if (!viewerL || !viewerR) return;
  viewerL.removeAllModels(); viewerR.removeAllModels();
  modL = {}; modR = {};
  if (rawRef)  modL.ref  = viewerL.addModel(rawRef,  rawFmt);
  if (rawPool) modL.pred = viewerL.addModel(rawPool, 'pdb');
  if (rawRef)  modR.ref  = viewerR.addModel(rawRef,  rawFmt);
  if (rawPair) modR.pred = viewerR.addModel(rawPair, 'pdb');
  updateSBSLayers();
  viewerL.zoomTo(); viewerR.zoomTo();
}
 
function loadIP() {
  if (!viewerL || !viewerR) return;
  viewerL.removeAllModels(); viewerR.removeAllModels();
  modL = {}; modR = {};
 
  if (rawPoolInput) {
    modL.pred   = viewerL.addModel(rawPoolInput, 'pdb');
    modL.chains = chainColorsFromText(rawPoolInput);
    buildChainLegend('ip-legend-l', modL.chains);
    applyChainStyle(viewerL, modL.pred, modL.chains, 1);
    viewerL.zoomTo();
  }
  if (rawPairInput) {
    modR.pred   = viewerR.addModel(rawPairInput, 'pdb');
    modR.chains = chainColorsFromText(rawPairInput);
    buildChainLegend('ip-legend-r', modR.chains);
    applyChainStyle(viewerR, modR.pred, modR.chains, 1);
    viewerR.zoomTo();
  }
  viewerL.render(); viewerR.render();
}
 
/* ── mode switching ───────────────────────────────────────────── */
function setMode(mode) {
  viewMode = mode;
 
  ['overlay','sbs','ip'].forEach(m =>
    document.getElementById('tab-' + m).classList.toggle('active', m === mode)
  );
 
  document.getElementById('overlay-controls').style.display =
    mode === 'overlay' ? '' : 'none';
  document.getElementById('overlay-view').style.display =
    mode === 'overlay' ? '' : 'none';
  document.getElementById('dual-view').style.display =
    mode !== 'overlay' ? 'flex' : 'none';
 
  /* swap inner controls inside each sbs-panel header */
  const showSBS = mode === 'sbs';
  const showIP  = mode === 'ip';
  document.getElementById('sbs-ctrl-l').style.display = showSBS ? '' : 'none';
  document.getElementById('sbs-ctrl-r').style.display = showSBS ? '' : 'none';
  document.getElementById('ip-ctrl-l').style.display  = showIP  ? '' : 'none';
  document.getElementById('ip-ctrl-r').style.display  = showIP  ? '' : 'none';
 
  if (mode !== 'overlay') {
    if (!dualInited) {
      initDual();
      /* first time: load whichever data we have */
      if (rawPool || rawPair || rawPoolInput || rawPairInput) {
        mode === 'sbs' ? loadSBS() : loadIP();
      }
    } else {
      /* already inited: switch content */
      mode === 'sbs' ? loadSBS() : loadIP();
    }
    setTimeout(() => { viewerL?.resize(); viewerR?.resize(); }, 50);
  } else {
    setTimeout(() => viewerO?.resize(), 50);
  }
}
 
/* ── spin helper ──────────────────────────────────────────────── */
function spin(id, on, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('on', on);
  if (msg != null) el.textContent = msg;
}
 
/* ── select complex ───────────────────────────────────────────── */
async function selectComplex(ac) {
  if (ac === curAc) return;
  curAc = ac;
  const c = COMPLEXES[ac];
 
  document.querySelectorAll('#cpx-tbody tr')
    .forEach(tr => tr.classList.toggle('selected', tr.dataset.ac === ac));
 
  document.getElementById('vwr-title').textContent = c.uniprot_id + ' ×' + c.n_chains;
  document.getElementById('vwr-pdb').textContent   = (c.pdb_id || '').toUpperCase();
 
  const fmtChip = (tm, rmsd) =>
    `TM <b>${fmt(tm)}</b>&nbsp;&nbsp;RMSD <b>${fmt(rmsd,2)}</b>`;
  document.getElementById('chip-pool').innerHTML     = fmtChip(c.tm_pool,  c.rmsd_pool);
  document.getElementById('chip-pair').innerHTML     = fmtChip(c.tm_pair,  c.rmsd_pair);
  document.getElementById('chip-pool-sbs').innerHTML = fmtChip(c.tm_pool,  c.rmsd_pool);
  document.getElementById('chip-pair-sbs').innerHTML = fmtChip(c.tm_pair,  c.rmsd_pair);
 
  spin('ov-main',  true, 'Loading…');
  if (dualInited) { spin('ov-left', true, 'Loading…'); spin('ov-right', true, 'Loading…'); }
 
  rawRef = rawPool = rawPair = rawPoolInput = rawPairInput = null;
  rawFmt = c.ref_fmt;
 
  try {
    await Promise.all([
      c.ref_gz         && ungzip(c.ref_gz).then(t         => { rawRef       = t; }),
      c.pool_gz        && ungzip(c.pool_gz).then(t        => { rawPool      = t; }),
      c.pair_gz        && ungzip(c.pair_gz).then(t        => { rawPair      = t; }),
      c.pool_input_gz  && ungzip(c.pool_input_gz).then(t  => { rawPoolInput = t; }),
      c.pair_input_gz  && ungzip(c.pair_input_gz).then(t  => { rawPairInput = t; }),
    ].filter(Boolean));
 
    loadOverlay();
    if (dualInited) {
      viewMode === 'sbs' ? loadSBS() : loadIP();
    }
 
    spin('ov-main', false);
    if (dualInited) { spin('ov-left', false); spin('ov-right', false); }
  } catch(e) {
    console.error(e);
    spin('ov-main', true, 'Error: ' + e.message);
  }
}
 
/* ── init ─────────────────────────────────────────────────────── */
initOverlay();
if (acs.length) selectComplex(acs[0]);
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
print(f"\n  Written : {OUT}")
print(f"  Size    : {size_kb:.0f} KB  (~{size_kb/1024:.1f} MB)")
 