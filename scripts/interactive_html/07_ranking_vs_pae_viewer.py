#!/usr/bin/env python3
"""
Self-contained HTML viewer: compare two CombFold input-model selection metrics
(default: ranking score  vs  PAE) on the benchmark complexes (n_proteins > 2).

Layout (like 05_homomultimer_viewer.py)
  Left  : sortable table of all complexes, TM / CF confidence for both metrics
  Right : two linked 3Dmol windows
            left window  = metric A (ranking score)
            right window = metric B (PAE)
          Mode "Assembled complex":
            best-CF-confidence CombFold output (US-aligned, taken from
            usalign_outputs_predX/complex/usalign.pdb — no re-alignment)
            overlaid on the reference PDB.  Each window header shows TM / RMSD /
            CF of that model, a reference toggle, reference-saturation and
            reference-opacity sliders.
          Mode "Input pairs":
            the AF pair models CombFold used as input for each metric
            (<complex>_input/pdbs/), pair selectable from a dropdown.

Structures are cleaned with gemmi (no waters/ligands/nucleic acids, backbone
only by default), gzip+base64 embedded -> run ON THE CLUSTER, download the HTML.
"""

import base64
import gzip
import json
import re
from pathlib import Path

import gemmi
import polars as pl
from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"
OUT = data_dir / "Pipeline/viewer/06_ranking_vs_pae_viewer.html"

# Which two pipelines to compare: (key, pipeline folder, display label).
# Swap in "16_CP_model_based_on_iptm" / "17_CP_model_based_on_ptm_avg" to compare others.
LEFT = ("rank", "10_all_CP_complexes", "Ranking score")
RIGHT = ("pae", "18_CP_model_based_on_pae", "PAE")

BACKBONE_ONLY = True        # cartoon only needs N/CA/C/O -> much smaller HTML
EMBED_INPUT_PAIRS = True    # set False for a smaller file without the input-pair mode
MAX_INPUT_PAIRS = 8         # max AF input pair models embedded per complex (per window)


# ── loading (same logic as the notebook) ──────────────────────────────────────

def add_confidence(df: pl.DataFrame) -> pl.DataFrame:
    conf = {}
    for d in {Path(p).parent for p in df["combfold_output_path"]}:
        with open(d / "confidence.txt") as f:
            conf.update({k: float(v) for k, v in (line.split() for line in f if line.strip())})
    return df.with_columns(
        pl.col("combfold_output_path")
        .replace_strict(conf, return_dtype=pl.Float64)
        .alias("CF_confidence")
    )


def load(folder: str) -> pl.DataFrame:
    df = add_confidence(
        pl.read_parquet(data_dir / "Pipeline" / folder / "cf_pdb_structure_similarity/cf_pdb_eval_all_metrics.parquet")
    )
    return (
        df.filter(pl.col("n_proteins") > 2)
        .sort("CF_confidence", descending=True)
        .group_by("complex_ac")
        .first()
    )


# ── structure helpers ─────────────────────────────────────────────────────────

def compress(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode())).decode()


def strip_pdb(text: str) -> str:
    keep = {"ATOM", "HETATM", "MODEL", "ENDMDL", "TER", "END"}
    return "\n".join(l for l in text.splitlines() if l[:6].strip() in keep)


def clean_structure(path: Path, model_name: str | None = None) -> tuple[str, str]:
    """Return (text, fmt) of a cleaned structure: one model, protein only, optionally backbone only."""
    st = gemmi.read_structure(str(path))
    # keep a single model (the one requested, else the first)
    if len(st) > 1:
        idx = next((i for i, m in enumerate(st) if model_name and m.name == str(model_name)), 0)
        for i in reversed(range(len(st))):
            if i != idx:
                del st[i]
    st.setup_entities()
    st.remove_ligands_and_waters()
    for model in st:
        drop = [
            i for i, ch in enumerate(model)
            if ch.get_polymer().check_polymer_type()
            in (gemmi.PolymerType.Dna, gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid)
        ]
        for i in reversed(drop):
            del model[i]
    st.remove_empty_chains()
    if BACKBONE_ONLY:
        keep = {"N", "CA", "C", "O"}
        for model in st:
            for chain in model:
                for res in chain:
                    for i in reversed([i for i, a in enumerate(res) if a.name not in keep]):
                        del res[i]
    # PDB format can't hold multi-character chain IDs (big assemblies) -> use mmCIF then
    if any(len(ch.name) > 1 for ch in st[0]):
        return st.make_mmcif_document().as_string(), "cif"
    return st.make_pdb_string(), "pdb"


def embed_structure(path: Path | None, label: str, model_name: str | None = None) -> dict | None:
    if path is None or not path.exists():
        print(f"    WARNING not found [{label}]: {path}")
        return None
    try:
        text, fmt = clean_structure(path, model_name)
    except Exception as e:  # fall back to the raw file
        print(f"    WARNING gemmi failed on {label} ({e}); embedding raw file")
        fmt = "cif" if path.suffix == ".cif" else "pdb"
        text = path.read_text()
        text = strip_pdb(text) if fmt == "pdb" else text
    return {"gz": compress(text), "fmt": fmt}


def input_pdbs_dir(cf_result_dir: Path) -> Path | None:
    """…/CombFold/<name>_pool_output  ->  …/CombFold/<name>_pool_input/pdbs"""
    cand = cf_result_dir.parent / (re.sub(r"_output$", "_input", cf_result_dir.name)) / "pdbs"
    if cand.exists():
        return cand
    print(f"    WARNING input pdbs dir not found: {cand}")
    return None


def find_pair_files(pdbs_dir: Path, a: str, b: str) -> list[Path]:
    """AF pair models for proteins a,b (either order; handles AFM_A_B_*.pdb and chunked names)."""
    files = sorted(p for p in pdbs_dir.iterdir() if p.suffix in (".pdb", ".cif"))
    pat = re.compile(rf"{re.escape(a)}.*{re.escape(b)}|{re.escape(b)}.*{re.escape(a)}")
    return [p for p in files if pat.search(p.name)]


def side_payload(row: dict, key: str) -> dict:
    pred_dir = Path(row["usalign_pred_dir"])
    aligned = pred_dir / "complex" / "usalign.pdb"
    return {
        "tm": row["usalign_cpx_tm_score"],
        "rmsd": row["usalign_cpx_rmsd"],
        "cf": row["CF_confidence"],
        "pred_name": pred_dir.name.replace("usalign_outputs_", ""),
        "cf_output": Path(row["combfold_output_path"]).name,
        "chain_mapping": row["chain_mapping"],
        "n_pairs_used": row["n_pairs_used"],
        "pred": embed_structure(aligned, f"{row['complex_ac']} {key} aligned pred"),
        "pairs": {},
    }


def available_pair_files(row: dict) -> dict[str, Path]:
    """pair -> first (best-ranked) AF model file that actually exists in <name>_input/pdbs.
    Pairs from pairs_used without a file (e.g. homo self-pairs not run as AFM) are skipped."""
    pdbs = input_pdbs_dir(Path(row["Combfold_result_path"]))
    if pdbs is None:
        return {}
    try:
        pairs = json.loads(row["pairs_used"].replace("'", '"'))
    except Exception:
        pairs = []
    found = {}
    for pair in pairs:
        a, b = pair.split("_", 1)
        files = find_pair_files(pdbs, a, b)
        if files:
            found[pair] = files[0]          # sorted -> rank_1 first
    missing = [p for p in pairs if p not in found]
    if missing:
        print(f"    note: no AF input file for {len(missing)} listed pair(s) (skipped): {', '.join(missing)}")
    return found


def embed_input_pairs(ac: str, sides: dict[str, dict]) -> None:
    """Embed at most MAX_INPUT_PAIRS pairs per complex (same pairs in both windows where possible)."""
    avail = {s: available_pair_files(row) for s, row in sides.items()}
    common = sorted(set.intersection(*(set(v) for v in avail.values()))) if avail else []
    rest = sorted(set().union(*(set(v) for v in avail.values())) - set(common))
    chosen = (common + rest)[:MAX_INPUT_PAIRS]
    n_total = len(common) + len(rest)
    if n_total > MAX_INPUT_PAIRS:
        print(f"    capping input pairs: {MAX_INPUT_PAIRS}/{n_total} embedded")
    for s, files in avail.items():
        for pair in chosen:
            f = files.get(pair)
            e = embed_structure(f, f"{ac} {s} {f.name}") if f else None
            EMBED[ac][s]["pairs"][pair] = [{"file": f.name, **e}] if e else []


# ── main ──────────────────────────────────────────────────────────────────────

(lk, lfolder, llabel), (rk, rfolder, rlabel) = LEFT, RIGHT
dfL, dfR = load(lfolder), load(rfolder)
print(f"{llabel}: {dfL.height} complexes   {rlabel}: {dfR.height} complexes  (expected 41)")

rowsL = {r["complex_ac"]: r for r in dfL.iter_rows(named=True)}
rowsR = {r["complex_ac"]: r for r in dfR.iter_rows(named=True)}
only_l, only_r = set(rowsL) - set(rowsR), set(rowsR) - set(rowsL)
if only_l or only_r:
    print(f"  only in {llabel}: {sorted(only_l)}\n  only in {rlabel}: {sorted(only_r)}")

EMBED: dict[str, dict] = {}
for ac in sorted(set(rowsL) | set(rowsR)):
    base = rowsL.get(ac) or rowsR.get(ac)
    print(f"  {ac}  ({base['pdb_id']}, {base['n_proteins']} proteins)")
    EMBED[ac] = {
        "complex_ac": ac,
        "pdb_id": base["pdb_id"],
        "n_proteins": base["n_proteins"],
        "identifiers": base["identifiers"],
        "ref": embed_structure(Path(base["reference_pdb_path"]), f"{ac} ref", base["reference_pdb_model"]),
        "L": side_payload(rowsL[ac], lk) if ac in rowsL else None,
        "R": side_payload(rowsR[ac], rk) if ac in rowsR else None,
    }
    if EMBED_INPUT_PAIRS:
        sides = {s: r[ac] for s, r in (("L", rowsL), ("R", rowsR)) if ac in r}
        embed_input_pairs(ac, sides)

META = {"L": {"label": llabel, "short": lk, "folder": lfolder},
        "R": {"label": rlabel, "short": rk, "folder": rfolder}}
print(f"\nEmbedded {len(EMBED)} complexes")


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>%%TITLE%%</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-size:13px; color:#242424;
       background:#f0f0f0; height:100vh; display:flex; flex-direction:column; gap:8px; padding:10px; }
.topbar { background:white; border:1px solid #ddd; border-radius:6px; padding:9px 14px;
          display:flex; align-items:center; gap:14px; flex-shrink:0; flex-wrap:wrap; }
.topbar h2 { font-size:14px; font-weight:700; }
.topbar .hint { font-size:12px; color:#999; }
.summary { font-size:12px; color:#444; font-family:monospace; }
.main { display:flex; gap:8px; flex:1; min-height:0; }

/* table */
.table-panel { width:500px; flex-shrink:0; background:white; border:1px solid #ddd; border-radius:6px;
               display:flex; flex-direction:column; overflow:hidden; }
.table-panel-hdr { padding:7px 12px; border-bottom:1px solid #eee; font-size:11px; font-weight:600; color:#666; }
.table-wrap { overflow-y:auto; flex:1; }
table { width:100%; border-collapse:collapse; font-size:12px; }
thead th { position:sticky; top:0; background:#f7f7f7; z-index:1; padding:6px 6px; font-weight:600;
           font-size:11px; color:#444; border-bottom:1px solid #ddd; white-space:nowrap; cursor:pointer; user-select:none; }
thead th:hover { background:#ececec; }
thead th.r, td.r { text-align:right; }
tbody tr { cursor:pointer; }
tbody tr:hover { background:#f0f5ff; }
tbody tr.selected { background:#ddeeff; }
tbody td { padding:5px 6px; border-bottom:1px solid #f2f2f2; white-space:nowrap; }
.mono { font-family:monospace; }
.tm-val { font-family:monospace; font-weight:700; text-align:right; }

/* viewer panel */
.viewer-panel { flex:1; min-width:0; background:white; border:1px solid #ddd; border-radius:6px;
                display:flex; flex-direction:column; overflow:hidden; }
.viewer-hdr { padding:8px 14px; border-bottom:1px solid #eee; display:flex; align-items:center;
              gap:12px; flex-shrink:0; flex-wrap:wrap; }
.viewer-hdr h3 { font-size:13px; font-weight:700; }
.badge { display:inline-block; padding:2px 9px; border-radius:10px; font-size:11px; font-weight:600;
         background:#eee; border:1px solid #ccc; font-family:monospace; }
.tabs { display:flex; gap:3px; }
.tab { background:#f0f0f0; border:1px solid #ccc; border-radius:4px; cursor:pointer; padding:2px 10px;
       font-size:12px; font-weight:600; line-height:1.6; }
.tab:hover:not(.active) { background:#e4e4e4; }
.tab.active { background:#0072B2; color:white; border-color:#0072B2; }
.opt { font-size:12px; display:flex; align-items:center; gap:4px; }
select { font-size:12px; padding:1px 4px; }

.dual { flex:1; min-height:0; display:flex; }
.panel { flex:1; min-width:0; display:flex; flex-direction:column; overflow:hidden; }
.panel + .panel { border-left:2px solid #e8e8e8; }
.phdr { padding:6px 10px; border-bottom:1px solid #eee; background:#fafafa; flex-shrink:0;
        display:flex; flex-direction:column; gap:5px; }
.prow { display:flex; align-items:center; gap:10px; flex-wrap:wrap; min-height:22px; }
.plabel { font-size:13px; font-weight:700; }
.pbody { flex:1; position:relative; min-height:0; }
.pbody > .vw { width:100%; height:100%; position:absolute; inset:0; }
.spin { position:absolute; inset:0; background:rgba(255,255,255,.85); display:none; align-items:center;
        justify-content:center; font-size:13px; color:#666; text-align:center; padding:20px; z-index:5; }
.spin.on { display:flex; }
.chip { font-size:12px; font-family:monospace; color:#333; background:#f5f5f5; border:1px solid #e2e2e2;
        border-radius:4px; padding:2px 8px; white-space:nowrap; }
.chip b { font-weight:700; }
.lc { display:flex; align-items:center; gap:5px; font-size:12px; white-space:nowrap; }
.sw { width:10px; height:10px; border-radius:2px; border:1px solid rgba(0,0,0,.15); flex-shrink:0; }
.sl { width:75px; cursor:pointer; accent-color:#555; }
.legend { display:flex; gap:6px; align-items:center; font-size:11px; color:#555; flex-wrap:wrap; }
</style>
</head>
<body>

<div class="topbar">
  <h2>%%TITLE%%</h2>
  <span class="summary" id="summary"></span>
  <span class="hint">Click a row to load · best-CF-confidence model per complex · both windows share the reference frame</span>
</div>

<div class="main">
  <div class="table-panel">
    <div class="table-panel-hdr" id="tbl-hdr">Complexes — click a column header to sort</div>
    <div class="table-wrap">
      <table>
        <thead><tr id="thead-row"></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="viewer-panel">
    <div class="viewer-hdr">
      <div class="tabs">
        <button class="tab active" id="tab-cpx" onclick="setMode('cpx')">Assembled complex</button>
        <button class="tab" id="tab-ip" onclick="setMode('ip')">Input pairs</button>
      </div>
      <h3 id="title">—</h3>
      <span class="badge" id="pdbbadge">—</span>
      <span class="opt" id="colormode-ctrl">Colour:
        <select id="colormode" onchange="restyleAll()">
          <option value="flat">Ref blue / pred flat</option>
          <option value="prot">By protein</option>
        </select>
      </span>
      <span class="opt" id="pair-ctrl" style="display:none">Pair:
        <select id="pairsel" onchange="curPair=this.value; loadAll()"></select>
      </span>
      <label class="opt"><input type="checkbox" id="sync" checked> sync rotation</label>
    </div>

    <div class="dual">
      %%PANELS%%
    </div>
  </div>
</div>

<script>
const D = %%JSON%%;
const META = %%META%%;
const SIDES = ['L','R'];
const COL_REF = '#0072B2';
const COL_PRED = {L:'#E69F00', R:'#009E73'};
const CC = ['#E69F00','#0072B2','#009E73','#CC79A7','#56B4E9','#D55E00','#F0E442','#999999'];
const HUES = [0,210,40,130,280,20,170,320,60,250];

/* ── utils ───────────────────────────────────────────────────── */
async function ungzip(b64) {
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).text();
}
const TXT = new Map();                     // decompression cache, keyed by object
async function text(obj) {
  if (!obj) return null;
  if (!TXT.has(obj)) TXT.set(obj, await ungzip(obj.gz));
  return TXT.get(obj);
}
function fmt(v, d=3) { return (v == null || Number.isNaN(v)) ? '—' : v.toFixed(d); }
function tmColor(v) {
  if (v == null) return '#aaa';
  const t = Math.min(1, Math.max(0, v));
  return t < 0.5 ? `rgb(210,${Math.round(t*2*180)},40)` : `rgb(${Math.round((1-(t-0.5)*2)*210)},170,40)`;
}
function hexToRgb(h) { h = h.replace('#',''); return [0,2,4].map(i => parseInt(h.substr(i,2),16)); }
function rgbToHex(c) { return '#' + c.map(x => Math.round(x).toString(16).padStart(2,'0')).join(''); }
/* desaturate: blend towards light grey; sat=1 -> original colour, sat=0 -> grey */
function fade(hex, sat) {
  const g = [200,200,200];
  return rgbToHex(hexToRgb(hex).map((x,i) => g[i] + (x - g[i]) * sat));
}
function hsl(h, s, l) {
  s/=100; l/=100;
  const k = n => (n + h/30) % 12, a = s*Math.min(l,1-l);
  const f = n => l - a*Math.max(-1, Math.min(k(n)-3, Math.min(9-k(n), 1)));
  return rgbToHex([f(0),f(8),f(4)].map(x => 255*x));
}
/* per-protein colours: same hue per UniProt in both windows; ref pale, pred vivid */
function protColors(mappingJson) {
  const pred = {}, ref = {}, legend = [];
  let m; try { m = JSON.parse(mappingJson); } catch(e) { return {pred, ref, legend}; }
  Object.keys(m).sort().forEach((p, i) => {
    const hue = HUES[i % HUES.length], ph = hsl(hue,70,45), rh = hsl(hue,45,70);
    const [cf, rf] = m[p];
    cf.forEach(c => pred[c] = ph); rf.forEach(c => ref[c] = rh);
    legend.push([p, ph]);
  });
  return {pred, ref, legend};
}
function chainsIn(txt, fmtName) {
  const chains = [];
  if (fmtName === 'pdb') {
    for (const l of txt.split('\n'))
      if ((l.startsWith('ATOM') || l.startsWith('HETATM')) && l.length > 21 && !chains.includes(l[21])) chains.push(l[21]);
  }
  return chains;
}
function el(id) { return document.getElementById(id); }
function spin(s, on, msg) { const e = el('spin-'+s); e.classList.toggle('on', on); if (msg != null) e.textContent = msg; }

/* ── summary ─────────────────────────────────────────────────── */
(function summary() {
  const acs = Object.keys(D);
  const both = acs.filter(a => D[a].L && D[a].R);
  const mean = s => both.reduce((t,a) => t + D[a][s].tm, 0) / both.length;
  const w = s => both.filter(a => D[a][s].tm > D[a][s==='L'?'R':'L'].tm + 1e-9).length;
  el('summary').innerHTML =
    `n=${acs.length} · mean TM ${META.L.label} <b>${fmt(mean('L'))}</b> vs ${META.R.label} <b>${fmt(mean('R'))}</b>` +
    ` · better: ${META.L.label} ${w('L')} / ${META.R.label} ${w('R')} / tie ${both.length - w('L') - w('R')}`;
})();

/* ── table ───────────────────────────────────────────────────── */
const COLS = [
  {k:'complex_ac', t:'Complex', v:c => c.complex_ac, cls:'mono'},
  {k:'pdb', t:'PDB', v:c => c.pdb_id, cls:'mono'},
  {k:'n', t:'N', v:c => c.n_proteins, cls:'r'},
  {k:'tmL', t:'TM '+META.L.short, v:c => c.L?.tm, tm:true},
  {k:'tmR', t:'TM '+META.R.short, v:c => c.R?.tm, tm:true},
  {k:'dtm', t:'ΔTM', v:c => (c.L && c.R) ? c.R.tm - c.L.tm : null, d:true},
  {k:'cfL', t:'CF '+META.L.short, v:c => c.L?.cf, cls:'r mono', dig:1},
  {k:'cfR', t:'CF '+META.R.short, v:c => c.R?.cf, cls:'r mono', dig:1},
];
let sortKey = 'dtm', sortDesc = true;
function buildTable() {
  el('thead-row').innerHTML = COLS.map(c =>
    `<th class="${c.cls?.includes('r')||c.tm||c.d?'r':''}" data-k="${c.k}">${c.t}${c.k===sortKey?(sortDesc?' ▼':' ▲'):''}</th>`).join('');
  el('thead-row').querySelectorAll('th').forEach(th => th.onclick = () => {
    if (sortKey === th.dataset.k) sortDesc = !sortDesc; else { sortKey = th.dataset.k; sortDesc = true; }
    buildTable();
  });
  const col = COLS.find(c => c.k === sortKey);
  const acs = Object.keys(D).sort((a,b) => {
    let x = col.v(D[a]), y = col.v(D[b]);
    if (x == null) return 1; if (y == null) return -1;
    const r = (typeof x === 'string') ? x.localeCompare(y) : x - y;
    return sortDesc ? -r : r;
  });
  el('tbody').innerHTML = '';
  acs.forEach(ac => {
    const c = D[ac], tr = document.createElement('tr');
    tr.dataset.ac = ac; if (ac === curAc) tr.classList.add('selected');
    tr.innerHTML = COLS.map(col => {
      const v = col.v(c);
      if (col.tm) return `<td class="tm-val" style="color:${tmColor(v)}">${fmt(v)}</td>`;
      if (col.d)  return `<td class="tm-val" style="color:${v==null?'#aaa':v>0.005?COL_PRED.R:v<-0.005?COL_PRED.L:'#888'}">${v==null?'—':(v>0?'+':'')+v.toFixed(3)}</td>`;
      if (col.dig != null) return `<td class="${col.cls}">${fmt(v, col.dig)}</td>`;
      return `<td class="${col.cls||''}" style="font-size:11px">${v ?? '—'}</td>`;
    }).join('');
    tr.onclick = () => selectComplex(ac);
    el('tbody').appendChild(tr);
  });
  el('tbl-hdr').textContent = `Complexes (${acs.length}) — ΔTM = ${META.R.label} − ${META.L.label}; click a header to sort`;
}

/* ── state & viewers ─────────────────────────────────────────── */
let mode = 'cpx', curAc = null, curPair = null;
const V = {}, M = {L:{}, R:{}};
let syncing = false;
SIDES.forEach(s => {
  V[s] = $3Dmol.createViewer(el('viewer-'+s), {backgroundColor:'white'});
});
function linkViews() {
  SIDES.forEach(s => {
    const o = s === 'L' ? 'R' : 'L';
    if (typeof V[s].setViewChangeCallback === 'function') {
      V[s].setViewChangeCallback(view => {
        if (syncing || !el('sync').checked) return;
        syncing = true; V[o].setView(view); syncing = false;
      });
    }
  });
}
linkViews();

/* ── styling ─────────────────────────────────────────────────── */
function styleCpx(s) {
  const v = V[s], m = M[s], c = D[curAc], side = c?.[s];
  if (!side) return;
  const showRef = el(`chk-ref-${s}`).checked, sat = +el(`sat-ref-${s}`).value, op = +el(`op-ref-${s}`).value;
  const showPred = el(`chk-pred-${s}`).checked, opP = +el(`op-pred-${s}`).value;
  const byProt = el('colormode').value === 'prot';
  const pc = byProt ? protColors(side.chain_mapping) : null;
  if (m.ref) {
    v.setStyle({model:m.ref}, showRef ? {cartoon:{color:fade(byProt?'#bbbbbb':COL_REF, sat), opacity:op}} : {});
    if (showRef && byProt) for (const [ch,col] of Object.entries(pc.ref))
      v.setStyle({model:m.ref, chain:ch}, {cartoon:{color:fade(col, sat), opacity:op}});
  }
  if (m.pred) {
    v.setStyle({model:m.pred}, showPred ? {cartoon:{color:byProt?'#888888':COL_PRED[s], opacity:opP}} : {});
    if (showPred && byProt) for (const [ch,col] of Object.entries(pc.pred))
      v.setStyle({model:m.pred, chain:ch}, {cartoon:{color:col, opacity:opP}});
  }
  el(`legend-${s}`).innerHTML = byProt
    ? pc.legend.map(([p,col]) => `<div class="sw" style="background:${col}"></div><span>${p}</span>`).join('')
    : '';
  v.render();
}
function styleIP(s) {
  const v = V[s], m = M[s];
  if (!m.pred) return;
  const op = +el(`op-ip-${s}`).value;
  if (m.chains.length) m.chains.forEach((ch,i) => v.setStyle({model:m.pred, chain:ch}, {cartoon:{color:CC[i%CC.length], opacity:op}}));
  else v.setStyle({model:m.pred}, {cartoon:{color:'spectrum', opacity:op}});
  v.render();
}
function restyle(s) { mode === 'cpx' ? styleCpx(s) : styleIP(s); }
function restyleAll() { SIDES.forEach(restyle); }

/* ── loading ─────────────────────────────────────────────────── */
async function loadCpx(s) {
  const v = V[s], c = D[curAc], side = c[s];
  v.removeAllModels(); M[s] = {};
  if (!side) { spin(s, true, `${META[s].label}: no model for this complex`); v.render(); return; }
  const [ref, pred] = await Promise.all([text(c.ref), text(side.pred)]);
  if (ref)  M[s].ref  = v.addModel(ref, c.ref.fmt);
  if (pred) M[s].pred = v.addModel(pred, side.pred.fmt);
  styleCpx(s);
  v.zoomTo(); v.render();
  spin(s, !pred, !pred ? 'US-aligned prediction not embedded (see build log)' : null);
}
async function loadIP(s) {
  const v = V[s], c = D[curAc], side = c[s];
  v.removeAllModels(); M[s] = {};
  const files = side?.pairs?.[curPair] || [];
  const fsel = el(`filesel-${s}`);
  const prev = fsel.value;
  fsel.innerHTML = files.map((f,i) => `<option value="${i}">${f.file}</option>`).join('');
  if (files[prev]) fsel.value = prev;
  fsel.style.display = files.length > 1 ? '' : 'none';
  el(`ipfile-${s}`).textContent = files.length === 1 ? files[0].file : (files.length ? `${files.length} files` : '');
  if (!files.length) { spin(s, true, `${META[s].label}: no input model for ${curPair || 'this pair'}`); v.render(); return; }
  const f = files[+fsel.value || 0];
  const t = await text(f);
  M[s].pred = v.addModel(t, f.fmt);
  M[s].chains = chainsIn(t, f.fmt);
  el(`legend-ip-${s}`).innerHTML = M[s].chains.map((ch,i) =>
    `<div class="sw" style="background:${CC[i%CC.length]}"></div><span>Chain ${ch}</span>`).join('');
  styleIP(s);
  v.zoomTo(); v.render();
  spin(s, false);
}
async function loadAll() {
  if (!curAc) return;
  SIDES.forEach(s => spin(s, true, 'Loading…'));
  try {
    await Promise.all(SIDES.map(s => mode === 'cpx' ? loadCpx(s) : loadIP(s)));
    if (el('sync').checked && M.L && Object.keys(M.L).length) { syncing = true; V.R.setView(V.L.getView()); syncing = false; V.R.render(); }
  } catch (e) { console.error(e); SIDES.forEach(s => spin(s, true, 'Error: ' + e.message)); }
}

/* ── header info ─────────────────────────────────────────────── */
function updateHeaders() {
  const c = D[curAc];
  el('title').textContent = `${c.complex_ac} · ${c.n_proteins} proteins`;
  el('pdbbadge').textContent = (c.pdb_id || '').toUpperCase();
  el('pdbbadge').title = c.identifiers;
  SIDES.forEach(s => {
    const x = c[s];
    el(`chip-${s}`).innerHTML = x
      ? `CF <b>${fmt(x.cf,1)}</b> &nbsp; TM <b style="color:${tmColor(x.tm)}">${fmt(x.tm)}</b> &nbsp; RMSD <b>${fmt(x.rmsd,2)}</b>`
      : 'no model';
    el(`sub-${s}`).textContent = x ? `${x.pred_name} · ${x.cf_output} · ${x.n_pairs_used} pairs used` : '';
  });
  // pair dropdown: union of pairs over both metrics
  const pairs = [...new Set(SIDES.flatMap(s => Object.keys(c[s]?.pairs || {})))].sort();
  el('pairsel').innerHTML = pairs.map(p => {
    const miss = SIDES.filter(s => !(c[s]?.pairs?.[p]?.length)).map(s => META[s].short);
    return `<option value="${p}">${p}${miss.length ? '  (missing: ' + miss.join(', ') + ')' : ''}</option>`;
  }).join('');
  if (!pairs.includes(curPair)) curPair = pairs[0] || null;
  if (curPair) el('pairsel').value = curPair;
}

async function selectComplex(ac) {
  if (ac === curAc) return;
  curAc = ac;
  document.querySelectorAll('#tbody tr').forEach(tr => tr.classList.toggle('selected', tr.dataset.ac === ac));
  updateHeaders();
  await loadAll();
}

function setMode(m) {
  mode = m;
  el('tab-cpx').classList.toggle('active', m === 'cpx');
  el('tab-ip').classList.toggle('active', m === 'ip');
  el('pair-ctrl').style.display = m === 'ip' ? '' : 'none';
  el('colormode-ctrl').style.display = m === 'cpx' ? '' : 'none';
  SIDES.forEach(s => {
    el(`cpx-ctrl-${s}`).style.display = m === 'cpx' ? '' : 'none';
    el(`ip-ctrl-${s}`).style.display  = m === 'ip'  ? '' : 'none';
  });
  loadAll();
}

window.addEventListener('resize', () => SIDES.forEach(s => V[s].resize()));
buildTable();
const first = el('tbody').querySelector('tr');
if (first) selectComplex(first.dataset.ac);
</script>
</body>
</html>
"""

PANEL = r"""
<div class="panel">
  <div class="phdr">
    <div class="prow">
      <span class="plabel" style="color:%%COL%%">%%LABEL%%</span>
      <span class="chip" id="chip-%%S%%">—</span>
      <span style="font-size:11px;color:#888" id="sub-%%S%%"></span>
    </div>
    <div class="prow" id="cpx-ctrl-%%S%%">
      <span class="lc">
        <input type="checkbox" id="chk-ref-%%S%%" checked onchange="restyle('%%S%%')">
        <div class="sw" style="background:#0072B2"></div><label for="chk-ref-%%S%%">Reference</label>
      </span>
      <span class="lc">saturation
        <input type="range" class="sl" id="sat-ref-%%S%%" min="0" max="1" step="0.05" value="1" oninput="restyle('%%S%%')">
      </span>
      <span class="lc">opacity
        <input type="range" class="sl" id="op-ref-%%S%%" min="0" max="1" step="0.05" value="1" oninput="restyle('%%S%%')">
      </span>
      <span class="lc" style="margin-left:8px">
        <input type="checkbox" id="chk-pred-%%S%%" checked onchange="restyle('%%S%%')">
        <div class="sw" style="background:%%COL%%"></div><label for="chk-pred-%%S%%">CombFold</label>
        <input type="range" class="sl" id="op-pred-%%S%%" min="0" max="1" step="0.05" value="1" oninput="restyle('%%S%%')">
      </span>
      <div class="legend" id="legend-%%S%%"></div>
    </div>
    <div class="prow" id="ip-ctrl-%%S%%" style="display:none">
      <span style="font-size:11px;color:#555;font-family:monospace" id="ipfile-%%S%%"></span>
      <select id="filesel-%%S%%" style="display:none" onchange="loadIP('%%S%%')"></select>
      <span class="lc">opacity
        <input type="range" class="sl" id="op-ip-%%S%%" min="0" max="1" step="0.05" value="1" oninput="restyle('%%S%%')">
      </span>
      <div class="legend" id="legend-ip-%%S%%"></div>
    </div>
  </div>
  <div class="pbody">
    <div class="vw" id="viewer-%%S%%"></div>
    <div class="spin on" id="spin-%%S%%">Select a complex from the table</div>
  </div>
</div>
"""

panels = "".join(
    PANEL.replace("%%S%%", s).replace("%%LABEL%%", META[s]["label"]).replace("%%COL%%", col)
    for s, col in (("L", "#E69F00"), ("R", "#009E73"))
)
title = f"{llabel} vs {rlabel} — benchmark complexes"
html_out = (
    HTML.replace("%%PANELS%%", panels)
    .replace("%%TITLE%%", title)
    .replace("%%META%%", json.dumps(META))
    .replace("%%JSON%%", json.dumps(EMBED, ensure_ascii=True))
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html_out, encoding="utf-8")
size_mb = OUT.stat().st_size / 1e6
print(f"\n  Written : {OUT}\n  Size    : {size_mb:.1f} MB")