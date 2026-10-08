#!/usr/bin/env python3
"""
Generate a self-contained HTML viewer: PAE (left) vs PAE + pLDDT-trimmed (right).

Complexes (n_proteins > 2, best CombFold output by CF_confidence in each run,
only complexes present in BOTH runs):
  Left  window : run 18  (PAE, no trimming)
  Right window : run 19  (PAE, pLDDT-trimmed)
  Each window  : US-align aligned prediction (<usalign_pred_dir>/complex/usalign.pdb,
                 already rotated onto the reference -> no transformation computed here)
                 + the reference structure, CF confidence, TM-score of that output,
                 and (how many residues of the full complex are in the model).
  Right window additionally shows how much of the complex's full length was trimmed.

Input pairs (mode "Input pairs"): up to MAX_PAIRS pairs per complex (those with the
most residues removed by trimming), left = untrimmed pair pdb (run 18),
right = trimmed pair pdb (run 19). Optional ghost of the removed residues.

Nothing falls back silently: missing files / unexpected data stop the script with an assert.

Speed: pairs are only stripped/compressed after the top MAX_PAIRS were selected, and
complexes are built in parallel threads (file reads + zlib release the GIL).
The visible output is identical to the previous version.
"""

import base64, gzip, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"

RUN_L = data_dir / "Pipeline/18_CP_model_based_on_pae"
RUN_R = data_dir / "Pipeline/19_CP_pae_plddt_trimmed"
SEQ_CSV = data_dir / "iPTM_and_pLDDT/all_yeast_proteins_uniprot_mapped_sequences.csv"
OUT = data_dir / "Pipeline/viewer/07_pae_vs_trimmed_viewer.html"

MAX_PAIRS = 8          # input pairs per complex shown in the "Input pairs" mode
BACKBONE_ONLY = True   # keep only N/CA/C/O atoms of pdb-format files (cartoon only needs these; much smaller html)
N_THREADS = 8

KEEP_RECORDS = frozenset({"ATOM", "HETATM", "MODEL", "ENDMDL", "TER", "END"})
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


# ── helpers ───────────────────────────────────────────────────────────────────

def compress(text: str) -> str:
    # mtime=0 -> reproducible output (no timestamp in the gzip header)
    return base64.b64encode(gzip.compress(text.encode(), mtime=0)).decode()


def strip_pdb(text: str) -> str:
    out = []
    for l in text.splitlines():
        rec = l[:6].strip()
        if rec not in KEEP_RECORDS:
            continue
        if BACKBONE_ONLY and rec in ("ATOM", "HETATM") and l[12:16].strip() not in BACKBONE_ATOMS:
            continue
        out.append(l)
    return "\n".join(out)


def ca_chains(pdb_text: str) -> list[str]:
    """Chain id of every CA atom (one entry per residue), same rule as the length counting."""
    return [
        l[21] for l in pdb_text.splitlines()
        if l.startswith("ATOM") and l[12:16] == " CA " and l[16:17] in (" ", "A")
    ]


def read_chain_list(cf_output_path: Path) -> dict[str, str]:
    """chain letter -> UniProt from CombFold's chain.list (two levels above the assembled pdb)."""
    chain_list = cf_output_path.parent.parent / "_unified_representation" / "assembly_output" / "chain.list"
    assert chain_list.exists(), f"chain.list not found: {chain_list}"
    mapping: dict[str, str] = {}
    for line in chain_list.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pdb$", line)
        assert m, f"unparseable chain.list line {line!r} in {chain_list}"
        mapping[m.group(2)] = m.group(1)
    assert mapping, f"empty chain.list: {chain_list}"
    return mapping


def add_confidence(df: pl.DataFrame) -> pl.DataFrame:
    conf = {}
    for d in {Path(p).parent for p in df["combfold_output_path"]}:
        with open(d / "confidence.txt") as f:
            conf.update({k: float(v) for k, v in (line.split() for line in f if line.strip())})
    # replace_strict raises if a path has no confidence -> acts as an assert
    return df.with_columns(
        pl.col("combfold_output_path").replace_strict(conf, return_dtype=pl.Float64).alias("CF_confidence")
    )


def load_run(run_dir: Path) -> dict[str, dict]:
    """complex_ac -> row (best CF output by confidence, n_proteins > 2)."""
    df = add_confidence(pl.read_parquet(run_dir / "cf_pdb_structure_similarity/cf_pdb_eval_all_metrics.parquet"))
    df = (
        df.sort("CF_confidence", descending=True)
        .group_by("complex_ac", maintain_order=True)
        .first()
        .filter(pl.col("n_proteins") > 2)
    )
    return {r["complex_ac"]: r for r in df.iter_rows(named=True)}


def input_pdbs_dir(row: dict) -> Path:
    res = Path(row["Combfold_result_path"])
    assert res.name.endswith("_pool_output"), f"unexpected result folder name: {res}"
    d = res.parent / (res.name.removesuffix("_pool_output") + "_pool_input") / "pdbs"
    assert d.exists(), f"input pdbs dir not found: {d}"
    return d


def pairs_in_dir(pdbs_dir: Path) -> dict[str, Path]:
    """pair name -> rank_1 pdb, for the pairs that actually exist in this input folder.
    (pairs_used in the parquet lists every pair considered for the complex over all stoichiometries,
    but a folder only holds the pairs needed for its own stoichiometry, e.g. no homodimer for x1.)"""
    out: dict[str, Path] = {}
    for f in sorted(pdbs_dir.glob("AFM_*_unrelaxed_rank_1_*.pdb")):
        pair = f.name.removeprefix("AFM_").split("_unrelaxed")[0]
        assert pair.count("_") == 1, f"unexpected pair name {pair!r} from {f.name}"
        assert pair not in out, f"more than one rank_1 pdb for {pair} in {pdbs_dir}: {out[pair].name}, {f.name}"
        out[pair] = f
    assert out, f"no AFM_*_unrelaxed_rank_1_*.pdb in {pdbs_dir}"
    return out


# ── sequence lengths (to get the full length of each complex) ─────────────────

assert SEQ_CSV.exists(), f"sequence csv not found: {SEQ_CSV}"
seq_df = pl.read_csv(SEQ_CSV)
assert {"uniprot_id", "sequence"} <= set(seq_df.columns), f"unexpected columns in {SEQ_CSV.name}: {seq_df.columns}"
seq_len = dict(zip(seq_df["uniprot_id"], seq_df["sequence"].str.len_chars()))

# ── load both runs ────────────────────────────────────────────────────────────

print("Loading runs...")
rows_l, rows_r = load_run(RUN_L), load_run(RUN_R)
only_l, only_r = sorted(set(rows_l) - set(rows_r)), sorted(set(rows_r) - set(rows_l))
print(f"  PAE: {len(rows_l)} complexes, trimmed: {len(rows_r)} complexes")
if only_l or only_r:
    print(f"  NOTE only in PAE run (left out): {only_l}")
    print(f"  NOTE only in trimmed run (left out): {only_r}")
acs = sorted(set(rows_l) & set(rows_r))
assert acs, "no complex present in both runs"
print(f"  {len(acs)} complexes in both runs")


def run_record(row: dict) -> dict:
    out = Path(row["combfold_output_path"])
    assert out.exists(), f"CF output not found: {out}"
    out_text = out.read_text()

    chain_map = read_chain_list(out)
    chains = ca_chains(out_text)
    missing = set(chains) - set(chain_map)
    assert not missing, f"{out}: chains {missing} not in chain.list"
    unknown = {chain_map[c] for c in set(chains)} - set(seq_len)
    assert not unknown, f"{out}: UniProt ids without sequence in {SEQ_CSV.name}: {unknown}"

    n_ca = len(chains)
    full_len = sum(seq_len[chain_map[c]] for c in dict.fromkeys(chains))  # each chain once
    assert n_ca <= full_len, f"{out}: {n_ca} CA atoms but full length of its chains is only {full_len}"

    aligned = Path(row["usalign_pred_dir"]) / "complex" / "usalign.pdb"
    assert aligned.exists(), f"US-align pdb not found: {aligned}"
    aligned_text = aligned.read_text()
    assert set(ca_chains(aligned_text)) == set(chains), f"{aligned}: chain ids differ from {out.name}"

    cm = json.loads(row["chain_mapping"])  # uniprot -> [[pred chains], [ref chains]]
    ref_chain_map = {rc: u for u, (_, rcs) in cm.items() for rc in rcs}

    return {
        "cf_confidence": round(row["CF_confidence"], 2),
        "tm": row["usalign_cpx_tm_score"],
        "rmsd": row["usalign_cpx_rmsd"],
        "n_ca": n_ca,
        "full_len": full_len,
        "trimmed_frac": 1 - n_ca / full_len,
        "pred_gz": compress(strip_pdb(aligned_text)),
        "pred_chain_map": chain_map,
        "ref_chain_map": ref_chain_map,
        "result_dir": row["Combfold_result_path"],
    }


def pair_records(row_l: dict, row_r: dict) -> tuple[list[dict], int, int, int]:
    files_l, files_r = pairs_in_dir(input_pdbs_dir(row_l)), pairs_in_dir(input_pdbs_dir(row_r))
    common = sorted(set(files_l) & set(files_r))

    # cheap pass: read + count residues only
    cand = []
    for pair in common:
        txt_l, txt_r = files_l[pair].read_text(), files_r[pair].read_text()
        n_l, n_r = len(ca_chains(txt_l)), len(ca_chains(txt_r))
        assert n_r <= n_l, f"{pair}: trimmed pair ({n_r}) longer than untrimmed pair ({n_l})"
        cand.append((pair, txt_l, txt_r, n_l, n_r))
    # stable sort, same tie order as before
    cand.sort(key=lambda t: t[3] - t[4], reverse=True)

    # expensive pass (strip + gzip) only for the pairs that are actually shown
    recs = []
    for pair, txt_l, txt_r, n_l, n_r in cand[:MAX_PAIRS]:
        p1, p2 = pair.split("_")
        recs.append({
            "pair": pair,
            "proteins": [p1, p2],
            "label": f"{p1} – {p2}" if p1 != p2 else f"{p1} (homo)",
            "n_l": n_l,
            "n_r": n_r,
            "removed_frac": 1 - n_r / n_l,
            "left_gz": compress(strip_pdb(txt_l)),
            "right_gz": compress(strip_pdb(txt_r)),
        })
    return recs, len(common), len(files_l), len(files_r)


def build_complex(ac: str) -> tuple[dict, str]:
    rl, rr = rows_l[ac], rows_r[ac]

    ref_l, ref_r = Path(rl["reference_pdb_path"]), Path(rr["reference_pdb_path"])
    assert ref_l == ref_r, f"{ac}: different reference files in the two runs: {ref_l} vs {ref_r}"
    assert ref_l.exists(), f"reference not found: {ref_l}"
    ref_fmt = "cif" if ref_l.suffix == ".cif" else "pdb"
    ref_text = ref_l.read_text()

    pairs, n_common, n_files_l, n_files_r = pair_records(rl, rr)
    same_folder = Path(rl["Combfold_result_path"]).name == Path(rr["Combfold_result_path"]).name
    log = (f"  {ac}: TM {rl['usalign_cpx_tm_score']:.3f} -> {rr['usalign_cpx_tm_score']:.3f} | "
           f"input pairs in folder: pae {n_files_l}, trimmed {n_files_r}, common {n_common}, showing {len(pairs)}"
           + ("" if same_folder else "  | NOTE: different CombFold folders (stoichiometry differs)"))

    rec = {
        "complex_ac": ac,
        "identifiers": rl["identifiers"],
        "n_proteins": int(rl["n_proteins"]),
        "pdb_id": rl["pdb_id"],
        "ref_fmt": ref_fmt,
        "ref_gz": compress(strip_pdb(ref_text) if ref_fmt == "pdb" else ref_text),
        "L": run_record(rl),
        "R": run_record(rr),
        "same_folder": same_folder,
        "n_pairs_common": n_common,
        "pairs": pairs,
    }
    return rec, log


# executor.map keeps input order and re-raises the first exception in the main thread,
# so the asserts above still stop the script and nothing is skipped silently.
with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
    results = list(ex.map(build_complex, acs))

EMBED: dict[str, dict] = {}
for ac, (rec, log) in zip(acs, results):
    print(log)
    EMBED[ac] = rec

print(f"\nEmbedded {len(EMBED)} complexes")


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PAE vs PAE + pLDDT trimmed</title>
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
  padding:9px 14px; display:flex; align-items:center; gap:12px; flex-shrink:0; flex-wrap:wrap;
}
.topbar h2   { font-size:14px; font-weight:700; color:#333; }
.topbar .hint { font-size:12px; color:#999; }
.topbar label.sync { font-size:12px; display:flex; align-items:center; gap:4px; cursor:pointer; user-select:none; }
.main { display:flex; gap:8px; flex:1; min-height:0; }

/* ── left table ───────────────────────────────────────────────── */
.table-panel {
  width:520px; flex-shrink:0;
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
  padding:6px 7px; font-weight:600; font-size:11px; color:#444;
  border-bottom:1px solid #ddd; white-space:nowrap; cursor:pointer; user-select:none;
}
thead th:hover { background:#ececec; }
thead th.r { text-align:right; }
tbody tr { cursor:pointer; transition:background .1s; }
tbody tr:hover    { background:#f0f5ff; }
tbody tr.selected { background:#ddeeff; }
tbody td { padding:5px 7px; border-bottom:1px solid #f2f2f2; white-space:nowrap; }
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
.mode-tabs { display:flex; gap:3px; flex-shrink:0; }
.mode-tab {
  background:#f0f0f0; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:2px 10px; font-size:12px; font-weight:600;
  line-height:1.6; white-space:nowrap;
}
.mode-tab:hover:not(.active) { background:#e4e4e4; }
.mode-tab.active { background:#0072B2; color:white; border-color:#0072B2; }

/* ── pair bar ─────────────────────────────────────────────────── */
.pair-bar {
  padding:6px 12px; border-bottom:1px solid #eee; background:#fafafa;
  display:none; align-items:center; gap:8px; flex-shrink:0;
}
.pair-bar-label { font-weight:600; font-size:12px; color:#555; white-space:nowrap; }
.navbtn {
  background:#f2f2f2; border:1px solid #ccc; border-radius:4px;
  cursor:pointer; padding:1px 8px; font-size:13px; line-height:1.6; flex-shrink:0;
}
.navbtn:hover:not(:disabled) { background:#e4e4e4; }
.navbtn:disabled { opacity:.35; cursor:default; }
.pair-list-wrap { flex:1; overflow-x:auto; display:flex; gap:5px; padding-bottom:3px; min-width:0; }
.pair-btn {
  white-space:nowrap; flex-shrink:0;
  padding:2px 10px; font-size:11px; font-family:monospace; font-weight:600;
  border:1px solid #ccc; border-radius:4px; cursor:pointer; background:#f2f2f2;
}
.pair-btn:hover:not(.active) { background:#e4e4e4; }
.pair-btn.active { background:#0072B2; color:white; border-color:#0072B2; }
.pair-count { font-size:11px; color:#999; white-space:nowrap; flex-shrink:0; }

/* ── two windows ──────────────────────────────────────────────── */
.dual-view { flex:1; min-height:0; display:flex; flex-direction:row; }
.sbs-panel { flex:1; min-width:0; display:flex; flex-direction:column; overflow:hidden; }
.sbs-panel + .sbs-panel { border-left:2px solid #e8e8e8; }
.sbs-hdr {
  padding:6px 10px; border-bottom:1px solid #eeeeee;
  background:#fafafa; flex-shrink:0;
  display:flex; flex-direction:column; gap:5px;
}
.sbs-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; min-height:22px; }
.sbs-label { font-size:12px; font-weight:700; white-space:nowrap; flex-shrink:0; }
.sbs-body  { flex:1; position:relative; min-height:0; }
.sbs-body > div:first-child { width:100%; height:100%; }
.spin {
  position:absolute; inset:0; background:rgba(255,255,255,.85);
  display:none; align-items:center; justify-content:center;
  font-size:13px; color:#666; text-align:center; padding:20px;
}
.spin.on { display:flex; }
.metric-chip {
  font-size:11px; font-family:monospace; color:#444;
  background:#f5f5f5; border:1px solid #e2e2e2;
  border-radius:4px; padding:1px 7px; white-space:nowrap;
}
.metric-chip b { font-weight:700; }
.metric-chip.warn { background:#fff3e0; border-color:#ffcc80; }
.layer-ctrl { display:flex; align-items:center; gap:5px; }
.swatch { width:10px; height:10px; border-radius:2px; flex-shrink:0; border:1px solid rgba(0,0,0,.15); }
.layer-ctrl label { font-size:12px; white-space:nowrap; cursor:pointer; user-select:none; }
.layer-ctrl input[type=checkbox] { cursor:pointer; }
.op-slider { width:65px; cursor:pointer; accent-color:#555; }
.chain-legend { display:flex; gap:6px; align-items:center; font-size:11px; color:#555; }
</style>
</head>
<body>

<div class="topbar">
  <h2>PAE vs PAE + pLDDT trimmed</h2>
  <div class="mode-tabs">
    <button class="mode-tab active" id="tab-cpx"   onclick="setMode('cpx')">Complexes</button>
    <button class="mode-tab"        id="tab-pairs" onclick="setMode('pairs')">Input pairs</button>
  </div>
  <label class="sync"><input type="checkbox" id="chk-sync" checked> sync views</label>
  <span class="hint">Click a row to load structures. Pairs mode: &larr; / &rarr; to step through pairs.</span>
</div>

<div class="main">

  <div class="table-panel">
    <div class="table-panel-hdr" id="table-hdr">Complexes (n proteins &gt; 2) &mdash; click a column to sort</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr id="thead-row"></tr>
        </thead>
        <tbody id="cpx-tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="viewer-panel">
    <div class="viewer-hdr">
      <h3 id="vwr-title">&mdash;</h3>
      <span class="badge" id="vwr-pdb">&mdash;</span>
      <span class="badge" id="vwr-n">&mdash;</span>
    </div>

    <div class="pair-bar" id="pair-bar">
      <span class="pair-bar-label">Input pairs:</span>
      <button class="navbtn" id="prev-pair" title="Previous pair (&larr;)">&#9664;</button>
      <div class="pair-list-wrap" id="pair-list"></div>
      <button class="navbtn" id="next-pair" title="Next pair (&rarr;)">&#9654;</button>
      <span class="pair-count" id="pair-count"></span>
    </div>

    <div class="dual-view">
      <div class="sbs-panel">
        <div class="sbs-hdr">
          <div class="sbs-row"><span class="sbs-label" id="title-L" style="color:#E69F00"></span><span id="chips-L" style="display:contents"></span></div>
          <div class="sbs-row" id="ctrl-L"></div>
        </div>
        <div class="sbs-body"><div id="viewer-L"></div><div class="spin on" id="spin-L">Select a complex from the table</div></div>
      </div>
      <div class="sbs-panel">
        <div class="sbs-hdr">
          <div class="sbs-row"><span class="sbs-label" id="title-R" style="color:#009E73"></span><span id="chips-R" style="display:contents"></span></div>
          <div class="sbs-row" id="ctrl-R"></div>
        </div>
        <div class="sbs-body"><div id="viewer-R"></div><div class="spin on" id="spin-R">Select a complex from the table</div></div>
      </div>
    </div>
  </div>

</div>

<script>
const COMPLEXES = %%JSON%%;

const COL_REF = '#0072B2';
const COL     = {L: '#E69F00', R: '#009E73'};
/* Okabe-Ito chain colours (pair mode: chain A, chain B) */
const CC = ['#E69F00','#0072B2','#009E73','#CC79A7','#56B4E9','#D55E00','#F0E442','#999999'];
const LABEL_STYLE = {backgroundColor:'black', backgroundOpacity:0.75, fontColor:'white', fontSize:12, padding:4, inFront:true};
const TITLE = {
  cpx:   {L: 'PAE',                R: 'PAE + pLDDT trimmed'},
  pairs: {L: 'PAE input (untrimmed)', R: 'pLDDT-trimmed input'},
};

const byId = id => document.getElementById(id);
const SIDES = ['L','R'];

/* ── gzip decompress ──────────────────────────────────────────── */
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

function tmColor(v) {
  if (v == null) return '#aaa';
  const t = Math.min(1, Math.max(0, v));
  if (t < 0.5) return `rgb(210,${Math.round(t*2*180)},40)`;
  return `rgb(${Math.round((1-(t-0.5)*2)*210)},170,40)`;
}
function fmt(v, d=3) { return (v == null) ? '&mdash;' : v.toFixed(d); }
function pct(v, d=1) { return (v == null) ? '&mdash;' : (100*v).toFixed(d) + '%'; }

/* ── table ────────────────────────────────────────────────────── */
const COLUMNS = [
  {key:'ac',      label:'Complex',    get:c => c.complex_ac,          cell:c => `<td class="mono" style="font-size:11px">${c.complex_ac}</td>`},
  {key:'n',       label:'N',          get:c => c.n_proteins,          cell:c => `<td class="r">${c.n_proteins}</td>`, r:true},
  {key:'pdb',     label:'PDB',        get:c => c.pdb_id,              cell:c => `<td class="mono" style="font-size:11px">${(c.pdb_id||'').toUpperCase()}</td>`},
  {key:'confL',   label:'CF PAE',     get:c => c.L.cf_confidence,     cell:c => `<td class="r mono">${fmt(c.L.cf_confidence,1)}</td>`, r:true},
  {key:'confR',   label:'CF trim',    get:c => c.R.cf_confidence,     cell:c => `<td class="r mono">${fmt(c.R.cf_confidence,1)}</td>`, r:true},
  {key:'tmL',     label:'TM PAE',     get:c => c.L.tm,                cell:c => `<td class="tm-val" style="color:${tmColor(c.L.tm)}">${fmt(c.L.tm)}</td>`, r:true},
  {key:'tmR',     label:'TM trim',    get:c => c.R.tm,                cell:c => `<td class="tm-val" style="color:${tmColor(c.R.tm)}">${fmt(c.R.tm)}</td>`, r:true},
  {key:'dtm',     label:'ΔTM',        get:c => (c.R.tm - c.L.tm),     cell:c => {const d=c.R.tm-c.L.tm; return `<td class="r mono" style="color:${d>0.02?'#0072B2':(d<-0.02?'#D55E00':'#888')}">${d>=0?'+':''}${d.toFixed(3)}</td>`;}, r:true},
  {key:'trimmed', label:'Trimmed',    get:c => c.R.trimmed_frac,      cell:c => `<td class="r mono">${pct(c.R.trimmed_frac)}</td>`, r:true},
];
let sortKey = 'tmR', sortDesc = true, curAc = null;

function buildTable() {
  byId('thead-row').innerHTML = COLUMNS.map(col =>
    `<th class="${col.r ? 'r' : ''}" data-key="${col.key}">${col.label}${col.key === sortKey ? (sortDesc ? ' ▼' : ' ▲') : ''}</th>`
  ).join('');
  document.querySelectorAll('#thead-row th').forEach(th => th.onclick = () => {
    const k = th.dataset.key;
    if (k === sortKey) sortDesc = !sortDesc; else { sortKey = k; sortDesc = true; }
    buildTable();
  });
  const col = COLUMNS.find(c => c.key === sortKey);
  const list = Object.values(COMPLEXES).sort((a, b) => {
    const va = col.get(a), vb = col.get(b);
    const cmp = (typeof va === 'string') ? va.localeCompare(vb) : ((va ?? -Infinity) - (vb ?? -Infinity));
    return sortDesc ? -cmp : cmp;
  });
  const tbody = byId('cpx-tbody');
  tbody.innerHTML = '';
  list.forEach(c => {
    const tr = document.createElement('tr');
    tr.dataset.ac = c.complex_ac;
    tr.innerHTML = COLUMNS.map(col => col.cell(c)).join('');
    if (c.complex_ac === curAc) tr.classList.add('selected');
    tr.onclick = () => selectComplex(c.complex_ac);
    tbody.appendChild(tr);
  });
}

/* ── viewer state ─────────────────────────────────────────────── */
let mode = 'cpx';
const V = {L: null, R: null};
const M = {L: {}, R: {}};            // models per side
const RAW = {ref: null, refFmt: 'pdb', L: null, R: null};
const PAIRRAW = {L: null, R: null};
let curPairIdx = -1, loadTok = 0;

function initViewers() {
  SIDES.forEach(s => {
    V[s] = $3Dmol.createViewer(byId('viewer-' + s), {backgroundColor: 'white', antialias: true});
    /* no depth fog: otherwise everything far from the camera fades into the white background */
    if (typeof V[s].enableFog === 'function') V[s].enableFog(false);
    else console.warn('This 3Dmol version has no enableFog: depth fog stays on');
  });
  /* link the two views (rotation / zoom / translation) */
  let busy = false;
  SIDES.forEach(s => {
    const o = s === 'L' ? 'R' : 'L';
    if (typeof V[s].setViewChangeCallback !== 'function') {
      console.warn('This 3Dmol version has no setViewChangeCallback: view sync disabled');
      return;
    }
    V[s].setViewChangeCallback(() => {
      if (busy || !byId('chk-sync').checked) return;
      const a = V[s].getView(), b = V[o].getView();
      if (a.every((x, i) => Math.abs(x - b[i]) < 1e-6)) return;
      busy = true; V[o].setView(a); busy = false;
    });
  });
}

function spin(side, on, msg) {
  const el = byId('spin-' + side);
  el.classList.toggle('on', on);
  if (msg != null) el.textContent = msg;
}

function fitViews() {
  V.L.zoomTo();
  if (byId('chk-sync').checked) V.R.setView(V.L.getView()); else V.R.zoomTo();
  V.L.render(); V.R.render();
}

/* ── hover labels ─────────────────────────────────────────────── */
function hoverCB(getMap, kind) {
  return (atom, viewer) => {
    const u = getMap()?.[atom.chain];
    viewer.removeAllLabels();
    viewer.addLabel(u ? `${kind} chain ${atom.chain}: ${u}` : `${kind} chain ${atom.chain} (unmapped)`,
                    {...LABEL_STYLE, position: atom});
    viewer.render();
  };
}
const unhoverCB = (atom, viewer) => { viewer.removeAllLabels(); viewer.render(); };

/* ── complex mode ─────────────────────────────────────────────── */
function cpxControls(side) {
  byId('ctrl-' + side).innerHTML = `
    <div class="layer-ctrl">
      <input type="checkbox" id="chk-ref-${side}" checked onchange="applyCpxStyle('${side}')">
      <div class="swatch" style="background:${COL_REF}"></div><label for="chk-ref-${side}">Reference</label>
      <input type="range" class="op-slider" id="op-ref-${side}" min="0" max="1" step="0.05" value="1" oninput="applyCpxStyle('${side}')">
    </div>
    <div class="layer-ctrl">
      <input type="checkbox" id="chk-pred-${side}" checked onchange="applyCpxStyle('${side}')">
      <div class="swatch" style="background:${COL[side]}"></div><label for="chk-pred-${side}">Prediction</label>
      <input type="range" class="op-slider" id="op-pred-${side}" min="0" max="1" step="0.05" value="1" oninput="applyCpxStyle('${side}')">
    </div>`;
}

function cpxChips(side) {
  const c = COMPLEXES[curAc], r = c[side];
  const full = `residues <b>${r.n_ca}</b> / ${r.full_len}`;
  const trimmed = side === 'R'
    ? `<span class="metric-chip warn">trimmed <b>${pct(r.trimmed_frac)}</b> of full length</span>` : '';
  const diffStoic = (side === 'R' && !c.same_folder)
    ? `<span class="metric-chip warn" title="${COMPLEXES[curAc].L.result_dir} vs ${r.result_dir}">different stoichiometry than the PAE run</span>` : '';
  byId('chips-' + side).innerHTML =
    `<span class="metric-chip">CF conf <b>${fmt(r.cf_confidence, 1)}</b></span>` +
    `<span class="metric-chip">TM <b>${fmt(r.tm)}</b>&nbsp;&nbsp;RMSD <b>${fmt(r.rmsd, 2)}</b></span>` +
    `<span class="metric-chip">${full}</span>` + trimmed + diffStoic;
}

function applyCpxStyle(side) {
  const m = M[side];
  if (m.ref) {
    V[side].setStyle({model: m.ref}, byId('chk-ref-' + side).checked
      ? {cartoon: {color: COL_REF, opacity: +byId('op-ref-' + side).value}} : {});
  }
  if (m.pred) {
    V[side].setStyle({model: m.pred}, byId('chk-pred-' + side).checked
      ? {cartoon: {color: COL[side], opacity: +byId('op-pred-' + side).value}} : {});
  }
  V[side].render();
}

function renderCpx() {
  const c = COMPLEXES[curAc];
  SIDES.forEach(s => {
    V[s].removeAllModels(); M[s] = {};
    if (RAW.ref) M[s].ref  = V[s].addModel(RAW.ref, RAW.refFmt);
    if (RAW[s])  M[s].pred = V[s].addModel(RAW[s], 'pdb');
    cpxControls(s); cpxChips(s);
    applyCpxStyle(s);
    if (M[s].ref)  V[s].setHoverable({model: M[s].ref},  true, hoverCB(() => COMPLEXES[curAc]?.[s].ref_chain_map,  'Ref'),  unhoverCB);
    if (M[s].pred) V[s].setHoverable({model: M[s].pred}, true, hoverCB(() => COMPLEXES[curAc]?.[s].pred_chain_map, 'Pred'), unhoverCB);
  });
  fitViews();
}

/* ── pair mode ────────────────────────────────────────────────── */
function chainsFromText(t) {
  const chains = [];
  for (const line of t.split('\n')) {
    if ((line.startsWith('ATOM') || line.startsWith('HETATM')) && line.length > 21) {
      const ch = line[21];
      if (!chains.includes(ch)) chains.push(ch);
    }
  }
  return chains;
}

function pairControls(side) {
  if (side === 'L') {
    byId('ctrl-L').innerHTML = `
      <div class="layer-ctrl"><label>Opacity</label>
        <input type="range" class="op-slider" id="op-pair-L" min="0" max="1" step="0.05" value="1" oninput="applyPairStyle('L')"></div>
      <div class="chain-legend" id="legend-L"></div>`;
  } else {
    byId('ctrl-R').innerHTML = `
      <div class="layer-ctrl"><label>Opacity</label>
        <input type="range" class="op-slider" id="op-pair-R" min="0" max="1" step="0.05" value="1" oninput="applyPairStyle('R')"></div>
      <div class="layer-ctrl">
        <input type="checkbox" id="chk-ghost" checked onchange="applyPairStyle('R')">
        <div class="swatch" style="background:#bbbbbb"></div><label for="chk-ghost">Show removed residues (grey)</label>
      </div>
      <div class="chain-legend" id="legend-R"></div>`;
  }
}

function applyPairStyle(side) {
  const m = M[side];
  const op = +byId('op-pair-' + side).value;
  if (m.ghost) V[side].setStyle({model: m.ghost}, byId('chk-ghost').checked ? {cartoon: {color: '#bbbbbb', opacity: 0.45}} : {});
  if (m.pred && m.chains) m.chains.forEach((ch, i) =>
    V[side].setStyle({model: m.pred, chain: ch}, {cartoon: {color: CC[i % CC.length], opacity: op}}));
  V[side].render();
}

function pairChips(p) {
  const removed = p.n_l - p.n_r;
  byId('chips-L').innerHTML = `<span class="metric-chip">residues <b>${p.n_l}</b></span>`;
  byId('chips-R').innerHTML =
    `<span class="metric-chip">residues <b>${p.n_r}</b></span>` +
    `<span class="metric-chip warn">removed <b>${removed}</b> (${pct(p.removed_frac)})</span>`;
  const legend = SIDES.map(s => {
    const chains = M[s].chains || [];
    return chains.map((ch, i) =>
      `<div class="swatch" style="background:${CC[i % CC.length]}"></div><span>${ch}: ${p.proteins[i] ?? '?'}</span>`).join('');
  });
  byId('legend-L').innerHTML = legend[0];
  byId('legend-R').innerHTML = legend[1];
}

async function selectPair(idx) {
  const c = COMPLEXES[curAc];
  if (!c || idx < 0 || idx >= c.pairs.length) return;
  curPairIdx = idx;
  document.querySelectorAll('.pair-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
  document.querySelectorAll('.pair-btn')[idx]?.scrollIntoView({block: 'nearest', inline: 'nearest'});
  byId('prev-pair').disabled = idx <= 0;
  byId('next-pair').disabled = idx >= c.pairs.length - 1;

  const p = c.pairs[idx], tok = ++loadTok;
  SIDES.forEach(s => spin(s, true, 'Loading…'));
  const [tl, tr] = await Promise.all([ungzip(p.left_gz), ungzip(p.right_gz)]);
  if (tok !== loadTok) return;
  PAIRRAW.L = tl; PAIRRAW.R = tr;

  SIDES.forEach(s => { V[s].removeAllModels(); M[s] = {}; pairControls(s); });
  M.L.pred = V.L.addModel(tl, 'pdb'); M.L.chains = chainsFromText(tl);
  M.R.pred = V.R.addModel(tr, 'pdb'); M.R.chains = chainsFromText(tr);
  M.R.ghost = V.R.addModel(tl, 'pdb');       // untrimmed pair underneath the trimmed one
  SIDES.forEach(s => {
    const lab = (atom, viewer) => {
      const i = M[s].chains.indexOf(atom.chain);
      viewer.removeAllLabels();
      viewer.addLabel(`Chain ${atom.chain}: ${p.proteins[i] ?? '?'}`, {...LABEL_STYLE, position: atom});
      viewer.render();
    };
    V[s].setHoverable({model: M[s].pred}, true, lab, unhoverCB);
    applyPairStyle(s);
  });
  pairChips(p);
  fitViews();
  SIDES.forEach(s => spin(s, false));
}

function buildPairBar() {
  const c = COMPLEXES[curAc];
  const list = byId('pair-list');
  list.innerHTML = '';
  c.pairs.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'pair-btn';
    b.textContent = p.label;
    b.title = `${p.pair}: ${p.n_l} → ${p.n_r} residues (−${pct(p.removed_frac)})`;
    b.onclick = () => selectPair(i);
    list.appendChild(b);
  });
  byId('pair-count').textContent =
    `${c.pairs.length} of ${c.n_pairs_common} pairs (most trimmed first)`;
}

/* ── mode / select ────────────────────────────────────────────── */
function setMode(m) {
  mode = m;
  byId('tab-cpx').classList.toggle('active', m === 'cpx');
  byId('tab-pairs').classList.toggle('active', m === 'pairs');
  byId('pair-bar').style.display = m === 'pairs' ? 'flex' : 'none';
  SIDES.forEach(s => { byId('title-' + s).textContent = TITLE[m][s]; });
  if (curAc) refresh();
  setTimeout(() => SIDES.forEach(s => V[s]?.resize()), 50);
}

async function refresh() {
  if (!curAc) return;
  if (mode === 'cpx') { renderCpx(); SIDES.forEach(s => spin(s, false)); }
  else {
    buildPairBar(); curPairIdx = -1;
    if (COMPLEXES[curAc].pairs.length) await selectPair(0);
    else SIDES.forEach(s => spin(s, true, 'No input pair exists in both runs for this complex'));
  }
}

async function selectComplex(ac) {
  if (ac === curAc) return;
  curAc = ac;
  const c = COMPLEXES[ac], tok = ++loadTok;
  document.querySelectorAll('#cpx-tbody tr').forEach(tr => tr.classList.toggle('selected', tr.dataset.ac === ac));
  byId('vwr-title').textContent = ac;
  byId('vwr-title').title = c.identifiers;
  byId('vwr-pdb').textContent = (c.pdb_id || '').toUpperCase();
  byId('vwr-n').textContent = `n = ${c.n_proteins}`;
  SIDES.forEach(s => spin(s, true, 'Loading…'));
  const [ref, l, r] = await Promise.all([ungzip(c.ref_gz), ungzip(c.L.pred_gz), ungzip(c.R.pred_gz)]);
  if (tok !== loadTok) return;
  RAW.ref = ref; RAW.refFmt = c.ref_fmt; RAW.L = l; RAW.R = r;
  await refresh();
}

/* ── keyboard / nav ───────────────────────────────────────────── */
byId('prev-pair').onclick = () => selectPair(curPairIdx - 1);
byId('next-pair').onclick = () => selectPair(curPairIdx + 1);
document.addEventListener('keydown', e => {
  if (mode !== 'pairs' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft')  selectPair(curPairIdx - 1);
  if (e.key === 'ArrowRight') selectPair(curPairIdx + 1);
});

/* ── init ─────────────────────────────────────────────────────── */
initViewers();
SIDES.forEach(s => { byId('title-' + s).textContent = TITLE.cpx[s]; });
buildTable();
const first = document.querySelector('#cpx-tbody tr')?.dataset.ac;
if (first) selectComplex(first);
</script>
</body>
</html>
"""

# ── write output ──────────────────────────────────────────────────────────────

json_blob = json.dumps(EMBED, ensure_ascii=True)
html_out = HTML.replace("%%JSON%%", json_blob)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html_out, encoding="utf-8")

size_kb = OUT.stat().st_size / 1024
print(f"\n  Written : {OUT}")
print(f"  Size    : {size_kb:.0f} KB  (~{size_kb/1024:.1f} MB)")