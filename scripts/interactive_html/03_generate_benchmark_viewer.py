#!/usr/bin/env python3
"""
Generate self-contained HTML viewer for benchmark complexes (n_proteins > 2,
exact PDB match, CF output present on disk).

Replicates the notebook pair-joining logic so every embedded complex carries
the same (complex_ac, pair) metrics that the notebook computed.

For each complex (best CF output by CF_confidence):
  Left  panel : CombFold assembled model — chain hover shows UniProt ID
  Right panel : PDB reference (exact match, fetched live from RCSB)
  Pair  bar   : click any af_close_by_chainpair button; the pair's chains are
                highlighted blue/green simultaneously in both viewers, and
                ipTM-corrected / PAE-min / pLDDT is shown inline.

Usage:
    uv run python scripts/generate_benchmark_viewer.py

Output:
    data/Pipeline/10_all_CP_complexes/benchmark_structure_viewer/03_benchmark_viewer.html
"""

import base64, gzip, json, re, sys
from pathlib import Path

import polars as pl
from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"

CF_BASE     = data_dir / "Pipeline/10_all_CP_complexes/CombFold"
BENCH_PARQ  = data_dir / "Pipeline/10_all_CP_complexes/cf_pdb_structure_similarity/cf_pdb_eval_all_metrics.parquet"
METRICS_CSV = data_dir / "Pipeline/10_all_CP_complexes/10_all_CP_complexes_pool_pairs_metrics_with_plddt_corrected.csv"
PDB_MAP_CSV = data_dir / "complete_complex_pdb_mapping_v2/all_pdb_matches_with_match_class.csv"
SIFTS_PATH  = data_dir / "pdb/pdb_chain_uniprot.csv"
OUT         = data_dir / "Pipeline/10_all_CP_complexes/benchmark_structure_viewer/benchmark_viewer.html"


# ── helpers ───────────────────────────────────────────────────────────────────

def compress(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode())).decode()


def strip_pdb(text: str) -> str:
    _KEEP = frozenset({"ATOM", "HETATM", "MODEL", "ENDMDL", "TER", "END"})
    return "\n".join(l for l in text.splitlines() if l[:6].strip() in _KEEP)


def sniff_delimiter(path: Path, default: str = ",") -> str:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            return "\t" if line.count("\t") > line.count(",") else ","
    return default


def read_chain_list(cf_output_path: Path) -> dict[str, str] | None:
    """Chain letter → UniProt from CombFold's own chain.list manifest.

    cf_output_path points to a specific assembled PDB file, e.g.
    .../assembled_results/pred0.pdb.  chain.list lives at
    .../assembled_results/../../_unified_representation/assembly_output/chain.list
    (i.e. two levels up from the PDB file).
    """
    folder     = cf_output_path.parent.parent   # strip predN.pdb + assembled_results/
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
            mapping[m.group(2)] = m.group(1)   # chain → UniProt
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

    folder_name = cf_output_path.parent.parent.name
    print(f"  WARNING no chain.list for {folder_name} — folder-name fallback", file=sys.stderr)
    base = folder_name.replace("_pool_output", "")
    uniprot_list = [
        acc
        for acc, n in re.findall(r"([A-Z][A-Z0-9]{5,})x(\d+)", base)
        for _ in range(int(n))
    ]
    if len(chains_in_model) != len(uniprot_list):
        print(
            f"  WARNING chain/uniprot count mismatch: "
            f"{len(chains_in_model)} model chains vs {len(uniprot_list)} in folder name",
            file=sys.stderr,
        )
    return {ch: uniprot_list[i] for i, ch in enumerate(chains_in_model) if i < len(uniprot_list)}


# ── load benchmark eval ───────────────────────────────────────────────────────

print("Loading benchmark evaluation parquet...")
bench = pl.read_parquet(BENCH_PARQ)

try:
    from procompa import clean_identifiers
    bench = clean_identifiers(bench, "identifiers")
except Exception:
    pass

print(f"  {bench.height} rows total")

# ── PDB mapping ───────────────────────────────────────────────────────────────

print("Loading PDB mapping (exact_pdb_match only)...")
pdb_map = (
    pl.read_csv(PDB_MAP_CSV)
    .filter(pl.col("match_class") == "exact_pdb_match")
    .select("complex_ac", "pdb_id")
    .unique("complex_ac")   # one PDB per complex (first exact match)
)
print(f"  {pdb_map.height} complexes with exact_pdb_match")

bench = bench.join(pdb_map, on="complex_ac", how="inner")
bench = bench.filter(pl.col("n_proteins") > 2)
print(f"  {bench.height} rows after join + n_proteins > 2 filter")

# Drop rows where the CF output file no longer exists on disk
bench = bench.filter(
    pl.col("combfold_output_path").map_elements(
        lambda p: Path(p).exists(), return_dtype=pl.Boolean
    )
)
print(f"  {bench.height} rows with CF output on disk")

# ── CF confidence ─────────────────────────────────────────────────────────────

print("Reading CF confidence scores from confidence.txt files...")
conf: dict[str, float] = {}
for d in {Path(p).parent for p in bench["combfold_output_path"]}:
    conf_path = d / "confidence.txt"
    if conf_path.exists():
        for line in conf_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    conf[parts[0]] = float(parts[1])
                except ValueError:
                    pass

bench = bench.with_columns(
    pl.col("combfold_output_path")
      .map_elements(lambda p: conf.get(p, float("nan")), return_dtype=pl.Float64)
      .alias("CF_confidence")
)

# Keep best CF output per complex_ac
bench = (
    bench
    .sort("CF_confidence", descending=True, nulls_last=True)
    .unique("complex_ac", keep="first")
    .sort("usalign_cpx_tm_score", descending=True, nulls_last=True)
)
print(f"  {bench.height} unique complexes (best CF output each)")

# ── pair metrics (replicates notebook logic) ──────────────────────────────────

print("Loading pair metrics...")
metrics = pl.read_csv(METRICS_CSV)

bench = bench.with_row_index("row_id")


def extract_pairs(s: str | None) -> list[str]:
    if not s:
        return []
    try:
        return list(json.loads(s).keys())
    except Exception:
        return []


cf_pairs_long = (
    bench
    .select("row_id", "complex_ac", "cf_close_by_chainpairs")
    .with_columns(
        pl.col("cf_close_by_chainpairs")
          .map_elements(extract_pairs, return_dtype=pl.List(pl.String))
          .alias("pair")
    )
    .explode("pair")
    .drop_nulls("pair")
)

# Restrict metrics to only the (complex_ac, pair) combos that appear in CF outputs
metrics = metrics.join(
    cf_pairs_long.select("complex_ac", "pair").unique(),
    on=["complex_ac", "pair"],
    how="semi",
)

# Drop ambiguous (complex_ac, pair) rows in the metrics table
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

# Real-homodimer filter: keep homo pairs only when stoichiometry >= 2
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

# Aggregate per row_id: parallel lists of pair + metrics
METRIC_COLS = [c for c in [
    "chain_pair_iptm",
    "chain_pair_iptm_corrected",
    "chain_pair_pae_min",
    "plddt1",
    "plddt2",
] if c in metrics.columns]

metrics_by_row = (
    cf_pairs_long
    .join(metrics, on=["complex_ac", "pair"], how="inner")
    .group_by("row_id")
    .agg(
        pl.col("pair").alias("af_pairs"),
        *[pl.col(c) for c in METRIC_COLS],
    )
)

bench = bench.join(metrics_by_row, on="row_id", how="left")
n_with = bench.filter(pl.col("af_pairs").is_not_null()).height
print(f"  {n_with}/{bench.height} complexes have pair metrics")

# ── SIFTS ─────────────────────────────────────────────────────────────────────

print("Loading SIFTS...")
sifts_sep = sniff_delimiter(SIFTS_PATH)
sifts_df  = pl.read_csv(SIFTS_PATH, comment_prefix="#", infer_schema_length=0, separator=sifts_sep)
pdb_col   = sifts_df.columns[0]

sifts_lookup: dict[str, dict[str, str]] = {}
for pdb_id, chain, uni in sifts_df.select([pdb_col, "CHAIN", "SP_PRIMARY"]).iter_rows():
    if pdb_id and chain and uni:
        sifts_lookup.setdefault(pdb_id.lower(), {}).setdefault(chain, uni)
print(f"  {len(sifts_lookup)} PDB entries loaded")

# ── build embed dict ──────────────────────────────────────────────────────────

print("\nBuilding embed data...")
EMBED: dict[str, dict] = {}

for row in bench.iter_rows(named=True):
    ac       = row["complex_ac"]
    pdb_id   = row["pdb_id"]
    cf_conf  = row["CF_confidence"]
    n_prot   = row["n_proteins"]
    tm_score = row.get("usalign_cpx_tm_score")
    out_path = Path(row["combfold_output_path"])

    tm_str = f"{tm_score:.3f}" if tm_score is not None and tm_score == tm_score else "N/A"
    print(f"  {ac}  PDB={pdb_id}  TM={tm_str}  CF={cf_conf:.1f}  n={n_prot}")

    # Read and compress CF output
    try:
        cf_text      = out_path.read_text()
        cf_chain_map = build_cf_chain_map(cf_text, out_path)
        cf_gz        = compress(strip_pdb(cf_text))
    except Exception as e:
        print(f"    WARNING: skipping {ac}: {e}", file=sys.stderr)
        continue

    # PDB chain map: chain letter → UniProt (via SIFTS auth_asym_id)
    pdb_chain_map: dict[str, str] = dict(sifts_lookup.get(pdb_id.lower(), {}))
    if not pdb_chain_map:
        print(f"    WARNING: no SIFTS entries for {pdb_id}", file=sys.stderr)

    # Build pair list with parallel metrics
    af_pairs = row.get("af_pairs") or []
    pairs: list[dict] = []
    for i, pair_str in enumerate(af_pairs):
        parts = pair_str.split("_", 1)
        if len(parts) != 2:
            print(f"    WARNING: unrecognised pair format {pair_str!r} — skipping", file=sys.stderr)
            continue
        p1, p2  = parts
        entry: dict = {
            "pair"    : pair_str,
            "proteins": [p1, p2],
            "label"   : f"{p1} – {p2}" if p1 != p2 else f"{p1} (homo)",
        }
        for col in METRIC_COLS:
            vals = row.get(col)
            v    = vals[i] if vals is not None and i < len(vals) else None
            # Polars nulls come through as None; Python float NaN → None for JSON
            entry[col] = None if (v is not None and v != v) else v
        pairs.append(entry)
        pairs.sort(key=lambda p: (p["chain_pair_iptm_corrected"] is None, p["chain_pair_iptm_corrected"] or 0))

    EMBED[ac] = {
        "complex_ac"   : ac,
        "pdb_id"       : pdb_id,
        "cf_confidence": round(cf_conf, 2) if cf_conf == cf_conf else None,
        "n_proteins"   : int(n_prot),
        "cf_gz"        : cf_gz,
        "cf_chain_map" : cf_chain_map,
        "pdb_chain_map": pdb_chain_map,
        "pairs"        : pairs,
        "tm_score"     : row.get("usalign_cpx_tm_score"),
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
<title>Benchmark Structure Viewer</title>
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
.badge-pdb { background: #fff8e1; color: #5d4037; border: 1px solid #ffe082; }

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
.pair-bar-label {
  font-weight: 600; font-size: 12px; color: #555; white-space: nowrap;
}
.navbtn {
  background: #f2f2f2; border: 1px solid #ccc; border-radius: 4px;
  cursor: pointer; padding: 1px 8px; font-size: 13px; line-height: 1.6;
  flex-shrink: 0;
}
.navbtn:hover:not(:disabled) { background: #e4e4e4; }
.navbtn:disabled { opacity: .35; cursor: default; }
.navbtn.active-toggle {
  background: #0072B2; color: white; border-color: #005a8e;
}
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
.pair-count { font-size: 11px; color: #999; white-space: nowrap; flex-shrink: 0; }

/* ── metrics display ── */
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

/* ── color legend ── */
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
  <span class="badge badge-cf"  id="badge-cf"></span>
  <span class="badge badge-n"   id="badge-n"></span>
  <span class="badge badge-pdb" id="badge-pdb"></span>
  <span id="badge-tm" class="badge" style="background:#f3e5f5;color:#4a148c;border:1px solid #ce93d8;">TM ?</span>
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
      <h3 id="pdb-title">PDB Reference</h3>
    </div>
    <div class="panel-body">
      <div id="pdb-viewer"  class="viewer3d"></div>
      <div id="pdb-overlay" class="overlay">Loading...</div>
    </div>
  </div>

</div>

<!-- pair navigation bar -->
<div class="pair-bar">
  <span class="pair-bar-label">Pairs:</span>
  <button class="navbtn" id="prev-pair" title="Previous pair (←)">&#9664;</button>
  <div class="pair-list-wrap" id="pair-list"></div>
  <button class="navbtn" id="next-pair" title="Next pair (→)">&#9654;</button>
  <span class="pair-count" id="pair-count"></span>
  <button class="navbtn" id="toggle-others" title="Hide/show non-pair chains (H)">Hide others</button>
  <div class="pair-metrics" id="pair-metrics">
    <span class="m-na">Select a pair to highlight it</span>
  </div>
  <div class="legend">
    <span><span class="leg-sw" style="background:#0072B2"></span>Prot 1</span>
    <span><span class="leg-sw" style="background:#009E73"></span>Prot 2</span>
    <span><span class="leg-sw" style="background:#E69F00"></span>Other (CF)</span>
    <span><span class="leg-sw" style="background:#bbbbbb"></span>Other (PDB)</span>
  </div>
</div>

<script>
const COMPLEXES = %%JSON%%;

/* Okabe-Ito palette */
const CC     = ['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#F0E442','#999999'];
const YELLOW = '#E69F00';
const GRAY   = '#bbbbbb';

const LABEL_STYLE = {
  backgroundColor: 'black', backgroundOpacity: 0.75,
  fontColor: 'white', fontSize: 12, padding: 4, inFront: true,
};

/* ── state ───────────────────────────────────────────────────────────── */
let cfV = null, pdbV = null;
let cur = null;
let cfRaw = null, pdbRaw = null, pdbFmtCur = 'pdb';
let currentPairIdx = -1;
let pdbCache = {}, pdbCacheOwner = null;
let hideOthers = false;   // ← new: toggle for non-pair chains

/* ── gzip decompress ─────────────────────────────────────────────────── */
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

/* ── overlay ─────────────────────────────────────────────────────────── */
function spin(id, on, msg) {
  const el = document.getElementById(id);
  el.classList.toggle('on', on);
  if (msg !== undefined) el.textContent = msg;
}

/* ── init viewers ────────────────────────────────────────────────────── */
function initViewers() {
  const opts = {backgroundColor: 'white', antialias: true};
  cfV  = $3Dmol.createViewer(document.getElementById('cf-viewer'),  opts);
  pdbV = $3Dmol.createViewer(document.getElementById('pdb-viewer'), opts);
}

/* ── hover callbacks ─────────────────────────────────────────────────── */
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

function pdbHoverCB(atom, viewer) {
  const u = cur?.pdb_chain_map?.[atom.chain];
  viewer.removeAllLabels();
  viewer.addLabel(
    u ? `Chain ${atom.chain}: ${u}` : `Chain ${atom.chain} (no SIFTS)`,
    {...LABEL_STYLE, position: atom}
  );
  viewer.render();
}
function pdbUnhoverCB(atom, viewer) { viewer.removeAllLabels(); viewer.render(); }

/* ── default style (by chain, Okabe-Ito) ────────────────────────────── */
function styleByChain(v, text, fmt, hoverOn, hoverOff) {
  v.removeAllModels();
  const model  = v.addModel(text, fmt);
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();
  chains.forEach((ch, i) => v.setStyle({chain: ch}, {cartoon: {color: CC[i % CC.length]}}));
  v.setHoverable({}, true, hoverOn, hoverOff);
  v.zoomTo(); v.render();
}

/* ── pair highlighting ───────────────────────────────────────────────── */
/* proteins[0] chains → blue (CC[0])
   proteins[1] chains → green (CC[2])  (same for homo: all blue)
   other CF chains    → yellow (or hidden when hideOthers is true)
   other PDB chains   → gray   (or hidden when hideOthers is true)    */

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

function highlightPDBPair(proteins) {
  if (!pdbRaw || !pdbV) return;
  const [p1, p2] = proteins;
  const homo     = p1 === p2;

  pdbV.removeAllModels();
  const model  = pdbV.addModel(pdbRaw, pdbFmtCur);
  const chains = [...new Set(model.selectedAtoms({}).map(a => a.chain))].sort();

  chains.forEach(ch => {
    const u = cur?.pdb_chain_map?.[ch];
    if      (u === p1)          pdbV.setStyle({chain: ch}, {cartoon: {color: CC[0]}});  // blue
    else if (!homo && u === p2) pdbV.setStyle({chain: ch}, {cartoon: {color: CC[2]}});  // green
    else if (hideOthers)        pdbV.setStyle({chain: ch}, {});                          // hidden
    else                        pdbV.setStyle({chain: ch}, {cartoon: {color: GRAY}});   // gray
  });
  pdbV.setHoverable({}, true, pdbHoverCB, pdbUnhoverCB);
  pdbV.zoomTo(); pdbV.render();
}

/* ── pair navigation ─────────────────────────────────────────────────── */
function buildPairButtons() {
  const list = document.getElementById('pair-list');
  list.innerHTML = '';
  const pairs = cur?.pairs || [];
  pairs.forEach((pair, i) => {
    const b      = document.createElement('button');
    b.className  = 'pair-btn';
    b.textContent = pair.label;
    const iptm   = pair.chain_pair_iptm_corrected;
    b.title      = iptm != null ? `ipTM corr: ${iptm.toFixed(3)}` : pair.pair;
    b.onclick    = () => selectPair(i);
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

  // Scroll active button into view (pair bar may overflow horizontally)
  document.querySelectorAll('.pair-btn')[idx]?.scrollIntoView({block: 'nearest', inline: 'nearest'});

  updateNavButtons();
  updateMetricsDisplay(idx);
  if (cfRaw)  highlightCFPair(pair.proteins);
  if (pdbRaw) highlightPDBPair(pair.proteins);
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

  el.innerHTML = [
    `<span class="m-label">ipTM corr:</span> ${fmt(pair.chain_pair_iptm_corrected)}`,
    `<span class="m-sep">|</span>`,
    `<span class="m-label">ipTM:</span> ${fmt(pair.chain_pair_iptm)}`,
    `<span class="m-sep">|</span>`,
    `<span class="m-label">PAE min:</span> ${fmt(pair.chain_pair_pae_min, 1)} Å`,
    `<span class="m-sep">|</span>`,
    `<span class="m-label">pLDDT:</span> ${plddtStr}`,
  ].join(' ');
}

/* ── load CF (from embedded gzip) ───────────────────────────────────── */
async function loadCF() {
  if (!cur) return;
  cfRaw = null;
  spin('cf-overlay', true, 'Loading CF model...');
  try {
    const text = await ungzip(cur.cf_gz);
    cfRaw = text;

    document.getElementById('cf-path').textContent = cur.complex_ac;

    /* If a pair is already selected (shouldn't happen on fresh load, but be safe),
       apply its highlighting; otherwise render by-chain and auto-select pair 0. */
    if (currentPairIdx >= 0 && cur.pairs[currentPairIdx]) {
      highlightCFPair(cur.pairs[currentPairIdx].proteins);
    } else {
      styleByChain(cfV, cfRaw, 'pdb', cfHoverCB, cfUnhoverCB);
      if (cur.pairs.length > 0) selectPair(0);   // auto-select first pair
    }
    spin('cf-overlay', false);
  } catch (e) {
    console.error(e);
    spin('cf-overlay', true, 'CF load error: ' + e.message);
    setTimeout(() => spin('cf-overlay', false), 4000);
  }
}

/* ── load PDB (RCSB, cache-first) ────────────────────────────────────── */
async function loadPDB(pid) {
  if (!pid) return;
  pdbRaw = null;

  const applyHighlight = () => {
    if (currentPairIdx >= 0 && cur.pairs[currentPairIdx]) {
      highlightPDBPair(cur.pairs[currentPairIdx].proteins);
    } else {
      styleByChain(pdbV, pdbRaw, pdbFmtCur, pdbHoverCB, pdbUnhoverCB);
    }
  };

  if (pdbCache[pid]) {
    const {text, fmt} = pdbCache[pid];
    pdbRaw = text; pdbFmtCur = fmt;
    applyHighlight();
    return;
  }

  const owner = pdbCacheOwner;
  spin('pdb-overlay', true, `Fetching ${pid.toUpperCase()} from RCSB…`);
  try {
    /* Try PDB format first; fall back to mmCIF (e.g. very large structures) */
    const r1 = await fetch(`https://files.rcsb.org/download/${pid.toUpperCase()}.pdb`);
    let text, fmt;
    if (r1.ok) {
      text = await r1.text(); fmt = 'pdb';
    } else {
      const r2 = await fetch(`https://files.rcsb.org/download/${pid.toUpperCase()}.cif`);
      if (!r2.ok) throw new Error(`RCSB 404 for ${pid} (.pdb and .cif)`);
      text = await r2.text(); fmt = 'mmcif';
    }
    if (owner !== pdbCacheOwner) return;   // user switched complex while fetching
    pdbCache[pid] = {text, fmt};
    pdbRaw = text; pdbFmtCur = fmt;
    applyHighlight();
    spin('pdb-overlay', false);
  } catch (e) {
    console.error(e);
    spin('pdb-overlay', true, `Could not load ${pid.toUpperCase()}: ${e.message}`);
    setTimeout(() => spin('pdb-overlay', false), 4000);
  }
}

/* ── select complex ──────────────────────────────────────────────────── */
function selectComplex(ac) {
  cur = COMPLEXES[ac];
  if (!cur) return;

  /* Reset everything */
  cfRaw = null; pdbRaw = null;
  currentPairIdx = -1;
  pdbCache       = {};
  pdbCacheOwner  = ac;

  /* Update topbar */
  document.getElementById('badge-cf').textContent  = `CF ${cur.cf_confidence?.toFixed(1) ?? '?'}`;
  document.getElementById('badge-n').textContent   = `n = ${cur.n_proteins}`;
  document.getElementById('badge-pdb').textContent = `PDB: ${cur.pdb_id.toUpperCase()}`;
  document.getElementById('badge-tm').textContent =
  cur.tm_score != null ? `TM ${cur.tm_score.toFixed(3)}` : 'TM ?';
  document.getElementById('pdb-title').textContent =
    `PDB Reference — ${cur.pdb_id.toUpperCase()}`;

  buildPairButtons();
  loadCF();
  loadPDB(cur.pdb_id);
}

/* ── sort helper ─────────────────────────────────────────────────────── */
function sortByTM(a, b) {
  const tmA = a.tm_score ?? -1;
  const tmB = b.tm_score ?? -1;
  return tmB - tmA || b.n_proteins - a.n_proteins || (b.cf_confidence ?? 0) - (a.cf_confidence ?? 0);
}

/* ── dropdown ────────────────────────────────────────────────────────── */
function buildDropdown() {
  const sel = document.getElementById('complex-select');
  /* Sort by TM score desc, then n_proteins desc, then CF confidence desc */
  Object.values(COMPLEXES)
    .sort(sortByTM)
    .forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.complex_ac;
      opt.textContent = [
        c.complex_ac,
        `n=${c.n_proteins}`,
        `PDB: ${c.pdb_id.toUpperCase()}`,
        `CF: ${c.cf_confidence?.toFixed(1) ?? '?'}`,
        `pairs: ${c.pairs.length}`,
        `TM: ${c.tm_score != null ? c.tm_score.toFixed(3) : '?'}`,
      ].join('  |  ');
      sel.appendChild(opt);
    });
  sel.onchange = () => selectComplex(sel.value);
}

/* ── toggle others ───────────────────────────────────────────────────── */
function applyHideToggle() {
  const btn = document.getElementById('toggle-others');
  btn.textContent = hideOthers ? 'Show others' : 'Hide others';
  btn.classList.toggle('active-toggle', hideOthers);
  /* Re-render only when a pair is already selected */
  if (currentPairIdx >= 0 && cur?.pairs[currentPairIdx]) {
    const proteins = cur.pairs[currentPairIdx].proteins;
    if (cfRaw)  highlightCFPair(proteins);
    if (pdbRaw) highlightPDBPair(proteins);
  }
}

document.getElementById('toggle-others').addEventListener('click', () => {
  hideOthers = !hideOthers;
  applyHideToggle();
});

/* ── keyboard shortcuts ──────────────────────────────────────────────── */
document.addEventListener('keydown', e => {
  /* Only fire when focus is not inside a text input or select */
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const n = cur?.pairs?.length ?? 0;
  if (e.key === 'ArrowLeft'  && currentPairIdx > 0)     selectPair(currentPairIdx - 1);
  if (e.key === 'ArrowRight' && currentPairIdx < n - 1) selectPair(currentPairIdx + 1);
  if (e.key === 'h' || e.key === 'H') { hideOthers = !hideOthers; applyHideToggle(); }
});

document.getElementById('prev-pair').onclick = () => selectPair(currentPairIdx - 1);
document.getElementById('next-pair').onclick = () => selectPair(currentPairIdx + 1);

/* ── init ────────────────────────────────────────────────────────────── */
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

# ── write output ───────────────────────────────────────────────────────────────

json_blob = json.dumps(EMBED, ensure_ascii=True)
html_out  = HTML.replace("%%JSON%%", json_blob)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html_out, encoding="utf-8")

size_kb = OUT.stat().st_size / 1024
print(f"\n  Written: {OUT}")
print(f"  Size:    {size_kb:.0f} KB  (~{size_kb/1024:.1f} MB)")