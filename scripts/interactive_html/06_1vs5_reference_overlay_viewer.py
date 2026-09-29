#!/usr/bin/env python
"""
Dual-panel viewer: CombFold output superposed on its US-align reference structure.

Left  panel = run with 1 AF3 input model per pair  (Pipeline/10_all_CP_complexes)
Right panel = run with 5 AF3 input models per pair (Pipeline/14_CF_multiple_models)

For each complex_ac (n_proteins > 2) present in BOTH runs, the CombFold output with
the highest CF_confidence is selected per run. Rather than re-running US-align, this
reuses the superposition already computed by the eval pipeline (X02/X03): for
combfold_output_path .../assembled_results/output_clustered_N.pdb, the matching
superposed structure lives at
.../usalign_outputs/usalign_outputs_predN/complex/usalign.pdb

Usage
-----
    python 06_1vs5_reference_overlay_viewer.py
    python 06_1vs5_reference_overlay_viewer.py --limit 20 -o /tmp/test.html
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from procompa import get_project_root

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"

ONE_MODEL_PARQUET = (
    DATA_DIR
    / "Pipeline/10_all_CP_complexes/cf_pdb_structure_similarity/cf_pdb_eval_all_metrics.parquet"
)
MULTI_MODEL_PARQUET = (
    DATA_DIR
    / "Pipeline/14_CF_multiple_models/cf_pdb_structure_similarity/cf_pdb_eval_all_metrics.parquet"
)

MIN_N_PROTEINS = 3  # i.e. n_proteins > 2

# Okabe-Ito, colour-blind safe
OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]
REFERENCE_COLOR = "#BBBBBB"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def add_cf_confidence(df: pl.DataFrame) -> pl.DataFrame:
    """Attach CombFold confidence by reading confidence.txt next to each output."""
    conf: dict[str, str] = {}
    for d in {Path(p).parent for p in df["combfold_output_path"]}:
        conf_file = d / "confidence.txt"
        assert conf_file.exists(), f"missing confidence.txt: {conf_file}"
        with open(conf_file) as fh:
            conf.update(dict(line.split() for line in fh if line.strip()))

    missing = [p for p in df["combfold_output_path"] if p not in conf]
    assert not missing, f"{len(missing)} outputs absent from confidence.txt, e.g. {missing[:3]}"

    return df.with_columns(
        pl.col("combfold_output_path")
        .map_elements(lambda p: float(conf[p]), return_dtype=pl.Float64)
        .alias("CF_confidence")
    )


def load_run(parquet: Path) -> pl.DataFrame:
    """
    Load one run, filter to n_proteins > 2, keep the highest-CF_confidence
    output per complex_ac.
    """
    assert parquet.exists(), f"parquet not found: {parquet}"
    df = pl.read_parquet(parquet)
    df = add_cf_confidence(df)
    df = df.filter(pl.col("n_proteins") > MIN_N_PROTEINS - 1)
    return df.sort("CF_confidence", descending=True).group_by("complex_ac").first()


# --------------------------------------------------------------------------- #
# Structure handling
# --------------------------------------------------------------------------- #

_CLUSTER_IDX_RE = re.compile(r"output_clustered_(\d+)\.pdb$")


def usalign_output_path(combfold_output_path: Path) -> Path:
    """
    Map a CombFold output to its precomputed US-align superposition.

    .../<run>_output/assembled_results/output_clustered_N.pdb
      -> .../<run>_output/usalign_outputs/usalign_outputs_predN/complex/usalign.pdb

    This reuses the superposition already produced by the eval pipeline (X02/X03)
    instead of re-running US-align and re-implementing the rotation ourselves.
    """
    m = _CLUSTER_IDX_RE.search(combfold_output_path.name)
    assert m, f"unexpected combfold_output_path name: {combfold_output_path.name}"
    run_dir = combfold_output_path.parent.parent  # .../<run>_output, up from assembled_results/
    return run_dir / "usalign_outputs" / f"usalign_outputs_pred{m.group(1)}" / "complex" / "usalign.pdb"


def read_reference(path: Path) -> tuple[str, str]:
    """Return (text, format) for the reference; keep mmCIF to survive >62 chains."""
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return path.read_text(), "cif"
    return path.read_text(), "pdb"


def pack(text: str) -> str:
    """gzip + base64 for embedding."""
    return base64.b64encode(gzip.compress(text.encode(), compresslevel=6)).decode()


# --------------------------------------------------------------------------- #
# Per-complex worker
# --------------------------------------------------------------------------- #


def build_panel(row: dict, label: str) -> dict:
    """Read the already-superposed CombFold output; return payload for the HTML."""
    model_path = Path(row["combfold_output_path"])
    ref_path = Path(row["reference_pdb_path"])
    assert ref_path.exists(), f"missing reference: {ref_path}"

    aligned_path = usalign_output_path(model_path)
    assert aligned_path.exists(), f"missing precomputed US-align output: {aligned_path}"

    ref_text, ref_fmt = read_reference(ref_path)

    return {
        "label": label,
        "tm": row["usalign_cpx_tm_score"],
        "cf_conf": row["CF_confidence"],
        "n_proteins": row["n_proteins"],
        "model_name": model_path.name,
        "ref_name": ref_path.name,
        "model": pack(aligned_path.read_text()),
        "model_fmt": "pdb",
        "ref": pack(ref_text),
        "ref_fmt": ref_fmt,
    }


def build_entry(complex_ac: str, row_one: dict, row_multi: dict) -> dict | None:
    try:
        return {
            "complex_ac": complex_ac,
            "left": build_panel(row_one, "1 input model"),
            "right": build_panel(row_multi, "5 input models"),
        }
    except Exception as exc:  # noqa: BLE001 - report and skip, do not abort the batch
        print(f"  SKIP {complex_ac}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CombFold 1 vs 5 input models - reference overlay</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666666;
    --line: #dddddd; --panel: #fafafa;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #1a1a1a; --fg: #eeeeee; --muted: #aaaaaa;
      --line: #3a3a3a; --panel: #242424;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 14px; }
  h1 { font-size: 17px; margin: 0; font-weight: 600; }
  select, button {
    font: inherit; padding: 6px 10px; border: 1px solid var(--line);
    border-radius: 6px; background: var(--panel); color: var(--fg);
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .panel { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--panel); }
  .panel h2 { font-size: 14px; margin: 0; padding: 9px 12px; border-bottom: 1px solid var(--line); font-weight: 600; }
  .viewer { position: relative; width: 100%; height: 520px; }
  .stats { display: flex; gap: 22px; padding: 10px 12px; border-top: 1px solid var(--line); flex-wrap: wrap; }
  .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  .stat-value { font-size: 19px; font-variant-numeric: tabular-nums; font-weight: 600; }
  .files { padding: 0 12px 10px; font-size: 11px; color: var(--muted); word-break: break-all; }
  .legend { display: flex; gap: 16px; align-items: center; font-size: 12px; color: var(--muted); margin-top: 12px; }
  .swatch { display: inline-block; width: 26px; height: 3px; vertical-align: middle; margin-right: 6px; }
</style>
</head>
<body>
<header>
  <h1>CombFold vs reference &mdash; 1 vs 5 AF3 input models</h1>
  <select id="picker"></select>
  <button id="prev">&larr;</button>
  <button id="next">&rarr;</button>
  <label style="font-size:12px;color:var(--muted)">
    <input type="checkbox" id="showref" checked> reference
  </label>
  <label style="font-size:12px;color:var(--muted)">
    <input type="checkbox" id="sync"> sync cameras
  </label>
  <span id="counter" style="font-size:12px;color:var(--muted)"></span>
</header>

<div class="grid">
  <div class="panel">
    <h2 id="left-title">1 input model</h2>
    <div class="viewer" id="left-viewer"></div>
    <div class="stats">
      <div><div class="stat-label">TM-score</div><div class="stat-value" id="left-tm">&ndash;</div></div>
      <div><div class="stat-label">CF confidence</div><div class="stat-value" id="left-conf">&ndash;</div></div>
      <div><div class="stat-label">Proteins</div><div class="stat-value" id="left-n">&ndash;</div></div>
    </div>
    <div class="files" id="left-files"></div>
  </div>
  <div class="panel">
    <h2 id="right-title">5 input models</h2>
    <div class="viewer" id="right-viewer"></div>
    <div class="stats">
      <div><div class="stat-label">TM-score</div><div class="stat-value" id="right-tm">&ndash;</div></div>
      <div><div class="stat-label">CF confidence</div><div class="stat-value" id="right-conf">&ndash;</div></div>
      <div><div class="stat-label">Proteins</div><div class="stat-value" id="right-n">&ndash;</div></div>
    </div>
    <div class="files" id="right-files"></div>
  </div>
</div>

<div class="legend">
  <span><span class="swatch" style="background:__REFCOLOR__"></span>reference (US-align target)</span>
  <span><span class="swatch" style="background:linear-gradient(90deg,#E69F00,#56B4E9,#009E73,#0072B2,#D55E00)"></span>CombFold model, coloured by chain</span>
</div>

<script>
const DATA = __DATA__;
const PALETTE = __PALETTE__;
const REFCOLOR = "__REFCOLOR__";

function inflate(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return pako.ungzip(bytes, { to: "string" });
}

const viewers = {
  left: $3Dmol.createViewer("left-viewer", { backgroundColor: "white" }),
  right: $3Dmol.createViewer("right-viewer", { backgroundColor: "white" }),
};

function paintPanel(side, panel, showRef) {
  const v = viewers[side];
  v.clear();

  if (showRef) {
    v.addModel(inflate(panel.ref), panel.ref_fmt);
    v.setStyle({ model: -1 }, { cartoon: { color: REFCOLOR, opacity: 0.55 } });
  }

  v.addModel(inflate(panel.model), panel.model_fmt);
  const chains = new Set();
  v.getModel(-1).selectedAtoms({}).forEach(a => chains.add(a.chain));
  [...chains].sort().forEach((c, i) => {
    v.setStyle({ model: -1, chain: c }, { cartoon: { color: PALETTE[i % PALETTE.length] } });
  });

  v.zoomTo();
  v.render();

  document.getElementById(side + "-title").textContent = panel.label;
  document.getElementById(side + "-tm").textContent = panel.tm.toFixed(3);
  document.getElementById(side + "-conf").textContent = panel.cf_conf.toFixed(1);
  document.getElementById(side + "-n").textContent = panel.n_proteins;
  document.getElementById(side + "-files").textContent =
    panel.model_name + "  vs  " + panel.ref_name;
}

let idx = 0;
function show(i) {
  idx = (i + DATA.length) % DATA.length;
  const entry = DATA[idx];
  const showRef = document.getElementById("showref").checked;
  paintPanel("left", entry.left, showRef);
  paintPanel("right", entry.right, showRef);
  document.getElementById("picker").value = String(idx);
  document.getElementById("counter").textContent = (idx + 1) + " / " + DATA.length;
  if (document.getElementById("sync").checked) linkCameras();
}

function linkCameras() {
  viewers.left.setView(viewers.right.getView());
  viewers.left.render();
}

const picker = document.getElementById("picker");
DATA.forEach((d, i) => {
  const o = document.createElement("option");
  o.value = String(i);
  o.textContent = d.complex_ac +
    "   (TM " + d.left.tm.toFixed(2) + " -> " + d.right.tm.toFixed(2) + ")";
  picker.appendChild(o);
});

picker.addEventListener("change", e => show(parseInt(e.target.value, 10)));
document.getElementById("prev").addEventListener("click", () => show(idx - 1));
document.getElementById("next").addEventListener("click", () => show(idx + 1));
document.getElementById("showref").addEventListener("change", () => show(idx));
document.getElementById("sync").addEventListener("change", e => { if (e.target.checked) linkCameras(); });
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") show(idx - 1);
  if (e.key === "ArrowRight") show(idx + 1);
});

show(0);
</script>
</body>
</html>
"""


def write_html(entries: list[dict], out_path: Path) -> None:
    html = (
        HTML_TEMPLATE.replace("__DATA__", json.dumps(entries))
        .replace("__PALETTE__", json.dumps(OKABE_ITO))
        .replace("__REFCOLOR__", REFERENCE_COLOR)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="only the first N complexes")
    ap.add_argument("--workers", type=int, default=32, help="I/O-bound now, safe to raise")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=DATA_DIR / "Pipeline/viewer/07_reference_overlay_viewer.html",
    )
    args = ap.parse_args()

    print("selecting the highest-CF_confidence output per complex")
    one = load_run(ONE_MODEL_PARQUET)
    multi = load_run(MULTI_MODEL_PARQUET)
    print(f"  1-model run : {one.height} complexes with n_proteins > 2")
    print(f"  5-model run : {multi.height} complexes with n_proteins > 2")

    one_rows = {r["complex_ac"]: r for r in one.iter_rows(named=True)}
    multi_rows = {r["complex_ac"]: r for r in multi.iter_rows(named=True)}

    shared = sorted(set(one_rows) & set(multi_rows))
    only_one = sorted(set(one_rows) - set(multi_rows))
    only_multi = sorted(set(multi_rows) - set(one_rows))
    if only_one:
        print(f"  WARNING: {len(only_one)} complexes only in 1-model run, e.g. {only_one[:5]}")
    if only_multi:
        print(f"  WARNING: {len(only_multi)} complexes only in 5-model run, e.g. {only_multi[:5]}")
    print(f"  shared      : {len(shared)}")

    if args.limit:
        shared = shared[: args.limit]

    entries: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(build_entry, ac, one_rows[ac], multi_rows[ac]): ac for ac in shared
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="loading superposed structures"):
            entry = fut.result()
            if entry is not None:
                entries.append(entry)

    assert entries, "no complexes could be loaded"
    entries.sort(key=lambda e: e["right"]["tm"] - e["left"]["tm"], reverse=True)

    write_html(entries, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {len(entries)} complexes -> {args.out}  ({size_mb:.1f} MB)")

    delta = np.array([e["right"]["tm"] - e["left"]["tm"] for e in entries])
    print(f"TM delta (5 models - 1 model): mean {delta.mean():+.3f}, median {np.median(delta):+.3f}")
    print(f"  improved: {(delta > 0.02).sum()}   unchanged: {(abs(delta) <= 0.02).sum()}   worse: {(delta < -0.02).sum()}")


if __name__ == "__main__":
    main()