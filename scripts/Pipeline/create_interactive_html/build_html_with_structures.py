"""
Build a single self-contained HTML file:
  - left: TM vs CF scatter, one point per complex (highest-CF-confidence output only)
  - right: a 3Dmol viewer that shows the ref+pred overlay for whichever point you click

Structures are cleaned (gemmi: drop waters/ligands/nucleic) and embedded as raw
PDB text directly in the HTML at build time — no live connection to the cluster
is needed once the file is downloaded, since nothing is fetched at view-time.

Run this ON THE CLUSTER (needs access to the actual structure files), then
download only the resulting HTML.
"""
import json
import polars as pl
import gemmi
from scipy.stats import spearmanr

PARQUET = "both_parts_benchmark_structure_similarity.parquet"
OUT_HTML = "tm_vs_cf_with_structures.html"


import gzip
import base64


def clean_structure_to_pdb_string(path: str, backbone_only: bool = True) -> str:
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_ligands_and_waters()
    for model in st:
        drop = [
            i for i, chain in enumerate(model)
            if chain.get_polymer().check_polymer_type() in (
                gemmi.PolymerType.Dna, gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid,
            )
        ]
        for i in reversed(drop):
            del model[i]
    st.remove_empty_chains()

    if backbone_only:
        # cartoon rendering only needs N, CA, C, O — side chains are pure bytes with
        # zero visual effect on a ribbon, and are most of the file size
        keep = {"N", "CA", "C", "O"}
        for model in st:
            for chain in model:
                for residue in chain:
                    drop_atoms = [i for i, a in enumerate(residue) if a.name not in keep]
                    for i in reversed(drop_atoms):
                        del residue[i]

    return st.make_pdb_string()


def compress_for_embed(pdb_text: str) -> str:
    """gzip + base64, decompressed client-side with pako.js — ~20x smaller in the HTML."""
    return base64.b64encode(gzip.compress(pdb_text.encode())).decode()


# ---- 1. select one row per complex: the highest-CF-confidence output ----
df = (
    pl.read_parquet(PARQUET)
    .filter(pl.col("n_proteins") > 2)
    .with_columns(usalign_pdb_path=pl.col("usalign_pred_dir") + "/complex/usalign.pdb")
    .sort("CF_confidence", descending=True)
    .group_by("complex_ac")
    .head(1)
)
n = df.height
rho, pval = spearmanr(df["CF_confidence"], df["usalign_cpx_tm_score"])
print(f"n={n} (one row per complex, highest CF)  rho={rho:.3f}  p={pval:.2e}")

# ---- 2. embed cleaned structures per complex, keyed by complex_ac ----
structure_data = {}
for row in df.iter_rows(named=True):
    try:
        ref_pdb = clean_structure_to_pdb_string(row["reference_pdb_path"])
        pred_pdb = clean_structure_to_pdb_string(row["usalign_pdb_path"])
    except Exception as e:
        print(f"  [WARN] skipping {row['complex_ac']}: {e}")
        continue
    structure_data[row["complex_ac"]] = {
        "ref": compress_for_embed(ref_pdb),
        "pred": compress_for_embed(pred_pdb),
        "chain_mapping": row["chain_mapping"],  # JSON string: {uniprot: [[cf_chains],[ref_chains]]}
    }

print(f"embedded structures for {len(structure_data)}/{n} complexes")

# ---- 3. build the plotly scatter (as an embeddable div, not full page) ----
import plotly.graph_objects as go

customdata = df.select(["complex_ac", "usalign_cpx_tm_score", "CF_confidence"]).rows()
fig = go.Figure(
    go.Scatter(
        x=df["CF_confidence"], y=df["usalign_cpx_tm_score"],
        mode="markers", marker=dict(size=10, color="#1f77b4", line=dict(width=0.5, color="white")),
        customdata=customdata,
        hovertemplate="<b>%{customdata[0]}</b><br>TM: %{customdata[1]:.2f}   CF: %{customdata[2]:.0f}<extra></extra>",
    )
)
fig.update_layout(
    title=f"TM vs CF confidence — one point per complex (highest CF output)<br><sup>n={n}, Spearman ρ={rho:.2f}</sup>",
    xaxis_title="CF confidence", yaxis_title="TM score",
    template="simple_white", width=650, height=600,
    margin=dict(t=80),
)
plot_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id="scatterDiv")

# ---- 4. assemble the final page: plot (left) + 3Dmol viewer (right) ----
page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TM vs CF confidence with structures</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  #container {{ display: flex; gap: 24px; align-items: flex-start; }}
  #viewerPane {{ display: flex; flex-direction: column; gap: 8px; }}
  #viewerDiv {{ width: 650px; height: 600px; border: 1px solid #ccc; position: relative; }}
  #infoBox {{ font-size: 14px; color: #333; }}
  #legend span {{ font-weight: bold; }}
  #legend .ref {{ color: #0072B2; }}
  #legend .pred {{ color: #E69F00; }}
</style>
</head>
<body>
<h3>Click a point to load its structure overlay</h3>
<div id="container">
  <div id="plotPane">{plot_html}</div>
  <div id="viewerPane">
    <div id="infoBox">Click a point on the left to load a structure.</div>
    <div id="colorControls">
      <label><input type="radio" name="colorMode" value="flat" checked> Flat (ref blue / pred orange)</label>
      &nbsp;&nbsp;
      <label><input type="radio" name="colorMode" value="chain"> By chain</label>
    </div>
    <div id="opacityControl">
      <label>Reference opacity: <span id="opacityValue">100%</span></label><br>
      <input type="range" id="refOpacitySlider" min="0" max="100" value="100" style="width: 300px;">
    </div>
    <div id="viewerDiv"></div>
    <div id="legend"><span class="ref">■ blue</span> = reference (PDB) &nbsp;&nbsp; <span class="pred">■ orange</span> = prediction (CF)</div>
  </div>
</div>

<script>
const structureData = {json.dumps(structure_data)};

let viewer = $3Dmol.createViewer(document.getElementById("viewerDiv"), {{backgroundColor: "white"}});

function decompress(b64) {{
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return pako.inflate(bytes, {{ to: "string" }});
}}

let currentRefPdb = null;
let currentPredPdb = null;

// one base hue per protein; reference = pale/desaturated ("paper" style),
// CF prediction = same hue, slightly richer/more saturated
const baseHues = [0, 210, 40, 130, 280, 20, 170, 320, 60, 250]; // spread around the wheel

function hslToHex(h, s, l) {{
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const toHex = x => Math.round(255 * x).toString(16).padStart(2, "0");
  return "#" + toHex(f(0)) + toHex(f(8)) + toHex(f(4));
}}

function buildColorMapsFromChainMapping(chainMappingJson) {{
  const predColor = {{}};
  const refColor = {{}};
  let mapping;
  try {{
    mapping = JSON.parse(chainMappingJson);
  }} catch (e) {{
    return {{predColor, refColor}};  // fall back to defaults below if parsing fails
  }}
  const proteins = Object.keys(mapping);
  proteins.forEach((protein, idx) => {{
    const hue = baseHues[idx % baseHues.length];
    const [cfChains, refChains] = mapping[protein];
    const predHex = hslToHex(hue, 70, 45);   // slightly stronger/vivid
    const refHex = hslToHex(hue, 35, 78);    // pale, "paper"-style
    cfChains.forEach(ch => {{ predColor[ch] = predHex; }});
    refChains.forEach(ch => {{ refColor[ch] = refHex; }});
  }});
  return {{predColor, refColor}};
}}

function makeMappedColorFunc(colorMap, fallback) {{
  return function(atom) {{
    return colorMap[atom.chain] || fallback;
  }};
}}

let currentMode = "flat";
let currentRefOpacity = 1.0;
let currentPredColorMap = {{}};
let currentRefColorMap = {{}};

function applyStyle() {{
  if (currentRefPdb === null) return;
  if (currentMode === "chain") {{
    viewer.setStyle({{model: 0}}, {{cartoon: {{colorfunc: makeMappedColorFunc(currentRefColorMap, "#cccccc"), opacity: currentRefOpacity}}}});
    viewer.setStyle({{model: 1}}, {{cartoon: {{colorfunc: makeMappedColorFunc(currentPredColorMap, "#999999")}}}});
  }} else {{
    viewer.setStyle({{model: 0}}, {{cartoon: {{color: "0x0072B2", opacity: currentRefOpacity}}}});
    viewer.setStyle({{model: 1}}, {{cartoon: {{color: "0xE69F00"}}}});
  }}
  viewer.render();
}}

document.querySelectorAll('input[name="colorMode"]').forEach(el => {{
  el.addEventListener("change", (e) => {{ currentMode = e.target.value; applyStyle(); }});
}});

document.getElementById("refOpacitySlider").addEventListener("input", (e) => {{
  currentRefOpacity = e.target.value / 100;
  document.getElementById("opacityValue").textContent = e.target.value + "%";
  applyStyle();
}});

function loadComplex(complexAc, tm, cf) {{
  const info = document.getElementById("infoBox");
  const data = structureData[complexAc];
  if (!data) {{
    info.innerHTML = `<b>${{complexAc}}</b> — no structure embedded for this point.`;
    return;
  }}
  info.innerHTML = `<b>${{complexAc}}</b>  TM=${{tm.toFixed(2)}}  CF=${{cf.toFixed(0)}}  (backbone-only, decompressing…)`;
  currentRefPdb = decompress(data.ref);
  currentPredPdb = decompress(data.pred);
  const maps = buildColorMapsFromChainMapping(data.chain_mapping);
  currentPredColorMap = maps.predColor;
  currentRefColorMap = maps.refColor;
  viewer.clear();
  viewer.addModel(currentRefPdb, "pdb");
  viewer.addModel(currentPredPdb, "pdb");
  applyStyle();
  viewer.zoomTo();
  viewer.render();
  info.innerHTML = `<b>${{complexAc}}</b>  TM=${{tm.toFixed(2)}}  CF=${{cf.toFixed(0)}}`;
}}

document.getElementById("scatterDiv").on("plotly_click", function(evt) {{
  const pt = evt.points[0];
  const [complexAc, tm, cf] = pt.customdata;
  loadComplex(complexAc, tm, cf);
}});
</script>
</body>
</html>
"""

with open(OUT_HTML, "w") as f:
    f.write(page)

import os
print(f"wrote {OUT_HTML}  ({os.path.getsize(OUT_HTML)/1e6:.1f} MB)")