#!/usr/bin/env python3
"""
Trimmed YeastMap CombFold viewer, compared to the untrimmed run or to the CP core complex.

Dropdown : YeastMap complexes whose best trimmed output (over pred_1/2/3) has confidence > CONF_THRESHOLD
Left     : trimmed run, best output per predicted stoichiometry, arrows step through the
           stoichiometries from highest to lowest confidence
Right    : switch between
           - untrimmed : best output of the SAME stoichiometry (same output folder name)
           - CP core   : the Complex Portal complex the YeastMap prediction is based on (input_12_yeastmap.tsv),
                         stoichiometry closest to the left one (shared proteins), ties -> highest confidence
Colors   : by chain | uniform | trimmed mapping | CP core proteins
Click    : on an atom -> chain, UniProt, residue, pLDDT
"""
import base64
import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from procompa import get_project_root
from tqdm import tqdm

DATA = get_project_root() / "data"
TRIMMED_RUN = "22_YM_pae_trimmed"
UNTRIMMED_RUN = "21_YM_pae"
CP_RUN = "23_all_CP_pae"
YEASTMAP_TSV = DATA / "Pipeline/12_Yeastmap/input_12_yeastmap.tsv"   # YeastMap complex -> CP complex
CONF_THRESHOLD = 80
OUT = DATA / "Pipeline/viewer/09_trimmed_vs_untrimmed_viewer.html"
BACKBONE_ATOMS = {"N", "CA", "C", "O"}   # enough for a cartoon, ~5x smaller than full atom
N_THREADS = 16                           # file reads are the slow part -> read complexes in parallel


def results_csv(run: str) -> Path:
    return DATA / "Pipeline" / run / f"all_pdb_present_{run}_pool_pipeline_complexes_combfold_results.csv"


for required_path in (results_csv(TRIMMED_RUN), results_csv(CP_RUN), YEASTMAP_TSV,
                      DATA / "Pipeline" / UNTRIMMED_RUN / "CombFold"):
    assert required_path.exists(), f"missing: {required_path}"


# ── helpers ───────────────────────────────────────────────────────────────────

def output_folder_name(protein_counts: dict[str, int]) -> str:
    """Same naming as the pipeline: 'P1x1_P2x2_pool_output' (hashed if > 200 chars)."""
    complex_name = "_".join(f"{p}x{protein_counts[p]}" for p in sorted(protein_counts))
    if len(complex_name) > 200:
        complex_name = complex_name[:180] + "_" + hashlib.sha1(complex_name.encode()).hexdigest()[:12]
    return f"{complex_name}_pool_output"


def backbone_pdb(pdb_text: str) -> str:
    return "\n".join(line for line in pdb_text.splitlines()
                     if line.startswith("ATOM") and line[12:16].strip() in BACKBONE_ATOMS)


def compress(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode(), compresslevel=6)).decode()


def csv_best_confidence(col: str) -> pl.Expr:
    """'{0: 78.5017, 1: 78.8618}' -> 78.8618; '{}' -> null"""
    return (pl.col(col).str.extract_all(r": -?[\d.]+")
            .list.eval(pl.element().str.slice(2).cast(pl.Float64)).list.max())


def parse_pred(row: dict, pred: int) -> tuple[dict[str, int], dict]:
    """pred_k = '{"P1":1,"P2":2},{"rank":1,"n_copies":3,"probability":0.33}' -> (counts, info)"""
    protein_counts, pred_info = json.loads("[" + row[f"pred_{pred}"] + "]")
    assert pred_info["rank"] == pred, f"{row['complex_ac']} pred_{pred}: rank is {pred_info['rank']}"
    return protein_counts, pred_info


def identifier_counts(identifiers: str) -> dict[str, int]:
    """'P1(2)|P2(0)|CHEBI:123(1)' -> {'P1': 2, 'P2': 0} (small molecules / RNA dropped, 0 = unknown)"""
    protein_counts = {}
    for token in identifiers.split("|"):
        if token.startswith(("CHEBI", "URS")):
            continue
        token_match = re.match(r"^([A-Za-z0-9_-]+)\((\d+)\)$", token.strip())
        assert token_match, f"unparseable identifier {token!r} in {identifiers!r}"
        protein_counts[token_match.group(1)] = int(token_match.group(2))
    return protein_counts


def read_chain_list(output_dir: Path) -> dict[str, str]:
    """CombFold's chain.list: one 'UNIPROT_CHAIN.pdb' per line -> {chain: uniprot}."""
    chain_list_path = output_dir / "_unified_representation" / "assembly_output" / "chain.list"
    assert chain_list_path.exists(), f"chain.list missing: {chain_list_path}"
    chain_to_protein = {}
    for line in filter(None, map(str.strip, chain_list_path.read_text().splitlines())):
        chain_match = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)\.pdb$", line)
        assert chain_match, f"unparseable chain.list line in {chain_list_path}: {line!r}"
        chain_to_protein[chain_match.group(2)] = chain_match.group(1)
    return chain_to_protein


def read_best_model(run: str, folder: str) -> dict | None:
    """Highest-confidence output of one CombFold folder; None if the folder has no outputs."""
    output_dir = DATA / "Pipeline" / run / "CombFold" / folder
    confidence_path = output_dir / "assembled_results" / "confidence.txt"
    if not confidence_path.exists():
        return None
    scored_lines = [line.split() for line in confidence_path.read_text().splitlines() if line.strip()]
    if not scored_lines:
        return None
    assert all(len(fields) == 2 for fields in scored_lines), f"malformed {confidence_path}"
    best_path, best_conf = max(((Path(p), float(s)) for p, s in scored_lines), key=lambda model: model[1])
    model_path = output_dir / "assembled_results" / best_path.name
    assert model_path.exists(), f"best model missing: {model_path}"

    model_text = backbone_pdb(model_path.read_text())
    chain_to_protein = read_chain_list(output_dir)
    model_chains = list(dict.fromkeys(line[21] for line in model_text.splitlines()))   # order of appearance
    assert set(model_chains) <= set(chain_to_protein), f"{model_path}: chains {set(model_chains) - set(chain_to_protein)} not in chain.list"
    plddts = [float(line[60:66]) for line in model_text.splitlines()]
    assert 0 <= min(plddts) and max(plddts) <= 100 and max(plddts) > 0, (
        f"{model_path}: B-factor column doesn't look like pLDDT (min {min(plddts)}, max {max(plddts)})")
    return {"conf": best_conf, "n_outputs": len(scored_lines), "folder": folder, "model": best_path.name,
            "model_chains": model_chains,
            "chains": {chain: chain_to_protein[chain] for chain in model_chains},
            "pdb_gz": compress(model_text)}


def core_candidates(cp_row: dict) -> dict[str, dict]:
    """All CP core stoichiometries with outputs: pred_1/2/3 + true (if run) -> {folder: {"labels", "counts", "model"}}."""
    candidates = {}
    for pred in (1, 2, 3):
        if cp_row[f"CF_{pred}_n_assemblies"] == 0:
            continue
        protein_counts, _ = parse_pred(cp_row, pred)
        folder = output_folder_name(protein_counts)
        if folder in candidates:
            candidates[folder]["labels"].append(f"pred {pred}")
            continue
        model = read_best_model(CP_RUN, folder)
        assert model is not None, f"{cp_row['complex_ac']} pred {pred}: no outputs in {CP_RUN}/{folder}"
        assert model["n_outputs"] == cp_row[f"CF_{pred}_n_assemblies"], f"{cp_row['complex_ac']} pred {pred}: output count differs from csv"
        candidates[folder] = {"labels": [f"pred {pred}"], "counts": protein_counts, "model": model}

    # true stoichiometry: unknown '(0)' counts were run as 1 copy (e.g. CPX-474: P07213(0) -> P07213x1)
    true_counts = {protein: n if n > 0 else 1 for protein, n in identifier_counts(cp_row["identifiers"]).items()}
    true_folder = output_folder_name(true_counts)
    if true_folder in candidates:
        candidates[true_folder]["labels"].append("true")
    elif cp_row["CF_true_n_assemblies"] > 0:
        model = read_best_model(CP_RUN, true_folder)
        assert model is not None, f"{cp_row['complex_ac']} true: no outputs in {CP_RUN}/{true_folder}"
        assert model["n_outputs"] == cp_row["CF_true_n_assemblies"], f"{cp_row['complex_ac']} true: output count differs from csv"
        candidates[true_folder] = {"labels": ["true"], "counts": true_counts, "model": model}
    return candidates


def closest_core(trimmed_counts: dict[str, int], candidates: dict[str, dict], core_proteins: set[str]) -> tuple[str, int]:
    """Core stoichiometry with the fewest copy differences on the shared proteins; ties -> highest confidence."""
    shared_proteins = set(trimmed_counts) & core_proteins
    def copy_difference(candidate: dict) -> int:
        return sum(abs(trimmed_counts[p] - candidate["counts"].get(p, 0)) for p in shared_proteins)
    best_folder = min(candidates, key=lambda f: (copy_difference(candidates[f]), -candidates[f]["model"]["conf"]))
    return best_folder, copy_difference(candidates[best_folder])


# ── load tables ───────────────────────────────────────────────────────────────

trimmed_results = pl.read_csv(results_csv(TRIMMED_RUN))
assert trimmed_results["complex_ac"].is_unique().all(), f"{TRIMMED_RUN}: more than one row per complex"
cp_results = pl.read_csv(results_csv(CP_RUN))
assert cp_results["complex_ac"].is_unique().all(), f"{CP_RUN}: more than one row per complex"
cp_row_by_ac = {cp_row["complex_ac"]: cp_row for cp_row in cp_results.iter_rows(named=True)}

yeastmap = pl.read_csv(YEASTMAP_TSV, separator="\t", infer_schema_length=0)
assert yeastmap["#Complex ac"].is_unique().all(), "YeastMap complex listed twice in tsv"
cp_ac_by_ym_ac = dict(zip(yeastmap["#Complex ac"], yeastmap["#Complex_ac_db"]))

# filter on the confidences already in the csv -> only read PDBs of complexes that end up in the viewer
confident_complexes = (
    trimmed_results
    .with_columns(*[csv_best_confidence(f"CF_{pred}_confidence").alias(f"best_conf_{pred}") for pred in (1, 2, 3)])
    .filter(pl.max_horizontal("best_conf_1", "best_conf_2", "best_conf_3") > CONF_THRESHOLD)
)
assert confident_complexes.height, f"no complex with trimmed confidence > {CONF_THRESHOLD}"
missing_cp = [ac for ac in confident_complexes["complex_ac"] if cp_ac_by_ym_ac.get(ac) not in cp_row_by_ac]
assert not missing_cp, f"no CP core complex (tsv or {CP_RUN} csv) for: {missing_cp}"
print(f"{confident_complexes.height}/{trimmed_results.height} complexes with trimmed conf > {CONF_THRESHOLD}")


# ── per complex ───────────────────────────────────────────────────────────────

def build_complex_entry(row: dict) -> dict:
    cp_row = cp_row_by_ac[cp_ac_by_ym_ac[row["complex_ac"]]]
    core_proteins = set(identifier_counts(cp_row["identifiers"]))
    candidates = core_candidates(cp_row)

    stois_by_folder = {}   # two preds can be the same stoichiometry -> one entry, both pred numbers
    for pred in (1, 2, 3):
        if row[f"CF_{pred}_n_assemblies"] == 0:
            continue
        protein_counts, pred_info = parse_pred(row, pred)
        folder = output_folder_name(protein_counts)
        if folder in stois_by_folder:
            stois_by_folder[folder]["preds"].append(pred)
            continue
        trimmed_model = read_best_model(TRIMMED_RUN, folder)
        assert trimmed_model is not None, f"{row['complex_ac']} pred {pred}: no outputs in {folder}"
        assert trimmed_model["n_outputs"] == row[f"CF_{pred}_n_assemblies"], f"{row['complex_ac']} pred {pred}: output count differs from csv"
        assert abs(trimmed_model["conf"] - row[f"best_conf_{pred}"]) < 1e-3, (
            f"{row['complex_ac']} pred {pred}: confidence.txt {trimmed_model['conf']} != csv {row[f'best_conf_{pred}']}")

        left_proteins = [trimmed_model["chains"][chain] for chain in trimmed_model["model_chains"]]
        core_folder, core_difference = closest_core(protein_counts, candidates, core_proteins) if candidates else (None, None)
        stois_by_folder[folder] = {
            "preds": [pred],
            "probability": pred_info["probability"],   # of the first (highest-ranked) pred with this stoi
            "counts": protein_counts,
            "interesting": any(n > 1 for n in protein_counts.values()),
            "trimmed": trimmed_model,
            "untrimmed": read_best_model(UNTRIMMED_RUN, folder),
            "core_folder": core_folder,
            "core_difference": core_difference,       # summed copy-number difference on shared proteins
            "core_proteins_in_left": len(set(left_proteins) & core_proteins),
            "core_chains_in_left": sum(p in core_proteins for p in left_proteins),
        }

    stois = sorted(stois_by_folder.values(), key=lambda stoi: -stoi["trimmed"]["conf"])
    used_core_folders = {stoi["core_folder"] for stoi in stois}
    return {
        "complex_ac": row["complex_ac"],
        "identifiers": row["identifiers"],
        "cp_complex_ac": cp_row["complex_ac"],
        "core_proteins": sorted(core_proteins),
        "core_models": {folder: c for folder, c in candidates.items() if folder in used_core_folders},
        "stois": stois,
    }


with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    viewer_complexes = list(tqdm(executor.map(build_complex_entry, confident_complexes.iter_rows(named=True)),
                                 total=confident_complexes.height, desc="Reading models"))

viewer_complexes.sort(key=lambda c: -c["stois"][0]["trimmed"]["conf"])
all_stois = [stoi for c in viewer_complexes for stoi in c["stois"]]
print(f"{len(all_stois)} stoichiometries")
print(f"  {sum(s['untrimmed'] is None for s in all_stois)} without untrimmed output -> right panel empty")
print(f"  {sum(s['core_folder'] is None for s in all_stois)} without any CP core output -> right panel empty")
print(f"  {sum(s['core_difference'] == 0 for s in all_stois)} with the same stoi in CP core (shared proteins)")


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trimmed YeastMap CombFold viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
* { box-sizing:border-box; margin:0; }
body { font-family:-apple-system,"Segoe UI",sans-serif; font-size:13px; background:#f0f0f0;
       padding:10px; height:100vh; display:flex; flex-direction:column; gap:8px; }
.topbar, .panel { background:white; border:1px solid #ddd; border-radius:6px; }
.topbar { padding:8px 12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
#complex-select { flex:1; max-width:600px; padding:3px; }
.panels { display:grid; grid-template-columns:1fr 1fr; gap:8px; flex:1; min-height:0; }
.panel { display:flex; flex-direction:column; overflow:hidden; }
.panel-hdr { padding:7px 12px; border-bottom:1px solid #eee; display:flex; gap:8px;
             align-items:center; flex-wrap:wrap; }
.panel-hdr h3 { font-size:13px; }
.stoi { flex-basis:100%; font-size:11px; color:#666; font-family:monospace; }
.badge { display:inline-block; margin:1px 0; padding:1px 9px; border-radius:10px; font-size:11px;
         font-weight:600; background:#ddeeff; color:#004a80; }
.badge.interesting { background:#ffe2c2; color:#8a4300; }
.badge.core { background:#d5f0ec; color:#1d6b61; }
.tab { padding:2px 9px; border:1px solid #ccc; border-radius:4px; background:#f2f2f2; cursor:pointer; font-weight:600; }
.tab.active { background:#0072B2; color:white; border-color:#0072B2; }
.body { flex:1; position:relative; min-height:0; }
.viewer { width:100%; height:100%; }
.folder, .click-info { position:absolute; bottom:6px; z-index:5; font-size:10px; font-family:monospace;
                       background:rgba(255,255,255,.85); padding:2px 6px; }
.folder { left:8px; color:#777; }
.click-info { right:8px; color:#222; font-size:12px; }
.empty { position:absolute; inset:0; display:none; align-items:center; justify-content:center; color:#888; }
.empty.on { display:flex; }
</style></head>
<body>

<div class="topbar">
  <button id="prev-complex">&#9664;</button>
  <select id="complex-select"></select>
  <button id="next-complex">&#9654;</button>
  <label>Color <select id="color-mode">
    <option value="chain">by chain</option>
    <option value="uniform">uniform</option>
    <option value="mapping">trimmed mapping</option>
    <option value="core">CP core proteins</option>
  </select></label>
  <label>pLDDT &ge; <input type="number" id="plddt-threshold" value="70" min="0" max="100" step="1" style="width:55px" disabled></label>
  <label><input type="checkbox" id="show-all" checked> show all</label>
</div>

<div class="panels">
  <div class="panel">
    <div class="panel-hdr">
      <h3>Trimmed</h3>
      <button id="prev-stoi">&#9664;</button><button id="next-stoi">&#9654;</button>
      <span id="left-stats"></span>
      <span id="left-stoi" class="stoi"></span>
    </div>
    <div class="body">
      <div id="left-viewer" class="viewer"></div>
      <span id="left-folder" class="folder"></span><span id="left-click" class="click-info"></span>
    </div>
  </div>
  <div class="panel">
    <div class="panel-hdr">
      <button class="tab active" data-mode="untrimmed">Untrimmed</button>
      <button class="tab" data-mode="core">CP core</button>
      <span id="right-stats"></span>
      <span id="right-stoi" class="stoi"></span>
    </div>
    <div class="body">
      <div id="right-viewer" class="viewer"></div>
      <span id="right-folder" class="folder"></span><span id="right-click" class="click-info"></span>
      <div id="right-empty" class="empty"></div>
    </div>
  </div>
</div>

<script>
const COMPLEXES = %%JSON%%;
const BLUE = '#1f77b4', YELLOW = '#f2c94c', CORE_COLOR = '#2a9d8f';
const RANK_NAMES = ['most likely stoi', '2nd most likely stoi', '3rd most likely stoi'];
const leftViewer  = $3Dmol.createViewer('left-viewer',  {backgroundColor: 'white'});
const rightViewer = $3Dmol.createViewer('right-viewer', {backgroundColor: 'white'});
const $ = id => document.getElementById(id);

let complexIndex = 0, complex = null, stoiIndex = 0, rightMode = 'untrimmed';
let leftModel = null, rightModel = null;     // models currently drawn (chain -> UniProt for colors + clicks)
let leftLoadId = 0, rightLoadId = 0;         // ignore results of older clicks

async function ungzip(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
}

function badge(text, cls = '') { return `<span class="badge ${cls}">${text}</span> `; }
function stoiText(counts) { return Object.keys(counts).sort().map(p => p + 'x' + counts[p]).join('  '); }
function chainStats(model) {
  const nUnique = new Set(model.model_chains.map(c => model.chains[c])).size;
  return badge(model.model_chains.length + ' chains') + badge(nUnique + ' unique proteins');
}

/* what the right panel shows: untrimmed model of the same stoi, or the matched CP core model */
function rightInfo() {
  const stoi = complex.stois[stoiIndex];
  if (rightMode === 'untrimmed') return stoi.untrimmed ? {model: stoi.untrimmed, counts: stoi.counts} : null;
  return complex.core_models[stoi.core_folder] || null;
}

function updateStats() {
  const stoi = complex.stois[stoiIndex];
  const trimmed = stoi.trimmed;
  const rank = badge(RANK_NAMES[stoiIndex] + ` (${stoiIndex + 1}/${complex.stois.length})`);
  const interesting = stoi.interesting ? badge('interesting stoi', 'interesting') : '';
  const core = complex.core_models[stoi.core_folder];

  $('left-stats').innerHTML = badge('CF ' + trimmed.conf.toFixed(2)) + badge('pred ' + stoi.preds.join('/'))
    + badge('p = ' + stoi.probability.toFixed(2)) + rank + badge(trimmed.n_outputs + ' outputs')
    + chainStats(trimmed) + interesting
    + badge(`CP core proteins: ${stoi.core_proteins_in_left}/${complex.core_proteins.length}`, 'core')
    + badge(`their chains: ${stoi.core_chains_in_left}` + (core ? ` (CP core: ${core.model.model_chains.length})` : ''), 'core');
  $('left-stoi').textContent = stoiText(stoi.counts);
  $('left-folder').textContent = trimmed.folder + ' / ' + trimmed.model;

  const right = rightInfo();
  if (!right) {
    $('right-stats').innerHTML = '';
    $('right-empty').textContent = rightMode === 'core' ? 'no CP core output for ' + complex.cp_complex_ac
                                                        : 'no untrimmed output for this stoichiometry';
  } else if (rightMode === 'untrimmed') {
    const delta = right.model.conf - trimmed.conf;
    $('right-stats').innerHTML = badge(`CF ${right.model.conf.toFixed(2)} (${delta >= 0 ? '+' : ''}${delta.toFixed(2)} vs trimmed)`)
      + badge('pred ' + stoi.preds.join('/')) + rank + badge(right.model.n_outputs + ' outputs')
      + chainStats(right.model) + interesting;
  } else {
    const match = stoi.core_difference === 0 ? 'same stoi as left (shared proteins)'
                                             : `closest stoi (${stoi.core_difference} copies differ)`;
    $('right-stats').innerHTML = badge('CF ' + right.model.conf.toFixed(2))
      + badge(complex.cp_complex_ac + ' ' + right.labels.join('/'), 'core') + badge(match, 'core')
      + badge(right.model.n_outputs + ' outputs') + chainStats(right.model)
      + (Object.values(right.counts).some(n => n > 1) ? badge('interesting stoi', 'interesting') : '');
  }
  $('right-stoi').textContent = right ? stoiText(right.counts) : '';
  $('right-folder').textContent = right ? right.model.folder + ' / ' + right.model.model : '';
  $('right-empty').classList.toggle('on', !right);
}

function onAtomClick(side, model, atom) {
  $(side + '-click').textContent =
    `chain ${atom.chain}  ${model.chains[atom.chain]}  ${atom.resn} ${atom.resi}  pLDDT ${atom.b.toFixed(1)}`;
}

async function drawModel(viewer, model, side) {
  const text = model ? await ungzip(model.pdb_gz) : null;
  return () => {   // drawing is returned as a function so a stale load can be dropped before touching the viewer
    viewer.clear();
    $(side + '-click').textContent = '';
    if (!model) return;
    viewer.addModel(text, 'pdb');
    viewer.setClickable({}, true, atom => onAtomClick(side, model, atom));
  };
}

async function loadLeft() {
  const id = ++leftLoadId, model = complex.stois[stoiIndex].trimmed;
  const draw = await drawModel(leftViewer, model, 'left');
  if (id !== leftLoadId) return false;
  draw(); leftModel = model;
  return true;
}

async function loadRight() {
  const id = ++rightLoadId, right = rightInfo(), model = right ? right.model : null;
  const draw = await drawModel(rightViewer, model, 'right');
  if (id !== rightLoadId) return false;
  draw(); rightModel = model;
  return true;
}

function applyColors() {
  if (!leftModel) return;
  const mode = $('color-mode').value;
  const coreProteins = new Set(complex.core_proteins);
  // residues are matched by (UniProt, residue number), so all copies of a protein are treated the same
  const trimmedResidues = new Set(leftViewer.selectedAtoms({atom: 'CA'}).map(a => leftModel.chains[a.chain] + ':' + a.resi));
  for (const [viewer, model, isLeft] of [[leftViewer, leftModel, true], [rightViewer, rightModel, false]]) {
    if (!model) continue;
    let cartoon;
    if (mode === 'chain')        cartoon = {colorscheme: 'chain'};
    else if (mode === 'uniform') cartoon = {color: YELLOW};
    else if (mode === 'mapping') cartoon = isLeft ? {color: BLUE}
      : {colorfunc: a => trimmedResidues.has(model.chains[a.chain] + ':' + a.resi) ? BLUE : YELLOW};
    else cartoon = {colorfunc: a => coreProteins.has(model.chains[a.chain]) ? CORE_COLOR : YELLOW};
    viewer.setStyle({}, {cartoon});
  }
  hideLowPlddt();
  leftViewer.render(); rightViewer.render();
}

function hideLowPlddt() {
  // pLDDT is read from the B-factor column; the trimmed mapping above still uses all residues
  if ($('show-all').checked) return;
  const threshold = Number($('plddt-threshold').value) || 0;
  [leftViewer, rightViewer].forEach(v => v.setStyle({predicate: a => a.b < threshold}, {}));
}

async function showStoi(index) {
  stoiIndex = index;
  $('prev-stoi').disabled = index === 0;
  $('next-stoi').disabled = index === complex.stois.length - 1;
  updateStats();
  const [leftDone, rightDone] = await Promise.all([loadLeft(), loadRight()]);
  if (!leftDone || !rightDone) return;
  applyColors();
  leftViewer.zoomTo(); rightViewer.zoomTo();
  leftViewer.render(); rightViewer.render();
}

async function setRightMode(mode) {
  rightMode = mode;
  document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.mode === mode));
  updateStats();
  if (!(await loadRight())) return;
  applyColors();
  rightViewer.zoomTo(); rightViewer.render();
}

function selectComplex(index) {
  complexIndex = index;
  complex = COMPLEXES[index];
  $('complex-select').value = index;
  $('prev-complex').disabled = index === 0;
  $('next-complex').disabled = index === COMPLEXES.length - 1;
  showStoi(0);
}

COMPLEXES.forEach((c, i) => {
  const ids = c.identifiers.length > 60 ? c.identifiers.slice(0, 57) + '...' : c.identifiers;
  $('complex-select').add(new Option(`${c.complex_ac}  (CF ${c.stois[0].trimmed.conf.toFixed(1)}, core ${c.cp_complex_ac})  ${ids}`, i));
});
$('complex-select').onchange = () => selectComplex(Number($('complex-select').value));
$('prev-complex').onclick = () => selectComplex(complexIndex - 1);
$('next-complex').onclick = () => selectComplex(complexIndex + 1);
$('prev-stoi').onclick = () => showStoi(stoiIndex - 1);
$('next-stoi').onclick = () => showStoi(stoiIndex + 1);
document.querySelectorAll('.tab').forEach(tab => tab.onclick = () => setRightMode(tab.dataset.mode));
$('color-mode').onchange = applyColors;
$('plddt-threshold').oninput = applyColors;
$('show-all').onchange = function() {
  $('plddt-threshold').disabled = this.checked;
  applyColors();
};

selectComplex(0);
</script>
</body></html>
"""

assert "%%JSON%%" in HTML
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML.replace("%%JSON%%", json.dumps(viewer_complexes)), encoding="utf-8")
print(f"Written: {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")