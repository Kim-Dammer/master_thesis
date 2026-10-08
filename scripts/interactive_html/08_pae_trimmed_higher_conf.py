import ast
import base64
import gzip
import json
import re
from pathlib import Path

import polars as pl
from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"

CONF_THRESHOLD = 80
N_PROTEINS_ABOVE = 2    # keep complexes with MORE than this many proteins
EXPECTED_N_GAINED = 10

# all four runs are joined (as in the notebook) so the complex set is identical to the gained/lost plot
RUNS = {
    "ranking": "10_all_CP_complexes",
    "pae": "18_CP_model_based_on_pae",
    "trimmed": "19_CP_pae_plddt_trimmed",
    "safeguard": "20_CP_pae_plddt_trimmed_interface_safeguard",
}
VIEWER_RUNS = {"pae": "pae (untrimmed)", "trimmed": "pae trimmed"}   # left, right

viewer_dir = data_dir / "Pipeline/viewer"
assert viewer_dir.parent.exists(), f"{viewer_dir.parent} does not exist"
viewer_dir.mkdir(exist_ok=True)
viewer_html_path = viewer_dir / "08_pae_vs_trimmed_viewer.html"


# ---------------- complexes without a structure match ----------------
pdb_mapping = pl.read_csv(data_dir / "complete_complex_pdb_mapping_v2/all_pdb_matches_with_match_class.csv")
complexes_without_structure = (
    pdb_mapping
    .filter(pl.col("match_class").is_in(["no_homology_match", "no_sufficient_match"]))
    .filter(pl.col("n_proteins") > N_PROTEINS_ABOVE)
    .select("complex_ac", "match_class")
    .unique()
)
assert complexes_without_structure["complex_ac"].is_unique().all(), "a complex has more than one match_class"


# ---------------- best confidence per complex and run ----------------
def read_combfold_results(run_name):
    folder = RUNS[run_name]
    path = data_dir / "Pipeline" / folder / f"all_pdb_present_{folder}_pool_pipeline_complexes_combfold_results.csv"
    assert path.exists(), f"missing results file: {path}"
    results = pl.read_csv(path).join(complexes_without_structure, on="complex_ac", how="inner")
    assert results["complex_ac"].is_unique().all(), f"{path.name}: more than one row per complex"
    return results


def highest_confidence(column):
    """highest CombFold confidence in a result cell (a dict stored as text); null if there is no assembly"""
    return pl.col(column).map_elements(
        lambda cell: max(ast.literal_eval(cell).values()) if cell not in (None, "{}") else None,
        return_dtype=pl.Float64,
    )


def best_confidence_per_complex(run_name):
    return read_combfold_results(run_name).select(
        "complex_ac",
        pl.max_horizontal(
            highest_confidence("CF_1_confidence"),
            highest_confidence("CF_2_confidence"),
            highest_confidence("CF_3_confidence"),
            pl.when(pl.col("correct_pred_rank") == "none").then(highest_confidence("CF_true_confidence")),
        ).alias(run_name),
    )


best_confidence_table = complexes_without_structure
for run_name in RUNS:
    best_confidence_table = best_confidence_table.join(best_confidence_per_complex(run_name), on="complex_ac", how="inner")
for run_name in RUNS:
    assert read_combfold_results(run_name).height == best_confidence_table.height, \
        f"{run_name}: its csv has other complexes than the other runs"

# passes in trimmed, but not in pae (no output counts as not passing)
complexes_gained_in_trimmed = (
    best_confidence_table
    .filter(~(pl.col("pae") > CONF_THRESHOLD).fill_null(False) & (pl.col("trimmed") > CONF_THRESHOLD).fill_null(False))
    .sort("trimmed", descending=True)
)
assert complexes_gained_in_trimmed.height == EXPECTED_N_GAINED, \
    f"expected {EXPECTED_N_GAINED} complexes, got {complexes_gained_in_trimmed.height}"
print(complexes_gained_in_trimmed)


# ---------------- find the best assembly pdb per complex ----------------
# pool folders are named by stoichiometry, e.g. O13297x1_Q01159x1_pool_output, not by complex id,
# so: complex -> candidate stoichiometries (from the results csv) -> pool folder -> confidence.txt
def stoich_from_pool_dir(dir_name):
    """'O13297x1_Q01159x1_pool_output' -> (('O13297', 1), ('Q01159', 1)), order-independent"""
    assert dir_name.endswith("_pool_output"), f"unexpected folder name: {dir_name}"
    entries = dir_name.removesuffix("_pool_output").split("_")
    parsed = [re.fullmatch(r"(.+)x(\d+)", entry) for entry in entries]
    assert all(parsed), f"cannot parse pool folder name: {dir_name}"
    return tuple(sorted((m.group(1), int(m.group(2))) for m in parsed))


def stoich_from_pred_cell(cell):
    """'{"P1":1,"P2":2},{"rank":1,...}' -> (('P1', 1), ('P2', 2))"""
    first_dict = re.match(r"^\{([^}]*)\}", cell)
    assert first_dict, f"cannot parse prediction cell: {cell[:80]}"
    entries = first_dict.group(1).replace('"', "").split(",")
    return tuple(sorted((acc, int(n)) for acc, n in (entry.rsplit(":", 1) for entry in entries)))


NON_PROTEIN_PREFIX = "CHEBI:"   # small molecules; the CombFold pool folders only contain proteins


def stoich_from_identifiers(cell):
    """'P1(1)|CHEBI:597326(2)|P2(2)' -> (('P1', 1), ('P2', 2)); small molecules are dropped"""
    parsed = [re.fullmatch(r"(.+)\((\d+)\)", entry) for entry in cell.split("|")]
    assert all(parsed), f"cannot parse identifiers: {cell}"
    proteins = [(m.group(1), int(m.group(2))) for m in parsed if not m.group(1).startswith(NON_PROTEIN_PREFIX)]
    assert proteins, f"no protein left after dropping small molecules: {cell}"
    return tuple(sorted(proteins))


HASHED_POOL_DIR = re.compile(r"__[0-9a-f]{12}_pool_output$")   # long names: truncated protein list + hash


def index_pool_folders(run_name):
    """{stoichiometry: confidence.txt path} of one run; folders with a hashed name are skipped (counted and printed)"""
    combfold_dir = data_dir / "Pipeline" / RUNS[run_name] / "CombFold"
    stoich_to_confidence_file = {}
    n_hashed = 0
    for confidence_file in combfold_dir.glob("*_pool_output/assembled_results/confidence.txt"):
        pool_dir_name = confidence_file.parent.parent.name
        if HASHED_POOL_DIR.search(pool_dir_name):
            n_hashed += 1
            continue
        stoich = stoich_from_pool_dir(pool_dir_name)
        assert stoich not in stoich_to_confidence_file, f"{run_name}: two pool folders with stoichiometry {stoich}"
        stoich_to_confidence_file[stoich] = confidence_file
    assert stoich_to_confidence_file, f"no pool folders found under {combfold_dir}"
    print(f"{run_name}: {len(stoich_to_confidence_file)} pool folders indexed, {n_hashed} hashed-name folders skipped")
    return stoich_to_confidence_file


def best_pdb_in_pool_folder(confidence_file):
    """(pdb path, confidence) of the highest-confidence assembly in one confidence.txt (ties: first line)"""
    pdb_to_confidence = {}
    for line in confidence_file.read_text().splitlines():
        if line.strip():
            pdb_path, conf = line.rsplit(None, 1)
            pdb_to_confidence[pdb_path] = float(conf)
    assert pdb_to_confidence, f"{confidence_file} is empty"
    best_pdb = max(pdb_to_confidence, key=pdb_to_confidence.get)
    assert Path(best_pdb).exists(), f"pdb does not exist: {best_pdb}"
    return best_pdb, pdb_to_confidence[best_pdb]


def best_assembly_of_complex(run_name, complex_row, stoich_to_confidence_file):
    """Same candidates as best_confidence_per_complex: pred_1/2/3, plus the true stoichiometry if no pred matched it."""
    candidates = []
    for rank in (1, 2, 3):
        if complex_row[f"CF_{rank}_confidence"] in (None, "{}"):
            continue   # this prediction has no assembly, it cannot be the best one
        candidates.append((f"pred_{rank}", stoich_from_pred_cell(complex_row[f"pred_{rank}"])))
    if complex_row["correct_pred_rank"] == "none" and complex_row["CF_true_confidence"] not in (None, "{}"):
        candidates.append(("true", stoich_from_identifiers(complex_row["identifiers"])))
    assert candidates, f"{complex_row['complex_ac']}/{run_name}: no candidate with an assembly"

    best = None
    for source, stoich in candidates:
        assert stoich in stoich_to_confidence_file, \
            f"{complex_row['complex_ac']}/{run_name}/{source}: no pool folder for stoichiometry {stoich}"
        pdb_path, conf = best_pdb_in_pool_folder(stoich_to_confidence_file[stoich])
        if best is None or conf > best[1]:
            best = (pdb_path, conf)
    return best


def gzip_base64(text):
    """pdb text -> gzipped, base64-encoded string (decompressed in the browser with DecompressionStream)"""
    return base64.b64encode(gzip.compress(text.encode())).decode()


pool_folders_per_run = {run_name: index_pool_folders(run_name) for run_name in VIEWER_RUNS}
results_per_run = {run_name: read_combfold_results(run_name) for run_name in VIEWER_RUNS}

viewer_data = {}
for complex_ac in complexes_gained_in_trimmed["complex_ac"]:
    viewer_data[complex_ac] = {}
    for run_name in VIEWER_RUNS:
        complex_row = results_per_run[run_name].filter(pl.col("complex_ac") == complex_ac).row(0, named=True)
        pdb_path, conf = best_assembly_of_complex(run_name, complex_row, pool_folders_per_run[run_name])
        # the score shown in the viewer must be the one from the table above
        conf_in_table = complexes_gained_in_trimmed.filter(pl.col("complex_ac") == complex_ac)[run_name].item()
        assert conf_in_table is not None, f"{complex_ac}/{run_name}: no output in the table, but an assembly was found"
        assert abs(conf - conf_in_table) < 0.1, f"{complex_ac}/{run_name}: viewer conf {conf} != table conf {conf_in_table}"
        viewer_data[complex_ac][run_name] = {"conf": round(conf, 1), "pdb_gz": gzip_base64(Path(pdb_path).read_text())}

# ---------------- html ----------------
VIEWER_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>pae vs trimmed</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body { font-family: sans-serif; margin: 12px; }
  #bar { margin-bottom: 8px; }
  #labels { display: flex; }
  #labels div { flex: 1; text-align: center; font-size: 18px; padding: 6px; }
  .good { color: #009E73; font-weight: bold; }
  .bad { color: #D55E00; font-weight: bold; }
  #grid { position: relative; width: 100%; height: 75vh; border: 1px solid #ccc; box-sizing: border-box; }
</style></head>
<body>
<div id="bar">
  <button id="prev">&larr;</button>
  <select id="complex"></select>
  <button id="next">&rarr;</button>
  <span style="color:#666"> both views rotate together, colored by chain, threshold: conf &gt; __THRESHOLD__</span>
</div>
<div id="labels"><div id="label0"></div><div id="label1"></div></div>
<div id="grid"></div>
<script>
window.onerror = (msg, src, line) => { document.body.insertAdjacentHTML('afterbegin', '<pre style="color:red">JS error: ' + msg + ' (line ' + line + ')</pre>'); };
window.onunhandledrejection = (event) => { document.body.insertAdjacentHTML('afterbegin', '<pre style="color:red">JS error: ' + event.reason + '</pre>'); };
if (typeof $3Dmol === 'undefined') throw new Error('3Dmol.js not loaded (no internet or domain blocked)');
if (typeof DecompressionStream === 'undefined') throw new Error('browser has no DecompressionStream (too old)');
const data = __DATA__;
const runs = __RUNS__;           // [[key, title], ...] left to right
const threshold = __THRESHOLD__;
const select = document.getElementById('complex');
const complexes = Object.keys(data);
complexes.forEach(ac => select.add(new Option(ac, ac)));

const viewers = $3Dmol.createViewerGrid(document.getElementById('grid'),
  {rows: 1, cols: 2, control_all: true}, {backgroundColor: 'white'});

async function ungzip(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).text();
}

async function show(ac) {
  const pdbTexts = await Promise.all(runs.map(([key]) => ungzip(data[ac][key].pdb_gz)));
  if (select.value !== ac) return;   // user switched complex while unzipping
  runs.forEach(([key, title], i) => {
    const entry = data[ac][key];
    const viewer = viewers[0][i];
    viewer.clear();
    viewer.addModel(pdbTexts[i], 'pdb');
    viewer.setStyle({}, {cartoon: {colorscheme: 'chain'}});
    viewer.zoomTo();
    viewer.render();
    const cls = entry.conf > threshold ? 'good' : 'bad';
    document.getElementById('label' + i).innerHTML =
      title + ': CF confidence <span class="' + cls + '">' + entry.conf + '</span>';
  });
}
function step(delta) {
  const i = (complexes.indexOf(select.value) + delta + complexes.length) % complexes.length;
  select.value = complexes[i];
  show(select.value);
}
select.onchange = () => show(select.value);
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
select.value = complexes[0];
show(complexes[0]);
</script></body></html>"""

viewer_html = (
    VIEWER_TEMPLATE
    .replace("__DATA__", json.dumps(viewer_data))
    .replace("__RUNS__", json.dumps(list(VIEWER_RUNS.items())))
    .replace("__THRESHOLD__", str(CONF_THRESHOLD))
)
viewer_html_path.write_text(viewer_html)
print(f"{len(viewer_data)} complexes, {viewer_html_path.stat().st_size / 1e6:.1f} MB -> {viewer_html_path}")