# combfold_eval — score CombFold assemblies against reference PDB structures

A standalone, dependency-light batch tool that runs **after CombFold**. For every
complex in a mapping file it locates the CombFold assembled model(s), acquires the
reference structure, maps chains **by sequence**, and reports three complementary
accuracy levels — **per-subunit fold, global assembly, and per-interface** — with
explicit coverage so every number is interpretable.

The guiding principle of this tool is that **every methodological choice is
explicit, documented, and justified** (see *Methodological choices* below), rather
than hidden inside a black box. Nothing is cherry-picked to look good.

---

## 1. What it computes

| Level | Metric | Tool | Why |
|-------|--------|------|-----|
| Per-subunit fold | TM-score (both normalizations) + CA-RMSD (2 variants) | US-align (`mm=0`) + in-house Kabsch | Is each subunit folded correctly, independent of assembly? |
| Global assembly | Complex TM-score (both normalizations) + global CA-RMSD | US-align (`mm=1`, multimer) + in-house Kabsch | Is the whole complex placed/oriented correctly? |
| Per-interface | DockQ, Fnat, iRMSD, LRMSD, CAPRI class + mean | DockQ v2 | Is each pairwise contact reproduced (the quantity CombFold optimizes)? |

Reporting all three is deliberate: a complex can have well-folded subunits (high
per-chain TM) but wrong relative placement (low complex TM / poor DockQ), or vice
versa. One global number would hide that.

---

## 2. Requirements

- **Python 3.9+** with: `gemmi`, `biopython`, `numpy`, `scipy`, `pandas`
- **DockQ v2** (`pip install DockQ`) — provides the `DockQ` CLI
- **US-align** binary (`USalign`) — https://zhanggroup.org/US-align/

Point the tool at the two binaries with `--usalign` / `--dockq` (or the CONFIG
block) if they are not on `PATH`. No PyMOL, no internet at run time except to
download reference structures (which are cached).

---

## 3. Quick start

```bash
# option A: edit the CONFIG block at the top of run_combfold_eval.py, then:
python run_combfold_eval.py
# (no --ref-cache needed: defaults to your persistent
#  /cluster/project/beltrao/kdammer/master_thesis/data/reference_pdb)

# option B: pass everything on the command line
python run_combfold_eval.py \
    --mapping   best_match_per_complex.csv \
    --combfold-base /path/to/CombFold \
    --uniprot-csv   all_yeast_proteins_uniprot_mapped_sequences.csv \
    --out-dir       combfold_eval_out \
    --ref-cache     /cluster/project/beltrao/kdammer/master_thesis/data/reference_pdb \
    --usalign /path/to/USalign --dockq DockQ

# score only the top-ranked cluster instead of every cluster
python run_combfold_eval.py --top-cluster-only

# restrict to a few complexes while debugging
python run_combfold_eval.py --only CPX-2800,CPX-737

# also write the exact scored PDBs (renamed/superposable) for visual inspection
python run_combfold_eval.py --save-structures
```

Run `python run_combfold_eval.py --help` for the full flag list (identity
thresholds, outlier cycles/cutoff, etc.).

---

## 4. Mapping-file column contract

The mapping file is a CSV. Only two columns are **required**:

| Column | Req? | Meaning |
|--------|------|---------|
| `complex_ac` | **yes** | Complex identifier (e.g. `CPX-2800`). Used to locate the CombFold output folder. |
| `pdb_id` | **yes** | Best-match reference PDB id (e.g. `2qlv`). Use `SELF` (or set `local_ref`) to score against a local file only. |
| `matching_hits` | no | `uniprot:pdb_chain,...` — used **only** as a fallback source of candidate UniProts. |
| `folder` | no | Explicit CombFold output folder (absolute, or relative to `--combfold-base`). |
| `model` | no | Explicit model PDB path or glob (overrides folder discovery). |
| `uniprots` | no | Explicit UniProt list (`P1;P2` or `P1x2,P3`) for this complex. |
| `local_ref` | no | Local reference structure file, treated as the asymmetric unit. |

**How each complex is resolved** (all steps are logged in `run_log.txt`):

- **Model(s):** the `model` glob if given; else the canonical CombFold layout
  `<base>/<folder-or-auto>/assembled_results/output_clustered_*.pdb`
  (auto = the `<complex_ac>`-matching folder, preferring `*_output`/`*_result`
  folders over `*_input`); else a flat `<base>/<complex_ac>*.pdb` search.
  All clusters are scored by default (`--top-cluster-only` to change this).
- **Confidence:** `assembled_results/confidence.txt` (both known formats parsed).
- **Candidate UniProts** (for sequence-based chain mapping): the `uniprots`
  column, else parsed from the folder/model name (`..._P06782x1_P12904x1_...`),
  else from `matching_hits`. Full-length sequences come from the offline CSV,
  falling back to the UniProt REST API (cached).

---

## 5. Output schema

Written to `--out-dir`:

- **`complex_summary.csv`** — one row per *(complex, cluster, reference form)*:
  cluster + CombFold confidence/rank, reference form + whether it is the primary
  form, composition match (shared/missing/extra subunits), complex TM (both
  norms), global RMSD (both variants) + `n_res_global`/`coverage_global`,
  `mean_dockq`, provenance columns (`pairing_source`, `candidate_source`,
  `ref_select_reason`), `status`, `flags`.
- **`per_chain.csv`** — one row per matched subunit: model↔reference chain, TM
  (both norms), RMSD (no-cycles + refined) with atom counts, coverage of
  resolved and of full length, sequence identity.
- **`per_interface.csv`** — one row per native interface: interface UniProts,
  native chain-pair key, model chain pair, DockQ / Fnat / iRMSD / LRMSD, CAPRI.
- **`json/<complex>_c<k>.json`** — full chain-mapping + reference-form provenance
  for that cluster (audit trail).
- **`run_log.txt`** — one status line per complex/cluster.
- **`structures/`** (only with `--save-structures`) — the exact renamed,
  shared-subunit PDBs of the primary form that were fed to the metrics, so the
  superposition can be reproduced/inspected in any viewer.

Every complex is wrapped so a single failure never aborts the batch; the failure
is recorded as an explicit `status` (`no_model_found`, `no_uniprots_resolved`,
`error`, ...) instead.

---

## 6. Methodological choices (and why)

This section exists because the goal is to be **aware of and justify every choice**.

### 6.1 Chain mapping — sequence-based and independent
Model and reference chains are matched by **local sequence alignment to the
complex's UniProt sequences**, independently on each structure; then model and
reference chains that resolve to the *same* UniProt are paired. Copy-number
ambiguity (homo-oligomers) is resolved by DockQ's own optimal chain assignment,
which is then reused as the authoritative pairing for TM/RMSD.

- **Not by chain ID:** CombFold labels chains `A,B,C…` in spec order; the PDB uses
  its own labels. Matching by ID would mis-pair subunits. *In the 2QLV example,
  model chain `B` (SNF4) correctly maps to reference chain `C`, not reference `B`.*
- **Not by length:** paralogs/similar-length subunits would be swapped.
- Assignments at ≥ `same_protein_identity` (default 90%) are "same protein";
  between `homolog_identity` (30%) and 90% are accepted but **flagged** as
  homolog/paralog, never silently.

### 6.2 Reference form — download both, select *before* scoring
For each `pdb_id` the tool downloads **all biological assemblies** from RCSB and
also keeps the **asymmetric unit** (or a user-supplied `local_ref`). The form
used as *primary* is chosen by a **composition rule applied before any metric is
computed**, ranked by: fewest missing subunits → fewest extra subunits →
biological assembly preferred over AU → most resolved residues.

- **Why pre-metric:** selecting the form that maximizes the score would be
  circular. Composition match to the model is a metric-independent criterion.
- **Why restrict the form:** scoring against a 6-chain AU when the model is a
  3-chain biological unit deflates the reference-normalized TM by construction.
- **All forms are still scored and reported** (one flagged `is_primary_ref`), so
  you can see the sensitivity to this choice rather than trusting it blindly.
  *In the 2QLV example, `assembly_1` and `assembly_2` model SIP2 slightly
  differently; both are in the output.*

### 6.3 TM-score — both normalizations
US-align normalizes TM by a chosen chain-length; short vs long reference changes
the value. We report **both**:
- `TM_ref` (normalized by the **reference** length) — "how much of the true
  structure did we recover".
- `TM_mod` (normalized by the **model** length) — penalizes extra/over-modeled
  residues in the model.
Reporting both makes coverage effects explicit instead of picking a flattering
denominator. (US-align is run with `-ter 1`, and `-mm 1` for the whole complex.)

### 6.4 RMSD — Bio.SVDSuperimposer fit, custom outlier-rejection loop
CA-RMSD is computed via **Kabsch superposition** (no PyMOL — more transparent,
and portable to a headless SLURM node). The rigid-body fit itself is delegated to
`Bio.SVDSuperimposer` (Biopython is already a pipeline dependency, so this adds
no new one) rather than a hand-rolled SVD, since correctly forcing a *proper*
rotation (never a mirror-image "best fit") is a solved problem better reused than
reimplemented. This was verified against a from-scratch implementation including
on an adversarial mirror-image test case (bit-identical RMSD/rotation in both a
normal and a reflective-input scenario) before switching, and the full pipeline
was re-run end-to-end afterward with zero change in any TM/RMSD/DockQ output.
The outer refinement logic — which is the actual design decision, and isn't
provided by any library — stays custom. Two variants are reported:
- `rmsd_nocyc` — over **all** co-observed CA pairs (no rejection); `n_res` equals
  the number of aligned CA, so the value is fully interpretable.
- `rmsd_wcyc` — up to `outlier_cycles` (default 5) refit iterations that reject
  CA pairs deviating beyond `outlier_cutoff` x current RMSD (dynamically
  recomputed each cycle, mirroring PyMOL `align`'s adaptive rejection). This
  reveals the well-superposed *core*.
`rmsd_nocyc ≥ rmsd_wcyc` always; the surviving atom count is always reported.

### 6.5 DockQ — per interface + mean over native interfaces
DockQ is the standard CAPRI-aware interface metric and is what CombFold's assembly
step effectively optimizes. We report it **per native interface** (with Fnat,
iRMSD, LRMSD, and CAPRI class: High ≥0.80, Medium ≥0.49, Acceptable ≥0.23, else
Incorrect) and the **mean over native interfaces** as a single assembly-quality
summary. Interfaces are keyed by native chains; DockQ's optimal chain mapping
handles homo-oligomer copy permutations.

**Mismatch tolerance (`dockq_allowed_mismatches`, default `10`).** DockQ v2 runs
its own internal sequence-homology re-check before it will pair a model chain
with a native chain, and by default that check requires **zero** sequence
mismatches (`--allowed_mismatches 0`). This duplicates — and is stricter than —
the pipeline's own independent sequence-based chain assignment (gated at
`same_protein_identity`, default 90% identity). In practice, any true residue
difference between the full-length UniProt-derived model and the actual
crystallized construct (strain polymorphism, an engineered point mutation, a
cloning-tag remnant, etc.) causes DockQ to reject the chain outright and return
`mean_dockq=NaN` for the whole complex, even though TM-score/RMSD compute
normally and the pairing is otherwise correct. This was caught during sanity
testing on a diverse complex set: two complexes with a single real residue
difference between model and reference chain had **100% of their DockQ scores
silently zeroed out** despite TM scores in the 0.6–0.9 range. The fix raises
DockQ's own tolerance to `--allowed_mismatches 10` by default — generous enough
to cover typical construct differences, but bounded so DockQ won't force-match a
genuinely different paralog, since DockQ only ever sees chain pairs the
pipeline's own alignment already accepted. Override via `Config(dockq_allowed_mismatches=...)`
or `--dockq-allowed-mismatches` if you want DockQ's stricter default back, or a
larger tolerance for unusually divergent constructs.

**When `mean_dockq` is still `NaN`.** If a model proposes more copies of a
homo-oligomeric subunit than the reference structure actually contains (a
CombFold stoichiometry guess can simply be wrong), there is no real chain left
to pair the surplus copy against. DockQ correctly refuses to score that pairing,
`mean_dockq` is `NaN` for the affected interfaces, and the row is flagged
`homo_oligomer_copy_pairing_arbitrary` together with `dockq_no_interface_or_failed`.
This is not a bug to raise `dockq_allowed_mismatches` further for — check
`ref_select_reason` / the actual PDB entity list first; the reference may simply
not contain the stoichiometry CombFold assembled.

### 6.6 Coverage is always reported
Because a subunit can be perfectly folded yet only partially resolved in the
crystal structure (or vice versa), every per-chain row carries coverage of
resolved residues and of full length. A high RMSD/low TM is only meaningful
alongside how much of the chain it was computed over.

---

## 7. Worked example (2QLV / CPX-2800)

`CPX_2800_...pool_output.pdb` (CombFold model, chains A/B/C) vs **2QLV** (the
heterotrimeric core of *S. cerevisiae* AMPK/SNF1, X-ray 2.6 Å). Reference forms
found: `assembly_1` (3 chains, **primary**), `assembly_2` (3 chains),
`asymmetric_unit` (6 chains, flagged `extra=3`, never primary).

Sequence-based mapping (100% identity; note B→C, C→B):

| Model | UniProt (protein) | Ref chain | coverage (full-len) |
|-------|-------------------|-----------|---------------------|
| A | P06782 (SNF1) | A | 0.21 |
| B | P12904 (SNF4) | C | 0.96 |
| C | P34164 (SIP2) | B | 0.37 |

Primary (`assembly_1`): complex `TM_ref`=0.813, `TM_mod`=0.374; global RMSD
8.84 Å (598 CA) → 2.22 Å refined; mean DockQ 0.288. Interfaces: SNF1–SNF4
DockQ 0.10 (Incorrect), SNF1–SIP2 0.48 (Acceptable), SNF4–SIP2 0.28 (Acceptable).

Interpretation: SNF4 is modeled very well (TM 0.97), SNF1/SIP2 are only partially
resolved in the crystal and less well placed — visible precisely because coverage
and both TM normalizations are reported.

---

## 8. Compute + parallelization

Each comparison is ~5–10 s and < 1 GB RAM, and is spent almost entirely in two
external calls (US-align, DockQ) plus structure parsing — not in any Python-side
bookkeeping. Different complexes are fully independent (own temp work dir, own
per-complex JSON, own subprocess calls), so **the batch runner scores complexes
concurrently by default** (`--n-workers`, default `min(8, cpu_count)`) via a
thread pool in `pipeline.py`. `--n-workers 1` reproduces the original strictly
serial code path (useful for debugging). The two genuinely shared resources —
the UniProt sequence cache (`sequences.py`) and the RCSB reference-structure
cache (`reference.py`) — are lock-protected so concurrent workers requesting the
same accession/PDB id don't race on the same cache file; no other change was made
to any metric. This was validated by re-running half of the 27-complex sanity set
under `--n-workers 8` and diffing every numeric column against the existing
serial baseline: max difference across TM-score/RMSD/coverage/DockQ was
9.5e-17 (floating-point noise), with identical status/flags on all rows —
i.e. parallelizing changes wall-clock time only, never a result. Measured speedup
on that run was ~2x (limited by load imbalance across a small row count and
per-row CPU contention from concurrent US-align/DockQ subprocesses, not by core
count); expect the ratio to improve on a larger, more uniform batch or more cores.

Beyond a single node, the batch is still embarrassingly parallel across mapping
rows: **shard the mapping file across a SLURM array** (one shard per task, each
with its own `--out-dir`) and concatenate the three CSVs afterward. The
reference cache (`--ref-cache`) is safe to share read-only across tasks once
populated.

### 8.1 Persistent reference-PDB cache

The CLI script's default `--ref-cache` is
`/cluster/project/beltrao/kdammer/master_thesis/data/reference_pdb` — a fixed
path independent of your working directory, so reruns from anywhere reuse it
rather than starting from an empty `./ref_cache`. Each PDB id gets its own
subfolder (`<ref_cache>/<pdb_id>/`); a file already present and non-empty is
reused as-is, never re-downloaded (see `reference.py::acquire_reference_files`).
Every batch run prints and logs one summary line so you can confirm this
directly, e.g.:
```
[ref cache] 22 file(s) reused from /cluster/.../reference_pdb, 0 newly downloaded, 0 failed
```
Validated by running the same 4-complex subset twice against the same (initially
empty) cache dir: run 1 reported 8 downloads/14 hits (later clusters of the same
complex already reusing that complex's just-downloaded reference), run 2
reported 0 downloads/22 hits, and the two runs' output tables were bit-identical
(max diff 0.0) — confirms caching changes only whether a network call happens,
never a result.

---

## 9. Layout

```
combfold_eval/
├── run_combfold_eval.py   # standalone CLI entry point (edit CONFIG or use flags)
├── config.py              # all tunable choices, each documented with its reason
├── sequences.py           # UniProt full-length sequences (CSV + REST fallback)
├── structure_utils.py     # mmCIF/PDB load, cleanup, modified-residue handling
├── mapping.py             # sequence-based chain assignment + residue correspondence
├── reference.py           # RCSB assembly/AU acquisition + pre-metric form selection
├── metrics.py             # US-align (TM), DockQ, Kabsch RMSD wrappers
├── compare.py             # one model vs one reference: the full comparison
└── pipeline.py            # batch driver + output schema
```
