#!/usr/bin/env python
"""
rm /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/logs/*.{out,err}
rm -r /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/5_fifth_setup_all_CP/*_{input,output}
rm -r /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/5_fifth_setup_all_CP/pdb_source_logs
===========================
uv run 01_combfold_with_stoic.py \
        --mode all \
        --input-tsv  /cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Saccharomyces_cerevisiae_ComplexTab.tsv\
        --seq-csv    /cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_yeast_proteins_uniprot_mapped_sequences.csv \
        --out-dir    /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/5_fifth_setup_all_CP\
        --stoic-sh   s1_run_stoic.sbatch \
        --combfold-sh s2_run_CombFold.sbatch \
        --analyze-sh s3_analyze_CF_results.sbatch \
        --top-n 10 --combfold-source pool

Single orchestrator for the second-setup Stoic + CombFold pipeline.

Pipeline:
    raw TSV -> fasta -> submit-stoic -> aggregate-stoic -> expand
            -> submit-combfold -> analyze

Front-end (new): the fasta stage reads a raw ComplexPortal TSV, applies
procompa.helpers.clean_identifiers to build a one-copy-per-protein
'cleaned_complex' seed, parses the original (n) counts into 'true_spec', and
writes FASTAs using ONLY the local --seq-csv sequences (no UniProt API).
Complexes with any protein ID missing from the local CSV are skipped + logged.

The 'combfold_submission' column (the spec actually sent to CombFold) is filled
in the expand stage from Stoic's predicted stoichiometries (plus the curated
true_spec row). Every CombFold job is submitted with SOURCE=pool by default.
data/Pipeline/3_third_setup
Run with --mode all to submit the whole chain via sbatch dependencies
(no blocking polling). Or run any single mode independently.

Layout written:
    <out_dir>/
        cleaned_complexes.csv          (#Complex ac, cleaned_complex, true_spec, ...)
        missing_ids_complexes.txt      (skipped complexes + missing IDs)
        fastas/<CPX>.fasta
        uniprot_mapped_seq_second_setup.csv
        stoic_results/<CPX>/{af3_input_*.json, results.json}
        stoic_results_aggregated_second_setup.csv
        second_setup_expanded.csv      (adds combfold_submission per Stoic pred)
        second_setup_job_registry.json
        CombFold/<complex_name>_output/...
        all_pdb_present_second_setup_pipeline_complexes_combfold_results.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import requests  # used ONLY for targeted PRO-chain sub-sequence resolution

# clean_identifiers comes from the project's helper package. It drops CHEBI
# entries and strips the trailing "(n)" stoichiometry suffix, turning
#   "P32797(0)|P38960(0)|CHEBI:49601(0)|Q07921(0)"
# into
#   "P32797 P38960 Q07921"
from procompa.helpers import clean_identifiers

# ---------------------------------------------------------------------------
# SETUP_NAME: the ONE place to rename a run. It is interpolated into every
# setup-tagged output filename (uniprot_mapped_seq_<name>.csv,
# <name>_expanded.csv, <name>_job_registry.json, stoic_results_aggregated_
# <name>.csv, all_pdb_present_<name>_pipeline_..._results.csv). Override at the
# command line with --setup-name without editing this file.
SETUP_NAME = "fifth_setup"

# Raw ComplexPortal column that holds the pipe-separated molecule identifiers.
MOLECULES_COL = "Identifiers (and stoichiometry) of molecules in complex"
COMPLEX_AC_COL = "#Complex ac"


# ---------------------------------------------------------------------------
# Spec parsing helpers (shared with 10_run_csv_combfold.py)
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(r"^([A-Za-z0-9_]+)\((\d+)\)$")


def parse_spec(spec: str) -> dict[str, int]:
    """Parse 'P00937(1),P00899(2)' -> {'P00937': 1, 'P00899': 2}.

    A non-empty token that does not match _ENTRY_RE is still skipped (behavior
    unchanged), but a one-line warning is emitted so silent drops become
    visible — e.g. a '-PRO_' id or a formatting glitch that slipped into a spec
    string would otherwise vanish without trace.
    """
    counts: dict[str, int] = {}
    for tok in [t.strip() for t in str(spec).split(",") if t.strip()]:
        m = _ENTRY_RE.match(tok)
        if m:
            counts[m.group(1)] = int(m.group(2))
        else:
            print(f"[parse_spec] WARNING: unparseable token skipped: {tok!r} "
                  f"(in spec {str(spec)!r})", file=sys.stderr)
    return counts


def canonical_spec(spec_or_dict: str | dict[str, int]) -> tuple[tuple[str, int], ...]:
    """Return a sorted, hashable canonical form."""
    d = spec_or_dict if isinstance(spec_or_dict, dict) else parse_spec(spec_or_dict)
    return tuple(sorted(d.items()))


def dict_to_spec_str(d: dict[str, int], protein_order: list[str] | None = None) -> str:
    """Format {uid: count} -> 'P00937(1),P00899(2)'.

    If protein_order is given, use that order (sorted otherwise).
    """
    order = protein_order if protein_order is not None else sorted(d)
    return ",".join(f"{p}({d[p]})" for p in order if p in d)


def spec_to_complex_name(spec: str) -> str:
    """Mirror 05_run_CombFold.sbatch: sorted UniProt IDs joined as '{id}x{count}'."""
    counts = parse_spec(spec)
    return "_".join(f"{p}x{counts[p]}" for p in sorted(counts))


def spec_to_proteins(spec: str) -> list[str]:
    return sorted(parse_spec(spec))


# ---------------------------------------------------------------------------
# Raw ComplexPortal TSV parsing helpers
# ---------------------------------------------------------------------------

# Plain UniProt accession (Swiss-Prot / TrEMBL syntax).
_UNIPROT_ACCESSION = r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
# A foldable subunit token = plain accession OR a processed-chain id
# "<accession>-PRO_<digits>". Each may carry a trailing "(n)" stoichiometry.
_SUBUNIT_RE = re.compile(rf"^((?:{_UNIPROT_ACCESSION})(?:-PRO_\d+)?)\((\d+)\)$")
# Tokens we intentionally drop WITHOUT skipping the complex.
_DROP_PREFIXES = ("CHEBI:", "URS")
# PRO-chain token detector (after stripping the "(n)" suffix).
_PRO_TOKEN_RE = re.compile(rf"^({_UNIPROT_ACCESSION})-PRO_(\d+)$")


def _strip_count(tok: str) -> tuple[str, int | None]:
    """Split 'X(2)' -> ('X', 2); 'X' -> ('X', None)."""
    m = re.match(r"^(.*)\((\d+)\)$", tok)
    if m:
        return m.group(1), int(m.group(2))
    return tok, None


def classify_molecules(molecules: str) -> tuple[dict[str, int], bool, list[str]]:
    """Classify the raw ComplexPortal molecules string.

    Input example: "P32797(0)|P38960(0)|CHEBI:49601(0)|Q07921(0)"

    Rules:
      * Keep plain UniProt accessions as foldable subunits. "(0)" (unknown) ->
        count 1; positive count kept as-is.
      * Processed-chain ids (X-PRO_NNN) are TEMPORARILY treated as blockers, so
        any complex containing one is skipped end-to-end (see note in body). The
        pairwise foldcomp DBs are keyed on plain full-length accessions, so a
        mature PRO sub-sequence has no residue-consistent pairwise structure.
      * Silently drop CHEBI and URS (RNAcentral) tokens.
      * Any OTHER token type (paralog sets '[A,B]', nested complex refs 'CPX-...',
        'EBI-...', etc.) is treated as a hard blocker: the whole complex must be
        skipped. Such tokens are returned in `blockers`.

    Returns:
      (counts, all_unknown, blockers)
        counts       {subunit_id: count>=1} in first-seen order (protein order)
        all_unknown  True if every retained subunit had "(0)"
        blockers     list of offending tokens; non-empty => skip the complex
    """
    counts: dict[str, int] = {}
    blockers: list[str] = []
    n_subunits = 0
    n_unknown = 0
    for tok in [t.strip() for t in str(molecules).split("|") if t.strip()]:
        base, _ = _strip_count(tok)
        m = _SUBUNIT_RE.match(tok)
        if m:
            uid = m.group(1)
            # TEMPORARY EXCLUSION: processed-chain ids (X-PRO_NNN) are dropped
            # end-to-end for now. STOIC folds the resolved mature sub-sequence,
            # but the pairwise foldcomp DBs are keyed on plain full-length
            # accessions (predicted on full-length seqs), so the mature
            # sub-sequence has no residue-consistent pairwise structure. Rather
            # than assemble an inconsistent complex, skip the WHOLE complex and
            # log it. Route the PRO token to blockers so stage_fasta's existing
            # skip+log path handles it (reason BLOCKED, detail = the PRO token).
            if _PRO_TOKEN_RE.match(uid):
                blockers.append(tok)
                continue
            raw = int(m.group(2))
            n_subunits += 1
            if raw == 0:
                n_unknown += 1
                counts[uid] = counts.get(uid, 0) + 1
            else:
                counts[uid] = counts.get(uid, 0) + raw
        elif base.upper().startswith("CHEBI:") or base.upper().startswith("URS"):
            continue  # non-protein, drop silently
        else:
            blockers.append(tok)  # set / nested-complex / EBI / unrecognized
    all_unknown = n_subunits > 0 and n_unknown == n_subunits
    return counts, all_unknown, blockers


def build_cleaned_complexes_df(tsv_path: Path) -> pl.DataFrame:
    """Read a raw ComplexPortal TSV and return a cleaned polars DataFrame.

    NOTE: cleaned_complex and true_spec are BOTH derived from classify_molecules
    (the single source of truth), so they always describe the same subunit set.
    procompa.clean_identifiers is applied too (as a raw reference column), but the
    orchestrator does not rely on it for the foldable set.

    Output columns (added):
      cleaned_ref            raw procompa.clean_identifiers output (reference)
      cleaned_complex        space-joined subunit ids, one copy each
      true_spec              curated stoichiometry "UID(n),..." ((0)->1 when at
                             least one subunit count is known; EMPTY string
                             when every subunit count was unknown -- there is
                             no curated ground truth to report in that case)
      true_spec_all_unknown  bool; True if every subunit count was unknown
                             (true_spec is "" iff this is True)
      blocker_tokens         comma-joined non-foldable tokens ('' if none)
    """
    df = pl.read_csv(tsv_path, separator="\t", quote_char=None, infer_schema_length=0)
    if MOLECULES_COL not in df.columns or COMPLEX_AC_COL not in df.columns:
        sys.exit(
            f"[fasta] TSV missing required columns. Need '{COMPLEX_AC_COL}' and "
            f"'{MOLECULES_COL}'. Found: {df.columns}"
        )

    # Reference-only: keep procompa's output for transparency/debugging.
    df = clean_identifiers(df, column=MOLECULES_COL, new_column="cleaned_ref")

    cleaned_complex: list[str] = []
    true_specs: list[str] = []
    all_unknown_flags: list[bool] = []
    blocker_cols: list[str] = []
    for mol in df[MOLECULES_COL].to_list():
        counts, all_unknown, blockers = classify_molecules(mol)
        cleaned_complex.append(" ".join(counts.keys()))
        # When EVERY subunit's ComplexPortal count was "(0)" (unknown), counts
        # values are just the classify_molecules (0)->1 filler, not a curated
        # ground truth. Leave true_spec EMPTY in that case rather than emitting
        # a fabricated "UID(1),..." string that looks like real ground truth
        # downstream (stage_expand relies on true_spec being empty/falsy to
        # correctly mark is_true_spec / stoic_pred_correct as "unknown").
        if all_unknown:
            true_specs.append("")
        else:
            true_specs.append(",".join(f"{uid}({counts[uid]})" for uid in counts))
        all_unknown_flags.append(all_unknown)
        blocker_cols.append(",".join(blockers))

    df = df.with_columns(
        pl.Series("cleaned_complex", cleaned_complex),
        pl.Series("true_spec", true_specs),
        pl.Series("true_spec_all_unknown", all_unknown_flags),
        pl.Series("blocker_tokens", blocker_cols),
    )
    return df


# ---------------------------------------------------------------------------
# Paths / config dataclass
# ---------------------------------------------------------------------------

def _opt_path(args: argparse.Namespace, name: str) -> Path | None:
    """Resolve an optional path CLI arg to an absolute Path, or None if unset.

    Uses getattr so it is safe for args that a given mode never defines; this
    matches the original mixed 'getattr(...)/args.x' access (both resolve to the
    same value here since argparse always defines these with default None)."""
    val = getattr(args, name, None)
    return Path(val).resolve() if val else None


class Paths:
    def __init__(self, args: argparse.Namespace):
        self.out_dir = Path(args.out_dir).resolve()

        # Raw ComplexPortal TSV is the new front-end input. When provided, the
        # fasta stage derives the working CSV (cleaned_complexes.csv) from it and
        # every downstream stage consumes that derived CSV.
        self.input_tsv = _opt_path(args, "input_tsv")
        self.cleaned_csv = self.out_dir / "cleaned_complexes.csv"

        # Local UniProt sequence CSV (uniprot_id,sequence). Sole sequence source.
        self.seq_csv = _opt_path(args, "seq_csv")

        # Working input CSV: explicit --input-csv, else the derived cleaned CSV.
        self.input_csv = _opt_path(args, "input_csv") or self.cleaned_csv
        self.stoic_sh = _opt_path(args, "stoic_sh")
        self.combfold_sh = _opt_path(args, "combfold_sh")
        self.analyze_sh = _opt_path(args, "analyze_sh")
        self.this_script = Path(__file__).resolve()

        # ------------------------------------------------------------------
        # SETUP NAME: single knob for the naming of this run's output files.
        # Change once (via --setup-name, or the SETUP_NAME default below) and
        # every setup-tagged filename updates. E.g. "fifth_setup" ->
        # uniprot_mapped_seq_fifth_setup.csv, fifth_setup_expanded.csv, ...
        # ------------------------------------------------------------------
        self.setup_name = getattr(args, "setup_name", None) or SETUP_NAME
        s = self.setup_name

        # Subpaths
        self.fastas_dir = self.out_dir / "fastas"
        self.uniprot_map_csv = self.out_dir / f"uniprot_mapped_seq_{s}.csv"
        self.stoic_results_dir = self.out_dir / "stoic_results"
        self.stoic_agg_csv = self.out_dir / f"stoic_results_aggregated_{s}.csv"
        self.expanded_csv = self.out_dir / f"{s}_expanded.csv"
        self.registry_json = self.out_dir / f"{s}_job_registry.json"
        self.combfold_out_base = self.out_dir / "CombFold"
        self.final_csv = self.out_dir / (
            f"all_pdb_present_{s}_pipeline_complexes_combfold_results.csv"
        )
        # Tidy long side table: one row per (complex, used pair) with the full
        # AF3 per-pair metrics. Companion to the wide final_csv (see stage_analyze).
        self.pairs_metrics_csv = self.out_dir / f"{s}_pairs_metrics.csv"
        self.missing_stoic_log = self.out_dir / "missing_stoic_cpxs.txt"
        self.missing_ids_log = self.out_dir / "missing_ids_complexes.txt"
        # Persistent cache of resolved PRO-chain sub-sequences (keeps reruns
        # offline without mutating the user's --seq-csv).
        self.pro_cache_csv = self.out_dir / "pro_chain_sequences.csv"

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fastas_dir.mkdir(parents=True, exist_ok=True)
        self.stoic_results_dir.mkdir(parents=True, exist_ok=True)
        self.combfold_out_base.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_registry(paths: Paths) -> dict[str, Any]:
    if paths.registry_json.exists():
        with open(paths.registry_json) as fh:
            return json.load(fh)
    return {}


def save_registry(paths: Paths, reg: dict[str, Any]) -> None:
    with open(paths.registry_json, "w") as fh:
        json.dump(reg, fh, indent=2)


# ---------------------------------------------------------------------------
# SLURM submit helper (shared by all sbatch-submitting stages)
# ---------------------------------------------------------------------------

_JOBID_RE = re.compile(r"Submitted batch job (\d+)")


def _submit_sbatch(
    cmd: list[str],
    tag: str,
    *,
    on_error: str = "exit",
    require_jobid: bool = False,
) -> int | None:
    """Run an sbatch command and return the parsed SLURM job id (or None).

    Consolidates the identical run -> check-returncode -> parse-"Submitted batch
    job N" boilerplate. Behavior knobs preserve each caller's ORIGINAL handling:

      on_error="exit"  -> sys.exit on non-zero returncode (submit-stoic,
                          submit-analyze, submit-chain).
      on_error="skip"  -> print the failure and return None, letting the caller
                          continue (submit-combfold's per-spec loop).
      require_jobid    -> if True, sys.exit when stdout has no parseable job id
                          (submit-stoic). If False, return None (all others).

    `tag` is the log prefix, e.g. "submit-stoic".
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if on_error == "exit":
            sys.exit(f"[{tag}] sbatch failed: {result.stderr.strip()}")
        # on_error == "skip": mirror submit-combfold's per-spec failure path.
        print(f"[{tag}]   FAILED: {result.stderr.strip()}")
        return None
    m = _JOBID_RE.search(result.stdout)
    if not m:
        if require_jobid:
            sys.exit(f"[{tag}] could not parse job id: {result.stdout}")
        return None
    return int(m.group(1))


def _redirect_sbatch_logs(text: str, logs_dir: Path, prefix: str) -> str:
    """Rewrite the first '#SBATCH --output=' / '--error=' lines to logs_dir.

    Shared by the stoic and combfold sbatch patchers; `prefix` names the log
    files (e.g. 'stoic' -> stoic_%j.out / stoic_%j.err). Same regexes, same
    count=1, same MULTILINE flag as the original inlined blocks, so output is
    byte-identical.
    """
    text = re.sub(
        r'^#SBATCH\s+--output=.*$',
        f'#SBATCH --output={logs_dir}/{prefix}_%j.out',
        text, count=1, flags=re.MULTILINE,
    )
    text = re.sub(
        r'^#SBATCH\s+--error=.*$',
        f'#SBATCH --error={logs_dir}/{prefix}_%j.err',
        text, count=1, flags=re.MULTILINE,
    )
    return text


# ===========================================================================
# Stage 1: FASTA generation (replaces 01)
# ===========================================================================

def _resolve_pro_chain(pro_token: str, timeout: int = 60) -> str | None:
    """Resolve a processed-chain token 'ACC-PRO_NNNN' to its mature sub-sequence.

    Queries UniProt for the parent accession, finds the feature whose featureId
    matches 'PRO_NNNN', and slices the full-length sequence to that chain's
    start-end coordinates. Returns None if the accession, feature, or coordinates
    cannot be resolved. Network is used ONLY here (targeted, not bulk).
    """
    m = _PRO_TOKEN_RE.match(pro_token)
    if not m:
        return None
    acc = m.group(1)
    feature_id = f"PRO_{m.group(2)}"
    try:
        r = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=timeout
        )
        if r.status_code != 200:
            return None
        data = r.json()
    except requests.RequestException:
        return None
    full = data.get("sequence", {}).get("value", "")
    if not full:
        return None
    for feat in data.get("features", []):
        if feat.get("featureId") == feature_id:
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if isinstance(start, int) and isinstance(end, int) and 1 <= start <= end <= len(full):
                return full[start - 1:end]
            return None
    return None


def _iter_seq_rows(df: pd.DataFrame):
    """Yield cleaned (uid, seq) pairs from a uniprot_id/sequence DataFrame.

    Shared cleaning rule for both sequence loaders: strip both fields and skip
    any row with an empty uid, empty seq, or a literal 'nan' sequence. Callers
    keep their own column-presence checks and build their own mapping shape.
    """
    for uid, seq in zip(df["uniprot_id"], df["sequence"]):
        uid = str(uid).strip()
        seq = str(seq).strip()
        if uid and seq and seq.lower() != "nan":
            yield uid, seq


def _load_local_sequences(seq_csv: Path) -> dict[str, str]:
    """Load {uniprot_id: sequence} from the local sequence CSV.

    The CSV must have columns 'uniprot_id' and 'sequence'. This is the ONLY
    sequence source (no UniProt API fallback).
    """
    if seq_csv is None:
        sys.exit("[fasta] --seq-csv is required (local uniprot_id,sequence CSV).")
    if not seq_csv.exists():
        sys.exit(f"[fasta] sequence CSV not found: {seq_csv}")
    sdf = pd.read_csv(seq_csv)
    if {"uniprot_id", "sequence"} - set(sdf.columns):
        sys.exit(f"[fasta] {seq_csv} must have 'uniprot_id' and 'sequence' columns.")
    return {uid: seq for uid, seq in _iter_seq_rows(sdf)}


def stage_fasta(paths: Paths) -> None:
    """Front-end stage: raw ComplexPortal TSV -> cleaned CSV + one FASTA per CPX.

    Steps:
      1. Read the raw TSV; apply procompa.clean_identifiers -> 'cleaned_complex'
         (one copy per protein, CHEBI dropped) and parse the original (n)
         counts -> 'true_spec' / 'true_spec_all_unknown'.
      2. Write cleaned_complexes.csv (the working input CSV for downstream stages).
      3. Load sequences from the LOCAL CSV only.
      4. Write one FASTA per complex (one record per protein, single copy).
         Complexes with any protein ID absent from the local CSV are SKIPPED
         and logged to missing_ids_complexes.txt.
      5. Write the seq->uid mapping CSV (subset actually used) for the aggregate
         stage.
    """
    if paths.input_tsv is None:
        sys.exit("[fasta] --input-tsv is required to build the cleaned complex CSV.")
    print(f"[fasta] Reading raw ComplexPortal TSV: {paths.input_tsv}")

    cleaned = build_cleaned_complexes_df(paths.input_tsv)
    print(f"[fasta] {cleaned.height} complexes read from TSV")

    # Load local sequences (sole bulk source) + any previously cached PRO chains.
    print(f"[fasta] Loading local sequences: {paths.seq_csv}")
    sequences = _load_local_sequences(paths.seq_csv)
    if paths.pro_cache_csv.exists():
        cached = _load_local_sequences(paths.pro_cache_csv)
        sequences.update(cached)
        print(f"[fasta] Loaded {len(cached)} cached PRO-chain sub-sequence(s)")
    print(f"[fasta] Sequence dictionary: {len(sequences)} entries")

    # Cache for resolved PRO-chain sub-sequences (avoid duplicate API calls).
    pro_cache: dict[str, str | None] = {}
    newly_resolved: dict[str, str] = {}

    def _get_sequence(sub_id: str) -> str | None:
        """Return sequence for a subunit id (plain accession or X-PRO_NNN).

        Plain accessions come from the local CSV. PRO-chain ids are resolved via
        a targeted UniProt call and cached (both in-run and into the local CSV).
        """
        if sub_id in sequences:
            return sequences[sub_id]
        if _PRO_TOKEN_RE.match(sub_id):
            if sub_id not in pro_cache:
                print(f"[fasta]   resolving PRO chain {sub_id} via UniProt...")
                pro_cache[sub_id] = _resolve_pro_chain(sub_id)
            seq = pro_cache[sub_id]
            if seq:
                sequences[sub_id] = seq
                newly_resolved[sub_id] = seq
            return seq
        return None

    # Decide which complexes are foldable
    kept_rows: list[dict[str, Any]] = []
    missing_records: list[str] = []
    used_ids: set[str] = set()
    n_written = 0

    for row in cleaned.iter_rows(named=True):
        cpx_id = str(row[COMPLEX_AC_COL]).strip()
        blockers = str(row["blocker_tokens"]).strip()
        cleaned_complex = str(row["cleaned_complex"]).strip()
        proteins = [p for p in cleaned_complex.split() if p]

        # (a) Hard blockers (paralog sets, nested complex refs, EBI ids) -> skip.
        if blockers and blockers.lower() != "nan":
            missing_records.append(f"{cpx_id}\tBLOCKED\t{blockers}")
            print(f"[fasta]   [SKIP] {cpx_id}: non-foldable token(s): {blockers}")
            continue

        # (b) No protein subunits after cleaning (RNA/small-molecule only) -> skip.
        if not proteins:
            missing_records.append(f"{cpx_id}\tNO_PROTEIN\t<only non-protein molecules>")
            continue

        # (c) Resolve every subunit sequence (local, or PRO-chain via API).
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for uid in proteins:
            seq = _get_sequence(uid)
            if seq is None:
                missing.append(uid)
            else:
                resolved[uid] = seq
        if missing:
            missing_records.append(f"{cpx_id}\tMISSING_SEQ\t{','.join(missing)}")
            print(f"[fasta]   [SKIP] {cpx_id}: {len(missing)} missing seq(s): {missing}")
            continue

        # (d) Write one FASTA per complex, one record per subunit (single copy).
        fasta_path = paths.fastas_dir / f"{cpx_id}.fasta"
        with open(fasta_path, "w") as f:
            for uid in proteins:
                seq = resolved[uid]
                used_ids.add(uid)
                f.write(f">{uid}\n")
                for j in range(0, len(seq), 80):
                    f.write(seq[j:j + 80] + "\n")
        n_written += 1
        kept_rows.append(row)

    # Persist newly resolved PRO-chain sub-sequences to a dedicated cache CSV so
    # subsequent runs stay fully offline (the user's --seq-csv is not mutated).
    if newly_resolved:
        prev = (_load_local_sequences(paths.pro_cache_csv)
                if paths.pro_cache_csv.exists() else {})
        prev.update(newly_resolved)
        pd.DataFrame(
            [{"uniprot_id": k, "sequence": v} for k, v in sorted(prev.items())]
        ).to_csv(paths.pro_cache_csv, index=False)
        print(f"[fasta] Cached {len(newly_resolved)} PRO-chain sub-sequence(s) "
              f"-> {paths.pro_cache_csv}")

    # Persist the working CSV (only complexes that produced a FASTA)
    kept_df = pl.DataFrame(kept_rows) if kept_rows else cleaned.head(0)
    kept_df.write_csv(paths.cleaned_csv)
    print(f"[fasta] Wrote working CSV ({kept_df.height} foldable complexes): "
          f"{paths.cleaned_csv}")

    # Write seq->uid mapping CSV (only the IDs actually used)
    map_df = pd.DataFrame(
        [{"uniprot_id": uid, "sequence": sequences[uid]} for uid in sorted(used_ids)]
    )
    map_df.to_csv(paths.uniprot_map_csv, index=False)
    print(f"[fasta] Wrote mapping ({len(map_df)} proteins): {paths.uniprot_map_csv}")

    # Log skipped complexes
    if missing_records:
        with open(paths.missing_ids_log, "w") as fh:
            fh.write("#Complex ac\treason\tdetail\n")
            for line in missing_records:
                fh.write(line + "\n")
        print(f"[fasta] Skipped {len(missing_records)} complex(es) with missing/"
              f"non-protein IDs -> {paths.missing_ids_log}")

    print(f"[fasta] Wrote {n_written} FASTA files to {paths.fastas_dir}")


# ===========================================================================
# Stage 2: Stoic submission (single sbatch, env-var overrides)
# ===========================================================================

def _stoic_already_complete(paths: Paths) -> bool:
    """Return True iff STOIC results are already present on disk for every
    expected complex.

    The expected-complex set is derived exactly as `stage_aggregate_stoic` does:
    the unique '#Complex ac' values in `paths.input_csv` (which falls back to
    `cleaned_complexes.csv`, written by `stage_fasta`). A complex counts as done
    when `stoic_results/<cpx>/results.json` exists, parses as JSON, and is a
    non-empty list. If ANY expected complex is missing that, STOIC is treated as
    not-yet-complete (returns False). Trigger is results-on-disk, so it survives
    loss of the job registry.
    """
    if not paths.input_csv.exists():
        print(f"[submit-stoic] No input CSV at {paths.input_csv}; cannot check "
              f"existing STOIC results.")
        return False

    df_input = pd.read_csv(paths.input_csv)
    cpx_ids = df_input["#Complex ac"].dropna().astype(str).unique().tolist()
    if not cpx_ids:
        print("[submit-stoic] Input CSV lists zero complexes; cannot skip STOIC.")
        return False

    missing: list[str] = []
    for cpx_id in cpx_ids:
        rj = paths.stoic_results_dir / cpx_id / "results.json"
        if not rj.exists():
            missing.append(cpx_id)
            continue
        try:
            with open(rj) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            missing.append(cpx_id)
            continue
        if not isinstance(data, list) or not data:
            missing.append(cpx_id)

    n_present = len(cpx_ids) - len(missing)
    if missing:
        preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        print(f"[submit-stoic] STOIC results present for {n_present}/{len(cpx_ids)} "
              f"complexes; {len(missing)} missing/empty ({preview}).")
        return False

    print(f"[submit-stoic] STOIC results present for all {len(cpx_ids)} complexes.")
    return True


def stage_submit_stoic(paths: Paths, top_n: int = 10, force: bool = False) -> None:
    """Submit the single Stoic GPU sbatch with env-var overrides for I/O paths.

    `top_n` is patched into the sbatch's 'stoic_predict_stoichiometry --top-n N'
    so STOIC generates exactly the number of predictions that will be expanded
    to CombFold (single source of truth).

    If `force` is False and results.json already exist on disk for every expected
    complex, submission is skipped (idempotent reruns). Pass `force=True` to
    resubmit regardless.
    """
    if paths.stoic_sh is None or not paths.stoic_sh.exists():
        sys.exit("[submit-stoic] --stoic-sh path does not exist.")

    if not force and _stoic_already_complete(paths):
        print("[submit-stoic] Skipping submission (use --force-stoic to override).")
        return

    print(f"[submit-stoic] Submitting {paths.stoic_sh} (top_n={top_n})")

    # Pass FASTA / output via env vars; the sbatch script needs to honour these.
    # If the user's existing sbatch hardcodes paths, we sed-patch a temp copy.
    patched_sh = _patch_stoic_sbatch(paths, top_n)

    env = os.environ.copy()
    env["SECOND_SETUP_FASTA_DIR"] = str(paths.fastas_dir)
    env["SECOND_SETUP_OUTPUT_DIR"] = str(paths.stoic_results_dir)

    # submit-stoic needs a custom env; run inline but reuse the shared parser via
    # a small closure-free path: same behavior as _submit_sbatch(on_error=exit,
    # require_jobid=True).
    result = subprocess.run(
        ["sbatch", str(patched_sh)],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        sys.exit(f"[submit-stoic] sbatch failed: {result.stderr.strip()}")

    m = _JOBID_RE.search(result.stdout)
    if not m:
        sys.exit(f"[submit-stoic] could not parse job id: {result.stdout}")
    stoic_job_id = int(m.group(1))
    print(f"[submit-stoic] -> stoic_job_id={stoic_job_id}")

    reg = load_registry(paths)
    reg["stoic_job_id"] = stoic_job_id
    reg["stoic_submitted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_registry(paths, reg)
    print(f"[submit-stoic] Registry: {paths.registry_json}")


def _patch_combfold_sbatch(paths: Paths) -> Path:
    """Patch the user's 05_run_CombFold.sbatch to write into second_setup/CombFold.

    Rewrites:
        OUTPUT_BASE="..."         -> second_setup/CombFold
        #SBATCH --output=...      -> absolute path in second_setup/logs/
        #SBATCH --error=...       -> absolute path in second_setup/logs/
    """
    text = paths.combfold_sh.read_text()
    logs_dir = paths.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    text = re.sub(
        r'^OUTPUT_BASE=.*$',
        f'OUTPUT_BASE="{paths.combfold_out_base}"',
        text, count=1, flags=re.MULTILINE,
    )
    text = _redirect_sbatch_logs(text, logs_dir, "combfold")

    patched_path = paths.out_dir / f"_patched_{paths.combfold_sh.name}"
    patched_path.write_text(text)
    patched_path.chmod(0o755)

    # Verify the OUTPUT_BASE replacement actually took effect
    if str(paths.combfold_out_base) not in patched_path.read_text():
        sys.exit(f"[combfold-patch] OUTPUT_BASE substitution failed in "
                 f"{paths.combfold_sh}. Expected an 'OUTPUT_BASE=...' line.")
    return patched_path


def _patch_stoic_sbatch(paths: Paths, top_n: int) -> Path:
    """Write a patched copy of the user's stoic sbatch that points at second_setup.

    The user's existing sbatch hardcodes FASTA_DIR / OUTPUT_DIR and '--top-n N';
    replace them with the second-setup paths and the orchestrator's --top-n so
    STOIC generates exactly as many predictions as will be expanded to CombFold.
    Also redirect SBATCH log paths to second_setup/logs/ so the job can run from
    any CWD.
    """
    text = paths.stoic_sh.read_text()
    logs_dir = paths.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Replace the FASTA_DIR / OUTPUT_DIR variable assignments
    text = re.sub(
        r'^FASTA_DIR=.*$',
        f'FASTA_DIR="{paths.fastas_dir}"',
        text, count=1, flags=re.MULTILINE,
    )
    text = re.sub(
        r'^OUTPUT_DIR=.*$',
        f'OUTPUT_DIR="{paths.stoic_results_dir}"',
        text, count=1, flags=re.MULTILINE,
    )
    # Wire STOIC's --top-n to the orchestrator's --top-n (single source of truth,
    # so STOIC-generated == expanded-to-CombFold). Matches '--top-n 10' or
    # '--top-n=10'. count=0 replaces every occurrence.
    text, n_topn = re.subn(
        r'--top-n(\s+|=)\d+',
        f'--top-n {top_n}',
        text,
    )
    # Redirect SBATCH --output / --error to absolute logs_dir paths.
    text = _redirect_sbatch_logs(text, logs_dir, "stoic")

    patched_path = paths.out_dir / f"_patched_{paths.stoic_sh.name}"
    patched_path.write_text(text)
    patched_path.chmod(0o755)

    # Verify the path replacements actually took effect
    pt = patched_path.read_text()
    if str(paths.fastas_dir) not in pt or str(paths.stoic_results_dir) not in pt:
        sys.exit(f"[stoic-patch] FASTA_DIR/OUTPUT_DIR substitution failed in "
                 f"{paths.stoic_sh}. Expected 'FASTA_DIR=' and 'OUTPUT_DIR=' "
                 f"lines.")
    # Verify the --top-n patch took effect (fail loud rather than let STOIC and
    # the expand stage silently disagree on prediction count).
    if n_topn == 0:
        sys.exit(f"[stoic-patch] no '--top-n N' found to patch in {paths.stoic_sh}. "
                 f"Expected a 'stoic_predict_stoichiometry ... --top-n N' line.")
    if f"--top-n {top_n}" not in pt:
        sys.exit(f"[stoic-patch] '--top-n {top_n}' substitution failed in "
                 f"{paths.stoic_sh}.")
    print(f"[stoic-patch] patched --top-n -> {top_n} ({n_topn} occurrence(s))")
    return patched_path


# ===========================================================================
# Stage 3: Aggregate Stoic results (modified 02)
# ===========================================================================

PREDICTION_META_KEYS = {"rank", "probability"}


class StoicMappingError(Exception):
    """One complex could not be mapped back from STOIC sequences to UniProt IDs.

    Raised (instead of aborting the whole batch) so stage_aggregate_stoic can
    skip + log just the offending complex and keep processing the rest.
    """


def _load_seq_to_uniprot(path: Path) -> dict[str, set[str]]:
    """Load the sequence -> {uniprot_id, ...} map from the mapping CSV.

    A single sequence can map to MORE THAN ONE UniProt ID: in yeast, whole-
    genome-duplication paralog pairs (e.g. histone HTA1/HTA2, HHF1/HHF2, several
    ribosomal-protein gene pairs) are 100% identical at the protein level. We
    therefore keep the FULL set of candidate IDs per sequence and warn (rather
    than abort). The per-complex resolution in _parse_one_stoic_pred then
    disambiguates using the complex's own expected ID set; only a collision
    WITHIN a single complex is genuinely unresolvable.
    """
    df = pd.read_csv(path)
    if {"uniprot_id", "sequence"} - set(df.columns):
        sys.exit(f"{path} missing 'uniprot_id' or 'sequence' column.")
    mapping: dict[str, set[str]] = {}
    for uid, seq in _iter_seq_rows(df):
        mapping.setdefault(seq, set()).add(uid)
    n_shared = sum(1 for uids in mapping.values() if len(uids) > 1)
    if n_shared:
        shared = {s[:20] + "...": sorted(u) for s, u in mapping.items() if len(u) > 1}
        print(f"[aggregate-stoic] NOTE: {n_shared} sequence(s) shared by >1 "
              f"UniProt ID (identical paralogs). Resolved per-complex where "
              f"possible: {shared}")
    return mapping


def _parse_one_stoic_pred(
    entry: dict[str, Any],
    seq_to_uid: dict[str, set[str]],
    cpx_id: str,
    idx: int,
    expected_uids: set[str],
) -> tuple[float, int | None, dict[str, int], list[str]]:
    """Return (probability, n_copies, {uid: count}, protein_order_as_returned).

    Resolution rule for each sequence key:
      * candidates = seq_to_uid[seq]  (may be >1 for identical paralogs)
      * intersect with `expected_uids` (this complex's own subunit IDs) to
        disambiguate. Exactly one survivor -> use it.
      * zero survivors (or seq missing entirely) -> unmappable -> raise.
      * >1 survivor (two identical-sequence paralogs BOTH in this complex) ->
        genuinely ambiguous -> raise. The caller skips + logs the complex.

    Raises StoicMappingError instead of aborting the batch.

    NOTE: stoic's raw "rank" field is NOT an ordinal rank; it is the total
    subunit copy number. It is informational only (not used to build the spec),
    so it is treated as OPTIONAL here (see bug #9): probability is the only hard
    requirement.
    """
    if "probability" not in entry:
        raise StoicMappingError(f"[{cpx_id}] pred {idx} missing 'probability'")
    prob = float(entry["probability"])
    # rank/n_copies is optional + informational; degrade to None on any problem.
    try:
        n_copies: int | None = int(entry["rank"]) if "rank" in entry else None
    except (TypeError, ValueError):
        n_copies = None

    stoich: dict[str, int] = {}
    order: list[str] = []
    for key, value in entry.items():
        if key in PREDICTION_META_KEYS:
            continue
        seq = str(key)
        candidates = seq_to_uid.get(seq)
        if not candidates:
            raise StoicMappingError(
                f"[{cpx_id}] pred {idx}: sequence not in mapping "
                f"(first 60 chars): {seq[:60]!r}"
            )
        resolved = candidates & expected_uids
        if len(resolved) == 1:
            uid = next(iter(resolved))
        elif len(resolved) == 0:
            # Sequence maps to IDs, but none are expected in THIS complex.
            if len(candidates) == 1:
                uid = next(iter(candidates))  # unambiguous globally; trust it
            else:
                raise StoicMappingError(
                    f"[{cpx_id}] pred {idx}: sequence maps to {sorted(candidates)} "
                    f"but none are in this complex's expected set "
                    f"{sorted(expected_uids)}"
                )
        else:
            # >1 identical-sequence paralog present IN this complex -> cannot
            # tell which count belongs to which ID from sequence alone.
            raise StoicMappingError(
                f"[{cpx_id}] pred {idx}: ambiguous — sequence maps to "
                f"{sorted(resolved)}, both present in this complex"
            )
        stoich[uid] = int(value)
        order.append(uid)
    return prob, n_copies, stoich, order


def stage_aggregate_stoic(paths: Paths, max_preds: int = 10) -> None:
    """Read stoic_results/CPX-*/results.json; emit aggregated CSV.

    `max_preds` (= the orchestrator's --top-n) caps how many predictions per
    complex are stored as pred_1..pred_N columns; it is threaded from --top-n so
    aggregate, STOIC generation, and expand all agree. Fewer are stored if STOIC
    returned fewer.

    Missing folders are logged to missing_stoic_cpxs.txt and skipped.
    """
    print(f"[aggregate-stoic] Loading mapping {paths.uniprot_map_csv}")
    seq_to_uid = _load_seq_to_uniprot(paths.uniprot_map_csv)

    df_input = pd.read_csv(paths.input_csv)
    cpx_ids_in_csv = df_input["#Complex ac"].dropna().astype(str).unique().tolist()
    print(f"[aggregate-stoic] Expecting {len(cpx_ids_in_csv)} CPX folders")

    # Per-complex expected UniProt IDs (from the cleaned foldable set) used to
    # disambiguate identical-sequence paralogs during mapping.
    expected_by_cpx: dict[str, set[str]] = {}
    if "cleaned_complex" in df_input.columns:
        for _, r in df_input.iterrows():
            cid = str(r["#Complex ac"]).strip()
            expected_by_cpx[cid] = {
                p for p in str(r["cleaned_complex"]).split() if p
            }

    rows: list[dict[str, Any]] = []
    missing: list[str] = []       # missing/empty results.json
    unmappable: list[str] = []    # results present but couldn't map to UniProt
    MAX_PREDS = int(max_preds)

    for cpx_id in cpx_ids_in_csv:
        folder = paths.stoic_results_dir / cpx_id
        rj = folder / "results.json"
        if not rj.exists():
            missing.append(cpx_id)
            print(f"[aggregate-stoic]   [WARN] {cpx_id}: missing results.json")
            continue
        with open(rj) as fh:
            data = json.load(fh)
        if not isinstance(data, list) or not data:
            missing.append(cpx_id)
            print(f"[aggregate-stoic]   [WARN] {cpx_id}: empty results.json")
            continue

        expected_uids = expected_by_cpx.get(cpx_id, set())
        # Map every prediction for this complex; on ANY mapping failure, skip the
        # WHOLE complex (partial predictions would give inconsistent specs) and
        # log why — but keep processing all other complexes.
        try:
            parsed: list[tuple[float, int | None, dict[str, int], list[str]]] = []
            for i, entry in enumerate(data):
                parsed.append(
                    _parse_one_stoic_pred(entry, seq_to_uid, cpx_id, i, expected_uids)
                )
        except StoicMappingError as e:
            unmappable.append(f"{cpx_id}\t{e}")
            print(f"[aggregate-stoic]   [SKIP] {e}")
            continue
        # Sort by probability descending
        parsed.sort(key=lambda t: t[0], reverse=True)
        kept = parsed[:MAX_PREDS]

        row: dict[str, Any] = {
            "cpx_id": cpx_id,
            "n_predictions": len(parsed),
        }
        for slot in range(1, MAX_PREDS + 1):
            if slot <= len(kept):
                prob, n_copies, stoich, order = kept[slot - 1]
                row[f"pred_{slot}_stoichiometry"] = json.dumps(
                    stoich, sort_keys=True, separators=(",", ":")
                )
                row[f"pred_{slot}_score"] = json.dumps(
                    {"rank": slot, "n_copies": n_copies, "probability": prob},
                    separators=(",", ":"),
                )
                row[f"pred_{slot}_protein_order"] = json.dumps(order)
            else:
                row[f"pred_{slot}_stoichiometry"] = ""
                row[f"pred_{slot}_score"] = ""
                row[f"pred_{slot}_protein_order"] = ""
        rows.append(row)

    if missing:
        with open(paths.missing_stoic_log, "w") as fh:
            for c in missing:
                fh.write(c + "\n")
        print(f"[aggregate-stoic] Wrote {len(missing)} missing CPX(s) to "
              f"{paths.missing_stoic_log}")

    if unmappable:
        unmappable_log = paths.out_dir / "unmappable_stoic_cpxs.txt"
        with open(unmappable_log, "w") as fh:
            fh.write("#Complex ac\treason\n")
            for line in unmappable:
                fh.write(line + "\n")
        print(f"[aggregate-stoic] Wrote {len(unmappable)} unmappable/ambiguous "
              f"CPX(s) to {unmappable_log}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(paths.stoic_agg_csv, index=False)
    print(f"[aggregate-stoic] Wrote {len(rows)} rows -> {paths.stoic_agg_csv} "
          f"({len(missing)} missing, {len(unmappable)} unmappable skipped)")


# ===========================================================================
# Stage 4: Expand each input row into 10–11 stoichiometry rows
# ===========================================================================

def stage_expand(paths: Paths, top_n: int = 10) -> None:
    """Build the expanded CSV with one row per stoichiometry to run.

    Per input complex: top-N Stoic predictions + the curated 'true_spec' from the
    ComplexPortal TSV (if it isn't already among the top-N).

    Each expanded row carries 'combfold_submission' = the per-row stoichiometry
    spec ("UID(n),...") that is passed verbatim to the CombFold sbatch. For Stoic
    predictions these are Stoic's predicted counts; for the appended true row it
    is the curated ComplexPortal stoichiometry ((0)->1). 'stoich_prediction' is
    kept as an alias for backward compatibility with downstream stages.
    """
    print(f"[expand] Reading {paths.input_csv}")
    df_input = pd.read_csv(paths.input_csv)
    if "true_spec" not in df_input.columns:
        sys.exit("[expand] input CSV missing 'true_spec' column; run --mode fasta "
                 "against the raw TSV first.")
    print(f"[expand] Reading {paths.stoic_agg_csv}")
    df_stoic = pd.read_csv(paths.stoic_agg_csv)
    stoic_by_cpx = df_stoic.set_index("cpx_id").to_dict(orient="index")

    expanded_rows: list[dict[str, Any]] = []
    MAX_PREDS = int(top_n)

    for _, row in df_input.iterrows():
        cpx_id = str(row["#Complex ac"])
        # pandas reads empty CSV cells as NaN (float); coerce to "". true_spec
        # is "" exactly when ComplexPortal's own stoichiometry annotation was
        # "(0)" (unknown) for every subunit -- see build_cleaned_complexes_df.
        # There is no curated ground truth to compare against in that case, so
        # is_true_spec / stoic_pred_correct must say "unknown" rather than a
        # real True/False derived from a fabricated "1 copy each" fallback.
        true_spec_raw = row["true_spec"]
        true_spec = true_spec_raw if isinstance(true_spec_raw, str) else ""
        spec_unknown = not true_spec.strip()
        true_dict = parse_spec(true_spec) if not spec_unknown else {}
        true_canon = canonical_spec(true_dict) if not spec_unknown else ()
        # Preserve the protein order from the true spec for the appended-true row
        true_order = [t.split("(")[0].strip()
                      for t in true_spec.split(",") if t.strip()]

        # Collect Stoic predictions for this complex (if available)
        stoic_entry = stoic_by_cpx.get(cpx_id)
        preds: list[tuple[int, float, dict[str, int], list[str]]] = []  # (rank, prob, stoich, order)
        if stoic_entry:
            for slot in range(1, MAX_PREDS + 1):
                stoich_str = stoic_entry.get(f"pred_{slot}_stoichiometry", "")
                score_str = stoic_entry.get(f"pred_{slot}_score", "")
                order_str = stoic_entry.get(f"pred_{slot}_protein_order", "")
                # pandas reads empty CSV cells as NaN (float); coerce to ""
                if not isinstance(stoich_str, str) or not stoich_str:
                    continue
                stoich = json.loads(stoich_str)
                score = (json.loads(score_str)
                         if isinstance(score_str, str) and score_str else {})
                order = (json.loads(order_str)
                         if isinstance(order_str, str) and order_str else sorted(stoich))
                preds.append((slot, float(score.get("probability", float("nan"))),
                              stoich, order))

        canon_set = {canonical_spec(p[2]) for p in preds}
        # Emit one expanded row per Stoic prediction
        for slot, prob, stoich, order in preds:
            stoich_str = dict_to_spec_str(stoich, protein_order=order)
            new_row = dict(row)
            new_row["combfold_submission"] = stoich_str
            new_row["stoich_prediction"] = stoich_str  # alias for downstream stages
            new_row["stoic_pred_rank"] = slot
            new_row["pred_score"] = prob
            if spec_unknown:
                # No curated ground truth exists for this complex (ComplexPortal
                # listed every subunit count as "(0)"). Do not fabricate a
                # True/False verdict against the "1 copy each" filler value.
                new_row["is_true_spec"] = "unknown"
                new_row["stoic_pred_correct"] = "unknown"
            else:
                new_row["is_true_spec"] = (canonical_spec(stoich) == true_canon)
                new_row["stoic_pred_correct"] = (canonical_spec(stoich) == true_canon)
            expanded_rows.append(new_row)

        # Append the true row if it wasn't already among the Stoic preds.
        # Skipped entirely when spec_unknown: there is no curated ground truth
        # to append, and true_canon is already () in that case anyway.
        if not spec_unknown and true_canon and true_canon not in canon_set:
            true_str = dict_to_spec_str(true_dict, protein_order=true_order)
            new_row = dict(row)
            new_row["combfold_submission"] = true_str
            new_row["stoich_prediction"] = true_str  # alias for downstream stages
            new_row["stoic_pred_rank"] = pd.NA
            new_row["pred_score"] = pd.NA
            new_row["is_true_spec"] = True
            # This row is a SYNTHETIC ground-truth row appended precisely because
            # STOIC did NOT predict the true spec (stoic_pred_rank is NA). It must
            # NOT count as a correct STOIC prediction, or accuracy computed as
            # mean(stoic_pred_correct) is inflated on exactly the complexes STOIC
            # got wrong. is_true_spec stays True (it genuinely is ground truth);
            # stoic_pred_correct answers "did STOIC predict this?" -> False here.
            new_row["stoic_pred_correct"] = False
            expanded_rows.append(new_row)

    df_out = pd.DataFrame(expanded_rows)
    df_out.to_csv(paths.expanded_csv, index=False)
    print(f"[expand] Wrote {len(df_out)} rows (top_n={MAX_PREDS}) -> {paths.expanded_csv}")
    if len(df_out):
        print(f"[expand]   per-complex counts: "
              f"{df_out['#Complex ac'].value_counts().to_dict()}")


# ===========================================================================
# Stage 5: Submit CombFold jobs
# ===========================================================================

def stage_submit_combfold(
    paths: Paths, dry_run: bool = False, source: str = "pool", force: bool = False
) -> list[int]:
    """Submit one CombFold sbatch per unique combfold_submission spec.

    The s2 CombFold sbatch takes TWO positional args: the spec and SOURCE
    ('pair' or 'pool'). Every job is submitted with the given `source`
    (default 'pool').

    Idempotent by default (`force=False`): a spec is treated as "already
    attempted" and NOT resubmitted if its run's stable pairs CSV
    (`pdb_source_logs/<run_name>_pairs.csv`) already exists on disk. s2 writes
    that file near the end of the AF3-pair-gathering phase, before the
    CombFold search itself launches, so its presence is a disk-truth signal
    that survives regardless of whether the job later succeeded, failed, or
    was killed by the SLURM time limit (e.g. a 4h TIMEOUT with 0 assemblies
    still leaves this file in place). This check is disk-based (not
    registry-based) so it survives registry loss, matching the pattern used
    by `_stoic_already_complete()`. Pass `force=True` to resubmit every spec
    unconditionally (old behavior).

    Returns only the job ids submitted THIS call (empty list if every spec
    was already attempted and none needed submitting).
    """
    if paths.combfold_sh is None or not paths.combfold_sh.exists():
        sys.exit("[submit-combfold] --combfold-sh path does not exist.")
    if source not in ("pair", "pool"):
        sys.exit(f"[submit-combfold] source must be 'pair' or 'pool', got '{source}'.")

    print(f"[submit-combfold] Reading {paths.expanded_csv}")
    df = pd.read_csv(paths.expanded_csv)

    # Prefer combfold_submission; fall back to stoich_prediction alias.
    spec_col = "combfold_submission" if "combfold_submission" in df.columns else "stoich_prediction"
    unique_specs = list(dict.fromkeys(df[spec_col].astype(str)))

    # Patch the combfold sbatch so OUTPUT_BASE points at second_setup/CombFold
    patched_sh = _patch_combfold_sbatch(paths)
    print(f"[submit-combfold] using patched sbatch: {patched_sh}")

    def _already_attempted(spec: str) -> bool:
        run_name = f"{spec_to_complex_name(spec)}_{source}"
        pairs_csv = paths.combfold_out_base / "pdb_source_logs" / f"{run_name}_pairs.csv"
        return pairs_csv.exists()

    # Existing registry job ids, keyed by spec, so skipped specs keep their
    # previously recorded slurm_job_id instead of it being lost.
    old_reg = load_registry(paths)
    old_job_by_spec: dict[str, int | None] = {}
    for entry in old_reg.get("combfold_jobs", []):
        sp = entry.get("combfold_submission")
        if sp is not None:
            old_job_by_spec[str(sp)] = entry.get("slurm_job_id")

    if force:
        to_submit = list(unique_specs)
        skipped: list[str] = []
    else:
        skipped = [s for s in unique_specs if _already_attempted(s)]
        skipped_set = set(skipped)
        to_submit = [s for s in unique_specs if s not in skipped_set]

    print(f"[submit-combfold] {len(df)} rows -> {len(unique_specs)} unique specs "
          f"(source={source}); {len(skipped)} already attempted (skipped), "
          f"{len(to_submit)} to submit." + (" [force]" if force else ""))

    spec_to_job: dict[str, int | None] = {}
    submitted_ids: list[int] = []
    for spec in to_submit:
        cmd = ["sbatch", str(patched_sh), spec, source]
        if dry_run:
            print(f"[DRY-RUN] {' '.join(cmd)}")
            spec_to_job[spec] = None
            continue
        jid = _submit_sbatch(cmd, "submit-combfold", on_error="skip")
        spec_to_job[spec] = jid
        if jid:
            submitted_ids.append(jid)
        print(f"[submit-combfold]   spec='{spec[:60]}...' -> job {jid}")

    for spec in skipped:
        # Not touched this call; carry forward whatever job id was on record.
        spec_to_job[spec] = old_job_by_spec.get(spec)

    # Record per-row registry entries. Rows for already-attempted specs keep
    # their previously recorded slurm_job_id (via spec_to_job above) instead
    # of this call's fresh dict silently dropping submission history.
    rows_reg: list[dict[str, Any]] = []
    for csv_row, row in df.iterrows():
        spec = str(row[spec_col])
        rows_reg.append({
            "csv_row": int(csv_row),
            "cpx_id": str(row.get("#Complex ac", "")),
            "combfold_submission": spec,
            "stoich_prediction": spec,
            "complex_name": spec_to_complex_name(spec),
            "source": source,
            "slurm_job_id": spec_to_job.get(spec),
        })

    reg = load_registry(paths)
    reg["combfold_jobs"] = rows_reg
    reg["combfold_submitted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_registry(paths, reg)
    return submitted_ids


# ===========================================================================
# Stage 6: Submit dependency analyze job
# ===========================================================================

def stage_submit_analyze_dependency(
    paths: Paths, job_ids: list[int], source: str = "pool"
) -> None:
    """Submit `s3_analyze_CF_results.sbatch` with --dependency=afterany on all
    CombFold jobs.

    `source` MUST match the --combfold-source the CombFold jobs were actually
    submitted with (passed through as the sbatch script's 3rd positional arg),
    since s2 writes results into '<complex_name>_<source>_output/' and
    stage_analyze needs the same suffix to find them. A mismatch does not
    error - it silently reports every CF slot as n_assemblies=0 / reason='else'.

    Also passes `paths.this_script` (this orchestrator's own resolved
    absolute path, computed once in THIS process) as a 4th positional arg.
    s3_analyze_CF_results.sbatch uses it directly instead of trying to
    rediscover its own sibling via `dirname "${BASH_SOURCE[0]}"` at runtime:
    under `sbatch`, SLURM copies the submitted script into a spool directory
    before executing it, so `BASH_SOURCE[0]` inside the running job resolves
    to that spool copy's path, not to the real scripts/Pipeline/ directory -
    self-discovery there fails with "No such file or directory" for
    01_combfold_with_stoic.py. Passing the known-good path explicitly avoids
    relying on spool-relative self-discovery entirely.
    """
    if paths.analyze_sh is None or not paths.analyze_sh.exists():
        sys.exit("[submit-analyze] --analyze-sh path does not exist.")
    if not job_ids:
        print("[submit-analyze] no CombFold jobs submitted; skipping dependency.")
        return

    logs_dir = paths.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    dep = ",".join(str(j) for j in job_ids)
    cmd = [
        "sbatch",
        f"--dependency=afterany:{dep}",
        "--kill-on-invalid-dep=yes",
        f"--output={logs_dir}/ss_analyze_%j.out",
        f"--error={logs_dir}/ss_analyze_%j.err",
        str(paths.analyze_sh),
        str(paths.input_csv),
        str(paths.out_dir),
        source,
        str(paths.this_script),
    ]
    print(f"[submit-analyze] {' '.join(cmd)}")
    analyze_jid = _submit_sbatch(cmd, "submit-analyze", on_error="exit")
    print(f"[submit-analyze] -> analyze job {analyze_jid}")

    reg = load_registry(paths)
    reg["analyze_job_id"] = analyze_jid
    save_registry(paths, reg)


# ===========================================================================
# Stage 7: Submit Stoic dependency chain (aggregate -> expand -> combfold -> analyze)
# ===========================================================================

def stage_submit_post_stoic_chain(
    paths: Paths, stoic_job_id: int, top_n: int = 10, source: str = "pool"
) -> None:
    """Submit a single sbatch (--dependency=afterok:stoic_job_id) that runs
    aggregate-stoic + expand + submit-combfold + (submit-analyze dependency).

    The chain script is auto-generated and lives alongside the orchestrator.
    """
    if paths.analyze_sh is None or not paths.analyze_sh.exists():
        sys.exit("[submit-chain] --analyze-sh path does not exist.")
    chain_sh = paths.out_dir / "_post_stoic_chain.sbatch"
    chain_sh.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name=post_stoic_chain\n"
        f"#SBATCH --output={paths.out_dir}/logs/post_stoic_chain_%j.out\n"
        f"#SBATCH --error={paths.out_dir}/logs/post_stoic_chain_%j.err\n"
        f"set -euo pipefail\n"
        f"mkdir -p {paths.out_dir}/logs\n"
        f"python {paths.this_script} \\\n"
        f"    --input-csv {paths.input_csv} \\\n"
        f"    --out-dir {paths.out_dir} \\\n"
        f"    --setup-name {paths.setup_name} \\\n"
        f"    --combfold-sh {paths.combfold_sh} \\\n"
        f"    --analyze-sh {paths.analyze_sh} \\\n"
        f"    --top-n {top_n} \\\n"
        f"    --combfold-source {source} \\\n"
        f"    --mode post-stoic-chain\n"
    )
    chain_sh.chmod(0o755)

    cmd = [
        "sbatch",
        f"--dependency=afterok:{stoic_job_id}",
        "--kill-on-invalid-dep=yes",
        str(chain_sh),
    ]
    print(f"[submit-chain] {' '.join(cmd)}")
    chain_jid = _submit_sbatch(cmd, "submit-chain", on_error="exit")
    print(f"[submit-chain] -> chain job {chain_jid}")
    reg = load_registry(paths)
    reg["post_stoic_chain_job_id"] = chain_jid
    save_registry(paths, reg)


# ===========================================================================
# Stage 8: Analyze CombFold outputs
# ===========================================================================

def _parse_confidence_txt(conf_path: Path) -> dict[int, float]:
    scores: dict[int, float] = {}
    if not conf_path.exists():
        return scores
    with open(conf_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                m = re.search(r"output_clustered_(\d+)\.pdb", parts[0])
                idx = int(m.group(1)) if m else len(scores)
                try:
                    scores[idx] = float(parts[1])
                except ValueError:
                    pass
    return scores


# --- new helpers ------------------------------------------------------------
# Full AF3 per-pair metric schema logged by s2 into <run_name>_pairs.csv.
_PAIR_METRIC_FIELDS = [
    "af3_id1", "af3_id2", "chain_id1", "chain_id2",
    "input_name", "input_type", "batch_id", "seed", "sample",
    "ranking_score", "chain_pair_iptm",
    "chain_pair_pae_min_min", "chain_pair_pae_min_max", "chain_pair_pae_min_mean",
]


def _stoichiometry_given(identifiers: str) -> bool | None:
    """User-provided semantics on the RAW ComplexPortal identifiers string.

    Ignore non-protein tokens (CPX-..., CHEBI:...). Return:
      * None  if no protein entries are present,
      * False if any protein has stoichiometry 0 (i.e. unknown '(0)'),
      * True  if all protein entries have stoichiometry != 0.
    """
    uniprot_entry_re = re.compile(r"([A-Za-z0-9:_-]+)\((\d+)\)")

    def _is_uniprot_id(ident: str) -> bool:
        return not (ident.startswith("CPX-") or ident.startswith("CHEBI:"))

    nums = [int(n) for ident, n in uniprot_entry_re.findall(str(identifiers))
            if _is_uniprot_id(ident)]
    if not nums:
        return None
    return all(n != 0 for n in nums)


def _read_run(run_name: str, combfold_out_base: Path, source: str) -> dict[str, Any]:
    """Read one CombFold run's outputs + pairs CSV. Cached by run_name upstream.

    Returns dict with:
      success (bool), n_outputs (int), confidence (dict[int,float]),
      pairs_written (list[dict] with protein1/protein2/pair_type + metrics),
      any_missing (bool: a required pair had status 'missing'),
      pairs_csv_exists (bool).
    """
    output_dir = combfold_out_base / f"{run_name}_output"
    assembled = output_dir / "assembled_results"
    if assembled.exists():
        pdbs = sorted(assembled.glob("output_clustered_*.pdb"))
        n_outputs = len(pdbs)
    else:
        n_outputs = 0
    success = n_outputs > 0
    confidence = _parse_confidence_txt(assembled / "confidence.txt")

    pairs_csv = combfold_out_base / "pdb_source_logs" / f"{run_name}_pairs.csv"
    pairs_written: list[dict[str, Any]] = []
    any_missing = False
    pairs_csv_exists = pairs_csv.exists()
    if pairs_csv_exists:
        pdf = pd.read_csv(pairs_csv)
        for _, prow in pdf.iterrows():
            status = str(prow.get("status", "")).strip()
            if status == "missing":
                any_missing = True
                continue
            if status != "written":
                continue
            entry = {
                "protein1": str(prow.get("protein1", "")).strip(),
                "protein2": str(prow.get("protein2", "")).strip(),
                "pair_type": str(prow.get("pair_type", "")).strip(),
            }
            for f in _PAIR_METRIC_FIELDS:
                entry[f] = prow[f] if f in pdf.columns else None
            pairs_written.append(entry)
    return {
        "success": success,
        "n_outputs": n_outputs,
        "confidence": confidence,
        "pairs_written": pairs_written,
        "any_missing": any_missing,
        "pairs_csv_exists": pairs_csv_exists,
    }


def _sacct_states(job_ids: list[int]) -> dict[int, str]:
    """Batched `sacct` lookup of the base state for each job id.

    One `sacct -j id1,id2,... --format=JobID,State --parsable2 --noheader`
    call for ALL job ids at once (not per-run), to avoid hundreds/thousands
    of subprocess calls when scaled to a full run. Rows for the '.batch' /
    '.extern' sub-steps are skipped (only the parent job id row is kept).
    States like 'CANCELLED by 12345' or 'CANCELLED+' are normalized to their
    leading word (e.g. 'CANCELLED'), and 'TIMEOUT'-suffixed variants match
    exactly on 'TIMEOUT'.

    Returns {} on any failure (sacct missing, non-zero exit, unparsable
    output, empty job_ids) -- this is non-fatal; callers must treat a missing
    job id in the returned dict the same as "state unknown".
    """
    if not job_ids:
        return {}
    ids_arg = ",".join(str(j) for j in job_ids)
    try:
        result = subprocess.run(
            ["sacct", "-j", ids_arg, "--format=JobID,State", "--parsable2", "--noheader"],
            capture_output=True, text=True,
        )
    except Exception as exc:
        print(f"[analyze] sacct lookup failed ({exc}); timeout detection disabled.")
        return {}
    if result.returncode != 0:
        print(f"[analyze] sacct returned non-zero ({result.returncode}); "
              f"timeout detection disabled. stderr: {result.stderr.strip()}")
        return {}

    states: dict[int, str] = {}
    try:
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            jobid_field, state_field = parts[0], parts[1]
            if "." in jobid_field:
                # Skip '.batch' / '.extern' / '.0' sub-step rows.
                continue
            try:
                jid = int(jobid_field)
            except ValueError:
                continue
            state = state_field.strip().split()[0] if state_field.strip() else ""
            states[jid] = state
    except Exception as exc:
        print(f"[analyze] sacct output unparsable ({exc}); timeout detection disabled.")
        return {}
    return states


def _reason(
    run_exists: bool,
    run_info: dict[str, Any] | None,
    run_name: str | None = None,
    job_id_by_run_name: dict[str, int] | None = None,
    sacct_states: dict[int, str] | None = None,
) -> str:
    """Failure-reason for one CF slot. '' if assembled; else the category.

    `run_name`/`job_id_by_run_name`/`sacct_states` are optional so existing
    callers (and tests) that don't need timeout detection keep working
    unchanged. When all three are supplied and the would-be 'else' case's
    job is found in `sacct_states` with state 'TIMEOUT', return 'timed_out'
    instead.
    """
    if not run_exists or run_info is None:
        return "not_run"
    if run_info["n_outputs"] > 0:
        return ""
    if run_info["any_missing"]:
        return "pair_not_found"
    if run_name is not None and job_id_by_run_name is not None and sacct_states is not None:
        jid = job_id_by_run_name.get(run_name)
        if jid is not None and sacct_states.get(jid) == "TIMEOUT":
            return "timed_out"
    return "else"


def _pair_key(p1: str, p2: str) -> tuple[str, str]:
    """Uppercased, order-independent (min,max) UniProt key."""
    a, b = p1.upper(), p2.upper()
    return (a, b) if a <= b else (b, a)


def stage_analyze(paths: Paths, source: str = "pool") -> None:
    """Build the per-complex WIDE final CSV + the tidy pairs-metrics side table.

    Wide CSV (one row per '#Complex ac'):
      complex_ac, identifiers (raw), pred_1..pred_N (stoich_json + ',' + score_json),
      correct_pred_count, correct_pred_rank, CF_1..CF_N (_n_assemblies/_confidence/_reason),
      CF_true_* , combfold_job_ids, n_pairs_used, pairs_used.
    Dict/list cells are written as their repr/JSON string (single-file constraint).

    Tidy CSV (paths.pairs_metrics_csv): one row per (complex_ac, used unordered
    pair) with the full AF3 per-pair metric schema as typed columns.

    `source` MUST match the SOURCE the CombFold jobs were submitted with (s2
    writes to '<complex_name>_<source>_output/').
    """
    if source not in ("pair", "pool"):
        sys.exit(f"[analyze] source must be 'pair' or 'pool', got '{source}'.")
    print(f"[analyze] Reading {paths.expanded_csv} (source={source})")
    df = pd.read_csv(paths.expanded_csv)

    spec_col = "combfold_submission" if "combfold_submission" in df.columns else "stoich_prediction"
    ac_col = "#Complex ac"
    id_col = "Identifiers (and stoichiometry) of molecules in complex"

    # ---- determine N (number of pred/CF slots) -----------------------------
    n_from_rank = 0
    if "stoic_pred_rank" in df.columns:
        ranks = pd.to_numeric(df["stoic_pred_rank"], errors="coerce").dropna()
        if len(ranks):
            n_from_rank = int(ranks.max())
    n_from_agg = 0
    try:
        agg_cols = pd.read_csv(paths.stoic_agg_csv, nrows=0).columns
        n_from_agg = sum(1 for c in agg_cols
                         if re.fullmatch(r"pred_\d+_stoichiometry", c))
    except Exception:
        pass
    N = max(n_from_rank, n_from_agg, 1)
    print(f"[analyze] slots N={N} (from rank={n_from_rank}, from agg={n_from_agg})")

    # ---- aggregated STOIC strings, keyed by cpx_id -------------------------
    agg = pd.read_csv(paths.stoic_agg_csv, dtype=str).fillna("")
    agg_by_cpx: dict[str, dict[str, str]] = {}
    if "cpx_id" in agg.columns:
        for _, arow in agg.iterrows():
            agg_by_cpx[str(arow["cpx_id"])] = dict(arow)

    # ---- job IDs per complex_name (from registry) --------------------------
    jobs_by_cpx: dict[str, list] = {}
    job_id_by_run_name: dict[str, int] = {}
    try:
        with open(paths.registry_json) as fh:
            reg = json.load(fh)
        for j in reg.get("combfold_jobs", []):
            cid = str(j.get("cpx_id", ""))
            jid = j.get("slurm_job_id")
            if cid and jid is not None:
                jobs_by_cpx.setdefault(cid, [])
                if jid not in jobs_by_cpx[cid]:
                    jobs_by_cpx[cid].append(jid)
            cname = j.get("complex_name")
            if cname and jid is not None:
                # run_name = "<complex_name>_<source>"; last-one-wins on dup.
                job_id_by_run_name[f"{cname}_{j.get('source', source)}"] = jid
    except Exception as exc:
        print(f"[analyze] registry not read ({exc}); combfold_job_ids empty.")

    # ---- sacct states, one batched call for every job id in the registry ---
    all_job_ids = sorted({jid for jid in job_id_by_run_name.values()})
    sacct_states = _sacct_states(all_job_ids)
    if sacct_states:
        still_running = [jid for jid in all_job_ids
                          if sacct_states.get(jid) in ("PENDING", "RUNNING")]
        if still_running:
            print(f"[analyze] WARNING: {len(still_running)} CombFold job(s) still "
                  f"PENDING/RUNNING at analyze-time: {still_running}")

    # ---- per-run cache (assemblies/confidence/pairs) -----------------------
    run_cache: dict[str, dict[str, Any]] = {}

    def get_run(spec: str) -> dict[str, Any]:
        rn = f"{spec_to_complex_name(spec)}_{source}"
        if rn not in run_cache:
            run_cache[rn] = _read_run(rn, paths.combfold_out_base, source)
        return run_cache[rn]

    wide_rows: list[dict[str, Any]] = []
    tidy_rows: list[dict[str, Any]] = []

    for cpx_id, grp in df.groupby(ac_col, sort=False):
        cpx_id = str(cpx_id)
        first = grp.iloc[0]
        identifiers = first.get(id_col, "")
        identifiers = identifiers if isinstance(identifiers, str) else ""

        out: dict[str, Any] = {"complex_ac": cpx_id, "identifiers": identifiers}

        agg_entry = agg_by_cpx.get(cpx_id, {})

        # -- STOIC prediction columns pred_1..N (from aggregated CSV) --------
        for i in range(1, N + 1):
            stoich = agg_entry.get(f"pred_{i}_stoichiometry", "")
            score = agg_entry.get(f"pred_{i}_score", "")
            out[f"pred_{i}"] = f"{stoich},{score}" if (stoich or score) else ""

        # -- correct_pred_count / correct_pred_rank --------------------------
        given = _stoichiometry_given(identifiers)
        pred_rows = grp[pd.to_numeric(grp["stoic_pred_rank"], errors="coerce").notna()]
        if given is not True:
            # None or False -> ground truth not given -> unknown
            out["correct_pred_count"] = "unknown"
            out["correct_pred_rank"] = "unknown"
        else:
            correct = pred_rows[pred_rows["stoic_pred_correct"]
                                .astype(str).str.lower().eq("true")]
            if len(correct) == 0:
                out["correct_pred_count"] = "none"
                out["correct_pred_rank"] = "none"
            else:
                ranks = pd.to_numeric(correct["stoic_pred_rank"], errors="coerce").dropna()
                out["correct_pred_count"] = int(len(correct))
                out["correct_pred_rank"] = int(ranks.min())

        # -- CF blocks per slot ---------------------------------------------
        # Map slot -> spec for this complex's prediction rows.
        slot_to_spec: dict[int, str] = {}
        for _, r in pred_rows.iterrows():
            slot = int(float(r["stoic_pred_rank"]))
            slot_to_spec[slot] = str(r[spec_col])

        pairs_union: dict[tuple[str, str], dict[str, Any]] = {}

        def collect_pairs(run_info: dict[str, Any]):
            for pw in run_info["pairs_written"]:
                key = _pair_key(pw["protein1"], pw["protein2"])
                if key not in pairs_union:
                    pairs_union[key] = pw

        for i in range(1, N + 1):
            if i in slot_to_spec:
                spec_i = slot_to_spec[i]
                ri = get_run(spec_i)
                run_name_i = f"{spec_to_complex_name(spec_i)}_{source}"
                out[f"CF_{i}_n_assemblies"] = ri["n_outputs"]
                out[f"CF_{i}_confidence"] = ri["confidence"]
                out[f"CF_{i}_reason"] = _reason(
                    True, ri, run_name_i, job_id_by_run_name, sacct_states
                )
                collect_pairs(ri)
            else:
                out[f"CF_{i}_n_assemblies"] = 0
                out[f"CF_{i}_confidence"] = {}
                out[f"CF_{i}_reason"] = "not_run"

        # -- CF_true block ---------------------------------------------------
        # Synthetic true row: stoic_pred_rank is NA AND is_true_spec == True.
        rank_na = pd.to_numeric(grp["stoic_pred_rank"], errors="coerce").isna()
        is_true = grp["is_true_spec"].astype(str).str.lower().eq("true")
        true_rows = grp[rank_na & is_true]
        if len(true_rows):
            true_spec = str(true_rows.iloc[0][spec_col])
            rt = get_run(true_spec)
            run_name_true = f"{spec_to_complex_name(true_spec)}_{source}"
            out["CF_true_n_assemblies"] = rt["n_outputs"]
            out["CF_true_confidence"] = rt["confidence"]
            out["CF_true_reason"] = _reason(
                True, rt, run_name_true, job_id_by_run_name, sacct_states
            )
            collect_pairs(rt)
        else:
            out["CF_true_n_assemblies"] = 0
            out["CF_true_confidence"] = {}
            out["CF_true_reason"] = "not_run"

        # -- job ids + pairs summary ----------------------------------------
        out["combfold_job_ids"] = jobs_by_cpx.get(cpx_id, [])

        pair_labels = sorted(f"{a}_{b}" for (a, b) in pairs_union)
        out["n_pairs_used"] = len(pair_labels)
        out["pairs_used"] = pair_labels

        # -- tidy rows -------------------------------------------------------
        for (a, b), pw in sorted(pairs_union.items()):
            trow = {
                "complex_ac": cpx_id,
                "protein1": pw["protein1"].upper(),
                "protein2": pw["protein2"].upper(),
                "pair": f"{a}_{b}",
                "pair_type": pw["pair_type"],
            }
            for f in _PAIR_METRIC_FIELDS:
                trow[f] = pw.get(f)
            tidy_rows.append(trow)

        wide_rows.append(out)

    # ---- write wide CSV ----------------------------------------------------
    # Fixed column order, identical across rows.
    cols = ["complex_ac", "identifiers"]
    cols += [f"pred_{i}" for i in range(1, N + 1)]
    cols += ["correct_pred_count", "correct_pred_rank"]
    for i in range(1, N + 1):
        cols += [f"CF_{i}_n_assemblies", f"CF_{i}_confidence", f"CF_{i}_reason"]
    cols += ["CF_true_n_assemblies", "CF_true_confidence", "CF_true_reason"]
    cols += ["combfold_job_ids", "n_pairs_used", "pairs_used"]

    wide = pd.DataFrame(wide_rows, columns=cols)
    wide.to_csv(paths.final_csv, index=False)
    print(f"[analyze] Wrote {len(wide)} complex rows -> {paths.final_csv}")

    # ---- write tidy pairs-metrics CSV --------------------------------------
    tidy_cols = ["complex_ac", "protein1", "protein2", "pair", "pair_type"] + _PAIR_METRIC_FIELDS
    tidy = pd.DataFrame(tidy_rows, columns=tidy_cols)
    tidy.to_csv(paths.pairs_metrics_csv, index=False)
    print(f"[analyze] Wrote {len(tidy)} pair rows -> {paths.pairs_metrics_csv}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-tsv", type=Path, default=None,
                    help="Raw ComplexPortal TSV (front-end input for --mode "
                         "fasta/all). Required unless --input-csv is given.")
    ap.add_argument("--input-csv", type=Path, default=None,
                    help="Pre-built working CSV (skips TSV cleaning). If omitted, "
                         "the derived <out-dir>/cleaned_complexes.csv is used.")
    ap.add_argument("--seq-csv", type=Path, default=None,
                    help="Local uniprot_id,sequence CSV (sole sequence source; "
                         "required for --mode fasta/all).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--setup-name", type=str, default=SETUP_NAME,
                    help=f"Single label interpolated into every setup-tagged "
                         f"output filename (default {SETUP_NAME!r}). Change this "
                         f"one flag instead of editing individual paths.")
    ap.add_argument("--stoic-sh", type=Path, default=None,
                    help="Path to 03_run_stoic.sbatch (required for submit-stoic/all)")
    ap.add_argument("--combfold-sh", type=Path, default=None,
                    help="Path to 05_run_CombFold.sbatch (required for submit-combfold/all)")
    ap.add_argument("--analyze-sh", type=Path, default=None,
                    help="Path to 11_analyze.sbatch (required for all/post-stoic-chain)")
    ap.add_argument("--top-n", type=int, default=10,
                    help="Number of Stoic predictions per complex to expand into "
                         "CombFold jobs (default 10; the curated true_spec is "
                         "always added if not already present).")
    ap.add_argument("--force-stoic", action="store_true",
                    help="Resubmit STOIC even if results.json already exist on "
                         "disk for all expected complexes (default: skip if present).")
    ap.add_argument("--force-combfold", action="store_true",
                    help="Resubmit CombFold for every spec even if it was already "
                         "attempted (has a pdb_source_logs/<run>_pairs.csv on disk), "
                         "regardless of whether that prior attempt succeeded, failed, "
                         "or timed out (default: skip specs already attempted).")
    ap.add_argument("--combfold-source", choices=["pair", "pool"], default="pool",
                    help="SOURCE arg passed to the CombFold sbatch (default pool).")
    ap.add_argument(
        "--mode",
        choices=[
            "fasta", "submit-stoic", "aggregate-stoic", "expand",
            "submit-combfold", "analyze",
            "post-stoic-chain",   # internal: aggregate+expand+submit-combfold+submit-analyze
            "all",
        ],
        default="all",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="(submit-combfold) print sbatch commands without running")
    args = ap.parse_args()

    paths = Paths(args)
    paths.ensure_dirs()

    # --- Mode-specific required-arg checks ----------------------------------
    if args.mode in ("fasta", "all") and not args.input_tsv:
        sys.exit(f"--mode {args.mode} requires --input-tsv (raw ComplexPortal TSV)")
    if args.mode in ("fasta", "all") and not args.seq_csv:
        sys.exit(f"--mode {args.mode} requires --seq-csv (local sequence CSV)")
    if args.mode in ("aggregate-stoic", "expand", "submit-combfold", "analyze",
                     "post-stoic-chain") and not (args.input_csv or paths.cleaned_csv.exists()):
        sys.exit(f"--mode {args.mode} requires --input-csv or an existing "
                 f"{paths.cleaned_csv} (run --mode fasta first)")
    if args.mode in ("submit-stoic", "all") and not args.stoic_sh:
        sys.exit(f"--mode {args.mode} requires --stoic-sh")
    if args.mode in ("submit-combfold", "post-stoic-chain", "all") and not args.combfold_sh:
        sys.exit(f"--mode {args.mode} requires --combfold-sh")
    if args.mode in ("post-stoic-chain", "all") and not args.analyze_sh:
        sys.exit(f"--mode {args.mode} requires --analyze-sh")
    if args.mode == "submit-combfold" and not args.analyze_sh:
        print("[warn] --analyze-sh not given; will NOT submit dependency analyze job.")

    if args.mode == "fasta":
        stage_fasta(paths)

    elif args.mode == "submit-stoic":
        if not paths.fastas_dir.exists() or not any(paths.fastas_dir.iterdir()):
            sys.exit("[submit-stoic] fastas/ is empty; run --mode fasta first.")
        stage_submit_stoic(paths, top_n=args.top_n, force=args.force_stoic)

    elif args.mode == "aggregate-stoic":
        stage_aggregate_stoic(paths, max_preds=args.top_n)

    elif args.mode == "expand":
        stage_expand(paths, top_n=args.top_n)

    elif args.mode == "submit-combfold":
        if not paths.expanded_csv.exists():
            sys.exit("[submit-combfold] expanded.csv missing; run --mode expand first.")
        ids = stage_submit_combfold(paths, dry_run=args.dry_run,
                                    source=args.combfold_source,
                                    force=args.force_combfold)
        if args.analyze_sh and not args.dry_run:
            stage_submit_analyze_dependency(paths, ids, source=args.combfold_source)

    elif args.mode == "analyze":
        stage_analyze(paths, source=args.combfold_source)

    elif args.mode == "post-stoic-chain":
        # Internal: this is what the dependency sbatch runs after Stoic finishes.
        stage_aggregate_stoic(paths, max_preds=args.top_n)
        stage_expand(paths, top_n=args.top_n)
        ids = stage_submit_combfold(paths, source=args.combfold_source,
                                    force=args.force_combfold)
        if ids:
            if args.analyze_sh:
                stage_submit_analyze_dependency(paths, ids, source=args.combfold_source)
        else:
            print("[post-stoic-chain] No new CombFold jobs submitted (all specs "
                  "already attempted); running analyze inline now.")
            stage_analyze(paths, source=args.combfold_source)

    elif args.mode == "all":
        # Full end-to-end via sbatch dependencies (no blocking polling).
        stage_fasta(paths)

        # If STOIC already ran (results.json on disk for every complex) and the
        # user did not force a rerun, there is no fresh stoic_job_id to chain an
        # afterok dependency on. Run the post-STOIC steps DIRECTLY and inline
        # (same body as --mode post-stoic-chain); the STOIC outputs already exist.
        if not args.force_stoic and _stoic_already_complete(paths):
            print("[all] STOIC results already present; skipping STOIC submission "
                  "and running aggregate -> expand -> combfold -> analyze inline "
                  "(use --force-stoic to rerun STOIC).")
            stage_aggregate_stoic(paths, max_preds=args.top_n)
            stage_expand(paths, top_n=args.top_n)
            ids = stage_submit_combfold(paths, source=args.combfold_source,
                                        force=args.force_combfold)
            if ids:
                if args.analyze_sh:
                    stage_submit_analyze_dependency(paths, ids, source=args.combfold_source)
                print(f"\n[all] Submitted CombFold (STOIC skipped). Monitor with "
                      f"squeue; final CSV will appear at:\n  {paths.final_csv}")
            else:
                print("[all] No new CombFold jobs needed (every spec already "
                      "attempted); running analyze inline now "
                      "(use --force-combfold to rerun CombFold).")
                stage_analyze(paths, source=args.combfold_source)
        else:
            stage_submit_stoic(paths, top_n=args.top_n, force=args.force_stoic)
            reg = load_registry(paths)
            stoic_jid = reg.get("stoic_job_id")
            if not stoic_jid:
                sys.exit("[all] No stoic_job_id in registry; cannot chain.")
            stage_submit_post_stoic_chain(paths, stoic_jid, top_n=args.top_n,
                                          source=args.combfold_source)
            print(f"\n[all] Submitted Stoic job {stoic_jid} and dependent chain. "
                  f"Monitor with squeue; final CSV will appear at:\n  {paths.final_csv}")


if __name__ == "__main__":
    main()
