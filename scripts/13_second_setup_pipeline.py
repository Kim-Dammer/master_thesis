#!/usr/bin/env python
"""
13_second_setup_pipeline.py
===========================
run with uv run scripts/13_second_setup_pipeline.py --mode all     --input-csv data/Pipeline/first_setup/all_pdb_present_first_setup_pipeline_complexes.csv     --out-dir   data/Pipeline/second_setup     --stoic-sh    scripts/03_run_stoic.sbatch     --combfold-sh scripts/05_run_CombFold.sbatch     --analyze-sh  scripts/12_analyze.sbatch

Single orchestrator for the second-setup Stoic + CombFold pipeline.

Pipeline:
    fasta -> submit-stoic -> aggregate-stoic -> expand -> submit-combfold -> analyze

Run with --mode all to submit the whole chain via sbatch dependencies
(no blocking polling). Or run any single mode independently.

Layout written:
    <out_dir>/
        fastas/<CPX>.fasta
        uniprot_mapped_seq_second_setup.csv
        stoic_results/<CPX>/{af3_input_*.json, results.json}
        stoic_results_aggregated_second_setup.csv
        second_setup_expanded.csv
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
import requests


# ---------------------------------------------------------------------------
# Spec parsing helpers (shared with 10_run_csv_combfold.py)
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(r"^([A-Za-z0-9_]+)\((\d+)\)$")


def parse_spec(spec: str) -> dict[str, int]:
    """Parse 'P00937(1),P00899(2)' -> {'P00937': 1, 'P00899': 2}."""
    counts: dict[str, int] = {}
    for tok in [t.strip() for t in str(spec).split(",") if t.strip()]:
        m = _ENTRY_RE.match(tok)
        if m:
            counts[m.group(1)] = int(m.group(2))
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
# Paths / config dataclass
# ---------------------------------------------------------------------------

class Paths:
    def __init__(self, args: argparse.Namespace):
        self.input_csv = Path(args.input_csv).resolve()
        self.out_dir = Path(args.out_dir).resolve()
        self.stoic_sh = Path(args.stoic_sh).resolve() if args.stoic_sh else None
        self.combfold_sh = Path(args.combfold_sh).resolve() if args.combfold_sh else None
        self.analyze_sh = Path(args.analyze_sh).resolve() if args.analyze_sh else None
        self.this_script = Path(__file__).resolve()

        # Subpaths
        self.fastas_dir = self.out_dir / "fastas"
        self.uniprot_map_csv = self.out_dir / "uniprot_mapped_seq_second_setup.csv"
        self.stoic_results_dir = self.out_dir / "stoic_results"
        self.stoic_agg_csv = self.out_dir / "stoic_results_aggregated_second_setup.csv"
        self.expanded_csv = self.out_dir / "second_setup_expanded.csv"
        self.registry_json = self.out_dir / "second_setup_job_registry.json"
        self.combfold_out_base = self.out_dir / "CombFold"
        self.final_csv = self.out_dir / (
            "all_pdb_present_second_setup_pipeline_complexes_combfold_results.csv"
        )
        self.missing_stoic_log = self.out_dir / "missing_stoic_cpxs.txt"

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


# ===========================================================================
# Stage 1: FASTA generation (replaces 01)
# ===========================================================================

def stage_fasta(paths: Paths) -> None:
    """Build one FASTA per #Complex ac using comb_fold_submission proteins.

    Also writes uniprot_mapped_seq_second_setup.csv (uniprot_id, sequence).
    """
    print(f"[fasta] Reading {paths.input_csv}")
    df = pd.read_csv(paths.input_csv)
    if "comb_fold_submission" not in df.columns or "#Complex ac" not in df.columns:
        sys.exit("Input CSV must have '#Complex ac' and 'comb_fold_submission' columns.")

    # Collect all unique UniProt IDs across rows
    all_ids: set[str] = set()
    for spec in df["comb_fold_submission"]:
        all_ids.update(parse_spec(spec).keys())
    all_ids_list = sorted(all_ids)
    print(f"[fasta] Total unique UniProt IDs across {len(df)} rows: {len(all_ids_list)}")

    # Fetch sequences in batches of 500
    BATCH = 500
    sequences: dict[str, str] = {}

    for i in range(0, len(all_ids_list), BATCH):
        batch = all_ids_list[i:i + BATCH]
        print(f"[fasta]  Batch {i // BATCH + 1}: {len(batch)} IDs...")
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/stream",
            params={
                "query": " OR ".join(f"accession:{a}" for a in batch),
                "format": "fasta",
                "size": len(batch),
            },
            timeout=120,
        )
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 10)))
            r = requests.get(
                "https://rest.uniprot.org/uniprotkb/stream",
                params={
                    "query": " OR ".join(f"accession:{a}" for a in batch),
                    "format": "fasta",
                    "size": len(batch),
                },
                timeout=120,
            )
        if r.status_code != 200:
            print(f"[fasta]  HTTP {r.status_code}, skipping batch")
            continue

        cur_id, cur_seq = None, []
        for line in r.text.splitlines():
            if line.startswith(">"):
                if cur_id:
                    sequences[cur_id] = "".join(cur_seq)
                cur_id = line.split("|")[1] if "|" in line else line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id:
            sequences[cur_id] = "".join(cur_seq)
        time.sleep(1)

    print(f"[fasta] Retrieved {len(sequences)}/{len(all_ids_list)} sequences")

    missing_ids = [uid for uid in all_ids_list if uid not in sequences]
    if missing_ids:
        sys.exit(f"[fasta] FATAL: {len(missing_ids)} UniProt ID(s) returned no "
                 f"sequence: {missing_ids[:10]}...")

    # Write the seq -> uniprot mapping CSV
    map_df = pd.DataFrame(
        [{"uniprot_id": uid, "sequence": sequences[uid]} for uid in all_ids_list]
    )
    map_df.to_csv(paths.uniprot_map_csv, index=False)
    print(f"[fasta] Wrote mapping: {paths.uniprot_map_csv}")

    # Write one FASTA per CPX (single copy per protein)
    rows = df[["#Complex ac", "comb_fold_submission"]].drop_duplicates(
        subset="#Complex ac"
    )
    n_written = 0
    for cpx_id, spec in rows.itertuples(index=False):
        proteins = spec_to_proteins(spec)
        fasta_path = paths.fastas_dir / f"{cpx_id}.fasta"
        with open(fasta_path, "w") as f:
            for uid in proteins:
                seq = sequences[uid]
                f.write(f">{uid}\n")
                for j in range(0, len(seq), 80):
                    f.write(seq[j:j + 80] + "\n")
        n_written += 1
    print(f"[fasta] Wrote {n_written} FASTA files to {paths.fastas_dir}")


# ===========================================================================
# Stage 2: Stoic submission (single sbatch, env-var overrides)
# ===========================================================================

def stage_submit_stoic(paths: Paths) -> None:
    """Submit the single Stoic GPU sbatch with env-var overrides for I/O paths."""
    if paths.stoic_sh is None or not paths.stoic_sh.exists():
        sys.exit("[submit-stoic] --stoic-sh path does not exist.")

    print(f"[submit-stoic] Submitting {paths.stoic_sh}")

    # Pass FASTA / output via env vars; the sbatch script needs to honour these.
    # If the user's existing sbatch hardcodes paths, we sed-patch a temp copy.
    patched_sh = _patch_stoic_sbatch(paths)

    env = os.environ.copy()
    env["SECOND_SETUP_FASTA_DIR"] = str(paths.fastas_dir)
    env["SECOND_SETUP_OUTPUT_DIR"] = str(paths.stoic_results_dir)

    result = subprocess.run(
        ["sbatch", str(patched_sh)],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        sys.exit(f"[submit-stoic] sbatch failed: {result.stderr.strip()}")

    m = re.search(r"Submitted batch job (\d+)", result.stdout)
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
    text = re.sub(
        r'^#SBATCH\s+--output=.*$',
        f'#SBATCH --output={logs_dir}/combfold_%j.out',
        text, count=1, flags=re.MULTILINE,
    )
    text = re.sub(
        r'^#SBATCH\s+--error=.*$',
        f'#SBATCH --error={logs_dir}/combfold_%j.err',
        text, count=1, flags=re.MULTILINE,
    )

    patched_path = paths.out_dir / f"_patched_{paths.combfold_sh.name}"
    patched_path.write_text(text)
    patched_path.chmod(0o755)

    # Verify the OUTPUT_BASE replacement actually took effect
    if str(paths.combfold_out_base) not in patched_path.read_text():
        sys.exit(f"[combfold-patch] OUTPUT_BASE substitution failed in "
                 f"{paths.combfold_sh}. Expected an 'OUTPUT_BASE=...' line.")
    return patched_path


def _patch_stoic_sbatch(paths: Paths) -> Path:
    """Write a patched copy of the user's stoic sbatch that points at second_setup.

    The user's existing sbatch hardcodes FASTA_DIR / OUTPUT_DIR; replace them
    with second-setup paths. Also redirect SBATCH log paths to second_setup/logs/
    so the job can run from any CWD.
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
    # Redirect SBATCH --output / --error to absolute logs_dir paths.
    text = re.sub(
        r'^#SBATCH\s+--output=.*$',
        f'#SBATCH --output={logs_dir}/stoic_%j.out',
        text, count=1, flags=re.MULTILINE,
    )
    text = re.sub(
        r'^#SBATCH\s+--error=.*$',
        f'#SBATCH --error={logs_dir}/stoic_%j.err',
        text, count=1, flags=re.MULTILINE,
    )

    patched_path = paths.out_dir / f"_patched_{paths.stoic_sh.name}"
    patched_path.write_text(text)
    patched_path.chmod(0o755)

    # Verify the path replacements actually took effect
    pt = patched_path.read_text()
    if str(paths.fastas_dir) not in pt or str(paths.stoic_results_dir) not in pt:
        sys.exit(f"[stoic-patch] FASTA_DIR/OUTPUT_DIR substitution failed in "
                 f"{paths.stoic_sh}. Expected 'FASTA_DIR=' and 'OUTPUT_DIR=' "
                 f"lines.")
    return patched_path


# ===========================================================================
# Stage 3: Aggregate Stoic results (modified 02)
# ===========================================================================

PREDICTION_META_KEYS = {"rank", "probability"}


def _load_seq_to_uniprot(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if {"uniprot_id", "sequence"} - set(df.columns):
        sys.exit(f"{path} missing 'uniprot_id' or 'sequence' column.")
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        seq = str(row["sequence"]).strip()
        uid = str(row["uniprot_id"]).strip()
        if not seq or not uid:
            continue
        if seq in mapping and mapping[seq] != uid:
            sys.exit(f"Duplicate sequence -> {mapping[seq]} and {uid}")
        mapping[seq] = uid
    return mapping


def _parse_one_stoic_pred(
    entry: dict[str, Any],
    seq_to_uid: dict[str, str],
    cpx_id: str,
    idx: int,
) -> tuple[float, int, dict[str, int], list[str]]:
    """Return (probability, rank, {uid: count}, protein_order_as_returned)."""
    if "probability" not in entry or "rank" not in entry:
        sys.exit(f"[{cpx_id}] pred {idx} missing rank/probability")
    prob = float(entry["probability"])
    rank = int(entry["rank"])
    stoich: dict[str, int] = {}
    order: list[str] = []
    for key, value in entry.items():
        if key in PREDICTION_META_KEYS:
            continue
        seq = str(key)
        if seq not in seq_to_uid:
            sys.exit(f"[{cpx_id}] pred {idx}: sequence not in mapping "
                     f"(first 60 chars): {seq[:60]!r}")
        uid = seq_to_uid[seq]
        stoich[uid] = int(value)
        order.append(uid)
    return prob, rank, stoich, order


def stage_aggregate_stoic(paths: Paths) -> None:
    """Read stoic_results/CPX-*/results.json; emit aggregated CSV.

    Missing folders are logged to missing_stoic_cpxs.txt and skipped.
    """
    print(f"[aggregate-stoic] Loading mapping {paths.uniprot_map_csv}")
    seq_to_uid = _load_seq_to_uniprot(paths.uniprot_map_csv)

    df_input = pd.read_csv(paths.input_csv)
    cpx_ids_in_csv = df_input["#Complex ac"].dropna().astype(str).unique().tolist()
    print(f"[aggregate-stoic] Expecting {len(cpx_ids_in_csv)} CPX folders")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    MAX_PREDS = 10

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

        parsed: list[tuple[float, int, dict[str, int], list[str]]] = []
        for i, entry in enumerate(data):
            parsed.append(_parse_one_stoic_pred(entry, seq_to_uid, cpx_id, i))
        # Sort by probability descending
        parsed.sort(key=lambda t: t[0], reverse=True)
        kept = parsed[:MAX_PREDS]

        row: dict[str, Any] = {
            "cpx_id": cpx_id,
            "n_predictions": len(parsed),
        }
        for slot in range(1, MAX_PREDS + 1):
            if slot <= len(kept):
                prob, rank, stoich, order = kept[slot - 1]
                row[f"pred_{slot}_stoichiometry"] = json.dumps(
                    stoich, sort_keys=True, separators=(",", ":")
                )
                row[f"pred_{slot}_score"] = json.dumps(
                    {"rank": rank, "probability": prob},
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

    out_df = pd.DataFrame(rows)
    out_df.to_csv(paths.stoic_agg_csv, index=False)
    print(f"[aggregate-stoic] Wrote {len(rows)} rows -> {paths.stoic_agg_csv}")


# ===========================================================================
# Stage 4: Expand each input row into 10–11 stoichiometry rows
# ===========================================================================

def stage_expand(paths: Paths) -> None:
    """Build second_setup_expanded.csv with one row per stoichiometry to run.

    Per input row: top-10 Stoic predictions + the true comb_fold_submission
    if it isn't already among the top-10.
    """
    print(f"[expand] Reading {paths.input_csv}")
    df_input = pd.read_csv(paths.input_csv)
    print(f"[expand] Reading {paths.stoic_agg_csv}")
    df_stoic = pd.read_csv(paths.stoic_agg_csv)
    stoic_by_cpx = df_stoic.set_index("cpx_id").to_dict(orient="index")

    expanded_rows: list[dict[str, Any]] = []
    MAX_PREDS = 10

    for _, row in df_input.iterrows():
        cpx_id = str(row["#Complex ac"])
        true_spec = str(row["comb_fold_submission"])
        true_dict = parse_spec(true_spec)
        true_canon = canonical_spec(true_dict)
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
            new_row["stoich_prediction"] = stoich_str
            new_row["stoic_pred_rank"] = slot
            new_row["pred_score"] = prob
            new_row["stoic_pred_correct"] = (canonical_spec(stoich) == true_canon)
            expanded_rows.append(new_row)

        # Append the true row if it wasn't already among the Stoic preds
        if true_canon not in canon_set:
            new_row = dict(row)
            new_row["stoich_prediction"] = dict_to_spec_str(true_dict, protein_order=true_order)
            new_row["stoic_pred_rank"] = pd.NA
            new_row["pred_score"] = pd.NA
            new_row["stoic_pred_correct"] = True
            expanded_rows.append(new_row)

    df_out = pd.DataFrame(expanded_rows)
    df_out.to_csv(paths.expanded_csv, index=False)
    print(f"[expand] Wrote {len(df_out)} rows -> {paths.expanded_csv}")
    print(f"[expand]   per-complex counts: "
          f"{df_out['#Complex ac'].value_counts().to_dict()}")


# ===========================================================================
# Stage 5: Submit CombFold jobs
# ===========================================================================

def stage_submit_combfold(paths: Paths, dry_run: bool = False) -> list[int]:
    """Submit one CombFold sbatch per unique stoich_prediction."""
    if paths.combfold_sh is None or not paths.combfold_sh.exists():
        sys.exit("[submit-combfold] --combfold-sh path does not exist.")

    print(f"[submit-combfold] Reading {paths.expanded_csv}")
    df = pd.read_csv(paths.expanded_csv)

    unique_specs = list(dict.fromkeys(df["stoich_prediction"].astype(str)))
    print(f"[submit-combfold] {len(df)} rows -> {len(unique_specs)} unique specs")

    # Patch the combfold sbatch so OUTPUT_BASE points at second_setup/CombFold
    patched_sh = _patch_combfold_sbatch(paths)
    print(f"[submit-combfold] using patched sbatch: {patched_sh}")

    spec_to_job: dict[str, int | None] = {}
    submitted_ids: list[int] = []
    for spec in unique_specs:
        cmd = ["sbatch", str(patched_sh), spec]
        if dry_run:
            print(f"[DRY-RUN] {' '.join(cmd)}")
            spec_to_job[spec] = None
            continue
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[submit-combfold]   FAILED for {spec}: {result.stderr.strip()}")
            spec_to_job[spec] = None
            continue
        m = re.search(r"Submitted batch job (\d+)", result.stdout)
        jid = int(m.group(1)) if m else None
        spec_to_job[spec] = jid
        if jid:
            submitted_ids.append(jid)
        print(f"[submit-combfold]   spec='{spec[:60]}...' -> job {jid}")

    # Record per-row registry entries
    rows_reg: list[dict[str, Any]] = []
    for csv_row, row in df.iterrows():
        spec = str(row["stoich_prediction"])
        rows_reg.append({
            "csv_row": int(csv_row),
            "cpx_id": str(row.get("#Complex ac", "")),
            "stoich_prediction": spec,
            "complex_name": spec_to_complex_name(spec),
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

def stage_submit_analyze_dependency(paths: Paths, job_ids: list[int]) -> None:
    """Submit `11_analyze.sbatch` with --dependency=afterany on all CombFold jobs."""
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
    ]
    print(f"[submit-analyze] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[submit-analyze] sbatch failed: {result.stderr.strip()}")
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    analyze_jid = int(m.group(1)) if m else None
    print(f"[submit-analyze] -> analyze job {analyze_jid}")

    reg = load_registry(paths)
    reg["analyze_job_id"] = analyze_jid
    save_registry(paths, reg)


# ===========================================================================
# Stage 7: Submit Stoic dependency chain (aggregate -> expand -> combfold -> analyze)
# ===========================================================================

def stage_submit_post_stoic_chain(paths: Paths, stoic_job_id: int) -> None:
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
        f"#SBATCH --time=01:00:00\n"
        f"#SBATCH --cpus-per-task=2\n"
        f"#SBATCH --mem-per-cpu=8G\n"
        f"set -euo pipefail\n"
        f"mkdir -p {paths.out_dir}/logs\n"
        f"python {paths.this_script} \\\n"
        f"    --input-csv {paths.input_csv} \\\n"
        f"    --out-dir {paths.out_dir} \\\n"
        f"    --combfold-sh {paths.combfold_sh} \\\n"
        f"    --analyze-sh {paths.analyze_sh} \\\n"
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[submit-chain] sbatch failed: {result.stderr.strip()}")
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    chain_jid = int(m.group(1)) if m else None
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


def stage_analyze(paths: Paths) -> None:
    """Parse CombFold outputs and write the final enriched CSV."""
    print(f"[analyze] Reading {paths.expanded_csv}")
    df = pd.read_csv(paths.expanded_csv)

    col_success: list[bool] = []
    col_n_outputs: list[int] = []
    col_confidence: list[str] = []

    cache: dict[str, dict[str, Any]] = {}

    for idx, row in df.iterrows():
        spec = str(row["stoich_prediction"])
        complex_name = spec_to_complex_name(spec)

        if complex_name in cache:
            c = cache[complex_name]
        else:
            output_dir = paths.combfold_out_base / f"{complex_name}_output"
            assembled = output_dir / "assembled_results"
            if assembled.exists():
                pdbs = sorted(assembled.glob("output_clustered_*.pdb"))
                n_outputs = len(pdbs)
                success = n_outputs > 0
            else:
                n_outputs = 0
                success = False
            scores = _parse_confidence_txt(assembled / "confidence.txt")
            confidence_str = ";".join(
                f"{k}:{v:.4f}" for k, v in sorted(scores.items())
            ) if scores else ""
            c = {
                "success": success,
                "n_outputs": n_outputs,
                "confidence_str": confidence_str,
            }
            cache[complex_name] = c
            print(f"[analyze]   row {idx}: {complex_name} -> "
                  f"success={c['success']}, n={c['n_outputs']}")

        col_success.append(c["success"])
        col_n_outputs.append(c["n_outputs"])
        col_confidence.append(c["confidence_str"])

    df["combfold_successfully"] = col_success
    df["n_assembled_outputs"] = col_n_outputs
    df["confidence_scores"] = col_confidence

    df.to_csv(paths.final_csv, index=False)
    print(f"[analyze] Wrote {len(df)} rows -> {paths.final_csv}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--stoic-sh", type=Path, default=None,
                    help="Path to 03_run_stoic.sbatch (required for submit-stoic/all)")
    ap.add_argument("--combfold-sh", type=Path, default=None,
                    help="Path to 05_run_CombFold.sbatch (required for submit-combfold/all)")
    ap.add_argument("--analyze-sh", type=Path, default=None,
                    help="Path to 11_analyze.sbatch (required for all/post-stoic-chain)")
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
        stage_submit_stoic(paths)

    elif args.mode == "aggregate-stoic":
        stage_aggregate_stoic(paths)

    elif args.mode == "expand":
        stage_expand(paths)

    elif args.mode == "submit-combfold":
        if not paths.expanded_csv.exists():
            sys.exit("[submit-combfold] expanded.csv missing; run --mode expand first.")
        ids = stage_submit_combfold(paths, dry_run=args.dry_run)
        if args.analyze_sh and not args.dry_run:
            stage_submit_analyze_dependency(paths, ids)

    elif args.mode == "analyze":
        stage_analyze(paths)

    elif args.mode == "post-stoic-chain":
        # Internal: this is what the dependency sbatch runs after Stoic finishes.
        stage_aggregate_stoic(paths)
        stage_expand(paths)
        ids = stage_submit_combfold(paths)
        if args.analyze_sh:
            stage_submit_analyze_dependency(paths, ids)

    elif args.mode == "all":
        # Full end-to-end via sbatch dependencies (no blocking polling).
        stage_fasta(paths)
        stage_submit_stoic(paths)
        reg = load_registry(paths)
        stoic_jid = reg.get("stoic_job_id")
        if not stoic_jid:
            sys.exit("[all] No stoic_job_id in registry; cannot chain.")
        stage_submit_post_stoic_chain(paths, stoic_jid)
        print(f"\n[all] Submitted Stoic job {stoic_jid} and dependent chain. "
              f"Monitor with squeue; final CSV will appear at:\n  {paths.final_csv}")


if __name__ == "__main__":
    main()
