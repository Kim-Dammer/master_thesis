#!/usr/bin/env python
"""

14_run_csv_combfold.py — Submit CombFold runs from a CSV, then parse results
and enrich the CSV with assembly outcome and confidence.

Usage (one-pass, blocks until all jobs finish):
 python 14_run_csv_combfold.py \
     --csv pdb_present_for_stoi_gr_two_third_setup_pipeline_complexes.csv \
     --sh scripts/05_run_CombFold.sbatch \
     --output-base data/Pipeline/third_setup/CombFold \
     --mode all

What gets added to the CSV (one column per):
 combfold_successfully True / False (any output_clustered_*.pdb in assembled_results/)
 n_assembled_outputs int (number of output_clustered_*.pdb files)
 confidence_scores str (semicolon-separated: "0:81.89;1:81.46" — cluster_idx:score)

Assumptions:
 - The CSV has a column called 'comb_fold_submission' with specs like
   'P00937(1),P00899(2)'.
 - CombFold output layout:
     <OUTPUT_BASE>/<complex_name>_output/assembled_results/output_clustered_N.pdb
     <OUTPUT_BASE>/<complex_name>_output/assembled_results/confidence.txt

"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VENV = "/cluster/project/beltrao/kdammer/master_thesis/.venv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spec_to_complex_name(spec: str) -> str:
    """Derive the CombFold complex name from a spec string.

    Mirrors the bash-side logic in 05_run_CombFold.sbatch:
    sorted UniProt IDs, joined as '{id}x{count}'.
    """
    entry_re = re.compile(r"^([A-Za-z0-9_]+)\((\d+)\)$")
    counts: dict[str, int] = {}
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        m = entry_re.match(tok)
        if m:
            counts[m.group(1)] = int(m.group(2))
    return "_".join(f"{p}x{counts[p]}" for p in sorted(counts))


def spec_to_proteins(spec: str) -> list[str]:
    """Extract sorted list of UniProt IDs from spec."""
    entry_re = re.compile(r"^([A-Za-z0-9_]+)\((\d+)\)$")
    proteins: list[str] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        m = entry_re.match(tok)
        if m:
            proteins.append(m.group(1))
    return sorted(set(proteins))


def patch_combfold_sbatch(sh_path: Path, output_base: Path) -> Path:
    """Patch 05_run_CombFold.sbatch so OUTPUT_BASE and log paths point at
    the requested directory.

    Returns the path to the patched copy.
    """
    text = sh_path.read_text()
    logs_dir = output_base.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Rewrite OUTPUT_BASE
    text = re.sub(
        r'^OUTPUT_BASE=.*$',
        f'OUTPUT_BASE="{output_base}"',
        text, count=1, flags=re.MULTILINE,
    )
    # Rewrite SBATCH log paths to absolute
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

    patched_path = output_base.parent / f"_patched_{sh_path.name}"
    patched_path.write_text(text)
    patched_path.chmod(0o755)

    # Verify the substitution took effect
    if str(output_base) not in patched_path.read_text():
        sys.exit(f"[patch] OUTPUT_BASE substitution failed in {sh_path}. "
                 f"Expected an 'OUTPUT_BASE=...' line.")
    return patched_path


# ---------------------------------------------------------------------------
# Submit mode
# ---------------------------------------------------------------------------

def submit(df: pd.DataFrame, csv_path: Path, sh_path: Path,
           output_base: Path, dry_run: bool = False) -> None:
    """Submit one sbatch per unique spec, record job IDs in a sidecar file.

    Duplicate specs in the CSV are submitted only once; all rows sharing
    a spec get the same slurm_job_id in the registry.
    """
    # --- Patch the sbatch ----------------------------------------------------
    output_base.mkdir(parents=True, exist_ok=True)
    patched_sh = patch_combfold_sbatch(sh_path, output_base)
    print(f"Using patched sbatch: {patched_sh}")

    # --- Deduplicate specs ---------------------------------------------------
    spec_to_rows: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        spec = str(row["comb_fold_submission"])
        spec_to_rows.setdefault(spec, []).append(int(idx))

    unique_specs = list(spec_to_rows.keys())
    print(f"CSV has {len(df)} rows, {len(unique_specs)} unique specs")

    # --- Submit each unique spec once ----------------------------------------
    spec_job_map: dict[str, int | None] = {}

    for spec in unique_specs:
        complex_name = spec_to_complex_name(spec)
        csv_rows = spec_to_rows[spec]

        if dry_run:
            print(f"[DRY-RUN] sbatch {patched_sh} '{spec}' -> {complex_name} "
                  f"(rows {csv_rows})")
            spec_job_map[spec] = None
            continue

        cmd = ["sbatch", str(patched_sh), spec]
        print(f"Submitting: {' '.join(cmd)} (rows {csv_rows})")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr.strip()}")
            spec_job_map[spec] = None
            continue

        m = re.search(r"Submitted batch job (\d+)", result.stdout)
        job_id = int(m.group(1)) if m else None
        print(f"  -> job {job_id}, complex {complex_name}")
        spec_job_map[spec] = job_id

    # --- Build per-row registry ----------------------------------------------
    registry: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        spec = str(row["comb_fold_submission"])
        complex_name = spec_to_complex_name(spec)
        entry: dict[str, Any] = {
            "csv_row": int(idx),
            "spec": spec,
            "complex_name": complex_name,
            "slurm_job_id": spec_job_map.get(spec),
        }
        if spec_job_map.get(spec) is None and not dry_run:
            entry["submit_error"] = "sbatch failed or no job ID parsed"
        registry.append(entry)

    # Write sidecar
    sidecar_path = csv_path.with_name(csv_path.stem + "_job_registry.json")
    with open(sidecar_path, "w") as fh:
        json.dump(registry, fh, indent=2)
    print(f"\nJob registry written to: {sidecar_path}")
    if not dry_run:
        unique_job_ids = list({r["slurm_job_id"] for r in registry if r["slurm_job_id"] is not None})
        if unique_job_ids:
            print(f"Monitor with: squeue -j {','.join(str(j) for j in unique_job_ids)}")


# ---------------------------------------------------------------------------
# Wait mode (used by --mode all)
# ---------------------------------------------------------------------------

def wait_for_jobs(registry: list[dict[str, Any]], poll_interval: int = 60) -> None:
    """Poll squeue until all submitted jobs are gone."""
    unique_job_ids = list({str(r["slurm_job_id"]) for r in registry if r["slurm_job_id"] is not None})
    if not unique_job_ids:
        return
    print(f"Waiting for {len(unique_job_ids)} unique job(s) to finish (polling every {poll_interval}s)...")
    while True:
        result = subprocess.run(
            ["squeue", "-j", ",".join(unique_job_ids), "-h", "-o", "%i"],
            capture_output=True, text=True,
        )
        running = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if not running:
            print("All jobs finished.")
            break
        print(f"  {len(running)} job(s) still running: {', '.join(running[:5])}{'...' if len(running)>5 else ''}")
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Analyze mode
# ---------------------------------------------------------------------------

def parse_confidence_txt(conf_path: Path) -> dict[int, float]:
    """Parse confidence.txt -> {cluster_idx: score}.

    Format (one line per cluster):
    /full/path/output_clustered_0.pdb 81.8879
    /full/path/output_clustered_1.pdb 81.4568
    """
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


def analyze(df: pd.DataFrame, csv_path: Path, output_base: Path) -> None:
    """Parse CombFold outputs and enrich the CSV (no iptm columns)."""
    # Prepare new columns
    col_success: list[bool] = []
    col_n_outputs: list[int] = []
    col_confidence: list[str] = []

    # Cache parsed results per complex_name (avoids re-reading same dir for duplicate specs)
    parsed_cache: dict[str, dict[str, Any]] = {}

    for idx, row in df.iterrows():
        spec = str(row["comb_fold_submission"])
        complex_name = spec_to_complex_name(spec)

        # Use cache if we already parsed this complex_name
        if complex_name in parsed_cache:
            cached = parsed_cache[complex_name]
            col_success.append(cached["success"])
            col_n_outputs.append(cached["n_outputs"])
            col_confidence.append(cached["confidence_str"])
        else:
            output_dir = output_base / f"{complex_name}_output"
            assembled_dir = output_dir / "assembled_results"

            # --- Assembly success ---
            # Skip if complex_name exceeds filesystem limit (255 bytes per component)
            # CombFold would have failed to create the directory anyway.
            name_too_long = len(complex_name) > 255
            if name_too_long:
                print(f"  row {idx}: {complex_name[:60]}... -> SKIPPED (name too long: {len(complex_name)} chars)")
                output_pdbs = []
                n_outputs = 0
                success = False
            else:
                try:
                    dir_exists = assembled_dir.exists()
                except OSError:
                    dir_exists = False
                if dir_exists:
                    output_pdbs = sorted(assembled_dir.glob("output_clustered_*.pdb"))
                    n_outputs = len(output_pdbs)
                    success = n_outputs > 0
                else:
                    output_pdbs = []
                    n_outputs = 0
                    success = False

            # --- Confidence scores ---
            if name_too_long:
                scores = {}
            else:
                conf_path = assembled_dir / "confidence.txt"
                scores = parse_confidence_txt(conf_path)
            if scores:
                confidence_str = ";".join(f"{k}:{v:.4f}" for k, v in sorted(scores.items()))
            else:
                confidence_str = ""

            col_success.append(success)
            col_n_outputs.append(n_outputs)
            col_confidence.append(confidence_str)

            parsed_cache[complex_name] = {
                "success": success,
                "n_outputs": n_outputs,
                "confidence_str": confidence_str,
            }

            print(f"  row {idx}: {complex_name} -> success={success}, n_outputs={n_outputs}, "
                  f"confidence={scores if scores else 'N/A'}")

    # Add columns to DataFrame
    df["combfold_successfully"] = col_success
    df["n_assembled_outputs"] = col_n_outputs
    df["confidence_scores"] = col_confidence

    # Write enriched CSV
    out_path = csv_path.with_name(csv_path.stem + "_combfold_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nEnriched CSV written to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path,
                    help="Input CSV with comb_fold_submission column")
    ap.add_argument("--sh", type=Path,
                    default=Path(__file__).resolve().parent / "05_run_CombFold.sbatch",
                    help="Path to 05_run_CombFold.sbatch (default: same dir as this script)")
    ap.add_argument("--output-base", type=Path,
                    default=Path("/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/first_setup/CombFold"),
                    help="CombFold output base directory (default: first_setup/CombFold)")
    ap.add_argument("--mode", choices=["submit", "analyze", "all"], default="all",
                    help="submit: sbatch all rows | analyze: parse results | all: submit+wait+analyze")
    ap.add_argument("--dry-run", action="store_true",
                    help="(submit mode) print commands without running")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="Seconds between squeue polls (all mode)")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")

    output_base = args.output_base.resolve()

    df = pd.read_csv(args.csv)
    if "comb_fold_submission" not in df.columns:
        sys.exit(f"Column 'comb_fold_submission' not found in {args.csv}. "
                 f"Available: {list(df.columns)}")

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Output base: {output_base}")

    if args.mode in ("submit", "all"):
        if not args.sh.exists():
            sys.exit(f"05_run_CombFold.sbatch not found: {args.sh}")
        submit(df, args.csv, sh_path=args.sh, output_base=output_base,
               dry_run=args.dry_run)

    if args.mode == "all":
        sidecar_path = args.csv.with_name(args.csv.stem + "_job_registry.json")
        if sidecar_path.exists():
            with open(sidecar_path) as fh:
                registry = json.load(fh)
            if not args.dry_run:
                wait_for_jobs(registry, poll_interval=args.poll_interval)

    if args.mode in ("analyze", "all"):
        analyze(df, args.csv, output_base=output_base)


if __name__ == "__main__":
    main()
