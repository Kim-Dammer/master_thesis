#!/usr/bin/env python
"""
run_csv_combfold.py — Submit CombFold runs from a CSV, then parse results
and enrich the CSV with assembly outcome, confidence, and per-pair ipTM.

Usage (two-pass, recommended):
    # Pass 1: submit all jobs
    python run_csv_combfold.py --csv all_pdb_present_first_setup_pipeline_complexes.csv \
        --sh /cluster/project/beltrao/kdammer/master_thesis/scripts/05_05_run_CombFold.sbatch \
        --mode submit

    # Pass 2: after all SLURM jobs finish, parse results
    python run_csv_combfold.py --csv all_pdb_present_first_setup_pipeline_complexes.csv \
        --mode analyze

Usage (one-pass, blocks until all jobs finish):
    python run_csv_combfold.py --csv all_pdb_present_first_setup_pipeline_complexes.csv \
        --sh /cluster/project/beltrao/kdammer/master_thesis/scripts/05_run_combfold.sbatch \
        --mode all

What gets added to the CSV (one column per):
    combfold_successfully   True / False  (any output_clustered_*.pdb in assembled_results/)
    n_assembled_outputs     int  (number of output_clustered_*.pdb files)
    confidence_scores       str  (semicolon-separated: "0:81.89;1:81.46" — cluster_idx:score)
    iptm_<PAIR>            float per pair  (e.g. iptm_P00937_P00899, iptm_P00937_P00937)
                              heterodimer ipTM from pp.pairs.chain_pair_iptm_mean_corrected
                              homodimer ipTM from AFDB HEADER line (AF-XXXXX) via API lookup
                              NaN if not found

Assumptions:
    - The CSV has a column called 'comb_fold_submission' with specs like
      'P00937(1),P00899(2)'.
    - The .env file expected by 05_run_CombFold.sbatch is in the CWD.
    - The venv at VENV has pandas, pooled_ppi, af3io installed.
    - CombFold output layout:
        <OUTPUT_BASE>/<complex_name>_output/assembled_results/output_clustered_N.pdb
        <OUTPUT_BASE>/<complex_name>_output/assembled_results/confidence.txt

Note on duplicate specs:
    The CSV may contain duplicate comb_fold_submission values (e.g. the same
    spec appearing on multiple rows). The submit phase deduplicates: each
    unique spec is submitted exactly once. All CSV rows sharing a spec get
    the same slurm_job_id in the registry. The analyze phase reads the same
    output directory for every row that shares a spec, so all duplicate rows
    get the same result columns.
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

# ---------------------------------------------------------------------------
# Config — must match 05_run_CombFold.sbatch / .env
# ---------------------------------------------------------------------------
VENV = "/cluster/project/beltrao/kdammer/master_thesis/.venv"
OUTPUT_BASE = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/first_setup/CombFold"
MERGED_PDBS_DIR = "/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/merged_pdbs"
POOLED_PPI_DB = "/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.03"

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


def spec_to_pairs(spec: str) -> list[tuple[str, str]]:
    """Return all unique unordered pairs (including self-pairs) from spec.

    Self-pairs are included for proteins with stoich >= 2 (homodimers).
    """
    entry_re = re.compile(r"^([A-Za-z0-9_]+)\((\d+)\)$")
    counts: dict[str, int] = {}
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        m = entry_re.match(tok)
        if m:
            counts[m.group(1)] = int(m.group(2))
    proteins = sorted(counts)
    pairs: list[tuple[str, str]] = []
    for i, p1 in enumerate(proteins):
        for p2 in proteins[i:]:
            if p1 == p2:
                if counts[p1] >= 2:
                    pairs.append((p1, p2))
            else:
                pairs.append(tuple(sorted([p1, p2])))  # type: ignore[arg-type]
    return pairs


# ---------------------------------------------------------------------------
# Submit mode
# ---------------------------------------------------------------------------

def submit(df: pd.DataFrame, csv_path: Path, sh_path: Path, dry_run: bool = False) -> None:
    """Submit one sbatch per unique spec, record job IDs in a sidecar file.

    Duplicate specs in the CSV are submitted only once; all rows sharing
    a spec get the same slurm_job_id in the registry.
    """
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
            print(f"[DRY-RUN] sbatch {sh_path} '{spec}'  -> {complex_name}  "
                  f"(rows {csv_rows})")
            spec_job_map[spec] = None
            continue

        cmd = ["sbatch", str(sh_path), spec]
        print(f"Submitting: {' '.join(cmd)}  (rows {csv_rows})")
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
            print(f"Monitor with:  squeue -j {','.join(str(j) for j in unique_job_ids)}")


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


def lookup_heterodimer_iptm(pairs_list: list[tuple[str, str]]) -> dict[str, float | None]:
    """Look up ipTM for heterodimer pairs from pp.pairs.

    Returns dict keyed by "X__Y" (sorted) -> ipTM value or None.
    """
    import pooled_ppi

    pp = pooled_ppi.PooledPredictionsDb(POOLED_PPI_DB)

    # Determine which ipTM column is available
    iptm_col = None
    for candidate in [
        "chain_pair_iptm_mean_corrected",
        "chain_pair_iptm",
        "iptm",
        "ipTM",
        "average_iptm",
    ]:
        if candidate in pp.pairs.columns:
            iptm_col = candidate
            break

    if iptm_col is not None:
        print(f"  Using ipTM column: {iptm_col}")
    else:
        print(f"  [WARN] No ipTM column found in pp.pairs. Columns: {list(pp.pairs.columns)}")

    result: dict[str, float | None] = {}
    for p1, p2 in pairs_list:
        key = f"{p1}__{p2}"
        if p1 == p2:
            result[key] = None
            continue
        rows = pp.pairs[
            ((pp.pairs.uniprot_id1 == p1) & (pp.pairs.uniprot_id2 == p2))
            | ((pp.pairs.uniprot_id1 == p2) & (pp.pairs.uniprot_id2 == p1))
        ]
        if len(rows) == 0 or iptm_col is None:
            result[key] = None
        else:
            try:
                result[key] = float(rows.iloc[0][iptm_col])
            except (ValueError, TypeError):
                result[key] = None
    return result


def lookup_homodimer_iptm_afdb(proteins: list[str]) -> dict[str, float | None]:
    """Try to extract ipTM for homodimers from AFDB PDB HEADER lines.

    AFDB PDBs have a HEADER line like:
        HEADER AF-0000000066218867-MODEL_V1
    The AF-ID can be used to look up the entry at
    https://alphafold.ebi.ac.uk/api/prediction/<AF-ID>
    which returns JSON with 'iptm' field.
    """
    result: dict[str, float | None] = {}
    merged = Path(MERGED_PDBS_DIR)

    for prot in proteins:
        pdb_path = merged / f"{prot}.pdb"
        if not pdb_path.exists():
            result[prot] = None
            continue

        # Extract AF-ID from HEADER line
        af_id = None
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith("HEADER"):
                    m = re.search(r"(AF-[A-Z0-9]+-MODEL_V\d+)", line)
                    if m:
                        af_id = m.group(1)
                    break  # only check first HEADER

        if af_id is None:
            result[prot] = None
            continue

        # Try the EBI API for ipTM
        try:
            import urllib.request
            url = f"https://alphafold.ebi.ac.uk/api/prediction/{af_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "run_csv_combfold/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            if isinstance(data, list) and len(data):
                entry = data[0]
            elif isinstance(data, dict):
                entry = data
            else:
                entry = {}
            iptm_val = entry.get("iptm")
            result[prot] = float(iptm_val) if iptm_val is not None else None
        except Exception as e:
            print(f"  [WARN] Could not fetch ipTM for {prot} ({af_id}): {e}")
            result[prot] = None

    return result


def analyze(df: pd.DataFrame, csv_path: Path) -> None:
    """Parse CombFold outputs and enrich the CSV."""
    # Collect all unique pairs across all specs for batch ipTM lookup
    all_pairs: set[tuple[str, str]] = set()
    all_proteins: set[str] = set()
    for _, row in df.iterrows():
        spec = str(row["comb_fold_submission"])
        pairs = spec_to_pairs(spec)
        all_pairs.update(pairs)
        all_proteins.update(spec_to_proteins(spec))

    print(f"Looking up ipTM for {len(all_pairs)} unique pairs across {len(all_proteins)} proteins...")
    hetero_pairs = [p for p in all_pairs if p[0] != p[1]]
    homo_proteins = [p[0] for p in all_pairs if p[0] == p[1]]

    iptm_hetero = lookup_heterodimer_iptm(hetero_pairs) if hetero_pairs else {}
    iptm_homo = lookup_homodimer_iptm_afdb(homo_proteins) if homo_proteins else {}

    # Merge into one dict keyed by "X__Y"
    iptm_all: dict[str, float | None] = {}
    iptm_all.update(iptm_hetero)
    for prot, val in iptm_homo.items():
        iptm_all[f"{prot}__{prot}"] = val

    # Determine which pairs actually appear (for column naming)
    pair_keys_in_data: set[str] = set()
    for _, row in df.iterrows():
        spec = str(row["comb_fold_submission"])
        for p in spec_to_pairs(spec):
            pair_keys_in_data.add(f"{p[0]}__{p[1]}")

    # Prepare new columns
    col_success: list[bool] = []
    col_n_outputs: list[int] = []
    col_confidence: list[str] = []
    col_iptm: dict[str, list[float | None]] = {k: [] for k in sorted(pair_keys_in_data)}

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
            output_dir = Path(OUTPUT_BASE) / f"{complex_name}_output"
            assembled_dir = output_dir / "assembled_results"

            # --- Assembly success ---
            if assembled_dir.exists():
                output_pdbs = sorted(assembled_dir.glob("output_clustered_*.pdb"))
                n_outputs = len(output_pdbs)
                success = n_outputs > 0
            else:
                output_pdbs = []
                n_outputs = 0
                success = False

            # --- Confidence scores ---
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

        # --- ipTM per pair ---
        pairs = spec_to_pairs(spec)
        for key in sorted(pair_keys_in_data):
            pair_tuples = [(p[0], p[1]) for p in pairs]
            key_tuple = tuple(key.split("__"))
            if key_tuple in pair_tuples or (key_tuple[1], key_tuple[0]) in pair_tuples:
                col_iptm[key].append(iptm_all.get(key))
            else:
                col_iptm[key].append(None)

    # Add columns to DataFrame
    df["combfold_successfully"] = col_success
    df["n_assembled_outputs"] = col_n_outputs
    df["confidence_scores"] = col_confidence

    for key in sorted(pair_keys_in_data):
        col_name = f"iptm_{key.replace('__', '_')}"
        df[col_name] = col_iptm[key]

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
    ap.add_argument("--mode", choices=["submit", "analyze", "all"], default="all",
                    help="submit: sbatch all rows | analyze: parse results | all: submit+wait+analyze")
    ap.add_argument("--dry-run", action="store_true",
                    help="(submit mode) print commands without running")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="Seconds between squeue polls (all mode)")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")

    if not args.sh.exists():
        sys.exit(f"05_run_CombFold.sbatch not found: {args.sh}")

    df = pd.read_csv(args.csv)
    if "comb_fold_submission" not in df.columns:
        sys.exit(f"Column 'comb_fold_submission' not found in {args.csv}. "
                 f"Available: {list(df.columns)}")

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Using shell script: {args.sh}")

    if args.mode in ("submit", "all"):
        submit(df, args.csv, sh_path=args.sh, dry_run=args.dry_run)

    if args.mode == "all":
        sidecar_path = args.csv.with_name(args.csv.stem + "_job_registry.json")
        if sidecar_path.exists():
            with open(sidecar_path) as fh:
                registry = json.load(fh)
            if not args.dry_run:
                wait_for_jobs(registry, poll_interval=args.poll_interval)

    if args.mode in ("analyze", "all"):
        analyze(df, args.csv)


if __name__ == "__main__":
    main()
