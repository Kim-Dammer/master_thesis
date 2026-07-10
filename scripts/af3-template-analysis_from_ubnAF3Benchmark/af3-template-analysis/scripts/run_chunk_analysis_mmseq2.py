#!/usr/bin/env python3
"""
MMseqs2 replacement for run_chunk_analysis.py.

Does the same job as the original (find PDB homologs for input protein sequences)
but uses MMseqs2 with iterative profiling instead of jackhmmer -> hmmbuild -> hmmsearch.

CRITICAL: Produces the EXACT SAME output files that run_single_true_identity.py expects:
  - <protein_id>_<chain_id>_templates.txt  (hit list, with ">> " lines for PDB IDs)
  - <protein_id>_<chain_id>_templates.fasta (query sequence in FASTA format)
  - homology_check.csv (summary CSV, same format as original)

This means submit_true_identity.sh runs UNCHANGED after this script.

Usage (same CLI interface as the original):
    python3 run_chunk_analysis_mmseqs2.py \
        --output_dir chunk_analysis_results/uniqSeqs \
        --job_id 1 \
        --input_files P26449_uniqSeq.json P53157_uniqSeq.json

Requires mmseqs2 to be on PATH (module load mmseqs2 or conda install).
"""

import argparse
import csv
import json
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
PDB_SEQRES_PATH = Path("/cluster/project/alphafold/pdb_seqres/pdb_seqres.txt")
N_CPU = 8
SENSITIVITY = 7.5
EVALUE = 100
NUM_ITERATIONS = 3  # iterative profiling (approximates jackhmmer)


def run_mmseqs_search(sequence: str, query_name: str, output_prefix: Path):
    """
    Search a single query sequence against pdb_seqres using MMseqs2.

    Produces:
      - <output_prefix>.fasta  (query sequence)
      - <output_prefix>.txt    (hits in HMMER-like ">> " format for compatibility)
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Write query FASTA (same as original script)
    query_fasta_path = output_prefix.with_suffix(".fasta")
    with open(query_fasta_path, "w") as f:
        f.write(f">query\n{sequence}\n")

    # Use a temporary directory for MMseqs2 databases
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        query_db = tmp / "query_db"
        pdb_db = tmp / "pdb_db"
        result_db = tmp / "result_db"
        mmseqs_tmp = tmp / "mmseqs_tmp"
        hits_tsv = tmp / "hits.tsv"

        # Create query database (single sequence)
        subprocess.run(
            ["mmseqs", "createdb", str(query_fasta_path), str(query_db)],
            check=True, capture_output=True, text=True,
        )

        # Create PDB target database
        # Check if a pre-built database exists to avoid rebuilding every time
        pdb_db_cached = Path("/tmp/mmseqs_pdb_db")
        if pdb_db_cached.exists() and (pdb_db_cached / "query_db.idx").exists():
            pdb_db = pdb_db_cached
        else:
            subprocess.run(
                ["mmseqs", "createdb", str(PDB_SEQRES_PATH), str(pdb_db)],
                check=True, capture_output=True, text=True,
            )
            # Cache it for subsequent calls in the same job
            try:
                shutil.copytree(pdb_db, pdb_db_cached)
            except Exception:
                pass  # caching is optional

        # Run MMseqs2 search with iterative profiling
        subprocess.run(
            [
                "mmseqs", "search",
                str(query_db), str(pdb_db), str(result_db), str(mmseqs_tmp),
                "--threads", str(N_CPU),
                "-s", str(SENSITIVITY),
                "-e", str(EVALUE),
                "--num-iterations", str(NUM_ITERATIONS),
            ],
            check=True, capture_output=True, text=True,
        )

        # Convert results to TSV
        subprocess.run(
            [
                "mmseqs", "convertalis",
                str(query_db), str(pdb_db), str(result_db), str(hits_tsv),
                "--format-output", "target,pident,alnlen,evalue,qcov",
            ],
            check=True, capture_output=True, text=True,
        )

        # Convert MMseqs2 TSV to HMMER-like format for compatibility with
        # run_single_true_identity.py (which parses ">> " lines)
        hmmsearch_output_path = output_prefix.with_suffix(".txt")
        with open(hmmsearch_output_path, "w") as f:
            # Write a minimal HMMER-like header (the true-identity script only
            # looks for ">> " lines, but we include a header for readability)
            f.write(f"# MMseqs2 search results (replaces jackhmmer/hmmsearch)\n")
            f.write(f"# Query: {query_name}\n")
            f.write(f"# Sensitivity: {SENSITIVITY}, E-value: {EVALUE}, Iterations: {NUM_ITERATIONS}\n")
            f.write(f"#\n")

            # Read the TSV and write hits in HMMER ">> " format
            if hits_tsv.exists() and hits_tsv.stat().st_size > 0:
                with open(hits_tsv, "r") as tsv_in:
                    for line in tsv_in:
                        parts = line.strip().split("\t")
                        if len(parts) >= 4:
                            target = parts[0]  # PDB chain ID, e.g. "1ycc_A"
                            pident = parts[1]
                            alnlen = parts[2]
                            evalue = parts[3]
                            # Write in HMMER-like format:
                            # >> 1ycc_A  (the true-identity script parses this line)
                            f.write(f">> {target}\n")
                            # Include identity/evalue as a comment for debugging
                            f.write(f"   # pident={pident}, alnlen={alnlen}, evalue={evalue}\n")

    return hmmsearch_output_path


def parse_mmseqs_hits(hits_file: Path):
    """
    Parse the HMMER-like output to extract hits with identity and e-value.
    (Same return format as the original parse_hmmsearch_output)
    """
    hits = []
    try:
        with open(hits_file, "r") as f:
            for line in f:
                if line.startswith(">> "):
                    target_name = line[3:].strip().split()[0]
                    hits.append({"name": target_name, "identity": 0.0, "evalue": 0.0})
                elif line.strip().startswith("# pident="):
                    # Parse the comment line we wrote
                    parts = line.strip().split(",")
                    try:
                        pident = float(parts[0].split("=")[1])
                        evalue = float(parts[2].split("=")[1])
                        if hits:
                            hits[-1]["identity"] = pident
                            hits[-1]["evalue"] = evalue
                    except (IndexError, ValueError):
                        pass
    except FileNotFoundError:
        return []
    return hits


def process_single_file(input_file: Path, output_dir: Path):
    """Process a single JSON input file (same interface as original)."""
    try:
        protein_id = input_file.stem.split("_")[0].lower()
        with open(input_file, "r") as f:
            data = json.load(f)

        for molecule_definition in data.get("sequences", []):
            if "protein" in molecule_definition:
                protein_info = molecule_definition["protein"]
                sequence = protein_info.get("sequence")
                chain_id = protein_info.get("id", "A")

                file_output_dir = output_dir / protein_id
                file_output_dir.mkdir(exist_ok=True, parents=True)
                output_prefix = file_output_dir / f"{protein_id}_{chain_id}_templates"

                alignment_file = output_prefix.with_suffix(".txt")

                # Skip if already done (same optimization as original)
                if not alignment_file.exists() or alignment_file.stat().st_size == 0:
                    run_mmseqs_search(sequence, protein_id, output_prefix)
                else:
                    logging.info(f"Skipping search for {protein_id}, parsing existing results.")

                # Parse results
                hits = parse_mmseqs_hits(alignment_file)

                # Sort by identity (highest first)
                hits.sort(key=lambda x: x["identity"], reverse=True)

                # Write CSV summary (same format as original)
                summary_csv_path = file_output_dir / "homology_check.csv"
                with open(summary_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["File", "Chain", "Identity", "E-value", "Template_Name"])
                    for hit in hits:
                        writer.writerow([
                            input_file.name,
                            chain_id,
                            f"{hit['identity']:.2f}",
                            f"{hit['evalue']:.2e}",
                            hit["name"],
                        ])

    except Exception as e:
        logging.error(f"Error processing {input_file}: {e}")


def main(input_files: list[Path], output_dir: Path, job_id: str):
    logging.basicConfig(level=logging.INFO)
    for input_file in tqdm(input_files):
        process_single_file(input_file, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--job_id", type=str, default="1")
    parser.add_argument("--input_files", type=str, required=True, nargs="+")
    args = parser.parse_args()
    main([Path(f) for f in args.input_files], Path(args.output_dir), args.job_id)
