#!/usr/bin/env python3
"""
Calculate true sequence identity for a single protein results folder.

This script processes one protein folder (e.g., output_dir/protein_id/) and:
1. Extracts input sequences from FASTA files
2. Extracts hit sequences from the PDB database
3. Performs pairwise global alignment
4. Calculates actual % identity
5. Writes results to true_identity.csv in the same folder
"""

import csv
import logging
from pathlib import Path
import argparse
import sys
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def setup_logging(protein_folder: Path) -> logging.Logger:
    """Setup logging to both console and a log file in the protein folder."""
    logger = logging.getLogger('true_identity')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler - writes to protein_folder/true_identity.log
    log_file = protein_folder / "true_identity.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_pdb_sequences(pdb_seqres_path: Path, logger: logging.Logger) -> dict:
    """Load PDB sequences from pdb_seqres.txt into a dictionary."""
    pdb_sequences = {}
    
    try:
        with open(pdb_seqres_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                pdb_sequences[record.id] = str(record.seq)
        logger.info(f"✅ Loaded {len(pdb_sequences)} PDB sequences")
    except Exception as e:
        logger.error(f"❌ Failed to load PDB sequences: {e}")
        raise
    
    return pdb_sequences


def extract_query_sequence_from_fasta(fasta_file: Path, logger: logging.Logger = None) -> str:
    """Extract the original input sequence from the FASTA file."""
    query_seq = ""
    
    try:
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                query_seq = str(record.seq)
                break  # Only need the first sequence
    except FileNotFoundError:
        msg = f"⚠️ FASTA file not found: {fasta_file}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
    except Exception as e:
        msg = f"⚠️ Error reading {fasta_file}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
    
    return query_seq


def calculate_identity(seq1: str, seq2: str) -> float:
    """
    Calculate % identity between two sequences using global alignment.
    Uses Bio.Align.PairwiseAligner (faster than deprecated pairwise2).
    
    Returns: percentage of matching residues relative to query length
    """
    if not seq1 or not seq2:
        return 0.0
    
    # Use PairwiseAligner (much faster than pairwise2)
    # aligner = Align.PairwiseAligner()
    # aligner.mode = 'global'
    # aligner.match_score = 1
    # aligner.mismatch_score = 0
    # aligner.open_gap_score = -0.5
    # aligner.extend_gap_score = -0.1
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'                          # local, not global
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")    
    aligner.open_gap_score = -11                    # strict, not -0.5
    aligner.extend_gap_score = -1                   # strict, not -0.1
    # Get the best alignment
    alignments = aligner.align(seq1, seq2)
    
    try:
        best_alignment = next(iter(alignments))
    except StopIteration:
        return 0.0
    
    # Count matches
    #! The following section is hard-coded binary, in either case, use best_alignment.score directly 
    # aligned_seq1, aligned_seq2 = best_alignment
    # matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) 
    #               if a == b and a != '-' and b != '-')
    # # Calculate identity as matches / length of query sequence
    # identity = (matches / len(seq1)) * 100
    
    return best_alignment.score


def parse_hmmer_hits(hmmer_output_file: Path, max_hits: int = 10000000) -> list[str]:
    """
    Extract hit PDB IDs from HMMER output file.
    Returns list of PDB IDs found in the file.
    """
    hits = []
    
    try:
        with open(hmmer_output_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('>> '):
                    pdb_id = line[3:].strip().split()[0]
                    hits.append(pdb_id)
                    if len(hits) >= max_hits:
                        break
    except FileNotFoundError:
        pass
    
    return hits


def process_single_folder(
    protein_folder: Path,
    pdb_sequences: dict,
    logger: logging.Logger
):
    """
    Process a single protein results folder.
    Writes true_identity.csv to the folder.
    """
    protein_id = protein_folder.name
    error_count = 0
    
    # Find all hmmsearch output files in this folder
    hmmer_files = sorted(protein_folder.glob('*_templates.txt'))
    
    if not hmmer_files:
        logger.warning(f"No HMMER output files found in {protein_folder}")
        return
    
    logger.info(f"📁 Found {len(hmmer_files)} HMMER output files in {protein_id}")
    
    # Collect all jobs
    jobs = []
    for hmmer_file in hmmer_files:
        # Parse filename to get chain_id
        file_name = hmmer_file.stem
        parts = file_name.rsplit('_', 2)
        
        if len(parts) < 2:
            logger.debug(f"Skipping file with unexpected name format: {file_name}")
            continue
        
        chain_id = parts[1]
        
        # Extract query sequence from the FASTA file
        fasta_file = hmmer_file.with_suffix('.fasta')
        query_seq = extract_query_sequence_from_fasta(fasta_file, logger)
        
        if not query_seq:
            logger.warning(f"No query sequence found for {hmmer_file.name}")
            continue
        
        # Get all hits
        hit_pdb_ids = parse_hmmer_hits(hmmer_file)
        
        # Queue jobs for processing
        for hit_pdb_id in hit_pdb_ids:
            if hit_pdb_id in pdb_sequences:
                jobs.append((chain_id, hit_pdb_id, query_seq, pdb_sequences[hit_pdb_id]))
    
    if not jobs:
        logger.warning(f"No valid alignments to process for {protein_id}")
        return
    
    logger.info(f"📋 Processing {len(jobs)} alignments for {protein_id}...")
    
    # Results storage: (chain_id, hit_pdb_id) -> identity
    results = {}
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=min(8, multiprocessing.cpu_count())) as executor:
        futures = {
            executor.submit(calculate_identity, job[2], job[3]): job 
            for job in jobs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Alignments ({protein_id})"):
            job = futures[future]
            chain_id, hit_pdb_id = job[0], job[1]
            
            try:
                true_identity = future.result()
                results[(chain_id, hit_pdb_id)] = true_identity
            except Exception as e:
                error_count += 1
                logger.error(f"Error calculating identity for {hit_pdb_id}: {e}")
    
    # Write results to CSV in the protein folder
    output_csv = protein_folder / "true_identity_standard_mmseq2_scoring.csv"
    
    try:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['chain_id', 'hit_pdb_id', 'true_identity_percent', 'is_high_homology_30'])
            
            for (chain_id, hit_pdb_id), identity in sorted(results.items()):
                writer.writerow([
                    chain_id,
                    hit_pdb_id,
                    f"{identity:.2f}",
                    "YES" if identity > 30.0 else "NO"
                ])
        
        logger.info(f"✅ Wrote {len(results)} results to {output_csv}")
    except Exception as e:
        error_count += 1
        logger.error(f"Failed to write results CSV: {e}")
    
    # Log summary
    if error_count > 0:
        logger.warning(f"Completed with {error_count} errors")
    else:
        logger.info(f"Completed successfully with no errors")


def main(
    protein_folder: Path,
    pdb_seqres_path: Path
):
    """
    Main function to calculate true identities for a single protein folder.
    """
    
    if not protein_folder.is_dir():
        print(f"❌ Error: {protein_folder} is not a directory")
        return 1
    
    # Setup logging
    logger = setup_logging(protein_folder)
    logger.info(f"Starting true identity calculation for {protein_folder.name}")
    
    # Check if already processed
    output_csv = protein_folder / "true_identity.csv"
    if output_csv.exists():
        logger.info(f"⏭️ Skipping {protein_folder.name} - true_identity.csv already exists")
        return 0
    
    try:
        # Load PDB sequences
        logger.info(f"📂 Loading PDB sequences from {pdb_seqres_path.name}...")
        pdb_sequences = load_pdb_sequences(pdb_seqres_path, logger)
        
        # Process the folder
        process_single_folder(protein_folder, pdb_sequences, logger)
        
        logger.info(f"✨ Done processing {protein_folder.name}!")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error processing {protein_folder.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate true sequence identity for a single protein folder"
    )
    parser.add_argument(
        "--protein_folder",
        type=Path,
        required=True,
        help="Path to a single protein results folder (e.g., output_dir/protein_id/)"
    )
    parser.add_argument(
        "--pdb_seqres",
        type=Path,
        required=True,
        help="Path to pdb_seqres.txt file"
    )
    
    args = parser.parse_args()
    
    exit_code = main(
        protein_folder=args.protein_folder,
        pdb_seqres_path=args.pdb_seqres
    )
    sys.exit(exit_code)
