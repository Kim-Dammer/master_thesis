import argparse
import csv
import json
import logging
import subprocess
from pathlib import Path
import re
import tempfile
from tqdm import tqdm

# --- CONFIGURATION (Euler Paths) ---
PDB_SEQRES_PATH = Path("/cluster/project/alphafold/pdb_seqres/pdb_seqres.txt") 
UNIREF90_PATH = Path('/cluster/project/alphafold/uniref90/uniref90.fasta')
N_CPU = 8

# --- 1. The "Loose" Search (Matches AF3 Pipeline) ---
def run_template_search(sequence: str, output_prefix: Path):
    output_prefix.parent.mkdir(exist_ok=True, parents=True)
    query_fasta_path = output_prefix.with_suffix(".fasta")
    
    with open(query_fasta_path, "w") as f:
        f.write(f">query\n{sequence}\n")
    
    hmm_path = output_prefix.with_suffix(".hmm")
    
    # Build HMM
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".sto") as msa_tmp:
        msa_sto_path = Path(msa_tmp.name)
        subprocess.run(['jackhmmer', '--cpu', str(N_CPU), '-N', '1', '-A', str(msa_sto_path), str(query_fasta_path), UNIREF90_PATH], check=True, capture_output=True, text=True)
        subprocess.run(['hmmbuild', '--cpu', str(N_CPU), str(hmm_path), str(msa_sto_path)], check=True, capture_output=True, text=True)
    
    hmmsearch_output_path = output_prefix.with_suffix(".txt")
    
    # AF3 "Loose" Flags
    cmd = [
        'hmmsearch', '--cpu', str(N_CPU),
        '--F1', '0.1', '--F2', '0.1', '--F3', '0.1',
        '-E', '100', '--domE', '100', '--incE', '100', '--incdomE', '100',
        str(hmm_path), PDB_SEQRES_PATH
    ]
    
    with open(hmmsearch_output_path, "w") as f:
        subprocess.run(cmd, check=True, text=True, stdout=f)
        
    return hmmsearch_output_path

# --- 2. The "Strict" Calculation (Matches DeepMind Eval Paper) ---
def parse_hmmsearch_output(hmmsearch_file: Path):
    hits = []
    try:
        with open(hmmsearch_file, 'r') as f:
            content = f.read()
            # Split into Hit blocks (each starting with >>)
            hit_blocks = content.split('>> ')[1:]
    except (FileNotFoundError, IndexError):
        return []

    # 1. robustly find Query Name and Length from the header
    # Matches: "Query:   query-i1  [M=614]"
    header_match = re.search(r"Query:\s+(\S+)\s+\[[LM]=(\d+)\]", content)
    if not header_match:
        logging.warning(f"Could not parse header in {hmmsearch_file.name}")
        return []
    
    query_name_ref = header_match.group(1)
    query_len = int(header_match.group(2))

    for block in hit_blocks:
        # The first line of the block has the Target Name
        # e.g. "6uv0_A  mol:protein..."
        lines = block.split('\n')
        target_name_ref = lines[0].strip().split()[0]
        
        # Split the hit block into specific Domain Alignment blocks
        # (The first chunk is summary data, we skip it by taking [1:])
        domain_blocks = block.split('== domain ')[1:]
        
        for d_block in domain_blocks:
            # Extract E-value from domain header line
            # Format: "... score: XX.X bits;  conditional E-value: 1.2e-10"
            evalue_match = re.search(r'conditional E-value:\s+(\S+)', d_block)
            evalue = float('inf')
            if evalue_match:
                try:
                    evalue = float(evalue_match.group(1))
                except ValueError:
                    evalue = float('inf')
            
            # Filter by E-value < 10
            if evalue >= 10:
                continue
            
            # We must accumulate sequences because HMMER wraps long lines
            q_seq_fragments = []
            t_seq_fragments = []
            
            for line in d_block.split('\n'):
                parts = line.strip().split()
                # A valid alignment line has at least: Name, Start, Seq, End (4 parts)
                if len(parts) < 4: continue
                
                # Check if this line belongs to the Query (Model)
                if parts[0] == query_name_ref:
                    # [-2] is the sequence column (Name Start Sequence End)
                    q_seq_fragments.append(parts[-2])
                    
                # Check if this line belongs to the Target
                elif parts[0] == target_name_ref:
                    t_seq_fragments.append(parts[-2])
            
            # Join fragments (handles wrapping)
            query_alignment = "".join(q_seq_fragments)
            target_alignment = "".join(t_seq_fragments)
            
            # Safety check: if parser failed to find lines
            if not query_alignment or not target_alignment:
                continue

            # --- METRICS CALCULATION ---
            
            # 1. Coverage (Residues in Query that are aligned)
            # We count non-gap characters in the Model/Query sequence
            aligned_query_residues = sum(1 for q in query_alignment if q not in '.-')
            align_ratio = aligned_query_residues / query_len

            # 2. Pipeline Filter: Is coverage > 10%?
            if align_ratio < 0.1: continue

            # 3. Identity (Evaluation Metric)
            # Count EXACT matches
            identities = sum(1 for q, t in zip(query_alignment, target_alignment) 
                             if q.upper() == t.upper() and q not in '.-' and t not in '.-')
            
            percent_identity = (identities / query_len) * 100
            
            # Append all hits with E < 10
            hits.append({'name': target_name_ref, 'identity': percent_identity, 'evalue': evalue})

    return hits

def process_single_file(input_file: Path, output_dir: Path):
    try:
        protein_id = input_file.stem.split('_')[0].lower()
        with open(input_file, 'r') as f:
            data = json.load(f)

        for molecule_definition in data.get('sequences', []):
            if 'protein' in molecule_definition:
                protein_info = molecule_definition['protein']
                sequence = protein_info.get('sequence')
                chain_id = protein_info.get('id', 'A')

                file_output_dir = output_dir / protein_id
                file_output_dir.mkdir(exist_ok=True, parents=True)
                output_prefix = file_output_dir / f"{protein_id}_{chain_id}_templates"

                alignment_file = output_prefix.with_suffix(".txt")
                
                # OPTIMIZATION: If the .txt file already exists and has content, 
                # SKIP the search and just re-parse.
                if not alignment_file.exists() or alignment_file.stat().st_size == 0:
                    run_template_search(sequence, output_prefix)
                else:
                    logging.info(f"Skipping search for {protein_id}, parsing existing results.")
                
                # Parse results
                hits = parse_hmmsearch_output(alignment_file)
                
                # Sort by Identity (Highest first)
                hits.sort(key=lambda x: x['identity'], reverse=True)
                
                # Write CSV Summary with all hits
                summary_csv_path = file_output_dir / "homology_check.csv"
                with open(summary_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Header: File, Chain, Identity, E-value, Template_Name
                    writer.writerow([
                        "File", "Chain", "Identity", "E-value", "Template_Name"
                    ])
                    # Write each hit as a row
                    for hit in hits:
                        writer.writerow([
                            input_file.name,
                            chain_id,
                            f"{hit['identity']:.2f}",
                            f"{hit['evalue']:.2e}",
                            hit['name']
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
    parser.add_argument("--input_files", type=str, required=True, nargs='+')
    args = parser.parse_args()
    main([Path(f) for f in args.input_files], Path(args.output_dir), args.job_id)