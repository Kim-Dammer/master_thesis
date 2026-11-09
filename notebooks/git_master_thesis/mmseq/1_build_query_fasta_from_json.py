#!/usr/bin/env python3
"""
Build a combined query FASTA directly from your AF3-style input JSON files
(the ones in .../af3-template-analysis/data), for use with mmseqs2.

This reads straight from the source JSONs, so it doesn't require having
already run the HMMER pipeline (unlike pulling from the *_templates.fasta
files inside chunk_analysis_results/) - useful if you want to run mmseqs2
standalone, or on proteins you haven't sent through stage 1 yet.

Protein ID is derived the same way run_chunk_analysis.py / the sbatch
scripts derive it: the part of the filename before the first underscore,
lowercased - e.g. "O13297_uniqSeq.json" -> protein_id "o13297". This keeps
IDs consistent with your existing chunk_analysis_results/ folder names.

If a JSON has more than one chain, each chain gets its own FASTA entry,
headered as "{protein_id}_{chain_id}" to avoid collisions; single-chain
JSONs (your current data) just use "{protein_id}".

Usage:
    python build_query_fasta_from_json.py \\
        --input_dir /cluster/project/beltrao/kdammer/master_thesis/scripts/af3-template-analysis_from_ubnAF3Benchmark/af3-template-analysis/data \\
        --output all_queries.fasta

script_order: build_query_fasta_from_json.py -> run_mmseqs_search.sh -> run_chunk_analysis_mmseqs2.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def get_protein_id(json_path: Path) -> str:
    return json_path.stem.split("_")[0].lower()


def main():
    parser = argparse.ArgumentParser(description="Build a combined FASTA from AF3-style protein JSONs")
    parser.add_argument("--input_dir", required=True, help="Directory containing *.json input files")
    parser.add_argument("--output", required=True, help="Output combined FASTA path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"ERROR: no .json files found in {input_dir}")
        return

    # Track which protein_ids we've seen (from filename) to warn on collisions,
    # same case-insensitive dedup logic as submit_parallel_analysis.sh
    seen_protein_ids = defaultdict(list)
    fasta_entries = []  # list of (header, sequence)

    n_files = 0
    n_sequences = 0
    n_skipped_files = 0
    n_skipped_chains = 0

    for json_path in json_files:
        protein_id = get_protein_id(json_path)
        seen_protein_ids[protein_id].append(json_path.name)

        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [SKIP FILE] {json_path.name} - could not parse: {e}")
            n_skipped_files += 1
            continue

        chains = [m["protein"] for m in data.get("sequences", []) if "protein" in m]

        if not chains:
            print(f"  [SKIP FILE] {json_path.name} - no protein entries found in 'sequences'")
            n_skipped_files += 1
            continue

        multi_chain = len(chains) > 1

        for chain in chains:
            sequence = chain.get("sequence")
            chain_id = chain.get("id", "A")

            if not sequence:
                print(f"  [SKIP CHAIN] {json_path.name} chain {chain_id} - missing 'sequence'")
                n_skipped_chains += 1
                continue

            header = f"{protein_id}_{chain_id}" if multi_chain else protein_id
            fasta_entries.append((header, sequence))
            n_sequences += 1

        n_files += 1

    # Warn about protein_id collisions (multiple files mapping to the same
    # first-underscore-chunk protein_id - these would also collide in the
    # HMMER pipeline's output folder naming).
    collisions = {pid: files for pid, files in seen_protein_ids.items() if len(files) > 1}
    if collisions:
        print("\nWARNING: the following protein_ids come from multiple input files")
        print("(same first-underscore-chunk of the filename) - check these aren't")
        print("unintentionally overwriting each other's output downstream:")
        for pid, files in collisions.items():
            print(f"  {pid}: {files}")

    # Write combined FASTA
    with open(args.output, "w") as f:
        for header, sequence in fasta_entries:
            f.write(f">{header}\n{sequence}\n")

    print(f"\nProcessed {n_files} JSON files ({n_skipped_files} skipped).")
    print(f"Wrote {n_sequences} sequences ({n_skipped_chains} chains skipped) to {args.output}")


if __name__ == "__main__":
    main()