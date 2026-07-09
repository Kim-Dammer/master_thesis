#!/usr/bin/env python3
"""
Generate AF3-style JSON input files from a CSV of UniProt IDs and sequences.

The af3-template-analysis pipeline (run_chunk_analysis.py) expects one JSON file
per protein, in the AlphaFold3 input format:

    {
      "name": "a0_uniqSeq",
      "sequences": [
        {"protein": {"id": "A", "sequence": "MKL..."}}
      ]
    }

Filenames follow the convention used in the original benchmark:
    a0_uniqSeq.json, a1_uniqSeq.json, a2_uniqSeq.json, ...

Usage:
    uv run generate_input_json.py \\
        --input_csv  /cluster/project/beltrao/kdammer/master_thesis/data/iPTM_and_pLDDT/all_yeast_proteins_uniprot_mapped_sequences.csv \\
        --output_dir /cluster/project/beltrao/kdammer/master_thesis/scripts/af3-template-analysis_from_ubnAF3Benchmark/af3-template-analysis/data \\
        [--skip_missing_sequences]

Outputs:
    - <output_dir>/a0_uniqSeq.json, a1_uniqSeq.json, ...  (one per protein)
    - <output_dir>/id_mapping.csv  (index -> uniprot_id, for traceability)
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert UniProt ID + sequence CSV into AF3 JSON input files."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV with columns: uniprot_id, sequence",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write JSON files (created if it doesn't exist)",
    )
    parser.add_argument(
        "--skip_missing_sequences",
        action="store_true",
        help="Skip rows where sequence is empty/NaN instead of erroring out",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    if not input_csv.exists():
        print(f"ERROR: Input CSV not found: {input_csv}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    rows = []
    with open(input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "uniprot_id" not in reader.fieldnames or "sequence" not in reader.fieldnames:
            print(
                f"ERROR: CSV must have 'uniprot_id' and 'sequence' columns. "
                f"Found: {reader.fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from {input_csv}")

    # Check for duplicate UniProt IDs
    seen = set()
    duplicates = []
    for row in rows:
        uid = row["uniprot_id"]
        if uid in seen:
            duplicates.append(uid)
        seen.add(uid)
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate UniProt IDs found: {duplicates[:10]}")

    # Generate JSON files
    written = 0
    skipped = 0
    mapping_rows = []

    for idx, row in enumerate(rows):
        uid = row["uniprot_id"]
        sequence = row.get("sequence", "")

        # Handle missing sequences
        if not sequence or sequence.strip() == "" or sequence.lower() == "nan":
            if args.skip_missing_sequences:
                print(f"  SKIP (no sequence): {uid}")
                skipped += 1
                continue
            else:
                print(
                    f"ERROR: No sequence for {uid}. "
                    f"Use --skip_missing_sequences to skip these rows.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # Strip whitespace from sequence
        sequence = sequence.strip()

        # Build the AF3 JSON structure
        name = f"{uid}_uniqSeq"
        json_data = {
            "name": name,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": sequence,
                    }
                }
            ],
        }

        # Write JSON file
        out_path = output_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(json_data, f, indent=2)

        mapping_rows.append({"index": idx, "json_name": name, "uniprot_id": uid, "seq_length": len(sequence)})
        written += 1

    # Write the index -> uniprot_id mapping for traceability
    mapping_path = output_dir / "id_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "json_name", "uniprot_id", "seq_length"])
        writer.writeheader()
        writer.writerows(mapping_rows)

    print(f"\nDone.")
    print(f"  Written:   {written} JSON files to {output_dir}/")
    print(f"  Skipped:   {skipped} (missing sequences)")
    print(f"  Mapping:   {mapping_path}")
    print(f"\nExample file ({mapping_rows[0]['json_name']}.json):")
    example_path = output_dir / f"{mapping_rows[0]['json_name']}.json"
    with open(example_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
