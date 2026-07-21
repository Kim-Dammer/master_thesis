#!/usr/bin/env python3
"""
lookup_iptm.py — Submit the iptm lookup sbatch job.

Submits 08_lookup_iptm.sbatch to the cluster and prints the job ID.
All heavy computation (loading the pooled-PPI DB) runs on the compute node.

Usage (on cluster login node):
    source /cluster/project/beltrao/kdammer/master_thesis/.venv/bin/activate
    python lookup_iptm.py
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SBATCH_SCRIPT = Path(__file__).parent / "08_lookup_iptm.sh"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not SBATCH_SCRIPT.exists():
        sys.exit(f"ERROR: sbatch script not found at {SBATCH_SCRIPT}")

    print(f"Submitting {SBATCH_SCRIPT} ...")
    result = subprocess.run(
        ["sbatch", str(SBATCH_SCRIPT)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        sys.exit(f"sbatch failed:\n{result.stderr}")

    # sbatch prints "Submitted batch job <ID>"
    output = result.stdout.strip()
    print(output)

    # Extract job ID
    parts = output.split()
    if len(parts) >= 4 and parts[-1].isdigit():
        job_id = parts[-1]
        print(f"\nJob ID: {job_id}")
        print(f"Monitor with:  squeue -j {job_id}")
        print(f"Output log:    logs/lookup_iptm_{job_id}.out")
        print(f"Error log:     logs/lookup_iptm_{job_id}.err")
    else:
        print(f"\nCould not parse job ID from: {output}")


if __name__ == "__main__":
    main()
