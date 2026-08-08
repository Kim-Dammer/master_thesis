#!/usr/bin/env python3
"""
Scan CombFold output directories for chain overlap issues.

Detects three failure modes:
  1. FAILED  - CombFold produced no output PDBs (assembly failed entirely)
  2. OVERLAP - Output PDBs have chains with identical/near-identical coordinates
  3. PARTIAL - Output PDBs have fewer chains than expected from stoichiometry

For each output PDB, compares all chain pairs by CA-CA distance and classifies:
  exact : mean CA distance < 0.01 A  (perfect coordinate duplication)
  near  : mean CA distance < 1.0 A   (near-identical, likely identity transform)
  close : mean CA distance < 5.0 A   (suspiciously close, may indicate problem)
  none  : mean CA distance >= 5.0 A  (normal)

Also extracts diagnostics from CombFold's output.log:
  - Number of transformations extracted from AFM predictions
  - Representative subunit pLDDT
  - Maximum assembly iteration reached
  - Whether assembly explicitly failed

Usage:
    python scan_combfold_outputs.py /path/to/CombFold/
    uv run scan_combfold_outputs.py /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/6_sixth_subset_homomultimers_pool_vs_pair/CombFold --output /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/6_sixth_subset_homomultimers_pool_vs_pair/CombFold/assembled_report.csv --verbose

Output:
  - CSV file (one row per output PDB, plus one row per failed complex)
  - Summary statistics printed to stderr
  - List of problematic complexes with exact/near overlaps

Requirements:
  - Python 3.6+
  - numpy (already required by CombFold)
"""

import os
import sys
import csv
import json
import re
import argparse
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def parse_pdb_ca_atoms(pdb_path):
    """Parse CA atoms from a PDB file, grouped by chain.

    Returns: {chain_id: {resseq: np.array([x, y, z])}}
    """
    chains = defaultdict(dict)
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                chain_id = line[21]
                try:
                    resseq = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                chains[chain_id][resseq] = np.array([x, y, z])
    except Exception as e:
        print(f"  WARNING: could not parse {pdb_path}: {e}", file=sys.stderr)
    return dict(chains)


def check_chain_overlaps(chains, close_threshold=5.0):
    """Compare all chain pairs by CA-CA distance.

    Returns:
      overlaps: list of dicts with chain1, chain2, mean_dist, max_dist, classification
      min_mean_dist: minimum mean CA distance across all chain pairs (inf if <2 chains)
    """
    chain_ids = sorted(chains.keys())
    overlaps = []
    min_mean_dist = float("inf")

    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            c1, c2 = chain_ids[i], chain_ids[j]
            common = sorted(set(chains[c1].keys()) & set(chains[c2].keys()))
            if len(common) < 3:
                continue
            dists = [np.linalg.norm(chains[c1][r] - chains[c2][r]) for r in common]
            mean_dist = float(np.mean(dists))
            max_dist = float(np.max(dists))
            min_mean_dist = min(min_mean_dist, mean_dist)

            if mean_dist < close_threshold:
                overlaps.append({
                    "chain1": c1,
                    "chain2": c2,
                    "mean_dist": mean_dist,
                    "max_dist": max_dist,
                    "classification": classify_overlap(mean_dist),
                })

    return overlaps, min_mean_dist


def classify_overlap(mean_dist):
    """Classify the severity of a chain overlap."""
    if mean_dist < 0.01:
        return "exact"
    elif mean_dist < 1.0:
        return "near"
    elif mean_dist < 5.0:
        return "close"
    return "none"


# ---------------------------------------------------------------------------
# Log / config parsing
# ---------------------------------------------------------------------------

def parse_log(log_path):
    """Parse CombFold's output.log for diagnostics.

    Returns dict with:
      num_transformations: total transformations extracted from AFM PDBs
      rep_plddt: pLDDT of representative subunit (or None)
      max_iteration: highest assembly iteration reached (0 if none)
      failed: True if "Could not assemble" appears in log
      failure_reason: string describing why it failed (or "")
    """
    info = {
        "num_transformations": 0,
        "rep_plddt": None,
        "max_iteration": 0,
        "failed": False,
        "failure_reason": "",
    }

    if not os.path.exists(log_path):
        # Try alternate log file name
        alt = os.path.join(os.path.dirname(log_path), "log")
        if os.path.exists(alt):
            log_path = alt
        else:
            return info

    try:
        with open(log_path) as f:
            for line in f:
                # Count transformations
                m = re.search(r"found (\d+) transformations? between", line)
                if m:
                    info["num_transformations"] += int(m.group(1))

                # Representative pLDDT
                m = re.search(r"rep \S+ has plddt score ([\d.]+)", line)
                if m:
                    info["rep_plddt"] = float(m.group(1))

                # Assembly iteration tracking
                m = re.search(r"running iteration (\d+).*prev kept results: (\d+)", line)
                if m:
                    it = int(m.group(1))
                    kept = int(m.group(2))
                    info["max_iteration"] = max(info["max_iteration"], it)

                # Explicit failure
                if "Could not assemble" in line:
                    info["failed"] = True
                    info["failure_reason"] = "Could not assemble"

                # Connectivity failure
                if "Not enough transformations" in line:
                    info["failed"] = True
                    info["failure_reason"] = "Not enough transformations (disconnected graph)"

                # Missing interface
                if "Skipping transformation, missing interface" in line:
                    info["failure_reason"] = "Missing interface between subunits"
    except Exception as e:
        print(f"  WARNING: could not parse log {log_path}: {e}", file=sys.stderr)

    return info


def parse_subunits_json(json_path):
    """Parse subunits.json for stoichiometry.

    Returns: (total_chains, stoichiometry_str, num_unique_subunits)
      e.g., (4, "O13297x4", 1) for a homotetramer
    """
    if not os.path.exists(json_path):
        return None, None, None
    try:
        with open(json_path) as f:
            data = json.load(f)
        total_chains = 0
        stoich_parts = []
        for key, subunit in data.items():
            n = len(subunit.get("chain_names", []))
            total_chains += n
            name = subunit.get("name", key)
            stoich_parts.append(f"{name}x{n}")
        return total_chains, ";".join(stoich_parts), len(data)
    except Exception as e:
        print(f"  WARNING: could not parse {json_path}: {e}", file=sys.stderr)
        return None, None, None


def parse_confidence(conf_path):
    """Parse confidence.txt for assembly confidence scores."""
    if not os.path.exists(conf_path):
        return []
    scores = []
    try:
        with open(conf_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        scores.append(float(parts[-1]))
                    except ValueError:
                        pass
    except Exception:
        pass
    return scores


def count_input_pdbs(input_dir):
    """Count PDB files in the input pdbs/ folder."""
    pdbs_dir = os.path.join(input_dir, "pdbs")
    if not os.path.isdir(pdbs_dir):
        return 0
    return len([f for f in os.listdir(pdbs_dir) if f.endswith(".pdb")])


# ---------------------------------------------------------------------------
# Main scanning logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scan CombFold output directories for chain overlap issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scan_combfold_outputs.py /path/to/CombFold/
    python scan_combfold_outputs.py /path/to/CombFold/ -o report.csv -v

After running, filter the CSV:
    # Show only exact/near overlaps:
    awk -F',' 'NR==1 || $10=="exact" || $10=="near"' report.csv | column -t -s','
    # Show only failed assemblies:
    awk -F',' 'NR==1 || $5=="failed"' report.csv | column -t -s','
        """,
    )
    parser.add_argument("combfold_dir", help="Root CombFold directory containing *_output folders")
    parser.add_argument("--output", "-o", default="combfold_overlap_report.csv",
                        help="Output CSV path (default: combfold_overlap_report.csv)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-complex progress to stderr")
    args = parser.parse_args()

    if not os.path.isdir(args.combfold_dir):
        print(f"Error: {args.combfold_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find all *_output directories
    output_dirs = []
    for item in sorted(os.listdir(args.combfold_dir)):
        item_path = os.path.join(args.combfold_dir, item)
        if item.endswith("_output") and os.path.isdir(item_path):
            output_dirs.append(item)

    if not output_dirs:
        print(f"No *_output directories found in {args.combfold_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(output_dirs)} output directories to scan\n", file=sys.stderr)

    # Collect data
    rows = []
    stats = defaultdict(lambda: defaultdict(int))
    problematic = []  # exact/near overlaps
    stoich_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # [stoich][source][status]

    for dirname in output_dirs:
        # Parse complex name and source from directory name
        # e.g., "O13297x4_pool_output" -> complex="O13297x4", source="pool"
        base = dirname[: -len("_output")]
        if base.endswith("_pair"):
            complex_name = base[: -len("_pair")]
            source = "pair"
        elif base.endswith("_pool"):
            complex_name = base[: -len("_pool")]
            source = "pool"
        else:
            complex_name = base
            source = "unknown"

        output_dir = os.path.join(args.combfold_dir, dirname)
        input_dir = os.path.join(args.combfold_dir, dirname.replace("_output", "_input"))

        # Stoichiometry from subunits.json
        expected_chains, stoich_str, num_unique = parse_subunits_json(
            os.path.join(input_dir, "subunits.json")
        )

        # Input PDB count
        num_input_pdbs = count_input_pdbs(input_dir)

        # Parse assembly log
        log_path = os.path.join(output_dir, "_unified_representation", "assembly_output", "output.log")
        log_info = parse_log(log_path)

        # Check for assembled results
        assembled_dir = os.path.join(output_dir, "assembled_results")
        pdb_files = []
        if os.path.isdir(assembled_dir):
            pdb_files = sorted([f for f in os.listdir(assembled_dir) if f.endswith(".pdb")])

        # Check if output_clustered.res exists (assembly produced results)
        clustered_res_path = os.path.join(
            output_dir, "_unified_representation", "assembly_output", "output_clustered.res"
        )
        has_clustered_res = os.path.exists(clustered_res_path)

        # Confidence scores
        confidence_scores = parse_confidence(os.path.join(assembled_dir, "confidence.txt"))

        # --- Failed: no output PDBs ---
        if not pdb_files:
            row = {
                "complex": complex_name,
                "source": source,
                "stoichiometry": stoich_str or "",
                "expected_chains": expected_chains if expected_chains is not None else "",
                "status": "failed",
                "pdb_file": "",
                "actual_chains": "",
                "chain_ids": "",
                "has_overlap": "",
                "overlap_type": "",
                "overlapping_pairs": "",
                "min_pairwise_dist": "",
                "partial_assembly": "",
                "num_transformations": log_info["num_transformations"],
                "rep_plddt": log_info["rep_plddt"] if log_info["rep_plddt"] is not None else "",
                "max_iteration": log_info["max_iteration"],
                "failure_reason": log_info["failure_reason"],
                "has_clustered_res": has_clustered_res,
                "num_input_pdbs": num_input_pdbs,
                "confidence": "",
            }
            rows.append(row)
            stats[source]["failed"] += 1
            if stoich_str:
                stoich_stats[stoich_str][source]["failed"] += 1
            if args.verbose:
                reason = log_info["failure_reason"] or "no output PDBs"
                print(f"  {dirname}: FAILED ({reason})", file=sys.stderr)
            continue

        # --- Success: process each output PDB ---
        stats[source]["success"] += 1
        if stoich_str:
            stoich_stats[stoich_str][source]["success"] += 1

        for pdb_file in pdb_files:
            pdb_path = os.path.join(assembled_dir, pdb_file)
            chains = parse_pdb_ca_atoms(pdb_path)

            chain_ids = sorted(chains.keys())
            actual_chains = len(chain_ids)

            overlaps, min_dist = check_chain_overlaps(chains)

            has_overlap = len(overlaps) > 0
            overlap_type = ""
            overlap_pairs_str = ""

            if has_overlap:
                overlaps.sort(key=lambda x: x["mean_dist"])
                worst = overlaps[0]
                overlap_type = worst["classification"]
                overlap_pairs_str = "; ".join(
                    f"{o['chain1']}={o['chain2']}({o['mean_dist']:.4f}A)"
                    for o in overlaps
                )

                if overlap_type in ("exact", "near"):
                    stats[source]["overlap_problematic"] += 1
                    stoich_stats[stoich_str][source]["overlap_problematic"] += 1
                    problematic.append({
                        "complex": complex_name,
                        "source": source,
                        "pdb_file": pdb_file,
                        "overlap_type": overlap_type,
                        "overlapping_pairs": overlap_pairs_str,
                        "num_transformations": log_info["num_transformations"],
                        "rep_plddt": log_info["rep_plddt"],
                        "expected_chains": expected_chains,
                        "actual_chains": actual_chains,
                        "stoichiometry": stoich_str,
                    })
                elif overlap_type == "close":
                    stats[source]["overlap_close"] += 1
                    stoich_stats[stoich_str][source]["overlap_close"] += 1
            else:
                stats[source]["no_overlap"] += 1
                stoich_stats[stoich_str][source]["no_overlap"] += 1

            # Check for partial assembly
            partial = ""
            if expected_chains is not None and actual_chains < expected_chains:
                partial = f"{actual_chains}/{expected_chains}"
                stats[source]["partial"] += 1
            elif expected_chains is not None and actual_chains > expected_chains:
                partial = f"extra({actual_chains}/{expected_chains})"

            row = {
                "complex": complex_name,
                "source": source,
                "stoichiometry": stoich_str or "",
                "expected_chains": expected_chains if expected_chains is not None else "",
                "status": "success",
                "pdb_file": pdb_file,
                "actual_chains": actual_chains,
                "chain_ids": ",".join(chain_ids),
                "has_overlap": has_overlap,
                "overlap_type": overlap_type,
                "overlapping_pairs": overlap_pairs_str,
                "min_pairwise_dist": f"{min_dist:.4f}" if min_dist != float("inf") else "",
                "partial_assembly": partial,
                "num_transformations": log_info["num_transformations"],
                "rep_plddt": log_info["rep_plddt"] if log_info["rep_plddt"] is not None else "",
                "max_iteration": log_info["max_iteration"],
                "failure_reason": "",
                "has_clustered_res": has_clustered_res,
                "num_input_pdbs": num_input_pdbs,
                "confidence": confidence_scores[0] if confidence_scores else "",
            }
            rows.append(row)

            if args.verbose:
                if has_overlap and overlap_type in ("exact", "near"):
                    print(f"  {dirname}/{pdb_file}: OVERLAP ({overlap_type}) {overlap_pairs_str}",
                          file=sys.stderr)
                elif has_overlap:
                    print(f"  {dirname}/{pdb_file}: close chains {overlap_pairs_str}",
                          file=sys.stderr)
                elif partial:
                    print(f"  {dirname}/{pdb_file}: PARTIAL {partial}", file=sys.stderr)
                else:
                    print(f"  {dirname}/{pdb_file}: OK ({actual_chains} chains, min_dist={min_dist:.1f}A)",
                          file=sys.stderr)

    # --- Write CSV ---
    fieldnames = [
        "complex", "source", "stoichiometry", "expected_chains", "status",
        "pdb_file", "actual_chains", "chain_ids", "has_overlap", "overlap_type",
        "overlapping_pairs", "min_pairwise_dist", "partial_assembly",
        "num_transformations", "rep_plddt", "max_iteration", "failure_reason",
        "has_clustered_res", "num_input_pdbs", "confidence",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Print summary ---
    print("\n" + "=" * 72, file=sys.stderr)
    print("CombFold Output Scan Summary", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    total = len(output_dirs)
    print(f"\nTotal complexes: {total}", file=sys.stderr)

    for src in ["pair", "pool", "unknown"]:
        t = stats[src]["success"] + stats[src]["failed"]
        if t > 0:
            print(f"  {src}: {t} complexes", file=sys.stderr)

    # Status breakdown
    total_success = sum(stats[s]["success"] for s in stats)
    total_failed = sum(stats[s]["failed"] for s in stats)
    print(f"\nAssembly status:", file=sys.stderr)
    print(f"  success: {total_success:>4d} ({100*total_success/total:.1f}%)", file=sys.stderr)
    print(f"  failed:  {total_failed:>4d} ({100*total_failed/total:.1f}%)", file=sys.stderr)
    for src in ["pair", "pool"]:
        t = stats[src]["success"] + stats[src]["failed"]
        if t > 0:
            print(f"    {src}: {stats[src]['success']}/{t} success "
                  f"({100*stats[src]['success']/t:.1f}%), "
                  f"{stats[src]['failed']}/{t} failed", file=sys.stderr)

    # Overlap breakdown
    total_exact = sum(stats[s]["overlap_problematic"] for s in stats)
    total_close = sum(stats[s]["overlap_close"] for s in stats)
    total_clean = sum(stats[s]["no_overlap"] for s in stats)
    print(f"\nChain overlap among {total_success} successful outputs:", file=sys.stderr)
    print(f"  exact/near (< 1.0 A):  {total_exact:>4d}", file=sys.stderr)
    for src in ["pair", "pool"]:
        print(f"    {src}: {stats[src]['overlap_problematic']}", file=sys.stderr)
    print(f"  close (1-5 A):         {total_close:>4d}", file=sys.stderr)
    for src in ["pair", "pool"]:
        print(f"    {src}: {stats[src]['overlap_close']}", file=sys.stderr)
    print(f"  no overlap (>= 5 A):   {total_clean:>4d}", file=sys.stderr)
    for src in ["pair", "pool"]:
        print(f"    {src}: {stats[src]['no_overlap']}", file=sys.stderr)

    # Partial assemblies
    total_partial = sum(stats[s]["partial"] for s in stats)
    if total_partial > 0:
        print(f"\nPartial assemblies (fewer chains than expected): {total_partial}", file=sys.stderr)

    # Breakdown by stoichiometry
    if stoich_stats:
        print(f"\nBreakdown by stoichiometry:", file=sys.stderr)
        print(f"  {'Stoichiometry':<25} {'Source':<8} {'Success':>8} {'Failed':>8} "
              f"{'Overlap':>8} {'Clean':>8}", file=sys.stderr)
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", file=sys.stderr)
        for stoich in sorted(stoich_stats.keys()):
            for src in ["pair", "pool"]:
                s = stoich_stats[stoich][src]
                t = s["success"] + s["failed"]
                if t > 0:
                    print(f"  {stoich:<25} {src:<8} {s['success']:>8} {s['failed']:>8} "
                          f"{s['overlap_problematic']:>8} {s['no_overlap']:>8}", file=sys.stderr)

    # Problematic complexes list
    if problematic:
        print(f"\n{'=' * 72}", file=sys.stderr)
        print(f"Problematic complexes with exact/near chain overlap ({len(problematic)}):",
              file=sys.stderr)
        print(f"{'=' * 72}", file=sys.stderr)
        print(f"{'Complex':<20} {'Src':<6} {'PDB':<28} {'Overlapping pairs':<35} "
              f"{'Trans':>5} {'pLDDT':>7}", file=sys.stderr)
        print(f"{'-'*20} {'-'*6} {'-'*28} {'-'*35} {'-'*5} {'-'*7}", file=sys.stderr)
        for p in sorted(problematic, key=lambda x: (x["source"], x["complex"])):
            plddt_str = f"{p['rep_plddt']:.1f}" if p["rep_plddt"] is not None else ""
            print(f"{p['complex']:<20} {p['source']:<6} {p['pdb_file']:<28} "
                  f"{p['overlapping_pairs'][:35]:<35} {p['num_transformations']:>5} "
                  f"{plddt_str:>7}", file=sys.stderr)

    # Correlation: transformations vs overlap
    if rows:
        print(f"\n{'=' * 72}", file=sys.stderr)
        print("Correlation: number of transformations vs. outcome", file=sys.stderr)
        print(f"{'=' * 72}", file=sys.stderr)
        trans_buckets = defaultdict(lambda: defaultdict(int))
        for r in rows:
            try:
                n = int(r["num_transformations"])
            except (ValueError, KeyError):
                continue
            bucket = "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5 else "6+"
            if r["status"] == "failed":
                trans_buckets[bucket]["failed"] += 1
            elif r.get("overlap_type") in ("exact", "near"):
                trans_buckets[bucket]["overlap"] += 1
            elif r["status"] == "success":
                trans_buckets[bucket]["clean"] += 1
        print(f"  {'Transforms':<12} {'Failed':>8} {'Overlap':>8} {'Clean':>8} {'Total':>8}",
              file=sys.stderr)
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", file=sys.stderr)
        for bucket in ["0", "1", "2", "3-5", "6+"]:
            b = trans_buckets[bucket]
            t = sum(b.values())
            if t > 0:
                print(f"  {bucket:<12} {b['failed']:>8} {b['overlap']:>8} {b['clean']:>8} {t:>8}",
                      file=sys.stderr)

    print(f"\nDetailed CSV report: {args.output}", file=sys.stderr)
    print(f"\nQuick filters:", file=sys.stderr)
    print(f"  # Exact/near overlaps only:", file=sys.stderr)
    print(f"  awk -F',' 'NR==1 || $10==\"exact\" || $10==\"near\"' {args.output} | column -t -s','",
          file=sys.stderr)
    print(f"  # Failed assemblies only:", file=sys.stderr)
    print(f"  awk -F',' 'NR==1 || $5==\"failed\"' {args.output} | column -t -s','", file=sys.stderr)
    print(f"  # Partial assemblies only:", file=sys.stderr)
    print(f"  awk -F',' 'NR==1 || $13!=\"\" && $13!=\"\" ' {args.output} | column -t -s','",
          file=sys.stderr)


if __name__ == "__main__":
    main()
