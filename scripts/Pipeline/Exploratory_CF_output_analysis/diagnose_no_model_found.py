#!/usr/bin/env python3
"""Diagnose which complexes are `no_model_found` and WHY, for the 3 unexpected
cases (CPX-1640, CPX-1706, CPX-2800 -- CPX-940/CPX-1849 are expected to fail,
they have 0 CombFold assemblies per your results CSV).

For each target complex, prints:
  1. every candidate stoichiometry the manifest produces (identifiers/pred_1/2/3)
     and the exact folder stub + suffix combos the pipeline will look for
  2. whether each of those exact paths exists on disk
  3. a fallback: any folder under --combfold-base whose name contains ANY of
     the complex's UniProt accessions (case-insensitive substring match) --
     this reveals the REAL folder name if it differs from what we computed
     (different accession order, different suffix, extra text, etc.)

Usage (no editing needed -- just point the 3 paths at your real files):
    python3 diagnose_no_model_found.py \
        --manifest /cluster/.../t6_CF_test_Example_for_RM_TM/all_pdb_present_t4_CF_test_pipeline_complexes_combfold_results.csv \
        --combfold-base /cluster/.../t11_RM_TM_updated_CF_pipeline/CombFold \
        --only CPX-1640,CPX-1706,CPX-2800
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import json

import pandas as pd

_UNIPROT = r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
_ACC_ONLY = re.compile(_UNIPROT)
_IDENT_RE = re.compile(r"([A-Za-z0-9_.:-]+)\((\d+)\)")
_PRED_DICT_RE = re.compile(r"^\s*(\{[^{}]*\})")


def _is_uniprot(acc: str) -> bool:
    return bool(_ACC_ONLY.fullmatch(acc))


def parse_identifiers(text):
    out = {}
    if not isinstance(text, str):
        return out
    for m in _IDENT_RE.finditer(text):
        acc, cnt = m.group(1), int(m.group(2))
        if _is_uniprot(acc):
            out[acc] = cnt
    return out


def parse_pred_field(text):
    if not isinstance(text, str) or not text.strip():
        return {}
    m = _PRED_DICT_RE.match(text.strip())
    if not m:
        return {}
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    return {str(k): int(v) for k, v in d.items() if _is_uniprot(str(k))}


def folder_stub(stoich):
    return "_".join(f"{acc}x{cnt}" for acc, cnt in sorted(stoich.items()))


def candidate_stoichiometries(row):
    stub_to_label = {}
    ordered = []
    sources = [
        ("identifiers", parse_identifiers(row.get("identifiers"))),
        ("pred_1", parse_pred_field(row.get("pred_1"))),
        ("pred_2", parse_pred_field(row.get("pred_2"))),
        ("pred_3", parse_pred_field(row.get("pred_3"))),
    ]
    for label, stoich in sources:
        if not stoich:
            continue
        stub = folder_stub(stoich)
        if stub in stub_to_label:
            stub_to_label[stub] += "+" + label
        else:
            stub_to_label[stub] = label
            ordered.append((stub, stoich))
    return [(stub_to_label[s], st) for s, st in ordered]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--combfold-base", required=True)
    ap.add_argument("--only", default="", help="comma-separated complex_ac list; default = all rows in manifest")
    ap.add_argument("--suffixes", default="pool_output,pair_output")
    a = ap.parse_args()

    if not os.path.isdir(a.combfold_base):
        print(f"ERROR: --combfold-base does not exist or is not a directory: {a.combfold_base}", file=sys.stderr)
        return 1

    suffixes = [s.strip() for s in a.suffixes.split(",") if s.strip()]
    only = {x.strip() for x in a.only.split(",") if x.strip()} or None

    mf = pd.read_csv(a.manifest)
    if "complex_ac" not in mf.columns:
        print("ERROR: manifest has no 'complex_ac' column", file=sys.stderr)
        return 1

    try:
        all_entries = sorted(os.listdir(a.combfold_base))
    except OSError as e:
        print(f"ERROR: cannot list --combfold-base: {e}", file=sys.stderr)
        return 1
    all_dirs = [e for e in all_entries if os.path.isdir(os.path.join(a.combfold_base, e))]
    print(f"[info] --combfold-base has {len(all_dirs)} subdirectories total\n")

    for _, row in mf.iterrows():
        cac = str(row["complex_ac"]).strip()
        if only and cac not in only:
            continue
        print(f"===== {cac} =====")
        cands = candidate_stoichiometries(row)
        if not cands:
            print("  NO candidate stoichiometries parsed from manifest row (identifiers/pred_1/2/3 all empty/unparseable)")
            print()
            continue

        all_accs = set()
        any_exact_hit = False
        for label, stoich in cands:
            stub = folder_stub(stoich)
            all_accs.update(stoich.keys())
            for suffix in suffixes:
                expected = f"{stub}_{suffix}"
                path = os.path.join(a.combfold_base, expected)
                exists = os.path.isdir(path)
                any_exact_hit = any_exact_hit or exists
                mark = "EXISTS" if exists else "missing"
                print(f"  [{label:22s}] expected folder: {expected:60s} -> {mark}")

        if not any_exact_hit:
            print(f"\n  No exact match found. Searching --combfold-base for ANY folder containing "
                  f"any of this complex's accessions ({sorted(all_accs)})...")
            hits = [e for e in all_dirs if any(acc in e.upper() for acc in all_accs)]
            if hits:
                print(f"  Found {len(hits)} folder(s) containing at least one accession (real folder name, "
                      f"for comparison against what we expected above):")
                for h in hits[:15]:
                    print(f"    - {h}")
                if len(hits) > 15:
                    print(f"    ... and {len(hits) - 15} more")
            else:
                print(f"  Found NOTHING under --combfold-base containing any of {sorted(all_accs)}.")
                print(f"  -> This complex was very likely never folded under THIS --combfold-base "
                      f"(check if it belongs to a different experiment/run directory).")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
