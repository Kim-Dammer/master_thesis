#!/usr/bin/env python3
"""Standalone batch script: score CombFold assemblies against reference PDBs.

Runs AFTER CombFold. For every complex in a mapping file it locates the CombFold
assembled model(s) and the reference structure (biological assembly + asymmetric
unit, downloaded from RCSB), maps chains BY SEQUENCE, and reports three
complementary accuracy levels with coverage:

  * per-subunit fold      -> per-chain TM-score (both norms) + CA-RMSD (2 variants)
  * global assembly       -> complex TM-score (both norms) + global CA-RMSD
  * per-interface         -> DockQ / Fnat / iRMSD / LRMSD / CAPRI class + mean

Outputs: complex_summary.csv, per_chain.csv, per_interface.csv, per-complex JSON,
run_log.txt (see combfold_eval/pipeline.py for the full schema + mapping-file
column contract).

--------------------------------------------------------------------------------
USAGE
uv run 02_combfold_eval.py \
    --mapping /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/all_pdb_matches_with_match_class.csv \
    --combfold-base /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/t6_CF_test_Example_for_RM_TM/CombFold \
    --out-dir /cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/t6_CF_test_Example_for_RM_TM/comfold_eval \

    # score only the top-ranked cluster instead of all clusters:
    python run_combfold_eval.py --top-cluster-only

    # restrict to a few complexes (debugging):
    python run_combfold_eval.py --only CPX-2800,CPX-737

REQUIREMENTS
    python: gemmi, biopython, numpy, scipy, pandas, DockQ 

--------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import os
import sys

# make the package importable when this script is run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from procompa.combfold_eval.config import Config
from procompa.combfold_eval.sequences import UniProtSequences
from procompa.combfold_eval.pipeline import run_batch

# =============================== CONFIG ======================================
# Defaults; every one is overridable on the command line (see --help).
MAPPING_FILE   = "best_match_per_complex.csv"   # complex_ac, pdb_id (+ optional cols)
COMBFOLD_BASE  = "."                             # root holding per-complex CombFold output
UNIPROT_CSV    = "all_yeast_proteins_uniprot_mapped_sequences.csv"  # uniprot_id,sequence
OUT_DIR        = "combfold_eval_out"
# Persistent cluster-wide cache: one subfolder per PDB id, checked (exists +
# non-empty) before any download -- reruns from any working directory reuse it
# instead of re-fetching. Override with --ref-cache if running elsewhere.
REF_CACHE      = "/cluster/project/beltrao/kdammer/master_thesis/data/reference_pdb"
USALIGN_BIN    = "USalign"
DOCKQ_BIN      = "DockQ"

SCORE_ALL_CLUSTERS   = True    # score every CombFold cluster (else top-ranked only)
SAME_PROTEIN_IDENT   = 90.0    # %identity for "same protein" chain assignment
HOMOLOG_IDENT        = 30.0    # %identity floor to accept a homolog/paralog match
OUTLIER_CYCLES       = 5       # RMSD refinement cycles (0 = no outlier rejection)
OUTLIER_CUTOFF       = 2.0     # adaptive cutoff = OUTLIER_CUTOFF x current RMSD
SAVE_STRUCTURES      = False   # copy the exact scored PDBs of the primary form
DOCKQ_ALLOWED_MISMATCHES = 10  # DockQ's own homology re-check tolerance (see README 6.5)
N_WORKERS            = None    # complexes scored concurrently; None -> min(8, cpu_count)
# =============================================================================


def build_config(a: argparse.Namespace) -> Config:
    return Config(
        combfold_base=a.combfold_base,
        ref_cache=a.ref_cache,
        out_dir=a.out_dir,
        uniprot_seq_csv=a.uniprot_csv,
        usalign_bin=a.usalign,
        dockq_bin=a.dockq,
        score_all_clusters=not a.top_cluster_only,
        same_protein_identity=a.same_protein_identity,
        homolog_identity=a.homolog_identity,
        outlier_cycles=a.outlier_cycles,
        outlier_cutoff=a.outlier_cutoff,
        save_primary_structures=a.save_structures,
        dockq_allowed_mismatches=a.dockq_allowed_mismatches,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Score CombFold assemblies vs reference PDBs.")
    p.add_argument("--mapping", default=MAPPING_FILE)
    p.add_argument("--combfold-base", default=COMBFOLD_BASE)
    p.add_argument("--uniprot-csv", default=UNIPROT_CSV)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--ref-cache", default=REF_CACHE,
                   help="persistent reference-PDB cache dir, one subfolder per PDB id "
                        f"(default: {REF_CACHE})")
    p.add_argument("--usalign", default=USALIGN_BIN)
    p.add_argument("--dockq", default=DOCKQ_BIN)
    p.add_argument("--top-cluster-only", action="store_true",
                   default=not SCORE_ALL_CLUSTERS)
    p.add_argument("--same-protein-identity", type=float, default=SAME_PROTEIN_IDENT)
    p.add_argument("--homolog-identity", type=float, default=HOMOLOG_IDENT)
    p.add_argument("--outlier-cycles", type=int, default=OUTLIER_CYCLES)
    p.add_argument("--outlier-cutoff", type=float, default=OUTLIER_CUTOFF)
    p.add_argument("--dockq-allowed-mismatches", type=int, default=DOCKQ_ALLOWED_MISMATCHES,
                   help="DockQ's own homology re-check tolerance in residues (default 10; "
                        "0 = DockQ's stricter built-in default). See README 6.5.")
    p.add_argument("--save-structures", action="store_true", default=SAVE_STRUCTURES)
    p.add_argument("--only", default="", help="comma-separated complex_ac subset")
    p.add_argument("--no-json", action="store_true", help="skip per-complex JSON")
    p.add_argument("--n-workers", type=int, default=N_WORKERS,
                   help="complexes scored concurrently (default: min(8, cpu_count)). "
                        "Each complex is independent (own temp dir, own JSON file, "
                        "lock-protected shared caches) so this only changes wall-clock "
                        "time, never any metric. Use --n-workers 1 for the original "
                        "strictly-serial behavior.")
    a = p.parse_args(argv)

    cfg = build_config(a)
    if not os.path.exists(a.mapping):
        p.error(f"mapping file not found: {a.mapping}")
    seqs = UniProtSequences(a.uniprot_csv if os.path.exists(a.uniprot_csv) else None,
                            cache_dir=os.path.join(cfg.ref_cache, "_uniprot"))
    only = [x.strip() for x in a.only.split(",") if x.strip()] or None

    summary, chain, iface = run_batch(cfg, a.mapping, seqs,
                                      complexes=only, save_json=not a.no_json,
                                      n_workers=a.n_workers)

    n_scored = int((summary["status"] == "scored").sum()) if not summary.empty else 0
    n_rows = len(summary)
    print(f"[combfold_eval] wrote outputs to {os.path.abspath(cfg.out_dir)}")
    print(f"[combfold_eval] {n_scored}/{n_rows} (complex,cluster,form) rows scored; "
          f"{len(chain)} chain rows, {len(iface)} interface rows")
    if not summary.empty:
        bad = summary[summary["status"] != "scored"]
        if not bad.empty:
            print("[combfold_eval] non-scored rows (status):")
            for _, r in bad.iterrows():
                print(f"    {r['complex_ac']} / {r.get('pdb_id','')}: {r['status']} "
                      f"{('- ' + r['flags']) if r.get('flags') else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
