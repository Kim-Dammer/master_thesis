#!/usr/bin/env python3
"""
Run US-align (pool vs reference) for homomultimers where:
  - n_chains > 2
  - pool has a CombFold output
  - pair has NO CombFold output

US-align output is written to:
  {uniprot_id}x{n_chains}_pool_output/us_align/
"""

import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm import tqdm

from procompa import get_project_root

PRJ_ROOT = get_project_root()
DATA_DIR = PRJ_ROOT / "data"

CF_DIR      = DATA_DIR / "Pipeline/6_sixth_subset_homomultimers_pool_vs_pair/CombFold"
TSV_PATH    = DATA_DIR / "Pipeline/6_sixth_subset_homomultimers_pool_vs_pair/sixth_input_homomultimers_pool_vs_pair.tsv"
PDB_DIR     = DATA_DIR / "Homomultimer/pdb_files"
USALIGN_BIN = str(PRJ_ROOT / "tools/usalign/USalign/USalign")
OUT_PATH    = PRJ_ROOT / "scripts/homomultimer_pool_pair_comparison/homomultimers_pool_only_usalign.parquet"
LOG_PATH    = OUT_PATH.with_suffix(".log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── rebuild result table (same logic as notebook cell) ───────────────────────

def parse_confidence(conf_path: Path) -> list[tuple[str, float]]:
    return [
        (parts[0], float(parts[1]))
        for line in conf_path.read_text().strip().splitlines()
        if len(parts := line.rsplit(maxsplit=1)) == 2
    ]

def cluster_idx(path: str) -> int:
    return int(Path(path).stem.rsplit("_", 1)[-1])

def cf_stats(uniprot_id: str, n_chains: int, condition: str) -> dict:
    cols = [f"n_outputs_{condition}", f"conf_max_{condition}", f"cf_max_conf_path_{condition}"]
    null_row = dict.fromkeys(cols)

    out_dir   = CF_DIR / f"{uniprot_id}x{n_chains}_{condition}_output"
    if not out_dir.exists():
        return null_row

    conf_path = out_dir / "assembled_results" / "confidence.txt"
    if not conf_path.exists():
        return {**null_row, f"n_outputs_{condition}": 0}

    entries = parse_confidence(conf_path)
    if not entries:
        return {**null_row, f"n_outputs_{condition}": 0}

    max_conf  = max(e[1] for e in entries)
    best_path = min(
        (e[0] for e in entries if e[1] == max_conf),
        key=cluster_idx,
    )
    return {
        f"n_outputs_{condition}": len(entries),
        f"conf_max_{condition}":  max_conf,
        f"cf_max_conf_path_{condition}": best_path,
    }


def build_result_df() -> pl.DataFrame:
    df = (
        pl.read_csv(TSV_PATH, separator="\t", comment_prefix=None)
        .rename({"#Complex ac": "complex_ac"})
        .with_columns(pl.col("complex_ac").str.strip_chars_start("HOMO_").alias("uniprot_id"))
    )
    rows = [
        {**row, **cf_stats(row["uniprot_id"], row["n_chains"], "pool"),
                **cf_stats(row["uniprot_id"], row["n_chains"], "pair")}
        for row in df.iter_rows(named=True)
    ]
    return pl.DataFrame(rows)


# ── US-align helpers ──────────────────────────────────────────────────────────

def parse_usalign_stdout(text: str) -> dict:
    tm   = re.search(r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_2", text)
    rmsd = re.search(r"RMSD=\s*([\d.]+)", text)
    return {
        "tm_score": float(tm.group(1))   if tm   else None,
        "rmsd":     float(rmsd.group(1)) if rmsd else None,
    }

def run_usalign(pred_path: str, ref_path: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        USALIGN_BIN, pred_path, ref_path,
        "-mm", "1", "-ter", "1", "-mol", "prot",
        "-o", str(out_dir / "usalign"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    (out_dir / "usalign_stdout.txt").write_text(proc.stdout)
    return parse_usalign_stdout(proc.stdout)

def process_row(row: dict) -> dict:
    ac   = row["complex_ac"]
    pred = row["cf_max_conf_path_pool"]
    ref  = row["pdb_path"]
    # e.g. .../Q03148x3_pool_output/us_align/
    out_dir = Path(pred).parent.parent / "us_align"

    logger.info(f"{ac} [pool] starting — {Path(pred).parent.parent.name}")
    try:
        metrics = run_usalign(pred, ref, out_dir)
        logger.info(f"{ac} [pool] done — TM={metrics['tm_score']:.4f}, RMSD={metrics['rmsd']:.2f}")
        return {"complex_ac": ac, "tm_score_cpx_pool": metrics["tm_score"], "rmsd_cpx_pool": metrics["rmsd"]}
    except Exception as e:
        logger.warning(f"{ac} [pool] FAILED: {e}")
        return {"complex_ac": ac, "tm_score_cpx_pool": None, "rmsd_cpx_pool": None}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Building result DataFrame from TSV...")
    result = build_result_df()
    logger.info(f"  Total complexes: {result.height}")

    # filter: n_chains > 2, pool assembled, pair NOT assembled
    df = (
        result
        .filter(pl.col("n_chains") > 2)
        .filter(pl.col("cf_max_conf_path_pool").is_not_null())
        .filter(pl.col("cf_max_conf_path_pair").is_null())
        .with_columns(
            pdb_path=(
                pl.lit(str(PDB_DIR) + "/") +
                pl.col("pdb_id") + pl.lit("/") + pl.col("pdb_id") + pl.lit(".cif")
            )
        )
    )
    logger.info(f"  After filter (n_chains>2, pool✓, pair✗): {df.height} complexes")

    if df.height == 0:
        logger.warning("No complexes to process — exiting.")
        return

    rows = df.to_dicts()
    logger.info(f"Running US-align (pool only) for {df.height} complexes...")

    results = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(process_row, row): row["complex_ac"] for row in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="US-align", unit="complex"):
            ac = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                logger.error(f"{ac}: unhandled exception: {e}")

    results_df = pl.DataFrame(
        results,
        schema={
            "complex_ac":        pl.Utf8,
            "tm_score_cpx_pool": pl.Float64,
            "rmsd_cpx_pool":     pl.Float64,
        },
    )

    df_out = df.join(results_df, on="complex_ac", how="left")
    df_out.write_parquet(OUT_PATH)
    logger.info(f"Written {df_out.height} rows to {OUT_PATH}")
    logger.info("\n" + str(df_out.select(["complex_ac", "pdb_id", "n_chains",
                                          "tm_score_cpx_pool", "rmsd_cpx_pool"])))


if __name__ == "__main__":
    main()
