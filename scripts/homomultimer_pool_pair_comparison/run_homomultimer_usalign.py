#!/usr/bin/env python3
"""Run US-align (pool + pair vs reference) for all homomultimer complexes."""

import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm import tqdm
from procompa import get_project_root

PRJ_ROOT = get_project_root()


USALIGN_BIN = str(PRJ_ROOT / "tools/usalign/USalign/USalign")
CSV_PATH    = PRJ_ROOT / "scripts/homomultimer_poll_pair_comparison/homomultimers_pool_pair_outputs.csv"
OUT_PATH    = CSV_PATH.with_name("homomultimers_pool_pair_usalign.parquet")
LOG_PATH    = CSV_PATH.with_name("homomultimers_pool_pair_usalign.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),   # also print to stdout → visible in sbatch .out
    ],
)
logger = logging.getLogger(__name__)


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
    complex_ac = row["complex_ac"]
    result = {"complex_ac": complex_ac}

    for source in ("pool", "pair"):
        pred    = row[f"cf_max_conf_path_{source}"]
        ref     = row["pdb_path"]
        out_dir = Path(pred).parent.parent / "us_align"
        logger.info(f"{complex_ac} [{source}] starting — pred: {Path(pred).parent.parent.name}")
        try:
            metrics = run_usalign(pred, ref, out_dir)
            result[f"tm_score_cpx_{source}"] = metrics["tm_score"]
            result[f"rmsd_cpx_{source}"]     = metrics["rmsd"]
            logger.info(f"{complex_ac} [{source}] done — TM={metrics['tm_score']:.4f}, RMSD={metrics['rmsd']:.2f}")
        except Exception as e:
            logger.warning(f"{complex_ac} [{source}] FAILED: {e}")
            result[f"tm_score_cpx_{source}"] = None
            result[f"rmsd_cpx_{source}"]     = None

    return result


def main():
    df   = pl.read_csv(CSV_PATH)
    rows = df.to_dicts()
    logger.info(f"Running US-align for {len(rows)} complexes ({len(rows) * 2} alignments total).")
    logger.info(f"Log: {LOG_PATH}")

    results = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(process_row, row): row["complex_ac"] for row in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="US-align", unit="complex"):
            complex_ac = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                logger.error(f"{complex_ac}: unhandled exception: {e}")

    results_df = pl.DataFrame(
        results,
        schema={
            "complex_ac":        pl.Utf8,
            "tm_score_cpx_pool": pl.Float64,
            "rmsd_cpx_pool":     pl.Float64,
            "tm_score_cpx_pair": pl.Float64,
            "rmsd_cpx_pair":     pl.Float64,
        },
    )
    df_out = df.join(results_df, on="complex_ac", how="left")
    df_out.write_parquet(OUT_PATH)
    logger.info(f"Written to {OUT_PATH}")
    logger.info("\n" + str(df_out.select(["complex_ac", "tm_score_cpx_pool", "rmsd_cpx_pool", "tm_score_cpx_pair", "rmsd_cpx_pair"])))


if __name__ == "__main__":
    main()