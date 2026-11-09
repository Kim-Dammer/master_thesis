#!/usr/bin/env python3
"""Run US-align for every CombFold output model (all COPI stoichiometries) against 5A1U.

The reference is 5A1U with the Arf1 chain (chain H) removed, since Arf1 is not part
of the predicted coatomer. TM-scores are reported normalized by the reference length.
"""

import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from Bio.PDB import MMCIFIO, MMCIFParser, Select
from procompa import get_project_root
from tqdm import tqdm

PRJ_ROOT = get_project_root()

USALIGN_BIN = str(PRJ_ROOT / "tools/usalign/USalign/USalign")
COMBFOLD_DIR = PRJ_ROOT / "data/Pipeline/24_CPX_1652_pae_trimmed/CombFold"
REFERENCE_PDB_ID = "5A1U"
REFERENCE_CIF = PRJ_ROOT / f"data/pdb_mmcif/{REFERENCE_PDB_ID}.cif"
EXCLUDED_REFERENCE_CHAINS = {"H"}  # Arf1, not part of the predicted COPI complex
FILTERED_REFERENCE_CIF = COMBFOLD_DIR / (
    f"{REFERENCE_PDB_ID}_without_chain_{'_'.join(sorted(EXCLUDED_REFERENCE_CHAINS))}.cif"
)

RESULTS_DIR = PRJ_ROOT / "notebooks/ccprofiler_files_ym"
RESULTS_CSV = RESULTS_DIR / f"combfold_outputs_vs_{REFERENCE_PDB_ID}_usalign.csv"
LOG_PATH = RESULTS_DIR / f"combfold_outputs_vs_{REFERENCE_PDB_ID}_usalign.log"

STOICHIOMETRY_DIR_SUFFIX = "_pool_output"
MODEL_FILE_PATTERN = re.compile(r"output_clustered_(\d+)\.pdb")
SUBUNIT_COPY_PATTERN = re.compile(r"([A-Z0-9]+)x(\d+)")
N_WORKERS = 10

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ChainExcluder(Select):
    def __init__(self, excluded_chain_ids: set[str]):
        self.excluded_chain_ids = excluded_chain_ids

    def accept_chain(self, chain) -> bool:
        return chain.id not in self.excluded_chain_ids


def write_reference_without_chains(source_cif: Path, target_cif: Path, excluded_chain_ids: set[str]) -> None:
    reference_structure = MMCIFParser(QUIET=True).get_structure(REFERENCE_PDB_ID, str(source_cif))
    first_model = next(iter(reference_structure))
    residues_per_chain = {chain.id: len(chain) for chain in first_model}
    logger.info(f"{REFERENCE_PDB_ID} chains (id: n_residues) {residues_per_chain}")

    missing_chain_ids = excluded_chain_ids - set(residues_per_chain)
    assert not missing_chain_ids, f"Chains to exclude not found in {REFERENCE_PDB_ID}: {missing_chain_ids}"

    cif_writer = MMCIFIO()
    cif_writer.set_structure(reference_structure)
    cif_writer.save(str(target_cif), ChainExcluder(excluded_chain_ids))
    kept_chain_ids = sorted(set(residues_per_chain) - excluded_chain_ids)
    logger.info(f"Reference written to {target_cif}, kept chains {kept_chain_ids}")


def parse_usalign_stdout(usalign_stdout: str) -> dict:
    tm_norm_reference = re.search(r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_2", usalign_stdout)
    tm_norm_model = re.search(r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_1", usalign_stdout)
    alignment_stats = re.search(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)", usalign_stdout)
    assert tm_norm_reference and tm_norm_model and alignment_stats, (
        f"Could not parse US-align output\n{usalign_stdout}"
    )
    return {
        "tm_score_norm_by_reference": float(tm_norm_reference.group(1)),
        "tm_score_norm_by_model": float(tm_norm_model.group(1)),
        "aligned_length": int(alignment_stats.group(1)),
        "rmsd": float(alignment_stats.group(2)),
    }


def run_usalign(model_pdb: Path, reference_cif: Path, alignment_dir: Path) -> dict:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    usalign_cmd = [
        USALIGN_BIN, str(model_pdb), str(reference_cif),
        "-mm", "1", "-ter", "1", "-mol", "prot",
        "-o", str(alignment_dir / "usalign"),
    ]
    usalign_process = subprocess.run(usalign_cmd, capture_output=True, text=True)
    if usalign_process.returncode != 0:
        raise RuntimeError(
            f"US-align failed for {model_pdb} (exit {usalign_process.returncode})\n{usalign_process.stderr}"
        )
    (alignment_dir / "usalign_stdout.txt").write_text(usalign_process.stdout)
    return parse_usalign_stdout(usalign_process.stdout)


def collect_alignment_jobs() -> list[dict]:
    stoichiometry_dirs = sorted(COMBFOLD_DIR.glob(f"*{STOICHIOMETRY_DIR_SUFFIX}"))
    assert stoichiometry_dirs, f"No *{STOICHIOMETRY_DIR_SUFFIX} folders found in {COMBFOLD_DIR}"

    alignment_jobs = []
    for stoichiometry_dir in stoichiometry_dirs:
        stoichiometry = stoichiometry_dir.name.removesuffix(STOICHIOMETRY_DIR_SUFFIX)
        model_pdbs = sorted(
            (stoichiometry_dir / "assembled_results").glob("output_clustered_*.pdb"),
            key=lambda model_path: int(MODEL_FILE_PATTERN.fullmatch(model_path.name).group(1)),
        )
        assert model_pdbs, f"No output_clustered_*.pdb in {stoichiometry_dir / 'assembled_results'}"
        for model_pdb in model_pdbs:
            model_index = int(MODEL_FILE_PATTERN.fullmatch(model_pdb.name).group(1))
            alignment_jobs.append({
                "stoichiometry": stoichiometry,
                "model_index": model_index,
                "model_pdb": model_pdb,
                "alignment_dir": stoichiometry_dir / f"output_{model_index}_us_align",
            })
    logger.info(f"Found {len(stoichiometry_dirs)} stoichiometries and {len(alignment_jobs)} models in total.")
    return alignment_jobs


def align_one_model(alignment_job: dict) -> dict:
    alignment_metrics = run_usalign(alignment_job["model_pdb"], FILTERED_REFERENCE_CIF, alignment_job["alignment_dir"])
    logger.info(
        f"{alignment_job['stoichiometry']} model {alignment_job['model_index']} "
        f"TM(ref)={alignment_metrics['tm_score_norm_by_reference']:.4f}, RMSD={alignment_metrics['rmsd']:.2f}"
    )
    subunit_copies = {
        f"copies_{uniprot_ac}": int(n_copies)
        for uniprot_ac, n_copies in SUBUNIT_COPY_PATTERN.findall(alignment_job["stoichiometry"])
    }
    return {
        "stoichiometry": alignment_job["stoichiometry"],
        "model_index": alignment_job["model_index"],
        **subunit_copies,
        **alignment_metrics,
        "model_pdb": str(alignment_job["model_pdb"]),
        "alignment_dir": str(alignment_job["alignment_dir"]),
    }


def main():
    write_reference_without_chains(REFERENCE_CIF, FILTERED_REFERENCE_CIF, EXCLUDED_REFERENCE_CHAINS)
    alignment_jobs = collect_alignment_jobs()

    alignment_records = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(align_one_model, alignment_job) for alignment_job in alignment_jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="US-align", unit="model"):
            alignment_records.append(future.result())  # raises on any failed alignment

    usalign_scores = pl.DataFrame(alignment_records).sort(["stoichiometry", "model_index"])
    usalign_scores.write_csv(RESULTS_CSV)
    logger.info(f"Written to {RESULTS_CSV}")
    with pl.Config(tbl_rows=-1, fmt_str_lengths=100):
        logger.info("\n" + str(usalign_scores.select(
            ["stoichiometry", "model_index", "tm_score_norm_by_reference", "tm_score_norm_by_model", "rmsd"]
        )))


if __name__ == "__main__":
    main()
