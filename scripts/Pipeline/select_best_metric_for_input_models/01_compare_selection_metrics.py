#!/usr/bin/env python3
"""
Compare 5 AF3-model selection metrics for the pairwise contacts required by
the complexes in a ComplexPortal-format TSV -- WITHOUT running CombFold.

Metrics: ranking_score, pae (chain_pair_pae_min, ascending), iptm
(chain_pair_iptm_corrected), ptm_avg (plain mean of the two chains'
chain_ptm -- individual fold confidence, NOT interface quality), ptm_weighted
(same, weighted by each chain's sequence length).

For every required contact (homodimer self-pair / heteropair), writes which
AF3 model each metric would pick (batch_id, seed, sample, af3_id1/2,
chain_id1/2, ...) plus a wide side-by-side comparison. Stats only -- no
SQLite lookups, no PDBs written.

ASSUMPTIONS (edit the CONFIG block if wrong):
  - Stoichiometry per complex = classify_molecules() on the TSV's raw "(n)"
    tags, "(0)" -> 1, same rule 01_combfold_with_stoic.py uses for true_spec.
    No STOIC re-prediction.
  - Complexes with a hard-blocker token are skipped entirely.
  - Homodimer contacts: input_type == SOURCE ("pool") only, no fallback.
    Confirmed 2026-09-24: candidate pairs come tagged input_type in
    {"pair", "pool"}; only "pool" is wanted, for both homo and hetero.
  - Heteropair contacts: always input_type == 'pool'.
  - A contact having NO candidate model at all is expected (logged as
    "missing") -- but once a candidate row exists, every field this script
    needs from it (chain_ptm, seq_len, chain index) is asserted present:
    that's a data-integrity bug, not a normal gap, so it fails loudly.
  - Tie-breaking: when >1 candidate is exactly tied at the best value for a
    metric, the one with the LOWEST `sample` value is picked. Every row
    belonging to a pair_id where such a tie occurred gets `other_options =
    True`, so a tie-broken pick is visible rather than indistinguishable
    from a clean win. This is deliberate and deterministic -- NOT the old
    `rank(method="ordinal")` behavior, which broke ties by incidental
    DataFrame row order.

CHANGES vs the original version (2026-09-24), all called out explicitly so
nothing is silently different:
  - FIX: `load_candidates()`'s pair_id is now `.str.to_lowercase()`'d, to
    match `required_contacts_from_tsv()`'s lower-cased pair_id.
    `diagnose_pair_id_matching()` reports at runtime whether this changes
    anything for your data.
  - ADD: diagnose_pair_id_matching(), diagnose_metric_coverage_and_ties(),
    diagnose_pairwise_agreement() -- see each docstring.
  - ADD: deterministic tie-breaking (lowest `sample` wins) + `other_options`
    column, via a single shared helper `_rank_and_flag_ties()` used by both
    `select_per_metric()` and `annotate_all_candidates()`, so the two can't
    drift apart from each other.
  - ADD: annotate_all_candidates() -- every pool-input candidate model for
    every required contact, with a per-metric rank + other_options column,
    written to all_candidates_annotated.parquet, for manual spot-checking.
"""
from __future__ import annotations

import re
from pathlib import Path

import af3io.input
import polars as pl

# --- CONFIG -----------------------------------------------------------------
TSV_PATH = Path("/cluster/project/beltrao/kdammer/master_thesis/data/Pipeline/14_CF_multiple_models/fourteenth_input_benchmark_and_unknown_pdb_structures_for_cf_with_more_models.tsv")
POOLED_PPI_DB = Path("/cluster/work/beltrao/jjaenes/25.12_pooled-ppi-yeast/data-26.08")
OUTPUT_DIR = Path("/cluster/project/beltrao/kdammer/master_thesis/scripts/Pipeline/select_best_metric_for_input_models")
SOURCE = "pool"          # homodimer input_type, no fallback -- confirmed correct
TOP_N_MODELS = 1         # top-N models kept per contact per metric

MOLECULES_COL = "Identifiers (and stoichiometry) of molecules in complex"
COMPLEX_AC_COL = "#Complex ac"
NAME_COL = "Recommended name"

METRICS = {
    "ranking_score": {"col": "ranking_score", "descending": True},
    "pae": {"col": "chain_pair_pae_min", "descending": False},
    "iptm": {"col": "chain_pair_iptm_corrected", "descending": True},
    "ptm_avg": {"col": "ptm_avg", "descending": True},
    "ptm_weighted": {"col": "ptm_weighted", "descending": True},
}
ID_COLS = ["model_uid", "af3_id1", "af3_id2", "input_type", "input_name",
           "batch_id", "seed", "sample", "chain_id1", "chain_id2"]
VALUE_COLS = ["ranking_score", "chain_pair_iptm_corrected", "chain_pair_pae_min",
              "ptm_avg", "ptm_weighted"]

# --- classify_molecules, copied from 01_combfold_with_stoic.py so this ------
# --- script has no dependency on it / procompa. Keep in sync if that changes.
_UNIPROT_ACCESSION = r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
_SUBUNIT_RE = re.compile(rf"^((?:{_UNIPROT_ACCESSION})(?:-PRO_\d+)?)\((\d+)\)$")
_PRO_TOKEN_RE = re.compile(rf"^({_UNIPROT_ACCESSION})-PRO_(\d+)$")


def _strip_count(tok: str) -> tuple[str, int | None]:
    m = re.match(r"^(.*)\((\d+)\)$", tok)
    return (m.group(1), int(m.group(2))) if m else (tok, None)


def classify_molecules(molecules: str) -> tuple[dict[str, int], bool, list[str]]:
    counts: dict[str, int] = {}
    blockers: list[str] = []
    n_subunits = n_unknown = 0
    for tok in [t.strip() for t in str(molecules).split("|") if t.strip()]:
        base, _ = _strip_count(tok)
        m = _SUBUNIT_RE.match(tok)
        if m:
            uid = m.group(1)
            if _PRO_TOKEN_RE.match(uid):
                blockers.append(tok)
                continue
            raw = int(m.group(2))
            n_subunits += 1
            if raw == 0:
                n_unknown += 1
                counts[uid] = counts.get(uid, 0) + 1
            else:
                counts[uid] = counts.get(uid, 0) + raw
        elif base.upper().startswith("CHEBI:") or base.upper().startswith("URS"):
            continue
        else:
            blockers.append(tok)
    all_unknown = n_subunits > 0 and n_unknown == n_subunits
    return counts, all_unknown, blockers


# --- Step 1: required pairwise contacts, one row per (complex, contact) ----
def required_contacts_from_tsv(tsv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(tsv_path, separator="\t", quote_char=None, infer_schema_length=0)
    assert MOLECULES_COL in df.columns and COMPLEX_AC_COL in df.columns, \
        f"TSV missing required columns. Found: {df.columns}"

    rows = []
    n_skipped_blocked = 0
    for r in df.iter_rows(named=True):
        counts, _, blockers = classify_molecules(r[MOLECULES_COL])
        if blockers:
            n_skipped_blocked += 1
            continue
        if not counts:
            continue
        proteins = sorted(counts)
        assert NAME_COL in r, f"TSV row missing expected column {NAME_COL!r}."
        common = {"complex_ac": r[COMPLEX_AC_COL], "complex_name": r[NAME_COL],
                  "spec": ",".join(f"{p}({counts[p]})" for p in proteins)}
        for p in proteins:
            if counts[p] >= 2:
                rows.append({**common, "pair_type": "homo", "protein1": p, "protein2": p,
                             "pair_id": f"{p.lower()}__{p.lower()}"})
        for i, p1 in enumerate(proteins):
            for p2 in proteins[i + 1:]:
                lo, hi = sorted([p1.lower(), p2.lower()])
                rows.append({**common, "pair_type": "hetero", "protein1": p1, "protein2": p2,
                             "pair_id": f"{lo}__{hi}"})
    print(f"[contacts] {n_skipped_blocked} complexes skipped (hard-blocker token).")
    return pl.DataFrame(rows)


# --- Diagnostic: is the pair_id join actually working? ----------------------
def diagnose_pair_id_matching(contacts: pl.DataFrame) -> None:
    """Checks whether af3_id casing silently drops candidates. Prints
    findings; only raises if something is internally inconsistent."""
    wanted = set(contacts["pair_id"])
    raw = (
        pl.scan_parquet(POOLED_PPI_DB / "summary_models.parquet")
        .select(
            (pl.min_horizontal(["af3_id1", "af3_id2"]) + "__" +
             pl.max_horizontal(["af3_id1", "af3_id2"])).alias("pair_id_raw")
        )
        .unique()
        .collect()["pair_id_raw"]
    )
    raw_set = set(raw)
    lowered_set = {s.lower() for s in raw_set}
    n_raw = len(wanted & raw_set)
    n_lowered = len(wanted & lowered_set)
    print(f"[diag/pair_id] {len(wanted)} unique contact pair_ids wanted")
    print(f"[diag/pair_id] matched against RAW (as-stored) af3_id casing: {n_raw}")
    print(f"[diag/pair_id] matched after lower-casing the af3_id side: {n_lowered}")
    assert n_lowered >= n_raw, "unexpected: lower-casing REDUCED the match count -- investigate."
    if n_lowered > n_raw:
        print(f"[diag/pair_id] *** CASE MISMATCH CONFIRMED: {n_lowered - n_raw} additional "
              f"contacts only match once af3_id is lower-cased. load_candidates() handles this. ***")
    else:
        print("[diag/pair_id] no casing effect detected.")


# --- Step 2: candidate AF3 models + all 5 metrics' raw ingredients ---------
_CHAIN_INDEX = {label: i for i, label in zip(range(2000), af3io.input.enumerate_chains())}


def _chain_ptm_at(chain_ptm: list[float] | None, chain_id: str) -> float:
    assert chain_id in _CHAIN_INDEX, \
        f"chain_id={chain_id!r} not in _CHAIN_INDEX (only built up to 2000 chains)."
    idx = _CHAIN_INDEX[chain_id]
    assert chain_ptm is not None and idx < len(chain_ptm), \
        f"chain_ptm missing or too short for chain_id={chain_id!r} (idx={idx})"
    return chain_ptm[idx]


def load_candidates(contacts: pl.DataFrame) -> pl.DataFrame:
    homo_keys = set(contacts.filter(pl.col("pair_type") == "homo")["pair_id"])
    hetero_keys = set(contacts.filter(pl.col("pair_type") == "hetero")["pair_id"])

    models = (
        pl.scan_parquet(POOLED_PPI_DB / "summary_models.parquet")
        .with_columns(
            (pl.min_horizontal(["af3_id1", "af3_id2"]) + "__" +
             pl.max_horizontal(["af3_id1", "af3_id2"])).str.to_lowercase().alias("pair_id")
        )
        .filter(pl.col("pair_id").is_in(homo_keys | hetero_keys))
        .filter(
            (pl.col("pair_id").is_in(homo_keys) & (pl.col("input_type") == SOURCE)) |
            (pl.col("pair_id").is_in(hetero_keys) & (pl.col("input_type") == "pool"))
        )
        .with_columns(
            pl.format("{}|{}|{}|{}", "input_name", "input_type", "seed", "sample").alias("model_uid")
        )
        .collect()
    )

    confidences = (
        pl.scan_parquet(POOLED_PPI_DB / "summary_confidences.parquet")
        .select("input_name", "input_type", "seed", "sample", "chain_ptm")
        .collect()
    )
    dupe_counts = confidences.group_by(["input_name", "input_type", "seed", "sample"]).len()
    assert (dupe_counts["len"] == 1).all(), (
        "summary_confidences.parquet has duplicate (input_name, input_type, seed, sample) keys -- "
        "a left join on this table would silently fan out candidate rows."
    )
    proteins = pl.scan_parquet(POOLED_PPI_DB / "proteins.parquet").select("af3_id", "seq_len").collect()

    n_models = models.height
    df = models.join(confidences, on=["input_name", "input_type", "seed", "sample"], how="left")
    assert df.height == n_models, (
        f"join with summary_confidences.parquet changed row count "
        f"({n_models} -> {df.height}) -- fan-out from a duplicate key."
    )

    df = df.join(proteins.rename({"af3_id": "af3_id1", "seq_len": "seq_len1"}), on="af3_id1", how="left")
    assert df.height == n_models, (
        f"join with proteins.parquet (af3_id1) changed row count "
        f"({n_models} -> {df.height}) -- af3_id1 not unique in proteins.parquet."
    )

    df = df.join(proteins.rename({"af3_id": "af3_id2", "seq_len": "seq_len2"}), on="af3_id2", how="left")
    assert df.height == n_models, (
        f"join with proteins.parquet (af3_id2) changed row count "
        f"({n_models} -> {df.height}) -- af3_id2 not unique in proteins.parquet."
    )

    assert df["seq_len1"].null_count() == 0 and df["seq_len2"].null_count() == 0, \
        "af3_id not found in proteins.parquet for some candidate rows -- data-integrity bug."

    ptm1 = df.select(pl.struct("chain_ptm", "chain_id1").map_elements(
        lambda s: _chain_ptm_at(s["chain_ptm"], s["chain_id1"]), return_dtype=pl.Float64
    ).alias("ptm1"))["ptm1"]
    ptm2 = df.select(pl.struct("chain_ptm", "chain_id2").map_elements(
        lambda s: _chain_ptm_at(s["chain_ptm"], s["chain_id2"]), return_dtype=pl.Float64
    ).alias("ptm2"))["ptm2"]

    return df.with_columns(
        ptm1.alias("_ptm1"), ptm2.alias("_ptm2"),
    ).with_columns(
        ((pl.col("_ptm1") + pl.col("_ptm2")) / 2).alias("ptm_avg"),
        ((pl.col("_ptm1") * pl.col("seq_len1") + pl.col("_ptm2") * pl.col("seq_len2"))
         / (pl.col("seq_len1") + pl.col("seq_len2"))).alias("ptm_weighted"),
    )


# --- Diagnostic: per-metric candidate-pool coverage --------------------------
def diagnose_metric_coverage_and_ties(candidates: pl.DataFrame) -> None:
    print("\n[diag/coverage] non-null candidate rows and pair_id coverage per metric:")
    for name, spec in METRICS.items():
        col = spec["col"]
        n_non_null = candidates[col].drop_nulls().len()
        n_pairs = candidates.filter(pl.col(col).is_not_null())["pair_id"].n_unique()
        print(f"  {name:14s} col={col:28s} non-null_rows={n_non_null:6d} pair_ids_covered={n_pairs}")
    print("  (uneven coverage across metrics means each is choosing from a different pool -- "
          "that alone can explain a lot of 'disagreement' without any of it being a bug.)")


# --- Shared ranking + deterministic tie-break, used by selection AND dump ---
def _rank_and_flag_ties(candidates: pl.DataFrame, name: str, col: str, descending: bool) -> pl.DataFrame:
    """Rank candidates by `col` within each pair_id (best = rank 1). Ties at
    the best value are broken by ascending `sample` (lowest sample wins),
    not by incidental row order. `{name}_other_options` is True on every
    row of a pair_id where more than one candidate was tied at the best
    value, so a tie-broken pick stays visible.
    Returns one row per candidate: model_uid, pair_id, {name}_rank,
    {name}_other_options.
    """
    ranked = (
        candidates.filter(pl.col(col).is_not_null())
        .sort(["pair_id", col, "sample"], descending=[False, descending, False])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("pair_id").alias("rank")
        )
    )
    check = ranked.group_by("pair_id").agg(
        pl.col("rank").n_unique().alias("n_unique"), pl.len().alias("n")
    )
    assert (check["n_unique"] == check["n"]).all(), (
        f"rank is not unique within some pair_id group for metric {name!r} -- ranking logic broken."
    )

    best_val = ranked.filter(pl.col("rank") == 1).select("pair_id", pl.col(col).alias("_best_val"))
    n_tied = (
        ranked.join(best_val, on="pair_id", how="inner")
        .filter(pl.col(col) == pl.col("_best_val"))
        .group_by("pair_id")
        .len()
        .rename({"len": "_n_tied"})
    )
    ranked = ranked.join(n_tied, on="pair_id", how="left")
    assert ranked["_n_tied"].null_count() == 0, (
        f"tie-count join dropped rows for metric {name!r} -- every pair_id has a rank==1 row "
        f"by construction, so this should never be null."
    )
    return ranked.select(
        "model_uid", "pair_id",
        pl.col("rank").alias(f"{name}_rank"),
        (pl.col("_n_tied") > 1).alias(f"{name}_other_options"),
    )


# --- Step 3: per-metric top-N selection -------------------------------------
def select_per_metric(candidates: pl.DataFrame, contacts: pl.DataFrame) -> dict[str, pl.DataFrame]:
    contact_lookup = contacts.select("pair_id", "pair_type", "protein1", "protein2").unique()
    results = {}
    for name, spec in METRICS.items():
        rank_tbl = _rank_and_flag_ties(candidates, name, spec["col"], spec["descending"])
        top = (
            candidates.join(
                rank_tbl.filter(pl.col(f"{name}_rank") <= TOP_N_MODELS),
                on=["model_uid", "pair_id"], how="inner",
            )
            .rename({f"{name}_rank": "model_rank", f"{name}_other_options": "other_options"})
        )
        found = (
            contact_lookup.join(top, on="pair_id", how="inner")
            .select(["pair_id", "pair_type", "protein1", "protein2"]
                     + ID_COLS + VALUE_COLS + ["model_rank", "other_options"])
            .with_columns(pl.lit("found").alias("status"))
        )
        missing = (
            contact_lookup.filter(~pl.col("pair_id").is_in(set(top["pair_id"])))
            .with_columns(pl.lit("missing").alias("status"))
        )
        results[name] = pl.concat([found, missing], how="diagonal_relaxed") \
            .sort(["pair_type", "protein1", "protein2", "model_rank"])
        if missing.height:
            print(f"[{name}] {missing.height} required contact(s) with no eligible model.")
        n_tied = found.filter(pl.col("other_options"))["pair_id"].n_unique() if found.height else 0
        if n_tied:
            print(f"[{name}] {n_tied} pick(s) were tie-broken by lowest sample (other_options=True).")
    return results


# --- Diagnostic: full NxN pairwise top-1 agreement, not just vs ranking_score
def diagnose_pairwise_agreement(selections: dict[str, pl.DataFrame]) -> None:
    names = list(METRICS)
    top1 = {
        m: df.filter((pl.col("status") == "found") & (pl.col("model_rank") == 1))
              .select("pair_id", "model_uid")
        for m, df in selections.items()
    }
    print("\n[diag/agreement] pairwise top-1 agreement matrix (fraction of shared contacts "
          "where both metrics pick the SAME model_uid):")
    print(" " * 14 + "".join(f"{n:>14s}" for n in names))
    for a in names:
        cells = []
        for b in names:
            joined = top1[a].join(top1[b], on="pair_id", suffix="_b")
            n_shared = joined.height
            if n_shared == 0:
                cells.append(f"{'n/a':>13s} ")
                continue
            n_agree = joined.filter(pl.col("model_uid") == pl.col("model_uid_b")).height
            cells.append(f"{n_agree / n_shared:13.0%} ")
        print(f"{a:14s}" + "".join(cells))
    print("  (ptm_avg/ptm_weighted measure per-monomer fold confidence, not interface quality -- "
          "low agreement with the other three is expected. Low agreement AMONG ranking_score/"
          "pae/iptm themselves would be the more surprising thing.)")


# --- Full per-pair candidate dump, for manual spot-checking -----------------
def annotate_all_candidates(candidates: pl.DataFrame, contacts: pl.DataFrame) -> pl.DataFrame:
    """Every pool-input candidate model for every required contact, with a
    per-metric rank + other_options column attached -- NOT reduced to
    top-N. One row per (pair_id, model). Filter to one pair_id to manually
    verify that e.g. the row with iptm_rank == 1 really is what
    selection_iptm.csv reports as picked, and that other_options matches.
    """
    dup = candidates.group_by("model_uid").len().filter(pl.col("len") > 1)
    assert dup.height == 0, (
        f"model_uid is not unique in candidates ({dup.height} duplicated) -- "
        f"can't safely attach per-metric ranks by joining on model_uid."
    )

    out = candidates
    for name, spec in METRICS.items():
        rank_tbl = _rank_and_flag_ties(candidates, name, spec["col"], spec["descending"]).drop("pair_id")
        n_before = out.height
        out = out.join(rank_tbl, on="model_uid", how="left")
        assert out.height == n_before, (
            f"join for {name}_rank changed row count ({n_before} -> {out.height}) -- "
            f"ranking join fanned out unexpectedly."
        )

    contact_lookup = contacts.select("pair_id", "pair_type", "protein1", "protein2").unique()
    out = contact_lookup.join(out, on="pair_id", how="inner")

    n_no_candidates = contact_lookup.join(
        out.select("pair_id").unique(), on="pair_id", how="anti"
    ).height
    if n_no_candidates:
        print(f"[all_candidates] {n_no_candidates} contact(s) have zero pool candidates at all -- "
              f"not present in this dump (should match 'missing' status for every metric above).")

    return out.sort(["pair_type", "protein1", "protein2", "ranking_score_rank"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contacts = required_contacts_from_tsv(TSV_PATH)
    print(f"[contacts] {contacts.height} (complex, contact) rows; "
          f"{contacts['pair_id'].n_unique()} unique contacts, "
          f"{contacts['complex_ac'].n_unique()} complexes.")
    contacts.write_csv(OUTPUT_DIR / "required_contacts_by_complex.csv")

    diagnose_pair_id_matching(contacts)

    candidates = load_candidates(contacts)
    print(f"[candidates] {candidates.height} candidate AF3 samples for "
          f"{candidates['pair_id'].n_unique()} unique contacts.")

    diagnose_metric_coverage_and_ties(candidates)

    selections = select_per_metric(candidates, contacts)
    for name, df in selections.items():
        df.write_csv(OUTPUT_DIR / f"selection_{name}.csv")
        print(f"[{name}] wrote {df.height} rows.")

    diagnose_pairwise_agreement(selections)

    all_candidates = annotate_all_candidates(candidates, contacts)
    all_candidates.write_parquet(OUTPUT_DIR / "all_candidates_annotated.parquet")
    print(f"[all_candidates] wrote {all_candidates.height} rows (all pool candidates x rank + "
          f"other_options per metric) -> {OUTPUT_DIR / 'all_candidates_annotated.parquet'}")

    # Wide side-by-side: top-1 model per metric per contact, + differ flags vs ranking_score.
    top1 = {m: df.filter(pl.col("model_rank") == 1) for m, df in selections.items()}
    wide = top1["ranking_score"].select("pair_id", "pair_type", "protein1", "protein2")
    for m, df in top1.items():
        wide = wide.join(
            df.select("pair_id",
                      pl.col("model_uid").alias(f"{m}_model_uid"),
                      pl.col("input_name").alias(f"{m}_input_name"),
                      pl.col("sample").alias(f"{m}_sample"),
                      *[pl.col(c).alias(f"{m}_{c}") for c in VALUE_COLS],
                      pl.col("other_options").alias(f"{m}_other_options"),
                      pl.col("status").alias(f"{m}_status")),
            on="pair_id", how="left",
        )
    for m in list(METRICS)[1:]:
        wide = wide.with_columns(
            (pl.col("ranking_score_model_uid") != pl.col(f"{m}_model_uid")).alias(f"ranking_score_vs_{m}_differ")
        )
    wide.write_csv(OUTPUT_DIR / "selection_comparison_wide.csv")
    print(f"[compare] wrote wide comparison -> {OUTPUT_DIR / 'selection_comparison_wide.csv'}")


if __name__ == "__main__":
    main()