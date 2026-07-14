#!/usr/bin/env python3
"""
match_complexes_to_pdb.py

Match Complex Portal complexes against PDB entries using an mmseqs search
of complex-member proteins (and, if you ran it that way, their homologues)
against pdb_seqres.

Single-threshold model: any mmseq hit with identity_percent >= --threshold
counts as "this protein is present in that PDB entry" -- no separate
exact/homologue tiers. Default threshold is 30.

For each PDB entry we also need how many *distinct* proteins it contains
(chains with identical sequence = one protein, e.g. homo-oligomers)
so we can report how many "extra" proteins a PDB brings beyond your
complex. That comes from pdb_seqres.txt.

For every complex we report:
  - EXACT match: a PDB entry whose distinct-protein count equals the
    complex's protein count, AND every complex protein is matched
    (same size, full coverage, no extra, nothing missing).
  - For every complex WITHOUT an exact match: the single BEST match --
    ranked by (1) how many of the complex's proteins are found
    [coverage_frac, higher better], (2) how few extra unrelated
    proteins the PDB entry brings [extra_proteins, lower better].
    This naturally covers both of your cases:
      * PDB has only 5/6 of the complex's proteins (partial coverage)
      * PDB has all of the complex's proteins plus extra ones (superset)
    -- the ranking just picks whichever PDB entry is closest to your
    complex overall.
  - All other candidate PDB matches per complex are also written out,
    in case you want to browse alternatives instead of just the top pick.

Assumptions:
  * mmseq_results.hit_pdb_id is "<pdbid>_<chain>" (e.g. "6sv4_zY") --
    split on the FIRST "_" to recover the 4-char pdb id.
  * protein_id / Cleaned Identifiers matched case-insensitively.
  * "distinct proteins in a PDB entry" = number of unique sequences
    among mol:protein chains in that entry.
  * A complex protein counts as matched in a pdb entry if ANY of its
    mmseq hits against that pdb clears --threshold (you said: above
    threshold = good enough, no further tiering needed).

Outputs (csv) written to --out-dir:
  - summary_per_complex.csv       one row per complex: status + best match
  - exact_matches.csv             complex <-> pdb, exact matches
  - best_match_per_complex.csv    single best match for complexes w/o exact
  - all_candidate_matches.csv     every (complex, pdb) pair that matched >=1 protein
  - pdb_protein_counts.parquet    cached per-pdb distinct-protein counts

Usage:
  uv run 4_match_complexes_to_pdb.py \
      --complexes /cluster/project/beltrao/kdammer/master_thesis/data/Complex_Portal/Sc_ComplexTab_cleaned.csv \ 
      --mmseq /cluster/project/beltrao/kdammer/master_thesis/scripts/mmseq_homology_match/mmseqs/mmseqs_run_max_sensitivity/results/mmseqs_identity_similarity_max_sensitivity.parquet\
      --pdb-seqres /cluster/project/beltrao/kdammer/master_thesis/data/pdb/pdb_seqres.txt \
      --threshold 30 \
      --out-dir /cluster/project/beltrao/kdammer/master_thesis/data/complete_complex_pdb_mapping_v2/homology_pdb_mapping/
"""
import argparse
from pathlib import Path

import polars as pl


# --------------------------------------------------------------------------
# 1. Parse pdb_seqres.txt -> per-pdb distinct protein counts
# --------------------------------------------------------------------------
def parse_pdb_seqres(path: str) -> pl.DataFrame:
    """Stream-parse a pdb_seqres fasta file into a long DataFrame:
    pdb_id, chain, moltype, length, seq
    """
    pdb_ids, chains, moltypes, lengths, seqs = [], [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line[0] == ">":
                parts = line[1:].split(None, 3)  # id, mol:x, length:N, description...
                idpart = parts[0]
                if "_" in idpart:
                    pdbid, chain = idpart.split("_", 1)
                else:
                    pdbid, chain = idpart, ""
                mol = parts[1].split(":", 1)[1] if len(parts) > 1 and ":" in parts[1] else ""
                try:
                    length = int(parts[2].split(":", 1)[1]) if len(parts) > 2 and ":" in parts[2] else None
                except ValueError:
                    length = None
                pdb_ids.append(pdbid.lower())
                chains.append(chain)
                moltypes.append(mol)
                lengths.append(length)
            else:
                seqs.append(line.strip())
    if len(seqs) != len(pdb_ids):
        raise ValueError(
            f"Parsed {len(pdb_ids)} headers but {len(seqs)} sequence lines -- "
            "pdb_seqres format assumption (one sequence line per header) violated."
        )
    return pl.DataFrame(
        {"pdb_id": pdb_ids, "chain": chains, "moltype": moltypes, "length": lengths, "seq": seqs}
    )


def build_pdb_reference(pdb_seqres_path: str, cache_dir: Path):
    """Returns (pdb_counts, chain_clusters):
      pdb_counts: pdb_id -> n_distinct_proteins_pdb
      chain_clusters: hit_key ("pdbid_chain", lowercased) -> cluster_id, where
        cluster_id is shared by all chains with an identical sequence within
        the same pdb entry (e.g. all copies of a homo-oligomer subunit).
    Cached to cache_dir so re-runs skip re-parsing pdb_seqres.txt.
    """
    counts_path = cache_dir / "pdb_protein_counts.parquet"
    clusters_path = cache_dir / "pdb_chain_clusters.parquet"
    if counts_path.exists() and clusters_path.exists():
        return pl.read_parquet(counts_path), pl.read_parquet(clusters_path)

    raw = parse_pdb_seqres(pdb_seqres_path)
    protein_raw = raw.filter(pl.col("moltype") == "protein").with_columns(
        pl.concat_str([pl.col("pdb_id"), pl.lit("_"), pl.col("chain")]).str.to_lowercase().alias("hit_key")
    )
    # cluster_id: one id per unique (pdb_id, seq) pair -- shared by every chain
    # of that pdb entry with an identical sequence.
    clustered = protein_raw.with_columns(
        pl.concat_str([pl.col("pdb_id"), pl.lit(":"), pl.col("seq").hash().cast(pl.Utf8)]).alias("cluster_id")
    )
    chain_clusters = clustered.select(["hit_key", "cluster_id"]).unique(subset=["hit_key"])

    counts = (
        clustered.unique(subset=["pdb_id", "cluster_id"])
        .group_by("pdb_id")
        .agg(pl.len().alias("n_distinct_proteins_pdb"))
    )

    counts.write_parquet(counts_path)
    chain_clusters.write_parquet(clusters_path)
    return counts, chain_clusters


# --------------------------------------------------------------------------
# 2. Load + normalize complexes and mmseq results
# --------------------------------------------------------------------------
def load_complexes_long(path: str) -> pl.DataFrame:
    df = pl.read_csv(path) if path.endswith(".csv") else pl.read_parquet(path)
    id_col = "Cleaned Identifiers"
    complex_col = "#Complex ac"
    long = (
        df.select([complex_col, id_col, "n_proteins"])
        .with_columns(
            pl.col(id_col)
            .str.split(" ")
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        )
        .explode(id_col)
        .rename({id_col: "protein_id", complex_col: "complex_ac"})
        .with_columns(pl.col("protein_id").str.to_lowercase())
    )
    return long


def load_mmseq_filtered(path: str, min_identity: float) -> pl.LazyFrame:
    scan = pl.scan_csv(path) if path.endswith(".csv") else pl.scan_parquet(path)
    return (
        scan.with_columns(
            [
                pl.col("protein_id").str.to_lowercase().alias("protein_id"),
                pl.col("hit_pdb_id")
                .str.splitn("_", 2)
                .struct.field("field_0")
                .str.to_lowercase()
                .alias("pdb_id"),
            ]
        )
        .filter(pl.col("identity_percent") >= min_identity)
        .select(["protein_id", "pdb_id", "hit_pdb_id", "identity_percent"])
    )


def best_hit_per_protein_pdb(mmseq_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Collapse to one representative hit_pdb_id per (protein_id, pdb_id) --
    the highest-identity hit -- used to report which query:hit pair caused
    each protein to count as 'present' in a given pdb entry."""
    return mmseq_lf.group_by(["protein_id", "pdb_id"]).agg(
        pl.col("hit_pdb_id").sort_by(pl.col("identity_percent"), descending=True).first().alias("hit_pdb_id")
    )


# --------------------------------------------------------------------------
# 3. Core matching logic
# --------------------------------------------------------------------------
def build_complex_pdb_agg(
    complexes_long: pl.DataFrame,
    mmseq_lf: pl.LazyFrame,
    complex_sizes: pl.DataFrame,
    pdb_counts: pl.DataFrame,
    chain_clusters: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (complex_ac, pdb_id) that has >=1 matching protein,
    with n_matched / coverage / extra_proteins / matching_hits.

    Each target "cluster" (= one physical protein/subunit in the pdb entry,
    possibly present as several identical chains) can only be claimed by
    ONE complex protein. If several complex proteins independently hit the
    same cluster (e.g. close paralogs both matching the same pdb chain),
    only the alphabetically-first protein_id counts towards n_matched; the
    rest are still shown, in brackets, in matching_hits, e.g.:
        "p07249:1mnm_A(p11746)"
    This guarantees n_matched can never exceed n_distinct_proteins_pdb for
    that pdb entry (so extra_proteins is always >= 0).

    matching_hits format: "protein_id:hit_pdb_id" per counted match, with
    any conflicting protein_ids that hit the same target in parentheses.
    """
    rep = best_hit_per_protein_pdb(mmseq_lf)

    matches = (
        complexes_long.lazy()
        .join(rep, on="protein_id", how="inner")
        .with_columns(pl.col("hit_pdb_id").str.to_lowercase().alias("hit_key"))
        .join(chain_clusters.lazy(), on="hit_key", how="left")
        # fall back to the raw hit_key as cluster_id if the chain wasn't
        # found in pdb_seqres (e.g. naming mismatch) -- keeps that hit
        # usable without letting it silently disappear
        .with_columns(pl.coalesce([pl.col("cluster_id"), pl.col("hit_key")]).alias("cluster_id"))
    )

    # one row per (complex_ac, pdb_id, cluster_id): pick the alphabetically
    # first protein_id as the counted match, keep the rest as alternates
    cluster_groups = (
        matches.group_by(["complex_ac", "pdb_id", "cluster_id"])
        .agg(
            [
                pl.col("protein_id").sort().alias("proteins"),
                pl.col("hit_pdb_id").sort_by(pl.col("protein_id")).alias("hit_ids"),
            ]
        )
        .with_columns(
            [
                pl.col("proteins").list.first().alias("primary_protein"),
                pl.col("hit_ids").list.first().alias("primary_hit_pdb_id"),
                pl.col("proteins").list.slice(1).alias("alt_proteins"),
            ]
        )
        .with_columns(
            pl.when(pl.col("alt_proteins").list.len() > 0)
            .then(
                pl.concat_str(
                    [
                        pl.col("primary_protein"),
                        pl.lit(":"),
                        pl.col("primary_hit_pdb_id"),
                        pl.lit("("),
                        pl.col("alt_proteins").list.join(","),
                        pl.lit(")"),
                    ]
                )
            )
            .otherwise(
                pl.concat_str([pl.col("primary_protein"), pl.lit(":"), pl.col("primary_hit_pdb_id")])
            )
            .alias("hit_pair")
        )
    )

    agg = cluster_groups.group_by(["complex_ac", "pdb_id"]).agg(
        [
            pl.len().alias("n_matched"),
            pl.col("hit_pair").sort_by(pl.col("primary_protein")).str.join(",").alias("matching_hits"),
        ]
    )

    out = (
        agg.join(complex_sizes.lazy(), on="complex_ac", how="left")
        .join(pdb_counts.lazy(), on="pdb_id", how="left")
        .with_columns(
            [
                (pl.col("n_matched") / pl.col("n_proteins")).alias("coverage_frac"),
                (pl.col("n_distinct_proteins_pdb") - pl.col("n_matched")).alias("extra_proteins"),
            ]
        )
        .collect()
    )
    return out


def classify(agg: pl.DataFrame, complex_sizes: pl.DataFrame):
    """Returns (exact_df, best_match_df, all_candidates_df, no_match_df).

    Rule: if a pdb entry has MORE distinct proteins than the complex, it is
    only a valid candidate when it covers ALL of the complex's proteins
    (a true superset). A bigger pdb with only partial overlap (e.g. 5/8 of
    your complex, in an 11-protein structure) is dropped entirely -- it's
    not a meaningful "best match" candidate, just noise from an unrelated
    larger assembly that happens to share a few proteins.
    Pdb entries that are the same size or smaller than the complex are
    still allowed to be partial matches (e.g. 3/4 proteins present).
    """
    valid = agg.filter(
        ~(
            (pl.col("n_distinct_proteins_pdb") > pl.col("n_proteins"))
            & (pl.col("n_matched") < pl.col("n_proteins"))
        )
    )

    exact_df = valid.filter(
        (pl.col("n_matched") == pl.col("n_proteins"))
        & (pl.col("n_distinct_proteins_pdb") == pl.col("n_proteins"))
    ).sort(["complex_ac", "pdb_id"])
    exact_complexes = set(exact_df["complex_ac"].to_list())

    candidates = valid.filter(~pl.col("complex_ac").is_in(exact_complexes)).sort(
        ["complex_ac", "coverage_frac", "extra_proteins"],
        descending=[False, True, False],
    )

    best_match_df = candidates.group_by("complex_ac", maintain_order=True).first()

    matched_or_exact = exact_complexes | set(best_match_df["complex_ac"].to_list())
    no_match = complex_sizes.filter(~pl.col("complex_ac").is_in(matched_or_exact))

    return exact_df, best_match_df, candidates, no_match


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complexes", required=True, help="unmapped complexes csv/parquet")
    ap.add_argument("--mmseq", required=True, help="mmseq_results csv/parquet (can be large)")
    ap.add_argument("--pdb-seqres", required=True, help="pdb_seqres.txt")
    ap.add_argument(
        "--threshold", type=float, default=30.0,
        help="identity_percent threshold (%%) above which a hit counts as this protein being present (default: 30)",
    )
    ap.add_argument("--out-dir", default="results")
    ap.add_argument(
        "--pdb-cache", default=None,
        help="optional directory to cache parsed pdb_seqres tables, reused on later runs (default: --out-dir)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.pdb_cache) if args.pdb_cache else out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] parsing pdb_seqres.txt ...")
    pdb_counts, chain_clusters = build_pdb_reference(args.pdb_seqres, cache_dir)
    print(f"      {pdb_counts.height} pdb entries with protein chains")

    print("[2/4] loading complexes ...")
    complexes_long = load_complexes_long(args.complexes)
    complex_sizes = complexes_long.select(["complex_ac", "n_proteins"]).unique()
    print(f"      {complex_sizes.height} complexes, {complexes_long.height} complex-protein rows")

    print(f"[3/4] loading + filtering mmseq_results at >= {args.threshold}% identity ...")
    mmseq_lf = load_mmseq_filtered(args.mmseq, args.threshold)

    print("      joining + aggregating ...")
    agg = build_complex_pdb_agg(complexes_long, mmseq_lf, complex_sizes, pdb_counts, chain_clusters)

    print("[4/4] classifying + writing outputs ...")
    exact_df, best_match_df, all_candidates_df, no_match_df = classify(agg, complex_sizes)

    exact_df.write_csv(out_dir / "exact_matches.csv")
    best_match_df.write_csv(out_dir / "best_match_per_complex.csv")
    all_candidates_df.write_csv(out_dir / "all_candidate_matches.csv")

    exact_complexes = set(exact_df["complex_ac"].to_list())

    # one representative row per exact-matched complex (first pdb_id alphabetically),
    # reshaped to the same columns as best_match_df so both can feed the same
    # best_match_* summary columns -- plus the full list of exact pdb ids,
    # since a complex can in principle have more than one exact-match structure.
    exact_all_ids = (
        exact_df.group_by("complex_ac")
        .agg(pl.col("pdb_id").sort().str.join(",").alias("exact_match_pdb_ids"))
    )
    exact_representative = (
        exact_df.sort(["complex_ac", "pdb_id"])
        .group_by("complex_ac", maintain_order=True)
        .first()
        .select(
            [
                "complex_ac",
                pl.col("pdb_id").alias("best_match_pdb"),
                pl.col("n_matched").alias("best_match_n_matched"),
                pl.col("coverage_frac").alias("best_match_coverage_frac"),
                pl.col("n_distinct_proteins_pdb").alias("best_match_pdb_size"),
                pl.col("extra_proteins").alias("best_match_extra_proteins"),
                pl.col("matching_hits").alias("best_match_hits"),
            ]
        )
    )
    best_match_cols = pl.concat(
        [
            exact_representative,
            best_match_df.select(
                [
                    "complex_ac",
                    pl.col("pdb_id").alias("best_match_pdb"),
                    pl.col("n_matched").alias("best_match_n_matched"),
                    pl.col("coverage_frac").alias("best_match_coverage_frac"),
                    pl.col("n_distinct_proteins_pdb").alias("best_match_pdb_size"),
                    pl.col("extra_proteins").alias("best_match_extra_proteins"),
                    pl.col("matching_hits").alias("best_match_hits"),
                ]
            ),
        ],
        how="vertical",
    )

    summary = (
        complex_sizes.with_columns(pl.col("complex_ac").is_in(exact_complexes).alias("has_exact_pdb"))
        .join(best_match_cols, on="complex_ac", how="left")
        .join(exact_all_ids, on="complex_ac", how="left")
        .sort("complex_ac")
    )
    summary.write_csv(out_dir / "summary_per_complex.csv")

    n_exact = len(exact_complexes)
    n_best = best_match_df.height
    n_none = no_match_df.height
    n_total = complex_sizes.height

    print(
        f"""
Done. {n_total} complexes total.
  exact match (same size, full coverage) : {n_exact}
  best (partial or superset) match found : {n_best}
  no match at all (>= {args.threshold}% identity)   : {n_none}

Outputs written to {out_dir}/
"""
    )


if __name__ == "__main__":
    main()
