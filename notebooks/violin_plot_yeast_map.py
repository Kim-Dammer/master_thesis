import polars as pl
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import math
from procompa import get_project_root, get_data_dir

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"
data_dir_YM = get_data_dir() / "26.03_yeast.MAP"


def read_data():
    test_complex  = pl.read_csv(
    data_dir_YM / "complex_portal_scerevisiae_559292_reduced_complexes_20251007.test.txt", separator = "\t", has_header=False, truncate_ragged_lines=True).with_columns(
    pl.col("column_1").str.split(" ").list.len().alias("n_subcomplexes") )#236

    train_complex =pl.read_csv(
        data_dir_YM / "complex_portal_scerevisiae_559292_reduced_complexes_20251007.train.txt", separator = "\t", has_header=False, truncate_ragged_lines=True).with_columns(
        pl.col("column_1").str.split(" ").list.len().alias("n_subcomplexes") )#234

    predicted_complex = pl.read_csv(
        data_dir_YM / "yeast.MAP_complexes_wConfidenceScores_wGenenames_total779_20251214.csv", separator = ",", has_header = True).with_columns(
        pl.col("UniProt_ACCs").str.split(" ").list.len().alias("n_subcomplexes")
    ) #779

    db_complexes = pl.read_csv( data_dir / "Complex_Portal/Saccharomyces cerevisiae_ComplexTab.tsv", truncate_ragged_lines=True , separator = "\t")
    db_complexes = db_complexes.select(["#Complex ac","Identifiers (and stoichiometry) of molecules in complex", "Evidence Code"])
    complex_YM_complexes = db_complexes.with_columns( pl.col("Identifiers (and stoichiometry) of molecules in complex")
      .str.replace_all(r"\(\d+\)", "")  
      .str.replace_all(r"\|", " ")       
      .alias("cleaned_entries")
    )

    test_train_complexes = pl.concat([
    train_complex.with_columns(split=pl.lit("train")),
    test_complex.with_columns(split=pl.lit("test")) 
    ]).with_columns(
        true_list = pl.col("column_1").str.split(" "),
        size_true = pl.col("column_1").str.split(" ").list.len()
    ).with_row_index("true_id")

    return test_train_complexes, complex_YM_complexes, predicted_complex

if __name__ == "__main__":
    test_train_complexes, complex_YM_complexes, predicted_complex = read_data()

        
    #to ensure stuff like CHEBI:29035 doen't prevent from finding a match
    uniprot_pattern = r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b"
    # 1. prepare lists
    tt = test_train_complexes.with_columns(
        pl.col("column_1")
        .str.extract_all(uniprot_pattern)
        .list.sort()
        .alias("plist")
    )

    ym = complex_YM_complexes.with_columns(
        pl.col("cleaned_entries")
        .str.extract_all(uniprot_pattern)   # <-- already list[str]
        .list.sort()
        .alias("plist")
    )
    # 2. match existing
    tt_matched = tt.join(
        ym.select(["plist", "#Complex ac", "Evidence Code"]),
        on="plist",
        how="left"
    ).with_columns(
        pl.col("#Complex ac").is_not_null().alias("on_complex_db")
    )

    # 3. add missing complexes
    new_rows = (
        ym.join(tt.select("plist"), on="plist", how="anti")
        .with_columns([
            (pl.int_range(pl.len(), dtype=pl.UInt32) + (tt_matched["true_id"].max() + 1)).alias("true_id"),
            pl.col("plist").list.join(" ").alias("column_1"),
            pl.col("plist").alias("true_list"),

            pl.col("plist").list.len().alias("n_subcomplexes"),

            pl.lit("complex_Db").alias("split"),

            pl.col("plist").list.len().alias("size_true"),

            pl.lit(True).alias("on_complex_db"),
        ])
        .select(tt_matched.columns)
    )



    # 4. concat
    all_true_complexes = pl.concat([tt_matched, new_rows])

    all_true_complexes = all_true_complexes.with_columns(
        confidence_score = pl.when(pl.col("Evidence Code").str.contains("ECO:0000353"))
                        .then(pl.lit(5.0))
                        .when(pl.col("Evidence Code").str.contains("ECO:0005547"))
                        .then(pl.lit(3.0))
                        .when(pl.col("Evidence Code").str.contains("ECO:0005546"))
                        .then(pl.lit(4.0))
    ).rename({"Evidence Code": "evidence_code"})

    # 2. Prepare predicted complexes
    pred_prep = predicted_complex.with_columns(
        pred_list = pl.col("UniProt_ACCs").str.split(" "),
        size_pred = pl.col("UniProt_ACCs").str.split(" ").list.len()
    ).with_row_index("pred_id")

    #  Overlapping Pairs including metrics for anything with >= 1 overlapping protein
    overlaps_raw = (
        all_true_complexes.explode("true_list")
        .join(
            pred_prep.explode("pred_list"), 
            left_on="true_list", 
            right_on="pred_list"
        )
        .group_by(["true_id", "pred_id"])
        .agg([
            pl.all().first(),
            pl.len().alias("match_count")
        ])
        .with_columns(
            exact_size_match = (pl.col("size_true") == pl.col("size_pred")),
            jaccard_similarity = pl.col("match_count") / (pl.col("size_true") + pl.col("size_pred") - pl.col("match_count"))
        )
    )

    overlaps = overlaps_raw.select([
        "column_1", "UniProt_ACCs", "size_true", "size_pred", 
        "match_count", "jaccard_similarity", "split", "exact_size_match", "#Complex ac" , "confidence_score",
        "ComplexConfidence"
    ])


    # Completely Unmatched Predictions 

    unmatched_preds = (
        pred_prep.join(overlaps_raw, on="pred_id", how="anti")
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("#Complex ac"),
            
            column_1 = pl.lit(None, dtype=pl.String),
            size_true = pl.lit(None, dtype=pl.Int64),
            match_count = pl.lit(0, dtype=pl.Int64),
            jaccard_similarity = pl.lit(0.0, dtype=pl.Float64),
            split = pl.lit(None, dtype=pl.String),
            exact_size_match = pl.lit(False, dtype=pl.Boolean),
            confidence_score = pl.lit(None, dtype=pl.Float64),
            ComplexConfidence = pl.lit(None, dtype=pl.String)
        )
        .select([
            "column_1", "UniProt_ACCs", "size_true", "size_pred", 
            "match_count", "jaccard_similarity", "split", "exact_size_match", "#Complex ac", "confidence_score", "ComplexConfidence"
        ])
    )


    complete_overlap_by_complex = pl.concat([overlaps, unmatched_preds], how="vertical_relaxed")
    complete_overlap_by_complex = complete_overlap_by_complex.rename({"column_1": "true_complex", "UniProt_ACCs": "predicted_complex"})


    # 1. Load the 779 predicted complexes and generate the "CPX_N" IDs based on row order
    complex_mapping = (
        pl.read_csv(data_dir_YM / "yeast.MAP_complexes_wConfidenceScores_wGenenames_total779_20251214.csv")
        .with_row_index("row_num", offset=1) # Creates a 1-indexed count (1, 2, 3...)
        .select([
            pl.col("UniProt_ACCs").alias("predicted_complex"), # Match column name of your main LDF
            pl.format("CPX_{}", pl.col("row_num")).alias("predicted_complex_id")
        ])
    )

    # 2. Join the IDs into your main dataframe
    complete_overlap_by_complex = complete_overlap_by_complex.join(
        complex_mapping, 
        on="predicted_complex", 
        how="left"
    )

    #gett all theoretical possible pairins
    unique_complexes = complete_overlap_by_complex.unique(subset=["predicted_complex_id"])

    # 2. Explode the unique complexes into individual proteins
    exploded = unique_complexes.select(
        "predicted_complex_id", 
        protein=pl.col("predicted_complex").str.split(" ")
    ).explode("protein")

    # get unique, undirected pairs
    all_possible_pairs = (
        exploded.join(exploded, on="predicted_complex_id", suffix="_B")
        .rename({"protein": "protein_A"})
        .filter(pl.col("protein_A") < pl.col("protein_B"))     # (drops self-pairs and B-A duplicates)
        .unique() 
    )

    # 1. Sort by Jaccard similarity descending, then keep only the FIRST (highest) entry per complex
    best_overlap_by_complex = (
        complete_overlap_by_complex
        .sort("jaccard_similarity", descending=True)
        .unique(subset=["predicted_complex_id"], keep="first")
    )

    # 2. Join with all possible pairs (this will no longer explode the rows!)
    complex_table_with_pairs = best_overlap_by_complex.join(
        all_possible_pairs,
        on="predicted_complex_id",
        how="inner"
    )

    # 3. Create the display column for the pairs
    complex_table_with_pairs = complex_table_with_pairs.with_columns(
        pl.format("{}-{}", pl.col("protein_A"), pl.col("protein_B")).alias("complex_pairs")
    )

    yeastmap_predicted_complex_pairs = pl.read_csv( data_dir_YM / "yeastmapV3_complex_pairs_20260308.pairsWProb", separator = "\t", has_header = False)

    pair_scores = yeastmap_predicted_complex_pairs.rename({"column_1": "p1", "column_2": "p2", "column_3": "probability_score"})

    standardized_pairs = (
        pair_scores.with_columns(
            p_min = pl.min_horizontal("p1", "p2"),
            p_max = pl.max_horizontal("p1", "p2")
        )
        # Deduplicate the 481 bidirectional pairs (since both directions have identical scores)
        .unique(subset=["p_min", "p_max"])
        .select(["p_min", "p_max", "probability_score"])
    )

    # 2. Map the scores directly into your main complex table using a left join
    complex_pairs_with_scores = complex_table_with_pairs.join(
        standardized_pairs,
        left_on=["protein_A", "protein_B"],
        right_on=["p_min", "p_max"],
        how="left"
    )

    # 1. Label rows into Groups A, B, and C using Polars conditional logic
    complex_pairs_with_scores_filtered = complex_pairs_with_scores.filter(pl.col("size_pred")< 30)
    grouped_df = complex_pairs_with_scores_filtered.with_columns(
        pl.when(pl.col("exact_size_match") == True)
        .then(pl.lit("Same as in complex"))

        .when((pl.col("exact_size_match") == False) & (pl.col("match_count") == 1))
        .then(pl.lit("Pred 1 prot overlap with complex"))
        
        .when((pl.col("exact_size_match") == False) & (pl.col("jaccard_similarity") >= 0.5))
        .then(pl.lit("Pred at least 50% overlap with complex"))
        
        .when(pl.col("match_count") == 0)
        .then(pl.lit("None of the proteins are in complex"))
        
        .otherwise(pl.lit(None))
        .alias("Match Category")
    ).filter(pl.col("Match Category").is_not_null())

    counts = grouped_df.group_by("Match Category").len()
    counts_dict = dict(zip(counts["Match Category"], counts["len"]))

    order = [
        "Same as in complex",
        "Pred 1 prot overlap with complex",
        "Pred at least 50% overlap with complex",
        "None of the proteins are in complex",
    ]
    labels_with_n = [f"{cat}\n(n={counts_dict.get(cat, 0):,})" for cat in order]

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=grouped_df.to_pandas(),
        x="Match Category",
        y="probability_score",
        order=order,
        palette="Set2",
    )
    ax.set_xticklabels(labels_with_n)

    plt.title(
        "YeastMAP Probability Scores Across Match Groups",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("YeastMAP Probability Score", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(Path.cwd() / "Figures/Yeast_Map/yeastmap_probscores_across_matchgroups.png", dpi=500)
    

    complex_pairs_with_scores = complex_pairs_with_scores.with_columns(
        pair_on_complex_db=(
            pl.col("true_complex").str.split(" ").list.contains(pl.col("protein_A"))
            & pl.col("true_complex").str.split(" ").list.contains(pl.col("protein_B"))
        ).fill_null(False),
        predicted_known_percentage=(pl.col("match_count") / pl.col("size_pred")).round(2)
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    sizes = [2, 3, 4, 5]
    palette_colors = {True: "green", False: "darkred"}

    for i, (ax, size) in enumerate(zip(axes, sizes)):
        subset = complex_pairs_with_scores.filter(
            pl.col("size_pred") == size
        ).to_pandas()

        n_complexes = subset["predicted_complex_id"].nunique()
        expected_order = [round(match / size, 2) for match in range(0, size + 1)]

        # ensure both hue values exist for every x-category so split violins are always centered on the tick
        dummy_rows = []
        for xval in expected_order:
            for hue_val in [False, True]:
                if not ((subset["predicted_known_percentage"] == xval) & 
                        (subset["pair_on_complex_db"] == hue_val)).any():
                    dummy_rows.append({
                        "predicted_known_percentage": xval,
                        "pair_on_complex_db": hue_val,
                        "probability_score": float("nan"),
                    })
        if dummy_rows:
            subset = pd.concat([subset, pd.DataFrame(dummy_rows)], ignore_index=True)

        sns.violinplot(
            data=subset,
            x="predicted_known_percentage",
            y="probability_score",
            hue="pair_on_complex_db",
            split=True,
            hue_order=[False, True],
            order=expected_order,
            palette=palette_colors,
            inner="quartile",
            ax=ax,
            legend=(i == 3),
            density_norm="width",
        )

        ax.set_title(f"Complex Size: {size}\n(n={n_complexes} complexes)", fontsize=12)
        ax.set_xlabel("Overlap with COMPLEX / predicted complex size", fontsize=9 )
        ax.set_ylabel("Pairwise Probability Score", fontsize= 10)
        ax.grid(True, linestyle="--", alpha=0.5)
    leg = axes[3].get_legend()
    axes[3].legend_.remove()
    axes[0].legend(handles=leg.legend_handles, labels=[t.get_text() for t in leg.get_texts()], title="Pair on Complex", loc="lower left")

    plt.savefig(Path.cwd() / "Figures/Yeast_Map/yeastmap_probscores_by_predicted_knownpercentage_and_paironcomplex.png", dpi=500)

    rows_false, rows_true = [], []
    for size in sizes:
        subset = complex_pairs_with_scores.filter(pl.col("size_pred") == size).to_pandas()
        expected_order = [round(match / size, 2) for match in range(0, size + 1)]
        row_f, row_t = [f"Size {size}"], [f"Size {size}"]
        for xval in expected_order:
            n_false = ((subset["predicted_known_percentage"] == xval) & (subset["pair_on_complex_db"] == False)).sum()
            n_true  = ((subset["predicted_known_percentage"] == xval) & (subset["pair_on_complex_db"] == True)).sum()
            row_f.append(str(n_false))
            row_t.append(str(n_true))
        rows_false.append(row_f)
        rows_true.append(row_t)

    # Build combined rows: for each size, one red row then one green row
    max_cols = max(len(r) for r in rows_false)  # size + n_overlaps
    col_labels = [""] + [f"{round(m/s,2)}" for s in [sizes[-1]] for m in range(0, s+1)]  # placeholder, overridden per size

    fig, axes = plt.subplots(1, len(sizes), figsize=(12, 2.5),
                            gridspec_kw={"width_ratios": [s + 1 for s in sizes]})
    for ax in axes:
        ax.axis("off")

    for i, size in enumerate(sizes):
        ax = axes[i]
        expected_order = [round(match / size, 2) for match in range(0, size + 1)]
        col_labels = [""] + [str(x) for x in expected_order]

        cell_text  = [rows_false[i][1:], rows_true[i][1:]]  # drop the "Size X" label
        row_labels = ["not on DB", "on DB"]
        cell_colors = [
            ["#ffe5e5"] * len(expected_order),
            ["#e5f5e5"] * len(expected_order),
        ]
        row_label_colors = [["#c0392b"], ["#27ae60"]]

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=expected_order,
            cellLoc="center",
            loc="center",
            cellColours=cell_colors,
            rowColours=["#c0392b", "#27ae60"],
        )
        table.auto_set_column_width(list(range(len(expected_order))))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        # Header styling
        for j in range(len(expected_order)):
            table[0, j].set_facecolor("#2c2c2c")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Row label text color white
        for r in [1, 2]:
            table[r, -1].set_text_props(color="white", fontweight="bold")

        ax.set_title(f"Complex Size: {size}\n(n={complex_pairs_with_scores.filter(pl.col('size_pred')==size).to_pandas()['predicted_complex_id'].nunique()} complexes)", fontsize=10)

    fig.suptitle("Datapoints per violin half ", fontsize=12)
    plt.tight_layout()
    plt.savefig(Path.cwd() / "Figures/Yeast_Map/table_probscores_by_predicted_knownpercentage_and_paironcomplex_with_counts.png", dpi=500, bbox_inches="tight")

    counts = complex_pairs_with_scores.group_by("pair_on_complex_db").len()

    n_false = counts.filter(pl.col("pair_on_complex_db") == False).select("len").item() if False in counts["pair_on_complex_db"] else 0
    n_true = counts.filter(pl.col("pair_on_complex_db") == True).select("len").item() if True in counts["pair_on_complex_db"] else 0

    # 2. Add dummy column and convert to Pandas
    df_for_plot = complex_pairs_with_scores.with_columns(
        pl.lit("All Data").alias("group")
    ).to_pandas()

    plt.figure(figsize=(6, 6))


    ax = sns.violinplot(
        data=df_for_plot, 
        x="group",                 
        y="probability_score", 
        hue="pair_on_complex_db", 
        split=True,
        hue_order=[False, True],
        palette={True: "green", False: "darkred"},
        inner="quartile"
    )

    legend_labels = [f"False (n = {n_false:,})", f"True (n = {n_true:,})"]


    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=legend_labels, title="Pair on Complex", loc="upper left")

    plt.title("YeastMAP Pairwise Probability Scores")
    plt.xlabel("") 
    plt.ylabel("YeastMAP Probability Score")


    plt.tight_layout()
    plt.savefig(Path.cwd() / "Figures/Yeast_Map/yeastmap_probscores_by_paironcomplex.png", dpi=500)
    complex_pairs_with_scores.write_csv(Path.cwd() / "Dataframes/Yeast_Map/yeastmap_complex_pairs_with_scores_incl_db.csv")