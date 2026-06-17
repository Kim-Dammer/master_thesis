import polars as pl
from pathlib import Path
from procompa import get_project_root

PRJ_ROOT = get_project_root()
data_dir = PRJ_ROOT / "data"

IntAct = pl.read_csv(
    data_dir / "IntAct/yeast_miTab_2.7.txt",
    separator="\t",
    has_header=True,
    ignore_errors=True,
    infer_schema_length=0,

)

#filter so it only includes complexes, where both interactors are from yeast (specifically S. cerevisiae)
IntAct_yeast_proteins = IntAct.filter(
    (IntAct["Taxid interactor A"].str.contains("559292"))
    & (IntAct["Taxid interactor B"].str.contains("559292")) #139114
    & (IntAct["Type(s) interactor A"].str.contains("protein"))
    & (IntAct["Type(s) interactor B"].str.contains("protein"))
)

#for some proteins uniprot Id is nor given
non_uniprot = IntAct_yeast_proteins.filter(~IntAct_yeast_proteins["#ID(s) interactor A"].str.contains("uniprot")
                                           | ~IntAct_yeast_proteins["ID(s) interactor B"].str.contains("uniprot"))

#mapping for ids, for which none uniprot ID is given
intact_uniprot_mapping = {
    "intact:EBI-7857625" : "uniprotkb:Q12064",
    "intact:EBI-8222677" : "uniprotkb:Q07790",
    "intact:EBI-8225396" : "uniprotkb:Q03361",
    "intact:EBI-8225414" : "uniprotkb:A0A023PXB9",
    "intact:EBI-8225486" : "uniprotkb:Q6B0Y7",
    "intact:EBI-7330403" : "uniprotkb:P40341",
    "intact:EBI-16145422" : "uniprotkb:P0CG63",
}

# Apply the mapping to both Interactor A and Interactor B columns
IntAct_yeast_proteins_uni_prot = IntAct_yeast_proteins.with_columns([
    pl.col("#ID(s) interactor A").replace_strict(intact_uniprot_mapping, default=pl.col("#ID(s) interactor A")),
    pl.col("ID(s) interactor B").replace_strict(intact_uniprot_mapping, default=pl.col("ID(s) interactor B")
    )
])

IntAct_yeast_proteins_uni_prot = IntAct_yeast_proteins_uni_prot.with_columns([
    pl.col("#ID(s) interactor A").str.replace_all("uniprotkb:", ""),
    pl.col("ID(s) interactor B").str.replace_all("uniprotkb:", "")
])

IntAct_yeast_proteins_uni_prot.write_csv(Path.cwd() / "Dataframes/Yeast_Map/IntAct_yeast_proteins_uni_prot.csv")