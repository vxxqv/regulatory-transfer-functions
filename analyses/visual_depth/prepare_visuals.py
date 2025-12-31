"""Prepare source tables for locus, network, convergence, and volcano figures."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/visual_depth/results"
H5AD = ROOT / "data/raw/gwt/GWCD4i.DE_stats.h5ad"
EXEMPLARS = [("GATA3", "Rest", "amplified"), ("RELB", "Stim48hr", "buffered"), ("NFAT5", "Rest", "context_switching")]


def extract_rows(data, phenotypes: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    genes = pd.DataFrame({"feature_id": data.var_names.astype(str)})
    genes["gene_name"] = data.var["gene_name"].astype(str).to_numpy() if "gene_name" in data.var else genes["feature_id"]
    extracted = {}
    for gene, state, role in EXEMPLARS:
        row = phenotypes[(phenotypes.target_contrast_gene_name == gene) & (phenotypes.culture_condition == state)].iloc[0]
        index = data.obs_names.get_loc(str(row["index"]))
        table = genes.copy()
        table["log_fc"] = np.asarray(data.layers["log_fc"][index, :], dtype=float).ravel()
        table["lfc_se"] = np.asarray(data.layers["lfcSE"][index, :], dtype=float).ravel()
        table["q_value"] = np.asarray(data.layers["adj_p_value"][index, :], dtype=float).ravel()
        table["target_gene"] = gene; table["culture_condition"] = state; table["role"] = role
        table["tested"] = table.q_value.notna() & table.log_fc.notna()
        table["significant"] = table.tested & table.q_value.le(0.10)
        table.to_csv(OUTPUT / f"volcano_{gene.lower()}_{state.lower()}.csv", index=False)
        extracted[(gene, state)] = table
    return extracted


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phenotypes = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    data = ad.read_h5ad(H5AD, backed="r")
    extracted = extract_rows(data, phenotypes)

    programs = pd.read_csv(ROOT / "analyses/loci/results/transfer_programs.csv")
    edge_rows = []
    for (gene, state), block in programs.groupby(["gene", "condition"]):
        row = phenotypes[(phenotypes.target_contrast_gene_name == gene) & (phenotypes.culture_condition == state)]
        if row.empty:
            continue
        key = str(row.iloc[0]["index"]); index = data.obs_names.get_loc(key)
        names = pd.Series(data.var["gene_name"].astype(str).to_numpy(), index=data.var_names.astype(str))
        q_values = np.asarray(data.layers["adj_p_value"][index, :], dtype=float).ravel()
        q_map = pd.Series(q_values, index=names.to_numpy()).groupby(level=0).min()
        for item in block.itertuples(index=False):
            edge_rows.append({**item._asdict(), "q_value": q_map.get(item.downstream_gene, np.nan), "supported": bool(q_map.get(item.downstream_gene, np.inf) <= 0.10)})
    edges = pd.DataFrame(edge_rows)
    edges.to_csv(OUTPUT / "signed_bipartite_edges.csv", index=False)
    data.file.close()

    locus_summary = pd.read_csv(ROOT / "analyses/loci/results/locus_summary.csv")
    grades = pd.read_csv(ROOT / "analyses/causal_triangulation/results/locus_grades.csv")
    assignments = phenotypes[phenotypes.target_contrast_gene_name.isin(locus_summary.gene)][["target_contrast_gene_name", "culture_condition", "cluster", "transfer_class", "transfer_z"]].drop_duplicates()
    convergence = pd.read_csv(ROOT / "analyses/disease/results/convergence_edges.csv")
    circos = []
    for row in locus_summary.itertuples(index=False):
        tier = grades.loc[grades.locus == row.locus, "evidence_tier"].iloc[0]
        circos.append({"source": row.lead_rsid, "target": row.gene, "source_type": "variant", "target_type": "gene", "status": "available", "weight": row.lead_posterior_probability, "evidence_tier": tier, "condition": "locus"})
        local = assignments[(assignments.target_contrast_gene_name == row.gene) & assignments.cluster.notna()]
        if local.empty:
            circos.append({"source": row.gene, "target": "program unavailable", "source_type": "gene", "target_type": "program", "status": "unavailable", "weight": 0, "evidence_tier": tier, "condition": "all"})
        for item in local.itertuples(index=False):
            program = f"Program {int(item.cluster)}"
            circos.append({"source": row.gene, "target": program, "source_type": "gene", "target_type": "program", "status": "available", "weight": abs(item.transfer_z), "evidence_tier": tier, "condition": item.culture_condition})
            disease = convergence[(convergence.cluster == item.cluster) & (convergence.culture_condition == item.culture_condition)]
            if disease.empty:
                circos.append({"source": program, "target": "disease link unavailable", "source_type": "program", "target_type": "disease", "status": "unavailable", "weight": 0, "evidence_tier": tier, "condition": item.culture_condition})
            for link in disease.itertuples(index=False):
                circos.append({"source": program, "target": link.disease, "source_type": "program", "target_type": "disease", "status": "available", "weight": -np.log10(link.p_adj_fdr), "evidence_tier": tier, "condition": item.culture_condition})
    pd.DataFrame(circos).drop_duplicates().to_csv(OUTPUT / "circos_edges.csv", index=False)

    pd.read_csv(ROOT / "analyses/grn_benchmark/results/method_results.csv").to_csv(OUTPUT / "alternative_grn_methods.csv", index=False)
    pd.read_csv(ROOT / "analyses/grn_benchmark/results/unavailable_datasets.csv").to_csv(OUTPUT / "alternative_grn_unavailable.csv", index=False)
    audit = {"volcano_exemplars": [list(x) for x in EXEMPLARS], "bipartite_edges": int(len(edges)), "circos_edges": int(len(circos)), "volcano_rule": "adjusted p value <= 0.10", "volcano_all_tested_genes": True}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
