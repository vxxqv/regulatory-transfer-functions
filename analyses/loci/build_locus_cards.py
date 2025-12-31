"""Bind public fine mapping and L2G evidence to measured CD4 transfer programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


LOCUS_META = {
    "gata3": {"gene": "GATA3", "trait": "asthma", "catalog_study": "GCST90652529"},
    "stat3": {
        "gene": "STAT3",
        "trait": "inflammatory bowel disease",
        "catalog_study": "FINNGEN_R12_K11_IBD_STRICT",
    },
    "ptpn22": {
        "gene": "PTPN22",
        "trait": "rheumatoid arthritis",
        "catalog_study": "GCST000679",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opentargets", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, default=Path("data/interim/gwt_vectors"))
    parser.add_argument(
        "--phenotypes", type=Path, default=Path("analyses/primary/results/transfer_phenotypes.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("analyses/loci/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.opentargets.read_text(encoding="utf-8"))["data"]
    rows = pd.read_parquet(args.vectors / "rows.parquet").reset_index(drop=True)
    genes = pd.read_parquet(args.vectors / "genes.parquet")
    matrix = sparse.load_npz(args.vectors / "normalized_significant_logfc.npz").tocsr()
    phenotypes = pd.read_parquet(args.phenotypes)
    gene_names = genes["gene_name"].astype(str).to_numpy()
    summary_records = []
    variant_records = []
    feature_records = []
    program_records = []
    sensitivity_records = []

    for key, meta in LOCUS_META.items():
        credible = payload[key]
        locus_rows = credible["locus"]["rows"]
        prediction = credible["l2GPredictions"]["rows"][0]
        posterior_mass_95 = sum(
            variant["posteriorProbability"] for variant in locus_rows if variant["is95CredibleSet"]
        )
        posterior_mass_99 = sum(
            variant["posteriorProbability"] for variant in locus_rows if variant["is99CredibleSet"]
        )
        lead = max(locus_rows, key=lambda variant: variant["posteriorProbability"])
        summary_records.append(
            {
                "locus": key,
                **meta,
                "study_locus_id": credible["studyLocusId"],
                "fine_mapping_method": credible["finemappingMethod"],
                "fine_mapping_confidence": credible["confidence"],
                "credible_variants": credible["locus"]["count"],
                "lead_variant": lead["variant"]["id"],
                "lead_rsid": ",".join(lead["variant"].get("rsIds") or []),
                "lead_posterior_probability": lead["posteriorProbability"],
                "l2g_target": prediction["target"]["approvedSymbol"],
                "l2g_score": prediction["score"],
            }
        )
        for variant in locus_rows:
            item = variant["variant"]
            variant_records.append(
                {
                    "locus": key,
                    "gene": meta["gene"],
                    "trait": meta["trait"],
                    "variant_id": item["id"],
                    "rsids": ",".join(item.get("rsIds") or []),
                    "chromosome": item["chromosome"],
                    "position": item["position"],
                    "reference_allele": item["referenceAllele"],
                    "alternate_allele": item["alternateAllele"],
                    "posterior_probability": variant["posteriorProbability"],
                    "is_95_credible_set": variant["is95CredibleSet"],
                    "is_99_credible_set": variant["is99CredibleSet"],
                }
            )
        for feature in prediction["features"]:
            feature_records.append(
                {
                    "locus": key,
                    "gene": meta["gene"],
                    "trait": meta["trait"],
                    "feature": feature["name"],
                    "value": feature["value"],
                    "shap_value": feature["shapValue"],
                }
            )
        sensitivity_records.extend(
            [
                {
                    "locus": key,
                    "threshold": "95%",
                    "variants": sum(item["is95CredibleSet"] for item in locus_rows),
                    "posterior_mass": posterior_mass_95,
                    "nominated_gene": prediction["target"]["approvedSymbol"],
                },
                {
                    "locus": key,
                    "threshold": "99%",
                    "variants": sum(item["is99CredibleSet"] for item in locus_rows),
                    "posterior_mass": posterior_mass_99,
                    "nominated_gene": prediction["target"]["approvedSymbol"],
                },
            ]
        )

        response_rows = rows.index[rows["target_contrast_gene_name"].eq(meta["gene"])].tolist()
        for response_row in response_rows:
            vector = matrix.getrow(response_row)
            order = np.argsort(np.abs(vector.data))[::-1][:20]
            phenotype_row = phenotypes.loc[phenotypes["index"].eq(rows.at[response_row, "index"])].iloc[0]
            for rank, local_position in enumerate(order, start=1):
                column = vector.indices[local_position]
                program_records.append(
                    {
                        "locus": key,
                        "gene": meta["gene"],
                        "trait": meta["trait"],
                        "condition": rows.at[response_row, "culture_condition"],
                        "transfer_class": phenotype_row["transfer_class"],
                        "transfer_residual": phenotype_row["transfer_residual"],
                        "rank": rank,
                        "downstream_gene": gene_names[column],
                        "normalized_signed_effect": vector.data[local_position],
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_records)
    variants = pd.DataFrame(variant_records)
    features = pd.DataFrame(feature_records)
    programs = pd.DataFrame(program_records)
    sensitivity = pd.DataFrame(sensitivity_records)
    summary.to_csv(args.output / "locus_summary.csv", index=False)
    variants.to_csv(args.output / "credible_set_variants.csv", index=False)
    features.to_csv(args.output / "l2g_features.csv", index=False)
    programs.to_csv(args.output / "transfer_programs.csv", index=False)
    sensitivity.to_csv(args.output / "fine_mapping_sensitivity.csv", index=False)
    audit = {
        "loci": len(summary),
        "credible_variants": int(len(variants)),
        "l2g_features": int(len(features)),
        "program_edges": int(len(programs)),
        "genes_match_l2g": bool((summary["gene"] == summary["l2g_target"]).all()),
        "states": sorted(programs["condition"].unique().tolist()),
    }
    (args.output / "locus_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
