"""Prepare frozen disease, fine-mapping, state, and support-stratum outputs for S13 to S16."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/supplementary_disease/results"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    disease = ROOT / "analyses/disease/results"
    causal = ROOT / "analyses/causal_triangulation/results"
    loci = ROOT / "analyses/loci/results"

    enrichment = pd.read_csv(disease / "amplified_disease_enrichment.csv")
    convergence = pd.read_csv(disease / "convergence_edges.csv")
    enrichment.to_csv(OUTPUT / "s13_all_disease_tests.csv", index=False)
    convergence.groupby("disease").agg(
        significant_programs=("cluster", "size"),
        states=("culture_condition", "nunique"),
        minimum_q=("p_adj_fdr", "min"),
        median_odds_ratio=("odds_ratio", "median"),
    ).reset_index().to_csv(OUTPUT / "s13_disease_program_summary.csv", index=False)
    pd.read_csv(disease / "model_comparison.csv").to_csv(OUTPUT / "s13_model_comparison.csv", index=False)
    pd.read_csv(disease / "fold_metrics.csv").to_csv(OUTPUT / "s13_fold_metrics.csv", index=False)

    pd.read_csv(loci / "fine_mapping_sensitivity.csv").to_csv(OUTPUT / "s14_fine_mapping.csv", index=False)
    pd.read_csv(causal / "evidence_gates.csv").to_csv(OUTPUT / "s14_evidence_gates.csv", index=False)
    pd.read_csv(causal / "locus_grades.csv").to_csv(OUTPUT / "s14_locus_grades.csv", index=False)
    pd.read_csv(causal / "unavailable_tests.csv").to_csv(OUTPUT / "s14_unavailable_tests.csv", index=False)

    ancestry = pd.DataFrame(
        [
            {
                "analysis": "ancestry-stratified transfer",
                "status": "unavailable",
                "eligible_strata": 0,
                "reason": "individual-level ancestry labels were not present in the frozen public perturbation summaries",
            },
            {
                "analysis": "ancestry-stratified locus evidence",
                "status": "unavailable",
                "eligible_strata": 0,
                "reason": "the three source credible sets did not provide a harmonized ancestry field for comparative estimation",
            },
        ]
    )
    ancestry.to_csv(OUTPUT / "s15_ancestry_availability.csv", index=False)
    enrichment.groupby("condition").agg(
        disease_tests=("disease", "size"),
        median_odds_ratio=("odds_ratio", "median"),
        fdr_significant=("q_value", lambda x: int((x < 0.05).sum())),
    ).reset_index().to_csv(OUTPUT / "s15_state_disease_summary.csv", index=False)
    pd.read_csv(ROOT / "analyses/primary/results/condition_summary.csv").to_csv(OUTPUT / "s15_state_transfer_summary.csv", index=False)
    programs = pd.read_csv(loci / "transfer_programs.csv")
    programs.groupby(["locus", "condition"]).agg(
        displayed_program_genes=("downstream_gene", "nunique"),
        median_signed_effect=("normalized_signed_effect", "median"),
        positive_fraction=("normalized_signed_effect", lambda x: float((x > 0).mean())),
    ).reset_index().to_csv(OUTPUT / "s15_locus_state_programs.csv", index=False)

    phenotypes = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    for variable, name in [("n_cells_target", "cells"), ("n_guides", "guides")]:
        phenotypes.groupby("culture_condition")[variable].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).reset_index().to_csv(
            OUTPUT / f"s16_{name}_support.csv", index=False
        )
    scalar = pd.read_parquet(ROOT / "analyses/multiverse/results/scalar_specifications.parquet")
    scalar.groupby(["minimum_cells", "minimum_guides", "result_family"]).agg(
        specifications=("estimate", "size"),
        median_estimate=("estimate", "median"),
        expected_direction_fraction=("direction_expected", "mean"),
        supported_fraction=("supported", "mean"),
    ).reset_index().to_csv(OUTPUT / "s16_support_thresholds.csv", index=False)
    complete = pd.read_parquet(ROOT / "analyses/primary/results/context_switching.parquet")
    denominator = pd.DataFrame(
        [
            {"category": "all eligible targets", "targets": phenotypes["target_contrast"].nunique()},
            {"category": "complete three-state targets", "targets": complete["target_contrast"].nunique()},
            {"category": "incomplete state coverage", "targets": phenotypes["target_contrast"].nunique() - complete["target_contrast"].nunique()},
        ]
    )
    denominator.to_csv(OUTPUT / "s16_state_coverage.csv", index=False)
    audit = {
        "disease_tests": int(len(enrichment)),
        "disease_fdr_significant": int((enrichment["q_value"] < 0.05).sum()),
        "causal_loci": 3,
        "ancestry_strata": 0,
        "phenotype_rows": int(len(phenotypes)),
        "complete_three_state_targets": int(len(complete)),
    }
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
