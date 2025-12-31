"""Prepare existing null and negative-control outputs for Supplementary Figures S9 to S12."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/supplementary_nulls/results"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rewiring = pd.read_parquet(ROOT / "analyses/external_benchmark/results/degree_preserving_rewiring_null.parquet")
    external_summary = pd.read_csv(ROOT / "analyses/external_benchmark/results/null_summary.csv")
    rewiring.to_csv(OUTPUT / "s09_rewiring.csv", index=False)
    external_summary.to_csv(OUTPUT / "s09_null_summary.csv", index=False)
    rewiring.groupby("status").size().reset_index(name="replicates").to_csv(OUTPUT / "s09_rewiring_status.csv", index=False)
    quantiles = rewiring["r2"].quantile(np.linspace(0.01, 0.99, 99)).rename_axis("quantile").reset_index(name="null_r2")
    quantiles.to_csv(OUTPUT / "s09_rewiring_quantiles.csv", index=False)

    replication = json.loads((ROOT / "analyses/replication/results/replication_results.json").read_text(encoding="utf-8"))
    sign_flip = pd.read_parquet(ROOT / "analyses/replication/results/sign_flip_null.parquet")
    sign_flip["observed_delta"] = replication["median_correlation_delta"]
    sign_flip.to_csv(OUTPUT / "s10_sign_flip.csv", index=False)
    matched = pd.read_parquet(ROOT / "analyses/natural_genetics/results/matched_null.parquet")
    natural = json.loads((ROOT / "analyses/natural_genetics/results/natural_genetics_results.json").read_text(encoding="utf-8"))
    matched["observed_agreement"] = natural["direction_agreement"]
    matched["observed_rho"] = natural["score_spearman_rho"]
    matched.to_csv(OUTPUT / "s10_natural_matched_null.csv", index=False)
    pd.read_csv(ROOT / "analyses/replication/results/condition_summary.csv").to_csv(OUTPUT / "s10_k562_conditions.csv", index=False)
    pd.read_csv(ROOT / "analyses/natural_genetics/results/condition_summary.csv").to_csv(OUTPUT / "s10_genetic_conditions.csv", index=False)

    chromosome = pd.read_csv(ROOT / "analyses/natural_genetics/results/chromosome_holdout.csv")
    chromosome.to_csv(OUTPUT / "s11_chromosome_holdout.csv", index=False)
    pairs = pd.read_parquet(ROOT / "analyses/natural_genetics/results/directional_pairs.parquet")
    pairs["absolute_perturbation_effect"] = pairs["perturbation_effect"].abs()
    pairs["effect_bin"] = pd.qcut(pairs["absolute_perturbation_effect"], 8, duplicates="drop")
    binned = pairs.groupby("effect_bin", observed=True).agg(effect_median=("absolute_perturbation_effect", "median"), direction_agreement=("direction_match", "mean"), pairs=("direction_match", "size"), snps=("SNP", "nunique")).reset_index(drop=True)
    binned.to_csv(OUTPUT / "s11_effect_bins.csv", index=False)
    condition = pd.read_csv(ROOT / "analyses/natural_genetics/results/condition_summary.csv")
    condition.to_csv(OUTPUT / "s11_condition_matching.csv", index=False)
    strict = pd.DataFrame([{"analysis": "all mapped trans pairs", "pairs": natural["pairs"], "direction_agreement": natural["direction_agreement"]}, {"analysis": "Bonferroni trans subset", "pairs": natural["bonferroni_trans_pairs"], "direction_agreement": natural["bonferroni_trans_direction_agreement"]}])
    strict.to_csv(OUTPUT / "s11_strict_mappability.csv", index=False)

    switch_null = pd.read_parquet(ROOT / "analyses/primary/results/context_switch_null.parquet")
    switch_observed = pd.read_parquet(ROOT / "analyses/primary/results/context_switching.parquet")["residual_range"].median()
    switch_null["observed_median"] = switch_observed
    switch_null.to_csv(OUTPUT / "s12_context_labels.csv", index=False)
    feature_null = pd.read_parquet(ROOT / "analyses/external_benchmark/results/matched_permutation_null.parquet")
    observed_feature = external_summary.loc[external_summary["null"] == "matched_cd4_feature_permutation", "observed"].iloc[0]
    feature_null["observed_delta"] = observed_feature
    feature_null.to_csv(OUTPUT / "s12_feature_labels.csv", index=False)
    disease = pd.read_csv(ROOT / "analyses/disease/results/negative_control_rows.csv")
    disease.to_csv(OUTPUT / "s12_negative_diseases.csv", index=False)
    disease.groupby("culture_condition").agg(tests=("p_adj_fdr", "size"), minimum_q=("p_adj_fdr", "min"), median_q=("p_adj_fdr", "median"), corrected_significant=("significant", "sum")).reset_index().to_csv(OUTPUT / "s12_negative_disease_summary.csv", index=False)
    audit = {"degree_rewirings": int(len(rewiring)), "sign_flips": int(len(sign_flip)), "matched_genetic_permutations": int(len(matched)), "directional_pairs": int(len(pairs)), "chromosomes": int(len(chromosome)), "context_label_permutations": int(len(switch_null)), "feature_label_permutations": int(len(feature_null)), "negative_disease_tests": int(len(disease))}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
