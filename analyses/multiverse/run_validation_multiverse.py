"""Run cross-system, natural-genetic, and disease-convergence specification grids."""

from __future__ import annotations

import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyses.multiverse.run_scalar_multiverse import ROOT, adjust_q, multiplier_summary


OUTPUT = ROOT / "analyses/multiverse/results"


def score_columns(rows: pd.DataFrame) -> dict[str, str]:
    rows["score_distal_gene_count_residual"] = rows["transfer_z"]
    for name, source in [
        ("debiased_response_l2_gain", "gamma_l2_debiased"),
        ("significant_response_l1_gain", "gamma_l1_significant"),
    ]:
        logged = np.log1p(rows[source].clip(lower=0))
        rows[f"score_{name}"] = logged.groupby(rows["culture_condition"]).transform(
            lambda values: (values - values.mean()) / values.std()
        )
    return {name: f"score_{name}" for name in [
        "distal_gene_count_residual", "debiased_response_l2_gain", "significant_response_l1_gain"
    ]}


def disease_predictions(table: pd.DataFrame, feature_columns: list[str]) -> tuple[float, pd.Series]:
    y = table["significant"].astype(int).to_numpy()
    clusters = table["cluster"].astype(str).to_numpy()
    folds = np.array([
        int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) % 5 for value in clusters
    ])
    state = pd.get_dummies(table["culture_condition"], prefix="state", dtype=float)
    baseline = pd.concat([table[["cluster_size"]].reset_index(drop=True), state.reset_index(drop=True)], axis=1)
    augmented = pd.concat([baseline, table[feature_columns].reset_index(drop=True)], axis=1)
    predictions = {}
    for name, features in [("baseline", baseline), ("augmented", augmented)]:
        pred = np.full(len(table), np.nan)
        for heldout in range(5):
            train = folds != heldout
            test = folds == heldout
            model = Pipeline([
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=500, class_weight="balanced")),
            ])
            model.fit(features.loc[train], y[train])
            pred[test] = model.predict_proba(features.loc[test])[:, 1]
        predictions[name] = pred
    delta_auc = float(
        roc_auc_score(y, predictions["augmented"])
        - roc_auc_score(y, predictions["baseline"])
    )
    # Positive paired contributions mean the transfer-augmented model improves
    # out-of-fold Brier loss. Their mean is additive, so the target-cluster
    # multiplier interval below estimates uncertainty for the reported effect.
    brier_improvement = pd.Series(
        (y - predictions["baseline"]) ** 2 - (y - predictions["augmented"]) ** 2,
        index=table.index,
        name="brier_improvement",
    )
    return delta_auc, brier_improvement


def main() -> None:
    config = yaml.safe_load((ROOT / "config/multiverse.yaml").read_text(encoding="utf-8"))
    seed = int(config["multiverse"]["seeds"][0])
    replicates = int(config["multiverse"]["bootstrap_replicates"])
    rows = pd.read_parquet(ROOT / "analyses/vectors/results/network_gain_rows.parquet")
    phenotypes = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    rows = rows.merge(
        phenotypes[["index", "transfer_z", "cluster"]], on="index", how="left", validate="one_to_one"
    )
    scores = score_columns(rows)
    k562 = pd.read_parquet(ROOT / "analyses/replication/results/k562_replication_rows.parquet")
    natural = pd.read_parquet(ROOT / "analyses/natural_genetics/results/directional_pairs.parquet")
    disease = pd.read_parquet(ROOT / "analyses/disease/results/disease_model_rows.parquet")
    records = []
    unavailable = []
    spec = 0
    start = time.perf_counter()
    for minimum_cells, minimum_guides, on_target_rule, definition, tail in product(
        config["preprocessing"]["minimum_cells"],
        config["preprocessing"]["minimum_guides"],
        config["preprocessing"]["on_target_requirement"],
        config["transfer_definitions"],
        config["tail_cutoffs"],
    ):
        spec += 1
        mask = (
            rows["ontarget_significant"].fillna(False).astype(bool)
            & ~rows["low_target_gex"].fillna(True).astype(bool)
            & ~rows["neighboring_gene_KD"].fillna(True).astype(bool)
            & ~rows["distal_offtarget_flag"].fillna(True).astype(bool)
            & (rows["n_cells_target"] >= minimum_cells)
            & (rows["n_guides"] >= minimum_guides)
        )
        if on_target_rule == "significant_and_absolute_z_at_least_2":
            mask &= rows["ontarget_effect_size"].abs() >= 2
        selected = rows.loc[mask, ["target_contrast_gene_name", "culture_condition", "cluster", scores[definition]]].copy()
        selected = selected.rename(columns={scores[definition]: "score"}).dropna(subset=["score"])
        threshold = selected["score"].quantile(1 - float(tail))
        selected["tail_selected"] = selected["score"] >= threshold
        descriptor = {
            "minimum_cells": minimum_cells,
            "minimum_guides": minimum_guides,
            "on_target_rule": on_target_rule,
            "transfer_definition": definition,
            "tail_cutoff": tail,
            "module_resolution": 30,
        }

        cross = k562.merge(
            selected,
            left_on=["target_contrast_gene_name", "condition"],
            right_on=["target_contrast_gene_name", "culture_condition"],
            how="inner",
        )
        cross = cross[cross["tail_selected"]]
        cross_result = multiplier_summary(
            cross["correlation_delta"], cross["target_contrast_gene_name"], seed + spec, replicates
        )
        records.append(
            {
                **descriptor,
                "specification_id": f"cross_{spec}",
                "result_family": "cross_system_conservation",
                "rows": len(cross),
                "expected_direction": "positive",
                "negative_control": "three_random_gene_set_correlations_subtracted",
                **cross_result,
            }
        )

        genetics = natural.merge(
            selected,
            left_on=["GeneSymbol_cis", "condition"],
            right_on=["target_contrast_gene_name", "culture_condition"],
            how="inner",
        )
        genetics = genetics[genetics["tail_selected"]].copy()
        genetics["centered_match"] = genetics["direction_match"].astype(float) - 0.5
        genetic_result = multiplier_summary(
            genetics["centered_match"], genetics["SNP"], seed + 1000 + spec, replicates
        )
        records.append(
            {
                **descriptor,
                "specification_id": f"genetic_{spec}",
                "result_family": "natural_genetic_concordance",
                "rows": len(genetics),
                "expected_direction": "positive",
                "negative_control": "degree_matched_and_chromosome_holdout_controls",
                **genetic_result,
            }
        )

        cluster_features = selected.groupby(["cluster", "culture_condition"], as_index=False).agg(
            spec_tail_fraction=("tail_selected", "mean"),
            spec_median_transfer=("score", "median"),
        )
        disease_spec = disease.merge(
            cluster_features, on=["cluster", "culture_condition"], how="inner"
        )
        for covariate_set in config["covariate_sets"]:
            if covariate_set == "minimal":
                features = ["spec_tail_fraction"]
            elif covariate_set == "observability":
                features = ["spec_tail_fraction", "spec_median_transfer", "regulators", "median_log_response"]
            elif covariate_set == "reproducibility":
                features = ["spec_tail_fraction", "spec_median_transfer", "regulators"]
            else:
                features = ["spec_tail_fraction", "spec_median_transfer"]
            for resolution in config["module_resolutions"]:
                if int(resolution) != 30:
                    unavailable.append(
                        {
                            **descriptor,
                            "covariate_set": covariate_set,
                            "module_resolution": resolution,
                            "result_family": "disease_convergence",
                            "reason": "disease labels are tied to the source study cluster resolution",
                        }
                    )
                    continue
                delta_auc, brier_improvement = disease_predictions(disease_spec, features)
                uncertainty = multiplier_summary(
                    brier_improvement,
                    disease_spec["cluster"],
                    seed + 2000 + spec,
                    replicates,
                )
                records.append(
                    {
                        **descriptor,
                        "specification_id": f"disease_{spec}_{covariate_set}",
                        "result_family": "disease_convergence",
                        "covariate_set": covariate_set,
                        "rows": len(disease_spec),
                        "expected_direction": "positive",
                        "effect_measure": "out_of_fold_brier_loss_improvement",
                        "secondary_delta_auroc": delta_auc,
                        "negative_control": "source_negative_control_diseases_retained_separately",
                        **uncertainty,
                    }
                )

    table = adjust_q(pd.DataFrame(records))
    table["direction_expected"] = table["estimate"] > 0
    table["supported"] = table["direction_expected"] & (table["q_value"] < 0.05)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT / "validation_specifications.parquet", index=False)
    pd.DataFrame(unavailable).to_csv(OUTPUT / "validation_unavailable.csv", index=False)
    summary = table.groupby("result_family").agg(
        specifications=("specification_id", "size"),
        expected_direction_fraction=("direction_expected", "mean"),
        q_supported_fraction=("supported", "mean"),
        median_estimate=("estimate", "median"),
        minimum_estimate=("estimate", "min"),
        maximum_estimate=("estimate", "max"),
    ).reset_index()
    summary.to_csv(OUTPUT / "validation_family_summary.csv", index=False)
    audit = {
        "specification_rows": len(table),
        "unavailable_rows": len(unavailable),
        "wall_seconds": time.perf_counter() - start,
    }
    (OUTPUT / "validation_multiverse_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
