"""Run the frozen primary empirical transfer analysis."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kruskal, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("analyses/primary/results"))
    parser.add_argument("--config", type=Path, default=Path("config/analysis.yaml"))
    return parser.parse_args()


def safe_spearman(left: pd.Series, right: pd.Series) -> dict[str, float | int | None]:
    complete = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(complete) < 3 or complete.nunique().min() < 2:
        return {"rho": None, "p_value": None, "n": int(len(complete))}
    result = spearmanr(complete["left"], complete["right"])
    return {"rho": float(result.statistic), "p_value": float(result.pvalue), "n": int(len(complete))}


def parse_cluster_assignments(clusters: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    conditions = ("Stim48hr", "Stim8hr", "Rest")
    for row in clusters.itertuples(index=False):
        members = ast.literal_eval(row.cluster_member_with_condition)
        for member in members:
            matched = next((condition for condition in conditions if member.endswith(f"_{condition}")), None)
            if matched is None:
                raise ValueError(f"Unrecognized condition suffix: {member}")
            gene = member[: -(len(matched) + 1)]
            records.append(
                {
                    "target_contrast_gene_name": gene,
                    "culture_condition": matched,
                    "cluster": int(row.cluster),
                    "cluster_annotation": row.manual_annotation,
                    "cluster_condition_specificity": row.condition_specificity,
                }
            )
    result = pd.DataFrame(records).drop_duplicates()
    keys = ["target_contrast_gene_name", "culture_condition"]
    if result.duplicated(keys).any():
        raise ValueError("A perturbation-condition pair maps to more than one cluster")
    return result.sort_values(keys).reset_index(drop=True)


def entropy(values: pd.Series) -> float:
    frequencies = values.value_counts(normalize=True)
    raw = float(-(frequencies * np.log(frequencies)).sum())
    return raw / np.log(3) if len(values) == 3 else np.nan


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.models.empirical_transfer import cross_fitted_transfer, quality_filter
    from src.qc.summary_qc import audit_counts
    from src.statistics.resampling import condition_switch_null, grouped_bootstrap_interval

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    qc_config = config["quality_control"]
    analysis_config = config["transfer_phenotypes"]
    seed = int(config["study"]["seed"])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    source = args.source_root / "metadata" / "suppl_tables"
    de = pd.read_csv(source / "DE_stats.suppl_table.csv")
    qc = audit_counts(de, qc_config["minimum_guides"], qc_config["minimum_cells"])
    qc.to_csv(output / "qc_filter_counts.csv", index=False)

    selected = de.loc[
        quality_filter(de, qc_config["minimum_guides"], qc_config["minimum_cells"])
    ].copy()
    fit = cross_fitted_transfer(
        selected,
        folds=analysis_config["cross_validation"]["folds"],
        seed=seed,
    )
    transfer = fit.table.sort_values(["target_contrast", "culture_condition"]).reset_index(drop=True)

    cluster_source = pd.read_csv(source / "clustering_results_and_annotations.csv")
    assignments = parse_cluster_assignments(cluster_source)
    assignments.to_parquet(output / "cluster_assignments.parquet", index=False)
    transfer = transfer.merge(
        assignments,
        on=["target_contrast_gene_name", "culture_condition"],
        how="left",
        validate="many_to_one",
    )

    complete_targets = (
        transfer.groupby("target_contrast")["culture_condition"].nunique().loc[lambda x: x == 3].index
    )
    complete = transfer[transfer["target_contrast"].isin(complete_targets)].copy()
    context = (
        complete.groupby(["target_contrast", "target_contrast_gene_name"], as_index=False)
        .agg(
            residual_range=("transfer_z", lambda x: float(x.max() - x.min())),
            residual_sd=("transfer_z", "std"),
            minimum_transfer_z=("transfer_z", "min"),
            maximum_transfer_z=("transfer_z", "max"),
            states_amplified=("transfer_class", lambda x: int((x == "amplified").sum())),
            states_buffered=("transfer_class", lambda x: int((x == "buffered").sum())),
            clusters_observed=("cluster", "count"),
            distinct_clusters=("cluster", "nunique"),
            cluster_entropy=("cluster", entropy),
        )
        .sort_values("residual_range", ascending=False)
    )
    context["cluster_switch"] = context["distinct_clusters"] > 1

    null_statistic, null_values, null_p = condition_switch_null(
        complete,
        "target_contrast",
        "culture_condition",
        "transfer_z",
        replicates=int(config["negative_controls"]["permutations"]),
        seed=seed,
    )
    null_p_lower = float((1 + np.sum(null_values <= null_statistic)) / (len(null_values) + 1))
    pd.DataFrame({"null_median_residual_range": null_values}).to_parquet(
        output / "context_switch_null.parquet", index=False
    )

    switch_ci = grouped_bootstrap_interval(
        context,
        "target_contrast",
        "residual_range",
        statistic="median",
        replicates=int(analysis_config["uncertainty"]["replicates"]),
        seed=seed,
    )
    condition_values = [
        part["transfer_z"].to_numpy() for _, part in transfer.groupby("culture_condition", sort=True)
    ]
    condition_test = kruskal(*condition_values)

    state_matrix = complete.pivot(
        index="target_contrast", columns="culture_condition", values="transfer_z"
    )
    state_correlations: dict[str, dict[str, float | int | None]] = {}
    ordered_states = ["Rest", "Stim8hr", "Stim48hr"]
    for left_index, left_state in enumerate(ordered_states):
        for right_state in ordered_states[left_index + 1 :]:
            state_correlations[f"{left_state}_vs_{right_state}"] = safe_spearman(
                state_matrix[left_state], state_matrix[right_state]
            )

    technical_checks = {
        "cis_magnitude": safe_spearman(transfer["transfer_z"], transfer["cis_magnitude"]),
        "target_expression": safe_spearman(transfer["transfer_z"], transfer["target_baseMean"]),
        "target_cells": safe_spearman(transfer["transfer_z"], transfer["n_cells_target"]),
        "guide_reproducibility": safe_spearman(
            transfer["transfer_z"], transfer["guide_correlation_all"]
        ),
        "donor_reproducibility": safe_spearman(
            transfer["transfer_z"], transfer["donor_correlation_all_mean"]
        ),
    }
    class_counts = transfer["transfer_class"].value_counts().to_dict()
    condition_summary = (
        transfer.groupby("culture_condition")
        .agg(
            tests=("target_contrast", "size"),
            targets=("target_contrast", "nunique"),
            median_downstream=("n_downstream", "median"),
            mean_transfer_z=("transfer_z", "mean"),
            median_transfer_z=("transfer_z", "median"),
        )
        .reset_index()
    )
    condition_summary.to_csv(output / "condition_summary.csv", index=False)

    results = {
        "source_rows": int(len(de)),
        "source_targets": int(de["target_contrast"].nunique()),
        "primary_rows": int(len(transfer)),
        "primary_targets": int(transfer["target_contrast"].nunique()),
        "complete_three_state_targets": int(len(context)),
        "out_of_fold_r2": float(fit.r2),
        "cross_validation_folds": int(fit.folds),
        "transfer_class_counts": {key: int(value) for key, value in class_counts.items()},
        "median_context_switch": {
            "estimate": float(switch_ci[0]),
            "ci_95_low": float(switch_ci[1]),
            "ci_95_high": float(switch_ci[2]),
            "permutation_null_median": float(np.median(null_values)),
            "permutation_p_upper": float(null_p),
            "permutation_p_lower": null_p_lower,
            "null_definition": "condition-stratified independent target reassignment",
            "observed_statistic": float(null_statistic),
        },
        "cluster_switch": {
            "targets_with_all_three_cluster_assignments": int((context["clusters_observed"] == 3).sum()),
            "switching_targets": int(
                ((context["clusters_observed"] == 3) & context["cluster_switch"]).sum()
            ),
        },
        "condition_residual_test": {
            "kruskal_h": float(condition_test.statistic),
            "p_value": float(condition_test.pvalue),
        },
        "cross_state_transfer_correlations": state_correlations,
        "technical_and_reproducibility_checks": technical_checks,
    }
    (output / "primary_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    transfer.to_parquet(output / "transfer_phenotypes.parquet", index=False)
    context.to_parquet(output / "context_switching.parquet", index=False)
    context.head(100).to_csv(output / "top_context_switches.csv", index=False)


if __name__ == "__main__":
    main()
