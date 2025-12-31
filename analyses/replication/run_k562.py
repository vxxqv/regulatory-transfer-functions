"""Analyze the prespecified K562 second-system comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument(
        "--phenotypes", type=Path, default=Path("analyses/primary/results/transfer_phenotypes.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("analyses/replication/results"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def clustered_bootstrap(table: pd.DataFrame, replicates: int, seed: int) -> tuple[float, float]:
    grouped = [group["correlation_delta"].to_numpy() for _, group in table.groupby("target_contrast_gene_name")]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        draw = rng.integers(0, len(grouped), len(grouped))
        estimates[index] = np.median(np.concatenate([grouped[position] for position in draw]))
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.comparison, index_col=0)
    source = source.dropna(subset=["target_contrast_gene_name", "condition", "logfc_pearson_r"]).copy()
    random_columns = ["random_r1", "random_r2", "random_r3"]
    source["random_mean"] = source[random_columns].mean(axis=1)
    source["correlation_delta"] = source["logfc_pearson_r"] - source["random_mean"]
    pooled_random = source[random_columns].to_numpy(dtype=float).ravel()
    pooled_random = pooled_random[np.isfinite(pooled_random)]
    lower, upper = np.quantile(pooled_random, [0.025, 0.975])
    cd4_count_column = source["condition"].map(
        {"Rest": "n_degs_MASH_Rest", "Stim8hr": "n_degs_MASH_Stim8hr", "Stim48hr": "n_degs_MASH_Stim48hr"}
    )
    source["cd4_response_genes"] = [
        source.at[index, column] for index, column in cd4_count_column.items()
    ]
    source["shared_response"] = source["logfc_pearson_r"] > upper
    source["private_or_inverted"] = source["logfc_pearson_r"] < lower
    source["confidence_tier"] = np.select(
        [
            source["shared_response"]
            & (source["n_degs_MASH_K562"] >= 10)
            & (source["cd4_response_genes"] >= 10),
            source["shared_response"],
        ],
        ["high", "moderate"],
        default="unsupported",
    )

    phenotypes = pd.read_parquet(args.phenotypes)[
        ["target_contrast_gene_name", "culture_condition", "transfer_class", "transfer_residual"]
    ]
    source = source.merge(
        phenotypes,
        left_on=["target_contrast_gene_name", "condition"],
        right_on=["target_contrast_gene_name", "culture_condition"],
        how="left",
        validate="one_to_one",
    )
    ci_low, ci_high = clustered_bootstrap(source, args.bootstrap, args.seed)
    target_delta = source.groupby("target_contrast_gene_name")["correlation_delta"].median().to_numpy()
    rng = np.random.default_rng(args.seed + 1)
    sign_null = np.empty(args.permutations)
    for index in range(args.permutations):
        sign_null[index] = np.median(target_delta * rng.choice([-1.0, 1.0], len(target_delta)))
    observed_delta = float(source["correlation_delta"].median())

    condition_records = []
    for condition, group in source.groupby("condition", sort=True):
        degree = group[["n_degs_MASH_K562", "cd4_response_genes"]].dropna()
        degree_rho = spearmanr(degree.iloc[:, 0], degree.iloc[:, 1]).statistic if len(degree) >= 3 else np.nan
        condition_records.append(
            {
                "condition": condition,
                "rows": len(group),
                "targets": group["target_contrast_gene_name"].nunique(),
                "median_observed_correlation": group["logfc_pearson_r"].median(),
                "median_random_correlation": group[random_columns].stack().median(),
                "median_correlation_delta": group["correlation_delta"].median(),
                "response_degree_spearman_rho": degree_rho,
                "high_confidence_shared": (group["confidence_tier"] == "high").sum(),
                "private_or_inverted": group["private_or_inverted"].sum(),
            }
        )
    condition_summary = pd.DataFrame(condition_records)
    class_summary = (
        source.dropna(subset=["transfer_class"])
        .groupby(["condition", "transfer_class"], as_index=False)
        .agg(
            rows=("logfc_pearson_r", "size"),
            median_correlation=("logfc_pearson_r", "median"),
            shared_fraction=("shared_response", "mean"),
        )
    )
    regression_data = source.dropna(
        subset=["transfer_class", "n_degs_MASH_K562", "cd4_response_genes"]
    ).copy()
    regression_data["log1p_k562_response_genes"] = np.log1p(regression_data["n_degs_MASH_K562"])
    regression_data["log1p_cd4_response_genes"] = np.log1p(regression_data["cd4_response_genes"])
    model = smf.ols(
        "logfc_pearson_r ~ C(transfer_class, Treatment(reference='buffered')) + "
        "C(condition) + log1p_k562_response_genes + log1p_cd4_response_genes",
        data=regression_data,
    ).fit(cov_type="cluster", cov_kwds={"groups": regression_data["target_contrast_gene_name"]})
    confidence = model.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.to_numpy(),
            "standard_error": model.bse.to_numpy(),
            "ci_low": confidence.iloc[:, 0].to_numpy(),
            "ci_high": confidence.iloc[:, 1].to_numpy(),
            "p_value": model.pvalues.to_numpy(),
        }
    )
    class_mask = coefficients["term"].str.contains("transfer_class", regex=False)
    coefficients["q_value"] = np.nan
    coefficients.loc[class_mask, "q_value"] = multipletests(
        coefficients.loc[class_mask, "p_value"], method="fdr_bh"
    )[1]
    failures = source.sort_values("logfc_pearson_r").head(100)
    args.output.mkdir(parents=True, exist_ok=True)
    source.to_parquet(args.output / "k562_replication_rows.parquet", index=False)
    condition_summary.to_csv(args.output / "condition_summary.csv", index=False)
    class_summary.to_csv(args.output / "transfer_class_summary.csv", index=False)
    coefficients.to_csv(args.output / "class_concordance_regression.csv", index=False)
    failures.to_csv(args.output / "lowest_concordance_targets.csv", index=False)
    pd.DataFrame({"median_delta": sign_null}).to_parquet(args.output / "sign_flip_null.parquet", index=False)
    audit = {
        "rows": len(source),
        "targets": int(source["target_contrast_gene_name"].nunique()),
        "random_correlation_interval_95": [float(lower), float(upper)],
        "median_observed_correlation": float(source["logfc_pearson_r"].median()),
        "median_random_correlation": float(np.median(pooled_random)),
        "median_correlation_delta": observed_delta,
        "target_cluster_bootstrap_ci_95": [ci_low, ci_high],
        "target_sign_flip_p": float((1 + np.sum(sign_null >= observed_delta)) / (args.permutations + 1)),
        "high_confidence_shared_rows": int((source["confidence_tier"] == "high").sum()),
        "moderate_shared_rows": int((source["confidence_tier"] == "moderate").sum()),
        "private_or_inverted_rows": int(source["private_or_inverted"].sum()),
    }
    (args.output / "replication_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
