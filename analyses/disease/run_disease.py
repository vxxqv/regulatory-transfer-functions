"""Test whether frozen transfer features improve cluster-level disease convergence models."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrichment", type=Path, required=True)
    parser.add_argument(
        "--phenotypes", type=Path, default=Path("analyses/primary/results/transfer_phenotypes.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("analyses/disease/results"))
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def performance(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(observed, predicted)),
        "average_precision": float(average_precision_score(observed, predicted)),
        "brier": float(brier_score_loss(observed, predicted)),
    }


def main() -> None:
    args = parse_args()
    disease_all = pd.read_csv(args.enrichment)
    disease = disease_all.loc[disease_all["gene_set"].str.startswith("downstream_")].copy()
    disease["culture_condition"] = disease["gene_set"].str.replace("downstream_", "", regex=False)
    disease["significant"] = (disease["p_adj_fdr"] < 0.05) & ~disease["negative_control_disease"]
    disease["intersecting_gene_count"] = disease["intersecting_genes"].fillna("[]").map(
        lambda value: len(ast.literal_eval(value))
    )
    all_phenotype = pd.read_parquet(args.phenotypes)
    phenotype = all_phenotype.dropna(subset=["cluster"]).copy()
    phenotype["cluster"] = phenotype["cluster"].astype(int)
    cluster_features = (
        phenotype.groupby(["cluster", "culture_condition"], as_index=False)
        .agg(
            regulators=("target_contrast_gene_name", "nunique"),
            median_transfer_residual=("transfer_residual", "median"),
            median_abs_transfer_residual=("transfer_residual", lambda values: np.median(np.abs(values))),
            amplified_fraction=("transfer_class", lambda values: np.mean(values == "amplified")),
            buffered_fraction=("transfer_class", lambda values: np.mean(values == "buffered")),
            median_log_response=("log1p_n_downstream", "median"),
        )
    )
    joined = disease.merge(
        cluster_features,
        on=["cluster", "culture_condition"],
        how="inner",
        validate="many_to_one",
    )
    real = joined.loc[~joined["negative_control_disease"]].reset_index(drop=True)
    outcome = real["significant"].astype(int).to_numpy()
    groups = real["cluster"].to_numpy()
    base_numeric = ["cluster_size", "regulators"]
    transfer_numeric = [
        "median_transfer_residual",
        "median_abs_transfer_residual",
        "amplified_fraction",
        "buffered_fraction",
        "median_log_response",
    ]
    categorical = ["culture_condition"]
    predictions = {"size_state": np.full(len(real), np.nan), "transfer_augmented": np.full(len(real), np.nan)}
    fold_records = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(real, outcome, groups), start=1):
        for name, numeric in [("size_state", base_numeric), ("transfer_augmented", base_numeric + transfer_numeric)]:
            process = ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                        numeric,
                    ),
                    ("state", OneHotEncoder(handle_unknown="ignore"), categorical),
                ]
            )
            model = Pipeline(
                [
                    ("process", process),
                    (
                        "model",
                        LogisticRegression(
                            C=0.5,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=args.seed + fold,
                        ),
                    ),
                ]
            )
            model.fit(real.iloc[train], outcome[train])
            probability = model.predict_proba(real.iloc[test])[:, 1]
            predictions[name][test] = probability
            fold_records.append({"fold": fold, "model": name, "rows": len(test), **performance(outcome[test], probability)})
    for name, values in predictions.items():
        real[f"probability_{name}"] = values
    comparison = pd.DataFrame(
        [{"model": name, **performance(outcome, values)} for name, values in predictions.items()]
    )
    negative = joined.loc[joined["negative_control_disease"]].copy()
    top_forest = real.loc[real["significant"]].sort_values("p_adj_fdr").head(40)
    convergence_edges = real.loc[real["significant"], [
        "cluster", "disease", "culture_condition", "odds_ratio", "p_adj_fdr", "intersecting_gene_count"
    ]].copy()
    regulator_sets = disease_all.loc[
        (disease_all["gene_set"] == "regulators") & ~disease_all["negative_control_disease"]
    ].copy()
    disease_genes = {
        trait: set().union(
            *[set(ast.literal_eval(value)) for value in group["intersecting_genes"].dropna()]
        )
        for trait, group in regulator_sets.groupby("disease")
    }
    enrichment_records = []
    # Disease membership is only observable for regulators assigned to a source
    # response cluster, so the enrichment denominator is restricted to that
    # same eligible population.
    for condition, group in phenotype.groupby("culture_condition"):
        group = group.drop_duplicates("target_contrast_gene_name")
        amplified = group["transfer_class"].eq("amplified").to_numpy()
        genes = group["target_contrast_gene_name"].astype(str).to_numpy()
        for trait, members in disease_genes.items():
            in_trait = np.array([gene in members for gene in genes])
            a = int(np.sum(amplified & in_trait))
            b = int(np.sum(amplified & ~in_trait))
            c = int(np.sum(~amplified & in_trait))
            d = int(np.sum(~amplified & ~in_trait))
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            corrected = np.array([a, b, c, d], dtype=float) + 0.5
            log_or = np.log((corrected[0] * corrected[3]) / (corrected[1] * corrected[2]))
            standard_error = np.sqrt(np.sum(1.0 / corrected))
            enrichment_records.append(
                {
                    "condition": condition,
                    "disease": trait,
                    "amplified_disease": a,
                    "amplified_not_disease": b,
                    "other_disease": c,
                    "other_not_disease": d,
                    "odds_ratio": odds_ratio,
                    "ci_low": np.exp(log_or - 1.96 * standard_error),
                    "ci_high": np.exp(log_or + 1.96 * standard_error),
                    "p_value": p_value,
                }
            )
    amplified_enrichment = pd.DataFrame(enrichment_records)
    amplified_enrichment["q_value"] = multipletests(
        amplified_enrichment["p_value"], method="fdr_bh"
    )[1]
    args.output.mkdir(parents=True, exist_ok=True)
    real.to_parquet(args.output / "disease_model_rows.parquet", index=False)
    comparison.to_csv(args.output / "model_comparison.csv", index=False)
    pd.DataFrame(fold_records).to_csv(args.output / "fold_metrics.csv", index=False)
    top_forest.to_csv(args.output / "top_disease_enrichments.csv", index=False)
    convergence_edges.to_csv(args.output / "convergence_edges.csv", index=False)
    negative.to_csv(args.output / "negative_control_rows.csv", index=False)
    amplified_enrichment.to_csv(args.output / "amplified_disease_enrichment.csv", index=False)
    indexed = comparison.set_index("model")
    audit = {
        "joined_rows": len(joined),
        "real_disease_rows": len(real),
        "negative_control_rows": len(negative),
        "clusters": int(real["cluster"].nunique()),
        "diseases": int(real["disease"].nunique()),
        "significant_cluster_disease_links": int(outcome.sum()),
        "negative_control_significant_links": int((negative["p_adj_fdr"] < 0.05).sum()),
        "size_state_auroc": float(indexed.at["size_state", "auroc"]),
        "transfer_augmented_auroc": float(indexed.at["transfer_augmented", "auroc"]),
        "transfer_minus_size_state_auroc": float(
            indexed.at["transfer_augmented", "auroc"] - indexed.at["size_state", "auroc"]
        ),
        "amplified_disease_tests": len(amplified_enrichment),
        "amplified_disease_fdr_significant": int((amplified_enrichment["q_value"] < 0.05).sum()),
    }
    (args.output / "disease_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
