"""Target-held-out benchmark on integrated ENCODE noncoding CRISPRi screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=Path, default=Path("data/interim/encode_crispri/elements.parquet"))
    parser.add_argument(
        "--phenotypes", type=Path, default=Path("analyses/primary/results/transfer_phenotypes.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("analyses/encode_validation/results"))
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def score(observed: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(observed, probability)),
        "average_precision": float(average_precision_score(observed, probability)),
        "brier": float(brier_score_loss(observed, probability)),
    }


def main() -> None:
    args = parse_args()
    elements = pd.read_parquet(args.elements)
    elements = elements.loc[elements["valid_connection"]].copy()
    elements["outcome"] = elements["significant"].astype(int)
    elements["log1p_abs_distance"] = np.log1p(elements["distance_to_tss"].abs())
    elements["promoter_proximal"] = elements["distance_to_tss"].abs() <= 2000
    phenotype = pd.read_parquet(args.phenotypes)
    target_features = (
        phenotype.groupby("target_contrast_gene_name", as_index=False)
        .agg(
            median_transfer_residual=("transfer_residual", "median"),
            max_abs_transfer_residual=("transfer_residual", lambda values: np.max(np.abs(values))),
            median_log_response=("log1p_n_downstream", "median"),
            state_range=("transfer_residual", lambda values: values.max() - values.min()),
        )
        .rename(columns={"target_contrast_gene_name": "measuredGeneSymbol"})
    )
    elements = elements.merge(target_features, on="measuredGeneSymbol", how="left", validate="many_to_one")
    elements["transfer_feature_available"] = elements["median_transfer_residual"].notna()
    source_valid_rows = len(elements)
    source_targets = int(elements["measuredGeneSymbol"].nunique())
    source_significant = int(elements["outcome"].sum())
    elements = elements.loc[elements["transfer_feature_available"]].reset_index(drop=True)
    base_numeric = ["log1p_abs_distance", "promoter_proximal"]
    transfer_numeric = [
        "median_transfer_residual",
        "max_abs_transfer_residual",
        "median_log_response",
        "state_range",
        "transfer_feature_available",
    ]
    categorical = ["biosample", "assembly"]
    groups = elements["measuredGeneSymbol"].astype(str).to_numpy()
    outcome = elements["outcome"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    predictions = {
        "distance_assay": np.full(len(elements), np.nan),
        "transfer_augmented": np.full(len(elements), np.nan),
    }
    fold_records = []
    for fold, (train, test) in enumerate(splitter.split(elements, outcome, groups), start=1):
        for model_name, numeric in [
            ("distance_assay", base_numeric),
            ("transfer_augmented", base_numeric + transfer_numeric),
        ]:
            processor = ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                        numeric,
                    ),
                    ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
                ]
            )
            model = Pipeline(
                [
                    ("process", processor),
                    (
                        "classify",
                        LogisticRegression(
                            C=0.5,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=args.seed + fold,
                        ),
                    ),
                ]
            )
            model.fit(elements.iloc[train], outcome[train])
            probability = model.predict_proba(elements.iloc[test])[:, 1]
            predictions[model_name][test] = probability
            if len(np.unique(outcome[test])) == 2:
                fold_records.append(
                    {"fold": fold, "model": model_name, "rows": len(test), **score(outcome[test], probability)}
                )
    output_rows = elements.copy()
    for name, probability in predictions.items():
        if not np.isfinite(probability).all():
            raise ValueError(f"Missing out-of-fold probabilities for {name}")
        output_rows[f"probability_{name}"] = probability
    summary = pd.DataFrame(
        [{"model": name, **score(outcome, probability)} for name, probability in predictions.items()]
    )
    calibration = []
    for name, probability in predictions.items():
        bins = pd.qcut(pd.Series(probability).rank(method="first"), 10, labels=False)
        part = pd.DataFrame({"bin": bins, "probability": probability, "outcome": outcome})
        grouped = part.groupby("bin", as_index=False).agg(
            rows=("outcome", "size"), predicted=("probability", "mean"), observed=("outcome", "mean")
        )
        grouped["model"] = name
        calibration.append(grouped)
    grouped_indices = [group.index.to_numpy() for _, group in output_rows.groupby("measuredGeneSymbol")]
    rng = np.random.default_rng(args.seed + 100)
    auc_difference = np.empty(2000)
    for index in range(len(auc_difference)):
        draw = rng.integers(0, len(grouped_indices), len(grouped_indices))
        sampled = np.concatenate([grouped_indices[position] for position in draw])
        sampled_outcome = outcome[sampled]
        if len(np.unique(sampled_outcome)) < 2:
            auc_difference[index] = np.nan
        else:
            auc_difference[index] = roc_auc_score(
                sampled_outcome, predictions["transfer_augmented"][sampled]
            ) - roc_auc_score(sampled_outcome, predictions["distance_assay"][sampled])
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows.to_parquet(args.output / "oof_cre_predictions.parquet", index=False)
    summary.to_csv(args.output / "model_comparison.csv", index=False)
    pd.DataFrame(fold_records).to_csv(args.output / "fold_metrics.csv", index=False)
    pd.concat(calibration, ignore_index=True).to_csv(args.output / "calibration.csv", index=False)
    pd.DataFrame({"auc_difference": auc_difference}).to_parquet(
        args.output / "target_bootstrap_auc_difference.parquet", index=False
    )
    audit = {
        "source_valid_rows": source_valid_rows,
        "source_targets": source_targets,
        "source_significant_elements": source_significant,
        "benchmark_rows": len(elements),
        "benchmark_targets": int(elements["measuredGeneSymbol"].nunique()),
        "significant_elements": int(outcome.sum()),
        "prevalence": float(outcome.mean()),
        "targets_with_transfer_features": int(elements["measuredGeneSymbol"].nunique()),
        "distance_assay_auroc": float(summary.set_index("model").at["distance_assay", "auroc"]),
        "transfer_augmented_auroc": float(summary.set_index("model").at["transfer_augmented", "auroc"]),
        "transfer_minus_distance_auc": float(
            summary.set_index("model").at["transfer_augmented", "auroc"]
            - summary.set_index("model").at["distance_assay", "auroc"]
        ),
        "transfer_minus_distance_auc_ci_95": [
            float(np.nanquantile(auc_difference, 0.025)),
            float(np.nanquantile(auc_difference, 0.975)),
        ],
    }
    (args.output / "encode_validation_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
