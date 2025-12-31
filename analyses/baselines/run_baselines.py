"""Compare prespecified transfer phenotype baselines under target-held-out folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import kneighbors_graph
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phenotypes", type=Path, default=Path("analyses/primary/results/transfer_phenotypes.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("analyses/baselines/results"))
    parser.add_argument("--config", type=Path, default=Path("config/analysis.yaml"))
    return parser.parse_args()


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rho = (
        float(spearmanr(observed, predicted).statistic)
        if np.ptp(observed) > 0 and np.ptp(predicted) > 0
        else np.nan
    )
    return {
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_rho": rho,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.models.baselines import GraphNeuralBaseline, SparseNonlinearBaseline

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["study"]["seed"])
    folds = int(config["transfer_phenotypes"]["cross_validation"]["folds"])
    table = pd.read_parquet(args.phenotypes).reset_index(drop=True)
    table["n_guides"] = table["n_guides"].astype(float)
    numeric = ["cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "n_guides"]
    categorical = ["culture_condition"]
    features = table[numeric + categorical]
    outcome = table["log1p_n_downstream"].to_numpy(dtype=float)
    groups = table["target_contrast"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)
    model_names = ["global_mean", "state_pool", "ridge", "sparse_nonlinear", "feature_graph"]
    predictions = {name: np.full(len(table), np.nan, dtype=float) for name in model_names}
    fold_records: list[dict[str, int | float | str]] = []

    for fold, (train, test) in enumerate(splitter.split(features, outcome, groups), start=1):
        processor = ColumnTransformer(
            [
                ("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), numeric),
                ("state", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ]
        )
        x_train = processor.fit_transform(features.iloc[train])
        x_test = processor.transform(features.iloc[test])
        y_train = outcome[train]
        predictions["global_mean"][test] = np.mean(y_train)

        global_mean = float(np.mean(y_train))
        state_means = table.iloc[train].assign(outcome=y_train).groupby("culture_condition")["outcome"].mean()
        state_counts = table.iloc[train].groupby("culture_condition").size()
        prior_strength = 20.0
        pooled = (
            state_means * state_counts + global_mean * prior_strength
        ) / (state_counts + prior_strength)
        predictions["state_pool"][test] = (
            table.iloc[test]["culture_condition"].map(pooled).fillna(global_mean).to_numpy()
        )

        ridge = Ridge(alpha=10.0).fit(x_train, y_train)
        predictions["ridge"][test] = ridge.predict(x_test)
        nonlinear = SparseNonlinearBaseline(seed + fold).fit(x_train, y_train)
        predictions["sparse_nonlinear"][test] = nonlinear.predict(x_test)

        train_graph = kneighbors_graph(x_train, n_neighbors=10, mode="connectivity", include_self=False)
        test_neighbors = min(10, len(test) - 1)
        test_graph = kneighbors_graph(
            x_test, n_neighbors=test_neighbors, mode="connectivity", include_self=False
        )
        graph = GraphNeuralBaseline(seed + fold).fit(x_train, train_graph, y_train)
        predictions["feature_graph"][test] = graph.predict(x_test, test_graph)

        for name in model_names:
            fold_records.append(
                {
                    "fold": fold,
                    "model": name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    **metrics(outcome[test], predictions[name][test]),
                }
            )

    prediction_table = table[["index", "target_contrast", "culture_condition"]].copy()
    prediction_table["observed"] = outcome
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Missing out-of-fold predictions for {name}")
        prediction_table[name] = values
    summary = pd.DataFrame(
        [{"model": name, **metrics(outcome, values)} for name, values in predictions.items()]
    ).sort_values("r2", ascending=False)
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_table.to_parquet(args.output / "oof_predictions.parquet", index=False)
    pd.DataFrame(fold_records).to_csv(args.output / "fold_metrics.csv", index=False)
    summary.to_csv(args.output / "model_comparison.csv", index=False)
    audit = {
        "rows": len(table),
        "targets": int(table["target_contrast"].nunique()),
        "folds": folds,
        "outcome": "log1p_n_downstream",
        "best_model_by_r2": str(summary.iloc[0]["model"]),
        "best_oof_r2": float(summary.iloc[0]["r2"]),
        "graph_scope": "transductive covariate graph without outcome propagation",
    }
    (args.output / "baseline_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
