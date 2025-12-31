"""Run the frozen RPE1 model competition, ablations, nulls, and calibration audit."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor, kneighbors_graph
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/external_benchmark/results"
LEVELS = (0.50, 0.80, 0.95)


def hash_bucket(value: str, modulo: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % modulo


def numeric_pipeline(model: object) -> Pipeline:
    return Pipeline(
        [("impute", SimpleImputer()), ("scale", StandardScaler()), ("model", model)]
    )


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rho = spearmanr(observed, predicted).statistic if np.ptp(predicted) > 0 else np.nan
    return {
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_rho": float(rho),
    }


def feature_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    external = pd.read_parquet(
        ROOT / "analyses/external_benchmark/rpe1_prepared/eligible_targets.parquet"
    )
    phenotypes = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    context = pd.read_parquet(ROOT / "analyses/primary/results/context_switching.parquet")
    network = pd.read_parquet(ROOT / "analyses/vectors/results/network_gain_rows.parquet")
    tensor = pd.read_parquet(ROOT / "analyses/vectors/results/contextual_transfer_tensor.parquet")

    proximal_columns = ["cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "n_guides"]
    proximal = phenotypes.groupby("target_contrast_gene_name")[proximal_columns].median().add_prefix("cd4_")
    scalar = phenotypes.groupby("target_contrast_gene_name")[["transfer_residual", "transfer_z"]].agg(
        ["mean", "median", "std", "min", "max"]
    )
    scalar.columns = ["cd4_" + "_".join(column) for column in scalar.columns]
    classes = pd.crosstab(
        phenotypes["target_contrast_gene_name"], phenotypes["transfer_class"], normalize="index"
    ).add_prefix("cd4_class_fraction_")
    context_features = context.set_index("target_contrast_gene_name")[
        ["residual_range", "residual_sd", "cluster_entropy", "cluster_switch"]
    ].rename(columns=lambda value: "cd4_context_" + value)
    context_features["cd4_context_cluster_switch"] = context_features[
        "cd4_context_cluster_switch"
    ].astype(float)
    network_features = network.groupby("target_contrast_gene_name")[
        ["gamma_l2_debiased", "effective_response_dimension", "sign_balance", "dominant_module_fraction"]
    ].median().add_prefix("cd4_network_")
    modules = tensor.pivot_table(
        index="target_gene", columns="module", values="score", aggfunc="mean"
    )
    modules.columns = [f"cd4_module_{int(column):02d}" for column in modules.columns]
    modules.index.name = "target_contrast_gene_name"

    table = external.set_index("target_gene").join(
        [proximal, scalar, classes, context_features, network_features, modules], how="left"
    ).reset_index()
    table["external_log1p_cells"] = np.log1p(table["n_cells"])
    table["external_abs_on_target"] = table["on_target_mean_difference"].abs()
    simple = ["external_log1p_cells", "control_target_expression", "external_abs_on_target"]
    blocks = {
        "simple": simple,
        "proximal": [column for column in table if column.startswith("cd4_cis_") or column.startswith("cd4_log1p_") or column == "cd4_n_guides"],
        "scalar_transfer": [column for column in table if column.startswith("cd4_transfer_") or column.startswith("cd4_class_")],
        "context": [column for column in table if column.startswith("cd4_context_")],
        "vector": [column for column in table if column.startswith("cd4_module_")],
        "network": [column for column in table if column.startswith("cd4_network_")],
    }
    if any(not columns for name, columns in blocks.items() if name != "simple"):
        raise ValueError(f"At least one frozen CD4 feature block is empty: {blocks}")
    return table, blocks


def predict_model(
    name: str,
    table: pd.DataFrame,
    blocks: dict[str, list[str]],
    train: np.ndarray,
    test: np.ndarray,
    outcome: np.ndarray,
    seed: int,
) -> np.ndarray:
    all_cd4 = blocks["proximal"] + blocks["scalar_transfer"] + blocks["context"] + blocks["vector"] + blocks["network"]
    feature_sets = {
        "simple_covariates": blocks["simple"],
        "ridge": blocks["simple"] + blocks["proximal"],
        "sparse_nonlinear": blocks["simple"] + all_cd4,
        "graph_readout": blocks["simple"] + all_cd4,
        "frozen_cd4_transfer": blocks["simple"] + all_cd4,
        "without_proximal": blocks["simple"] + [c for c in all_cd4 if c not in blocks["proximal"]],
        "without_scalar_transfer": blocks["simple"] + [c for c in all_cd4 if c not in blocks["scalar_transfer"]],
        "without_context": blocks["simple"] + [c for c in all_cd4 if c not in blocks["context"]],
        "without_vector": blocks["simple"] + [c for c in all_cd4 if c not in blocks["vector"]],
        "without_network": blocks["simple"] + [c for c in all_cd4 if c not in blocks["network"]],
    }
    if name == "global_mean":
        return np.full(len(test), outcome[train].mean())
    if name == "standard_network_propagation":
        columns = blocks["vector"]
        imputer = SimpleImputer()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(table.iloc[train][columns]))
        x_test = scaler.transform(imputer.transform(table.iloc[test][columns]))
        model = KNeighborsRegressor(n_neighbors=min(15, len(train)), weights="distance")
        model.fit(x_train, outcome[train])
        return model.predict(x_test)

    columns = feature_sets[name]
    if name == "sparse_nonlinear":
        estimator = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
        )
    elif name == "graph_readout":
        imputer = SimpleImputer()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(table.iloc[train][columns]))
        x_test = scaler.transform(imputer.transform(table.iloc[test][columns]))
        train_graph = kneighbors_graph(x_train, n_neighbors=min(10, len(train) - 1), include_self=False)
        test_graph = kneighbors_graph(x_test, n_neighbors=min(10, len(test) - 1), include_self=False)
        def propagate(x: np.ndarray, adjacency: sparse.spmatrix) -> np.ndarray:
            adjacency = sparse.csr_matrix(adjacency) + sparse.eye(len(x), format="csr")
            degree = np.asarray(adjacency.sum(axis=1)).ravel()
            inverse = sparse.diags(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
            normalized = inverse @ adjacency @ inverse
            first = normalized @ x
            second = normalized @ first
            return np.hstack([x, np.asarray(first), np.asarray(second)])
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=500, early_stopping=True, random_state=seed
        )
        model.fit(propagate(x_train, train_graph), outcome[train])
        return model.predict(propagate(x_test, test_graph))
    else:
        estimator = Ridge(alpha=10.0)
    model = numeric_pipeline(estimator)
    model.fit(table.iloc[train][columns], outcome[train])
    return model.predict(table.iloc[test][columns])


def target_bootstrap(
    observed: np.ndarray, prediction: np.ndarray, replicates: int, seed: int
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    values = {"r2": [], "mae": [], "spearman_rho": []}
    for _ in range(replicates):
        draw = rng.integers(0, len(observed), len(observed))
        result = metrics(observed[draw], prediction[draw])
        for key in values:
            values[key].append(result[key])
    return {key: tuple(np.nanquantile(value, [0.025, 0.975])) for key, value in values.items()}


def main() -> None:
    config = yaml.safe_load((ROOT / "config/external_benchmark.yaml").read_text(encoding="utf-8"))
    seed = int(config["benchmark"]["seed"])
    bootstraps = int(config["benchmark"]["bootstrap_replicates"])
    table, blocks = feature_table()
    outcome = table["log1p_differentially_expressed_genes"].to_numpy(dtype=float)
    folds = table["fold"].to_numpy(dtype=int)
    model_names = list(config["models"]) + list(config["ablations"])
    predictions = {name: np.full(len(table), np.nan) for name in model_names}
    intervals = {(name, level): (np.full(len(table), np.nan), np.full(len(table), np.nan)) for name in model_names for level in LEVELS}
    compute_records: list[dict[str, object]] = []

    for name in model_names:
        start = time.perf_counter()
        tracemalloc.start()
        for fold in sorted(np.unique(folds)):
            test = np.flatnonzero(folds == fold)
            train_all = np.flatnonzero(folds != fold)
            calibration_mask = np.array(
                [hash_bucket(str(table.iloc[index]["target_gene"]) + "|cal", 5) == 0 for index in train_all]
            )
            calibration = train_all[calibration_mask]
            proper_train = train_all[~calibration_mask]
            test_prediction = predict_model(name, table, blocks, proper_train, test, outcome, seed + fold)
            calibration_prediction = predict_model(
                name, table, blocks, proper_train, calibration, outcome, seed + fold
            )
            predictions[name][test] = test_prediction
            residual = np.abs(outcome[calibration] - calibration_prediction)
            for level in LEVELS:
                rank = min(len(residual) - 1, int(np.ceil((len(residual) + 1) * level)) - 1)
                radius = float(np.sort(residual)[rank])
                intervals[(name, level)][0][test] = test_prediction - radius
                intervals[(name, level)][1][test] = test_prediction + radius
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        compute_records.append(
            {"model": name, "wall_seconds": time.perf_counter() - start, "peak_memory_mb": peak / 1024**2}
        )

    prediction_rows = table[["target_gene", "fold", "n_cells", "log1p_differentially_expressed_genes"]].copy()
    summaries = []
    calibration_records = []
    for index, name in enumerate(model_names):
        if not np.isfinite(predictions[name]).all():
            raise ValueError(f"Incomplete predictions for {name}")
        prediction_rows[name] = predictions[name]
        result = metrics(outcome, predictions[name])
        ci = target_bootstrap(outcome, predictions[name], bootstraps, seed + index)
        summaries.append(
            {
                "model": name,
                **result,
                **{f"{metric}_ci_low": bounds[0] for metric, bounds in ci.items()},
                **{f"{metric}_ci_high": bounds[1] for metric, bounds in ci.items()},
            }
        )
        design = np.column_stack([np.ones(len(table)), predictions[name]])
        intercept, slope = np.linalg.lstsq(design, outcome, rcond=None)[0]
        for level in LEVELS:
            lower, upper = intervals[(name, level)]
            calibration_records.append(
                {
                    "model": name,
                    "nominal_coverage": level,
                    "empirical_coverage": float(np.mean((outcome >= lower) & (outcome <= upper))),
                    "mean_interval_width": float(np.mean(upper - lower)),
                    "calibration_intercept": float(intercept),
                    "calibration_slope": float(slope),
                }
            )

    summary = pd.DataFrame(summaries).sort_values("r2", ascending=False)
    simple_r2 = float(summary.loc[summary["model"] == "simple_covariates", "r2"].iloc[0])
    full_prediction = predictions["frozen_cd4_transfer"]
    simple_prediction = predictions["simple_covariates"]
    rng = np.random.default_rng(seed + 900)
    paired_delta = np.empty(bootstraps)
    for replicate in range(bootstraps):
        draw = rng.integers(0, len(table), len(table))
        paired_delta[replicate] = r2_score(outcome[draw], full_prediction[draw]) - r2_score(
            outcome[draw], simple_prediction[draw]
        )
    delta_ci = np.quantile(paired_delta, [0.025, 0.975])
    full_r2 = float(summary.loc[summary["model"] == "frozen_cd4_transfer", "r2"].iloc[0])
    delta = full_r2 - simple_r2
    minimal = float(config["benchmark"]["minimally_relevant_delta_r2"])
    decision = "passed" if delta_ci[0] > minimal else "failed" if delta_ci[1] < minimal else "unresolved"
    hypotheses = pd.DataFrame(
        [
            {
                "hypothesis": "Frozen CD4 transfer features improve external RPE1 prediction beyond simple covariates",
                "primary_metric": "delta_r2",
                "estimate": delta,
                "ci_low": delta_ci[0],
                "ci_high": delta_ci[1],
                "minimally_relevant_effect": minimal,
                "decision": decision,
            }
        ]
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_rows.to_parquet(OUTPUT / "all_eligible_target_predictions.parquet", index=False)
    summary.to_csv(OUTPUT / "model_comparison.csv", index=False)
    pd.DataFrame(calibration_records).to_csv(OUTPUT / "calibration_coverage.csv", index=False)
    pd.DataFrame(compute_records).to_csv(OUTPUT / "computational_cost.csv", index=False)
    hypotheses.to_csv(OUTPUT / "hypothesis_decisions.csv", index=False)
    pd.DataFrame({"delta_r2": paired_delta}).to_parquet(OUTPUT / "paired_target_bootstrap.parquet", index=False)
    audit = {
        "eligible_targets": len(table),
        "all_eligible_targets_reported": True,
        "models": model_names,
        "folds": sorted(np.unique(folds).tolist()),
        "primary_delta_r2": delta,
        "primary_delta_r2_ci_95": delta_ci.tolist(),
        "primary_decision": decision,
        "pending_null_analyses": ["matched_permutations", "degree_preserving_network_rewiring"],
    }
    (OUTPUT / "benchmark_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
