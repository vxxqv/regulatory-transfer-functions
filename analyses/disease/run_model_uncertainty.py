from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "disease_model_uncertainty.yaml"
RESULTS = ROOT / "analyses" / "disease" / "results"
INPUT = RESULTS / "disease_model_rows.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auroc(observed: np.ndarray, predicted: np.ndarray) -> float:
    positives = observed == 1
    positive_count = int(positives.sum())
    negative_count = len(observed) - positive_count
    if positive_count == 0 or negative_count == 0:
        return np.nan
    ranks = pd.Series(predicted).rank(method="average").to_numpy()
    return float((ranks[positives].sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def average_precision(observed: np.ndarray, predicted: np.ndarray) -> float:
    order = np.argsort(-predicted, kind="mergesort")
    sorted_observed = observed[order]
    sorted_predicted = predicted[order]
    positives = int(sorted_observed.sum())
    if positives == 0:
        return np.nan
    threshold_indices = np.r_[np.flatnonzero(np.diff(sorted_predicted)), len(sorted_predicted) - 1]
    true_positives = np.cumsum(sorted_observed)[threshold_indices]
    precision = true_positives / (threshold_indices + 1)
    recall = true_positives / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def metrics(observed: np.ndarray, size_state: np.ndarray, transfer: np.ndarray) -> dict[str, float]:
    return {
        "auroc": auroc(observed, transfer) - auroc(observed, size_state),
        "average_precision": average_precision(observed, transfer) - average_precision(observed, size_state),
        "brier": float(np.mean(np.square(observed - size_state)) - np.mean(np.square(observed - transfer))),
    }


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if sha256(INPUT) != config["input_sha256"]:
        raise RuntimeError("Disease model input hash does not match the frozen configuration")
    frame = pd.read_parquet(INPUT).sort_values(["cluster", "disease", "culture_condition"]).reset_index(drop=True)
    observed = frame["significant"].astype(int).to_numpy()
    size_state = frame["probability_size_state"].to_numpy(dtype=float)
    transfer = frame["probability_transfer_augmented"].to_numpy(dtype=float)
    clusters = frame["cluster"].to_numpy()
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 50 or len(np.unique(observed)) != 2:
        raise RuntimeError("Disease model uncertainty minimum-power gate failed")
    cluster_rows = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    point = metrics(observed, size_state, transfer)
    rng = np.random.default_rng(int(config["seed"]))
    bootstrap_records = []
    bootstrap_replicates = int(config["bootstrap_replicates"])
    permutation_replicates = int(config["permutation_replicates"])
    for replicate in range(bootstrap_replicates):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        rows = np.concatenate([cluster_rows[cluster] for cluster in sampled])
        values = metrics(observed[rows], size_state[rows], transfer[rows])
        bootstrap_records.extend({"replicate": replicate, "metric": metric, "effect": value} for metric, value in values.items())
    bootstrap = pd.DataFrame(bootstrap_records)
    null_records = []
    for replicate in range(permutation_replicates):
        swapped = set(rng.choice(unique_clusters, size=rng.binomial(len(unique_clusters), 0.5), replace=False))
        mask = np.fromiter((cluster in swapped for cluster in clusters), dtype=bool, count=len(clusters))
        first = np.where(mask, transfer, size_state)
        second = np.where(mask, size_state, transfer)
        values = metrics(observed, first, second)
        null_records.extend({"replicate": replicate, "metric": metric, "effect": value} for metric, value in values.items())
    null = pd.DataFrame(null_records)
    records = []
    for metric in ["auroc", "average_precision", "brier"]:
        boot = bootstrap.loc[bootstrap["metric"].eq(metric), "effect"].dropna().to_numpy()
        permuted = null.loc[null["metric"].eq(metric), "effect"].dropna().to_numpy()
        estimate = point[metric]
        records.append(
            {
                "metric": metric,
                "effect_definition": config["effect_direction"][metric],
                "estimate": estimate,
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "p_two_sided": float((1 + np.sum(np.abs(permuted) >= abs(estimate))) / (len(permuted) + 1)),
                "rows": len(frame),
                "clusters": len(unique_clusters),
                "diseases": int(frame["disease"].nunique()),
                "bootstrap_replicates": len(boot),
                "permutation_replicates": len(permuted),
            }
        )
    summary = pd.DataFrame(records)
    summary["q_two_sided"] = bh(summary["p_two_sided"].to_numpy())
    summary["decision"] = np.select(
        [
            (summary["ci_low"] > 0) & (summary["q_two_sided"] < 0.05),
            (summary["ci_high"] < 0) & (summary["q_two_sided"] < 0.05),
        ],
        ["transfer_augmented_superior", "size_state_superior"],
        default="unresolved",
    )
    summary_path = RESULTS / "model_uncertainty.csv"
    null_path = RESULTS / "model_null_distribution.parquet"
    summary.to_csv(summary_path, index=False)
    null.to_parquet(null_path, index=False)
    audit = {
        "analysis": "frozen_cluster_paired_disease_model_uncertainty",
        "rows": len(frame),
        "clusters": int(len(unique_clusters)),
        "diseases": int(frame["disease"].nunique()),
        "positive_rows": int(observed.sum()),
        "config_sha256": sha256(CONFIG),
        "input_sha256": sha256(INPUT),
        "point_estimates_reproduced": point,
        "all_bootstrap_replicates_valid": bool(bootstrap.groupby("metric")["effect"].count().eq(bootstrap_replicates).all()),
        "multiplicity_family": "two-sided tests across three frozen performance metrics",
        "model_refitting": False,
        "output_sha256": {
            "model_uncertainty.csv": sha256(summary_path),
            "model_null_distribution.parquet": sha256(null_path),
        },
    }
    (RESULTS / "disease_uncertainty_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
