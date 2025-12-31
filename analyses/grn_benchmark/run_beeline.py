"""Benchmark fixed network-inference procedures on official BEELINE v4 data."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/beeline_v4/data/BEELINE-data/inputs/scRNA-Seq"
NETWORKS = ROOT / "work/beeline_v4/networks/Networks"
OUTPUT = ROOT / "analyses/grn_benchmark/results"
PANEL = {
    "hESC": ("human", "hESC-ChIP-seq-network.csv", "human-tfs.csv"),
    "hHep": ("human", "HepG2-ChIP-seq-network.csv", "human-tfs.csv"),
    "mDC": ("mouse", "mDC-ChIP-seq-network.csv", "mouse-tfs.csv"),
    "mESC": ("mouse", "mESC-ChIP-seq-network.csv", "mouse-tfs.csv"),
    "mHSC-E": ("mouse", "mHSC-ChIP-seq-network.csv", "mouse-tfs.csv"),
}


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positives = int(labels.sum())
    order = np.argsort(scores)[::-1]
    top = labels[order[: max(positives, 1)]].mean()
    prevalence = labels.mean()
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "early_precision_ratio": float(top / prevalence) if prevalence > 0 else np.nan,
    }


def target_bootstrap(edges: pd.DataFrame, replicates: int, seed: int) -> dict[str, tuple[float, float]]:
    targets = edges["target"].unique()
    by_target = {target: block.index.to_numpy() for target, block in edges.groupby("target")}
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "auprc": [], "early_precision_ratio": []}
    for _ in range(replicates):
        draw = rng.choice(targets, len(targets), replace=True)
        index = np.concatenate([by_target[target] for target in draw])
        result = metrics(edges.loc[index, "gold"].to_numpy(), edges.loc[index, "score"].to_numpy())
        for key in values:
            values[key].append(result[key])
    return {key: tuple(np.quantile(value, [0.025, 0.975])) for key, value in values.items()}


def select_genes(dataset: str, expression: pd.DataFrame, tf_set: set[str]) -> list[str]:
    ordering = pd.read_csv(DATA / dataset / "GeneOrdering.csv", index_col=0)
    ordering.index = ordering.index.astype(str)
    available = ordering.index.intersection(expression.index)
    ordering = ordering.loc[available]
    threshold = 0.01 / max(len(ordering), 1)
    significant_tfs = ordering.index[(ordering["VGAMpValue"] < threshold) & ordering.index.isin(tf_set)]
    non_tfs = ordering.loc[~ordering.index.isin(tf_set)].nlargest(500, "Variance").index
    return sorted(set(significant_tfs).union(non_tfs))


def finite_horizon(operator: np.ndarray, depth: int = 3) -> np.ndarray:
    eigenvalues = np.linalg.eigvals(operator)
    radius = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 0.0
    stable = operator * (0.95 / radius) if radius > 0.95 else operator.copy()
    result = stable.copy()
    current = stable.copy()
    for _ in range(1, depth):
        current = current @ stable
        result += current
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_edges = []
    summary_records = []
    selection_records = []
    for dataset_index, (dataset, (species, network_file, tf_file)) in enumerate(PANEL.items()):
        expression = pd.read_csv(DATA / dataset / "ExpressionData.csv", index_col=0)
        expression.index = expression.index.astype(str)
        pseudotime = pd.read_csv(DATA / dataset / "PseudoTime.csv", index_col=0).iloc[:, 0]
        pseudotime = pseudotime.reindex(expression.columns)
        tf_set = set(pd.read_csv(ROOT / "work/beeline_v4/networks" / tf_file)["TF"].astype(str))
        genes = select_genes(dataset, expression, tf_set)
        regulators = sorted(set(genes).intersection(tf_set))
        gold_source = pd.read_csv(NETWORKS / species / network_file)
        gold = set(zip(gold_source["Gene1"].astype(str), gold_source["Gene2"].astype(str)))
        x = expression.loc[genes].to_numpy(dtype=float).T
        order = np.argsort(pseudotime.to_numpy(dtype=float))
        x_ordered = x[order]
        current = x_ordered[:-1]
        following = x_ordered[1:]
        gene_index = {gene: index for index, gene in enumerate(genes)}
        tf_indices = np.array([gene_index[gene] for gene in regulators], dtype=int)
        selection_records.append(
            {"dataset": dataset, "cells": x.shape[0], "selected_genes": len(genes), "regulators": len(regulators), "candidate_edges": len(regulators) * (len(genes) - 1)}
        )

        method_scores = {}
        start = time.perf_counter()
        tracemalloc.start()
        correlation = np.corrcoef(x, rowvar=False)
        method_scores["absolute_pearson"] = np.abs(correlation[:, tf_indices].T)
        _, peak_pearson = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing = {"absolute_pearson": (time.perf_counter() - start, peak_pearson)}

        start = time.perf_counter()
        tracemalloc.start()
        lag = np.empty((len(regulators), len(genes)))
        for regulator_position, regulator_index in enumerate(tf_indices):
            left = current[:, regulator_index]
            left_centered = left - left.mean()
            right_centered = following - following.mean(axis=0)
            denominator = np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum(axis=0))
            lag[regulator_position] = np.divide(left_centered @ right_centered, denominator, out=np.zeros(len(genes)), where=denominator > 0)
        method_scores["pseudotime_lagged_correlation"] = np.abs(lag)
        _, peak_lag = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing["pseudotime_lagged_correlation"] = (time.perf_counter() - start, peak_lag)

        start = time.perf_counter()
        tracemalloc.start()
        transition = Ridge(alpha=10.0).fit(current, following).coef_
        method_scores["ridge_transition"] = np.abs(transition[:, tf_indices].T)
        _, peak_ridge = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing["ridge_transition"] = (time.perf_counter() - start, peak_ridge)

        start = time.perf_counter()
        tracemalloc.start()
        propagated = finite_horizon(transition)
        method_scores["finite_horizon_transfer"] = np.abs(propagated[:, tf_indices].T)
        _, peak_transfer = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing["finite_horizon_transfer"] = (time.perf_counter() - start, peak_transfer)

        for method, score_matrix in method_scores.items():
            rows = []
            for regulator_position, regulator in enumerate(regulators):
                for target_position, target in enumerate(genes):
                    if regulator == target:
                        continue
                    rows.append(
                        {"dataset": dataset, "method": method, "regulator": regulator, "target": target, "score": score_matrix[regulator_position, target_position], "gold": int((regulator, target) in gold)}
                    )
            edges = pd.DataFrame(rows)
            result = metrics(edges["gold"].to_numpy(), edges["score"].to_numpy())
            ci = target_bootstrap(edges, 2000, 20260912 + dataset_index * 10 + len(summary_records))
            wall, peak = timing[method]
            summary_records.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "edges": len(edges),
                    "gold_edges": int(edges["gold"].sum()),
                    **result,
                    **{f"{key}_ci_low": value[0] for key, value in ci.items()},
                    **{f"{key}_ci_high": value[1] for key, value in ci.items()},
                    "wall_seconds": wall,
                    "peak_memory_mb": peak / 1024**2,
                }
            )
            all_edges.append(edges)
        print(f"completed {dataset}", flush=True)
    pd.concat(all_edges, ignore_index=True).to_parquet(OUTPUT / "all_candidate_edges.parquet", index=False)
    pd.DataFrame(summary_records).to_csv(OUTPUT / "method_results.csv", index=False)
    pd.DataFrame(selection_records).to_csv(OUTPUT / "dataset_selection.csv", index=False)
    audit = {
        "datasets": len(PANEL),
        "methods": 4,
        "gold_used_for_training_or_selection": False,
        "all_candidate_edges_reported": True,
        "bootstrap_replicates": 2000,
    }
    (OUTPUT / "beeline_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
