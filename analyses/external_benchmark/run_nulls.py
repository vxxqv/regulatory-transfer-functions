"""Run matched feature permutations and exact directed degree-preserving rewiring."""

from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import kneighbors_graph
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyses.external_benchmark.run_benchmark import ROOT, feature_table


OUTPUT = ROOT / "analyses/external_benchmark/results"


def ridge_prediction(
    table: pd.DataFrame, columns: list[str], outcome: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    prediction = np.full(len(table), np.nan)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        model = Pipeline(
            [("impute", SimpleImputer()), ("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))]
        )
        model.fit(table.loc[train, columns], outcome[train])
        prediction[test] = model.predict(table.loc[test, columns])
    return prediction


def graph_prediction(graph: nx.DiGraph, outcome: np.ndarray, folds: np.ndarray) -> np.ndarray:
    prediction = np.full(len(outcome), np.nan)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        global_mean = float(outcome[train].mean())
        for node in np.flatnonzero(folds == fold):
            neighbors = [neighbor for neighbor in graph.successors(int(node)) if train[neighbor]]
            prediction[node] = float(outcome[neighbors].mean()) if neighbors else global_mean
    return prediction


def main() -> None:
    config = yaml.safe_load((ROOT / "config/external_benchmark.yaml").read_text(encoding="utf-8"))
    seed = int(config["benchmark"]["seed"])
    replicates = int(config["benchmark"]["permutation_replicates"])
    rewirings = int(config["benchmark"]["network_rewirings"])
    table, blocks = feature_table()
    outcome = table["log1p_differentially_expressed_genes"].to_numpy(dtype=float)
    folds = table["fold"].to_numpy(dtype=int)
    cd4 = blocks["proximal"] + blocks["scalar_transfer"] + blocks["context"] + blocks["vector"] + blocks["network"]
    columns = blocks["simple"] + cd4
    observed_full = ridge_prediction(table, columns, outcome, folds)
    observed_simple = ridge_prediction(table, blocks["simple"], outcome, folds)
    observed_delta = r2_score(outcome, observed_full) - r2_score(outcome, observed_simple)

    expression_bin = pd.qcut(table["control_target_expression"], 5, duplicates="drop").astype(str)
    degree_bin = pd.qcut(table["n_differentially_expressed_genes"], 5, duplicates="drop").astype(str)
    strata = expression_bin + "|" + degree_bin
    groups = [np.asarray(index, dtype=int) for index in table.groupby(strata, observed=False).groups.values()]
    rng = np.random.default_rng(seed + 1200)
    permutation_records = []
    permutation_start = time.perf_counter()
    for replicate in range(replicates):
        permuted = table.copy()
        order = np.arange(len(table))
        for group in groups:
            order[group] = rng.permutation(group)
        permuted.loc[:, cd4] = table.loc[order, cd4].to_numpy()
        prediction = ridge_prediction(permuted, columns, outcome, folds)
        permutation_records.append(
            {"replicate": replicate + 1, "delta_r2": r2_score(outcome, prediction) - r2_score(outcome, observed_simple)}
        )
    permutations = pd.DataFrame(permutation_records)
    matched_p = float((1 + (permutations["delta_r2"] >= observed_delta).sum()) / (replicates + 1))

    module_data = SimpleImputer().fit_transform(table[blocks["vector"]])
    module_data = StandardScaler().fit_transform(module_data)
    adjacency = kneighbors_graph(module_data, n_neighbors=15, mode="connectivity", include_self=False)
    graph = nx.from_scipy_sparse_array(adjacency, create_using=nx.DiGraph)
    observed_graph_prediction = graph_prediction(graph, outcome, folds)
    observed_graph_r2 = float(r2_score(outcome, observed_graph_prediction))
    rewiring_records = []
    rewiring_start = time.perf_counter()
    swaps = 3 * graph.number_of_edges()
    for replicate in range(rewirings):
        rewired = graph.copy()
        try:
            nx.directed_edge_swap(
                rewired,
                nswap=swaps,
                max_tries=swaps * 20,
                seed=seed + 5000 + replicate,
            )
            status = "ok"
        except nx.NetworkXAlgorithmError:
            status = "incomplete_swap"
        prediction = graph_prediction(rewired, outcome, folds)
        rewiring_records.append(
            {"replicate": replicate + 1, "r2": r2_score(outcome, prediction), "status": status}
        )
    rewiring = pd.DataFrame(rewiring_records)
    rewiring_p = float((1 + (rewiring["r2"] >= observed_graph_r2).sum()) / (rewirings + 1))

    permutations.to_parquet(OUTPUT / "matched_permutation_null.parquet", index=False)
    rewiring.to_parquet(OUTPUT / "degree_preserving_rewiring_null.parquet", index=False)
    summary = pd.DataFrame(
        [
            {
                "null": "matched_cd4_feature_permutation",
                "replicates": replicates,
                "observed": observed_delta,
                "null_median": permutations["delta_r2"].median(),
                "null_95_low": permutations["delta_r2"].quantile(0.025),
                "null_95_high": permutations["delta_r2"].quantile(0.975),
                "one_sided_p": matched_p,
                "wall_seconds": time.perf_counter() - permutation_start,
            },
            {
                "null": "directed_degree_preserving_rewiring",
                "replicates": rewirings,
                "observed": observed_graph_r2,
                "null_median": rewiring["r2"].median(),
                "null_95_low": rewiring["r2"].quantile(0.025),
                "null_95_high": rewiring["r2"].quantile(0.975),
                "one_sided_p": rewiring_p,
                "wall_seconds": time.perf_counter() - rewiring_start,
            },
        ]
    )
    summary.to_csv(OUTPUT / "null_summary.csv", index=False)


if __name__ == "__main__":
    main()
