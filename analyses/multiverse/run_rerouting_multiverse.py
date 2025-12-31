"""Re-estimate context rerouting across frozen module resolutions and seeds."""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/multiverse/results"


def energies(scores: np.ndarray) -> np.ndarray:
    squared = np.square(scores)
    return squared / np.maximum(squared.sum(axis=1, keepdims=True), np.finfo(float).tiny)


def js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    left_term = np.zeros_like(left)
    right_term = np.zeros_like(right)
    left_positive = left > 0
    right_positive = right > 0
    left_term[left_positive] = left[left_positive] * np.log(
        left[left_positive] / np.maximum(midpoint[left_positive], 1e-300)
    )
    right_term[right_positive] = right[right_positive] * np.log(
        right[right_positive] / np.maximum(midpoint[right_positive], 1e-300)
    )
    return 0.5 * (left_term.sum(axis=1) + right_term.sum(axis=1))


def multiplier(values: np.ndarray, groups: np.ndarray, seed: int, replicates: int) -> dict[str, float]:
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    cluster = frame.groupby("group")["value"].mean()
    estimate = float(cluster.mean())
    standard_error = float(cluster.std(ddof=1) / np.sqrt(len(cluster)))
    draws = estimate + standard_error * np.random.default_rng(seed).standard_normal(replicates)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(2 * norm.sf(abs(estimate / standard_error))) if standard_error > 0 else np.nan,
        "clusters": len(cluster),
        "bootstrap_method": "analytically_collapsed_target_cluster_gaussian_multiplier",
        "bootstrap_replicates": replicates,
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "config/multiverse.yaml").read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["multiverse"]["seeds"]]
    resolutions = [int(value) for value in config["module_resolutions"]]
    replicates = int(config["multiverse"]["bootstrap_replicates"])
    rows = pd.read_parquet(ROOT / "data/interim/gwt_vectors/rows.parquet").reset_index(drop=True)
    matrix = sparse.load_npz(ROOT / "data/interim/gwt_vectors/normalized_significant_logfc.npz").tocsr()
    pairs = pd.read_parquet(ROOT / "analyses/vectors/results/context_rerouting_pairs.parquet")
    left = pairs["left_index"].to_numpy(dtype=int)
    right = pairs["right_index"].to_numpy(dtype=int)
    distance = 1.0 - pairs["response_cosine"].to_numpy(dtype=float)
    records = []
    start = time.perf_counter()
    spec = 0
    for resolution, seed in product(resolutions, seeds):
        model = TruncatedSVD(n_components=resolution, random_state=seed)
        module_energy = energies(model.fit_transform(matrix))
        divergence = js(module_energy[left], module_energy[right])
        for minimum_cells, minimum_guides, on_target_rule in product(
            config["preprocessing"]["minimum_cells"],
            config["preprocessing"]["minimum_guides"],
            config["preprocessing"]["on_target_requirement"],
        ):
            spec += 1
            valid_rows = (
                rows["ontarget_significant"].fillna(False).astype(bool)
                & ~rows["low_target_gex"].fillna(True).astype(bool)
                & ~rows["neighboring_gene_KD"].fillna(True).astype(bool)
                & ~rows["distal_offtarget_flag"].fillna(True).astype(bool)
                & (rows["n_cells_target"] >= minimum_cells)
                & (rows["n_guides"] >= minimum_guides)
            ).to_numpy()
            if on_target_rule == "significant_and_absolute_z_at_least_2":
                valid_rows &= rows["ontarget_effect_size"].abs().to_numpy() >= 2
            valid = valid_rows[left] & valid_rows[right] & np.isfinite(distance) & np.isfinite(divergence)
            if valid.sum() < 30:
                continue
            z_divergence = (divergence[valid] - divergence[valid].mean()) / divergence[valid].std()
            z_distance = (distance[valid] - distance[valid].mean()) / distance[valid].std()
            contribution = z_divergence * z_distance
            result = multiplier(
                contribution,
                pairs.loc[valid, "target_contrast"].astype(str).to_numpy(),
                seed + spec,
                replicates,
            )
            records.append(
                {
                    "specification_id": spec,
                    "result_family": "rerouting",
                    "minimum_cells": minimum_cells,
                    "minimum_guides": minimum_guides,
                    "on_target_rule": on_target_rule,
                    "module_resolution": resolution,
                    "cross_validation_seed": seed,
                    "rows": int(valid.sum()),
                    "expected_direction": "positive",
                    **result,
                }
            )
        print(f"completed module resolution {resolution}, seed {seed}", flush=True)
    table = pd.DataFrame(records)
    p = table["p_value"].to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    table["q_value"] = np.empty_like(adjusted)
    table.loc[order, "q_value"] = np.clip(adjusted, 0, 1)
    table["direction_expected"] = table["estimate"] > 0
    table["supported"] = table["direction_expected"] & (table["q_value"] < 0.05)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT / "rerouting_specifications.parquet", index=False)
    summary = {
        "specifications": len(table),
        "expected_direction_fraction": float(table["direction_expected"].mean()),
        "q_supported_fraction": float(table["supported"].mean()),
        "median_estimate": float(table["estimate"].median()),
        "estimate_range": [float(table["estimate"].min()), float(table["estimate"].max())],
        "wall_seconds": time.perf_counter() - start,
    }
    (OUTPUT / "rerouting_multiverse_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
