"""Run the complete scalar specification grid and dependent central-result families."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/multiverse/results"


def fold(target: str, seed: int, folds: int) -> int:
    value = hashlib.sha256(f"{target}|{seed}".encode("utf-8")).hexdigest()[:16]
    return int(value, 16) % folds


def multiplier_summary(values: pd.Series, groups: pd.Series, seed: int, replicates: int) -> dict[str, float]:
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    cluster = frame.groupby("group")["value"].mean()
    if len(cluster) == 0:
        return {
            "estimate": np.nan,
            "standard_error": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "clusters": 0,
            "bootstrap_method": "analytically_collapsed_target_cluster_gaussian_multiplier",
            "bootstrap_replicates": replicates,
        }
    estimate = float(cluster.mean())
    standard_error = float(cluster.std(ddof=1) / np.sqrt(len(cluster))) if len(cluster) > 1 else np.nan
    rng = np.random.default_rng(seed)
    # For a scalar cluster-mean statistic, the weighted sum of independent
    # Gaussian cluster multipliers is itself Gaussian. Drawing that collapsed
    # distribution is exactly distribution-equivalent to materializing the
    # full replicate-by-cluster weight matrix, while avoiding prohibitive
    # memory use across the frozen multiverse.
    draws = estimate + standard_error * rng.standard_normal(replicates) if np.isfinite(standard_error) else np.full(replicates, np.nan)
    low, high = np.nanquantile(draws, [0.025, 0.975])
    if not np.isfinite(standard_error) or standard_error == 0:
        p_value = np.nan
    else:
        p_value = float(2 * norm.sf(abs(estimate / standard_error)))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": p_value,
        "clusters": int(len(cluster)),
        "bootstrap_method": "analytically_collapsed_target_cluster_gaussian_multiplier",
        "bootstrap_replicates": replicates,
    }


def model_for(name: str, seed: int) -> object:
    if name == "ridge":
        estimator = Ridge(alpha=10.0)
    elif name == "huber_regression":
        estimator = SGDRegressor(
            loss="huber", penalty="l2", alpha=0.001, max_iter=1000, tol=1e-4, random_state=seed
        )
    elif name == "sparse_nonlinear":
        estimator = HistGradientBoostingRegressor(
            max_iter=60, learning_rate=0.08, max_leaf_nodes=12, l2_regularization=1.0, random_state=seed
        )
    elif name == "graph_readout":
        estimator = KNeighborsRegressor(n_neighbors=15, weights="distance")
    else:
        raise ValueError(name)
    return Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("model", estimator)])


def crossfit(frame: pd.DataFrame, features: list[str], outcome: str, model: str, seed: int, folds: int) -> np.ndarray:
    assignments = frame["target_contrast"].astype(str).map(lambda value: fold(value, seed, folds)).to_numpy()
    y = frame[outcome].to_numpy(dtype=float)
    prediction = np.full(len(frame), np.nan)
    for heldout in range(folds):
        train = assignments != heldout
        test = assignments == heldout
        if train.sum() < 30 or test.sum() == 0:
            continue
        estimator = model_for(model, seed + heldout)
        estimator.fit(frame.loc[train, features], y[train])
        prediction[test] = estimator.predict(frame.loc[test, features])
    return prediction


def adjust_q(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["q_value"] = np.nan
    for family, indices in table.groupby("result_family").groups.items():
        p = table.loc[indices, "p_value"].to_numpy(dtype=float)
        valid = np.isfinite(p)
        if not valid.any():
            continue
        order = np.argsort(p[valid])
        ranked = p[valid][order]
        adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.clip(adjusted, 0, 1)
        target_indices = np.asarray(list(indices))[valid]
        table.loc[target_indices, "q_value"] = restored
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.shard_index < 0 or args.shard_index >= args.shards:
        raise ValueError("shard-index must be in [0, shards)")
    config = yaml.safe_load((ROOT / "config/multiverse.yaml").read_text(encoding="utf-8"))
    main_config = yaml.safe_load((ROOT / "config/analysis.yaml").read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["multiverse"]["seeds"]]
    replicates = int(config["multiverse"]["bootstrap_replicates"])
    context_null_replicates = int(config["multiverse"]["context_null_replicates"])
    folds = int(main_config["transfer_phenotypes"]["cross_validation"]["folds"])
    raw_path = ROOT / "work/upstream/GWT_perturbseq_analysis_2025/metadata/suppl_tables/DE_stats.suppl_table.csv"
    raw = pd.read_csv(raw_path)
    network = pd.read_parquet(ROOT / "analyses/vectors/results/network_gain_rows.parquet")
    vector_columns = ["index", "gamma_l2_debiased", "gamma_l1_significant"]
    raw = raw.merge(network[vector_columns], on="index", how="left", validate="one_to_one")
    raw["cis_magnitude"] = raw["ontarget_effect_size"].abs()
    raw["log1p_target_baseMean"] = np.log1p(raw["target_baseMean"])
    raw["log1p_n_cells_target"] = np.log1p(raw["n_cells_target"])
    raw["outcome_distal_gene_count_residual"] = np.log1p(raw["n_downstream"])
    raw["outcome_debiased_response_l2_gain"] = np.log1p(raw["gamma_l2_debiased"])
    raw["outcome_significant_response_l1_gain"] = np.log1p(raw["gamma_l1_significant"])
    state_dummies = pd.get_dummies(raw["culture_condition"], prefix="state", dtype=float)
    raw = pd.concat([raw, state_dummies], axis=1)
    covariates = {
        "minimal": ["cis_magnitude"] + list(state_dummies.columns),
        "observability": ["cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "n_guides"] + list(state_dummies.columns),
        "reproducibility": ["cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "n_guides", "guide_correlation_signif", "donor_correlation_hits_mean"] + list(state_dummies.columns),
        "no_state": ["cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "n_guides"],
    }
    outcome_map = {name: "outcome_" + name for name in config["transfer_definitions"]}
    records: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []
    start = time.perf_counter()
    processed = 0

    grid = list(product(
        config["preprocessing"]["minimum_cells"],
        config["preprocessing"]["minimum_guides"],
        config["preprocessing"]["on_target_requirement"],
        config["transfer_definitions"],
        config["model_families"],
        covariates,
        seeds,
    ))
    for specification_id, values in enumerate(grid, start=1):
        if (specification_id - 1) % args.shards != args.shard_index:
            continue
        processed += 1
        minimum_cells, minimum_guides, on_target_rule, transfer_definition, model, covariate_set, seed = values
        mask = (
            raw["ontarget_significant"].fillna(False).astype(bool)
            & ~raw["low_target_gex"].fillna(True).astype(bool)
            & ~raw["neighboring_gene_KD"].fillna(True).astype(bool)
            & ~raw["distal_offtarget_flag"].fillna(True).astype(bool)
            & (raw["n_guides"] >= minimum_guides)
            & (raw["n_cells_target"] >= minimum_cells)
            & raw["target_baseMean"].notna()
            & raw["ontarget_effect_size"].notna()
        )
        if on_target_rule == "significant_and_absolute_z_at_least_2":
            mask &= raw["cis_magnitude"] >= 2
        outcome_column = outcome_map[transfer_definition]
        frame = raw.loc[mask & raw[outcome_column].notna()].copy().reset_index(drop=True)
        descriptor = {
            "specification_id": specification_id,
            "minimum_cells": minimum_cells,
            "minimum_guides": minimum_guides,
            "on_target_rule": on_target_rule,
            "transfer_definition": transfer_definition,
            "model_family": model,
            "covariate_set": covariate_set,
            "cross_validation_seed": seed,
            "module_resolution": np.nan,
        }
        if len(frame) < 100 or frame["target_contrast"].nunique() < folds:
            unavailable.append({**descriptor, "reason": "insufficient_rows_or_targets"})
            continue
        prediction = crossfit(frame, covariates[covariate_set], outcome_column, model, seed, folds)
        valid = np.isfinite(prediction)
        frame = frame.loc[valid].copy()
        prediction = prediction[valid]
        observed = frame[outcome_column].to_numpy(dtype=float)
        residual = observed - prediction
        variance = float(np.var(observed))
        target_loss = pd.Series(np.square(observed - prediction) / variance, index=frame.index)
        gain_values = 1.0 - target_loss
        gain = multiplier_summary(gain_values, frame["target_contrast"], seed + 1, replicates)
        records.append(
            {
                **descriptor,
                "result_family": "transfer_gain",
                "tail_cutoff": np.nan,
                "rows": len(frame),
                **gain,
                "expected_direction": "positive",
            }
        )
        frame["residual"] = residual
        scale = frame.groupby("culture_condition")["residual"].transform("std").replace(0, np.nan)
        frame["transfer_z_spec"] = frame["residual"] / scale

        complete = frame.pivot_table(
            index="target_contrast", columns="culture_condition", values="transfer_z_spec", aggfunc="first"
        ).dropna()
        observed_range = complete.max(axis=1) - complete.min(axis=1)
        null_range = np.zeros(len(complete), dtype=float)
        null_rng = np.random.default_rng(seed + 2)
        matrix = complete.to_numpy(dtype=float)
        for _ in range(context_null_replicates):
            permuted = np.column_stack(
                [null_rng.permutation(matrix[:, column]) for column in range(matrix.shape[1])]
            )
            null_range += permuted.max(axis=1) - permuted.min(axis=1)
        null_range /= context_null_replicates
        switching = observed_range - null_range
        switch = multiplier_summary(
            pd.Series(switching, index=complete.index),
            pd.Series(complete.index, index=complete.index),
            seed + 3,
            replicates,
        )
        records.append(
            {
                **descriptor,
                "result_family": "context_switching",
                "tail_cutoff": np.nan,
                "rows": len(switching),
                "effect_measure": "observed_state_range_minus_state_matched_cross_target_null",
                "null_replicates": context_null_replicates,
                **switch,
                "expected_direction": "positive",
            }
        )
        for tail in config["tail_cutoffs"]:
            residual_matrix = frame.pivot_table(
                index="target_contrast", columns="culture_condition", values="residual", aggfunc="first"
            ).dropna()
            heldout_records = []
            for heldout_state in residual_matrix.columns:
                training_states = [state for state in residual_matrix.columns if state != heldout_state]
                training_score = residual_matrix[training_states].mean(axis=1)
                threshold = training_score.quantile(float(tail))
                selected_targets = training_score[training_score <= threshold].index
                heldout_records.append(
                    pd.DataFrame(
                        {
                            "target_contrast": selected_targets,
                            "heldout_state": heldout_state,
                            "heldout_residual": residual_matrix.loc[selected_targets, heldout_state].to_numpy(),
                        }
                    )
                )
            selected = pd.concat(heldout_records, ignore_index=True)
            buffered = multiplier_summary(
                selected["heldout_residual"],
                selected["target_contrast"],
                seed + int(tail * 1000),
                replicates,
            )
            records.append(
                {
                    **descriptor,
                    "result_family": "buffering",
                    "tail_cutoff": tail,
                    "rows": len(selected),
                    "effect_measure": "two_state_selected_heldout_state_residual",
                    **buffered,
                    "expected_direction": "negative",
                }
            )
        if specification_id % 100 == 0:
            print(f"processed {specification_id} scalar specifications", flush=True)

    table = adjust_q(pd.DataFrame(records))
    table["supported"] = (table["q_value"] < 0.05) & np.where(
        table["expected_direction"] == "positive", table["estimate"] > 0, table["estimate"] < 0
    )
    table["direction_expected"] = np.where(
        table["expected_direction"] == "positive", table["estimate"] > 0, table["estimate"] < 0
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.shards == 1 else f"_part{args.shard_index + 1}of{args.shards}"
    table.to_parquet(OUTPUT / f"scalar_specifications{suffix}.parquet", index=False)
    pd.DataFrame(unavailable).to_csv(OUTPUT / f"scalar_unavailable{suffix}.csv", index=False)
    family = table.groupby("result_family").agg(
        specifications=("specification_id", "size"),
        expected_direction_fraction=("direction_expected", "mean"),
        q_supported_fraction=("supported", "mean"),
        median_estimate=("estimate", "median"),
        minimum_estimate=("estimate", "min"),
        maximum_estimate=("estimate", "max"),
    ).reset_index()
    family["robustness_label"] = np.select(
        [
            (family["expected_direction_fraction"] >= 0.80) & (family["q_supported_fraction"] >= 0.50),
            family["expected_direction_fraction"] >= 0.60,
        ],
        ["broad", "mixed"],
        default="narrow",
    )
    family.to_csv(OUTPUT / f"scalar_family_summary{suffix}.csv", index=False)
    audit = {
        "scalar_base_specifications_total": len(grid),
        "scalar_base_specifications_processed": processed,
        "shard_index": args.shard_index,
        "shards": args.shards,
        "result_rows": len(table),
        "unavailable_base_specifications": len(unavailable),
        "cross_validation_folds": folds,
        "bootstrap_replicates_per_estimate": replicates,
        "wall_seconds": time.perf_counter() - start,
    }
    (OUTPUT / f"scalar_multiverse_results{suffix}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
