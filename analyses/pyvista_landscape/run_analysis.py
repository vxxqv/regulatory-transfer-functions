"""Derive the frozen statistics and source tables for Supplementary Figure S24."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.linalg import eigh
from scipy.stats import chi2, gaussian_kde, spearmanr


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/pyvista_transfer_landscape.yaml"
TRANSFER_PATH = ROOT / "analyses/primary/results/transfer_phenotypes.parquet"
PAIR_PATH = ROOT / "analyses/vectors/results/context_rerouting_pairs.parquet"
OUTPUT = ROOT / "figures/supplement/S24/figure_data"
AUDIT = ROOT / "analyses/pyvista_landscape/results"


def bh_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def scale_value(values: pd.Series | np.ndarray, lower: float, upper: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return (array - lower) / (upper - lower)


def complete_cohort(transfer: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    required = set(states)
    observed = transfer.groupby("target_contrast")["culture_condition"].agg(lambda values: set(values))
    complete = observed[observed.map(lambda values: values == required)].index
    cohort = transfer[transfer["target_contrast"].isin(complete)].copy()
    cohort = cohort.sort_values(["target_contrast", "culture_condition"], kind="stable")
    if cohort.duplicated(["target_contrast", "culture_condition"]).any():
        raise ValueError("Complete cohort contains duplicate target-state rows")
    return cohort


def derive_state_rerouting(pairs: pd.DataFrame, targets: set[str], states: list[str]) -> pd.DataFrame:
    pairs = pairs[pairs["target_contrast"].isin(targets)].copy()
    expected_pairs = {tuple(sorted(pair)) for index, left in enumerate(states) for pair in [(left, right) for right in states[index + 1 :]]}
    observed_pairs = {tuple(sorted(pair)) for pair in pairs[["left_state", "right_state"]].drop_duplicates().itertuples(index=False, name=None)}
    if expected_pairs != observed_pairs:
        raise ValueError("Pairwise state coverage disagrees with the frozen state set")
    left = pairs[["target_contrast", "left_state", "right_state", "pair_evaluable", "module_js_divergence"]].rename(
        columns={"left_state": "culture_condition", "right_state": "other_state"}
    )
    right = pairs[["target_contrast", "right_state", "left_state", "pair_evaluable", "module_js_divergence"]].rename(
        columns={"right_state": "culture_condition", "left_state": "other_state"}
    )
    long = pd.concat([left, right], ignore_index=True)
    long["finite_pair"] = long["pair_evaluable"] & np.isfinite(long["module_js_divergence"])
    rows = []
    for (target, state), block in long.groupby(["target_contrast", "culture_condition"], sort=False):
        finite = block.loc[block["finite_pair"], "module_js_divergence"]
        value = float(finite.mean()) if len(block) == 2 and len(finite) == 2 else np.nan
        rows.append(
            {
                "target_contrast": target,
                "culture_condition": state,
                "rerouting_pair_rows": int(len(block)),
                "rerouting_evaluable_pairs": int(len(finite)),
                "state_rerouting_js": value,
            }
        )
    result = pd.DataFrame(rows)
    expected = len(targets) * len(states)
    if len(result) != expected:
        raise ValueError(f"Expected {expected} target-state rerouting rows, observed {len(result)}")
    return result


def add_plot_coordinates(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    axes = {
        "x": ("cis_effect", "Absolute cis perturbation strength"),
        "y": ("transfer_gain", "Cross-fitted distal transfer gain"),
        "z": ("rerouting_value", "Mean pairwise module JS divergence"),
    }
    transforms = []
    source = source.copy()
    for axis, (field, label) in axes.items():
        finite = source[field][np.isfinite(source[field])]
        lower = float(finite.min())
        upper = float(finite.max())
        if upper <= lower:
            raise ValueError(f"Non-positive range for {axis}")
        source[f"plot_{axis}"] = scale_value(source[field], lower, upper)
        transforms.append(
            {
                "axis": axis,
                "source_field": field,
                "axis_label": label,
                "original_min": lower,
                "original_max": upper,
                "plot_min": 0.0,
                "plot_max": 1.0,
                "formula": "(value - original_min) / (original_max - original_min)",
                "clipped": False,
            }
        )
    return source, pd.DataFrame(transforms)


def threshold_table(transfer: pd.DataFrame, states: list[str], config: dict) -> pd.DataFrame:
    rows = []
    values = transfer["transfer_z"]
    for name, quantile in [
        ("buffered", config["thresholds"]["buffered_quantile"]),
        ("amplified", config["thresholds"]["amplified_quantile"]),
    ]:
        rows.append(
            {
                "state": "All",
                "threshold_class": name,
                "quantile": quantile,
                "transfer_gain_threshold": float(values.quantile(quantile)),
                "source_rows": int(values.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def select_trajectories(source: pd.DataFrame, states: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = []
    for target, block in source.groupby("target_id", sort=True):
        block = block.set_index("state").reindex(states)
        guide_correlations = block["guide_correlation_all"].dropna()
        finite = bool(np.isfinite(block[["cis_effect", "transfer_gain", "rerouting_value"]].to_numpy()).all())
        eligible = bool(
            finite
            and block["ontarget_significant"].fillna(False).all()
            and (~block["distal_offtarget_flag"].fillna(True)).all()
            and (block["n_guides"].fillna(0) >= 2).all()
            and (~block["single_guide_estimate"].fillna(True)).all()
        )
        target_rows.append(
            {
                "target_id": target,
                "target": block["target"].dropna().iloc[0],
                "eligible": eligible,
                "amplified": bool((block["transfer_class"] == "amplified").any()),
                "buffered": bool((block["transfer_class"] == "buffered").any()),
                "constant_transfer_class": bool(block["transfer_class"].nunique(dropna=True) == 1),
                "mean_rerouting": float(block["rerouting_value"].mean()) if finite else np.nan,
                "guide_supported_states": int(block["guide_support"].fillna(False).sum()),
                "median_guide_correlation": float(guide_correlations.median()) if not guide_correlations.empty else -np.inf,
                "minimum_target_cells": float(block["n_cells_target"].min()),
            }
        )
    candidates = pd.DataFrame(target_rows)
    finite_target = candidates.loc[candidates["eligible"] & np.isfinite(candidates["mean_rerouting"]), "mean_rerouting"]
    low = float(finite_target.quantile(0.10))
    high = float(finite_target.quantile(0.90))
    candidates["stable_core"] = candidates["constant_transfer_class"] & (candidates["mean_rerouting"] <= low)
    candidates["strongly_rerouted"] = candidates["mean_rerouting"] >= high
    used: set[str] = set()
    selected = []
    reasons = {
        "amplified": "at least one amplified state",
        "buffered": "at least one buffered state",
        "stable_core": "bottom rerouting decile with constant transfer class",
        "strongly_rerouted": "top rerouting decile",
    }
    for class_name in ["amplified", "buffered", "stable_core", "strongly_rerouted"]:
        block = candidates[candidates["eligible"] & candidates[class_name] & ~candidates["target_id"].isin(used)].copy()
        block = block.sort_values(
            ["guide_supported_states", "median_guide_correlation", "minimum_target_cells", "target_id"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        if block.empty:
            raise ValueError(f"No eligible trajectory for class {class_name}")
        row = block.iloc[0].to_dict()
        row["selected_class"] = class_name
        row["trajectory_number"] = len(selected) + 1
        row["selection_reason"] = (
            f"{reasons[class_name]}; fixed support rank: {int(row['guide_supported_states'])} guide-supported states, "
            f"median guide correlation {row['median_guide_correlation']:.3f}, minimum {int(row['minimum_target_cells'])} target cells"
        )
        selected.append(row)
        used.add(row["target_id"])
    labels = pd.DataFrame(selected)
    source = source.copy()
    source["trajectory_selection_status"] = "not_selected"
    source["trajectory_order"] = pd.Series(pd.NA, index=source.index, dtype="Int64")
    source["trajectory_number"] = pd.Series(pd.NA, index=source.index, dtype="Int64")
    source["selection_reason"] = ""
    order_map = {state: index for index, state in enumerate(states)}
    for row in labels.itertuples(index=False):
        mask = source["target_id"] == row.target_id
        source.loc[mask, "trajectory_selection_status"] = row.selected_class
        source.loc[mask, "trajectory_order"] = source.loc[mask, "state"].map(order_map).astype("Int64")
        source.loc[mask, "trajectory_number"] = int(row.trajectory_number)
        source.loc[mask, "selection_reason"] = row.selection_reason
    return source, labels


def bootstrap_centroids(source: pd.DataFrame, states: list[str], replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summary_rows = []
    replicate_rows = []
    ellipsoid_rows = []
    critical = float(chi2.ppf(0.95, df=3))
    for state in states:
        block = source[(source["state"] == state) & source[["plot_x", "plot_y", "plot_z"]].notna().all(axis=1)].copy()
        values = block[["cis_effect", "transfer_gain", "rerouting_value"]].to_numpy(dtype=float)
        plotted = block[["plot_x", "plot_y", "plot_z"]].to_numpy(dtype=float)
        samples = rng.integers(0, len(block), size=(replicates, len(block)))
        boot_original = values[samples].mean(axis=1)
        boot_plot = plotted[samples].mean(axis=1)
        center_original = values.mean(axis=0)
        center_plot = plotted.mean(axis=0)
        for replicate in range(replicates):
            replicate_rows.append(
                {
                    "state": state,
                    "replicate": replicate + 1,
                    "cis_centroid": boot_original[replicate, 0],
                    "transfer_centroid": boot_original[replicate, 1],
                    "rerouting_centroid": boot_original[replicate, 2],
                    "plot_x": boot_plot[replicate, 0],
                    "plot_y": boot_plot[replicate, 1],
                    "plot_z": boot_plot[replicate, 2],
                }
            )
        summary_rows.append(
            {
                "state": state,
                "finite_xyz_targets": int(len(block)),
                "missing_rerouting_targets": int((source["state"].eq(state) & source["rerouting_value"].isna()).sum()),
                "cis_centroid": center_original[0],
                "cis_ci_low": np.quantile(boot_original[:, 0], 0.025),
                "cis_ci_high": np.quantile(boot_original[:, 0], 0.975),
                "transfer_centroid": center_original[1],
                "transfer_ci_low": np.quantile(boot_original[:, 1], 0.025),
                "transfer_ci_high": np.quantile(boot_original[:, 1], 0.975),
                "rerouting_centroid": center_original[2],
                "rerouting_ci_low": np.quantile(boot_original[:, 2], 0.025),
                "rerouting_ci_high": np.quantile(boot_original[:, 2], 0.975),
                "plot_x": center_plot[0],
                "plot_y": center_plot[1],
                "plot_z": center_plot[2],
                "bootstrap_replicates": replicates,
            }
        )
        covariance = np.cov(boot_plot, rowvar=False, ddof=1)
        eigenvalues, eigenvectors = eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        eigenvectors = eigenvectors[:, order]
        radii = np.sqrt(critical * eigenvalues)
        for axis in range(3):
            ellipsoid_rows.append(
                {
                    "state": state,
                    "ellipsoid_level": 0.95,
                    "axis_index": axis + 1,
                    "radius_scaled": radii[axis],
                    "eigenvalue": eigenvalues[axis],
                    "eigenvector_x": eigenvectors[0, axis],
                    "eigenvector_y": eigenvectors[1, axis],
                    "eigenvector_z": eigenvectors[2, axis],
                    "center_plot_x": center_plot[0],
                    "center_plot_y": center_plot[1],
                    "center_plot_z": center_plot[2],
                    "cov_xx": covariance[0, 0],
                    "cov_xy": covariance[0, 1],
                    "cov_xz": covariance[0, 2],
                    "cov_yy": covariance[1, 1],
                    "cov_yz": covariance[1, 2],
                    "cov_zz": covariance[2, 2],
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(replicate_rows), pd.DataFrame(ellipsoid_rows)


def correlations(source: pd.DataFrame, states: list[str], replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for state in states:
        block = source[source["state"] == state].dropna(subset=["transfer_gain", "rerouting_value"])
        values = block[["transfer_gain", "rerouting_value"]].to_numpy(dtype=float)
        estimate, p_value = spearmanr(values[:, 0], values[:, 1])
        samples = rng.integers(0, len(values), size=(replicates, len(values)))
        boot = np.array([spearmanr(values[index, 0], values[index, 1]).statistic for index in samples])
        rows.append(
            {
                "state": state,
                "targets": int(len(values)),
                "spearman_rho": float(estimate),
                "ci_low": float(np.nanquantile(boot, 0.025)),
                "ci_high": float(np.nanquantile(boot, 0.975)),
                "p_value": float(p_value),
                "bootstrap_replicates": replicates,
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = bh_adjust(result["p_value"])
    result["decision"] = np.where(
        (result["q_value"] <= 0.05) & (result["ci_low"] > 0),
        "supported_positive",
        np.where((result["q_value"] <= 0.05) & (result["ci_high"] < 0), "supported_negative", "unresolved"),
    )
    return result


def consolidate_strata(frame: pd.DataFrame, minimum: int) -> pd.Series:
    rank = frame["baseline_cis"].rank(method="first")
    cis_bin = pd.qcut(rank, 5, labels=False).astype(int)
    strata = cis_bin.astype(str) + "|" + frame["baseline_guide_support"].astype(int).astype(str)
    counts = strata.value_counts()
    small = strata.map(counts) < minimum
    strata.loc[small] = cis_bin.loc[small].astype(str) + "|all"
    counts = strata.value_counts()
    strata.loc[strata.map(counts) < minimum] = "all"
    return strata


def displacement_analysis(source: pd.DataFrame, transitions: list[list[str]], replicates: int, seed: int, minimum: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    axis_fields = {"cis effect": "cis_effect", "transfer gain": "transfer_gain", "rerouting": "rerouting_value"}
    pooled_sd = {name: float(source[field].std(ddof=1)) for name, field in axis_fields.items()}
    estimate_rows = []
    null_rows = []
    for state_a, state_b in transitions:
        baseline = source[source["state"] == state_a][["target_id", "cis_effect", "guide_support"]].rename(
            columns={"cis_effect": "baseline_cis", "guide_support": "baseline_guide_support"}
        )
        for axis, field in axis_fields.items():
            wide = source[source["state"].isin([state_a, state_b])].pivot(index="target_id", columns="state", values=field).reset_index()
            frame = baseline.merge(wide, on="target_id", how="inner").dropna(subset=[state_a, state_b, "baseline_cis"])
            frame["stratum"] = consolidate_strata(frame, minimum)
            observed_changes = (frame[state_b].to_numpy(dtype=float) - frame[state_a].to_numpy(dtype=float)) / pooled_sd[axis]
            observed = float(np.median(observed_changes))
            samples = rng.integers(0, len(frame), size=(replicates, len(frame)))
            bootstrap = np.median(observed_changes[samples], axis=1)
            null_values = np.empty(replicates, dtype=float)
            destination = frame[state_b].to_numpy(dtype=float)
            origin = frame[state_a].to_numpy(dtype=float)
            groups = [np.asarray(index, dtype=int) for index in frame.groupby("stratum", sort=True).indices.values()]
            for replicate in range(replicates):
                permuted = destination.copy()
                for indices in groups:
                    permuted[indices] = rng.permutation(destination[indices])
                null_values[replicate] = np.median((permuted - origin) / pooled_sd[axis])
                null_rows.append(
                    {
                        "transition": f"{state_a}_to_{state_b}",
                        "axis": axis,
                        "replicate": replicate + 1,
                        "null_median_standardized_displacement": null_values[replicate],
                    }
                )
            lower_tail = (1 + np.sum(null_values <= observed)) / (replicates + 1)
            upper_tail = (1 + np.sum(null_values >= observed)) / (replicates + 1)
            p_value = min(1.0, 2 * min(lower_tail, upper_tail))
            estimate_rows.append(
                {
                    "transition": f"{state_a}_to_{state_b}",
                    "state_from": state_a,
                    "state_to": state_b,
                    "axis": axis,
                    "targets": int(len(frame)),
                    "standardization_sd": pooled_sd[axis],
                    "median_standardized_displacement": observed,
                    "ci_low": float(np.quantile(bootstrap, 0.025)),
                    "ci_high": float(np.quantile(bootstrap, 0.975)),
                    "null_median": float(np.median(null_values)),
                    "null_ci_low": float(np.quantile(null_values, 0.025)),
                    "null_ci_high": float(np.quantile(null_values, 0.975)),
                    "p_value": float(p_value),
                    "permutation_strata": int(frame["stratum"].nunique()),
                    "bootstrap_replicates": replicates,
                    "permutation_replicates": replicates,
                }
            )
    estimates = pd.DataFrame(estimate_rows)
    estimates["q_value"] = bh_adjust(estimates["p_value"])
    estimates["decision"] = np.where(
        (estimates["q_value"] <= 0.05) & ((estimates["ci_low"] > 0) | (estimates["ci_high"] < 0)),
        "supported_displacement",
        "unresolved",
    )
    return estimates, pd.DataFrame(null_rows)


def hdr_contours(x: np.ndarray, y: np.ndarray, masses: list[float], grid_points: int) -> list[tuple[float, np.ndarray]]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
    x_grid = np.linspace(x.min(), x.max(), grid_points)
    y_grid = np.linspace(y.min(), y.max(), grid_points)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    sorted_density = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(sorted_density)
    cumulative /= cumulative[-1]
    thresholds = [float(sorted_density[np.searchsorted(cumulative, mass)]) for mass in masses]
    order = np.argsort(thresholds)
    figure, axis = plt.subplots(figsize=(2, 2))
    contour = axis.contour(xx, yy, density, levels=np.asarray(thresholds)[order])
    segments = []
    for ordered_index, segment_group in enumerate(contour.allsegs):
        mass = masses[int(order[ordered_index])]
        for segment in segment_group:
            if len(segment) >= 3:
                segments.append((float(mass), np.asarray(segment, dtype=float)))
    plt.close(figure)
    return segments


def density_tables(source: pd.DataFrame, states: list[str], config: dict, transforms: pd.DataFrame) -> pd.DataFrame:
    limits = transforms.set_index("axis")
    planes = {
        "xy": ("cis_effect", "transfer_gain", "x", "y"),
        "yz": ("transfer_gain", "rerouting_value", "y", "z"),
        "xz": ("cis_effect", "rerouting_value", "x", "z"),
    }
    rows = []
    for state in states:
        block = source[source["state"] == state]
        for plane, (field_a, field_b, axis_a, axis_b) in planes.items():
            values = block[[field_a, field_b]].dropna().to_numpy(dtype=float)
            segments = hdr_contours(
                values[:, 0],
                values[:, 1],
                config["density"]["cumulative_mass_levels"],
                int(config["density"]["grid_points_per_axis"]),
            )
            for segment_id, (mass, segment) in enumerate(segments, start=1):
                plot_a = scale_value(segment[:, 0], limits.loc[axis_a, "original_min"], limits.loc[axis_a, "original_max"])
                plot_b = scale_value(segment[:, 1], limits.loc[axis_b, "original_min"], limits.loc[axis_b, "original_max"])
                for vertex, (original_a, original_b, scaled_a, scaled_b) in enumerate(
                    zip(segment[:, 0], segment[:, 1], plot_a, plot_b, strict=True), start=1
                ):
                    rows.append(
                        {
                            "state": state,
                            "plane": plane,
                            "cumulative_mass": mass,
                            "segment_id": segment_id,
                            "vertex_order": vertex,
                            "original_a": original_a,
                            "original_b": original_b,
                            "plot_a": scaled_a,
                            "plot_b": scaled_b,
                            "source_rows": int(len(values)),
                            "bandwidth_rule": config["density"]["bandwidth_rule"],
                        }
                    )
    return pd.DataFrame(rows)


def source_digest(source: pd.DataFrame) -> str:
    subset = source.sort_values(["target_id", "state"]).head(60).to_csv(index=False, float_format="%.12g")
    return hashlib.sha256(subset.encode("utf-8")).hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    states = config["states"]["order"]
    transfer = pd.read_parquet(TRANSFER_PATH)
    cohort = complete_cohort(transfer, states)
    expected_targets = int(config["cohort"]["expected_complete_targets"])
    if cohort["target_contrast"].nunique() != expected_targets or len(cohort) != expected_targets * len(states):
        raise ValueError("Complete cohort denominator changed")
    pairs = pd.read_parquet(PAIR_PATH)
    rerouting = derive_state_rerouting(pairs, set(cohort["target_contrast"]), states)
    source = cohort.merge(rerouting, on=["target_contrast", "culture_condition"], how="left", validate="one_to_one")
    source = source.rename(
        columns={
            "target_contrast": "target_id",
            "target_contrast_gene_name": "target",
            "culture_condition": "state",
            "cis_magnitude": "cis_effect",
            "transfer_z": "transfer_gain",
            "state_rerouting_js": "rerouting_value",
        }
    )
    source["response_support"] = source["ontarget_significant"] & (source["n_total_de_genes"] > 0) & ~source["distal_offtarget_flag"]
    source["guide_support"] = (
        (source["n_guides"].fillna(0) >= 2)
        & ~source["single_guide_estimate"].fillna(True)
        & (source["guide_correlation_signif_pval"].fillna(1) <= 0.05)
    )
    source["missingness_status"] = source["rerouting_evaluable_pairs"].map(
        {2: "rerouting_complete", 1: "rerouting_missing_one_pair", 0: "rerouting_missing_both_pairs"}
    )
    source, transforms = add_plot_coordinates(source)
    source, labels = select_trajectories(source, states)
    thresholds = threshold_table(transfer, states, config)
    y_transform = transforms.set_index("axis").loc["y"]
    thresholds["plot_y"] = scale_value(thresholds["transfer_gain_threshold"], y_transform.original_min, y_transform.original_max)
    centroids, centroid_replicates, ellipsoids = bootstrap_centroids(
        source, states, int(config["inference"]["bootstrap_replicates"]), int(config["seed"]) + 101
    )
    correlation = correlations(source, states, int(config["inference"]["bootstrap_replicates"]), int(config["seed"]) + 202)
    displacement, permutation = displacement_analysis(
        source,
        config["displacement"]["transitions"],
        int(config["inference"]["permutation_replicates"]),
        int(config["seed"]) + 303,
        int(config["displacement"]["minimum_permutation_stratum"]),
    )
    density = density_tables(source, states, config, transforms)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    source_columns = [
        "target_id",
        "target",
        "state",
        "cis_effect",
        "transfer_gain",
        "rerouting_value",
        "transfer_class",
        "response_support",
        "guide_support",
        "missingness_status",
        "trajectory_selection_status",
        "trajectory_number",
        "trajectory_order",
        "plot_x",
        "plot_y",
        "plot_z",
        "selection_reason",
        "rerouting_pair_rows",
        "rerouting_evaluable_pairs",
        "n_guides",
        "guide_correlation_all",
        "n_cells_target",
        "ontarget_significant",
        "distal_offtarget_flag",
        "single_guide_estimate",
    ]
    source[source_columns].to_csv(OUTPUT / "landscape_observations.tsv", sep="\t", index=False)
    transforms.to_csv(OUTPUT / "axis_transform.tsv", sep="\t", index=False)
    thresholds.to_csv(OUTPUT / "transfer_thresholds.tsv", sep="\t", index=False)
    centroids.to_csv(OUTPUT / "bootstrap_centroids.tsv", sep="\t", index=False)
    centroid_replicates.to_parquet(OUTPUT / "bootstrap_centroid_replicates.parquet", index=False)
    ellipsoids.to_csv(OUTPUT / "uncertainty_ellipsoids.tsv", sep="\t", index=False)
    correlation.to_csv(OUTPUT / "pairwise_correlations.tsv", sep="\t", index=False)
    displacement.to_csv(OUTPUT / "displacement_estimates.tsv", sep="\t", index=False)
    permutation.to_parquet(OUTPUT / "permutation_nulls.parquet", index=False)
    labels.to_csv(OUTPUT / "displayed_target_labels.tsv", sep="\t", index=False)
    density.to_parquet(OUTPUT / "density_contours.parquet", index=False)
    selected_rows = source[source["trajectory_selection_status"] != "not_selected"]
    audit = {
        "complete_targets": int(source["target_id"].nunique()),
        "source_rows": int(len(source)),
        "state_rows": source.groupby("state").size().astype(int).to_dict(),
        "finite_xyz_rows": int(source[["plot_x", "plot_y", "plot_z"]].notna().all(axis=1).sum()),
        "missing_rerouting_rows": int(source["rerouting_value"].isna().sum()),
        "missingness_status": source["missingness_status"].value_counts(dropna=False).astype(int).to_dict(),
        "selected_targets": labels[["trajectory_number", "target_id", "target", "selected_class"]].to_dict("records"),
        "trajectory_rows": int(len(selected_rows)),
        "trajectory_states_per_target": selected_rows.groupby("target_id")["state"].nunique().astype(int).to_dict(),
        "correlation_decisions": correlation.set_index("state")["decision"].to_dict(),
        "displacement_decisions": displacement["decision"].value_counts().astype(int).to_dict(),
        "density_vertices": int(len(density)),
        "source_subset_sha256": source_digest(source[source_columns]),
        "bootstrap_replicates": int(config["inference"]["bootstrap_replicates"]),
        "permutation_replicates": int(config["inference"]["permutation_replicates"]),
        "coordinate_imputation": False,
    }
    (AUDIT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
