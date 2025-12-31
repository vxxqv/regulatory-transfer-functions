"""Evaluate the frozen transportability hypotheses and their availability gates."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses/transportability/results"
SEED = 20260913
STATES = ["Rest", "Stim8hr", "Stim48hr"]
SOURCE_PATHS = [
    "config/cell_systems_expansion.yaml",
    "analyses/cell_systems_expansion/freeze_manifest.json",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/guide_concordance/results/all_target_states.parquet",
    "analyses/molecular_cascade/results/edge_evidence_matrix.parquet",
    "analyses/state_response_decomposition/results/target_decomposition.parquet",
    "analyses/replication/results/k562_replication_rows.parquet",
    "analyses/external_benchmark/results/all_eligible_target_predictions.parquet",
    "analyses/external_benchmark/rpe1_prepared/eligible_targets.parquet",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_hash_candidates(path: Path) -> dict[str, str]:
    data = path.read_bytes()
    candidates = {"raw": hashlib.sha256(data).hexdigest()}
    if path.suffix.lower() in {".json", ".md", ".tsv", ".txt", ".yaml", ".yml"}:
        lf = data.replace(b"\r\n", b"\n")
        candidates["lf"] = hashlib.sha256(lf).hexdigest()
        candidates["crlf"] = hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
    return candidates


def bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def hash_fold(value: str, folds: int, suffix: str) -> int:
    text = f"{value}|{suffix}".encode("utf-8")
    return 1 + int(hashlib.sha256(text).hexdigest()[:16], 16) % folds


def quantile_bin(series: pd.Series, bins: int = 5) -> pd.Series:
    valid = series.notna()
    output = pd.Series(-1, index=series.index, dtype=int)
    if valid.any():
        output.loc[valid] = pd.qcut(series.loc[valid].rank(method="first"), bins, labels=False).astype(int)
    return output


def residualized_fit(data: pd.DataFrame, predictor: str, outcome: str, covariates: list[str]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    frame = data.copy()
    numeric = frame[covariates].astype(float)
    for column in covariates:
        missing = numeric[column].isna()
        frame[f"{column}_missing"] = missing.astype(float)
        median = numeric[column].median()
        numeric[column] = numeric[column].fillna(0.0 if pd.isna(median) else median)
    scale = numeric.std(ddof=0).replace(0, 1)
    numeric = (numeric - numeric.mean()) / scale
    state = pd.get_dummies(frame["culture_condition"], dtype=float, drop_first=True)
    missing_columns = [f"{column}_missing" for column in covariates if frame[f"{column}_missing"].any()]
    nuisance = np.column_stack([np.ones(len(frame)), numeric.to_numpy(), state.to_numpy(), frame[missing_columns].to_numpy() if missing_columns else np.empty((len(frame), 0))])
    x = frame[predictor].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    residual_x = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
    residual_y = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    estimate = float(residual_x @ residual_y / (residual_x @ residual_x))
    return estimate, residual_x, residual_y, nuisance


def clustered_diagnostic(
    data: pd.DataFrame,
    predictor: str,
    outcome: str,
    covariates: list[str],
    name: str,
    rng: np.random.Generator,
    bootstraps: int,
    permutations: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    required = ["target_gene", "culture_condition", predictor, outcome]
    frame = data.dropna(subset=required).copy().reset_index(drop=True)
    estimate, residual_x, residual_y, _ = residualized_fit(frame, predictor, outcome, covariates)
    targets, codes = np.unique(frame["target_gene"], return_inverse=True)
    boot = np.empty(bootstraps)
    for index in range(bootstraps):
        sampled = rng.integers(0, len(targets), len(targets))
        weights = np.bincount(sampled, minlength=len(targets))[codes]
        denom = np.sum(weights * residual_x * residual_x)
        boot[index] = np.sum(weights * residual_x * residual_y) / denom if denom > 0 else np.nan

    target_meta = frame.groupby("target_gene").agg(
        expression=("source_target_expression", "mean"),
        degree=("source_response_degree", "mean"),
        network=("curated_network_degree", "mean"),
        pattern=("culture_condition", lambda values: "|".join(sorted(values))),
    )
    target_meta["expression_bin"] = quantile_bin(target_meta["expression"])
    target_meta["degree_bin"] = quantile_bin(target_meta["degree"])
    target_meta["network_bin"] = quantile_bin(target_meta["network"])
    target_rows = {target: block.sort_values("culture_condition").index.to_numpy() for target, block in frame.groupby("target_gene")}
    strata = []
    swappable = set()
    for _, block in target_meta.groupby(["pattern", "expression_bin", "degree_bin", "network_bin"]):
        members = block.index.to_numpy()
        if len(members) > 1:
            strata.append(members)
            swappable.update(members)
    null = np.empty(permutations)
    for replicate in range(permutations):
        permuted = residual_x.copy()
        for members in strata:
            for destination, origin in zip(members, rng.permutation(members)):
                permuted[target_rows[destination]] = residual_x[target_rows[origin]]
        denom = permuted @ permuted
        null[replicate] = permuted @ residual_y / denom if denom > 0 else np.nan
    p_value = float((1 + np.sum(np.abs(null) >= abs(estimate))) / (permutations + 1))
    result = {
        "analysis": name,
        "predictor": predictor,
        "outcome": outcome,
        "rows": len(frame),
        "targets": len(targets),
        "swappable_targets": len(swappable),
        "estimate": estimate,
        "ci_low": float(np.nanquantile(boot, 0.025)),
        "ci_high": float(np.nanquantile(boot, 0.975)),
        "permutation_p": p_value,
        "bootstrap_replicates": bootstraps,
        "permutation_replicates": permutations,
        "interpretation": "developmental observed-concordance diagnostic",
    }
    null_rows = pd.DataFrame({"analysis": name, "replicate": np.arange(permutations), "null_coefficient": null})
    boot_rows = pd.DataFrame({"analysis": name, "replicate": np.arange(bootstraps), "coefficient": boot})
    return result, null_rows, boot_rows


def molecular_summary(edges: pd.DataFrame) -> pd.DataFrame:
    physical = edges[["physical_active_link", "promoter_bound", "enhancer_linked"]].any(axis=1)
    evidence_types = edges[["curated_edge", "motif_supported"]].astype(int).sum(axis=1) + physical.astype(int)
    table = edges.assign(
        molecular_two_types=evidence_types.ge(2),
        molecular_any=evidence_types.ge(1),
    )
    return (
        table.groupby(["target_gene", "culture_condition"], as_index=False)
        .agg(
            molecular_support_fraction=("molecular_two_types", "mean"),
            molecular_any_fraction=("molecular_any", "mean"),
            curated_network_degree=("curated_edge", "sum"),
            molecular_response_edges=("response_gene", "size"),
        )
    )


def build_k562(primary: pd.DataFrame, guides: pd.DataFrame, decomposition: pd.DataFrame, molecular: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    k562 = pd.read_parquet(ROOT / "analyses/replication/results/k562_replication_rows.parquet").rename(
        columns={"target_contrast_gene_name": "target_gene", "culture_condition": "prior_primary_condition", "condition": "culture_condition"}
    )
    source = primary.rename(columns={"target_contrast_gene_name": "target_gene"})[
        [
            "target_gene",
            "target_contrast",
            "culture_condition",
            "log1p_target_baseMean",
            "cis_magnitude",
            "n_downstream",
            "guide_correlation_all",
            "donor_correlation_all_mean",
            "transfer_class",
            "ontarget_significant",
            "distal_offtarget_flag",
            "n_guides",
            "single_guide_estimate",
        ]
    ].rename(
        columns={
            "log1p_target_baseMean": "source_target_expression",
            "n_downstream": "source_response_degree",
            "guide_correlation_all": "primary_guide_concordance",
            "donor_correlation_all_mean": "source_donor_concordance",
        }
    )
    table = k562.merge(source, on=["target_gene", "culture_condition"], how="left", suffixes=("", "_source"), validate="many_to_one")
    guide_fields = guides[
        ["target", "culture_condition", "correlation_all", "quality_complete", "eligible_pair", "target_inference_status", "min_guide_cells"]
    ].rename(
        columns={
            "target": "target_gene",
            "correlation_all": "source_guide_concordance",
            "quality_complete": "source_guide_quality_complete",
            "eligible_pair": "source_guide_pair_eligible",
            "target_inference_status": "source_guide_inference_status",
            "min_guide_cells": "source_min_guide_cells",
        }
    )
    table = table.merge(guide_fields, on=["target_gene", "culture_condition"], how="left", validate="many_to_one")
    table = table.merge(molecular, on=["target_gene", "culture_condition"], how="left", validate="many_to_one")
    target = decomposition[["target_gene", "target_contrast", "rerouting", "core_fraction", "total_energy"]].copy()
    target = target.groupby("target_gene", as_index=False).agg(
        target_contrast=("target_contrast", "first"),
        mean_rerouting=("rerouting", "mean"),
        core_fraction=("core_fraction", "mean"),
        total_energy=("total_energy", "mean"),
    )
    table = table.merge(target, on="target_gene", how="left", suffixes=("", "_decomposition"), validate="many_to_one")

    complete = source.groupby("target_contrast").filter(lambda block: set(block.culture_condition) == set(STATES) and len(block) == 3)
    quality = complete.groupby("target_contrast").agg(
        states=("culture_condition", "nunique"),
        ontarget=("ontarget_significant", "all"),
        no_offtarget=("distal_offtarget_flag", lambda values: (~values.fillna(True)).all()),
        guides=("n_guides", "min"),
        no_single=("single_guide_estimate", lambda values: (~values.fillna(True)).all()),
        constant_transfer_class=("transfer_class", "nunique"),
    ).reset_index()
    target_by_id = decomposition[["target_contrast", "target_gene", "rerouting"]].merge(quality, on="target_contrast", how="left")
    target_by_id["s24_eligible"] = (
        target_by_id["rerouting"].notna()
        & target_by_id["states"].eq(3)
        & target_by_id["ontarget"].eq(True)
        & target_by_id["no_offtarget"].eq(True)
        & target_by_id["guides"].ge(2)
        & target_by_id["no_single"].eq(True)
    )
    eligible_rerouting = target_by_id.loc[target_by_id.s24_eligible, "rerouting"]
    low = float(eligible_rerouting.quantile(0.10))
    high = float(eligible_rerouting.quantile(0.90))
    target_by_id["rerouting_group"] = "intermediate"
    target_by_id.loc[target_by_id.s24_eligible & target_by_id.rerouting.ge(high), "rerouting_group"] = "strongly_rerouted"
    target_by_id.loc[target_by_id.s24_eligible & target_by_id.rerouting.le(low) & target_by_id.constant_transfer_class.eq(1), "rerouting_group"] = "invariant"
    labels = target_by_id.groupby("target_gene", as_index=False).agg(rerouting_group=("rerouting_group", "first"), s24_eligible=("s24_eligible", "max"))
    table = table.merge(labels, on="target_gene", how="left", validate="many_to_one")
    table["k562_row_status"] = np.where(table["source_target_expression"].notna(), "source_features_available", "source_target_state_unavailable")
    table["external_reliability_status"] = "unavailable_no_target_level_replicate_summary"
    table["reliability_corrected_status"] = "unavailable_external_reliability"
    return table, {"rerouting_low": low, "rerouting_high": high, "s24_eligible_targets": int(target_by_id.s24_eligible.sum())}


def reliability_bounds(k562: pd.DataFrame, rng: np.random.Generator, replicates: int, minimum_reliability: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    draws = []
    groups = [("All", k562)] + [(state, block) for state, block in k562.groupby("culture_condition", sort=True)]
    for label, block in groups:
        by_target = {target: values.logfc_pearson_r.to_numpy(float) for target, values in block.groupby("target_gene")}
        targets = np.array(sorted(by_target))
        observed = float(block.logfc_pearson_r.median())
        boot = np.empty(replicates)
        lower = np.empty(replicates)
        upper = np.empty(replicates)
        for index in range(replicates):
            chosen = rng.choice(targets, len(targets), replace=True)
            value = float(np.median(np.concatenate([by_target[target] for target in chosen])))
            boot[index] = value
            if value >= 0:
                lower[index], upper[index] = value, min(1.0, value / minimum_reliability)
            else:
                lower[index], upper[index] = max(-1.0, value / minimum_reliability), value
            draws.append({"condition": label, "replicate": index, "observed_median": value, "corrected_lower": lower[index], "corrected_upper": upper[index]})
        corrected = (observed, min(1.0, observed / minimum_reliability)) if observed >= 0 else (max(-1.0, observed / minimum_reliability), observed)
        records.append(
            {
                "system": "K562",
                "condition": label,
                "rows": len(block),
                "targets": len(targets),
                "observed_median": observed,
                "observed_ci_low": np.quantile(boot, 0.025),
                "observed_ci_high": np.quantile(boot, 0.975),
                "corrected_identified_lower": corrected[0],
                "corrected_identified_upper": corrected[1],
                "corrected_lower_ci_low": np.quantile(lower, 0.025),
                "corrected_lower_ci_high": np.quantile(lower, 0.975),
                "corrected_upper_ci_low": np.quantile(upper, 0.025),
                "corrected_upper_ci_high": np.quantile(upper, 0.975),
                "fraction_of_ceiling_lower": minimum_reliability,
                "fraction_of_ceiling_upper": 1.0,
                "noise_attributable_fraction_lower": 0.0,
                "noise_attributable_fraction_upper": 1.0,
                "status": "partially_identified_external_reliability_unavailable",
            }
        )
    return pd.DataFrame(records), pd.DataFrame(draws)


def applicability_pipeline(alpha: float, l1_ratio: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, tol=1e-6, random_state=seed)),
        ]
    )


def choose_elastic_net(x: pd.DataFrame, y: np.ndarray, targets: pd.Series, alpha_grid: list[float], ratio_grid: list[float], seed: int) -> tuple[float, float, float]:
    inner = np.array([hash_fold(str(target), 5, "transport-inner") for target in targets])
    candidates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for alpha in alpha_grid:
            for ratio in ratio_grid:
                predictions = np.full(len(y), np.nan)
                for fold in range(1, 6):
                    test = inner == fold
                    train = ~test
                    if test.sum() == 0 or train.sum() < 20:
                        continue
                    model = applicability_pipeline(alpha, ratio, seed + fold)
                    model.fit(x.loc[train], y[train])
                    predictions[test] = model.predict(x.loc[test])
                valid = np.isfinite(predictions)
                score = float(np.mean(np.abs(y[valid] - predictions[valid]))) if valid.any() else np.inf
                candidates.append((score, alpha, ratio))
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))


def build_rpe1(primary: pd.DataFrame, guides: pd.DataFrame, decomposition: pd.DataFrame, molecular: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    predictions = pd.read_parquet(ROOT / "analyses/external_benchmark/results/all_eligible_target_predictions.parquet")
    external = pd.read_parquet(ROOT / "analyses/external_benchmark/rpe1_prepared/eligible_targets.parquet")
    table = predictions.merge(external, on=["target_gene", "fold", "n_cells", "log1p_differentially_expressed_genes"], how="left", validate="one_to_one")
    source = (
        primary.rename(columns={"target_contrast_gene_name": "target_gene"})
        .groupby("target_gene", as_index=False)
        .agg(
            source_target_expression=("log1p_target_baseMean", "median"),
            source_donor_concordance=("donor_correlation_all_mean", "median"),
            source_response_degree=("n_downstream", "mean"),
        )
    )
    guide_source = guides.loc[guides.in_primary_study].copy()
    guide_source["source_guide_efficacy_state"] = guide_source[["knockdown_1", "knockdown_2"]].abs().median(axis=1, skipna=True)
    guide_summary = (
        guide_source
        .groupby("target", as_index=False)
        .agg(
            source_guide_concordance=("correlation_all", "median"),
            source_guide_efficacy=("source_guide_efficacy_state", "median"),
            source_guide_target_states=("culture_condition", "nunique"),
            source_guide_pair_eligible_fraction=("eligible_pair", "mean"),
        )
        .rename(columns={"target": "target_gene"})
    )
    modules = pd.read_parquet(ROOT / "analyses/vectors/results/contextual_transfer_tensor.parquet")
    module_availability = modules.groupby("target_gene", as_index=False).agg(module_availability=("score", lambda values: np.isfinite(values).mean()))
    mol_target = molecular.groupby("target_gene", as_index=False).agg(
        source_occupancy_proxy=("molecular_support_fraction", "mean"),
        source_curated_degree=("curated_network_degree", "mean"),
    )
    decomp = decomposition[["target_gene", "total_energy"]].copy()
    table = table.merge(source, on="target_gene", how="left", validate="one_to_one")
    table = table.merge(guide_summary, on="target_gene", how="left", validate="one_to_one")
    table = table.merge(module_availability, on="target_gene", how="left", validate="one_to_one")
    table = table.merge(decomp, on="target_gene", how="left", validate="one_to_one")
    table = table.merge(mol_target, on="target_gene", how="left", validate="one_to_one")
    table["baseline_target_expression"] = table["control_target_expression"]
    source_rank = table["source_target_expression"].rank(pct=True)
    external_rank = table["control_target_expression"].rank(pct=True)
    table["expression_range_compatibility"] = 1 - (source_rank - external_rank).abs()
    table["guide_efficacy"] = table["source_guide_efficacy"]
    table["source_response_reliability"] = table[["source_guide_concordance", "source_donor_concordance"]].clip(0, 1).mean(axis=1, skipna=True)
    table["target_degree"] = np.log1p(table["source_response_degree"])
    table["response_magnitude"] = np.sqrt(table["total_energy"].clip(lower=0))
    table["uncertainty_only"] = (table["frozen_cd4_transfer"] - table["without_vector"]).abs()
    table["reliability_only"] = table["source_response_reliability"]
    features = [
        "baseline_target_expression",
        "expression_range_compatibility",
        "guide_efficacy",
        "source_guide_concordance",
        "source_response_reliability",
        "module_availability",
        "target_degree",
    ]
    return table, features


def fit_applicability(
    table: pd.DataFrame,
    features: list[str],
    prediction_column: str,
    cfg: dict,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = table.copy()
    observed = frame["log1p_differentially_expressed_genes"].to_numpy(float)
    error = np.abs(observed - frame[prediction_column].to_numpy(float))
    frame["absolute_error"] = error
    frame["predicted_risk_raw"] = np.nan
    levels = [0.50, 0.80, 0.95]
    for level in levels:
        frame[f"risk_lower_{int(level * 100)}"] = np.nan
        frame[f"risk_upper_{int(level * 100)}"] = np.nan
    frame["bad_error_threshold"] = np.nan
    fold_records = []
    alpha_grid = [float(value) for value in cfg["applicability_model"]["alpha_grid"]]
    ratio_grid = [float(value) for value in cfg["applicability_model"]["l1_ratio_grid"]]
    for outer in sorted(frame.fold.astype(int).unique()):
        test = frame.fold.astype(int).eq(outer).to_numpy()
        outer_train = ~test
        calibration = outer_train & np.array([hash_fold(str(target), 5, "transport-calibration") == 1 for target in frame.target_gene])
        proper = outer_train & ~calibration
        score, alpha, ratio = choose_elastic_net(frame.loc[proper, features], error[proper], frame.loc[proper, "target_gene"], alpha_grid, ratio_grid, seed + outer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = applicability_pipeline(alpha, ratio, seed + outer)
            model.fit(frame.loc[proper, features], error[proper])
        prediction = model.predict(frame.loc[test, features])
        calibration_prediction = model.predict(frame.loc[calibration, features])
        frame.loc[test, "predicted_risk_raw"] = prediction
        residual = np.abs(error[calibration] - calibration_prediction)
        for level in levels:
            rank = min(len(residual) - 1, int(np.ceil((len(residual) + 1) * level)) - 1)
            radius = float(np.sort(residual)[rank])
            frame.loc[test, f"risk_lower_{int(level * 100)}"] = np.maximum(0, prediction - radius)
            frame.loc[test, f"risk_upper_{int(level * 100)}"] = prediction + radius
        frame.loc[test, "bad_error_threshold"] = float(np.quantile(error[outer_train], 1 - cfg["conformal_alpha"]))
        fold_records.append(
            {
                "prediction_model": prediction_column,
                "outer_fold": outer,
                "proper_training_targets": int(proper.sum()),
                "calibration_targets": int(calibration.sum()),
                "test_targets": int(test.sum()),
                "alpha": alpha,
                "l1_ratio": ratio,
                "inner_mae": score,
            }
        )
    frame["predicted_risk"] = frame["predicted_risk_raw"].clip(lower=0)
    frame["prediction_model"] = prediction_column
    return frame, pd.DataFrame(fold_records)


def calibration_summary(frame: pd.DataFrame, rng: np.random.Generator, replicates: int, permutations: int) -> tuple[dict[str, object], pd.DataFrame]:
    x = frame.predicted_risk.to_numpy(float)
    y = frame.absolute_error.to_numpy(float)
    rho = float(spearmanr(x, y).statistic)
    design = np.column_stack([np.ones(len(frame)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    boot = np.empty(replicates)
    for index in range(replicates):
        chosen = rng.integers(0, len(frame), len(frame))
        boot[index] = spearmanr(x[chosen], y[chosen]).statistic
    null = np.empty(permutations)
    folds = frame.fold.astype(int).to_numpy()
    for index in range(permutations):
        permuted = y.copy()
        for fold in np.unique(folds):
            rows = np.flatnonzero(folds == fold)
            permuted[rows] = rng.permutation(permuted[rows])
        null[index] = spearmanr(x, permuted).statistic
    p_value = float((1 + np.sum(np.abs(null) >= abs(rho))) / (permutations + 1))
    record = {
        "prediction_model": frame.prediction_model.iloc[0],
        "targets": len(frame),
        "spearman_rho": rho,
        "ci_low": np.nanquantile(boot, 0.025),
        "ci_high": np.nanquantile(boot, 0.975),
        "permutation_p": p_value,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "mae_of_risk_prediction": float(np.mean(np.abs(y - x))),
        "coverage_50": float(np.mean((y >= frame.risk_lower_50) & (y <= frame.risk_upper_50))),
        "coverage_80": float(np.mean((y >= frame.risk_lower_80) & (y <= frame.risk_upper_80))),
        "coverage_95": float(np.mean((y >= frame.risk_lower_95) & (y <= frame.risk_upper_95))),
        "interpretation": "developmental target-held-out diagnostic",
    }
    nulls = pd.DataFrame({"prediction_model": frame.prediction_model.iloc[0], "replicate": np.arange(permutations), "null_rho": null})
    return record, nulls


def select_indices(frame: pd.DataFrame, selector: str, coverage: float) -> np.ndarray:
    n = max(1, int(np.ceil(len(frame) * coverage)))
    if selector == "predict_all":
        return np.arange(len(frame))
    rules = {
        "applicability": ("predicted_risk", True),
        "response_magnitude": ("response_magnitude", False),
        "degree": ("target_degree", False),
        "uncertainty_only": ("uncertainty_only", True),
        "reliability_only": ("reliability_only", False),
    }
    column, ascending = rules[selector]
    values = frame[column].to_numpy(float)
    fill = np.inf if ascending else -np.inf
    order = np.argsort(np.nan_to_num(values, nan=fill))
    if not ascending:
        order = order[::-1]
    return order[:n]


def risk_coverage(
    frames: list[pd.DataFrame],
    coverages: list[float],
    rng: np.random.Generator,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selectors = ["predict_all", "random_abstention", "response_magnitude", "degree", "uncertainty_only", "reliability_only", "applicability"]
    records = []
    random_rows = []
    bootstrap_rows = []
    for frame in frames:
        model = frame.prediction_model.iloc[0]
        error = frame.absolute_error.to_numpy(float)
        for coverage in coverages:
            n = max(1, int(np.ceil(len(frame) * coverage)))
            for selector in selectors:
                if selector == "random_abstention":
                    values = np.empty(replicates)
                    for replicate in range(replicates):
                        chosen = rng.choice(len(frame), n, replace=False)
                        values[replicate] = error[chosen].mean()
                        random_rows.append({"prediction_model": model, "coverage": coverage, "replicate": replicate, "risk": values[replicate]})
                    estimate = float(np.median(values))
                    lo, hi = np.quantile(values, [0.025, 0.975])
                    selected_n = n
                else:
                    selected = select_indices(frame, selector, coverage)
                    selected_error = error[selected]
                    estimate = float(selected_error.mean())
                    draws = np.array([rng.choice(selected_error, len(selected_error), replace=True).mean() for _ in range(replicates)])
                    lo, hi = np.quantile(draws, [0.025, 0.975])
                    selected_n = len(selected)
                    bootstrap_rows.extend(
                        {"prediction_model": model, "selector": selector, "coverage": coverage, "replicate": index, "risk": value}
                        for index, value in enumerate(draws)
                    )
                records.append(
                    {
                        "prediction_model": model,
                        "selector": selector,
                        "coverage": coverage,
                        "selected_targets": selected_n,
                        "risk": estimate,
                        "ci_low": lo,
                        "ci_high": hi,
                        "status": "developmental_only",
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(random_rows), pd.DataFrame(bootstrap_rows)


def primary_coverage_comparisons(
    frames: list[pd.DataFrame],
    risk: pd.DataFrame,
    random_null: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
    coverage: float,
) -> pd.DataFrame:
    comparators = ["random_abstention", "response_magnitude", "degree", "uncertainty_only", "reliability_only"]
    records = []
    for frame in frames:
        model = frame.prediction_model.iloc[0]
        error = frame.absolute_error.to_numpy(float)
        app = select_indices(frame, "applicability", coverage)
        app_mask = np.zeros(len(frame), bool)
        app_mask[app] = True
        app_risk = error[app].mean()
        for comparator in comparators:
            if comparator == "random_abstention":
                null = random_null.loc[(random_null.prediction_model == model) & np.isclose(random_null.coverage, coverage), "risk"].to_numpy(float)
                delta_draw = null - np.array([rng.choice(error[app], len(app), replace=True).mean() for _ in range(len(null))])
                comparator_risk = float(np.median(null))
            else:
                other = select_indices(frame, comparator, coverage)
                other_mask = np.zeros(len(frame), bool)
                other_mask[other] = True
                comparator_risk = error[other].mean()
                delta_draw = np.empty(replicates)
                for index in range(replicates):
                    sampled = rng.integers(0, len(frame), len(frame))
                    weights = np.bincount(sampled, minlength=len(frame))
                    app_den = weights[app_mask].sum()
                    other_den = weights[other_mask].sum()
                    delta_draw[index] = (weights[other_mask] @ error[other_mask] / other_den) - (weights[app_mask] @ error[app_mask] / app_den)
            estimate = float(comparator_risk - app_risk)
            lo, hi = np.nanquantile(delta_draw, [0.025, 0.975])
            p_value = float((1 + np.sum(delta_draw <= 0)) / (len(delta_draw) + 1))
            app_fcr = float(np.mean(error[app] > frame.bad_error_threshold.to_numpy(float)[app]))
            if comparator == "random_abstention":
                comparator_fcr = float(np.mean(error > frame.bad_error_threshold.to_numpy(float)))
            else:
                comparator_fcr = float(np.mean(error[other] > frame.bad_error_threshold.to_numpy(float)[other]))
            records.append(
                {
                    "prediction_model": model,
                    "coverage": coverage,
                    "comparator": comparator,
                    "applicability_risk": app_risk,
                    "comparator_risk": comparator_risk,
                    "risk_improvement": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p_value,
                    "applicability_false_confidence_rate": app_fcr,
                    "comparator_false_confidence_rate": comparator_fcr,
                    "false_confidence_rate_increased": app_fcr > comparator_fcr,
                    "status": "developmental_only",
                }
            )
    result = pd.DataFrame(records)
    result["q_value"] = np.nan
    for model, index in result.groupby("prediction_model").groups.items():
        result.loc[index, "q_value"] = bh(result.loc[index, "p_value"])
    return result


def feature_availability() -> pd.DataFrame:
    status = {
        "baseline_target_expression": ("available_RPE1", "RPE1 control expression is retained before distal outcome evaluation."),
        "expression_range_compatibility": ("available_RPE1", "Percentile-rank compatibility uses source and RPE1 control expression."),
        "guide_efficacy": ("available_source_summary", "Median absolute CD4 guide knockdown is retained; RPE1 perturbation outcomes are excluded."),
        "source_guide_concordance": ("available_source_summary", "Two-guide source summary is descriptive and below the reliability-unit gate."),
        "source_response_reliability": ("available_source_summary", "Guide and donor summaries are combined for ranking only; no correction uses this score."),
        "baseline_transcriptomic_distance": ("unavailable", "No compatible baseline transcriptome distance is retained for all systems."),
        "network_neighborhood_conservation": ("unavailable", "No RPE1 or K562 matched regulatory-network conservation score is retained."),
        "module_availability": ("available_source_summary", "Fraction of finite frozen CD4 module scores is retained."),
        "target_degree": ("available_source_summary", "Mean CD4 distal response degree is retained."),
        "chromatin_accessibility_compatibility": ("unavailable", "No matched RPE1 or K562 accessibility matrix is available."),
        "TF_occupancy_support": ("unavailable_compatibility", "Source occupancy cannot establish external-state compatibility."),
        "enhancer_link_compatibility": ("unavailable", "No external-state enhancer-link map is available."),
        "intervention_type": ("constant", "Both external datasets are CRISPRi."),
        "assay_type": ("constant", "Both retained external outcomes use transcriptomic assays."),
        "timepoint_compatibility": ("unavailable", "External timepoint mapping is not comparable to all CD4 states."),
        "cell_state_similarity": ("unavailable", "No frozen common-space cell-state similarity score is retained."),
    }
    return pd.DataFrame([{"feature": feature, "status": value[0], "reason": value[1]} for feature, value in status.items()])


def system_availability() -> pd.DataFrame:
    rows = [
        ("CD4", "development source", "guide-to-guide", 2, "underpowered", "Two targeting guides are below the frozen three-unit reliability gate."),
        ("CD4", "development source", "donor-pair", np.nan, "summary_only", "A mean donor concordance is retained, but target-level replicate estimates are not."),
        ("K562", "previously inspected external", "response-vector", 1, "unavailable", "One target-level cross-system correlation is retained without external replicate reliability."),
        ("K562", "previously inspected external", "module-score", 0, "unavailable", "No replicate module-score reliability is retained."),
        ("RPE1", "previously inspected external", "response-vector", 1, "unavailable", "Stored outcomes do not include split-half or replicate response reliability."),
        ("RPE1", "previously inspected external", "module-score", 0, "unavailable", "No target-level external module concordance is retained."),
        ("future external", "confirmatory holdout", "all", 0, "unavailable", "No metadata-eligible untouched external outcome dataset is available."),
    ]
    return pd.DataFrame(rows, columns=["system", "role", "estimator", "replicate_units", "status", "reason"])


def add_storage_costs(costs: list[dict[str, object]]) -> pd.DataFrame:
    groups = {
        "K562 partial identification and diagnostics": [
            "k562_portability_rows.parquet",
            "reliability_bounds.csv",
            "reliability_bootstrap.parquet",
            "k562_diagnostics.csv",
            "k562_matched_permutation_nulls.parquet",
            "k562_target_bootstrap.parquet",
        ],
        "RPE1 developmental risk coverage": [
            "rpe1_applicability_rows.parquet",
            "applicability_folds.csv",
            "applicability_calibration.csv",
            "applicability_permutation_nulls.parquet",
            "risk_coverage.csv",
            "random_abstention_null.parquet",
            "risk_coverage_bootstrap.parquet",
            "primary_coverage_comparisons.csv",
        ],
    }
    table = pd.DataFrame(costs)
    table["output_storage_mb"] = table["block"].map(
        {block: sum((OUT / name).stat().st_size for name in names) / 1024**2 for block, names in groups.items()}
    )
    return table


def write_source_manifest() -> Path:
    freeze = json.loads((ROOT / "analyses/cell_systems_expansion/freeze_manifest.json").read_text(encoding="utf-8"))
    frozen = {record["path"]: record for record in freeze["artifacts"]}
    verified = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        actual = sha256(path)
        candidates = frozen_hash_candidates(path)
        expected_record = frozen.get(relative)
        expected = expected_record["sha256"] if expected_record else actual
        matched_form = next((name for name, value in candidates.items() if value == expected), None)
        if matched_form is None:
            raise ValueError(f"Frozen source changed: {relative}")
        verified.append(
            {
                "path": relative,
                "checkout_bytes": path.stat().st_size,
                "checkout_sha256": actual,
                "frozen_bytes": expected_record["bytes"] if expected_record else path.stat().st_size,
                "frozen_sha256": expected,
                "matched_form": matched_form,
                "matched_expansion_freeze": matched_form is not None if expected_record is not None else None,
                "verification_status": "matched_expansion_freeze" if expected_record is not None else "frozen_at_transport_start",
            }
        )
    path = OUT.parent / "source_manifest.json"
    path.write_text(json.dumps({"hash_comparison": "raw_or_lf_or_crlf_for_text_binary_exact", "sources": verified}, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config/cell_systems_expansion.yaml").read_text(encoding="utf-8"))
    source_manifest = write_source_manifest()

    transport = cfg["transportability"]
    bootstraps = int(cfg["common_inference"]["bootstrap_replicates"])
    permutations = int(cfg["common_inference"]["permutation_replicates"])
    rng = np.random.default_rng(SEED)
    costs = []
    tracemalloc.start()
    start = time.perf_counter()
    primary = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    guides = pd.read_parquet(ROOT / "analyses/guide_concordance/results/all_target_states.parquet")
    decomposition = pd.read_parquet(ROOT / "analyses/state_response_decomposition/results/target_decomposition.parquet")
    edges = pd.read_parquet(ROOT / "analyses/molecular_cascade/results/edge_evidence_matrix.parquet")
    molecular = molecular_summary(edges)
    k562, rerouting_thresholds = build_k562(primary, guides, decomposition, molecular)
    k562.to_parquet(OUT / "k562_portability_rows.parquet", index=False)
    bounds, bound_draws = reliability_bounds(k562, rng, bootstraps, float(transport["minimum_reliability"]))
    bounds.to_csv(OUT / "reliability_bounds.csv", index=False)
    bound_draws.to_parquet(OUT / "reliability_bootstrap.parquet", index=False)

    diagnostic_rows = []
    diagnostic_nulls = []
    diagnostic_boots = []
    h8 = k562.loc[k562.rerouting_group.isin(["invariant", "strongly_rerouted"])].copy()
    h8["strongly_rerouted"] = h8.rerouting_group.eq("strongly_rerouted").astype(float)
    covariates = ["source_target_expression", "cis_magnitude", "source_response_degree", "source_guide_concordance", "curated_network_degree"]
    h8_result, h8_null, h8_boot = clustered_diagnostic(h8, "strongly_rerouted", "logfc_pearson_r", covariates, "H8_observed_K562", rng, bootstraps, permutations)
    diagnostic_rows.append(h8_result)
    diagnostic_nulls.append(h8_null)
    diagnostic_boots.append(h8_boot)
    h11 = k562.dropna(subset=["molecular_support_fraction"]).copy()
    h11_result, h11_null, h11_boot = clustered_diagnostic(h11, "molecular_support_fraction", "logfc_pearson_r", covariates, "H11_observed_K562", rng, bootstraps, permutations)
    diagnostic_rows.append(h11_result)
    diagnostic_nulls.append(h11_null)
    diagnostic_boots.append(h11_boot)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics["q_value"] = bh(diagnostics.permutation_p)
    diagnostics["diagnostic_decision"] = np.where(
        diagnostics.q_value.lt(0.05) & diagnostics.ci_low.gt(0),
        "supported_positive",
        np.where(diagnostics.q_value.lt(0.05) & diagnostics.ci_high.lt(0), "supported_negative", "unresolved"),
    )
    diagnostics.to_csv(OUT / "k562_diagnostics.csv", index=False)
    pd.concat(diagnostic_nulls, ignore_index=True).to_parquet(OUT / "k562_matched_permutation_nulls.parquet", index=False)
    pd.concat(diagnostic_boots, ignore_index=True).to_parquet(OUT / "k562_target_bootstrap.parquet", index=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    costs.append({"block": "K562 partial identification and diagnostics", "wall_time_seconds": time.perf_counter() - start, "peak_memory_mb": peak / 1024**2})

    tracemalloc.start()
    start = time.perf_counter()
    rpe1, features = build_rpe1(primary, guides, decomposition, molecular)
    frames = []
    fold_tables = []
    calibration_records = []
    applicability_nulls = []
    for index, model in enumerate(["frozen_cd4_transfer", "without_vector"]):
        frame, folds = fit_applicability(rpe1, features, model, transport, SEED + 100 * (index + 1))
        record, null = calibration_summary(frame, rng, bootstraps, permutations)
        frames.append(frame)
        fold_tables.append(folds)
        calibration_records.append(record)
        applicability_nulls.append(null)
    calibration = pd.DataFrame(calibration_records)
    calibration["q_value"] = bh(calibration.permutation_p)
    calibration["diagnostic_decision"] = np.where(calibration.q_value.lt(0.05) & calibration.ci_low.gt(0), "supported_positive", "unresolved")
    combined_frames = pd.concat(frames, ignore_index=True)
    combined_frames.to_parquet(OUT / "rpe1_applicability_rows.parquet", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(OUT / "applicability_folds.csv", index=False)
    calibration.to_csv(OUT / "applicability_calibration.csv", index=False)
    pd.concat(applicability_nulls, ignore_index=True).to_parquet(OUT / "applicability_permutation_nulls.parquet", index=False)
    risk, random_null, risk_boot = risk_coverage(frames, [float(value) for value in transport["risk_coverage_grid"]], rng, bootstraps)
    risk.to_csv(OUT / "risk_coverage.csv", index=False)
    random_null.to_parquet(OUT / "random_abstention_null.parquet", index=False)
    risk_boot.to_parquet(OUT / "risk_coverage_bootstrap.parquet", index=False)
    primary_comparisons = primary_coverage_comparisons(frames, risk, random_null, rng, bootstraps, float(transport["primary_coverage"]))
    primary_comparisons.to_csv(OUT / "primary_coverage_comparisons.csv", index=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    costs.append({"block": "RPE1 developmental risk coverage", "wall_time_seconds": time.perf_counter() - start, "peak_memory_mb": peak / 1024**2})

    availability = system_availability()
    availability.to_csv(OUT / "system_availability.csv", index=False)
    features_table = feature_availability()
    features_table.to_csv(OUT / "feature_availability.csv", index=False)
    pooled = bounds.loc[bounds.condition == "All"].iloc[0]
    h8_diag = diagnostics.loc[diagnostics.analysis == "H8_observed_K562"].iloc[0]
    h11_diag = diagnostics.loc[diagnostics.analysis == "H11_observed_K562"].iloc[0]
    h9_diag = calibration.loc[calibration.prediction_model == "frozen_cd4_transfer"].iloc[0]
    h10_diag = primary_comparisons.loc[(primary_comparisons.prediction_model == "frozen_cd4_transfer") & (primary_comparisons.comparator == "random_abstention")].iloc[0]
    hypotheses = pd.DataFrame(
        [
            {
                "hypothesis": "H7",
                "estimate": pooled.observed_median,
                "ci_low": pooled.observed_ci_low,
                "ci_high": pooled.observed_ci_high,
                "diagnostic_p": np.nan,
                "diagnostic_q": np.nan,
                "decision": "unavailable",
                "reason": "External target-level reliability is absent; the corrected median is only identified between the reported bounds.",
                "diagnostic": f"Observed K562 median; corrected identified set {pooled.corrected_identified_lower:.3f} to {pooled.corrected_identified_upper:.3f}.",
            },
            {
                "hypothesis": "H8",
                "estimate": h8_diag.estimate,
                "ci_low": h8_diag.ci_low,
                "ci_high": h8_diag.ci_high,
                "diagnostic_p": h8_diag.permutation_p,
                "diagnostic_q": h8_diag.q_value,
                "decision": "unavailable",
                "reason": "The rerouting comparison is observed-concordance only because K562 target reliability is absent.",
                "diagnostic": h8_diag.diagnostic_decision,
            },
            {
                "hypothesis": "H9",
                "estimate": h9_diag.spearman_rho,
                "ci_low": h9_diag.ci_low,
                "ci_high": h9_diag.ci_high,
                "diagnostic_p": h9_diag.permutation_p,
                "diagnostic_q": h9_diag.q_value,
                "decision": "unavailable",
                "reason": "Only two previously inspected external systems with incompatible endpoints are available; leave-one-system-out validation requires three.",
                "diagnostic": h9_diag.diagnostic_decision,
            },
            {
                "hypothesis": "H10",
                "estimate": h10_diag.risk_improvement,
                "ci_low": h10_diag.ci_low,
                "ci_high": h10_diag.ci_high,
                "diagnostic_p": h10_diag.p_value,
                "diagnostic_q": h10_diag.q_value,
                "decision": "unavailable",
                "reason": "RPE1 is a previously inspected developmental system and no untouched held-out system is available.",
                "diagnostic": "developmental risk-coverage comparison against random abstention",
            },
            {
                "hypothesis": "H11",
                "estimate": h11_diag.estimate,
                "ci_low": h11_diag.ci_low,
                "ci_high": h11_diag.ci_high,
                "diagnostic_p": h11_diag.permutation_p,
                "diagnostic_q": h11_diag.q_value,
                "decision": "unavailable",
                "reason": "Molecular support can be compared with observed K562 concordance, but reliability-corrected portability is unavailable.",
                "diagnostic": h11_diag.diagnostic_decision,
            },
        ]
    )
    hypotheses.to_csv(OUT / "hypothesis_decisions.csv", index=False)
    add_storage_costs(costs).to_csv(OUT / "computational_cost.csv", index=False)
    exclusions = pd.DataFrame(
        [
            {"branch": "confirmatory external portability", "status": "unavailable", "denominator": 0, "reason": "No new metadata-eligible untouched outcome dataset is available."},
            {"branch": "leave-one-system-out applicability", "status": "underpowered", "denominator": 2, "reason": "Three compatible external systems are required; K562 and RPE1 endpoints are not common."},
            {"branch": "target-level reliability correction", "status": "unavailable", "denominator": 0, "reason": "Neither external output retains three target-level reliability units."},
            {"branch": "RPE1 signed-profile concordance", "status": "unavailable", "denominator": 490, "reason": "Stored RPE1 benchmark output retains response-count predictions, not target-level signed cross-system concordance."},
        ]
    )
    exclusions.to_csv(OUT / "exclusions.csv", index=False)
    audit = {
        "k562_rows": len(k562),
        "k562_targets": int(k562.target_gene.nunique()),
        "rpe1_targets": int(rpe1.target_gene.nunique()),
        "complete_state_targets": int(len(decomposition)),
        "compatible_external_systems": 2,
        "minimum_systems_required": int(transport["minimum_systems_for_applicability"]),
        "confirmatory_holdout_available": False,
        "all_hypotheses_reported": hypotheses.hypothesis.tolist(),
        "all_hypothesis_decisions": hypotheses.decision.value_counts().to_dict(),
        "reliability_method": "bounded partial identification; no point division",
        "rerouting_thresholds": rerouting_thresholds,
        "bootstrap_replicates": bootstraps,
        "permutation_replicates": permutations,
        "rpe1_interpretation": "developmental only",
        "rpe1_applicability_features": features,
        "source_manifest_sha256": sha256(source_manifest),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
