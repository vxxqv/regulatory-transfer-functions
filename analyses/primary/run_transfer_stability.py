from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses/primary/results"
CONFIG = ROOT / "config/strengthening_extensions.yaml"
STATES = ["Rest", "Stim8hr", "Stim48hr"]


def fold_id(target: str) -> int:
    return 1 + int(hashlib.sha256(target.encode()).hexdigest()[:16], 16) % 10


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(ranked, 1.0)
    return result


def prepare() -> tuple[pd.DataFrame, list[str]]:
    rows = pd.read_parquet(OUT / "transfer_phenotypes.parquet")
    evidence = pd.read_parquet(
        ROOT / "analyses/molecular_cascade/results/target_state_evidence.parquet",
        columns=[
            "target_contrast",
            "culture_condition",
            "mean_knockdown_fraction",
            "correlation_all",
        ],
    )
    rows = rows.merge(
        evidence,
        on=["target_contrast", "culture_condition"],
        how="left",
        validate="one_to_one",
    )
    curated = pd.read_parquet(ROOT / "analyses/molecular_cascade/results/curated_regulatory_edges.parquet")
    degree = (
        curated.loc[curated["primary_curated"]]
        .groupby("source_gene")["response_gene"]
        .nunique()
    )
    rows["curated_out_degree"] = rows["target_contrast_gene_name"].str.upper().map(degree)
    rows["curated_degree_covered"] = rows["curated_out_degree"].notna().astype(float)
    rows["curated_out_degree"] = np.log1p(rows["curated_out_degree"].fillna(0))
    rows["log1p_guide_count"] = np.log1p(rows["n_guides"])
    columns = [
        "cis_magnitude",
        "log1p_target_baseMean",
        "log1p_n_cells_target",
        "log1p_guide_count",
        "guide_correlation_all",
        "donor_correlation_all_mean",
        "mean_knockdown_fraction",
        "curated_out_degree",
        "curated_degree_covered",
    ]
    return rows, columns


def matrix(frame: pd.DataFrame, columns: list[str], medians: pd.Series, means: pd.Series, scales: pd.Series) -> np.ndarray:
    numeric = frame[columns].copy()
    missing = numeric[["guide_correlation_all", "donor_correlation_all_mean", "mean_knockdown_fraction"]].isna().astype(float)
    missing.columns = [f"{column}_missing" for column in missing.columns]
    numeric = numeric.fillna(medians)
    numeric = (numeric - means) / scales
    state = pd.get_dummies(frame["culture_condition"], dtype=float).reindex(columns=STATES, fill_value=0)
    return np.column_stack([numeric.to_numpy(float), missing.to_numpy(float), state.to_numpy(float)])


def cross_fitted_residuals(rows: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(len(rows), np.nan)
    folds = rows["target_contrast"].astype(str).map(fold_id).to_numpy()
    y = rows["transfer_z"].to_numpy(float)
    for fold in range(1, 11):
        train = folds != fold
        test = folds == fold
        medians = rows.loc[train, columns].median()
        means = rows.loc[train, columns].fillna(medians).mean()
        scales = rows.loc[train, columns].fillna(medians).std().replace(0, 1)
        x_train = matrix(rows.loc[train], columns, medians, means, scales)
        x_test = matrix(rows.loc[test], columns, medians, means, scales)
        model = Ridge(alpha=1.0).fit(x_train, y[train])
        predictions[test] = model.predict(x_test)
    if not np.isfinite(predictions).all():
        raise RuntimeError("cross-fitted predictions are incomplete")
    return predictions, y - predictions


def variance_components(targets: pd.DataFrame, multiplier: float) -> tuple[float, float, np.ndarray]:
    denominator = float((targets["observations"] - 1).clip(lower=0).sum())
    sigma2 = float(targets["within_ss"].sum() / denominator) if denominator > 0 else np.nan
    tau2 = float(max(targets["unshrunken"].var(ddof=1) - sigma2 * np.mean(1 / targets["observations"]), 0))
    if not np.isfinite(sigma2) or not np.isfinite(tau2):
        raise RuntimeError("variance components are not finite")
    if tau2 == 0:
        weight = np.zeros(len(targets))
    else:
        weight = tau2 / (tau2 + multiplier * sigma2 / targets["observations"].to_numpy(float))
    return sigma2, tau2, weight


def classify(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where(values <= low, "buffered", np.where(values >= high, "amplified", "intermediate"))


def bootstrap(targets: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    values = targets["unshrunken"].to_numpy(float)
    for replicate in range(replicates):
        sample = targets.iloc[rng.integers(0, len(targets), len(targets))].reset_index(drop=True)
        _, _, weight = variance_components(sample, 1.0)
        unshrunken = sample["unshrunken"].to_numpy(float)
        shrunken = weight * unshrunken
        low, high = np.quantile(unshrunken, [0.1, 0.9])
        slow, shigh = np.quantile(shrunken, [0.1, 0.9])
        original = classify(unshrunken, low, high)
        rank_class = classify(shrunken, slow, shigh)
        fixed_class = classify(shrunken, low, high)
        records.append(
            {
                "replicate": replicate,
                "spearman": spearmanr(unshrunken, shrunken).statistic,
                "buffered_rank_retention": np.mean(rank_class[original == "buffered"] == "buffered"),
                "amplified_rank_retention": np.mean(rank_class[original == "amplified"] == "amplified"),
                "buffered_fixed_retention": np.mean(fixed_class[original == "buffered"] == "buffered"),
                "amplified_fixed_retention": np.mean(fixed_class[original == "amplified"] == "amplified"),
            }
        )
    return pd.DataFrame(records)


def permutation_null(targets: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unshrunken = targets["unshrunken"].to_numpy(float)
    shrunken = targets["shrunken"].to_numpy(float)
    low, high = np.quantile(unshrunken, [0.1, 0.9])
    slow, shigh = np.quantile(shrunken, [0.1, 0.9])
    original = classify(unshrunken, low, high)
    strata = targets.groupby(["state_pattern", "degree_quintile"]).indices.values()
    records = []
    for replicate in range(replicates):
        permuted = shrunken.copy()
        for indices in strata:
            permuted[indices] = rng.permutation(permuted[indices])
        permuted_class = classify(permuted, slow, shigh)
        records.append(
            {
                "replicate": replicate,
                "buffered_rank_retention": np.mean(permuted_class[original == "buffered"] == "buffered"),
                "amplified_rank_retention": np.mean(permuted_class[original == "amplified"] == "amplified"),
            }
        )
    return pd.DataFrame(records)


def influence(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    medians = rows[columns].median()
    means = rows[columns].fillna(medians).mean()
    scales = rows[columns].fillna(medians).std().replace(0, 1)
    x = matrix(rows, columns, medians, means, scales)
    x = np.column_stack([np.ones(len(x)), x])
    y = rows["transfer_z"].to_numpy(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    inverse = np.linalg.pinv(x.T @ x)
    leverage = np.einsum("ij,jk,ik->i", x, inverse, x)
    mse = np.square(residual).sum() / max(len(y) - x.shape[1], 1)
    cooks = np.square(residual) / (x.shape[1] * mse) * leverage / np.square(np.maximum(1 - leverage, 1e-12))
    table = pd.DataFrame({"target": rows["target_contrast"], "cooks_distance": cooks, "leverage": leverage})
    return table.groupby("target", as_index=False).agg(max_cooks_distance=("cooks_distance", "max"), max_leverage=("leverage", "max")).sort_values("max_cooks_distance", ascending=False)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    spec = config["families"]["transfer_gain_buffering"]
    rows, columns = prepare()
    rows["oof_covariate_prediction"], rows["adjusted_transfer"] = cross_fitted_residuals(rows, columns)
    grouped = rows.groupby("target_contrast", sort=True)
    targets = grouped.agg(
        target_gene=("target_contrast_gene_name", "first"),
        observations=("culture_condition", "size"),
        unshrunken=("adjusted_transfer", "mean"),
        response_degree=("n_downstream", "mean"),
        state_pattern=("culture_condition", lambda x: "|".join(sorted(x))),
    ).reset_index().rename(columns={"target_contrast": "target"})
    mean_lookup = targets.set_index("target")["unshrunken"]
    rows["target_mean"] = rows["target_contrast"].map(mean_lookup)
    rows["within_square"] = np.square(rows["adjusted_transfer"] - rows["target_mean"])
    within = rows.groupby("target_contrast")["within_square"].sum()
    targets["within_ss"] = targets["target"].map(within).fillna(0)
    targets["degree_quintile"] = pd.qcut(targets["response_degree"].rank(method="first"), 5, labels=False)
    sigma2, tau2, weight = variance_components(targets, 1.0)
    targets["shrinkage_weight"] = weight
    targets["shrunken"] = targets["unshrunken"] * weight
    low, high = targets["unshrunken"].quantile([0.1, 0.9])
    slow, shigh = targets["shrunken"].quantile([0.1, 0.9])
    targets["original_class"] = classify(targets["unshrunken"].to_numpy(), low, high)
    targets["rank_shrunken_class"] = classify(targets["shrunken"].to_numpy(), slow, shigh)
    targets["fixed_threshold_class"] = classify(targets["shrunken"].to_numpy(), low, high)
    boot = bootstrap(targets, int(config["bootstrap_replicates"]), int(config["seed"]))
    null = permutation_null(targets, int(config["permutation_replicates"]), int(config["seed"]) + 1)
    metrics = {
        "spearman": spearmanr(targets["unshrunken"], targets["shrunken"]).statistic,
        "buffered_rank_retention": np.mean(targets.loc[targets.original_class.eq("buffered"), "rank_shrunken_class"].eq("buffered")),
        "amplified_rank_retention": np.mean(targets.loc[targets.original_class.eq("amplified"), "rank_shrunken_class"].eq("amplified")),
        "buffered_fixed_retention": np.mean(targets.loc[targets.original_class.eq("buffered"), "fixed_threshold_class"].eq("buffered")),
        "amplified_fixed_retention": np.mean(targets.loc[targets.original_class.eq("amplified"), "fixed_threshold_class"].eq("amplified")),
    }
    summary = []
    for metric, estimate in metrics.items():
        low_ci, high_ci = np.quantile(boot[metric], [0.025, 0.975])
        row = {"metric": metric, "estimate": estimate, "ci_low": low_ci, "ci_high": high_ci, "targets": len(targets)}
        if metric in null:
            p = (1 + np.sum(null[metric] >= estimate)) / (len(null) + 1)
            row.update({"permutation_p": p, "null_median": null[metric].median()})
        summary.append(row)
    summary = pd.DataFrame(summary)
    tested = summary["permutation_p"].notna()
    summary.loc[tested, "q_value"] = bh(summary.loc[tested, "permutation_p"].to_numpy(float))
    decision = (
        summary.set_index("metric").loc["spearman", "ci_low"] > 0.60
        and summary.set_index("metric").loc["buffered_rank_retention", "ci_low"] > 0.50
        and summary.set_index("metric").loc["amplified_rank_retention", "ci_low"] > 0.50
    )
    summary["decision"] = np.where(summary.metric.isin(["spearman", "buffered_rank_retention", "amplified_rank_retention"]), "supported" if decision else "mixed", "descriptive")
    sensitivity = []
    for multiplier in spec["measurement_error_sensitivity"]:
        local_sigma2, local_tau2, local_weight = variance_components(targets, float(multiplier))
        values = targets["unshrunken"].to_numpy() * local_weight
        qlow, qhigh = np.quantile(values, [0.1, 0.9])
        local_class = classify(values, qlow, qhigh)
        sensitivity.append(
            {
                "residual_variance_multiplier": multiplier,
                "sigma2": local_sigma2,
                "tau2": local_tau2,
                "median_shrinkage_weight": np.median(local_weight),
                "spearman": spearmanr(targets["unshrunken"], values).statistic,
                "buffered_rank_retention": np.mean(local_class[targets.original_class.eq("buffered")] == "buffered"),
                "amplified_rank_retention": np.mean(local_class[targets.original_class.eq("amplified")] == "amplified"),
            }
        )
    influence_table = influence(rows, columns)
    targets.to_parquet(OUT / "transfer_stability_targets.parquet", index=False)
    summary.to_csv(OUT / "transfer_stability_summary.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "transfer_measurement_error_sensitivity.csv", index=False)
    influence_table.to_csv(OUT / "transfer_stability_influence.csv", index=False)
    audit = {
        "rows": len(rows),
        "targets": len(targets),
        "original_buffered_rows": int(rows.transfer_class.eq("buffered").sum()),
        "original_amplified_rows": int(rows.transfer_class.eq("amplified").sum()),
        "target_tail_size": int(targets.original_class.eq("buffered").sum()),
        "sigma2": sigma2,
        "tau2": tau2,
        "decision": "supported" if decision else "mixed",
        "response_degree_adjustment": "stratified diagnostic only because response degree defines the transfer outcome",
        "measurement_error_status": "bounded sensitivity because target-state sampling variances are unavailable",
        "seed": int(config["seed"]),
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "permutation_replicates": int(config["permutation_replicates"]),
    }
    (OUT / "transfer_stability_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
