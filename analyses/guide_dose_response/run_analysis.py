"""Fit pre-specified guide-level regulatory dose-response models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/guide_dose_response/results"
UPSTREAM = ROOT / "work/upstream/GWT_perturbseq_analysis_2025/metadata"


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def pair_center(matrix: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(matrix)
    return (frame - frame.groupby(groups).transform("mean")).to_numpy()


def prepare(config: dict) -> pd.DataFrame:
    kd = pd.read_csv(UPSTREAM / "suppl_tables/guide_kd_efficiency.suppl_table.csv", index_col=0).rename_axis("guide_id").reset_index()
    pairs = pd.read_csv(UPSTREAM / "DE_by_guide.correlation_results.csv", index_col=0)
    library = pd.read_csv(UPSTREAM / "sgrna_df_final.csv", index_col=0)
    long = []
    for suffix in ["1", "2"]:
        block = pairs[["target", "culture_condition", f"guide_id_{suffix}", f"n_signif_{suffix}", "correlation_signif", "correlation_all", "n_signif_union"]].copy()
        block = block.rename(columns={f"guide_id_{suffix}": "guide_id", f"n_signif_{suffix}": "downstream_genes"})
        block["guide_position"] = int(suffix)
        long.append(block)
    data = pd.concat(long, ignore_index=True)
    data["pair_id"] = data["target"].astype(str) + "|" + data["culture_condition"].astype(str)
    data = data.merge(kd, on=["guide_id", "culture_condition"], how="left", validate="many_to_one")
    quality = library.rename(columns={"sgRNA": "guide_id"}).drop_duplicates("guide_id")
    quality["gc_fraction"] = quality["seq"].astype(str).str.upper().map(lambda x: (x.count("G") + x.count("C")) / max(len(x), 1))
    quality["library_flag"] = quality["flag"].fillna(False).astype(bool).astype(int)
    quality["bidirectional_promoter"] = quality["putative_bidirectional_promoter"].fillna(False).astype(bool).astype(int)
    quality["secondary_alignment"] = quality["other_alignment_chromosome"].notna().astype(int)
    quality["log1p_tss_distance"] = np.log1p(pd.to_numeric(quality["distance_to_closest_target_tss"], errors="coerce").abs())
    data = data.merge(quality[["guide_id", "gc_fraction", "library_flag", "bidirectional_promoter", "secondary_alignment", "log1p_tss_distance"]], on="guide_id", how="left", validate="many_to_one")
    data["log1p_guide_cells"] = np.log1p(data["guide_n"])
    floor = float(config["eligibility"]["minimum_ntc_expression"])
    data["knockdown_fraction"] = ((data["ntc_mean_expr"] - data["guide_mean_expr"]) / data["ntc_mean_expr"].clip(lower=floor)).clip(-1.0, 1.5)
    pooled_sd = np.sqrt((data["guide_std_expr"].pow(2) + data["ntc_std_expr"].pow(2)) / 2).clip(lower=0.01)
    data["standardized_difference"] = ((data["ntc_mean_expr"] - data["guide_mean_expr"]) / pooled_sd).clip(-5.0, 5.0)
    data["t_scaled"] = (-data["t_statistic"] / np.sqrt(data["guide_n"].clip(lower=1))).clip(-5.0, 5.0)
    data["response"] = np.log1p(data["downstream_genes"].clip(lower=0))
    minimum_cells = int(config["eligibility"]["minimum_guide_cells"])
    row_ok = data["guide_n"].ge(minimum_cells) & data["ntc_mean_expr"].ge(floor) & data["response"].notna() & data["knockdown_fraction"].notna()
    counts = data.loc[row_ok].groupby("pair_id")["guide_id"].nunique()
    eligible_pairs = counts[counts == int(config["eligibility"]["guides_per_target_state"])].index
    data = data[row_ok & data["pair_id"].isin(eligible_pairs)].copy()
    for column in config["quality_covariates"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())
    return data.sort_values(["culture_condition", "pair_id", "guide_position"]).reset_index(drop=True)


def fit_score(train: pd.DataFrame, family: str, dose: str, quality: list[str]):
    x = train[dose].to_numpy(dtype=float)
    groups = train["pair_id"].to_numpy()
    q = train[quality].to_numpy(dtype=float)
    q_mean, q_sd = q.mean(axis=0), q.std(axis=0)
    q_sd[q_sd == 0] = 1.0
    qz = (q - q_mean) / q_sd
    yc = pair_center(train[["response"]].to_numpy(), groups).ravel()

    def solve(features: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
        centered = pair_center(features, groups)
        return np.linalg.solve(centered.T @ centered + ridge * np.eye(centered.shape[1]), centered.T @ yc)

    if family == "null":
        beta = solve(qz)
        return lambda frame: ((frame[quality].to_numpy(dtype=float) - q_mean) / q_sd) @ beta, {}
    if family == "linear":
        beta = solve(np.column_stack([x, qz]))
        return lambda frame: np.column_stack([frame[dose].to_numpy(dtype=float), (frame[quality].to_numpy(dtype=float) - q_mean) / q_sd]) @ beta, {}
    if family == "spline":
        transformer = SplineTransformer(n_knots=5, degree=3, include_bias=False)
        basis = transformer.fit_transform(x[:, None])
        beta = solve(np.column_stack([basis, qz]), ridge=1.0)
        return lambda frame: np.column_stack([transformer.transform(frame[[dose]].to_numpy(dtype=float)), (frame[quality].to_numpy(dtype=float) - q_mean) / q_sd]) @ beta, {"knots": 5}
    if family == "breakpoint":
        knot = float(np.median(x))
        beta = solve(np.column_stack([x, np.maximum(x - knot, 0), qz]))
        return lambda frame: np.column_stack([frame[dose].to_numpy(dtype=float), np.maximum(frame[dose].to_numpy(dtype=float) - knot, 0), (frame[quality].to_numpy(dtype=float) - q_mean) / q_sd]) @ beta, {"breakpoint": knot}
    if family == "hill":
        best = None
        for exponent in [0.5, 1.0, 2.0, 4.0]:
            for half in np.quantile(x, [0.25, 0.5, 0.75]):
                positive = np.clip(x, 0, None)
                hill = positive**exponent / (positive**exponent + max(float(half), 0.01) ** exponent)
                features = np.column_stack([hill, qz])
                beta = solve(features)
                residual = yc - pair_center(features, groups) @ beta
                aic = len(train) * np.log(np.mean(residual**2) + 1e-12) + 2 * len(beta)
                if best is None or aic < best[0]:
                    best = (aic, exponent, float(half), beta)
        _, exponent, half, beta = best
        def hill_score(frame: pd.DataFrame) -> np.ndarray:
            values = np.clip(frame[dose].to_numpy(dtype=float), 0, None)
            curve = values**exponent / (values**exponent + max(half, 0.01) ** exponent)
            return np.column_stack([curve, (frame[quality].to_numpy(dtype=float) - q_mean) / q_sd]) @ beta
        return hill_score, {"hill_exponent": exponent, "hill_half": half}
    qbeta = solve(qz)
    residual = yc - pair_center(qz, groups) @ qbeta
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, residual)
    return lambda frame: ((frame[quality].to_numpy(dtype=float) - q_mean) / q_sd) @ qbeta + isotonic.predict(frame[dose].to_numpy(dtype=float)), {}


def predict_pairs(test: pd.DataFrame, scorer) -> pd.DataFrame:
    records = []
    for _, block in test.groupby("pair_id"):
        if len(block) != 2:
            continue
        block = block.reset_index(drop=True)
        scores = scorer(block)
        for held, anchor in [(0, 1), (1, 0)]:
            records.append({
                "pair_id": block.loc[held, "pair_id"], "target": block.loc[held, "target"], "culture_condition": block.loc[held, "culture_condition"],
                "held_out_guide": block.loc[held, "guide_id"], "anchor_guide": block.loc[anchor, "guide_id"], "observed": block.loc[held, "response"],
                "predicted": block.loc[anchor, "response"] + scores[held] - scores[anchor], "dose": block.loc[held, "knockdown_fraction"],
            })
    return pd.DataFrame(records)


def metrics(predictions: pd.DataFrame) -> dict:
    observed = predictions["observed"].to_numpy()
    predicted = predictions["predicted"].to_numpy()
    slope, intercept = np.polyfit(predicted, observed, 1) if np.std(predicted) > 0 else (np.nan, np.nan)
    return {"rows": len(predictions), "pairs": predictions["pair_id"].nunique(), "r2": r2_score(observed, predicted), "rmse": mean_squared_error(observed, predicted) ** 0.5, "mae": mean_absolute_error(observed, predicted), "calibration_slope": slope, "calibration_intercept": intercept}


def main() -> None:
    config = yaml.safe_load((ROOT / "config/guide_dose_response.yaml").read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))
    data = prepare(config)
    quality = list(config["quality_covariates"])
    families = list(config["models"])
    fold_rows, prediction_rows, parameter_rows = [], [], []
    for state, state_data in data.groupby("culture_condition"):
        pair_ids = state_data["pair_id"].drop_duplicates().to_numpy()
        splitter = KFold(n_splits=int(config["validation"]["target_pair_folds"]), shuffle=True, random_state=int(config["seed"]))
        for fold, (train_index, test_index) in enumerate(splitter.split(pair_ids), start=1):
            train_pairs, test_pairs = pair_ids[train_index], pair_ids[test_index]
            train = state_data[state_data["pair_id"].isin(train_pairs)]
            test = state_data[state_data["pair_id"].isin(test_pairs)]
            for family in families:
                scorer, params = fit_score(train, family, "knockdown_fraction", quality)
                prediction = predict_pairs(test, scorer)
                prediction["model"] = family; prediction["fold"] = fold
                prediction_rows.append(prediction)
                fold_rows.append({"culture_condition": state, "fold": fold, "model": family, **metrics(prediction)})
                parameter_rows.append({"culture_condition": state, "fold": fold, "model": family, **params})
    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    model_metrics = pd.DataFrame([
        {"culture_condition": state_name, "model": model_name, **metrics(group)}
        for (state_name, model_name), group in predictions.groupby(["culture_condition", "model"])
    ])
    comparisons = []
    for state, block in predictions.groupby("culture_condition"):
        null = block[block["model"] == "null"].copy()
        null_error = null.assign(error=(null.observed - null.predicted) ** 2).groupby("pair_id").error.mean()
        for family in families:
            current = block[block["model"] == family].copy()
            current_error = current.assign(error=(current.observed - current.predicted) ** 2).groupby("pair_id").error.mean()
            shared = null_error.index.intersection(current_error.index)
            p = 1.0 if family == "null" else float(wilcoxon(current_error.loc[shared], null_error.loc[shared], alternative="less").pvalue)
            current_match = model_metrics[(model_metrics["culture_condition"] == state) & (model_metrics["model"] == family)]
            null_match = model_metrics[(model_metrics["culture_condition"] == state) & (model_metrics["model"] == "null")]
            current_r2 = float(current_match["r2"].iloc[0])
            null_r2 = float(null_match["r2"].iloc[0])
            boot = []
            pair_array = np.array(shared)
            for _ in range(int(config["validation"]["bootstrap_replicates"])):
                sample = rng.choice(pair_array, len(pair_array), replace=True)
                chosen = current.set_index("pair_id").loc[sample].reset_index()
                chosen_null = null.set_index("pair_id").loc[sample].reset_index()
                boot.append(r2_score(chosen.observed, chosen.predicted) - r2_score(chosen_null.observed, chosen_null.predicted))
            comparisons.append({"culture_condition": state, "model": family, "delta_r2_vs_null": current_r2 - null_r2, "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975), "paired_error_p": p})
    comparisons = pd.DataFrame(comparisons)
    comparisons["q_value"] = bh(comparisons["paired_error_p"].to_numpy())
    comparisons["supported"] = (comparisons["delta_r2_vs_null"] > 0) & (comparisons["q_value"] < float(config["fdr"]))

    curves = []
    for state, state_data in data.groupby("culture_condition"):
        grid = np.linspace(state_data["knockdown_fraction"].quantile(0.01), state_data["knockdown_fraction"].quantile(0.99), 120)
        reference = pd.DataFrame({column: state_data[column].median() for column in quality}, index=range(len(grid)))
        reference["knockdown_fraction"] = grid
        for family in families:
            scorer, params = fit_score(state_data, family, "knockdown_fraction", quality)
            score = scorer(reference); score = score - score[0]
            for dose_value, value in zip(grid, score, strict=True): curves.append({"culture_condition": state, "model": family, "dose": dose_value, "relative_log1p_downstream": value, **params})
    curves = pd.DataFrame(curves)

    sensitivity = []
    for state, state_data in data.groupby("culture_condition"):
        for dose in config["dose_definitions"]:
            scorer, _ = fit_score(state_data, "linear", dose, quality)
            score = scorer(state_data)
            centered_score = pair_center(score[:, None], state_data["pair_id"].to_numpy()).ravel()
            centered_y = pair_center(state_data[["response"]].to_numpy(), state_data["pair_id"].to_numpy()).ravel()
            slope = np.dot(centered_score, centered_y) / max(np.dot(centered_score, centered_score), 1e-12)
            sensitivity.append({"culture_condition": state, "dose_definition": dose, "scaled_association": slope, "rows": len(state_data), "pairs": state_data.pair_id.nunique()})
    sensitivity = pd.DataFrame(sensitivity)

    permutation_rows = []
    for state, block in data.groupby("culture_condition"):
        pivot = block.pivot(index="pair_id", columns="guide_position", values=["knockdown_fraction", "response"]).dropna()
        dx = pivot[("knockdown_fraction", 2)] - pivot[("knockdown_fraction", 1)]
        dy = pivot[("response", 2)] - pivot[("response", 1)]
        observed = float(np.dot(dx, dy) / max(np.dot(dx, dx), 1e-12))
        for replicate in range(int(config["validation"]["permutation_replicates"])):
            signs = rng.choice([-1.0, 1.0], len(dx))
            permuted = dx.to_numpy() * signs
            slope = float(np.dot(permuted, dy) / max(np.dot(permuted, permuted), 1e-12))
            permutation_rows.append({"culture_condition": state, "replicate": replicate + 1, "permuted_slope": slope, "observed_slope": observed})
    permutations = pd.DataFrame(permutation_rows)

    target_report = data.groupby(["target", "culture_condition", "pair_id"]).agg(guides=("guide_id", "nunique"), minimum_cells=("guide_n", "min"), dose_range=("knockdown_fraction", lambda x: x.max() - x.min()), downstream_range=("downstream_genes", lambda x: x.max() - x.min()), high_confidence_no_effect_guides=("high_confidence_no_effect_guides", "sum")).reset_index()
    target_report["target_specific_nonlinearity"] = "underpowered"
    target_report["minimum_required_dose_levels"] = 4
    negative = data.groupby(["culture_condition", "high_confidence_no_effect_guides"]).agg(guides=("guide_id", "size"), median_downstream=("downstream_genes", "median"), median_dose=("knockdown_fraction", "median")).reset_index()
    unavailable = pd.DataFrame([
        {"estimand": "donor-held-out guide-dose validation", "status": "unavailable", "reason": "donor-specific effects were released only at target level, not jointly by guide"},
        {"estimand": "signed response norm by guide", "status": "unavailable", "reason": "guide-level signed coefficient vectors were not released"},
        {"estimand": "module activity and rerouting by guide", "status": "unavailable", "reason": "guide-level signed coefficient vectors were not released"},
        {"estimand": "disease-program convergence by guide", "status": "unavailable", "reason": "disease programs were defined at target-state level"},
        {"estimand": "non-targeting trans-response control", "status": "unavailable", "reason": "the released guide-response table contains paired targeting guides only"},
        {"estimand": "cis mediation", "status": "not tested", "reason": "two guide doses per target and no joint donor-specific guide effects do not identify mediation"},
    ])
    data.to_csv(OUTPUT / "eligible_guide_rows.csv", index=False)
    target_report.to_csv(OUTPUT / "eligible_target_states.csv", index=False)
    predictions.to_parquet(OUTPUT / "guide_held_out_predictions.parquet", index=False)
    fold_metrics.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    model_metrics.to_csv(OUTPUT / "model_metrics.csv", index=False)
    comparisons.to_csv(OUTPUT / "model_comparisons.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(OUTPUT / "model_parameters.csv", index=False)
    curves.to_csv(OUTPUT / "state_curves.csv", index=False)
    sensitivity.to_csv(OUTPUT / "measurement_error_sensitivity.csv", index=False)
    permutations.to_csv(OUTPUT / "matched_guide_permutations.csv", index=False)
    negative.to_csv(OUTPUT / "negative_controls.csv", index=False)
    unavailable.to_csv(OUTPUT / "unavailable_estimands.csv", index=False)
    audit = {"eligible_guide_rows": int(len(data)), "eligible_target_states": int(data.pair_id.nunique()), "targets": int(data.target.nunique()), "models": families, "states": sorted(data.culture_condition.unique()), "target_specific_nonlinearity_estimable": 0, "donor_held_out_estimable": False}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from analyses.guide_dose_response.finalize_results import finalize
    finalize(OUTPUT, int(config["seed"]), int(config["validation"]["bootstrap_replicates"]), quality)


if __name__ == "__main__":
    main()
