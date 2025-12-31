"""Analyse released guide-pair summaries without reconstructing unavailable vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from freeze import sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses/guide_concordance/results"
KEY = ["target", "culture_condition"]


def bh(values):
    values = np.asarray(values, float)
    output = np.full(len(values), np.nan)
    ok = np.flatnonzero(np.isfinite(values))
    order = ok[np.argsort(values[ok])]
    output[order] = np.minimum(1, np.minimum.accumulate((values[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return output


def write(frame, name):
    path = OUT / name
    frame.to_parquet(path, index=False) if name.endswith("parquet") else frame.to_csv(path, index=False)


def bootstrap_median(values, rng, count, statistic="median"):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    function = np.mean if statistic == "mean" else np.median
    draws = np.array([function(rng.choice(values, len(values), replace=True)) for _ in range(count)])
    return function(values), *np.quantile(draws, [.025, .975])


def prepare(source, cfg):
    meta = source / "work/upstream/GWT_perturbseq_analysis_2025/metadata"
    pairs = pd.read_csv(meta / "DE_by_guide.correlation_results.csv", index_col=0)
    pairs["released_pair"] = True
    primary = pd.read_parquet(source / "analyses/primary/results/transfer_phenotypes.parquet").rename(columns={"target_contrast_gene_name": "target"})
    primary = primary[KEY + ["cis_magnitude", "target_baseMean", "n_downstream", "transfer_z", "transfer_class"]]
    primary["in_primary_study"] = True
    rows = pairs.merge(primary, on=KEY, how="outer", validate="one_to_one")
    rows["released_pair"] = rows.released_pair.eq(True)
    rows["in_primary_study"] = rows.in_primary_study.eq(True)
    kd = pd.read_csv(meta / "suppl_tables/guide_kd_efficiency.suppl_table.csv", index_col=0).rename_axis("guide_id").reset_index()
    quality = pd.read_csv(meta / "sgrna_df_final.csv", index_col=0).rename(columns={"sgRNA": "guide_id"})
    assert not quality.guide_id.duplicated().any()
    quality["gc_fraction"] = quality.seq.str.upper().map(lambda x: (x.count("G") + x.count("C")) / len(x))
    quality["log1p_tss_distance"] = np.log1p(pd.to_numeric(quality.distance_to_closest_target_tss, errors="coerce").abs())
    quality["library_flag"] = quality.flag.fillna(False).astype(bool).astype(float)
    quality["bidirectional_promoter"] = quality.putative_bidirectional_promoter.fillna(False).astype(bool).astype(float)
    quality["secondary_alignment"] = quality.other_alignment_chromosome.notna().astype(float)
    qcols = ["gc_fraction", "log1p_tss_distance", "library_flag", "bidirectional_promoter", "secondary_alignment"]
    kd = kd.merge(quality[["guide_id"] + qcols], on="guide_id", how="left", validate="many_to_one")
    for suffix in ["1", "2"]:
        block = kd.rename(columns={c: f"{c}_{suffix}" for c in kd.columns if c != "culture_condition"})
        rows = rows.merge(block, on=[f"guide_id_{suffix}", "culture_condition"], how="left", validate="many_to_one")
        rows[f"knockdown_{suffix}"] = ((rows[f"ntc_mean_expr_{suffix}"] - rows[f"guide_mean_expr_{suffix}"]) / rows[f"ntc_mean_expr_{suffix}"]).clip(-1, 1.5)
    e = cfg["eligibility"]
    rows["min_guide_cells"] = rows[["guide_n_1", "guide_n_2"]].min(axis=1, skipna=False)
    rows["min_ntc_expression"] = rows[["ntc_mean_expr_1", "ntc_mean_expr_2"]].min(axis=1, skipna=False)
    rows["quality_complete"] = rows[[f"{c}_{s}" for c in qcols for s in ["1", "2"]]].notna().all(axis=1)
    rows["eligible_pair"] = rows.released_pair & rows.min_guide_cells.ge(e["minimum_guide_cells"]) & rows.min_ntc_expression.ge(e["minimum_ntc_expression"]) & rows[["knockdown_1", "knockdown_2", "n_signif_1", "n_signif_2"]].notna().all(axis=1)
    rows["significant_union_eligible"] = rows.eligible_pair & rows.n_signif_union.ge(e["minimum_significant_union"])
    rows["eligibility_status"] = np.select([~rows.released_pair, rows.min_guide_cells.isna() | rows.min_ntc_expression.isna(), rows.min_guide_cells.lt(e["minimum_guide_cells"]), rows.min_ntc_expression.lt(e["minimum_ntc_expression"]), rows.eligible_pair], ["pair_unavailable", "knockdown_metadata_unavailable", "low_cell_count", "low_proximal_expression", "eligible_descriptive_pair"], default="incomplete_pair")
    rows["target_inference_status"] = "underpowered_two_guides"
    rows.loc[~rows.released_pair, "target_inference_status"] = "unavailable"
    rows["cis_difference"] = rows.knockdown_2 - rows.knockdown_1
    rows["cis_absolute_difference"] = rows.cis_difference.abs()
    rows["count_difference"] = np.log1p(rows.n_signif_2) - np.log1p(rows.n_signif_1)
    rows["count_absolute_difference"] = rows.count_difference.abs()
    rows["count_sign_agreement"] = np.where((rows.cis_difference != 0) & (rows.count_difference != 0), (np.sign(rows.cis_difference) == np.sign(rows.count_difference)).astype(float), np.nan)
    intersection = rows.n_signif_1 + rows.n_signif_2 - rows.n_signif_union
    rows["count_set_valid"] = intersection.ge(0) & intersection.le(rows[["n_signif_1", "n_signif_2"]].min(axis=1))
    rows["significant_gene_jaccard"] = np.where(rows.count_set_valid & rows.n_signif_union.gt(0), intersection / rows.n_signif_union, np.nan)
    rows["log1p_min_guide_cells"] = np.log1p(rows.min_guide_cells)
    rows["mean_gc_fraction"] = rows[["gc_fraction_1", "gc_fraction_2"]].mean(axis=1)
    rows["mean_log1p_tss_distance"] = rows[["log1p_tss_distance_1", "log1p_tss_distance_2"]].mean(axis=1)
    for c in ["library_flag", "bidirectional_promoter", "secondary_alignment"]:
        rows[f"any_{c}"] = rows[[f"{c}_1", f"{c}_2"]].max(axis=1, skipna=False)
    rows["log1p_response_degree"] = np.log1p(rows.n_signif_union)
    rows["log1p_proximal_expression"] = np.log1p(rows.min_ntc_expression)
    denom = rows.min_ntc_expression
    guide_var = rows.guide_std_expr_1.pow(2) / rows.guide_n_1 + rows.guide_std_expr_2.pow(2) / rows.guide_n_2
    ntc_var = rows.ntc_std_expr_1.pow(2) / rows.ntc_n_1
    rows["cis_difference_se"] = np.sqrt(guide_var / denom.pow(2) + (rows.guide_mean_expr_1 - rows.guide_mean_expr_2).pow(2) * ntc_var / denom.pow(4))
    rows["shared_ntc_verified"] = np.isclose(rows.ntc_mean_expr_1, rows.ntc_mean_expr_2, equal_nan=False) & np.isclose(rows.ntc_n_1, rows.ntc_n_2, equal_nan=False)
    rows.loc[~rows.shared_ntc_verified, "cis_difference_se"] = np.nan
    for c in ["gc_fraction", "log1p_tss_distance", "library_flag", "bidirectional_promoter", "secondary_alignment"]:
        rows[f"delta_{c}"] = rows[f"{c}_2"] - rows[f"{c}_1"]
    rows["delta_log1p_cells"] = np.log1p(rows.guide_n_2) - np.log1p(rows.guide_n_1)
    return rows.sort_values(KEY).reset_index(drop=True)


def molecular_support(source, rows):
    path = source / "analyses/molecular_cascade/results/edge_evidence_matrix.parquet"
    if not path.exists():
        return rows, False
    edges = pd.read_parquet(path)
    audit = json.loads((path.parent / "audit.json").read_text())
    if len(edges) != audit["response_edge_denominator"]:
        return rows, False
    physical = edges.physical_active_link | edges.promoter_bound | edges.enhancer_linked
    edges["molecular_types"] = edges.curated_edge.astype(int) + edges.motif_supported.astype(int) + physical.astype(int)
    edges["molecular_two_types"] = edges.molecular_types.ge(2)
    edges["molecular_any_type"] = edges.molecular_types.ge(1)
    support = edges.groupby(["target_gene", "culture_condition"]).agg(molecular_support_fraction=("molecular_two_types", "mean"), molecular_any_fraction=("molecular_any_type", "mean"), evidence_response_edges=("response_gene", "size")).reset_index().rename(columns={"target_gene": "target"})
    write(support, "molecular_support.csv")
    return rows.merge(support, on=KEY, how="left", validate="one_to_one"), True


def target_demean(values, targets):
    frame = pd.DataFrame(values)
    return (frame - frame.groupby(np.asarray(targets)).transform("mean")).to_numpy()


def cluster_fit(frame, predictor, outcome, covars, cfg, rng, name, permutation="matched", target_control=False):
    columns = [predictor, outcome] + covars
    data = frame.dropna(subset=columns).copy().reset_index(drop=True)
    if target_control:
        counts = data.groupby("target").size()
        data = data.loc[data.target.isin(counts[counts >= 2].index)].reset_index(drop=True)
    n_targets = data.target.nunique()
    base = {"analysis": name, "predictor": predictor, "outcome": outcome, "n_rows": len(data), "n_targets": n_targets, "estimator": "target_controlled" if target_control else "pooled", "evaluation_scale": "within-target deviations" if target_control else "absolute outcome"}
    if n_targets < cfg["eligibility"]["minimum_targets_pooled"] or data[predictor].nunique() < 2:
        return {**base, "status": "underpowered"}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if predictor.startswith("molecular") and data.loc[data[predictor].gt(0), "target"].nunique() < cfg["eligibility"]["minimum_targets_evidence_group"]:
        return {**base, "status": "underpowered_supported_group"}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if target_control:
        varying = data.groupby("target")[predictor].agg(lambda x: float(np.ptp(x)) > 1e-12)
        base["varying_targets"] = int(varying.sum())
        if varying.sum() < cfg["eligibility"]["minimum_targets_evidence_group"]:
            return {**base, "status": "underpowered_within_target_variation"}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    state = pd.get_dummies(data.culture_condition, dtype=float, drop_first=True).to_numpy()
    cov = data[covars].to_numpy(float)
    csd = cov.std(axis=0)
    csd[csd == 0] = 1
    cov = (cov - cov.mean(axis=0)) / csd
    x = data[predictor].to_numpy(float)
    y = data[outcome].to_numpy(float)
    design = np.column_stack([np.ones(len(data)), x, cov, state])
    if target_control:
        design = target_demean(design, data.target)
        y = target_demean(y, data.target).ravel()
        x = design[:, 1]
        state = target_demean(state, data.target)
        nuisance = np.delete(design, 1, axis=1)
        residual_x = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
        if np.dot(residual_x, residual_x) < 1e-12:
            return {**base, "status": "unidentified_after_target_and_state_control"}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    groups, group_codes = np.unique(data.target, return_inverse=True)
    p = design.shape[1]
    a = np.zeros((len(groups), p, p))
    b = np.zeros((len(groups), p))
    np.add.at(a, group_codes, design[:, :, None] * design[:, None, :])
    np.add.at(b, group_codes, design * y[:, None])
    boot = []
    for _ in range(cfg["bootstrap_replicates"]):
        weights = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        boot.append(np.linalg.lstsq(np.einsum("g,gij->ij", weights, a), weights @ b, rcond=None)[0][1])
    ci = np.quantile(boot, [.025, .975])
    nuisance = np.delete(design, 1, axis=1)
    residual_y = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    residual_x = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
    null = []
    if permutation == "guide_swap":
        for _ in range(cfg["permutation_replicates"]):
            signs = rng.choice([-1, 1], len(groups))[group_codes]
            perm_y = residual_y * signs
            null.append(residual_x @ perm_y / (residual_x @ residual_x))
    else:
        target_meta = data.groupby("target").agg(expression=("log1p_proximal_expression", "mean"), degree=("log1p_response_degree", "mean"), pattern=("culture_condition", lambda x: "|".join(sorted(x))))
        target_meta["expression_bin"] = pd.qcut(target_meta.expression.rank(method="first"), 5, labels=False)
        target_meta["degree_bin"] = pd.qcut(target_meta.degree.rank(method="first"), 5, labels=False)
        target_rows = {target: block.sort_values("culture_condition").index.to_numpy() for target, block in data.groupby("target")}
        strata = [block.index.to_numpy() for _, block in target_meta.groupby(["pattern", "expression_bin", "degree_bin"])]
        for _ in range(cfg["permutation_replicates"]):
            px = residual_x.copy()
            for targets in strata:
                for destination, origin in zip(targets, rng.permutation(targets)):
                    px[target_rows[destination]] = residual_x[target_rows[origin]]
            null.append(px @ residual_y / (px @ px))
    pvalue = (1 + np.sum(np.abs(null) >= abs(beta[1]))) / (len(null) + 1)
    folds = GroupKFold(cfg["validation"]["target_held_out_folds"])
    predictions = []
    for fold, (train, test) in enumerate(folds.split(data, groups=data.target)):
        for kind, predictors in [("quality_baseline", [c for c in covars]), ("with_predictor", [predictor] + covars)]:
            all_values = data[predictors].to_numpy(float)
            if target_control:
                all_values = target_demean(all_values, data.target)
            train_values = all_values[train]
            test_values = all_values[test]
            mean, sd = train_values.mean(axis=0), train_values.std(axis=0)
            sd[sd == 0] = 1
            train_values = np.column_stack([(train_values - mean) / sd, state[train]])
            test_values = np.column_stack([(test_values - mean) / sd, state[test]])
            model = Ridge(alpha=1.0).fit(train_values, y[train])
            preds = model.predict(test_values)
            for i, pred in zip(test, preds):
                predictions.append({"analysis": name, "model": kind, "fold": fold, "target": data.target.iloc[i], "culture_condition": data.culture_condition.iloc[i], "observed": y[i], "prediction": pred, "estimator": base["estimator"], "evaluation_scale": base["evaluation_scale"]})
    pred = pd.DataFrame(predictions)
    influence = []
    for g, target in enumerate(groups):
        val = np.linalg.lstsq(a.sum(axis=0) - a[g], b.sum(axis=0) - b[g], rcond=None)[0][1]
        influence.append({"analysis": name, "target": target, "leave_one_target_coefficient": val, "change": val - beta[1]})
    result = {**base, "status": "estimated", "coefficient": beta[1], "ci_low": ci[0], "ci_high": ci[1], "permutation_p": pvalue, "null_replicates": len(null), "bootstrap_replicates": len(boot), "adjustment": ";".join(covars) + ";state" + (";target_fixed_effects" if target_control else ""), "supported_targets": data.loc[data[predictor].gt(0), "target"].nunique()}
    draws = pd.DataFrame({"analysis": name, "replicate": np.arange(len(null)), "null_coefficient": null})
    return result, draws, pred, pd.DataFrame(influence)


def state_comparisons(eligible, cfg, rng):
    records, nulls = [], []
    states = cfg["states"]
    for outcome in ["cis_absolute_difference", "count_absolute_difference"]:
        wide = eligible.pivot(index="target", columns="culture_condition", values=outcome)
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            paired = wide[[states[i], states[j]]].dropna()
            delta = (paired[states[j]] - paired[states[i]]).to_numpy()
            estimate, lo, hi = bootstrap_median(delta, rng, cfg["bootstrap_replicates"])
            null = [np.median(delta * rng.choice([-1, 1], len(delta))) for _ in range(cfg["permutation_replicates"])]
            p = (1 + np.sum(np.abs(null) >= abs(estimate))) / (len(null) + 1)
            name = f"{outcome}:{states[i]}:{states[j]}"
            records.append({"comparison": name, "outcome": outcome, "from_state": states[i], "to_state": states[j], "n_targets": len(delta), "median_change": estimate, "ci_low": lo, "ci_high": hi, "permutation_p": p})
            nulls.extend({"comparison": name, "replicate": k, "null_median": v} for k, v in enumerate(null))
    table = pd.DataFrame(records)
    table["q_value"] = bh(table.permutation_p)
    table["decision"] = np.where(table.q_value.lt(cfg["fdr"]) & ((table.ci_low > 0) | (table.ci_high < 0)), "supported_difference", "unresolved")
    write(table, "matched_state_comparisons.csv")
    write(pd.DataFrame(nulls), "matched_state_nulls.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--summaries-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config/guide_concordance.yaml").read_text())
    manifest = json.loads((ROOT / "analyses/guide_concordance/freeze_manifest.json").read_text())
    config_text = (ROOT / "config/guide_concordance.yaml").read_text(encoding="utf-8")
    assert hashlib.sha256(config_text.encode("utf-8")).hexdigest() == manifest["config_sha256"]
    for record in manifest["inputs"]:
        if record["available"]:
            assert sha256(args.source_root / record["path"]) == record["sha256"], record["path"]
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    rows = prepare(args.source_root, cfg)
    rows, evidence_available = molecular_support(args.source_root, rows)
    write(rows, "all_target_states.parquet")
    eligible = rows.loc[rows.eligible_pair].copy()
    eligible["significance_status"] = np.where(eligible.significant_union_eligible, "supported_union_size", "small_or_empty_union")
    write(eligible, "eligible_pairs.parquet")
    summary = []
    metrics = ["correlation_all", "correlation_signif", "frac_sign_agreement_signif_union", "significant_gene_jaccard", "cis_absolute_difference", "count_absolute_difference", "count_sign_agreement"]
    for state, frame in eligible.groupby("culture_condition"):
        for metric in metrics:
            subset = frame.loc[frame.significant_union_eligible] if metric in metrics[1:4] else frame
            vals = subset[metric].dropna()
            if len(vals):
                statistic = "mean" if metric == "count_sign_agreement" else "median"
                med, lo, hi = bootstrap_median(vals, rng, cfg["bootstrap_replicates"], statistic)
                summary.append({"culture_condition": state, "metric": metric, "n_targets": len(vals), "statistic": statistic, "estimate": med, "ci_low": lo, "ci_high": hi})
    write(pd.DataFrame(summary), "state_summaries.csv")
    counts = rows.groupby("culture_condition").agg(complete_denominator=("target", "size"), released_pairs=("released_pair", "sum"), primary_target_states=("in_primary_study", "sum"), eligible_pairs=("eligible_pair", "sum"), significant_union_eligible=("significant_union_eligible", "sum")).reset_index()
    write(counts, "denominators.csv")
    if args.summaries_only:
        print("Descriptive summaries refreshed; validated models retained.")
        return
    results, nulls, predictions, influences = [], [], [], []
    qualities = cfg["quality_covariates"] + ["log1p_proximal_expression", "log1p_response_degree", "cis_magnitude"]
    for predictor in ["molecular_support_fraction", "molecular_any_fraction"] if evidence_available else []:
        for outcome in ["correlation_all", "correlation_signif", "frac_sign_agreement_signif_union"]:
            subset = eligible.loc[eligible.in_primary_study & eligible.quality_complete].copy()
            if outcome != "correlation_all":
                subset = subset.loc[subset.significant_union_eligible]
            result, null, pred, influence = cluster_fit(subset, predictor, outcome, qualities, cfg, rng, f"{predictor}:{outcome}")
            results.append(result)
            nulls.append(null)
            predictions.append(pred)
            influences.append(influence)
    difference_covars = ["delta_gc_fraction", "delta_log1p_tss_distance", "delta_library_flag", "delta_bidirectional_promoter", "delta_secondary_alignment", "delta_log1p_cells"]
    result, null, pred, influence = cluster_fit(eligible, "cis_difference", "count_difference", difference_covars, cfg, rng, "paired_cis_count", permutation="guide_swap")
    results.append(result)
    nulls.append(null)
    predictions.append(pred)
    influences.append(influence)
    tests = pd.DataFrame(results)
    tests["q_value"] = bh(tests.permutation_p)
    tests["decision"] = np.where(tests.q_value.lt(cfg["fdr"]) & tests.ci_low.gt(0), "supported_positive", np.where(tests.q_value.lt(cfg["fdr"]) & tests.ci_high.lt(0), "supported_negative", "unresolved"))
    tests.loc[tests.status.ne("estimated"), "decision"] = tests.loc[tests.status.ne("estimated"), "status"]
    write(tests, "hypothesis_tests.csv")
    write(pd.concat(nulls, ignore_index=True), "permutation_nulls.parquet")
    pred = pd.concat(predictions, ignore_index=True)
    write(pred, "target_held_out_predictions.parquet")
    write(pd.concat(influences, ignore_index=True), "target_influence.csv")
    calibration = []
    for (analysis, model), data in pred.groupby(["analysis", "model"]):
        slope, intercept = np.polyfit(data.prediction, data.observed, 1)
        calibration.append({"analysis": analysis, "model": model, "n_rows": len(data), "r2": r2_score(data.observed, data.prediction), "mae": mean_absolute_error(data.observed, data.prediction), "calibration_slope": slope, "calibration_intercept": intercept})
    write(pd.DataFrame(calibration), "calibration.csv")
    sensitivity = []
    data = eligible.dropna(subset=["cis_difference", "cis_difference_se", "count_difference"] + difference_covars).copy()
    design = np.column_stack([np.ones(len(data)), data.cis_difference, data[difference_covars], pd.get_dummies(data.culture_condition, dtype=float, drop_first=True)])
    for scale in cfg["validation"]["measurement_error_scales"]:
        samples = []
        for _ in range(200 if scale else 1):
            x = design.copy()
            x[:, 1] += rng.normal(0, scale * data.cis_difference_se)
            samples.append(np.linalg.lstsq(x, data.count_difference, rcond=None)[0][1])
        sensitivity.append({"error_scale": scale, "n_pairs": len(data), "coefficient_median": np.median(samples), "simulation_low": np.quantile(samples, .025), "simulation_high": np.quantile(samples, .975), "interval_type": "measurement-error simulation range, not a sampling confidence interval"})
    write(pd.DataFrame(sensitivity), "measurement_error_sensitivity.csv")
    state_comparisons(eligible, cfg, rng)
    unavailable = [
        ("guide-by-donor pseudobulk", "unavailable", "No joint count matrix, donor labels and guide assignments among frozen inputs."),
        ("full and significant-union Spearman, cosine, rank, signed norm", "unavailable", "Released vectors are target-state aggregates, not individual-guide coefficients."),
        ("guide-level module activity, transfer gain, buffering, rerouting and disease convergence", "unavailable", "No guide-resolution expression vectors; target summaries cannot substitute."),
        ("donor effects and leave-one-donor validation", "unavailable", "Released donor robustness summaries do not identify donor-by-guide effects."),
        ("leave-one-guide stability and target heterogeneity", "underpowered", "All released comparisons use two guides; removing one leaves no guide-pair estimate."),
        ("negative-control trans genes and non-targeting trans vectors", "unavailable", "Non-targeting cis summaries are available, but joint trans-effect controls are not."),
        ("target-specific dose thresholds and cis mediation", "underpowered", "Two guides cannot identify a target dose-response shape or validate mediation assumptions."),
        ("matched guide-vector permutation and guide-state vector shuffle", "unavailable", "Without guide vectors, nulls for vector concordance cannot be reconstructed; predictor and paired-label nulls are reported separately."),
        ("gene-level correlation uncertainty", "unavailable", "Gene-level independence is implausible, upstream p-value assignment is questionable, and raw vectors are unavailable."),
        ("edge-specific guide concordance", "unavailable", "Concordance is released at target-state grain; molecular comparisons operate at that same grain."),
    ]
    write(pd.DataFrame(unavailable, columns=["estimand", "status", "reason"]), "availability_gates.csv")
    report = {"released_pairs": int(rows.released_pair.sum()), "complete_denominator": len(rows), "primary_denominator": int(rows.in_primary_study.sum()), "eligible_pairs": len(eligible), "eligible_targets": eligible.target.nunique(), "significant_union_eligible": int(rows.significant_union_eligible.sum()), "target_specific_underpowered": int(rows.released_pair.sum()), "invalid_gene_set_rows": int((rows.released_pair & ~rows.count_set_valid).sum()), "shared_ntc_eligible_pairs": int(eligible.shared_ntc_verified.sum()), "molecular_source_complete": evidence_available, "outcome_leakage_excluded": True, "supplied_correlation_pvalues_used": False, "config_sha256": manifest["config_sha256"], "amendment_sha256": sha256(ROOT / "analyses/guide_concordance/method_amendment.json"), "tests": tests[["analysis", "decision"]].to_dict("records")}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
