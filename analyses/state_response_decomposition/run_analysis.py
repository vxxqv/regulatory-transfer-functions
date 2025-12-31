"""Decompose frozen signed responses and test state generalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses/state_response_decomposition/results"


def sha(path):
    if path.suffix == ".yaml":
        return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b > 0)


def norm2(x):
    return np.asarray(x.multiply(x).sum(axis=1)).ravel() if sparse.issparse(x) else np.square(x).sum(axis=1)


def dotrow(x, y):
    return np.asarray(x.multiply(y).sum(axis=1)).ravel() if sparse.issparse(x) else (x * y).sum(axis=1)


def decompose(xs):
    core = sum(xs) / len(xs)
    energies = np.stack([norm2(x) for x in xs], axis=1)
    core_energy = len(xs) * norm2(core)
    residual = np.stack([norm2(x - core) for x in xs], axis=1)
    return core, energies, core_energy, residual


def bh(p):
    values = np.asarray(p, float)
    out = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    out[order] = np.minimum(1, np.minimum.accumulate((values[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return out


def bootstrap(values, count, seed, statistic=np.mean):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(count)]
    return tuple(np.quantile(estimates, [.025, .975]))


def permute_indices(groups, rng):
    indices = np.arange(sum(map(len, groups)))
    for group in groups:
        indices[group] = rng.permutation(group)
    return indices


def correlation_test(frame, outcome, cfg, offset):
    table = frame[["target_contrast", "core_fraction", outcome, "stratum"]].dropna().reset_index(drop=True)
    record = {"outcome": outcome, "targets": len(table), "eligible_denominator": len(frame), "rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "status": "underpowered"}
    if len(table) < cfg["minimum_correlation_targets"]:
        return record, pd.DataFrame()
    x, y = table["core_fraction"].to_numpy(), table[outcome].to_numpy()
    rng = np.random.default_rng(cfg["seed"] + offset)
    estimate = spearmanr(x, y).statistic
    boot = []
    for _ in range(cfg["bootstrap_replicates"]):
        index = rng.integers(0, len(x), len(x))
        boot.append(spearmanr(x[index], y[index]).statistic)
    groups = list(table.groupby("stratum").indices.values())
    ranked_x, ranked_y = rankdata(x), rankdata(y)
    ranked_x = (ranked_x - ranked_x.mean()) / np.linalg.norm(ranked_x - ranked_x.mean())
    ranked_y = (ranked_y - ranked_y.mean()) / np.linalg.norm(ranked_y - ranked_y.mean())
    null = np.array([ranked_x[permute_indices(groups, rng)] @ ranked_y for _ in range(cfg["permutations"])])
    centered = null - np.median(null)
    p = (1 + np.sum(np.abs(centered) >= abs(estimate - np.median(null)))) / (len(null) + 1)
    low, high = np.quantile(boot, [.025, .975])
    center = float(np.median(null))
    record.update(rho=float(estimate), ci_low=float(low), ci_high=float(high), p_value=float(p), null_median=center, rho_excess=float(estimate - center), excess_ci_low=float(low - center), excess_ci_high=float(high - center), status="evaluated")
    return record, pd.DataFrame({"outcome": outcome, "replicate": np.arange(len(null)), "rho": null})


def prediction_test(table, outcome, cfg):
    cols = cfg["prediction_covariates"]
    table = table.dropna(subset=[outcome, "core_fraction"]).copy()
    if len(table) < cfg["minimum_prediction_targets"]:
        return {"outcome": outcome, "targets": len(table), "status": "underpowered"}, pd.DataFrame()
    base = np.log1p(table[cols].clip(lower=0))
    x = np.column_stack([base, table["core_fraction"]])
    y = table[outcome].to_numpy()
    predictions = np.full((len(y), 2), np.nan)
    for fold in range(cfg["folds"]):
        test = table["fold"].to_numpy() == fold
        for model_index, features in enumerate([x[:, :-1], x]):
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=cfg["ridge_alpha"]))
            model.fit(features[~test], y[~test])
            predictions[test, model_index] = model.predict(features[test])
    loss = np.square(y[:, None] - predictions)
    delta = loss[:, 0] - loss[:, 1]
    low, high = bootstrap(delta, cfg["bootstrap_replicates"], cfg["seed"] + 800)
    denom = np.mean((y - y.mean()) ** 2)
    rng = np.random.default_rng(cfg["seed"] + 800)
    draws = np.array([np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(cfg["bootstrap_replicates"])])
    p = (1 + np.sum(np.abs(draws - delta.mean()) >= abs(delta.mean()))) / (len(draws) + 1)
    result = {"outcome": outcome, "targets": len(table), "covariate_r2": 1 - loss[:, 0].mean() / denom, "with_core_r2": 1 - loss[:, 1].mean() / denom, "delta_mse": delta.mean(), "ci_low": low, "ci_high": high, "delta_r2": delta.mean() / denom, "delta_r2_low": low / denom, "delta_r2_high": high / denom, "p_value": p, "status": "evaluated"}
    record = table[["target_contrast", "target_gene", "fold", "core_fraction"]].copy()
    record["outcome"] = outcome
    record["observed"] = y
    record["covariate_prediction"] = predictions[:, 0]
    record["core_prediction"] = predictions[:, 1]
    record["paired_loss_improvement"] = delta
    return result, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    source = args.source_root
    cfg_path = ROOT / "config/state_response_decomposition.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())["state_response_decomposition"]
    freeze_path = ROOT / "analyses/state_response_decomposition/freeze_manifest.json"
    manifest = json.loads(freeze_path.read_text())
    assert sha(cfg_path) == manifest["config_sha256"]
    for name, expected in manifest["sources"].items():
        assert sha(source / name) == expected, name
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(ROOT / "analyses/state_response_decomposition/frozen_cohort.tsv", sep="\t")
    cohort_text = (ROOT / "analyses/state_response_decomposition/frozen_cohort.tsv").read_text(encoding="utf-8")
    assert hashlib.sha256(cohort_text.encode()).hexdigest() == manifest["canonical_cohort_sha256"]
    rows = pd.read_parquet(source / "data/interim/gwt_vectors/rows.parquet").reset_index(drop=True)
    matrix = sparse.load_npz(source / "data/interim/gwt_vectors/normalized_significant_logfc.npz").astype(float).tocsr()
    targets, states = cohort["target_contrast"].tolist(), cfg["states"]
    indexed = rows.reset_index(names="matrix_row").set_index(["target_contrast", "culture_condition"])
    state_rows = [indexed.loc[[(target, state) for target in targets]].reset_index() for state in states]
    xs = [matrix[frame["matrix_row"].to_numpy()] for frame in state_rows]
    core, energy, core_energy, residual_energy = decompose(xs)
    total = energy.sum(axis=1)
    fraction = ratio(core_energy, total)
    np.testing.assert_allclose(total, core_energy + residual_energy.sum(axis=1), atol=1e-7)
    consensus = xs[0].minimum(xs[1]).minimum(xs[2]).maximum(0) + xs[0].maximum(xs[1]).maximum(xs[2]).minimum(0)
    target = cohort.copy()
    target["target_gene"] = state_rows[0]["target_contrast_gene_name"].to_numpy()
    target["total_energy"] = total
    target["core_energy"] = core_energy
    target["residual_energy"] = residual_energy.sum(axis=1)
    target["core_fraction"] = fraction
    target["residual_fraction"] = 1 - fraction
    target["signed_consensus_energy_fraction"] = ratio(3 * norm2(consensus), total)
    target["consensus_genes"] = np.asarray((consensus != 0).sum(axis=1)).ravel()
    union = sum((x != 0).astype(int) for x in xs)
    target["response_union_genes"] = np.asarray((union != 0).sum(axis=1)).ravel()
    target["signed_consensus_gene_fraction"] = ratio(target["consensus_genes"].to_numpy(), target["response_union_genes"].to_numpy())
    target["status"] = np.where(total > 0, "evaluated", "no_significant_response")
    covariates = {"mean_target_expression": "target_baseMean", "mean_cis_magnitude": "ontarget_effect_size", "mean_n_downstream": "n_downstream", "mean_target_cells": "n_cells_target", "mean_guides": "n_guides", "mean_guide_concordance": "guide_correlation_all", "mean_donor_concordance": "donor_correlation_all_mean"}
    for new, old in covariates.items():
        values = np.stack([f[old].to_numpy(float) for f in state_rows], axis=1)
        target[new] = pd.DataFrame(np.abs(values) if new == "mean_cis_magnitude" else values).mean(axis=1).to_numpy()
    target["stratum"] = pd.qcut(target["mean_n_downstream"].rank(method="first"), cfg["match_quantiles"], labels=False).astype(str) + ":" + pd.qcut(target["mean_cis_magnitude"].rank(method="first"), cfg["match_quantiles"], labels=False).astype(str)
    groups = list(target.groupby("stratum").indices.values())
    state_records, lso_records = [], []
    for i, state in enumerate(states):
        row = state_rows[i][["target_contrast", "target_contrast_gene_name", "culture_condition", "matrix_row", "n_downstream", "n_guides", "guide_correlation_all", "donor_correlation_all_mean"]].copy()
        row["fold"] = cohort["fold"].to_numpy()
        row["response_energy"] = energy[:, i]
        row["core_norm_squared"] = core_energy / 3
        row["residual_energy"] = residual_energy[:, i]
        row["core_residual_cross_term"] = 2 * dotrow(core, xs[i] - core)
        row["status"] = np.where(energy[:, i] > 0, "evaluated", "no_significant_response")
        state_records.append(row)
        other = [xs[j] for j in range(3) if j != i]
        prediction = sum(other) / 2
        pred_energy = norm2(prediction)
        cosine = ratio(dotrow(prediction, xs[i]), np.sqrt(pred_energy * energy[:, i]))
        pair_total = sum(norm2(x) for x in other)
        lso = pd.DataFrame({"target_contrast": targets, "heldout_state": state, "prediction_energy": pred_energy, "heldout_energy": energy[:, i], "cosine": cosine, "relative_squared_error": ratio(norm2(xs[i] - prediction), energy[:, i]), "two_state_core_fraction": ratio(2 * pred_energy, pair_total)})
        lso["status"] = np.where(np.isfinite(cosine), "evaluated", "no_evaluable_prediction_or_response")
        lso_records.append(lso)
    pd.concat(state_records).to_parquet(OUT / "target_state_decomposition.parquet", index=False)
    lso = pd.concat(lso_records, ignore_index=True)
    lso.to_parquet(OUT / "leave_one_state.parquet", index=False)
    print("Primary energy decomposition complete", flush=True)

    pair = pd.read_parquet(source / "analyses/vectors/results/context_rerouting_pairs.parquet")
    target["rerouting"] = target["target_contrast"].map(pair.groupby("target_contrast")["module_js_divergence"].agg(lambda x: x.mean() if x.notna().sum() == 3 else np.nan))
    tf = pd.read_parquet(source / "analyses/context_dynamics/results/tf_activity_association.parquet")
    target["tf_motif_range"] = target["target_contrast"].map(tf.set_index("target_contrast")["maximum_tf_activity_range"])
    k562 = pd.read_parquet(source / "analyses/replication/results/k562_replication_rows.parquet")
    target["k562_concordance"] = target["target_gene"].map(k562.groupby("target_contrast_gene_name")["logfc_pearson_r"].mean())
    rpe = pd.read_parquet(source / "analyses/external_benchmark/results/all_eligible_target_predictions.parquet")
    rpe["negative_transfer"] = np.square(rpe["log1p_differentially_expressed_genes"] - rpe["frozen_cd4_transfer"]) - np.square(rpe["log1p_differentially_expressed_genes"] - rpe["without_vector"])
    target["rpe1_negative_transfer"] = target["target_gene"].map(rpe.set_index("target_gene")["negative_transfer"])
    phenotype = pd.read_parquet(source / "analyses/primary/results/transfer_phenotypes.parquet")
    target["mean_transfer_gain"] = target["target_contrast"].map(phenotype.groupby("target_contrast")["transfer_residual"].mean())
    for name in cfg["association_outcomes"]:
        target[name + "_status"] = np.where(target[name].notna(), "available_summary", "unavailable")
    finite = np.isfinite(fraction)
    target["leave_one_target_mean_change"] = np.where(finite, (np.nansum(fraction) - fraction) / (finite.sum() - 1) - np.nanmean(fraction), np.nan)
    target.to_parquet(OUT / "target_decomposition.parquet", index=False)

    module = pd.read_parquet(source / "analyses/vectors/results/contextual_transfer_tensor.parquet")
    module = module[module["target_contrast"].isin(targets)].pivot(index="target_contrast", columns=["condition", "module"], values="score").reindex(targets)
    module_xs = [module[state].to_numpy(float) for state in states]
    _, me, mc, mr = decompose(module_xs)
    pd.DataFrame({"target_contrast": targets, "module_core_fraction": ratio(mc, me.sum(axis=1)), "module_residual_energy": mr.sum(axis=1), "status": "descriptive_reference_basis_fitted_previously"}).to_parquet(OUT / "module_coherence.parquet", index=False)

    rng = np.random.default_rng(cfg["seed"] + 100)
    null_records = []
    for rep in range(cfg["permutations"]):
        permuted = [xs[0], xs[1][permute_indices(groups, rng)], xs[2][permute_indices(groups, rng)]]
        e = sum(norm2(x) for x in permuted)
        common = sum(permuted) / 3
        null_records.append({"replicate": rep, "mean_core_fraction": np.nanmean(ratio(3 * norm2(common), e)), "median_core_fraction": np.nanmedian(ratio(3 * norm2(common), e))})
    null = pd.DataFrame(null_records)
    null.to_parquet(OUT / "matched_target_null.parquet", index=False)
    summary = []
    for metric in ["core_fraction", "signed_consensus_energy_fraction", "signed_consensus_gene_fraction"]:
        values = target[metric].dropna().to_numpy()
        low, high = bootstrap(values, cfg["bootstrap_replicates"], cfg["seed"] + 200)
        summary.append({"metric": metric, "targets": len(values), "estimate": np.mean(values), "ci_low": low, "ci_high": high})
    for state in states:
        values = lso.loc[lso["heldout_state"] == state, "cosine"].dropna().to_numpy()
        low, high = bootstrap(values, cfg["bootstrap_replicates"], cfg["seed"] + 201)
        summary.append({"metric": "heldout_cosine_" + state, "targets": len(values), "estimate": np.mean(values), "ci_low": low, "ci_high": high})
    pd.DataFrame(summary).to_csv(OUT / "summary_estimates.tsv", sep="\t", index=False)
    print("Matched target null complete", flush=True)

    sensitivity, model_fits = [], []
    row_indices = np.stack([x["matrix_row"].to_numpy() for x in state_rows], axis=1)
    for fold in range(cfg["folds"]):
        train = cohort["fold"].to_numpy() != fold
        heldout = ~train
        svd = TruncatedSVD(n_components=max(cfg["sensitivity_components"]), n_iter=5, random_state=cfg["seed"] + fold)
        svd.fit(matrix[row_indices[train].ravel()])
        projected = [svd.transform(x[heldout]) for x in xs]
        model_fits.append({"fold": fold, "training_targets": int(train.sum()), "heldout_targets": int(heldout.sum()), "components": int(svd.n_components), "training_target_sha256": hashlib.sha256("\n".join(cohort.loc[train, "target_contrast"]).encode()).hexdigest(), "heldout_target_sha256": hashlib.sha256("\n".join(cohort.loc[heldout, "target_contrast"]).encode()).hexdigest(), "component_sha256": hashlib.sha256(svd.components_.tobytes()).hexdigest()})
        for resolution in cfg["sensitivity_components"]:
            _, pe, pc, pr = decompose([p[:, :resolution] for p in projected])
            part = pd.DataFrame({"target_contrast": cohort.loc[heldout, "target_contrast"].to_numpy(), "fold": fold, "components": resolution, "core_fraction": ratio(pc, pe.sum(axis=1)), "captured_energy_fraction": ratio(pe.sum(axis=1), total[heldout]), "core_energy": pc, "residual_energy": pr.sum(axis=1)})
            sensitivity.append(part)
    pd.DataFrame(model_fits).to_csv(OUT / "component_fit_audit.tsv", sep="\t", index=False)
    pd.concat(sensitivity).to_parquet(OUT / "component_sensitivity.parquet", index=False)
    print("Target-held-out component sensitivity complete", flush=True)

    correlations, correlation_nulls = [], []
    for i, outcome in enumerate(cfg["association_outcomes"]):
        result, draws = correlation_test(target, outcome, cfg, 400 + i)
        correlations.append(result)
        correlation_nulls.append(draws)
    associations = pd.DataFrame(correlations)
    associations["q_value"] = bh(associations["p_value"])
    for i, row in associations.iterrows():
        if row["status"] == "evaluated":
            associations.loc[i, "status"] = "above_matched_null" if row["q_value"] < cfg["fdr_alpha"] and row["excess_ci_low"] > 0 else "below_matched_null" if row["q_value"] < cfg["fdr_alpha"] and row["excess_ci_high"] < 0 else "unresolved"
    associations.to_csv(OUT / "associations.tsv", sep="\t", index=False)
    pd.concat(correlation_nulls, ignore_index=True).to_parquet(OUT / "association_nulls.parquet", index=False)
    pred_results, pred_rows = [], []
    for outcome in cfg["prediction_outcomes"]:
        result, details = prediction_test(target, outcome, cfg)
        pred_results.append(result)
        pred_rows.append(details)
    comparisons = pd.DataFrame(pred_results)
    comparisons["q_value"] = bh(comparisons["p_value"])
    comparisons["status"] = np.where((comparisons["q_value"] < cfg["fdr_alpha"]) & (comparisons["ci_low"] > 0), "supported", np.where((comparisons["q_value"] < cfg["fdr_alpha"]) & (comparisons["ci_high"] < 0), "contradictory", "unresolved"))
    comparisons.to_csv(OUT / "heldout_prediction_comparison.tsv", sep="\t", index=False)
    pd.concat(pred_rows).to_parquet(OUT / "heldout_prediction_rows.parquet", index=False)

    independent = pd.read_csv(source / "analyses/context_dynamics/results/independent_state_validation.csv")
    gates = [
        {"branch": "independent_Tcell_state_replication", "targets": len(independent), "minimum_targets": cfg["minimum_external_tcell_targets"], "status": "underpowered", "reason": "Existing eight-target Th1/Th2 summary is previously inspected; no new independent signed T-cell perturbation matrix is supplied."},
        {"branch": "K562_core_vs_residual_vectors", "targets": int(target["k562_concordance"].notna().sum()), "status": "unavailable", "reason": "Frozen inputs contain whole-response correlation summaries, not gene-aligned external signed vectors."},
        {"branch": "RPE1_core_vs_residual_vectors", "targets": int(target["rpe1_negative_transfer"].notna().sum()), "status": "unavailable", "reason": "Frozen inputs contain inspected model predictions, not gene-aligned external signed vectors."},
        {"branch": "independent_chromatin_enhancer_cascade_validation", "targets": 0, "status": "unavailable", "reason": "No frozen target-state matched occupancy, accessibility, enhancer or cascade matrix is provided to this block. Motif range is response-derived."},
        {"branch": "component_specific_guide_and_donor_replication", "targets": 0, "status": "unavailable", "reason": "Guide and donor correlation summaries cannot identify component-specific signed response vectors."},
    ]
    pd.DataFrame(gates).to_csv(OUT / "availability_gates.tsv", sep="\t", index=False)
    audit = {"targets": len(target), "target_state_rows": len(target) * 3, "nonzero_targets": int(finite.sum()), "zero_energy_targets": int((~finite).sum()), "signed_consensus_targets": int((target["consensus_genes"] > 0).sum()), "state_nonzero": {state: int((energy[:, i] > 0).sum()) for i, state in enumerate(states)}, "maximum_energy_identity_error": float(np.max(np.abs(total - core_energy - residual_energy.sum(axis=1)))), "mean_core_fraction": float(np.nanmean(fraction)), "matched_null_mean": float(null["mean_core_fraction"].mean()), "matched_null_p": float((1 + (null["mean_core_fraction"] >= np.nanmean(fraction)).sum()) / (len(null) + 1)), "maximum_absolute_leave_one_target_change": float(target["leave_one_target_mean_change"].abs().max()), "source_sha256_verified": True, "freeze_manifest_sha256": sha(freeze_path), "elapsed_seconds": time.perf_counter() - start, "interpretation": "Mean component is a descriptive energy projection. Biological conservation also requires signed consensus and held-out or independent reproducibility."}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
