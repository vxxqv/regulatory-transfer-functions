"""Run frozen conditional decoders without refitting study outputs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from scipy.optimize import minimize

from freeze import HERE, ROOT, sha, verify

OUT = HERE / "results"
STATES = ["Rest", "Stim8hr", "Stim48hr"]
MODELS = ["nearest_centroid", "RNA_fingerprint_cosine", "frozen_module_decoder"]
BETAS = np.array([0., .1, .3, 1., 3., 10., 30.])
SEED = 20260913


def bh(values):
    p = np.asarray(values, float)
    out = np.full(len(p), np.nan)
    order = np.flatnonzero(np.isfinite(p))
    order = order[np.argsort(p[order])]
    out[order] = np.minimum(1., np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return out


def norm2(x):
    return np.asarray(x.multiply(x).sum(1)).ravel() if sparse.issparse(x) else np.square(x).sum(1)


def gram(a, b):
    result = a @ b.T
    return result.toarray() if sparse.issparse(result) else np.asarray(result)


def nuisance_basis(x, metadata):
    cols = ["cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "n_guides"]
    z = metadata[cols].to_numpy(float)
    med = np.nanmedian(z, axis=0)
    med[~np.isfinite(med)] = 0
    z = np.where(np.isfinite(z), z, med)
    scale = z.std(0)
    scale[scale == 0] = 1
    z = (z - z.mean(0)) / scale
    state = pd.get_dummies(metadata["culture_condition"], dtype=float).to_numpy()[:, 1:]
    z = np.column_stack([np.ones(len(z)), z, state])
    beta = np.linalg.pinv(z.T @ z, rcond=1e-10) @ np.asarray(x.T @ z).T
    _, singular, q = np.linalg.svd(beta, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0], 1e-12) * 1e-10))
    return q[:rank].T


def score_matrix(query, centroid, train, q, model):
    qp, cp, tp = np.asarray(query @ q), np.asarray(centroid @ q), np.asarray(train @ q)
    cross = gram(query, centroid).astype(float) - qp @ cp.T
    qn = np.maximum(0, norm2(query) - np.square(qp).sum(1))
    cn = np.maximum(0, norm2(centroid) - np.square(cp).sum(1))
    if model == "nearest_centroid":
        tn = np.maximum(0, norm2(train) - np.square(tp).sum(1))
        scale = float(np.median(tn[tn > 1e-12])) if np.any(tn > 1e-12) else 1.
        return -(np.maximum(0, qn[:, None] + cn[None, :] - 2 * cross)) / scale
    denom = np.sqrt(qn[:, None] * cn[None, :])
    return np.clip(np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1e-12), -1, 1)


def load_data(source):
    folds = pd.read_csv(source / "analyses/cell_systems_expansion/target_folds.tsv", sep="\t").sort_values("target").reset_index(drop=True)
    targets = folds.target.tolist()
    assert len(targets) == len(set(targets)) == 4399
    rows = pd.read_parquet(source / "data/interim/gwt_vectors/rows.parquet")
    primary = pd.read_parquet(source / "analyses/primary/results/transfer_phenotypes.parquet")
    genes = pd.read_parquet(source / "data/interim/gwt_vectors/genes.parquet")
    matrix = sparse.load_npz(source / "data/interim/gwt_vectors/normalized_significant_logfc.npz").astype(float).tocsr()
    assert matrix.shape == (len(rows), len(genes))
    rows["source_row"] = np.arange(len(rows))
    selected = rows[rows.target_contrast_gene_name.isin(targets)].copy()
    assert not selected.duplicated(["target_contrast_gene_name", "culture_condition"]).any()
    assert len(selected) == 4399 * 3
    cis_ids = set(selected.cis_feature_id.dropna().astype(str))
    excluded = genes.gene_name.isin(targets) | genes.feature_id.astype(str).isin(cis_ids)
    safe = matrix[:, ~excluded.to_numpy()].tocsr()
    mask = genes.copy()
    mask["excluded_target_feature"] = excluded
    mask["reason"] = np.where(excluded, "cohort_cis_feature_global_mask", "retained_non_target_gene")
    mask.to_csv(OUT / "feature_mask.tsv", sep="\t", index=False)
    extra = ["index", "cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "transfer_class", "transfer_residual", "transfer_z"]
    selected = selected.merge(primary[extra], on="index", how="left", validate="one_to_one")
    meta, xs = [], []
    for state in STATES:
        m = selected[selected.culture_condition == state].set_index("target_contrast_gene_name").loc[targets].reset_index()
        m["target"] = targets
        m["fold"] = folds.fold.to_numpy()
        x = safe[m.source_row.to_numpy()]
        m["full_detected_norm"] = np.sqrt(norm2(matrix[m.source_row.to_numpy()]))
        m["masked_detected_norm"] = np.sqrt(norm2(x))
        m["zero_full_profile"] = m.full_detected_norm == 0
        m["zero_masked_profile"] = m.masked_detected_norm == 0
        m["missing_profile"] = False
        m["independent_guide_profiles_available"] = False
        m["independent_donor_profiles_available"] = False
        meta.append(m)
        xs.append(x)
    loadings = pd.read_parquet(source / "analyses/vectors/results/reference_module_loadings.parquet")
    basis = loadings.pivot(index="feature_id", columns="module", values="loading").loc[genes.feature_id].to_numpy()[~excluded.to_numpy()]
    ms = [np.asarray(x @ basis) for x in xs]
    return folds, meta, xs, ms


def bootstrap_ci(values, seed):
    rng = np.random.default_rng(seed)
    estimates = np.array([np.mean(values[rng.integers(len(values), size=len(values))]) for _ in range(1000)])
    return np.quantile(estimates, [.025, .975]), estimates


def strata(metadata):
    e = pd.qcut(metadata.target_baseMean, 5, labels=False, duplicates="drop").fillna(-1).astype(int)
    d = pd.qcut(metadata.n_downstream, 5, labels=False, duplicates="drop").fillna(-1).astype(int)
    return e.astype(str) + "|" + d.astype(str)


def calibration_parameters(correct, confidence):
    if np.ptp(confidence) < 1e-10 or not 0 < correct.mean() < 1:
        return np.nan, np.nan
    x = np.log(np.clip(confidence, 1e-12, 1 - 1e-12) / np.clip(1 - confidence, 1e-12, 1))
    def objective(par):
        z = par[0] + par[1] * x
        return float(np.mean(np.logaddexp(0, z) - correct * z))
    result = minimize(objective, [0., 1.], method="BFGS", options={"maxiter": 200, "gtol": 1e-8})
    return (float(result.x[0]), float(result.x[1])) if np.isfinite(result.fun) else (np.nan, np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = json.loads((HERE / "freeze_manifest.json").read_text())
    records, expansion_hash = verify(args.source_root)
    assert records == manifest["artifacts"] and expansion_hash == manifest["expansion_freeze_sha256"]
    assert sha(HERE / "specification.json", True) == manifest["canonical_specification_sha256"]
    OUT.mkdir(exist_ok=True)
    folds, meta, xs, ms = load_data(args.source_root)
    n = len(folds)
    all_meta = pd.concat(meta, ignore_index=True)
    columns = ["target", "culture_condition", "source_row", "index", "target_contrast", "fold", "cis_magnitude", "target_baseMean", "n_cells_target", "n_guides", "n_downstream", "transfer_class", "transfer_residual", "full_detected_norm", "masked_detected_norm", "zero_full_profile", "zero_masked_profile", "missing_profile", "independent_guide_profiles_available", "independent_donor_profiles_available"]
    all_meta[columns].to_parquet(OUT / "profile_accounting.parquet", index=False)
    predictions, tuning, summaries, nulls, calibrations, coverage, cost, basis_audit = [], [], [], [], [], [], [], []
    independent_rows = []
    for outer, state in enumerate(STATES):
        others = [i for i in range(3) if i != outer]
        for model in MODELS:
            tick = time.perf_counter()
            source_x = ms if model == "frozen_module_decoder" else xs
            losses = []
            for train_i, val_i in [others, others[::-1]]:
                train = source_x[train_i]
                q = nuisance_basis(train, meta[train_i])
                scores = score_matrix(source_x[val_i], train, train, q, model)
                inner_losses = []
                for beta in BETAS:
                    loss = float(np.mean(logsumexp(beta * scores, axis=1) - beta * np.diag(scores)))
                    tuning.append({"heldout_state": state, "model": model, "inner_training_state": STATES[train_i], "inner_validation_state": STATES[val_i], "inverse_temperature": beta, "log_loss": loss, "targets": n})
                    inner_losses.append(loss)
                losses.append(inner_losses)
            beta = float(BETAS[np.argmin(np.mean(losses, axis=0))])
            a, b = others
            train = np.vstack([source_x[a], source_x[b]]) if model == "frozen_module_decoder" else sparse.vstack([source_x[a], source_x[b]], format="csr")
            train_meta = pd.concat([meta[a], meta[b]], ignore_index=True)
            q = nuisance_basis(train, train_meta)
            centroid = (source_x[a] + source_x[b]) / 2
            scores = score_matrix(source_x[outer], centroid, train, q, model)
            log_z = logsumexp(beta * scores, axis=1)
            log_probs = beta * scores - log_z[:, None]
            true_log = np.diag(log_probs).copy()
            information = np.log(n) + true_log
            probs = np.exp(log_probs)
            confidence = probs.max(1)
            ties = np.isclose(probs, confidence[:, None], rtol=0, atol=1e-14)
            correct = ties[np.arange(n), np.arange(n)] / ties.sum(1)
            pred = meta[outer][["target", "culture_condition", "transfer_class", "fold", "source_row", "full_detected_norm", "masked_detected_norm", "zero_full_profile", "zero_masked_profile"]].copy()
            pred["model"] = model
            pred["target_classes"] = n
            pred["inverse_temperature"] = beta
            pred["true_log_probability"] = true_log
            pred["true_probability"] = np.exp(true_log)
            pred["log_normalizer"] = log_z
            pred["true_score"] = np.diag(scores)
            pred["information_nats"] = information
            pred["top_confidence"] = confidence
            pred["fractional_top1_correct"] = correct
            pred["leave_one_target_mean_change"] = (information.mean() - information) / (n - 1)
            pred["partial_match_stratum"] = strata(meta[outer]).to_numpy()
            pred["inference_scope"] = "conditional_diagnostic_only"
            predictions.append(pred)
            groups = [g.index.to_numpy() for _, g in pred.groupby("partial_match_stratum", sort=True)]
            rng = np.random.default_rng(SEED + outer)
            distribution = []
            for rep in range(1000):
                order = np.arange(n)
                for group in groups:
                    order[group] = rng.permutation(group)
                value = float(np.log(n) + np.mean(log_probs[np.arange(n), order]))
                nulls.append({"state": state, "model": model, "replicate": rep, "information_nats": value, "null": "state_expression_response_degree_matched_fixed_decoder"})
                distribution.append(value)
            ci, _ = bootstrap_ci(information, SEED + outer)
            p = (1 + np.sum(np.asarray(distribution) >= information.mean())) / 1001
            summaries.append({"state": state, "model": model, "targets": n, "classes": n, "information_nats": information.mean(), "ci_low": ci[0], "ci_high": ci[1], "mean_log_loss": -true_log.mean(), "fractional_top1_accuracy": correct.mean(), "chance_accuracy": 1 / n, "inverse_temperature": beta, "p_value": p, "null_mean": np.mean(distribution), "permuteable_targets": sum(len(g) for g in groups if len(g) > 1), "scope": "conditional_not_confirmatory"})
            intercept, slope = calibration_parameters(correct, confidence)
            bins = np.minimum((confidence * 10).astype(int), 9)
            for j in range(10):
                selected = bins == j
                calibrations.append({"state": state, "model": model, "bin": j, "targets": int(selected.sum()), "mean_confidence": float(confidence[selected].mean()) if selected.any() else np.nan, "fractional_accuracy": float(correct[selected].mean()) if selected.any() else np.nan, "intercept": intercept, "slope": slope})
            true_p = np.exp(true_log)
            greater_mass = np.where(probs > true_p[:, None], probs, 0).sum(1)
            equal_mass = np.where(probs == true_p[:, None], probs, 0).sum(1)
            for alpha in [.5, .8, .95]:
                include = np.clip((alpha - greater_mass) / equal_mass, 0, 1)
                cci, _ = bootstrap_ci(include, SEED + outer)
                coverage.append({"state": state, "model": model, "nominal_mass": alpha, "observed_label_set_coverage": include.mean(), "ci_low": cci[0], "ci_high": cci[1], "targets": n, "tie_rule": "randomized_boundary_inclusion", "interval_type": "posterior_label_set_not_conformal"})
            np.savez_compressed(OUT / f"nuisance_{state}_{model}.npz", basis=q)
            basis_audit.append({"state": state, "model": model, "training_states": "|".join(STATES[i] for i in others), "training_profiles": 2 * n, "test_profiles": n, "shared_target_classes": n, "shared_profile_rows": 0, "nuisance_rank": q.shape[1], "orthogonality_max_error": np.max(np.abs(q.T @ q - np.eye(q.shape[1]))), "fixed_full_cohort_module_basis": model == "frozen_module_decoder"})
            for index in np.linspace(0, n - 1, 25, dtype=int):
                independent_rows.append({"state": state, "model": model, "target": folds.target.iloc[index], "target_position": int(index), "score": scores[index, index], "log_normalizer": log_z[index], "true_probability": true_p[index], "information_nats": information[index], "inverse_temperature": beta})
            cost.append({"state": state, "model": model, "wall_time_seconds": time.perf_counter() - tick})
            print(f"Completed {state} {model}", flush=True)
    result = pd.DataFrame(summaries)
    result["q_value"] = bh(result.p_value)
    result["conditional_support"] = (result.q_value < .05) & (result.ci_low > 0)
    result.to_csv(OUT / "decoder_summary.tsv", sep="\t", index=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_parquet(OUT / "target_predictions.parquet", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "nested_tuning.tsv", sep="\t", index=False)
    pd.DataFrame(nulls).to_parquet(OUT / "matched_nulls.parquet", index=False)
    pd.DataFrame(calibrations).to_csv(OUT / "calibration.tsv", sep="\t", index=False, na_rep="NA")
    pd.DataFrame(coverage).to_csv(OUT / "coverage.tsv", sep="\t", index=False)
    pd.DataFrame(basis_audit).to_csv(OUT / "split_audit.tsv", sep="\t", index=False)
    pd.DataFrame(independent_rows).to_csv(OUT / "independent_subset.tsv", sep="\t", index=False)
    pd.DataFrame(cost).to_csv(OUT / "computational_cost.tsv", sep="\t", index=False)
    prior = pd.DataFrame({"state": STATES, "model": "label_prior", "classes": n, "targets": n, "information_nats": 0., "log_loss": np.log(n), "fractional_top1_accuracy": 1 / n})
    prior.to_csv(OUT / "label_prior.tsv", sep="\t", index=False)
    by_class = []
    for (state, model, cls), frame in predictions.groupby(["culture_condition", "model", "transfer_class"]):
        ci, _ = bootstrap_ci(frame.information_nats.to_numpy(), SEED)
        by_class.append({"state": state, "model": model, "transfer_class": cls, "targets": len(frame), "global_classes": n, "mean_information_score_nats": frame.information_nats.mean(), "ci_low": ci[0], "ci_high": ci[1], "median_detected_norm": frame.full_detected_norm.median(), "zero_profiles": int(frame.zero_full_profile.sum()), "interpretation": "subgroup_performance_not_subgroup_mutual_information"})
    pd.DataFrame(by_class).to_csv(OUT / "transfer_class_summary.tsv", sep="\t", index=False)
    state_values = predictions[predictions.model == "RNA_fingerprint_cosine"].pivot(index="target", columns="culture_condition", values="information_nats").loc[folds.target]
    contrasts = []
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        delta = (state_values[STATES[b]] - state_values[STATES[a]]).to_numpy()
        ci, boot = bootstrap_ci(delta, SEED + a + b)
        p = (1 + np.sum(np.abs(boot - delta.mean()) >= abs(delta.mean()))) / 1001
        contrasts.append({"from_state": STATES[a], "to_state": STATES[b], "targets": n, "difference_nats": delta.mean(), "ci_low": ci[0], "ci_high": ci[1], "p_value": p, "test": "two_sided_centered_target_bootstrap", "scope": "conditional_diagnostic_only"})
    contrasts = pd.DataFrame(contrasts)
    contrasts["q_value"] = bh(contrasts.p_value)
    contrasts.to_csv(OUT / "state_contrasts.tsv", sep="\t", index=False)
    hypotheses = pd.read_csv(args.source_root / "analyses/cell_systems_expansion/hypotheses.tsv", sep="\t")
    hypotheses = hypotheses[hypotheses.block == "information_retention"].copy()
    hypotheses["decision"] = "unavailable"
    hypotheses["targets_accounted"] = n
    hypotheses["p_value"] = np.nan
    hypotheses["q_value"] = np.nan
    hypotheses["reason"] = [
        "Required generic templates and independent replicate profiles absent; only conditional sampled-state decoding estimated",
        "Fully adjusted replicated information unavailable; magnitude and subgroup diagnostic scores retained without H13 claim",
        "Fully adjusted information and reliability-resolved carrying-module comparisons unavailable",
        "Conditional state contrasts estimated, but full required adjustment and independent replication unavailable",
        "Reliable stable equivalence classes and eligible external redundancy outcomes unavailable",
        "Reliable information channels and eligible external emergence/synergy outcomes unavailable"
    ]
    hypotheses.to_csv(OUT / "hypothesis_decisions.tsv", sep="\t", index=False)
    gates = [
        ("complete_state_accounting", n, n, "available", "All frozen target classes retained"),
        ("class_known_state_validation", n, n, "available", "Two training-state summaries per class; one held-out-state summary"),
        ("independent_profiles_per_class", 0, n, "unavailable", "Three sampled-state summaries do not establish three independent guide/donor profiles"),
        ("complete_generic_template_removal", 0, n, "unavailable", "Library-size, stress and proliferation templates absent; unseen-state-specific mean removal not identifiable"),
        ("guide_or_donor_information_replication", 0, n, "unavailable", "Aggregate correlations are not response vectors"),
        ("external_profile_information", 0, n, "unavailable", "K562 concordance summaries do not define external decoder profiles"),
        ("regulator_family_outcome", 0, n, "unavailable", "No independent frozen family labels"),
        ("functional_module_outcome", 0, n, "unavailable", "Response-derived modules cannot be independent decoder labels"),
        ("ridge_multinomial", 0, n, "not_run_gate_failed", "Confirmatory model stopped before fitting because mandatory inputs are absent"),
        ("full_network_degree_matched_null", 0, n, "unavailable", "Assigned frozen inputs lack independent network-degree annotation"),
        ("equivalence_stability", 0, n, "unavailable", "Profile reliability, 20/40 loadings, negative-gene controls and full stability checks absent"),
        ("external_composition_association", 0, n, "unavailable", "Deferred until external and equivalence gates pass"),
        ("primary_information_pass", 0, 1, "not_passed", "Required adjustment, guide/donor replication and independent external association do not pass")
    ]
    pd.DataFrame(gates, columns=["gate", "numerator", "denominator", "status", "reason"]).to_csv(OUT / "availability_gates.tsv", sep="\t", index=False)
    try:
        import psutil
        memory = psutil.Process().memory_info()
        peak = getattr(memory, "peak_wset", memory.rss) / 1024 ** 2
    except ImportError:
        peak = None
    audit = {"targets": n, "state_profiles": len(all_meta), "missing_profiles": 0, "zero_full_profiles": int(all_meta.zero_full_profile.sum()), "zero_masked_profiles": int(all_meta.zero_masked_profile.sum()), "targets_zero_in_all_states": int(all_meta.groupby("target").zero_full_profile.all().sum()), "retained_non_target_genes": int(xs[0].shape[1]), "decoder_predictions": len(predictions), "primary_information_pass": False, "h12_h17_confirmatory_tests_run": 0, "conditional_tests": 9, "conditional_bh_family_size": 9, "full_degree_matched_null_available": False, "guide_donor_replication_available": False, "external_association_available": False, "analysis_wall_time_seconds": time.perf_counter() - start, "peak_memory_mb": peak, "output_storage_mb": sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) / 1024 ** 2}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
