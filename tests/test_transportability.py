import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/transportability/results"


def test_frozen_sources_match_across_portable_line_endings():
    manifest = json.loads((ROOT / "analyses/transportability/source_manifest.json").read_text())
    assert manifest["hash_comparison"] == "raw_or_lf_or_crlf_for_text_binary_exact"
    for record in manifest["sources"]:
        path = ROOT / record["path"]
        raw = path.read_bytes()
        candidates = {hashlib.sha256(raw).hexdigest(): len(raw)}
        if path.suffix.lower() in {".json", ".md", ".tsv", ".txt", ".yaml", ".yml"}:
            lf = raw.replace(b"\r\n", b"\n")
            crlf = lf.replace(b"\n", b"\r\n")
            candidates[hashlib.sha256(lf).hexdigest()] = len(lf)
            candidates[hashlib.sha256(crlf).hexdigest()] = len(crlf)
        assert record["checkout_sha256"] in candidates
        assert candidates[record["checkout_sha256"]] == record["checkout_bytes"]
        assert record["frozen_sha256"] in candidates
        if record["matched_expansion_freeze"] is None:
            assert record["verification_status"] == "frozen_at_transport_start"
        else:
            assert record["matched_expansion_freeze"]
            assert record["verification_status"] == "matched_expansion_freeze"
        assert record["frozen_bytes"] > 0


def test_denominators_gates_and_complete_reporting():
    audit = json.loads((RESULTS / "audit.json").read_text())
    k562 = pd.read_parquet(RESULTS / "k562_portability_rows.parquet")
    rpe1 = pd.read_parquet(RESULTS / "rpe1_applicability_rows.parquet")
    decisions = pd.read_csv(RESULTS / "hypothesis_decisions.csv")
    assert len(k562) == audit["k562_rows"] == 5640
    assert k562.target_gene.nunique() == audit["k562_targets"] == 1880
    assert k562.groupby("target_gene").culture_condition.nunique().eq(3).all()
    assert set(k562.culture_condition) == {"Rest", "Stim8hr", "Stim48hr"}
    assert rpe1.groupby("prediction_model").size().eq(490).all()
    assert rpe1.groupby("prediction_model").target_gene.nunique().eq(490).all()
    assert audit["complete_state_targets"] == 4399
    assert set(decisions.hypothesis) == {"H7", "H8", "H9", "H10", "H11"}
    assert decisions.decision.eq("unavailable").all()
    assert audit["compatible_external_systems"] == 2 < audit["minimum_systems_required"] == 3
    assert not audit["confirmatory_holdout_available"]
    assert audit["rpe1_interpretation"] == "developmental only"


def test_bounded_reliability_without_point_correction():
    bounds = pd.read_csv(RESULTS / "reliability_bounds.csv")
    draws = pd.read_parquet(RESULTS / "reliability_bootstrap.parquet")
    assert not any("point" in column and "corrected" in column for column in bounds.columns)
    assert bounds.corrected_identified_lower.between(-1, 1).all()
    assert bounds.corrected_identified_upper.between(-1, 1).all()
    assert (bounds.corrected_identified_lower <= bounds.observed_median).all()
    assert (bounds.observed_median <= bounds.corrected_identified_upper).all()
    expected_upper = np.minimum(1, bounds.observed_median / 0.10)
    np.testing.assert_allclose(bounds.corrected_identified_upper, expected_upper)
    assert draws.groupby("condition").size().eq(1000).all()
    assert draws.corrected_lower.between(-1, 1).all()
    assert draws.corrected_upper.between(-1, 1).all()
    spec = json.loads((ROOT / "analyses/transportability/implementation_spec.json").read_text())
    assert spec["reliability"]["prohibit_naive_point_division"]
    assert spec["reliability"]["method"].startswith("partial identification")


def test_k562_source_correspondence_and_diagnostics():
    original = pd.read_parquet(ROOT / "analyses/replication/results/k562_replication_rows.parquet")
    rows = pd.read_parquet(RESULTS / "k562_portability_rows.parquet")
    bounds = pd.read_csv(RESULTS / "reliability_bounds.csv").set_index("condition")
    np.testing.assert_allclose(original.logfc_pearson_r, rows.logfc_pearson_r)
    assert original.target_contrast_gene_name.equals(rows.target_gene)
    assert original.condition.equals(rows.culture_condition)
    np.testing.assert_allclose(original.logfc_pearson_r.median(), bounds.loc["All", "observed_median"])
    for condition, group in original.groupby("condition"):
        np.testing.assert_allclose(group.logfc_pearson_r.median(), bounds.loc[condition, "observed_median"])
    diagnostics = pd.read_csv(RESULTS / "k562_diagnostics.csv")
    assert set(diagnostics.analysis) == {"H8_observed_K562", "H11_observed_K562"}
    assert diagnostics.bootstrap_replicates.eq(1000).all()
    assert diagnostics.permutation_replicates.eq(1000).all()
    assert diagnostics.q_value.between(0, 1).all()
    assert (diagnostics.ci_low <= diagnostics.estimate).all()
    assert (diagnostics.estimate <= diagnostics.ci_high).all()
    covariates = ["source_target_expression", "cis_magnitude", "source_response_degree", "source_guide_concordance", "curated_network_degree"]
    for name, predictor, frame in [
        ("H8_observed_K562", "strongly_rerouted", rows.loc[rows.rerouting_group.isin(["invariant", "strongly_rerouted"])].assign(strongly_rerouted=lambda data: data.rerouting_group.eq("strongly_rerouted").astype(float))),
        ("H11_observed_K562", "molecular_support_fraction", rows.dropna(subset=["molecular_support_fraction"])),
    ]:
        frame = frame.dropna(subset=[predictor, "logfc_pearson_r"]).copy()
        numeric = frame[covariates].astype(float)
        missing_columns = []
        for column in covariates:
            missing = numeric[column].isna().astype(float)
            median = numeric[column].median()
            numeric[column] = numeric[column].fillna(0 if pd.isna(median) else median)
            if missing.any():
                frame[f"{column}_missing"] = missing
                missing_columns.append(f"{column}_missing")
        numeric = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
        state = pd.get_dummies(frame.culture_condition, dtype=float, drop_first=True)
        design = np.column_stack([np.ones(len(frame)), numeric, state, frame[missing_columns] if missing_columns else np.empty((len(frame), 0))])
        x = frame[predictor].to_numpy(float)
        y = frame.logfc_pearson_r.to_numpy(float)
        rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        estimate = rx @ ry / (rx @ rx)
        reported = diagnostics.loc[diagnostics.analysis.eq(name), "estimate"].iloc[0]
        np.testing.assert_allclose(estimate, reported, atol=1e-12)


def test_rpe1_oof_errors_feature_freeze_and_calibration():
    source = pd.read_parquet(ROOT / "analyses/external_benchmark/results/all_eligible_target_predictions.parquet")
    rows = pd.read_parquet(RESULTS / "rpe1_applicability_rows.parquet")
    audit = json.loads((RESULTS / "audit.json").read_text())
    features = audit["rpe1_applicability_features"]
    expected = {
        "baseline_target_expression",
        "expression_range_compatibility",
        "guide_efficacy",
        "source_guide_concordance",
        "source_response_reliability",
        "module_availability",
        "target_degree",
    }
    assert set(features) == expected
    assert not ({"log1p_differentially_expressed_genes", "absolute_error", "frozen_cd4_transfer", "without_vector", "source_occupancy_proxy"} & set(features))
    for model, group in rows.groupby("prediction_model"):
        group = group.sort_values("target_gene").reset_index(drop=True)
        reference = source.sort_values("target_gene").reset_index(drop=True)
        assert group.target_gene.equals(reference.target_gene)
        np.testing.assert_allclose(group[model], reference[model])
        np.testing.assert_allclose(group.absolute_error, np.abs(group.log1p_differentially_expressed_genes - group[model]))
        assert group.target_gene.is_unique
        assert group.fold.between(1, 10).all()
        assert group.predicted_risk.notna().all()
        for level in [50, 80, 95]:
            assert (group[f"risk_lower_{level}"] <= group[f"risk_upper_{level}"]).all()
    calibration = pd.read_csv(RESULTS / "applicability_calibration.csv")
    assert calibration.targets.eq(490).all()
    assert calibration[["coverage_50", "coverage_80", "coverage_95"]].apply(lambda column: column.between(0, 1).all()).all()
    decisions = pd.read_csv(RESULTS / "hypothesis_decisions.csv").set_index("hypothesis")
    full = rows.loc[rows.prediction_model.eq("frozen_cd4_transfer")]
    rho = full[["predicted_risk", "absolute_error"]].corr(method="spearman").iloc[0, 1]
    np.testing.assert_allclose(rho, decisions.loc["H9", "estimate"])
    comparison = pd.read_csv(RESULTS / "primary_coverage_comparisons.csv")
    h10 = comparison.loc[comparison.prediction_model.eq("frozen_cd4_transfer") & comparison.comparator.eq("random_abstention")].iloc[0]
    np.testing.assert_allclose(h10.risk_improvement, decisions.loc["H10", "estimate"])


def test_applicability_features_reconstruct_without_rpe1_perturbation_outcomes():
    rows = pd.read_parquet(RESULTS / "rpe1_applicability_rows.parquet")
    actual = rows.loc[rows.prediction_model.eq("frozen_cd4_transfer")].sort_values("target_gene").reset_index(drop=True)
    external = pd.read_parquet(ROOT / "analyses/external_benchmark/rpe1_prepared/eligible_targets.parquet")
    primary = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet").rename(columns={"target_contrast_gene_name": "target_gene"})
    primary = primary.groupby("target_gene", as_index=False).agg(
        source_target_expression=("log1p_target_baseMean", "median"),
        source_donor_concordance=("donor_correlation_all_mean", "median"),
        source_response_degree=("n_downstream", "mean"),
    )
    guides = pd.read_parquet(ROOT / "analyses/guide_concordance/results/all_target_states.parquet")
    guides = guides.loc[guides.in_primary_study].copy()
    guides["source_guide_efficacy_state"] = guides[["knockdown_1", "knockdown_2"]].abs().median(axis=1, skipna=True)
    guides = guides.groupby("target", as_index=False).agg(
        source_guide_concordance=("correlation_all", "median"),
        guide_efficacy=("source_guide_efficacy_state", "median"),
    ).rename(columns={"target": "target_gene"})
    modules = pd.read_parquet(ROOT / "analyses/vectors/results/contextual_transfer_tensor.parquet")
    modules = modules.groupby("target_gene", as_index=False).agg(module_availability=("score", lambda values: np.isfinite(values).mean()))
    expected = external[["target_gene", "control_target_expression"]].merge(primary, on="target_gene", how="left", validate="one_to_one")
    expected = expected.merge(guides, on="target_gene", how="left", validate="one_to_one")
    expected = expected.merge(modules, on="target_gene", how="left", validate="one_to_one").sort_values("target_gene").reset_index(drop=True)
    expected["baseline_target_expression"] = expected.control_target_expression
    expected["expression_range_compatibility"] = 1 - (
        expected.source_target_expression.rank(pct=True) - expected.control_target_expression.rank(pct=True)
    ).abs()
    expected["source_response_reliability"] = expected[["source_guide_concordance", "source_donor_concordance"]].clip(0, 1).mean(axis=1, skipna=True)
    expected["target_degree"] = np.log1p(expected.source_response_degree)
    assert actual.target_gene.equals(expected.target_gene)
    for column in [
        "baseline_target_expression",
        "expression_range_compatibility",
        "guide_efficacy",
        "source_guide_concordance",
        "source_response_reliability",
        "module_availability",
        "target_degree",
    ]:
        np.testing.assert_allclose(actual[column], expected[column], equal_nan=True)
    spec = json.loads((ROOT / "analyses/transportability/implementation_spec.json").read_text())
    prohibited = set(spec["applicability_feature_provenance"]["RPE1_perturbation_outcomes_excluded"])
    audit = json.loads((RESULTS / "audit.json").read_text())
    assert prohibited.isdisjoint(audit["rpe1_applicability_features"])


def test_risk_coverage_nulls_and_uncertainty_are_complete():
    risk = pd.read_csv(RESULTS / "risk_coverage.csv")
    comparisons = pd.read_csv(RESULTS / "primary_coverage_comparisons.csv")
    random = pd.read_parquet(RESULTS / "random_abstention_null.parquet")
    bootstrap = pd.read_parquet(RESULTS / "risk_coverage_bootstrap.parquet")
    assert len(risk) == 2 * 9 * 7
    assert set(risk.selector) == {"predict_all", "random_abstention", "response_magnitude", "degree", "uncertainty_only", "reliability_only", "applicability"}
    assert set(risk.coverage.round(1)) == set(np.arange(0.2, 1.01, 0.1).round(1))
    assert (risk.ci_low <= risk.risk).all() and (risk.risk <= risk.ci_high).all()
    assert random.groupby(["prediction_model", "coverage"]).size().eq(1000).all()
    assert bootstrap.groupby(["prediction_model", "selector", "coverage"]).size().eq(1000).all()
    assert comparisons.groupby("prediction_model").size().eq(5).all()
    assert comparisons.coverage.eq(0.8).all()
    assert comparisons.q_value.between(0, 1).all()
    assert comparisons.status.eq("developmental_only").all()
    costs = pd.read_csv(RESULTS / "computational_cost.csv")
    assert set(costs.block) == {"K562 partial identification and diagnostics", "RPE1 developmental risk coverage"}
    assert costs[["wall_time_seconds", "peak_memory_mb", "output_storage_mb"]].gt(0).all().all()


def test_no_long_dash_characters():
    paths = list((ROOT / "analyses/transportability").rglob("*.py"))
    paths += list((ROOT / "analyses/transportability").rglob("*.md"))
    paths += list((ROOT / "analyses/transportability").rglob("*.json"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "\u2013" not in text and "\u2014" not in text


def test_s29_exports_and_exact_figure_sources():
    figure = ROOT / "figures/supplement/S29"
    source_pairs = {
        "A_reliability_bounds.csv": "reliability_bounds.csv",
        "B_k562_diagnostics.csv": "k562_diagnostics.csv",
        "C_risk_coverage.csv": "risk_coverage.csv",
        "D_calibration.csv": "applicability_calibration.csv",
        "D_hypothesis_decisions.csv": "hypothesis_decisions.csv",
    }
    for figure_name, result_name in source_pairs.items():
        assert (figure / "figure_data" / figure_name).read_bytes() == (RESULTS / result_name).read_bytes()
    gates = pd.read_csv(figure / "figure_data/D_gate_matrix.csv")
    assert list(gates.hypothesis) == ["H7", "H8", "H9", "H10", "H11"]
    assert (gates.iloc[:, 1:] == "fail").sum().sum() == 7
    tree = ElementTree.parse(figure / "S29.svg")
    assert not tree.findall(".//{http://www.w3.org/2000/svg}image")
    assert len(tree.findall(".//{http://www.w3.org/2000/svg}text")) > 35
    qc = json.loads((figure / "export_qc.json").read_text())
    assert qc["png_minimum_width_pass"]
    assert qc["svg_raster_layers"] == 0
    assert qc["edge_nonwhite_pixels"] == 0
    assert not qc["outside_canvas_text"]
    assert all(qc["source_file_hashes_exact"].values())
    for name in ["S29.png", "S29.pdf", "S29.svg"]:
        assert (figure / name).stat().st_size > 50_000
