import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/guide_concordance/results"
SOURCE = Path(os.environ.get("STUDY_SOURCE_ROOT", ROOT))


def test_complete_denominators_and_pair_grain():
    rows = pd.read_parquet(RESULTS / "all_target_states.parquet")
    audit = json.loads((RESULTS / "audit.json").read_text())
    assert not rows.duplicated(["target", "culture_condition"]).any()
    assert len(rows) == audit["complete_denominator"]
    assert rows.in_primary_study.sum() == 15807
    assert rows.released_pair.sum() == 26496
    assert rows.eligible_pair.sum() == audit["eligible_pairs"]
    eligible = rows[rows.eligible_pair]
    assert eligible.min_guide_cells.ge(100).all()
    assert eligible.min_ntc_expression.ge(.05).all()
    assert (eligible.guide_id_1 != eligible.guide_id_2).all()
    assert set(eligible.target_inference_status) == {"underpowered_two_guides"}
    assert 1 - eligible.quality_complete.mean() < .20
    summaries = pd.read_csv(RESULTS / "state_summaries.csv")
    for row in summaries[summaries.metric == "count_sign_agreement"].itertuples():
        assert row.statistic == "mean"
        values = eligible.loc[eligible.culture_condition == row.culture_condition, "count_sign_agreement"]
        assert np.isclose(row.estimate, values.mean())


def test_independent_source_subset_and_gene_set_overlap():
    meta = SOURCE / "work/upstream/GWT_perturbseq_analysis_2025/metadata"
    if not meta.exists():
        pytest.skip("Frozen upstream input required for independent source reproduction")
    pairs = pd.read_csv(meta / "DE_by_guide.correlation_results.csv", index_col=0).set_index(["target", "culture_condition"])
    kd = pd.read_csv(meta / "suppl_tables/guide_kd_efficiency.suppl_table.csv", index_col=0).rename_axis("guide_id").reset_index().set_index(["guide_id", "culture_condition"])
    rows = pd.read_parquet(RESULTS / "eligible_pairs.parquet").sort_values(["target", "culture_condition"])
    subset = rows.iloc[np.linspace(0, len(rows) - 1, 31, dtype=int)]
    for row in subset.itertuples():
        original = pairs.loc[(row.target, row.culture_condition)]
        assert np.isclose(row.correlation_all, original.correlation_all, equal_nan=True)
        assert np.isclose(row.correlation_signif, original.correlation_signif, equal_nan=True)
        guide1 = kd.loc[(row.guide_id_1, row.culture_condition)]
        guide2 = kd.loc[(row.guide_id_2, row.culture_condition)]
        dose1 = np.clip(1 - guide1.guide_mean_expr / guide1.ntc_mean_expr, -1, 1.5)
        dose2 = np.clip(1 - guide2.guide_mean_expr / guide2.ntc_mean_expr, -1, 1.5)
        assert np.isclose(row.cis_difference, dose2 - dose1)
        assert np.isclose(row.count_difference, np.log((original.n_signif_2 + 1) / (original.n_signif_1 + 1)))
        if row.n_signif_union > 0 and row.count_set_valid:
            shared = original.n_signif_1 + original.n_signif_2 - original.n_signif_union
            assert np.isclose(row.significant_gene_jaccard, shared / original.n_signif_union)


def test_no_outcome_leakage_in_molecular_predictors():
    path = SOURCE / "analyses/molecular_cascade/results/edge_evidence_matrix.parquet"
    if not path.exists():
        pytest.skip("Frozen molecular evidence source required")
    edges = pd.read_parquet(path)
    selected = sorted(edges.target_gene.unique())[::max(1, edges.target_gene.nunique() // 20)]
    edges = edges[edges.target_gene.isin(selected)].copy()
    count = edges[["curated_edge", "motif_supported"]].astype(int).sum(axis=1)
    count += edges[["physical_active_link", "promoter_bound", "enhancer_linked"]].any(axis=1).astype(int)
    expected = (count >= 2).groupby([edges.target_gene, edges.culture_condition]).mean()
    actual = pd.read_csv(RESULTS / "molecular_support.csv").set_index(["target", "culture_condition"])
    assert np.allclose(expected, actual.loc[expected.index, "molecular_support_fraction"])


def test_bootstrap_null_and_heldout_integrity():
    results = pd.read_csv(RESULTS / "hypothesis_tests.csv")
    estimated = results[results.status == "estimated"]
    assert estimated.bootstrap_replicates.eq(2000).all()
    assert estimated.null_replicates.eq(1000).all()
    assert estimated.q_value.between(0, 1).all()
    assert (estimated.ci_low <= estimated.ci_high).all()
    null = pd.read_parquet(RESULTS / "permutation_nulls.parquet")
    assert set(null.groupby("analysis").size()) == {1000}
    assert np.isfinite(null.null_coefficient).all()
    predictions = pd.read_parquet(RESULTS / "target_held_out_predictions.parquet")
    assert (predictions.groupby(["analysis", "model", "target"]).fold.nunique() == 1).all()
    assert not predictions.duplicated(["analysis", "model", "target", "culture_condition"]).any()
    assert set(predictions.model) == {"quality_baseline", "with_predictor"}
    assert np.allclose(estimated.q_value, false_discovery_control(estimated.permutation_p.to_numpy()))
    for row in estimated.itertuples():
        values = null.loc[null.analysis == row.analysis, "null_coefficient"]
        expected_p = (1 + values.abs().ge(abs(row.coefficient)).sum()) / (1 + len(values))
        assert np.isclose(expected_p, row.permutation_p)


def test_independent_paired_coefficient_by_residualization():
    data = pd.read_parquet(RESULTS / "eligible_pairs.parquet")
    covars = ["delta_gc_fraction", "delta_log1p_tss_distance", "delta_library_flag", "delta_bidirectional_promoter", "delta_secondary_alignment", "delta_log1p_cells"]
    data = data.dropna(subset=["cis_difference", "count_difference"] + covars)
    controls = np.column_stack([np.ones(len(data)), data[covars], pd.get_dummies(data.culture_condition, dtype=float, drop_first=True)])
    q, _ = np.linalg.qr(controls)
    residual_x = data.cis_difference.to_numpy() - q @ (q.T @ data.cis_difference.to_numpy())
    residual_y = data.count_difference.to_numpy() - q @ (q.T @ data.count_difference.to_numpy())
    independent = np.dot(residual_x, residual_y) / np.dot(residual_x, residual_x)
    results = pd.read_csv(RESULTS / "hypothesis_tests.csv")
    assert np.isclose(independent, results.loc[results.analysis == "paired_cis_count", "coefficient"].iloc[0], atol=1e-8)


def test_unavailable_estimands_remain_unavailable():
    gates = pd.read_csv(RESULTS / "availability_gates.csv")
    assert len(gates) >= 10
    assert gates[gates.estimand.str.contains("donor")].status.eq("unavailable").all()
    audit = json.loads((RESULTS / "audit.json").read_text())
    assert audit["outcome_leakage_excluded"]
    assert not audit["supplied_correlation_pvalues_used"]
    assert not (RESULTS / "guide_expression_vectors.parquet").exists()


def test_figure_sources_and_editable_vector_export():
    figure = ROOT / "figures/supplement/S25"
    tree = ET.parse(figure / "S25.svg")
    assert not tree.findall(".//{http://www.w3.org/2000/svg}image")
    assert len(tree.findall(".//{http://www.w3.org/2000/svg}text")) > 30
    for name in ["denominators.csv", "state_summaries.csv", "hypothesis_tests.csv", "matched_state_comparisons.csv"]:
        pd.testing.assert_frame_equal(pd.read_csv(figure / "figure_data" / name), pd.read_csv(RESULTS / name))
