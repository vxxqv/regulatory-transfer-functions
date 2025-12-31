from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import linregress, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("REGULATORY_SOURCE_ROOT", ROOT))
RESULTS = ROOT / "analyses/molecular_cascade/results"
spec = importlib.util.spec_from_file_location("cascade", ROOT / "analyses/molecular_cascade/run_analysis.py")
cascade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cascade)


def test_complete_denominator_and_unique_edges():
    targets = pd.read_parquet(RESULTS / "target_state_evidence.parquet")
    edges = pd.read_parquet(RESULTS / "edge_evidence_matrix.parquet")
    assert len(targets) == 15807
    assert targets.response_row.is_unique
    assert targets.target_gene.nunique() == 6105
    assert not edges.duplicated(["response_row", "gene_column"]).any()
    count = edges.groupby("response_row").size()
    eligible = targets[targets.eligible_regulator]
    np.testing.assert_array_equal(eligible.response_row.map(count).fillna(0), eligible.significant_response_edges)
    assert (targets.loc[~targets.eligible_regulator, "eligibility_status"] == "ineligible_no_compatible_regulatory_resource").all()


def test_every_frozen_hash_matches():
    checks = cascade.verify_freeze()
    assert len(checks) == 27
    protocol = next(record for record in checks if record["path"] == "docs/protocol.md")
    assert protocol["source"] == "historical_git_object"
    assert protocol["sha256"] == "cd3f185cc0b043664ab7afb68b96199b0b90533e042de0a4862ac1f215b7cf9f"


def test_effects_and_rows_independently_match_frozen_matrix():
    matrix = sparse.load_npz(SOURCE / "data/interim/gwt_vectors/normalized_significant_logfc.npz").tocsr()
    rows = pd.read_parquet(SOURCE / "data/interim/gwt_vectors/rows.parquet")
    genes = pd.read_parquet(SOURCE / "data/interim/gwt_vectors/genes.parquet")
    edges = pd.read_parquet(RESULTS / "edge_evidence_matrix.parquet")
    selected = edges.sort_values(["response_row", "gene_column"]).iloc[::max(1, len(edges) // 100)]
    for edge in selected.itertuples(index=False):
        assert edge.crispr_effect == matrix[edge.response_row, edge.gene_column]
        assert edge.response_gene == genes.iloc[edge.gene_column].gene_name.upper()
        assert edge.target_gene == rows.iloc[edge.response_row].target_contrast_gene_name.upper()
        assert edge.culture_condition == rows.iloc[edge.response_row].culture_condition
        assert edge.expected_crispr_sign == np.sign(rows.iloc[edge.response_row].ontarget_effect_size) * edge.curated_direction


def test_molecular_tiers_exclude_validation_outcomes():
    edges = pd.read_parquet(RESULTS / "edge_evidence_matrix.parquet")
    count = edges.curated_edge.astype(int) + edges.motif_supported.astype(int) + (edges.promoter_bound | edges.enhancer_linked).astype(int) + edges.physical_active_link.astype(int)
    np.testing.assert_array_equal(count, edges.independent_evidence_types)
    tier1 = edges.evidence_tier.eq("Tier 1")
    assert (~tier1 | (edges.occupancy_compatible & edges.enhancer_linked & edges.physical_active_link & edges.guide_concordant & ~edges.directional_contradiction)).all()
    assert tier1.sum() == 0
    tier2 = edges.evidence_tier.eq("Tier 2")
    assert (~tier2 | (edges.curated_edge & edges.curated_direction.ne(0) & edges.direction_status.eq("concordant") & count.ge(3))).all()
    assert (~edges.directional_contradiction | edges.evidence_tier.eq("Tier 4")).all()


def test_matched_caliper_and_complete_sets():
    matched = pd.read_parquet(RESULTS / "motif_matched_backgrounds.parquet")
    assert matched.match_distance.le(3).all()
    assert matched.groupby("match_id").size().eq(6).all()
    assert matched.groupby("match_id").culture_condition.nunique().eq(1).all()
    assert matched.groupby("match_id").chromosome_class.nunique().eq(1).all()
    assert matched.loc[matched.role.eq("control"), "matched_gene"].ne(matched.loc[matched.role.eq("control"), "case_gene"]).all()


def test_primary_correlations_and_decisions():
    targets = pd.read_parquet(RESULTS / "target_state_evidence.parquet")
    table = targets[targets.eligible_regulator & targets.significant_response_edges.gt(0)]
    hypotheses = pd.read_csv(RESULTS / "hypothesis_decisions.csv")
    row = hypotheses[hypotheses.family.eq("transfer_gain")].iloc[0]
    assert np.isclose(spearmanr(table.transfer_residual, table.direct_support_fraction).statistic, row.estimate)
    supported = hypotheses[hypotheses.decision.eq("passed")]
    assert supported.ci_low.gt(0).all()
    assert supported.q_value.lt(0.05).all()
    direct = hypotheses[hypotheses.family.eq("direct_timing")].iloc[0]
    assert direct.decision == "unavailable"
    np.testing.assert_allclose(cascade.bh(np.array([0.01, 0.04, 0.03, np.nan])), [0.03, 0.04, 0.04, np.nan], equal_nan=True)


def test_ulm_is_centered_regression_statistic():
    genes = pd.DataFrame({"gene_name": ["A", "B", "C", "D", "E", "F", "G"]})
    rows = pd.DataFrame({"index": [1, 2], "target_contrast_gene_name": ["A", "A"], "culture_condition": ["Rest", "Stim8hr"]})
    expression = np.array([[1., 2, -3, 0, 1, 0, -2], [0., -1, 2, 2, -2, 1, 0]])
    direction = np.array([0, 1, -1, 1, -1, 1, 0])
    network = pd.DataFrame({"source_gene": ["A"] * 5, "response_gene": ["B", "C", "D", "E", "F"], "curated_direction": direction[1:6], "dorothea_a": True, "dorothea_b": False, "dorothea_any": True})
    config = {"eligibility": {"activity_minimum_regulon_targets": 5}, "inference": {"activity_top_per_target_state": 1}}
    top, _, _ = cascade.infer_tf_activity(rows, sparse.csr_matrix(expression), genes, network, np.array([0, 1]), config)
    expected = [linregress(direction, value).slope / linregress(direction, value).stderr for value in expression]
    np.testing.assert_allclose(top.ulm_score, expected, rtol=1e-12)


def test_no_sign_selected_secondary_path():
    edges = pd.read_parquet(RESULTS / "edge_evidence_matrix.parquet")
    matrix = sparse.load_npz(SOURCE / "data/interim/gwt_vectors/normalized_significant_logfc.npz").tocsr()
    names = pd.read_parquet(SOURCE / "data/interim/gwt_vectors/genes.parquet").gene_name.str.upper().tolist()
    index = {name: i for i, name in enumerate(names)}
    subset = edges[edges.candidate_path_count.gt(1)].sort_values(["response_row", "response_gene"]).head(30)
    for edge in subset.itertuples(index=False):
        expected = max(edge.candidate_intermediates.split(";"), key=lambda gene: (abs(matrix[edge.response_row, index[gene]]), gene))
        assert edge.higher_order_intermediate == expected
        assert edge.candidate_agreeing_paths + edge.candidate_conflicting_paths == edge.candidate_path_count


def test_degree_null_deterministic_subset_and_invariants():
    edges = pd.read_parquet(RESULTS / "edge_evidence_matrix.parquet")
    curated = pd.read_parquet(RESULTS / "curated_regulatory_edges.parquet")
    actual = cascade.degree_preserving_null(edges, curated, {"seed": 20260913, "inference": {"degree_rewiring_replicates": 2}})
    frozen = pd.read_parquet(RESULTS / "degree_preserving_rewiring_null.parquet").head(2)
    pd.testing.assert_frame_equal(actual, frozen)
    assert actual.degree_preserved.all()


def test_predictions_have_identical_target_splits_and_correct_r2():
    predictions = pd.read_parquet(RESULTS / "reproducibility_oof_predictions.parquet")
    for dataset, block in predictions.groupby("dataset"):
        assert block.groupby("target_gene").fold.nunique().eq(1).all()
        assert block.groupby("index").model.nunique().eq(2).all()
        assert block.predicted.notna().all()
        assert block.interval_method.eq("training_residual_quantile_not_conformal").all()
    metrics = pd.read_csv(RESULTS / "predictive_validation.csv")
    for row in metrics.itertuples(index=False):
        block = predictions[predictions.dataset.eq(row.dataset) & predictions.model.eq(row.model)]
        expected = 1 - np.square(block.observed - block.predicted).sum() / np.square(block.observed - block.observed.mean()).sum()
        assert np.isclose(expected, row.r2)
        assert "k562_vector_concordant" not in row.features
        assert "guide_concordant" not in row.features


def test_secondary_screen_and_replication_boundaries():
    screen = pd.read_parquet(RESULTS / "secondary_factor_screen.parquet")
    assert not screen.duplicated(["response_row", "secondary_factor", "resource_type"]).any()
    assert screen.overlap_genes.le(screen.response_genes).all()
    assert screen.overlap_genes.le(screen.regulon_genes).all()
    np.testing.assert_allclose(screen.q_value, cascade.bh(screen.p_value.to_numpy()))
    gates = pd.read_csv(RESULTS / "cascade_replication_gates.csv")
    egr1 = gates[gates.target_gene.eq("GATA3") & gates.secondary_factor.eq("EGR1")]
    assert len(egr1) == 3
    assert not egr1.strong_cascade_claim.any()
    assert egr1.loc[egr1.culture_condition.eq("Stim8hr"), "secondary_programme_genes"].iloc[0] == 6
    assert egr1.loc[egr1.culture_condition.eq("Stim48hr"), "secondary_programme_genes"].iloc[0] == 2


def test_state_summary_and_permutation_statistics():
    targets = pd.read_parquet(RESULTS / "target_state_evidence.parquet")
    eligible = targets[targets.eligible_regulator & targets.significant_response_edges.gt(0)]
    wide = eligible.pivot_table(index="target_gene", columns="culture_condition", values="state_rerouting_js").dropna()
    summary = pd.read_csv(RESULTS / "timing_summary.csv")
    for state in ["Rest", "Stim8hr", "Stim48hr"]:
        row = summary[summary.metric.eq("Module rerouting") & summary.culture_condition.eq(state)].iloc[0]
        assert row.targets == len(wide)
        assert np.isclose(row.estimate, wide[state].mean())
    comparisons = pd.read_csv(RESULTS / "paired_state_changes.csv")
    null = pd.read_parquet(RESULTS / "state_label_permutation_null.parquet")
    for row in comparisons.itertuples(index=False):
        draws = null[null.metric.eq(row.metric) & null.first_state.eq(row.first_state) & null.second_state.eq(row.second_state)].null_difference
        assert len(draws) == 1000
        assert draws.notna().all()
        assert np.isclose(row.p_value, (1 + (draws.abs() >= abs(row.mean_difference) - 1e-12).sum()) / 1001)
    np.testing.assert_allclose(comparisons.q_value, cascade.bh(comparisons.p_value.to_numpy()))


def test_figure_has_editable_vector_layers_and_exact_sources():
    directory = ROOT / "figures/fig08"
    assert (directory / "figure8.png").stat().st_size > 50000
    assert (directory / "figure8.pdf").read_bytes().startswith(b"%PDF-")
    tree = ET.parse(directory / "figure8.svg")
    assert not tree.findall(".//{http://www.w3.org/2000/svg}image")
    assert len(tree.findall(".//{http://www.w3.org/2000/svg}text")) > 30
    plotted = pd.read_csv(directory / "figure_data/prediction_comparison.tsv", sep="\t")
    pd.testing.assert_frame_equal(plotted, pd.read_csv(RESULTS / "paired_prediction_comparison.csv"))
    qc = json.loads((directory / "qc.json").read_text())
    assert qc["source_checks_passed"]
