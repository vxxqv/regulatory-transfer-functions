import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "analyses/information_retention"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("information_analysis", HERE / "run_analysis.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_balance_and_split_identity():
    frame = pd.read_parquet(RESULTS / "profile_accounting.parquet")
    assert len(frame) == 13197
    assert frame.target.nunique() == 4399
    assert not frame.duplicated(["target", "culture_condition"]).any()
    assert (frame.groupby("target").size() == 3).all()
    assert not frame.missing_profile.any()
    for state in module.STATES:
        train, test = frame[frame.culture_condition != state], frame[frame.culture_condition == state]
        assert not set(train.source_row) & set(test.source_row)
        assert set(train.target) == set(test.target)
        assert (train.groupby("target").size() == 2).all()
    expected = 1 + frame.target.map(lambda t: int(hashlib.sha256(t.upper().encode()).hexdigest()[:16], 16) % 10)
    assert np.array_equal(frame.fold, expected)


def test_nested_selection_ignores_outer_outcomes():
    tuning = pd.read_csv(RESULTS / "nested_tuning.tsv", sep="\t")
    summary = pd.read_csv(RESULTS / "decoder_summary.tsv", sep="\t")
    for _, row in summary.iterrows():
        subset = tuning[(tuning.heldout_state == row.state) & (tuning.model == row.model)]
        assert row.state not in set(subset.inner_training_state) | set(subset.inner_validation_state)
        mean_loss = subset.groupby("inverse_temperature").log_loss.mean().sort_index()
        assert row.inverse_temperature == mean_loss.idxmin()
        assert len(subset) == 14


def test_information_identity_and_independent_subset():
    pred = pd.read_parquet(RESULTS / "target_predictions.parquet")
    assert len(pred) == 3 * 3 * 4399
    np.testing.assert_allclose(pred.information_nats, np.log(pred.target_classes) + pred.true_log_probability, atol=1e-12)
    sample = pd.read_csv(RESULTS / "independent_subset.tsv", sep="\t")
    for row in sample.itertuples():
        independently_calculated = math.log(4399) + math.log(row.true_probability)
        assert abs(independently_calculated - row.information_nats) < 1e-10
        assert abs(row.inverse_temperature * row.score - row.log_normalizer - math.log(row.true_probability)) < 1e-10


def test_null_denominators_and_bh():
    summary = pd.read_csv(RESULTS / "decoder_summary.tsv", sep="\t")
    null = pd.read_parquet(RESULTS / "matched_nulls.parquet")
    assert len(null) == 9000
    assert len(summary) == 9
    for row in summary.itertuples():
        dist = null[(null.state == row.state) & (null.model == row.model)].information_nats
        assert len(dist) == 1000
        expected = (1 + (dist >= row.information_nats).sum()) / 1001
        assert row.p_value == pytest.approx(expected)
    p = summary.p_value.to_numpy()
    order = np.argsort(p)
    adjusted = np.empty(9)
    for rank, i in enumerate(order):
        adjusted[i] = min(1., min(p[j] * 9 / (r + 1) for r, j in enumerate(order) if r >= rank))
    np.testing.assert_allclose(summary.q_value, adjusted)


def test_unavailable_gates_cannot_pass():
    audit = json.loads((RESULTS / "audit.json").read_text())
    decisions = pd.read_csv(RESULTS / "hypothesis_decisions.tsv", sep="\t")
    assert decisions.hypothesis.tolist() == [f"H{i}" for i in range(12, 18)]
    assert set(decisions.decision) == {"unavailable"}
    assert decisions.p_value.isna().all() and decisions.q_value.isna().all()
    assert audit["primary_information_pass"] is False
    assert not audit["guide_donor_replication_available"]
    assert not audit["external_association_available"]
    assert audit["h12_h17_confirmatory_tests_run"] == 0


def test_zero_vector_and_nuisance_removal():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(20, 12))
    metadata = pd.DataFrame({"cis_magnitude": np.arange(20), "log1p_target_baseMean": rng.normal(size=20), "log1p_n_cells_target": rng.normal(size=20), "n_guides": 2., "culture_condition": ["Rest"] * 10 + ["Stim8hr"] * 10})
    q = module.nuisance_basis(x, metadata)
    residual = x - x @ q @ q.T
    for state in ["Rest", "Stim8hr"]:
        np.testing.assert_allclose(residual[metadata.culture_condition == state].mean(0), 0, atol=1e-12)
    score = module.score_matrix(np.zeros((2, 12)), x, x, q, "RNA_fingerprint_cosine")
    np.testing.assert_array_equal(score, np.zeros_like(score))


def test_calibration_coverage_and_influence():
    coverage = pd.read_csv(RESULTS / "coverage.tsv", sep="\t")
    assert len(coverage) == 27
    assert coverage.observed_label_set_coverage.between(0, 1).all()
    pred = pd.read_parquet(RESULTS / "target_predictions.parquet")
    for _, frame in pred.groupby(["culture_condition", "model"]):
        values = frame.information_nats.to_numpy()
        index = 17
        delta = np.delete(values, index).mean() - values.mean()
        assert delta == pytest.approx(frame.leave_one_target_mean_change.iloc[index], abs=1e-12)


def test_source_backed_subset():
    source = os.environ.get("INFORMATION_SOURCE_ROOT")
    if not source:
        pytest.skip("INFORMATION_SOURCE_ROOT is required for source-backed verification")
    source = Path(source)
    records, expansion = module.verify(source)
    frozen = json.loads((HERE / "freeze_manifest.json").read_text())
    assert records == frozen["artifacts"] and expansion == frozen["expansion_freeze_sha256"]
    matrix = sparse.load_npz(source / "data/interim/gwt_vectors/normalized_significant_logfc.npz")
    mask = pd.read_csv(RESULTS / "feature_mask.tsv", sep="\t")
    matrix = matrix[:, ~mask.excluded_target_feature.to_numpy()].astype(float)
    accounting = pd.read_parquet(RESULTS / "profile_accounting.parquet")
    sample = pd.read_csv(RESULTS / "independent_subset.tsv", sep="\t")
    for state in module.STATES:
        q = np.load(RESULTS / f"nuisance_{state}_RNA_fingerprint_cosine.npz")["basis"]
        selected = sample[(sample.state == state) & (sample.model == "RNA_fingerprint_cosine")]
        for row in selected.itertuples():
            target = accounting[accounting.target == row.target]
            query = matrix[int(target[target.culture_condition == state].source_row.iloc[0])].toarray().ravel()
            training = matrix[target[target.culture_condition != state].source_row.to_numpy()].toarray()
            centroid = training.mean(0)
            query = query - q @ (q.T @ query)
            centroid = centroid - q @ (q.T @ centroid)
            denominator = np.linalg.norm(query) * np.linalg.norm(centroid)
            independent = np.dot(query, centroid) / denominator if denominator > 1e-12 else 0.
            assert independent == pytest.approx(row.score, abs=1e-9)
    assert not set(mask.loc[~mask.excluded_target_feature, "gene_name"]) & set(accounting.target)
