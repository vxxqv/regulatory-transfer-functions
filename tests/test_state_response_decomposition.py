import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analyses/state_response_decomposition/results"
FIG = ROOT / "figures/supplement/S26"


def test_complete_cohort_and_energy_identity():
    target = pd.read_parquet(OUT / "target_decomposition.parquet")
    state = pd.read_parquet(OUT / "target_state_decomposition.parquet")
    assert len(target) == target["target_contrast"].nunique() == 4399
    assert len(state) == 13197
    assert set(state["culture_condition"]) == {"Rest", "Stim8hr", "Stim48hr"}
    assert state.groupby("target_contrast").size().eq(3).all()
    np.testing.assert_allclose(target["total_energy"], target["core_energy"] + target["residual_energy"], atol=1e-9)
    np.testing.assert_allclose(state["response_energy"], state["core_norm_squared"] + state["residual_energy"] + state["core_residual_cross_term"], atol=1e-9)
    defined = target["total_energy"] > 0
    assert target.loc[~defined, "core_fraction"].isna().all()
    assert target.loc[defined, "core_fraction"].between(0, 1).all()
    assert target["signed_consensus_energy_fraction"].dropna().between(0, 1).all()


def test_folds_and_sensitivity_denominators():
    cohort = pd.read_csv(ROOT / "analyses/state_response_decomposition/frozen_cohort.tsv", sep="\t")
    sensitivity = pd.read_parquet(OUT / "component_sensitivity.parquet")
    fit = pd.read_csv(OUT / "component_fit_audit.tsv", sep="\t")
    assert set(sensitivity["components"]) == {10, 20, 30}
    assert sensitivity.groupby("components").size().eq(4399).all()
    joined = sensitivity.merge(cohort, on="target_contrast", validate="many_to_one")
    assert joined["fold_x"].equals(joined["fold_y"])
    assert (fit["training_targets"] + fit["heldout_targets"]).eq(4399).all()
    assert (fit["training_target_sha256"] != fit["heldout_target_sha256"]).all()
    assert sensitivity["captured_energy_fraction"].dropna().between(-1e-10, 1 + 1e-8).all()
    manifest = json.loads((ROOT / "analyses/state_response_decomposition/freeze_manifest.json").read_text())
    cohort_text = (ROOT / "analyses/state_response_decomposition/frozen_cohort.tsv").read_text(encoding="utf-8")
    assert hashlib.sha256(cohort_text.encode()).hexdigest() == manifest["canonical_cohort_sha256"]
    assert hashlib.sha256(cohort_text.replace("\n", "\r\n").encode()).hexdigest() == manifest["original_crlf_cohort_sha256"]
    assert "cohort_sha256" not in manifest
    config_text = (ROOT / "config/state_response_decomposition.yaml").read_text(encoding="utf-8")
    assert hashlib.sha256(config_text.encode()).hexdigest() == manifest["config_sha256"]


def test_matched_null_and_direction():
    null = pd.read_parquet(OUT / "matched_target_null.parquet")
    audit = json.loads((OUT / "audit.json").read_text())
    assoc = pd.read_csv(OUT / "associations.tsv", sep="\t")
    assert len(null) == 999
    assert null["mean_core_fraction"].between(0, 1).all()
    expected_p = (1 + (null["mean_core_fraction"] >= audit["mean_core_fraction"]).sum()) / 1000
    assert expected_p == audit["matched_null_p"]
    assert "directional one-sided matched permutation" in audit["matched_null_p_tail"]
    assert "not separately preregistered" in audit["tail_reporting"]
    np.testing.assert_allclose(assoc["rho_excess"], assoc["rho"] - assoc["null_median"])
    assert (assoc.loc[assoc["status"] == "below_matched_null", "excess_ci_high"] < 0).all()
    assert (assoc.loc[assoc["status"] == "above_matched_null", "excess_ci_low"] > 0).all()


def test_predictions_reproduce_from_rows():
    rows = pd.read_parquet(OUT / "heldout_prediction_rows.parquet")
    comparison = pd.read_csv(OUT / "heldout_prediction_comparison.tsv", sep="\t").set_index("outcome")
    for outcome, group in rows.groupby("outcome"):
        assert group["target_contrast"].is_unique
        first = (group["observed"] - group["covariate_prediction"]) ** 2
        second = (group["observed"] - group["core_prediction"]) ** 2
        np.testing.assert_allclose(first - second, group["paired_loss_improvement"], atol=1e-14)
        np.testing.assert_allclose((first - second).mean(), comparison.loc[outcome, "delta_mse"], atol=1e-14)
        assert 0 <= comparison.loc[outcome, "q_value"] <= 1


def test_independent_deterministic_source_subset():
    source = Path(os.environ.get("STATE_DECOMPOSITION_SOURCE_ROOT", ROOT))
    rows = pd.read_parquet(source / "data/interim/gwt_vectors/rows.parquet")
    matrix = sparse.load_npz(source / "data/interim/gwt_vectors/normalized_significant_logfc.npz")
    result = pd.read_parquet(OUT / "target_decomposition.parquet").set_index("target_contrast")
    lso = pd.read_parquet(OUT / "leave_one_state.parquet").set_index(["target_contrast", "heldout_state"])
    chosen = result.index.to_numpy()[np.linspace(0, len(result) - 1, 25, dtype=int)]
    for target in chosen:
        index = rows.index[rows["target_contrast"] == target]
        x = matrix[index].toarray().astype(float)
        mean = np.mean(x, axis=0)
        total = np.sum(x ** 2)
        np.testing.assert_allclose(total, result.loc[target, "total_energy"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(3 * (mean @ mean), result.loc[target, "core_energy"], rtol=1e-12, atol=1e-12)
        shared = np.all(x > 0, axis=0) | np.all(x < 0, axis=0)
        consensus = np.where(shared, np.min(np.abs(x), axis=0), 0)
        expected_consensus = 3 * np.sum(consensus ** 2) / total if total else np.nan
        np.testing.assert_allclose(expected_consensus, result.loc[target, "signed_consensus_energy_fraction"], atol=1e-12, equal_nan=True)
        for i, row_index in enumerate(index):
            condition = rows.loc[row_index, "culture_condition"]
            estimate = np.mean(np.delete(x, i, axis=0), axis=0)
            denom = np.linalg.norm(estimate) * np.linalg.norm(x[i])
            expected = estimate @ x[i] / denom if denom else np.nan
            np.testing.assert_allclose(expected, lso.loc[(target, condition), "cosine"], atol=1e-12, equal_nan=True)
    associations = pd.read_csv(OUT / "associations.tsv", sep="\t")
    for _, row in associations.iterrows():
        valid = result[["core_fraction", row["outcome"]]].dropna()
        np.testing.assert_allclose(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic, row["rho"], atol=1e-12)


def test_figure_sources_and_vector_export():
    assert len(list((FIG / "figure_data").glob("*_source.tsv"))) == 6
    assert len(pd.read_csv(FIG / "figure_data/B_source.tsv", sep="\t")) == 4399
    svg = ElementTree.parse(FIG / "S26.svg")
    assert not svg.findall(".//{http://www.w3.org/2000/svg}image")
    qc = json.loads((FIG / "export_qc.json").read_text())
    assert not qc["outside_canvas_text"]
    assert qc["svg_raster_layers"] == 0
    for name in ["S26.png", "S26.pdf", "S26.svg"]:
        assert (FIG / name).stat().st_size > 10000


def test_each_panel_uses_exact_results():
    for panel, file in [("D", "associations.tsv"), ("E", "heldout_prediction_comparison.tsv")]:
        actual = pd.read_csv(FIG / f"figure_data/{panel}_source.tsv", sep="\t")
        expected = pd.read_csv(OUT / file, sep="\t")
        pd.testing.assert_frame_equal(actual, expected)
    f = pd.read_csv(FIG / "figure_data/F_source.tsv", sep="\t").set_index("components")
    sensitivity = pd.read_parquet(OUT / "component_sensitivity.parquet")
    for resolution, group in sensitivity.groupby("components"):
        np.testing.assert_allclose(group["core_fraction"].mean(), f.loc[resolution, "estimate"])
