from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/guide_dose_response/results"


def test_eligible_rows_are_complete_pairs():
    rows = pd.read_csv(RESULTS / "eligible_guide_rows.csv")
    assert len(rows) == 29130
    assert rows["pair_id"].nunique() == 14565
    assert set(rows.groupby("pair_id").size()) == {2}
    assert rows[["knockdown_fraction", "response", "downstream_genes"]].notna().all().all()


def test_model_contest_is_complete_and_honest():
    metrics = pd.read_csv(RESULTS / "model_metrics.csv", keep_default_na=False)
    comparisons = pd.read_csv(RESULTS / "model_comparisons.csv", keep_default_na=False)
    assert set(metrics["model"]) == {"null", "linear", "spline", "isotonic", "breakpoint", "hill"}
    assert len(metrics) == 18
    assert len(comparisons) == 18
    assert (metrics["r2"] < 0).all()


def test_measurement_sensitivity_and_boundaries_are_reported():
    sensitivity = pd.read_csv(RESULTS / "measurement_error_sensitivity.csv")
    unavailable = pd.read_csv(RESULTS / "unavailable_estimands.csv")
    targets = pd.read_csv(RESULTS / "eligible_target_states.csv")
    assert len(sensitivity) == 9
    assert (sensitivity["ci_low"] > 0).all()
    assert set(targets["target_specific_nonlinearity"]) == {"underpowered"}
    assert "donor-held-out guide-dose validation" in set(unavailable["estimand"])
    assert "cis mediation" in set(unavailable["estimand"])
