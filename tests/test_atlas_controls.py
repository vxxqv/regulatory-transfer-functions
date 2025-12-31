import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_atlas_controls_have_frozen_replication_counts_and_complete_bins() -> None:
    result_dir = ROOT / "analyses" / "primary" / "results"
    audit = json.loads((result_dir / "atlas_control_audit.json").read_text(encoding="utf-8"))
    histogram = pd.read_csv(result_dir / "matched_gain_control_histogram.csv")
    calibration = pd.read_csv(result_dir / "transfer_calibration.csv")
    assert audit["permutations"] == 1000
    assert audit["bootstrap_replicates"] == 2000
    assert set(histogram["series"]) == {"Observed", "Matched null"}
    assert histogram.groupby("series").size().nunique() == 1
    assert calibration["calibration_bin"].tolist() == list(range(10))
    assert (calibration["observed_ci_low"] <= calibration["mean_observed"]).all()
    assert (calibration["mean_observed"] <= calibration["observed_ci_high"]).all()
