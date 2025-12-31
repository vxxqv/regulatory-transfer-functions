from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/vectors/results"


def test_composition_sensitivity_denominators_and_decisions():
    pairs = pd.read_parquet(RESULTS / "composition_sensitivity_pairs.parquet")
    summary = pd.read_csv(RESULTS / "composition_sensitivity_summary.csv").set_index("scope")
    null = pd.read_parquet(RESULTS / "composition_sensitivity_null.parquet")
    assert len(pairs) == 8168
    assert pairs["target_contrast"].nunique() == 3696
    assert pairs.groupby("state_pair").size().to_dict() == {
        "Rest__Stim48hr": 2647,
        "Rest__Stim8hr": 2671,
        "Stim8hr__Stim48hr": 2850,
    }
    assert len(null) == 4000
    assert summary["status"].eq("failed").all()
    assert summary["observed_minus_null_median"].lt(0).all()
    assert summary["q_value_upper_bh"].eq(1).all()


def test_composition_subset_reproduces_log_ratio_distance():
    pairs = pd.read_parquet(RESULTS / "composition_sensitivity_pairs.parquet").head(101)
    tensor = pd.read_parquet(RESULTS / "contextual_transfer_tensor.parquet")
    wide = tensor.pivot(index=["target_contrast", "condition"], columns="module", values="energy_fraction")
    for row in pairs.itertuples(index=False):
        left = wide.loc[(row.target_contrast, row.left_state)].to_numpy(float)
        right = wide.loc[(row.target_contrast, row.right_state)].to_numpy(float)
        left /= left.sum()
        right /= right.sum()
        left_clr = np.log(left) - np.log(left).mean()
        right_clr = np.log(right) - np.log(right).mean()
        distance = np.sqrt(np.mean(np.square(left_clr - right_clr)))
        assert np.isclose(distance, row.normalized_aitchison_distance, atol=1e-12)


def test_composition_closure_and_finite_outputs():
    pairs = pd.read_parquet(RESULTS / "composition_sensitivity_pairs.parquet")
    closure = pairs[[
        "left_replaced_closure",
        "right_replaced_closure",
    ]].to_numpy(float)
    assert np.allclose(closure, 1.0, atol=1e-12)
    assert np.isfinite(pairs["normalized_aitchison_distance"]).all()
    assert (pairs["normalized_aitchison_distance"] >= 0).all()
