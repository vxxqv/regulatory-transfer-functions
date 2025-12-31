from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/disease/results"


def test_disease_model_uncertainty_point_estimates_reproduce():
    frame = pd.read_parquet(RESULTS / "disease_model_rows.parquet")
    observed = frame["significant"].astype(int)
    simple = frame["probability_size_state"]
    transfer = frame["probability_transfer_augmented"]
    expected = {
        "auroc": roc_auc_score(observed, transfer) - roc_auc_score(observed, simple),
        "average_precision": average_precision_score(observed, transfer) - average_precision_score(observed, simple),
        "brier": brier_score_loss(observed, simple) - brier_score_loss(observed, transfer),
    }
    summary = pd.read_csv(RESULTS / "model_uncertainty.csv").set_index("metric")
    assert len(frame) == 1708
    assert frame["cluster"].nunique() == 75
    assert frame["disease"].nunique() == 14
    for metric, estimate in expected.items():
        assert np.isclose(summary.loc[metric, "estimate"], estimate, atol=1e-12)


def test_disease_model_uncertainty_resampling_and_decisions_are_complete():
    summary = pd.read_csv(RESULTS / "model_uncertainty.csv").set_index("metric")
    null = pd.read_parquet(RESULTS / "model_null_distribution.parquet")
    assert len(null) == 6000
    assert null.groupby("metric").size().eq(2000).all()
    assert summary.loc["auroc", "decision"] == "unresolved"
    assert summary.loc["average_precision", "decision"] == "unresolved"
    assert summary.loc["brier", "decision"] == "unresolved"
    assert summary.loc["auroc", "ci_high"] < 0
    assert summary.loc["average_precision", "ci_low"] < 0 < summary.loc["average_precision", "ci_high"]
    assert np.isclose(summary.loc["auroc", "q_two_sided"], 0.0742128935532233)
    assert summary["q_two_sided"].ge(0.05).all()
