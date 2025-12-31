from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/grn_benchmark/results"


def test_paired_grn_denominators_and_pairing():
    targets = pd.read_parquet(RESULTS / "paired_target_metrics.parquet")
    contrasts = pd.read_csv(RESULTS / "paired_method_contrasts.csv")
    assert len(targets) == 2796
    assert targets.groupby("dataset").size().to_dict() == {
        "hESC": 815,
        "hHep": 874,
        "mDC": 415,
        "mHSC-E": 692,
    }
    assert not targets.duplicated(["dataset", "target"]).any()
    assert len(contrasts) == 12
    assert contrasts.groupby("dataset").size().eq(3).all()
    assert contrasts["n_targets"].ge(100).all()


def test_paired_grn_contrasts_reproduce_from_target_rows():
    targets = pd.read_parquet(RESULTS / "paired_target_metrics.parquet")
    contrasts = pd.read_csv(RESULTS / "paired_method_contrasts.csv").set_index(["dataset", "metric"])
    for dataset, block in targets.groupby("dataset"):
        for metric in ["auroc", "auprc", "early_precision_ratio"]:
            estimate = block[f"difference_{metric}"].mean()
            method = block[f"finite_horizon_transfer_{metric}"].mean()
            comparator = block[f"comparator_{metric}"].mean()
            result = contrasts.loc[(dataset, metric)]
            assert np.isclose(estimate, result["mean_paired_difference"], atol=1e-15)
            assert np.isclose(method, result["mean_method_metric"], atol=1e-15)
            assert np.isclose(comparator, result["mean_comparator_metric"], atol=1e-15)


def test_grn_calibration_and_heterogeneity_are_complete():
    calibration = pd.read_csv(RESULTS / "score_calibration.csv")
    heterogeneity = pd.read_csv(RESULTS / "dataset_heterogeneity.csv")
    assert len(calibration) == 80
    assert calibration.groupby(["dataset", "method"]).size().eq(10).all()
    assert calibration[["mean_held_out_probability", "observed_gold_frequency"]].apply(lambda x: x.between(0, 1).all()).all()
    assert set(heterogeneity["metric"]) == {"auroc", "auprc", "early_precision_ratio"}
    assert heterogeneity["k_datasets"].eq(4).all()
    assert heterogeneity["i_squared_percent"].between(0, 100).all()
    auroc = heterogeneity.set_index("metric").loc["auroc"]
    assert auroc["random_effect_ci_high"] < 0
