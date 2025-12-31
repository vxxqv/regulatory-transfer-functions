from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/primary/results"


def test_transfer_stability_denominator_and_classes():
    targets = pd.read_parquet(RESULTS / "transfer_stability_targets.parquet")
    assert len(targets) == 6105
    assert targets["target"].is_unique
    assert targets[["unshrunken", "shrunken", "shrinkage_weight"]].notna().all().all()
    assert targets["shrinkage_weight"].between(0, 1).all()
    assert targets["original_class"].value_counts().to_dict() == {
        "intermediate": 4883,
        "buffered": 611,
        "amplified": 611,
    }


def test_transfer_stability_key_statistics_reproduce():
    targets = pd.read_parquet(RESULTS / "transfer_stability_targets.parquet")
    summary = pd.read_csv(RESULTS / "transfer_stability_summary.csv").set_index("metric")
    correlation = spearmanr(targets["unshrunken"], targets["shrunken"]).statistic
    buffered = targets["original_class"].eq("buffered")
    amplified = targets["original_class"].eq("amplified")
    buffered_retention = targets.loc[buffered, "rank_shrunken_class"].eq("buffered").mean()
    amplified_retention = targets.loc[amplified, "rank_shrunken_class"].eq("amplified").mean()
    assert np.isclose(correlation, summary.loc["spearman", "estimate"])
    assert np.isclose(buffered_retention, summary.loc["buffered_rank_retention", "estimate"])
    assert np.isclose(amplified_retention, summary.loc["amplified_rank_retention", "estimate"])
    assert summary.loc[["spearman", "buffered_rank_retention", "amplified_rank_retention"], "decision"].eq("supported").all()


def test_transfer_sensitivity_and_influence_are_complete():
    sensitivity = pd.read_csv(RESULTS / "transfer_measurement_error_sensitivity.csv")
    influence = pd.read_csv(RESULTS / "transfer_stability_influence.csv")
    assert sensitivity["residual_variance_multiplier"].tolist() == [1.0, 1.25, 1.5]
    assert sensitivity[["spearman", "buffered_rank_retention", "amplified_rank_retention"]].notna().all().all()
    assert len(influence) == 6105
    assert influence["target"].is_unique
    assert (influence[["max_cooks_distance", "max_leverage"]] >= 0).all().all()
