from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analyses.grn_benchmark.run_beeline import finite_horizon
from analyses.multiverse.run_scalar_multiverse import adjust_q, fold, multiplier_summary
from analyses.multiverse.run_validation_multiverse import disease_predictions


ROOT = Path(__file__).resolve().parents[1]


def test_scalar_grid_has_frozen_denominator() -> None:
    config = yaml.safe_load((ROOT / "config/multiverse.yaml").read_text(encoding="utf-8"))
    dimensions = [
        config["preprocessing"]["minimum_cells"],
        config["preprocessing"]["minimum_guides"],
        config["preprocessing"]["on_target_requirement"],
        config["transfer_definitions"],
        config["model_families"],
        config["covariate_sets"],
        config["multiverse"]["seeds"],
    ]
    assert len(list(product(*dimensions))) == 4320


def test_hash_folds_and_multiplier_are_reproducible() -> None:
    targets = [f"target_{index}" for index in range(40)]
    first = [fold(target, 20260912, 10) for target in targets]
    second = [fold(target, 20260912, 10) for target in targets]
    assert first == second
    values = pd.Series(np.linspace(-1, 1, 40))
    groups = pd.Series(targets)
    left = multiplier_summary(values, groups, 13, 2000)
    right = multiplier_summary(values, groups, 13, 2000)
    assert left == right
    assert left["bootstrap_replicates"] == 2000
    assert "cluster_gaussian_multiplier" in left["bootstrap_method"]


def test_false_discovery_adjustment_is_within_family() -> None:
    table = pd.DataFrame(
        {
            "result_family": ["a", "a", "b", "b"],
            "p_value": [0.01, 0.04, 0.03, np.nan],
        }
    )
    adjusted = adjust_q(table)
    assert np.allclose(adjusted.loc[:1, "q_value"], [0.02, 0.04])
    assert adjusted.loc[2, "q_value"] == 0.03
    assert np.isnan(adjusted.loc[3, "q_value"])


def test_disease_effect_matches_paired_out_of_fold_brier_improvement() -> None:
    rng = np.random.default_rng(20260912)
    clusters = np.repeat([f"cluster_{index}" for index in range(10)], 12)
    signal = rng.normal(size=len(clusters))
    outcome = signal + rng.normal(scale=0.4, size=len(clusters)) > 0
    table = pd.DataFrame(
        {
            "significant": outcome,
            "cluster": clusters,
            "culture_condition": np.tile(["Th0", "Th2"], len(clusters) // 2),
            "cluster_size": rng.integers(20, 100, size=len(clusters)),
            "transfer_signal": signal,
        }
    )
    delta_auc, contributions = disease_predictions(table, ["transfer_signal"])
    assert np.isfinite(delta_auc)
    assert len(contributions) == len(table)
    assert contributions.notna().all()
    assert contributions.mean() > 0


def test_finite_horizon_operator_is_stabilized() -> None:
    operator = np.array([[1.5, 0.2], [0.0, 1.2]])
    propagated = finite_horizon(operator, depth=3)
    assert np.isfinite(propagated).all()
    assert propagated.shape == operator.shape

