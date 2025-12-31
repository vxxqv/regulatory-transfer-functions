"""Target-level resampling and permutation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_bootstrap_interval(
    data: pd.DataFrame,
    group: str,
    value: str,
    statistic: str = "median",
    replicates: int = 2000,
    seed: int = 20260912,
) -> tuple[float, float, float]:
    """Bootstrap a statistic by sampling independent groups with replacement."""

    grouped = [part[value].to_numpy(dtype=float) for _, part in data.groupby(group)]
    grouped = [values[np.isfinite(values)] for values in grouped]
    grouped = [values for values in grouped if len(values)]
    if not grouped:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    function = np.nanmedian if statistic == "median" else np.nanmean
    estimate = float(function(np.concatenate(grouped)))
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, len(grouped), len(grouped))
        draws[index] = function(np.concatenate([grouped[i] for i in chosen]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def condition_switch_null(
    complete: pd.DataFrame,
    target: str,
    condition: str,
    value: str,
    replicates: int = 1000,
    seed: int = 20260912,
) -> tuple[float, np.ndarray, float]:
    """Compare median within-target ranges with condition-stratified permutations."""

    observed = complete.groupby(target)[value].agg(lambda x: x.max() - x.min())
    statistic = float(observed.median())
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=float)
    source = complete[[target, condition, value]].copy()
    for index in range(replicates):
        permuted = source.copy()
        permuted[value] = (
            source.groupby(condition, group_keys=False)[value]
            .apply(lambda x: pd.Series(rng.permutation(x.to_numpy()), index=x.index))
            .sort_index()
        )
        ranges = permuted.groupby(target)[value].agg(lambda x: x.max() - x.min())
        null[index] = ranges.median()
    p_value = float((1 + np.sum(null >= statistic)) / (replicates + 1))
    return statistic, null, p_value
