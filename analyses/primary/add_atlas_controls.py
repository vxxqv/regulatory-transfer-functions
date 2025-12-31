"""Create matched-null gain and target-bootstrap calibration summaries for Figure 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analyses/primary/results/transfer_phenotypes.parquet"
OUTPUT = ROOT / "analyses/primary/results"
SEED = 20260912
PERMUTATIONS = 1000
BOOTSTRAPS = 2000


def quantile_code(values: pd.Series, bins: int = 4) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, bins, labels=False, duplicates="drop").astype(int)


def main() -> None:
    table = pd.read_parquet(SOURCE).copy()
    table["cis_bin"] = quantile_code(table["cis_magnitude"])
    table["cell_bin"] = quantile_code(table["n_cells_target"])
    table["expression_bin"] = quantile_code(table["target_baseMean"])
    table["guide_bin"] = table["n_guides"].clip(upper=4).astype(int)
    strata_columns = ["culture_condition", "cis_bin", "cell_bin", "expression_bin", "guide_bin"]
    group_indices = [block.index.to_numpy() for _, block in table.groupby(strata_columns, observed=True)]

    residual_scale = table.groupby("culture_condition")["transfer_residual"].transform("std").to_numpy()
    observed = table["transfer_z"].to_numpy(dtype=float)
    outcome = table["log1p_n_downstream"].to_numpy(dtype=float)
    expected = table["expected_log1p_n_downstream"].to_numpy(dtype=float)
    edges = np.linspace(-5.0, 8.0, 79)
    observed_counts, _ = np.histogram(np.clip(observed, edges[0], edges[-1]), bins=edges)
    null_counts = np.zeros(len(edges) - 1, dtype=np.int64)
    null_tail_rates = np.empty(PERMUTATIONS, dtype=float)
    threshold = float(np.quantile(observed, 0.90))
    rng = np.random.default_rng(SEED)
    for replicate in range(PERMUTATIONS):
        permuted = outcome.copy()
        for index in group_indices:
            permuted[index] = rng.permutation(permuted[index])
        null_z = (permuted - expected) / residual_scale
        counts, _ = np.histogram(np.clip(null_z, edges[0], edges[-1]), bins=edges)
        null_counts += counts
        null_tail_rates[replicate] = np.mean(null_z >= threshold)

    widths = np.diff(edges)
    histogram = pd.concat(
        [
            pd.DataFrame(
                {
                    "series": "Observed",
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "density": observed_counts / (observed_counts.sum() * widths),
                }
            ),
            pd.DataFrame(
                {
                    "series": "Matched null",
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "density": null_counts / (null_counts.sum() * widths),
                }
            ),
        ],
        ignore_index=True,
    )
    histogram.to_csv(OUTPUT / "matched_gain_control_histogram.csv", index=False)

    table["calibration_bin"] = pd.qcut(
        table["expected_log1p_n_downstream"].rank(method="first"), 10, labels=False
    )
    calibration_rows = []
    for calibration_bin, block in table.groupby("calibration_bin", sort=True):
        targets = block["target_contrast"].astype(str).unique()
        by_target = {
            target: block.index[block["target_contrast"].astype(str) == target].to_numpy()
            for target in targets
        }
        observed_bootstrap = np.empty(BOOTSTRAPS, dtype=float)
        predicted_bootstrap = np.empty(BOOTSTRAPS, dtype=float)
        for replicate in range(BOOTSTRAPS):
            draw = rng.choice(targets, len(targets), replace=True)
            index = np.concatenate([by_target[target] for target in draw])
            observed_bootstrap[replicate] = table.loc[index, "log1p_n_downstream"].mean()
            predicted_bootstrap[replicate] = table.loc[index, "expected_log1p_n_downstream"].mean()
        calibration_rows.append(
            {
                "calibration_bin": int(calibration_bin),
                "rows": len(block),
                "targets": len(targets),
                "mean_predicted": block["expected_log1p_n_downstream"].mean(),
                "mean_observed": block["log1p_n_downstream"].mean(),
                "observed_ci_low": np.quantile(observed_bootstrap, 0.025),
                "observed_ci_high": np.quantile(observed_bootstrap, 0.975),
                "predicted_ci_low": np.quantile(predicted_bootstrap, 0.025),
                "predicted_ci_high": np.quantile(predicted_bootstrap, 0.975),
            }
        )
    pd.DataFrame(calibration_rows).to_csv(OUTPUT / "transfer_calibration.csv", index=False)
    audit = {
        "rows": len(table),
        "matched_strata": len(group_indices),
        "singleton_strata": int(sum(len(index) == 1 for index in group_indices)),
        "permutations": PERMUTATIONS,
        "bootstrap_replicates": BOOTSTRAPS,
        "observed_amplified_rate": float(np.mean(observed >= threshold)),
        "matched_null_amplified_rate_mean": float(null_tail_rates.mean()),
        "matched_null_amplified_rate_ci": [
            float(np.quantile(null_tail_rates, 0.025)),
            float(np.quantile(null_tail_rates, 0.975)),
        ],
    }
    (OUTPUT / "atlas_control_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
