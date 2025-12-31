"""Reconcile matched-null direction and FDR from frozen numerical outputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_analysis import bh

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses/state_response_decomposition/results"


def main():
    cfg = yaml.safe_load((ROOT / "config/state_response_decomposition.yaml").read_text())["state_response_decomposition"]
    associations = pd.read_csv(OUT / "associations.tsv", sep="\t")
    for source, destination in [("rho", "rho_excess"), ("ci_low", "excess_ci_low"), ("ci_high", "excess_ci_high")]:
        associations[destination] = associations[source] - associations["null_median"]
    associations["status"] = np.where((associations["q_value"] < .05) & (associations["excess_ci_low"] > 0), "above_matched_null", np.where((associations["q_value"] < .05) & (associations["excess_ci_high"] < 0), "below_matched_null", "unresolved"))
    associations.to_csv(OUT / "associations.tsv", sep="\t", index=False)
    comparison = pd.read_csv(OUT / "heldout_prediction_comparison.tsv", sep="\t")
    rows = pd.read_parquet(OUT / "heldout_prediction_rows.parquet")
    for i, row in comparison.iterrows():
        delta = rows.loc[rows["outcome"] == row["outcome"], "paired_loss_improvement"].to_numpy()
        rng = np.random.default_rng(cfg["seed"] + 800)
        estimates = np.array([np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(cfg["bootstrap_replicates"])])
        np.testing.assert_allclose(np.quantile(estimates, [.025, .975]), [row["ci_low"], row["ci_high"]], atol=1e-14)
        comparison.loc[i, "p_value"] = (1 + np.sum(np.abs(estimates - delta.mean()) >= abs(delta.mean()))) / (len(estimates) + 1)
    comparison["q_value"] = bh(comparison["p_value"])
    comparison["status"] = np.where((comparison["q_value"] < .05) & (comparison["ci_low"] > 0), "supported", np.where((comparison["q_value"] < .05) & (comparison["ci_high"] < 0), "contradictory", "unresolved"))
    comparison.to_csv(OUT / "heldout_prediction_comparison.tsv", sep="\t", index=False)
    record = {"model_refits": 0, "source_values_changed": False, "correction": "Association direction is relative to the matched null, not the raw zero-null correlation. Prediction comparisons include FDR from centered target-bootstrap tests.", "intervals_reproduced": True}
    (OUT / "statistical_qc.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
