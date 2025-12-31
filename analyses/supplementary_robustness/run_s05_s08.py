"""Prepare source tables for Supplementary Figures S5 to S8 without refitting primary models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/supplementary_robustness/results"
STATES = ["Rest", "Stim8hr", "Stim48hr"]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scalar = pd.read_parquet(ROOT / "analyses/multiverse/results/scalar_specifications.parquet")
    unavailable = pd.read_csv(ROOT / "analyses/multiverse/results/scalar_unavailable.csv")
    rerouting = pd.read_parquet(ROOT / "analyses/multiverse/results/rerouting_specifications.parquet")
    validation = pd.read_parquet(ROOT / "analyses/multiverse/results/validation_specifications.parquet")
    tensor = pd.read_parquet(ROOT / "analyses/vectors/results/contextual_transfer_tensor.parquet")

    alternative = scalar.groupby(["transfer_definition", "result_family"], dropna=False).agg(specifications=("estimate", "size"), median_estimate=("estimate", "median"), low_estimate=("estimate", lambda x: x.quantile(0.025)), high_estimate=("estimate", lambda x: x.quantile(0.975)), expected_direction_fraction=("direction_expected", "mean"), supported_fraction=("supported", "mean")).reset_index()
    alternative.to_csv(OUTPUT / "s05_alternative_definitions.csv", index=False)
    tails = scalar[scalar["tail_cutoff"].notna()].groupby(["transfer_definition", "result_family", "tail_cutoff"]).agg(median_estimate=("estimate", "median"), supported_fraction=("supported", "mean"), specifications=("estimate", "size")).reset_index()
    tails.to_csv(OUTPUT / "s05_tail_thresholds.csv", index=False)
    unavailable.groupby(["transfer_definition", "reason"]).size().reset_index(name="unavailable_specifications").to_csv(OUTPUT / "s05_unavailable.csv", index=False)
    concordance = scalar.pivot_table(index=["minimum_cells", "minimum_guides", "on_target_rule", "model_family", "covariate_set", "cross_validation_seed", "result_family", "tail_cutoff"], columns="transfer_definition", values="estimate").corr(method="spearman")
    concordance.to_csv(OUTPUT / "s05_definition_concordance.csv")

    depth_records = []
    spectral_records = []
    for state in STATES:
        scores = tensor[tensor["condition"] == state].pivot(index="target_contrast", columns="module", values="score").fillna(0).to_numpy()
        adjacency = np.corrcoef(scores, rowvar=False)
        np.fill_diagonal(adjacency, 0.0)
        threshold = np.quantile(np.abs(adjacency[np.triu_indices_from(adjacency, 1)]), 0.90)
        adjacency[np.abs(adjacency) < threshold] = 0.0
        raw_radius = float(np.max(np.abs(np.linalg.eigvals(adjacency))))
        scaled = adjacency * (0.85 / raw_radius) if raw_radius > 0 else adjacency
        eigenvalues = np.linalg.eigvals(scaled)
        for index, value in enumerate(eigenvalues):
            spectral_records.append({"condition": state, "eigen_index": index + 1, "real": value.real, "imaginary": value.imag, "magnitude": abs(value), "edge_threshold": threshold, "raw_spectral_radius": raw_radius})
        cumulative = np.eye(scaled.shape[0])
        power = np.eye(scaled.shape[0])
        prior = cumulative.copy()
        for depth in range(1, 7):
            power = power @ scaled
            cumulative = cumulative + power
            depth_records.append({"condition": state, "depth": depth, "operator_frobenius": np.linalg.norm(cumulative, ord="fro"), "increment_frobenius": np.linalg.norm(power, ord="fro"), "successive_cosine": float(np.vdot(prior.ravel(), cumulative.ravel()) / (np.linalg.norm(prior) * np.linalg.norm(cumulative))), "nonzero_edges": int(np.count_nonzero(adjacency)), "scaled_spectral_radius": float(max(abs(eigenvalues)))})
            prior = cumulative.copy()
    pd.DataFrame(depth_records).to_csv(OUTPUT / "s06_network_depth.csv", index=False)
    pd.DataFrame(spectral_records).to_csv(OUTPUT / "s06_spectral_diagnostics.csv", index=False)
    rerouting.groupby(["module_resolution", "cross_validation_seed"]).agg(estimate=("estimate", "median"), ci_low=("ci_low", "median"), ci_high=("ci_high", "median"), supported=("supported", "mean")).reset_index().to_csv(OUTPUT / "s06_module_resolution.csv", index=False)

    pd.read_csv(ROOT / "analyses/baselines/results/model_comparison.csv").to_csv(OUTPUT / "s07_model_comparison.csv", index=False)
    pd.read_csv(ROOT / "analyses/baselines/results/fold_metrics.csv").to_csv(OUTPUT / "s07_fold_metrics.csv", index=False)
    pd.read_csv(ROOT / "analyses/vectors/results/module_folds.csv").to_csv(OUTPUT / "s07_module_folds.csv", index=False)
    scalar.groupby(["cross_validation_seed", "result_family"]).agg(median_estimate=("estimate", "median"), expected_direction_fraction=("direction_expected", "mean"), supported_fraction=("supported", "mean"), specifications=("estimate", "size")).reset_index().to_csv(OUTPUT / "s07_seed_stability.csv", index=False)

    calibration = pd.read_csv(ROOT / "analyses/primary/results/transfer_calibration.csv")
    calibration["absolute_gap"] = (calibration["mean_observed"] - calibration["mean_predicted"]).abs()
    calibration.to_csv(OUTPUT / "s08_internal_calibration.csv", index=False)
    pd.read_csv(ROOT / "analyses/external_benchmark/results/calibration_coverage.csv").to_csv(OUTPUT / "s08_external_coverage.csv", index=False)
    families = pd.concat([
        scalar.groupby("result_family").agg(specifications=("estimate", "size"), supported_fraction=("supported", "mean"), expected_direction_fraction=("direction_expected", "mean")).reset_index().assign(source="scalar"),
        rerouting.groupby("result_family").agg(specifications=("estimate", "size"), supported_fraction=("supported", "mean"), expected_direction_fraction=("direction_expected", "mean")).reset_index().assign(source="rerouting"),
        validation.groupby("result_family").agg(specifications=("estimate", "size"), supported_fraction=("supported", "mean"), expected_direction_fraction=("direction_expected", "mean")).reset_index().assign(source="validation"),
    ], ignore_index=True)
    families.to_csv(OUTPUT / "s08_family_support.csv", index=False)
    audit = {"scalar_specifications": int(len(scalar)), "scalar_unavailable": int(len(unavailable)), "rerouting_specifications": int(len(rerouting)), "validation_specifications": int(len(validation)), "network_depths": 6, "states": STATES, "module_graph_nodes": int(tensor["module"].nunique())}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
