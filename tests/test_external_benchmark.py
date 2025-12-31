from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_holdout_denominator_and_predictions_are_complete() -> None:
    prepared = pd.read_parquet(
        ROOT / "analyses/external_benchmark/rpe1_prepared/eligible_targets.parquet"
    )
    predictions = pd.read_parquet(
        ROOT / "analyses/external_benchmark/results/all_eligible_target_predictions.parquet"
    )
    assert len(prepared) == 490
    assert prepared["target_gene"].is_unique
    assert set(prepared["target_gene"]) == set(predictions["target_gene"])
    assert not predictions.isna().any().any()


def test_all_model_families_and_ablations_are_reported() -> None:
    models = pd.read_csv(ROOT / "analyses/external_benchmark/results/model_comparison.csv")
    expected = {
        "global_mean",
        "simple_covariates",
        "ridge",
        "sparse_nonlinear",
        "standard_network_propagation",
        "graph_readout",
        "frozen_cd4_transfer",
        "without_proximal",
        "without_scalar_transfer",
        "without_context",
        "without_vector",
        "without_network",
    }
    assert set(models["model"]) == expected
    assert models[["r2", "mae", "spearman_rho"]].notna().all().all()
    calibration = pd.read_csv(
        ROOT / "analyses/external_benchmark/results/calibration_coverage.csv"
    )
    assert set(calibration["nominal_coverage"]) == {0.5, 0.8, 0.95}
    assert set(calibration["model"]) == expected
    nulls = pd.read_csv(ROOT / "analyses/external_benchmark/results/null_summary.csv")
    assert set(nulls["null"]) == {
        "matched_cd4_feature_permutation",
        "directed_degree_preserving_rewiring",
    }
    assert (nulls["replicates"] == 1000).all()
    decisions = pd.read_csv(
        ROOT / "analyses/external_benchmark/results/hypothesis_decisions.csv"
    )
    assert len(decisions) == 3
    assert set(decisions["decision"]).issubset({"passed", "failed", "unresolved"})


def test_causal_denominator_and_mediation_stops_are_complete() -> None:
    grades = pd.read_csv(ROOT / "analyses/causal_triangulation/results/locus_grades.csv")
    gates = pd.read_csv(ROOT / "analyses/causal_triangulation/results/evidence_gates.csv")
    assert set(grades["locus"]) == {"gata3", "stat3", "ptpn22"}
    assert len(gates) == 21
    assert (gates.groupby("locus")["gate"].nunique() == 7).all()
    assert not grades["mediation_claim"].any()
    assert (grades["mediation_status"] == "not_tested_assumptions_failed").all()
