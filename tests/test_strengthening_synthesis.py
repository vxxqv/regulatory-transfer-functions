from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "analyses/cell_systems_expansion/results/strengthening_results.tsv"
TABLE = ROOT / "tables/table_s9_strengthening_results.tsv"


def test_strengthening_synthesis_is_complete_and_identical():
    result = pd.read_csv(RESULT, sep="\t")
    table = pd.read_csv(TABLE, sep="\t")
    pd.testing.assert_frame_equal(result, table)
    assert len(result) == 10
    assert result["result_family"].is_unique
    assert result.notna().all().all()


def test_strengthening_synthesis_retains_all_statuses():
    result = pd.read_csv(RESULT, sep="\t").set_index("result_family")
    assert result.loc["Transfer gain and buffering", "final_status"] == "mixed"
    assert result.loc["Signed programme rerouting and context dependence", "final_status"] == "mixed"
    assert result.loc["Guide level dose response and concordance", "final_status"] == "supported"
    assert result.loc["External perturbation transportability", "final_status"] == "unavailable"
    assert result.loc["GRN benchmarking", "final_status"] == "failed"
    assert result.loc["Natural genetic concordance", "final_status"] == "unresolved"
    assert result.loc["Disease convergence and locus causal triangulation", "final_status"] == "failed"


def test_strengthening_synthesis_uses_verified_extensions():
    result = pd.read_csv(RESULT, sep="\t").set_index("result_family")
    assert "0.9942" in result.loc["Transfer gain and buffering", "strengthened_estimate"]
    assert "-0.3430" in result.loc["Signed programme rerouting and context dependence", "strengthened_estimate"]
    assert "-0.0616" in result.loc["GRN benchmarking", "strengthened_estimate"]
    molecular_denominator = result.loc["Molecular regulatory support", "original_denominator"]
    assert "715 eligible regulator-state combinations" in molecular_denominator
    assert "600 nonnull summaries clustered across 265 regulators" in molecular_denominator
    assert "124798 edge rows" in molecular_denominator
    cross_state = result.loc["Cross state and cross system conservation"]
    assert "arithmetic-component energy fraction" in cross_state["confidence_interval"]
    assert "core fraction" not in " ".join(cross_state.astype(str))
    assert "165 trans genes" in result.loc["Natural genetic concordance", "original_denominator"]
    assert "-0.1864 to -0.0096" in result.loc["Disease convergence and locus causal triangulation", "confidence_interval"]
    assert "paired two-sided q 0.0742" in result.loc["Disease convergence and locus causal triangulation", "corrected_significance"]
    assert "did not establish reverse superiority" in result.loc["Disease convergence and locus causal triangulation", "effect_size_interpretation"]
    assert "matched negative-locus analysis is unresolved" in result.loc["Disease convergence and locus causal triangulation", "remaining_limitation"].lower()
    assert result["materially_improves_manuscript"].sum() == 5
