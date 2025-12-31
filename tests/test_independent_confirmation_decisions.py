from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analyses" / "independent_confirmation"
RESULTS = BASE / "results"
METADATA = BASE / "metadata"
CORE_COLUMNS = ["hypothesis_id", "status", "complete_denominator", "decision", "reason"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def test_decision_ledgers_cover_each_frozen_hypothesis_once():
    frozen = _read(BASE / "hypotheses.tsv")
    perturbation = _read(RESULTS / "perturbation_hypothesis_decisions.tsv")
    canonical = _read(RESULTS / "hypothesis_decisions.tsv")
    genetic = _read(RESULTS / "genetic_hypothesis_decisions.tsv")

    assert frozen["hypothesis_id"].tolist() == [f"H{i}" for i in range(18, 30)]
    assert frozen["status_at_freeze"].eq("not_tested").all()
    assert perturbation["hypothesis_id"].tolist() == [f"H{i}" for i in range(18, 25)]
    assert canonical["hypothesis_id"].tolist() == [f"H{i}" for i in range(18, 30)]
    assert canonical["hypothesis_id"].is_unique
    assert list(canonical.columns[:5]) == CORE_COLUMNS
    for packed_paths in canonical["source_paths"]:
        for relative_path in packed_paths.split(";"):
            assert (ROOT / relative_path).is_file(), relative_path

    pd.testing.assert_frame_equal(
        canonical.loc[canonical["hypothesis_id"].isin(perturbation["hypothesis_id"]), perturbation.columns].reset_index(drop=True),
        perturbation.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        canonical.loc[canonical["hypothesis_id"].isin(genetic["hypothesis_id"]), CORE_COLUMNS].reset_index(drop=True),
        genetic[CORE_COLUMNS].reset_index(drop=True),
    )


def test_perturbation_decisions_match_the_complete_screen_denominator():
    candidates = _read(METADATA / "perturbation_candidates.tsv")
    decisions = _read(RESULTS / "perturbation_hypothesis_decisions.tsv")

    assert len(candidates) == 19
    assert candidates["denominator_weight"].astype(int).sum() == 75
    assert candidates["portability_decision"].eq("ADVANCE").sum() == 1
    assert candidates["tcell_decision"].eq("STOP").all()
    assert candidates["guide_donor_decision"].eq("STOP").all()
    assert candidates["current_outcome_artifacts_opened"].str.lower().eq("false").all()

    assert decisions["status"].eq("unavailable").all()
    assert decisions["decision"].eq("unavailable").all()
    assert decisions["test_status"].eq("not_tested").all()
    assert decisions["screened_candidates"].astype(int).eq(75).all()
    assert decisions["metadata_advances"].astype(int).eq(1).all()
    assert decisions["outcome_opened_confirmations"].astype(int).eq(0).all()
    assert decisions["eligible_branch_datasets"].astype(int).eq(0).all()
    assert decisions["reason"].str.startswith("not tested because").all()


def test_stop_reasons_and_required_branch_denominators_match_sources():
    decisions = _read(RESULTS / "perturbation_hypothesis_decisions.tsv").set_index("hypothesis_id")
    routes = _read(METADATA / "kolf_access_feasibility.tsv").set_index("route_id")
    screen = (METADATA / "perturbation_screen.md").read_text(encoding="utf-8")

    r01 = routes.loc["R01"]
    assert r01["transfer_bytes"] == "189393177972"
    assert r01["route_decision"] == "STOP"
    assert r01["outcome_content_opened"] == "false"
    assert int(r01["free_space_snapshot_bytes"]) < int(r01["transfer_bytes"])

    portability = decisions.loc[["H18", "H19", "H20"]]
    assert portability["blocked_object_bytes"].eq(r01["transfer_bytes"]).all()
    assert portability["required_branch_datasets"].astype(int).eq(1).all()
    assert portability["source_paths"].str.contains("kolf_access_feasibility.tsv", regex=False).all()

    tcell = decisions.loc[["H21", "H22"]]
    assert tcell["required_branch_datasets"].astype(int).eq(2).all()
    assert tcell["reason"].str.contains("zero independent primary T-cell datasets", regex=False).all()

    guide = decisions.loc[["H23", "H24"]]
    assert guide["required_branch_datasets"].astype(int).eq(1).all()
    assert guide["reason"].str.contains("zero datasets jointly met", regex=False).all()

    assert "The denominator contains 75 entries" in screen
    assert "Zero datasets pass every per-dataset gate" in screen
    assert "Zero datasets establish three supported guide efficacies" in screen
    assert "H18-H20, H21-H22, and H23-H24 remain not tested" in screen
