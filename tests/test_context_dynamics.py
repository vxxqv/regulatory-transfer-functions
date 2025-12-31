import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/context_dynamics/results"


def test_context_dynamics_denominators_are_complete():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(RESULTS / "selected_switch_targets.csv")
    activity = pd.read_parquet(RESULTS / "tf_activity_association.parquet")
    external = pd.read_csv(RESULTS / "independent_state_validation.csv")
    assert audit["complete_three_state_targets"] == 4399
    assert len(selected) == audit["selected_switch_targets"] == 12
    assert len(activity) == audit["tf_activity_targets"] == 4399
    assert activity[["adjusted_tf_activity_range", "adjusted_transfer_switch_score"]].notna().all().all()
    assert len(external) == audit["independent_state_targets"] == 8
    assert audit["independent_state_underpowered"] is True


def test_alluvial_uses_complete_response_target():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    alluvial = pd.read_parquet(RESULTS / "alluvial_programs.parquet")
    assert audit["alluvial_target"] == "PLCG1"
    assert set(alluvial["condition"]) == {"Rest", "Stim8hr", "Stim48hr"}
    totals = alluvial.groupby("condition")["energy_fraction"].sum()
    assert ((totals - 1.0).abs() < 2e-7).all()
