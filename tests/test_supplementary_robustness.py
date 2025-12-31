import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/supplementary_robustness/results"


def test_s05_s08_use_complete_frozen_specifications():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    assert audit["scalar_specifications"] == 17280
    assert audit["scalar_unavailable"] == 1440
    assert audit["rerouting_specifications"] == 180
    assert audit["validation_specifications"] == 1008
    assert audit["network_depths"] == 6
    assert audit["module_graph_nodes"] == 30


def test_network_depth_is_bounded_and_complete():
    depth = pd.read_csv(RESULTS / "s06_network_depth.csv")
    assert set(depth["condition"]) == {"Rest", "Stim8hr", "Stim48hr"}
    assert set(depth["depth"]) == set(range(1, 7))
    assert (depth["scaled_spectral_radius"] <= 0.8500001).all()
