import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/supplementary_nulls/results"


def test_null_denominators_are_complete():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    assert audit == {
        "chromosomes": 19,
        "context_label_permutations": 1000,
        "degree_rewirings": 1000,
        "directional_pairs": 448,
        "feature_label_permutations": 1000,
        "matched_genetic_permutations": 1000,
        "negative_disease_tests": 366,
        "sign_flips": 10000,
    }


def test_negative_controls_are_not_silently_filtered():
    controls = pd.read_csv(RESULTS / "s12_negative_diseases.csv")
    assert len(controls) == 366
    assert (controls["p_adj_fdr"] < 0.05).any()
    assert not controls["significant"].any()
