from __future__ import annotations

import pandas as pd


def test_locus_cards_are_complete_and_match_l2g_targets() -> None:
    summary = pd.read_csv("analyses/loci/results/locus_summary.csv")
    variants = pd.read_csv("analyses/loci/results/credible_set_variants.csv")
    assert set(summary["gene"]) == {"GATA3", "STAT3", "PTPN22"}
    assert (summary["gene"] == summary["l2g_target"]).all()
    assert len(variants) == summary["credible_variants"].sum()
    assert variants["posterior_probability"].between(0, 1).all()
