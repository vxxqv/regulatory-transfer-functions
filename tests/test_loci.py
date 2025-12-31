from __future__ import annotations

import pandas as pd


def test_locus_cards_are_complete_and_match_l2g_targets() -> None:
    summary = pd.read_csv("analyses/loci/results/locus_summary.csv")
    variants = pd.read_csv("analyses/loci/results/credible_set_variants.csv")
    assert set(summary["gene"]) == {"GATA3", "STAT3", "PTPN22"}
    assert (summary["gene"] == summary["l2g_target"]).all()
    assert len(variants) == summary["credible_variants"].sum()
    assert variants["posterior_probability"].between(0, 1).all()


def test_pchic_contact_denominator_is_complete_and_nonfabricated() -> None:
    variants = pd.read_csv("analyses/loci/results/credible_set_variants.csv")
    tests = pd.read_csv("analyses/loci/results/credible_variant_pchic_tests.csv")
    contacts = pd.read_csv("analyses/loci/results/credible_variant_pchic_contacts.csv")
    summary = pd.read_csv("analyses/loci/results/pchic_locus_summary.csv")
    assert len(tests) == len(variants) == 50
    assert set(summary["gene"]) == {"GATA3", "STAT3", "PTPN22"}
    assert tests["called_contact"].sum() == len(contacts)
    assert tests.loc[tests["gene"] == "STAT3", "called_gene_interaction_rows"].eq(0).all()
    assert not tests["called_contact"].any()
