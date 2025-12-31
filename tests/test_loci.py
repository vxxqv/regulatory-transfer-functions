from __future__ import annotations

import json

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
    gene_contacts = pd.read_csv("analyses/loci/results/target_gene_pchic_contacts.csv")
    liftover = pd.read_csv("analyses/loci/results/pchic_liftover_rows.csv")
    summary = pd.read_csv("analyses/loci/results/pchic_locus_summary.csv")
    audit = json.loads(open("analyses/loci/results/pchic_audit.json", encoding="utf-8").read())
    assert len(tests) == len(variants) == 50
    assert set(summary["gene"]) == {"GATA3", "STAT3", "PTPN22"}
    assert tests["called_contact"].sum() == len(contacts)
    assert tests.loc[tests["gene"] == "STAT3", "called_gene_interaction_rows"].eq(0).all()
    assert (tests["grch38_position_0based"] + 1 == tests["grch38_position"]).all()
    assert not tests["called_contact"].any()
    assert len(gene_contacts) == 58
    assert set(gene_contacts["gene"]) == {"GATA3", "PTPN22"}
    assert gene_contacts[["called_activated", "called_nonactivated"]].any(axis=1).all()
    assert len(liftover) == 58
    assert liftover["liftover_status"].eq("mapped").all()
    assert liftover[
        [
            "bait_first_candidates",
            "bait_last_candidates",
            "prey_first_candidates",
            "prey_last_candidates",
        ]
    ].eq(1).all().all()
    assert audit["liftover_method"] == "UCSC hg19ToHg38 chain via pyliftover"
    assert audit["source_assembly"] == "GRCh37"
    assert audit["target_assembly"] == "GRCh38"
    assert audit["failed_liftover_rows"] == 0
    assert len(audit["chain_sha256"]) == 64
    nearest = summary.set_index("locus")["nearest_called_contact_bp"]
    assert nearest["gata3"] == 108305
    assert nearest["ptpn22"] == 66475
