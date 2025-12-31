import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analyses/network_decomposition/results"


def test_network_decomposition_preserves_negative_comparator():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(RESULTS / "buffered_match_candidates.csv")
    networks = pd.read_parquet(RESULTS / "exemplar_networks.parquet")
    assert audit["amplified_exemplar"] == "GATA3"
    assert audit["buffered_comparator"] == candidates.iloc[0]["target_contrast_gene_name"]
    assert audit["buffered_candidate_denominator"] == len(candidates)
    assert set(networks["exemplar"]) == {"Amplified"}
    assert len(networks) == 16


def test_motif_denominators_and_external_null_are_complete():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    motifs = pd.read_parquet(RESULTS / "motif_enrichment_all.parquet")
    null = pd.read_parquet(RESULTS / "external_degree_preserving_null.parquet")
    assert len(motifs) == audit["motif_tests_total"]
    assert (motifs["status"] == "estimable").sum() == audit["motif_tests_estimable"]
    assert motifs.loc[motifs["status"] != "estimable", "q_value"].isna().all()
    assert len(null) == audit["degree_null_replicates"] == 1000


def test_residual_gain_predictions_are_target_held_out():
    predictions = pd.read_parquet(RESULTS / "residual_gain_predictions.parquet")
    assert predictions[["Covariates", "Topology", "Combined"]].notna().all().all()
    assert predictions.groupby("target_contrast")["fold"].nunique().max() == 1
    assert predictions["fold"].nunique() == 10
