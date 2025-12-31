import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "analyses" / "independent_confirmation" / "metadata"
CANDIDATES = METADATA / "molecular_candidates.tsv"
SOURCES = METADATA / "molecular_sources.tsv"
SCREEN = METADATA / "molecular_screen.md"


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def digest(row, fields, excluded):
    basis = "\n".join(f"{field}={row[field]}" for field in fields if field not in excluded) + "\n"
    return hashlib.sha256(basis.encode()).hexdigest()


def test_schema_and_complete_denominator():
    candidates = read_tsv(CANDIDATES)
    sources = read_tsv(SOURCES)
    required = {
        "candidate_id",
        "accession",
        "evidence_family",
        "genome_build",
        "identifier_scheme",
        "factor_targets",
        "frozen_perturbed_tf_coverage_n",
        "frozen_secondary_tf_coverage_n",
        "donor_count",
        "biological_replicates",
        "technical_replicates",
        "qc_metadata",
        "build_gate",
        "identifier_gate",
        "peak_quality_gate",
        "state_gate",
        "time_gate",
        "license",
        "provider_checksum",
        "outcome_unopened",
        "decision",
        "provisional_metadata_sha256",
    }
    assert len(candidates) == 27
    assert len(sources) == 27
    assert required <= set(candidates[0])
    assert len({row["candidate_id"] for row in candidates}) == 27
    assert len({row["source_key"] for row in sources}) == 27
    assert {row["source_key"] for row in candidates} == {row["source_key"] for row in sources}
    assert all(all(value != "" for value in row.values()) for row in candidates)
    assert all(all(value != "" for value in row.values()) for row in sources)


def test_deterministic_metadata_hashes():
    candidates = read_tsv(CANDIDATES)
    sources = read_tsv(SOURCES)
    candidate_fields = list(candidates[0])
    source_fields = list(sources[0])
    for row in candidates:
        observed = row["provisional_metadata_sha256"]
        assert len(observed) == 64
        assert observed == digest(row, candidate_fields, {"selection_rank", "provisional_metadata_sha256"})
    for row in sources:
        observed = row["source_metadata_sha256"]
        assert len(observed) == 64
        assert observed == digest(row, source_fields, {"source_metadata_sha256"})


def test_no_outcome_payload_columns_or_access():
    candidates = read_tsv(CANDIDATES)
    sources = read_tsv(SOURCES)
    forbidden = {
        "effect_size",
        "fold_change",
        "p_value",
        "q_value",
        "expression_value",
        "peak_signal",
        "enrichment_score",
        "response_vector",
        "guide_effect",
    }
    assert forbidden.isdisjoint(candidates[0])
    assert forbidden.isdisjoint(sources[0])
    assert all(row["outcome_fields_opened"] == "false" for row in candidates)
    assert all(row["outcome_unopened"] == "true" for row in candidates if row["accession"] != "GSE314342")
    excluded = next(row for row in candidates if row["accession"] == "GSE314342")
    assert excluded["outcome_unopened"] == "false"
    assert excluded["decision"] == "STOP"


def test_factor_state_time_coverage_and_nonprivileged_screen():
    candidates = read_tsv(CANDIDATES)
    assert {row["frozen_perturbed_tf_denominator"] for row in candidates} == {"284"}
    assert {row["frozen_secondary_tf_denominator"] for row in candidates} == {"133"}
    ctcf = [row for row in candidates if row["factor_targets"] == "CTCF"]
    assert {row["accession"] for row in ctcf} == {"ENCSR470KCE", "ENCSR806WWX", "ENCSR093CTT"}
    assert all(row["frozen_perturbed_tf_coverage_n"] == "0" for row in ctcf)
    assert all(row["frozen_secondary_tf_coverage_n"] == "0" for row in ctcf)
    gse_occupancy = next(row for row in candidates if row["accession"] == "GSE271089")
    assert gse_occupancy["state_gate"] == "pass"
    assert gse_occupancy["time_gate"] == "pass"
    assert gse_occupancy["frozen_perturbed_tf_coverage_n"] == "0"
    assert gse_occupancy["frozen_secondary_tf_coverage_n"] == "0"
    assert all("GATA3" not in row["stop_reason"] and "EGR1" not in row["stop_reason"] for row in candidates)
    assert all(row["decision"] == "STOP" for row in candidates)


def test_context_motifs_and_curated_sources_cannot_advance():
    candidates = read_tsv(CANDIDATES)
    context = [row for row in candidates if row["chromatin_context_only"] == "true"]
    motifs = [row for row in candidates if row["motif_only"] == "true"]
    curated = [row for row in candidates if row["evidence_family"] == "curated_network_correlated"]
    assert context and motifs and curated
    assert all(row["decision"] == "STOP" for row in context + motifs)
    assert all(row["correlated_family"] == "curated_network_correlated" for row in curated)
    assert {row["accession"] for row in curated} >= {
        "DoRothEA-A",
        "TRRUSTv2",
        "RegNetwork-frozen",
        "RegNetwork2025",
        "CollecTRI2023",
        "TFTG2024",
        "OmniPath-2026-09-14",
    }


def test_selection_rank_is_metadata_only_and_deterministic():
    candidates = read_tsv(CANDIDATES)
    sources = {row["source_key"]: row for row in read_tsv(SOURCES)}

    def score(row):
        if row["source_key"] == "S17":
            return -999
        verified_positive = row["factor_coverage_status"].startswith("verified") and (
            row["frozen_perturbed_tf_coverage_n"] not in {"0", "unavailable"}
            or row["frozen_secondary_tf_coverage_n"] not in {"0", "unavailable"}
        )
        return (
            8 * verified_positive
            + 4 * (row["state_gate"] == "pass")
            + 4 * (row["time_gate"] == "pass")
            + (sources[row["source_key"]]["provider_record_status"] == "screened")
            + 3 * (row["physical_evidence"] == "true")
            + 3 * (row["perturbational_evidence"] == "true")
            + 2 * (row["alternative_cascade_capable"] == "true")
            + (row["primary_cell_flag"] == "true")
            - 3 * (row["chromatin_context_only"] == "true")
            - 5 * (row["motif_only"] == "true")
            - 5 * (row["correlated_family"] != "none")
        )

    assert all(int(row["metadata_rank_score"]) == score(row) for row in candidates)
    expected = sorted(candidates, key=lambda row: (-score(row), row["accession"]))
    assert [int(row["selection_rank"]) for row in expected] == list(range(1, 28))


def test_h25_h26_stop_and_ascii_hyphens():
    text = SCREEN.read_text(encoding="utf-8")
    assert "| H25 | STOP |" in text
    assert "| H26 | STOP |" in text
    assert "No candidate advances to downstream molecular outcome analysis." in text
    for path in (CANDIDATES, SOURCES, SCREEN, Path(__file__)):
        content = path.read_text(encoding="utf-8")
        assert not ({"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"} & set(content))
