import csv
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "analyses/independent_confirmation/metadata/perturbation_candidates.tsv"
SOURCES = ROOT / "analyses/independent_confirmation/metadata/perturbation_sources.tsv"
REPORT = ROOT / "analyses/independent_confirmation/metadata/perturbation_screen.md"
ALLOWED_CHANGED = {
    "analyses/independent_confirmation/metadata/perturbation_candidates.tsv",
    "analyses/independent_confirmation/metadata/perturbation_sources.tsv",
    "analyses/independent_confirmation/metadata/perturbation_screen.md",
    "tests/test_independent_perturbation_screen.py",
}


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_schema_denominator_and_decisions():
    rows = read_tsv(CANDIDATES)
    required = {
        "candidate_id",
        "denominator_weight",
        "dataset_accession",
        "dataset_version",
        "provider_url",
        "license",
        "mapped_cd4_targets",
        "non_targeting_controls",
        "guide_assignments",
        "guides_per_target",
        "donor_count",
        "activation_state_count",
        "matched_states",
        "repeated_units",
        "min_cells_per_target",
        "min_cells_per_target_state",
        "common_genes",
        "portability_decision",
        "tcell_decision",
        "guide_donor_decision",
        "provisional_metadata_sha256",
        "provisional_metadata_hash_basis",
    }
    assert len(rows) == 19
    assert sum(int(row["denominator_weight"]) for row in rows) == 75
    assert required <= set(rows[0])
    assert len({row["candidate_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["portability_decision"] in {"ADVANCE", "STOP"}
        assert row["tcell_decision"] in {"ADVANCE", "STOP"}
        assert row["guide_donor_decision"] in {"ADVANCE", "STOP"}
        assert row["guide_holdout_reuse_decision"] in {"ADVANCE", "STOP"}
        assert row["two_state_core_reuse_decision"] in {"ADVANCE", "STOP"}


def test_provisional_hashes_recompute_independently():
    rows = read_tsv(CANDIDATES)
    for row in rows:
        expected = hashlib.sha256(row["provisional_metadata_hash_basis"].encode("utf-8")).hexdigest()
        assert re.fullmatch(r"[0-9a-f]{64}", row["provisional_metadata_sha256"])
        assert row["provisional_metadata_sha256"] == expected


def test_outcome_blind_access_and_owned_file_boundary():
    rows = read_tsv(CANDIDATES)
    sources = read_tsv(SOURCES)
    prohibited_columns = {
        "expression_matrix_values",
        "response_vector_values",
        "differential_expression_values",
        "effect_sizes",
        "p_values",
        "q_values",
        "outcome_rankings",
        "outcome_figures",
    }
    assert prohibited_columns.isdisjoint(rows[0])
    assert all(row["current_outcome_artifacts_opened"] == "false" for row in rows)
    assert all(row["inspection_scope"] == "metadata_methods_file_inventory_checksums_only" for row in rows)
    assert all(row["current_outcome_artifacts_opened"] == "false" for row in sources)
    commit = subprocess.run(
        ["git", "log", "--format=%H", "--grep=^screen independent perturbations$", "-1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit
    changed = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert set(changed) <= ALLOWED_CHANGED
    assert not any(Path(path).suffix.lower() in {".h5ad", ".h5mu", ".parquet", ".h5"} for path in changed)


def test_sources_resolve_and_are_metadata_only():
    rows = read_tsv(CANDIDATES)
    sources = read_tsv(SOURCES)
    source_ids = {row["source_id"] for row in sources}
    assert len(source_ids) == len(sources)
    for row in rows:
        assert set(row["source_ids"].split(";")) <= source_ids
    assert all(row["inspection_scope"] for row in sources)


def test_frozen_gate_results():
    rows = {row["candidate_id"]: row for row in read_tsv(CANDIDATES)}
    advanced = [row for row in rows.values() if row["portability_decision"] == "ADVANCE"]
    assert [row["candidate_id"] for row in advanced] == ["KOLF2.1J_Perturbation_Cell_Atlas"]
    kolf = advanced[0]
    assert int(kolf["mapped_cd4_targets"]) >= 100
    assert kolf["mapping_sha256"] == "c22383a1e28fed9cd2e7c667c711a997f9373fd80c8f909628f150d21a8c481a"
    assert kolf["processed_counts"].startswith("yes_")
    assert kolf["non_targeting_controls"].startswith("yes_")
    assert kolf["guide_holdout_reuse_decision"] == "ADVANCE"
    assert kolf["two_state_core_reuse_decision"] == "STOP"
    assert kolf["outcome_opening_decision"].startswith("STOP_")
    assert all(row["tcell_decision"] == "STOP" for row in rows.values())
    assert all(row["guide_donor_decision"] == "STOP" for row in rows.values())
    assert rows["GSE314342_Current_Primary_CD4_Source"]["outcomes_inspected_before_freeze"] == "true"
    assert rows["GSE314342_Current_Primary_CD4_Source"]["novelty_gate"] == "STOP_permanent_exclusion"
    assert "KOLF_Strong_Perturbations.h5ad" not in kolf["processed_counts"]
    report = REPORT.read_text(encoding="utf-8")
    assert "Portability metadata selection for H18-H20 | ADVANCE" in report
    assert "Definitive primary T-cell replication for H21-H22 | STOP" in report
    assert "Guide-by-donor reconstruction for H23-H24 | STOP" in report
    assert "No full 189 GB download should occur" in report
    assert "does not replace any frozen control" in report


if __name__ == "__main__":
    tests = [
        test_schema_denominator_and_decisions,
        test_provisional_hashes_recompute_independently,
        test_outcome_blind_access_and_owned_file_boundary,
        test_sources_resolve_and_are_metadata_only,
        test_frozen_gate_results,
    ]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
