import csv
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "analyses/independent_confirmation/metadata/kolf_access_feasibility.tsv"
REPORT = ROOT / "analyses/independent_confirmation/metadata/kolf_access_feasibility.md"
ALLOWED_CHANGED = {
    "analyses/independent_confirmation/metadata/kolf_access_feasibility.tsv",
    "analyses/independent_confirmation/metadata/kolf_access_feasibility.md",
    "tests/test_kolf_access_feasibility.py",
}
FORBIDDEN_EXTENSIONS = {".h5ad", ".h5mu", ".h5", ".hdf5", ".loom", ".mtx", ".parquet", ".feather", ".npy", ".npz", ".bin"}


def read_rows():
    with TABLE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def git_lines(*args):
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line]


def test_complete_route_denominator_and_schema():
    rows = read_rows()
    required = {
        "route_id",
        "authority",
        "version_pin",
        "population_scope",
        "outcome_blind",
        "provider_checksum",
        "local_integrity",
        "transfer_bytes",
        "scientific_losslessness",
        "selection_risk",
        "supports_h18_h20",
        "supports_guide_holdout_same_download",
        "route_decision",
        "exact_blocker",
        "outcome_content_opened",
        "free_space_snapshot_bytes",
        "free_space_snapshot_retrieved_utc",
        "metadata_hash_basis",
        "metadata_sha256",
        "study_final_decision",
    }
    assert required <= set(rows[0])
    assert len(rows) == 12
    assert {row["route_id"] for row in rows} == {f"R{i:02d}" for i in range(1, 13)}
    assert len({row["route"] for row in rows}) == 12


def test_metadata_hashes_recompute():
    for row in read_rows():
        expected = hashlib.sha256(row["metadata_hash_basis"].encode("utf-8")).hexdigest()
        assert re.fullmatch(r"[0-9a-f]{64}", row["metadata_sha256"])
        assert row["metadata_sha256"] == expected


def test_one_unambiguous_final_decision():
    rows = read_rows()
    assert {row["study_final_decision"] for row in rows} == {"STOP"}
    assert all(row["route_decision"].startswith("STOP") for row in rows)
    assert [row["route_id"] for row in rows if row["closest_actionable"] == "yes"] == ["R12"]
    assert [row["route_id"] for row in rows if row["supports_h18_h20"] == "yes_in_principle"] == ["R01"]
    assert [row["route_id"] for row in rows if row["supports_h18_h20"] == "yes_conditionally"] == ["R12"]


def test_full_object_and_resource_gate_are_exact():
    rows = {row["route_id"]: row for row in read_rows()}
    full = rows["R01"]
    assert full["transfer_bytes"] == "189393177972"
    assert "afd30fde1e6ad32969c29868394385d1" in full["provider_checksum"]
    assert full["free_space_snapshot_bytes"] == "182568767488"
    assert full["free_space_snapshot_retrieved_utc"] == "2026-09-13T20:58:49.9958938Z"
    assert "6824410484" in full["exact_blocker"]
    assert "182763712512" in full["exact_blocker"]
    assert rows["R03"]["outcome_blind"] == "no"
    assert rows["R06"]["scientific_losslessness"] == "no"
    assert rows["R09"]["local_integrity"].startswith("LFS_SHA256_")
    assert rows["R11"]["transfer_bytes"] == "linked_object_size_not_exposed_in_release_inventory"


def test_no_outcome_artifact_or_large_file():
    rows = read_rows()
    assert all(row["outcome_content_opened"] == "false" for row in rows)
    commit = git_lines("log", "--format=%H", "--grep=^assess kolf access$", "-1")
    if commit:
        changed = git_lines("diff-tree", "--no-commit-id", "--name-only", "-r", commit[0])
    else:
        changed = sorted(set(git_lines("diff", "--name-only") + git_lines("ls-files", "--others", "--exclude-standard")))
    assert set(changed) <= ALLOWED_CHANGED
    assert not any(Path(path).suffix.lower() in FORBIDDEN_EXTENSIONS for path in changed)
    assert all((ROOT / path).stat().st_size < 1_000_000 for path in changed)


def test_report_preserves_opening_boundary_and_reuse_rule():
    text = REPORT.read_text(encoding="utf-8")
    assert "STOP. No currently published route" in text
    assert "No expression matrix" in text
    assert "HDF5 structure inspection is therefore not permitted" in text
    assert "same download can support external guide-held-out information retention" in text
    assert "cannot support donor-held-out or two-state claims" in text
    assert "No alternate dataset or endpoint was substituted" in text
    assert "An initial audit snapshot found 182,763,712,512 free bytes" in text
    assert "retrieved at 2026-09-13T20:58:49.9958938Z" in text
    assert "Later free-space changes require rechecking" in text
    assert "do not alter this audited STOP" in text


def test_text_hygiene():
    combined = TABLE.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8") + Path(__file__).read_text(encoding="utf-8")
    assert "\u2013" not in combined
    assert "\u2014" not in combined


if __name__ == "__main__":
    tests = [
        test_complete_route_denominator_and_schema,
        test_metadata_hashes_recompute,
        test_one_unambiguous_final_decision,
        test_full_object_and_resource_gate_are_exact,
        test_no_outcome_artifact_or_large_file,
        test_report_preserves_opening_boundary_and_reuse_rule,
        test_text_hygiene,
    ]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
