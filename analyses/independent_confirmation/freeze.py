"""Freeze independent-confirmation rules before external outcome access."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses/independent_confirmation"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".tsv", ".txt", ".yaml", ".yml"}

FROZEN_FILES = [
    "config/independent_confirmation.yaml",
    "analyses/independent_confirmation/README.md",
    "analyses/independent_confirmation/freeze.py",
    "analyses/independent_confirmation/hypotheses.tsv",
    "analyses/independent_confirmation/dataset_roles.tsv",
    "analyses/independent_confirmation/candidate_manifest_template.tsv",
    "tests/test_independent_confirmation_freeze.py",
]

REUSED_CD4_ARTIFACTS = [
    "config/analysis.yaml",
    "config/cell_systems_expansion.yaml",
    "config/external_benchmark.yaml",
    "config/guide_concordance.yaml",
    "config/guide_dose_response.yaml",
    "config/molecular_cascade.yaml",
    "config/causal_triangulation.yaml",
    "analyses/external_benchmark/freeze.py",
    "analyses/external_benchmark/freeze_manifest.json",
    "analyses/external_benchmark/target_folds.tsv",
    "analyses/cell_systems_expansion/freeze_manifest.json",
    "analyses/cell_systems_expansion/target_folds.tsv",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/primary/results/context_switching.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/vectors/results/network_gain_rows.parquet",
    "analyses/vectors/results/reference_module_loadings.parquet",
    "analyses/baselines/run_baselines.py",
    "analyses/baselines/results/oof_predictions.parquet",
    "analyses/external_benchmark/run_benchmark.py",
]


def canonical_bytes(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return data, "binary_exact_sha256"
    text = data.decode("utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.encode("utf-8"), "text_utf8_lf_sha256"


def record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    data, mode = canonical_bytes(path)
    clean = path.relative_to(ROOT).as_posix()
    if Path(clean).is_absolute() or ":" in clean:
        raise ValueError(f"Manifest path is not relative: {clean}")
    size_key = "exact_bytes" if mode == "binary_exact_sha256" else "canonical_lf_bytes"
    return {
        "path": clean,
        "hash_mode": mode,
        size_key: len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    frozen = [record(relative) for relative in FROZEN_FILES]
    reused = [record(relative) for relative in REUSED_CD4_ARTIFACTS]
    manifest = {
        "freeze_date": "2026-09-13",
        "status": "frozen_before_independent_outcome_inspection",
        "phase": "independent_confirmation",
        "git_commit_before_freeze_commit": git_commit(),
        "new_external_outcomes_opened": False,
        "candidate_registry_has_outcome_rows": False,
        "metadata_only_screening": True,
        "two_stage_dataset_hash_process": {
            "before_selection": "provisional_metadata_sha256",
            "before_outcome_opening": "final_full_file_sha256_and_verified_provider_checksum",
        },
        "immutable_fields": [
            "candidate_id",
            "dataset_accession",
            "dataset_version",
            "provisional_metadata_sha256",
            "mapping_sha256",
            "dataset_role",
            "selection_rank",
            "selection_locked_utc",
            "outcomes_inspected_before_freeze",
            "outcome_unopened",
        ],
        "excluded_systems": ["CD4", "K562", "RPE1", "Th1", "Th2"],
        "exclude_every_previously_inspected_dataset_or_outcome": True,
        "prior_eight_target_th1_th2_definitive_replication_allowed": False,
        "target_folds": {
            "path": "analyses/external_benchmark/target_folds.tsv",
            "folds": 10,
            "rule": "1_plus_first_16_hex_sha256_target_gene_modulo_10",
            "regenerate": False,
        },
        "hashing_policy": {
            "text": "SHA256 of UTF-8 text normalized to LF; raw, LF, and CRLF are equivalent",
            "binary": "SHA256 of exact bytes",
            "paths": "repository-relative POSIX paths only",
            "manifest_self_hash": "excluded to avoid recursion",
        },
        "frozen_files": frozen,
        "reused_cd4_artifacts": reused,
    }
    (OUT / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
