"""Freeze the extended validation plan before new external outcomes are inspected."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/cell_systems_expansion.yaml"
OUT = ROOT / "analyses/cell_systems_expansion"
STATES = ("Rest", "Stim8hr", "Stim48hr")

RESULT_INPUTS = [
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/primary/results/context_switching.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/vectors/results/network_gain_rows.parquet",
    "analyses/vectors/results/reference_module_loadings.parquet",
    "analyses/guide_concordance/results/eligible_pairs.parquet",
    "analyses/guide_concordance/results/hypothesis_tests.csv",
    "analyses/guide_dose_response/results/guide_held_out_predictions.parquet",
    "analyses/molecular_cascade/results/edge_evidence_matrix.parquet",
    "analyses/molecular_cascade/results/hypothesis_decisions.csv",
    "analyses/molecular_cascade/protocol_revision.json",
    "analyses/state_response_decomposition/results/target_decomposition.parquet",
    "analyses/state_response_decomposition/results/associations.tsv",
    "analyses/replication/results/k562_replication_rows.parquet",
    "analyses/external_benchmark/results/all_eligible_target_predictions.parquet",
    "analyses/external_benchmark/target_folds.tsv",
    "analyses/multiverse/freeze_manifest.json",
    "analyses/causal_triangulation/freeze_manifest.json",
    "data/interim/gwt_vectors/audit.json",
    "data/interim/gwt_vectors/genes.parquet",
    "data/interim/gwt_vectors/rows.parquet",
    "data/interim/gwt_vectors/normalized_significant_logfc.npz",
    "data_manifest/sources.tsv",
    "docs/protocol.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold(identifier: str, folds: int) -> int:
    token = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]
    return 1 + int(token, 16) % folds


def write_tsv(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")
    return path


def complete_targets() -> list[str]:
    frame = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    sets = frame.groupby("target_contrast_gene_name").culture_condition.agg(lambda x: tuple(sorted(set(x))))
    expected = tuple(sorted(STATES))
    targets = sorted(sets.index[sets.map(lambda value: value == expected)].astype(str))
    if len(targets) != 4399:
        raise ValueError(f"Expected 4,399 complete-state targets, found {len(targets):,}")
    return targets


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    targets = complete_targets()
    target_folds = pd.DataFrame({"target": targets})
    target_folds["fold"] = target_folds.target.map(lambda value: fold(value.upper(), cfg["splits"]["target"]["folds"]))
    fold_path = write_tsv(target_folds, "target_folds.tsv")

    hypotheses = []
    for block in ("compositionality", "transportability", "information_retention"):
        for identifier, expectation in cfg[block]["hypotheses"].items():
            hypotheses.append({"hypothesis": identifier, "block": block, "directional_expectation": expectation, "status_at_freeze": "not_tested"})
    hypothesis_path = write_tsv(pd.DataFrame(hypotheses), "hypotheses.tsv")

    datasets = [
        {"dataset": "GSE314342 CD4", "role": "development_source", "outcomes_inspected": True, "eligible_for_new_external_replication": False},
        {"dataset": "Replogle K562 CRISPRi", "role": "external_replication_previously_inspected", "outcomes_inspected": True, "eligible_for_new_external_replication": False},
        {"dataset": "Replogle RPE1 CRISPRi", "role": "external_development_previously_inspected", "outcomes_inspected": True, "eligible_for_new_external_replication": False},
        {"dataset": "Future metadata-eligible combinatorial holdout", "role": "confirmatory_holdout", "outcomes_inspected": False, "eligible_for_new_external_replication": True},
    ]
    dataset_path = write_tsv(pd.DataFrame(datasets), "dataset_roles.tsv")

    configuration_inputs = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "config").glob("*.yaml"))
    prior_freezes = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "analyses").glob("*/freeze_manifest.json")
        if path.parent != OUT
    )
    inputs = list(dict.fromkeys([*configuration_inputs, *prior_freezes, *RESULT_INPUTS]))
    artifacts = []
    for relative in inputs:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(relative)
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    for path in (fold_path, hypothesis_path, dataset_path):
        artifacts.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    target_hash = hashlib.sha256("\n".join(targets).encode("utf-8")).hexdigest()
    manifest = {
        "status": "frozen_before_new_external_outcome_inspection",
        "freeze_seed": cfg["freeze_seed"],
        "git_commit_before_freeze_commit": commit,
        "new_external_outcomes_present": False,
        "previously_inspected_external_systems": ["Replogle K562 CRISPRi", "Replogle RPE1 CRISPRi"],
        "rpe1_untouched_claim_allowed": False,
        "complete_state_targets": len(targets),
        "complete_target_sha256": target_hash,
        "target_fold_counts": {str(key): int(value) for key, value in target_folds.fold.value_counts().sort_index().items()},
        "unmaterialized_split_rules": {
            "pair": cfg["splits"]["pair"],
            "guide": cfg["splits"]["guide"],
            "dataset": cfg["splits"]["dataset"],
            "reason": "External pair and guide identifiers are unavailable before metadata-only eligibility screening; deterministic rules are frozen before acquisition.",
        },
        "dynamic_operator": cfg["dynamic_operator_gate"],
        "configuration_files_hashed": len(configuration_inputs),
        "prior_freeze_manifests_hashed": len(prior_freezes),
        "raw_source_note": "Validated processed response matrices are direct inputs. Large raw source objects are tracked by the existing data manifest and upstream audit rather than rehashed for this expansion.",
        "artifacts": artifacts,
    }
    (OUT / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
