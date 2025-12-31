import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/cell_systems_expansion.yaml"
FREEZE = ROOT / "analyses/cell_systems_expansion"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".tsv", ".txt", ".yaml", ".yml"}


def hash_candidates(path):
    data = path.read_bytes()
    candidates = [data]
    if path.suffix.lower() in TEXT_SUFFIXES:
        lf = data.replace(b"\r\n", b"\n")
        candidates.extend([lf, lf.replace(b"\n", b"\r\n")])
    return {hashlib.sha256(value).hexdigest(): len(value) for value in candidates}


def test_freeze_manifest_and_artifacts():
    manifest = json.loads((FREEZE / "freeze_manifest.json").read_text())
    assert manifest["status"] == "frozen_before_new_external_outcome_inspection"
    assert manifest["new_external_outcomes_present"] is False
    assert manifest["rpe1_untouched_claim_allowed"] is False
    assert manifest["complete_state_targets"] == 4399
    assert manifest["configuration_files_hashed"] >= 1
    assert manifest["prior_freeze_manifests_hashed"] >= 1
    paths = {record["path"] for record in manifest["artifacts"]}
    assert set(path.relative_to(ROOT).as_posix() for path in (ROOT / "config").glob("*.yaml")) <= paths
    for record in manifest["artifacts"]:
        path = ROOT / record["path"]
        candidates = hash_candidates(path)
        assert record["sha256"] in candidates
        assert candidates[record["sha256"]] == record["bytes"]


def test_target_splits_and_hypotheses():
    folds = pd.read_csv(FREEZE / "target_folds.tsv", sep="\t")
    assert len(folds) == 4399
    assert folds.target.is_unique
    assert set(folds.fold) == set(range(1, 11))
    assert folds.groupby("target").fold.nunique().eq(1).all()
    hypotheses = pd.read_csv(FREEZE / "hypotheses.tsv", sep="\t")
    assert set(hypotheses.hypothesis) == {f"H{i}" for i in range(1, 18)}
    assert hypotheses.status_at_freeze.eq("not_tested").all()


def test_required_gates_are_frozen():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["common_inference"]["bootstrap_replicates"] == 1000
    assert cfg["common_inference"]["permutation_replicates"] == 1000
    assert cfg["positioning_gate"]["minimum_primary_results_passed"] == 2
    assert cfg["positioning_gate"]["external_replication_required"] is True
    assert cfg["dynamic_operator_gate"]["run_operator_models"] is False
    assert cfg["existing_system_roles"]["Replogle_RPE1_CRISPRi"]["untouched_claim_allowed"] is False
