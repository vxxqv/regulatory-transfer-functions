"""Freeze source artifacts and rules for Supplementary Figure S24."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "config/pyvista_transfer_landscape.yaml",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/context_dynamics/results/switch_scores.parquet",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing artifacts: {missing}")
    config = yaml.safe_load((ROOT / FILES[0]).read_text(encoding="utf-8"))
    transfer = pd.read_parquet(ROOT / FILES[1], columns=["target_contrast", "culture_condition"])
    required = set(config["states"]["order"])
    observed = transfer.groupby("target_contrast")["culture_condition"].agg(lambda values: set(values))
    complete = observed[observed.map(lambda values: values == required)].index
    expected = int(config["cohort"]["expected_complete_targets"])
    if len(complete) != expected:
        raise ValueError(f"Expected {expected} complete-state targets, observed {len(complete)}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "freeze_date": "2026-09-13",
        "status": "retrospective_figure_rules_frozen_before_s24_specific_derivations",
        "not_a_prospective_preregistration": True,
        "complete_targets": int(len(complete)),
        "complete_target_state_rows": int(len(complete) * len(required)),
        "states": config["states"]["order"],
        "git_commit_before_freeze_commit": commit,
        "artifacts": [
            {
                "path": name,
                "bytes": (ROOT / name).stat().st_size,
                "sha256": sha256(ROOT / name),
            }
            for name in FILES
        ],
    }
    output = ROOT / "analyses/pyvista_landscape/freeze_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
