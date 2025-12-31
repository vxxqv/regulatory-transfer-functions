"""Freeze CD4 artifacts, external-challenge rules, and deterministic target folds."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "config/analysis.yaml",
    "config/external_benchmark.yaml",
    "config/multiverse.yaml",
    "docs/protocol.md",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/primary/results/context_switching.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/network_gain_rows.parquet",
    "analyses/vectors/results/reference_module_loadings.parquet",
    "analyses/baselines/results/oof_predictions.parquet",
    "analyses/replication/results/k562_replication_rows.parquet",
    "analyses/natural_genetics/results/directional_pairs.parquet",
    "analyses/disease/results/disease_model_rows.parquet",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_for_target(target: str, folds: int = 10) -> int:
    value = int(hashlib.sha256(target.encode("utf-8")).hexdigest()[:16], 16)
    return value % folds + 1


def main() -> None:
    missing = [relative for relative in FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing artifacts: {missing}")

    phenotype_path = ROOT / "analyses/primary/results/transfer_phenotypes.parquet"
    phenotypes = pd.read_parquet(phenotype_path, columns=["target_contrast_gene_name"])
    targets = sorted(phenotypes["target_contrast_gene_name"].dropna().astype(str).unique())
    assignments = pd.DataFrame(
        {"target_gene": targets, "fold": [fold_for_target(target) for target in targets]}
    )
    output = ROOT / "analyses/external_benchmark"
    assignments.to_csv(output / "target_folds.tsv", sep="\t", index=False)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    manifest = {
        "freeze_date": "2026-09-12",
        "status": "frozen_before_rpe1_holdout_acquisition",
        "development_replication_status": "k562_outcomes_previously_inspected",
        "falsification_holdout": "Replogle RPE1 CRISPRi Perturb-seq",
        "holdout_outcomes_present_at_freeze": False,
        "fold_rule": "1 + int(first_16_hex_sha256(target_gene), 16) modulo 10",
        "folds": 10,
        "targets": len(targets),
        "git_commit_before_freeze_commit": commit,
        "artifacts": [
            {
                "path": relative.replace("\\", "/"),
                "bytes": (ROOT / relative).stat().st_size,
                "sha256": sha256(ROOT / relative),
            }
            for relative in FILES
        ],
        "target_folds": {
            "path": "analyses/external_benchmark/target_folds.tsv",
            "bytes": (output / "target_folds.tsv").stat().st_size,
            "sha256": sha256(output / "target_folds.tsv"),
        },
    }
    (output / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
