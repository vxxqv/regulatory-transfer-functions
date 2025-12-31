"""Freeze inputs and target folds before decomposition."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "data/interim/gwt_vectors/rows.parquet",
    "data/interim/gwt_vectors/genes.parquet",
    "data/interim/gwt_vectors/normalized_significant_logfc.npz",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/primary/results/context_switching.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/vectors/results/network_gain_rows.parquet",
    "analyses/vectors/results/reference_module_loadings.parquet",
    "analyses/context_dynamics/results/tf_activity_association.parquet",
    "analyses/context_dynamics/results/independent_state_validation.csv",
    "analyses/replication/results/k562_replication_rows.parquet",
    "analyses/external_benchmark/results/all_eligible_target_predictions.parquet",
    "analyses/guide_dose_response/results/eligible_guide_rows.csv",
]


def digest(path):
    if path.suffix == ".yaml":
        return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    folder = ROOT / "analyses/state_response_decomposition"
    config_path = ROOT / "config/state_response_decomposition.yaml"
    cfg = yaml.safe_load(config_path.read_text())["state_response_decomposition"]
    rows = pd.read_parquet(args.source_root / FILES[0])
    groups = rows.groupby("target_contrast")["culture_condition"].agg(list)
    targets = sorted(groups[groups.map(lambda values: sorted(values) == sorted(cfg["states"]))].index)
    assert len(targets) == cfg["expected_targets"]
    rng = np.random.default_rng(cfg["seed"])
    fold = np.empty(len(targets), dtype=int)
    fold[rng.permutation(len(targets))] = np.arange(len(targets)) % cfg["folds"]
    cohort = pd.DataFrame({"target_contrast": targets, "fold": fold})
    cohort.to_csv(folder / "frozen_cohort.tsv", sep="\t", index=False, lineterminator="\n")
    manifest = {
        "status": "retrospective extension frozen before new decomposition outcomes",
        "targets": len(targets),
        "config_sha256": digest(config_path),
        "original_crlf_cohort_sha256": hashlib.sha256((folder / "frozen_cohort.tsv").read_text(encoding="utf-8").replace("\n", "\r\n").encode()).hexdigest(),
        "canonical_cohort_sha256": hashlib.sha256((folder / "frozen_cohort.tsv").read_text(encoding="utf-8").encode()).hexdigest(),
        "sources": {name: digest(args.source_root / name) for name in FILES},
        "outcomes_examined_before_freeze": False,
    }
    (folder / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"targets": len(targets), "sources": len(FILES), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
