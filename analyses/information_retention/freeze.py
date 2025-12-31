"""Verify the expansion inputs before computing new diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "analyses/information_retention"
SOURCES = [
    "config/analysis.yaml", "config/cell_systems_expansion.yaml",
    "analyses/cell_systems_expansion/target_folds.tsv",
    "analyses/cell_systems_expansion/hypotheses.tsv",
    "data/interim/gwt_vectors/rows.parquet", "data/interim/gwt_vectors/genes.parquet",
    "data/interim/gwt_vectors/normalized_significant_logfc.npz",
    "data/interim/gwt_vectors/audit.json",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/vectors/results/reference_module_loadings.parquet",
    "analyses/state_response_decomposition/results/target_decomposition.parquet",
    "analyses/guide_concordance/results/eligible_pairs.parquet",
    "analyses/guide_concordance/results/hypothesis_tests.csv",
    "analyses/replication/results/k562_replication_rows.parquet",
]


def sha(path, canonical=False):
    data = path.read_bytes()
    if canonical:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify(source):
    expansion = source / "analyses/cell_systems_expansion/freeze_manifest.json"
    manifest = json.loads(expansion.read_text())
    entries = {x["path"]: x for x in manifest["artifacts"]}
    records = []
    for rel in SOURCES:
        expected = entries[rel]["sha256"]
        actual = sha(source / rel)
        if actual != expected:
            raise ValueError(f"Frozen source mismatch: {rel}")
        records.append({"path": rel, "sha256": actual, "bytes": (source / rel).stat().st_size})
    return records, sha(expansion)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    args = p.parse_args()
    records, expansion_hash = verify(args.source_root)
    manifest = {
        "status": "frozen_before_conditional_decoding_outcomes",
        "seed": 20260913,
        "expansion_freeze_sha256": expansion_hash,
        "canonical_specification_sha256": sha(HERE / "specification.json", True),
        "source_hash_rule": "Exact source bytes must match the parent expansion freeze; no newline normalization of source artifacts",
        "artifacts": records,
        "previously_known_availability": {
            "guide_and_donor_profile_vectors": False,
            "external_profile_vectors": False,
            "library_size_direction": False,
            "generic_stress_template": False,
            "proliferation_template": False,
            "independent_regulator_family_labels": False,
            "independent_functional_module_labels": False,
            "independent_network_degree_annotation": False,
            "twenty_and_forty_module_loadings": False
        }
    }
    dest = HERE / "freeze_manifest.json"
    if dest.exists() and json.loads(dest.read_text()) != manifest:
        raise ValueError("Existing information freeze differs; do not overwrite it")
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"verified_inputs": len(records), "specification_sha256": manifest["canonical_specification_sha256"]}))


if __name__ == "__main__":
    main()
