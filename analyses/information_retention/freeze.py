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
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".tsv", ".txt", ".yaml", ".yml"}


def sha(path, canonical=False):
    data = path.read_bytes()
    if canonical:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def hash_candidates(path):
    data = path.read_bytes()
    candidates = [data]
    if path.suffix.lower() in TEXT_SUFFIXES:
        lf = data.replace(b"\r\n", b"\n")
        candidates.extend([lf, lf.replace(b"\n", b"\r\n")])
    return {hashlib.sha256(value).hexdigest(): len(value) for value in candidates}


def verify(source):
    expansion = source / "analyses/cell_systems_expansion/freeze_manifest.json"
    manifest = json.loads(expansion.read_text())
    entries = {x["path"]: x for x in manifest["artifacts"]}
    records = []
    for rel in SOURCES:
        expected = entries[rel]["sha256"]
        if expected not in hash_candidates(source / rel):
            raise ValueError(f"Frozen source mismatch: {rel}")
        records.append(entries[rel])
    expansion_candidates = hash_candidates(expansion)
    local_freeze = HERE / "freeze_manifest.json"
    expected_expansion = json.loads(local_freeze.read_text())["expansion_freeze_sha256"] if local_freeze.exists() else sha(expansion)
    if expected_expansion not in expansion_candidates:
        raise ValueError("Frozen expansion manifest mismatch")
    return records, expected_expansion


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
        "source_hash_rule": "Text artifacts accept only raw, LF-canonical or CRLF-canonical byte equivalence; binary artifacts require exact bytes",
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
