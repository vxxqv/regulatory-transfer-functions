from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/cell_systems_expansion/strengthening_freeze_manifest.json"
INPUTS = [
    "config/strengthening_extensions.yaml",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/primary/results/primary_results.json",
    "analyses/multiverse/results/scalar_family_summary.csv",
    "analyses/multiverse/results/rerouting_specifications.parquet",
    "analyses/multiverse/results/validation_family_summary.csv",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/vectors/results/network_gain_rows.parquet",
    "analyses/molecular_cascade/results/target_state_evidence.parquet",
    "analyses/molecular_cascade/results/curated_regulatory_edges.parquet",
    "analyses/guide_concordance/results/audit.json",
    "analyses/state_response_decomposition/results/audit.json",
    "analyses/external_benchmark/results/benchmark_results.json",
    "analyses/transportability/results/system_availability.csv",
    "analyses/grn_benchmark/results/all_candidate_edges.parquet",
    "analyses/grn_benchmark/results/method_results.csv",
    "analyses/natural_genetics/results_audit_corrected/natural_genetics_results.json",
    "analyses/disease/results/disease_results.json",
    "analyses/causal_triangulation/results/causal_triangulation_results.json",
    "analyses/independent_confirmation/results/hypothesis_decisions.tsv",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    records = []
    for relative in INPUTS:
        path = ROOT / relative
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    payload = {
        "version": 1,
        "status": "frozen before targeted strengthening outcomes",
        "outcomes_examined_before_freeze": False,
        "new_computations": [
            "hierarchical transfer stability",
            "CLR Aitchison rerouting sensitivity",
            "paired GRN contrasts and calibration",
        ],
        "stopped_families": [
            "guide dose and concordance",
            "cross-system conservation",
            "external transportability",
            "molecular regulatory support",
            "natural-genetic concordance",
            "disease and locus convergence",
        ],
        "inputs": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
