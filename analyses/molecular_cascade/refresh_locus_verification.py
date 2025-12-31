"""Refresh the locus-only molecular-cascade integration after audited gate correction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "analyses/molecular_cascade/results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "molecular_cascade", ROOT / "analyses/molecular_cascade/run_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = module.verify_freeze()
    targets = pd.read_parquet(RESULTS / "target_state_evidence.parquet")
    table = module.locus_verification(targets)
    table.to_csv(RESULTS / "locus_regulatory_verification.csv", index=False)
    audit = {
        "correction_id": "CT-001",
        "rows": len(table),
        "all_loci_present": set(table["locus"]) == {"gata3", "stat3", "ptpn22"},
        "evidence_gates_sha256": sha256(
            ROOT / "analyses/causal_triangulation/results/evidence_gates.csv"
        ),
        "locus_grades_sha256": sha256(
            ROOT / "analyses/causal_triangulation/results/locus_grades.csv"
        ),
        "target_state_evidence_sha256": sha256(RESULTS / "target_state_evidence.parquet"),
        "output_sha256": sha256(RESULTS / "locus_regulatory_verification.csv"),
        "freeze_checks": checks,
    }
    (RESULTS / "locus_verification_refresh_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
