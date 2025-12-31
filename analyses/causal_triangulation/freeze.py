"""Freeze the locus denominator and causal-triangulation rules before retrieval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "config/causal_triangulation.yaml",
    "docs/protocol.md",
    "analyses/loci/results/locus_summary.csv",
    "analyses/loci/results/credible_set_variants.csv",
    "analyses/loci/results/l2g_features.csv",
    "analyses/loci/results/transfer_programs.csv",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/natural_genetics/results/directional_pairs.parquet",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [relative for relative in FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing artifacts: {missing}")
    loci = pd.read_csv(ROOT / "analyses/loci/results/locus_summary.csv")
    expected = {"gata3", "stat3", "ptpn22"}
    observed = set(loci["locus"].astype(str))
    if observed != expected:
        raise ValueError(f"Frozen denominator mismatch: expected {expected}, observed {observed}")
    output = ROOT / "analyses/causal_triangulation"
    manifest = {
        "freeze_date": "2026-09-12",
        "status": "frozen_before_triangulation_retrieval",
        "eligible_locus_denominator": len(loci),
        "eligible_loci": loci[
            ["locus", "gene", "trait", "study_locus_id", "lead_rsid"]
        ].to_dict(orient="records"),
        "artifacts": [
            {
                "path": relative.replace("\\", "/"),
                "bytes": (ROOT / relative).stat().st_size,
                "sha256": sha256(ROOT / relative),
            }
            for relative in FILES
        ],
    }
    (output / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
