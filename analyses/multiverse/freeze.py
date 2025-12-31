"""Record the exact multiverse rules and upstream results present at freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "config/multiverse.yaml",
    "docs/protocol.md",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/primary/results/context_switching.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/vectors/results/network_gain_rows.parquet",
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


def main() -> None:
    missing = [relative for relative in FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing artifacts: {missing}")
    output = ROOT / "analyses/multiverse"
    manifest = {
        "freeze_date": "2026-09-12",
        "status": "post_primary_grid_frozen_before_multiverse_execution",
        "primary_outcomes_existed_before_freeze": True,
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
