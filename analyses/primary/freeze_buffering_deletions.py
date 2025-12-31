from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "buffering_deletion_extension.yaml"
INPUT = ROOT / "analyses" / "primary" / "results" / "transfer_phenotypes.parquet"
OUTPUT = ROOT / "analyses" / "primary" / "results" / "buffering_deletion_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    input_hash = sha256(INPUT)
    if input_hash != config["input_sha256"]:
        raise RuntimeError("Transfer phenotype input hash does not match the frozen configuration")
    payload = {
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG),
        "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": input_hash,
        "seed": int(config["seed"]),
        "bootstrap_replicates": 2000,
        "permutation_replicates": 2000,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
