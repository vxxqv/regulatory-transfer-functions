"""Verify frozen inputs for the metadata-only eligibility audit."""

from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("REGULATORY_SOURCE_ROOT", ROOT))
OUT = ROOT / "analyses/compositionality/metadata"


def verify_inputs():
    manifest = json.loads((SOURCE / "analyses/cell_systems_expansion/freeze_manifest.json").read_text())
    rows = []
    for item in manifest["artifacts"]:
        path = SOURCE / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({**item, "actual_sha256": digest, "verified": digest == item["sha256"]})
    assert all(row["verified"] for row in rows), "Frozen input hash mismatch"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_verification.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    print("Frozen inputs verified:", len(verify_inputs()))
