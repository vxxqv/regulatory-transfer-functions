"""Verify frozen inputs for the metadata-only eligibility audit."""

from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("REGULATORY_SOURCE_ROOT", ROOT))
OUT = ROOT / "analyses/compositionality/metadata"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".tsv", ".txt", ".yaml", ".yml"}
AMENDMENT_PATHS = [
    ROOT / "analyses/pyvista_landscape/render_amendment.json",
    ROOT / "analyses/cell_systems_expansion/reporting_amendment_01.json",
]


def hash_candidates(path):
    data = path.read_bytes()
    candidates = [data]
    if path.suffix.lower() in TEXT_SUFFIXES:
        lf = data.replace(b"\r\n", b"\n")
        candidates.extend([lf, lf.replace(b"\n", b"\r\n")])
    return {hashlib.sha256(value).hexdigest() for value in candidates}


def verify_inputs():
    manifest = json.loads((SOURCE / "analyses/cell_systems_expansion/freeze_manifest.json").read_text())
    amendments = {
        amendment["path"]: amendment
        for amendment_path in AMENDMENT_PATHS
        for amendment in [json.loads(amendment_path.read_text())]
    }
    rows = []
    for item in manifest["artifacts"]:
        path = SOURCE / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        candidates = hash_candidates(path)
        verified = item["sha256"] in candidates
        amendment_id = None
        if not verified and item["path"] in amendments:
            amendment = amendments[item["path"]]
            verified = (
                amendment["parent_sha256"] == item["sha256"]
                and amendment["current_sha256"] in candidates
                and amendment["analysis_values_changed"] is False
            )
            amendment_id = amendment["amendment_id"] if verified else None
        rows.append({**item, "actual_sha256": digest, "verified": verified, "amendment_id": amendment_id})
    assert all(row["verified"] for row in rows), "Frozen input hash mismatch"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_verification.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    print("Frozen inputs verified:", len(verify_inputs()))
