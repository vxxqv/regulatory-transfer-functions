"""Acquire and verify the published CD4 CHiCAGO interaction table."""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "work/cd4_pchic_interactions.tsv.gz"
URL = "https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs13059-017-1285-0/MediaObjects/13059_2017_1285_MOESM5_ESM.gz"
EXPECTED_SHA256 = "23eb3205b2d45922cb511f698ba490de686df9e57c3a81f60568c865752ff857"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(DESTINATION.suffix + ".partial")
    digest = hashlib.sha256()
    with urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("CD4 CHiCAGO source checksum mismatch")
    with gzip.open(temporary, "rt", encoding="utf-8") as handle:
        header = handle.readline()
    if "Total_CD4_Activated" not in header or "Total_CD4_NonActivated" not in header:
        temporary.unlink(missing_ok=True)
        raise ValueError("CD4 CHiCAGO source schema mismatch")
    os.replace(temporary, DESTINATION)


if __name__ == "__main__":
    main()
