"""Download the frozen Enrichr TRANSFAC and JASPAR PWM gene-set library."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=TRANSFAC_and_JASPAR_PWMs"
EXPECTED_SHA256 = "55424badcff1aec60889d8be5ef98e2698c9e85164373157a72b22ec3e6dd68e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work/motif/TRANSFAC_and_JASPAR_PWMs.gmt"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, args.output)
    observed = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if observed != EXPECTED_SHA256:
        raise ValueError(f"Checksum mismatch: {observed}")


if __name__ == "__main__":
    main()
