"""Download and verify the frozen primary-CD4 H3K27ac peak file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


URL = "https://www.encodeproject.org/files/ENCFF068XUG/@@download/ENCFF068XUG.bed.gz"
EXPECTED_MD5 = "17d4fe5b096669738a3d4a1c8d6a971f"
EXPECTED_SHA256 = "d4c003434581b8e1117c06477b484b53b8798e55daf636ba38da594da675f905"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work/causal_inputs/ENCFF068XUG.bed.gz"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL) as response, args.output.open("wb") as destination:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            destination.write(block)
    if digest(args.output, "md5") != EXPECTED_MD5:
        raise ValueError("ENCODE MD5 checksum mismatch")
    if digest(args.output, "sha256") != EXPECTED_SHA256:
        raise ValueError("ENCODE SHA256 checksum mismatch")


if __name__ == "__main__":
    main()
