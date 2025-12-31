"""Fetch released ENCODE element-quantification tables for overlapping screen targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--target-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/encode_crispri/element_quantifications"))
    return parser.parse_args()


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "regulatory-transfer-functions/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    screens = pd.read_excel(args.workbook, sheet_name="Table 1", header=1)
    targets = pd.read_csv(args.target_summary, usecols=["target_contrast_gene_name"])
    profiled = set(targets["target_contrast_gene_name"].dropna().astype(str))
    screens = screens.loc[
        screens["Readout target"].astype(str).isin(profiled)
        & screens["Accession"].astype(str).str.match(r"^ENCSR")
    ].copy()

    catalog: list[dict[str, object]] = []
    for accession in sorted(screens["Accession"].astype(str).unique()):
        record = get_json(
            f"https://www.encodeproject.org/functional-characterization-experiments/{accession}/?format=json"
        )
        files = [
            item
            for item in record.get("files", [])
            if item.get("status") == "released"
            and item.get("file_format") == "tsv"
            and item.get("output_type") == "element quantifications"
        ]
        for item in files:
            file_accession = item["accession"]
            url = f"https://www.encodeproject.org{item['href']}"
            destination = args.output / f"{file_accession}.tsv"
            if not destination.exists():
                request = urllib.request.Request(
                    url, headers={"User-Agent": "regulatory-transfer-functions/0.1"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    destination.write_bytes(response.read())
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            metadata = screens.loc[screens["Accession"].astype(str) == accession].iloc[0]
            catalog.append(
                {
                    "experiment_accession": accession,
                    "file_accession": file_accession,
                    "biosample": metadata["Biosample"],
                    "readout": metadata["Readout"],
                    "readout_target": metadata["Readout target"],
                    "modality": metadata["Modality"],
                    "assembly": metadata["assembly"],
                    "url": url,
                    "sha256": digest,
                    "bytes": destination.stat().st_size,
                    "local_name": destination.name,
                }
            )
        time.sleep(0.05)
    pd.DataFrame(catalog).sort_values(
        ["readout_target", "experiment_accession", "file_accession"]
    ).to_csv(args.output / "catalog.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
