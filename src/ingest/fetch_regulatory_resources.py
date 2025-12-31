"""Retrieve the frozen regulatory resources and verify their checksums."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "work/regulatory_inputs"
RESOURCES = {
    "ENCFF858TLX.bed.gz": (
        "https://www.encodeproject.org/files/ENCFF858TLX/@@download/ENCFF858TLX.bed.gz",
        "c8a9d42a44feab05e4b5a634a5bbc75746fe14ac4c08b9080f000ce3a19435ce",
    ),
    "omnipath_tf_target.tsv": (
        "https://omnipathdb.org/interactions?format=tsv&datasets=tf_target&organisms=9606&fields=sources,references&genesymbols=1",
        "e252ce9460fa2b2ad0c73a15c55158183f50698c8d1da27f3cb102e7bfb6cf5d",
    ),
    "hg19ToHg38.over.chain.gz": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
        "5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d",
    ),
}
BIOMART_QUERY = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Attribute name="ensembl_gene_id"/><Attribute name="external_gene_name"/>
    <Attribute name="chromosome_name"/><Attribute name="start_position"/>
    <Attribute name="end_position"/><Attribute name="strand"/>
    <Attribute name="percentage_gene_gc_content"/>
  </Dataset>
</Query>"""
BIOMART_SHA256 = "2f7ab71c084f4ada92fd81bab4c8adf85d65f854c566917e2fe46fc6e4520e3a"


def retrieve(url: str, destination: Path, expected: str) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": "regulatory-transfer-functions/1.0"})
    digest = hashlib.sha256()
    with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
    if digest.hexdigest() != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {destination.name}")
    os.replace(temporary, destination)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected) in RESOURCES.items():
        retrieve(url, DESTINATION / filename, expected)
    query_url = "https://www.ensembl.org/biomart/martservice?" + urlencode({"query": BIOMART_QUERY})
    retrieve(query_url, DESTINATION / "ensembl_gene_attributes.tsv", BIOMART_SHA256)


if __name__ == "__main__":
    main()
