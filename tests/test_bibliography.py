from __future__ import annotations

import csv
from pathlib import Path


def test_curated_bibliography_is_unique_and_dash_clean() -> None:
    audit_path = Path("references/curation.tsv")
    bib_path = Path("references/library.bib")
    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    included = [row for row in rows if row["decision"] == "include"]
    dois = [row["doi"] for row in included]
    citekeys = [row["citekey"] for row in included]
    assert 40 <= len(included) <= 70
    assert len(dois) == len(set(dois))
    assert len(citekeys) == len(set(citekeys))
    text = bib_path.read_text(encoding="utf-8")
    assert "\u2013" not in text
    assert "\u2014" not in text
