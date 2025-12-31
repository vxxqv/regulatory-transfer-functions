"""Build a curated BibTeX library from frozen Crossref records."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


EXCLUDED = {
    "10.1016/j.cell.2019.05.013": "unrelated cancer history study",
    "10.1186/s13059-024-03182-1": "unrelated RNA modification study",
    "10.1038/s41587-020-0605-1": "not required for the stated analyses",
    "10.1016/j.molcel.2021.11.017": "unrelated metabolic lncRNA study",
    "10.1093/nar/gkae1072": "microRNA resource not used in the workflow",
}


def clean_text(value: str) -> str:
    """Normalize metadata while excluding typographic dash characters."""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u2010", "-").replace("\u2011", "-")
    value = value.replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", ":")
    value = value.replace("\u2212", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def bib_escape(value: str) -> str:
    value = clean_text(value)
    return value.replace("\\", "\\textbackslash{}")


def issued_year(record: dict) -> int:
    parts = record.get("published-print", record.get("published-online", record.get("issued", {})))
    date_parts = parts.get("date-parts", [[0]])
    return int(date_parts[0][0]) if date_parts and date_parts[0] else 0


def first_text(record: dict, field: str, default: str = "") -> str:
    values = record.get(field) or []
    return str(values[0]) if values else default


def citekey(record: dict, used: set[str]) -> str:
    family = clean_text(record.get("author", [{}])[0].get("family", "Anon"))
    stem = re.sub(r"[^A-Za-z0-9]", "", family) or "Anon"
    title_words = re.findall(r"[A-Za-z0-9]+", clean_text(first_text(record, "title", "Study")))
    informative = next((word for word in title_words if len(word) > 3), "Study")
    base = f"{stem}{issued_year(record)}{informative}"
    key = base
    suffix = ord("a")
    while key in used:
        key = base + chr(suffix)
        suffix += 1
    used.add(key)
    return key


def author_field(record: dict) -> str:
    authors = []
    for author in record.get("author", []):
        family = bib_escape(author.get("family", ""))
        given = bib_escape(author.get("given", ""))
        if family:
            authors.append(f"{family}, {given}".strip().rstrip(","))
    return " and ".join(authors)


def build(source: Path, bib_path: Path, audit_path: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = sorted(payload["records"], key=lambda row: row.get("DOI", "").lower())
    used: set[str] = set()
    bib_blocks: list[str] = []
    audit_rows: list[dict[str, str | int]] = []
    for record in records:
        doi = record.get("DOI", "").lower()
        include = doi not in EXCLUDED
        reason = "included after topical and metadata review" if include else EXCLUDED[doi]
        key = ""
        if include:
            key = citekey(record, used)
            fields = {
                "author": author_field(record),
                "title": bib_escape(first_text(record, "title")),
                "journal": bib_escape(first_text(record, "container-title")),
                "year": str(issued_year(record)),
                "volume": bib_escape(str(record.get("volume", ""))),
                "number": bib_escape(str(record.get("issue", ""))),
                "pages": bib_escape(str(record.get("page", record.get("article-number", "")))),
                "doi": doi,
                "url": f"https://doi.org/{doi}",
            }
            rendered = [f"@article{{{key},"]
            rendered.extend(
                f"  {name} = {{{value}}}," for name, value in fields.items() if value
            )
            rendered.append("}")
            bib_blocks.append("\n".join(rendered))
        audit_rows.append(
            {
                "doi": doi,
                "citekey": key,
                "year": issued_year(record),
                "title": clean_text(first_text(record, "title")),
                "decision": "include" if include else "exclude",
                "reason": reason,
            }
        )
    bib_path.parent.mkdir(parents=True, exist_ok=True)
    bib_path.write_text("\n\n".join(bib_blocks) + "\n", encoding="utf-8")
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"included={sum(row['decision'] == 'include' for row in audit_rows)} excluded={len(EXCLUDED)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("work/crossref_verified.json"))
    parser.add_argument("--bib", type=Path, default=Path("references/library.bib"))
    parser.add_argument("--audit", type=Path, default=Path("references/curation.tsv"))
    args = parser.parse_args()
    build(args.source, args.bib, args.audit)


if __name__ == "__main__":
    main()
