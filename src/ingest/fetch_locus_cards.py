"""Fetch the frozen Open Targets credible sets used by the three locus cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


STUDY_LOCI = {
    "gata3": "08084ba2089ac9106beddbae944e4e56",
    "stat3": "374aec5062abfe605b8a0d984c4efd1e",
    "ptpn22": "b00610bcfd2485c4decb85fac85fbcdd",
}

FIELDS = """
studyLocusId studyId confidence finemappingMethod
locus(page: {index: 0, size: 100}) {
  count rows {
    is95CredibleSet is99CredibleSet posteriorProbability logBF
    variant { id rsIds chromosome position referenceAllele alternateAllele }
  }
}
l2GPredictions(page: {index: 0, size: 100}) {
  count rows {
    score target { id approvedSymbol }
    features { name value shapValue }
  }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work/opentargets_locus_cards.json"))
    args = parser.parse_args()
    blocks = " ".join(
        f'{key}: credibleSet(studyLocusId: "{identifier}") {{ {FIELDS} }}'
        for key, identifier in STUDY_LOCI.items()
    )
    query = f"query {{ {blocks} }}"
    response = requests.post(
        "https://api.platform.opentargets.org/api/v4/graphql",
        json={"query": query},
        timeout=60,
        headers={"User-Agent": "RegulatoryTransferFunctions/0.1"},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print({key: payload["data"][key]["locus"]["count"] for key in STUDY_LOCI})


if __name__ == "__main__":
    main()
