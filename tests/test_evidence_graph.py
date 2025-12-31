from __future__ import annotations

import pandas as pd

from src.evidence.graph import EvidenceGraph


def test_typed_evidence_graph_accepts_valid_signatures() -> None:
    nodes = pd.DataFrame(
        [
            {"node_id": "perturbation:STAT1", "node_type": "perturbation", "label": "STAT1", "provenance": "study"},
            {"node_id": "gene:STAT1", "node_type": "gene", "label": "STAT1", "provenance": "GRCh38"},
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "source": "perturbation:STAT1",
                "target": "gene:STAT1",
                "relation": "targets",
                "evidence_tier": "D0",
                "score": -1.2,
                "uncertainty": 0.1,
                "state": "resting",
                "provenance": "study",
            }
        ]
    )
    EvidenceGraph(nodes, edges).validate()
