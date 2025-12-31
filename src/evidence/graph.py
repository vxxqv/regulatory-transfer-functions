"""Typed, provenance-bearing evidence graph for regulatory transfer analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


NODE_TYPES = {
    "variant",
    "locus",
    "element",
    "perturbation",
    "gene",
    "state",
    "module",
    "trait",
    "study",
}

RELATION_SIGNATURES = {
    "contains": {("locus", "variant"), ("locus", "element"), ("module", "gene")},
    "targets": {("perturbation", "gene"), ("perturbation", "element")},
    "regulates": {("element", "gene"), ("gene", "gene")},
    "responds_in": {("gene", "state"), ("module", "state")},
    "loads_on": {("gene", "module")},
    "associates_with": {("variant", "gene"), ("variant", "trait"), ("gene", "trait")},
    "supported_by": {
        ("variant", "study"),
        ("element", "study"),
        ("perturbation", "study"),
        ("gene", "study"),
        ("trait", "study"),
    },
}


@dataclass(frozen=True)
class EvidenceGraph:
    nodes: pd.DataFrame
    edges: pd.DataFrame

    def validate(self) -> None:
        required_nodes = {"node_id", "node_type", "label", "provenance"}
        required_edges = {
            "source",
            "target",
            "relation",
            "evidence_tier",
            "score",
            "uncertainty",
            "state",
            "provenance",
        }
        if not required_nodes.issubset(self.nodes.columns):
            raise ValueError(f"Node fields missing: {sorted(required_nodes - set(self.nodes.columns))}")
        if not required_edges.issubset(self.edges.columns):
            raise ValueError(f"Edge fields missing: {sorted(required_edges - set(self.edges.columns))}")
        if self.nodes["node_id"].duplicated().any():
            raise ValueError("Node identifiers must be unique")
        unknown_types = set(self.nodes["node_type"]) - NODE_TYPES
        if unknown_types:
            raise ValueError(f"Unknown node types: {sorted(unknown_types)}")
        node_types = self.nodes.set_index("node_id")["node_type"].to_dict()
        missing = (set(self.edges["source"]) | set(self.edges["target"])) - set(node_types)
        if missing:
            raise ValueError(f"Edges reference absent nodes: {sorted(missing)[:5]}")
        for edge in self.edges.itertuples(index=False):
            if edge.relation not in RELATION_SIGNATURES:
                raise ValueError(f"Unknown relation: {edge.relation}")
            signature = (node_types[edge.source], node_types[edge.target])
            if signature not in RELATION_SIGNATURES[edge.relation]:
                raise ValueError(f"Invalid signature for {edge.relation}: {signature}")
        if not self.edges["evidence_tier"].isin({"D0", "D1", "D2", "D3"}).all():
            raise ValueError("Evidence tiers must be D0 through D3")

    def write(self, directory: Path) -> None:
        self.validate()
        directory.mkdir(parents=True, exist_ok=True)
        self.nodes.sort_values(["node_type", "node_id"]).to_parquet(directory / "nodes.parquet", index=False)
        self.edges.sort_values(["relation", "source", "target", "state"]).to_parquet(
            directory / "edges.parquet", index=False
        )
        self.nodes.sort_values(["node_type", "node_id"]).to_csv(directory / "nodes.tsv", sep="\t", index=False)
        self.edges.sort_values(["relation", "source", "target", "state"]).to_csv(
            directory / "edges.tsv", sep="\t", index=False
        )
