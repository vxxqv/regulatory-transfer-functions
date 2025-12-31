"""Refresh causal-tier annotations in the disease-convergence edge table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/visual_depth/results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    phenotypes = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    locus_summary = pd.read_csv(ROOT / "analyses/loci/results/locus_summary.csv")
    grades_path = ROOT / "analyses/causal_triangulation/results/locus_grades.csv"
    grades = pd.read_csv(grades_path)
    assignments = phenotypes[
        phenotypes.target_contrast_gene_name.isin(locus_summary.gene)
    ][
        [
            "target_contrast_gene_name",
            "culture_condition",
            "cluster",
            "transfer_class",
            "transfer_z",
        ]
    ].drop_duplicates()
    convergence = pd.read_csv(ROOT / "analyses/disease/results/convergence_edges.csv")
    rows: list[dict[str, object]] = []
    for row in locus_summary.itertuples(index=False):
        tier = grades.loc[grades.locus == row.locus, "evidence_tier"].iloc[0]
        rows.append(
            {
                "source": row.lead_rsid,
                "target": row.gene,
                "source_type": "variant",
                "target_type": "gene",
                "status": "available",
                "weight": row.lead_posterior_probability,
                "evidence_tier": tier,
                "condition": "locus",
            }
        )
        local = assignments[
            (assignments.target_contrast_gene_name == row.gene)
            & assignments.cluster.notna()
        ]
        if local.empty:
            rows.append(
                {
                    "source": row.gene,
                    "target": "program unavailable",
                    "source_type": "gene",
                    "target_type": "program",
                    "status": "unavailable",
                    "weight": 0,
                    "evidence_tier": tier,
                    "condition": "all",
                }
            )
        for item in local.itertuples(index=False):
            program = f"Program {int(item.cluster)}"
            rows.append(
                {
                    "source": row.gene,
                    "target": program,
                    "source_type": "gene",
                    "target_type": "program",
                    "status": "available",
                    "weight": abs(item.transfer_z),
                    "evidence_tier": tier,
                    "condition": item.culture_condition,
                }
            )
            disease = convergence[
                (convergence.cluster == item.cluster)
                & (convergence.culture_condition == item.culture_condition)
            ]
            if disease.empty:
                rows.append(
                    {
                        "source": program,
                        "target": "disease link unavailable",
                        "source_type": "program",
                        "target_type": "disease",
                        "status": "unavailable",
                        "weight": 0,
                        "evidence_tier": tier,
                        "condition": item.culture_condition,
                    }
                )
            for link in disease.itertuples(index=False):
                rows.append(
                    {
                        "source": program,
                        "target": link.disease,
                        "source_type": "program",
                        "target_type": "disease",
                        "status": "available",
                        "weight": -np.log10(link.p_adj_fdr),
                        "evidence_tier": tier,
                        "condition": item.culture_condition,
                    }
                )
    table = pd.DataFrame(rows).drop_duplicates()
    destination = OUTPUT / "circos_edges.csv"
    table.to_csv(destination, index=False)
    audit = {
        "correction_id": "CT-001",
        "rows": len(table),
        "locus_grades_sha256": sha256(grades_path),
        "output_sha256": sha256(destination),
        "tier_by_locus": grades.set_index("locus")["evidence_tier"].to_dict(),
    }
    (OUTPUT / "circos_refresh_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
