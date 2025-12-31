"""Combine bootstrap and structural-null decisions without rerunning models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/external_benchmark/results"


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def main() -> None:
    hypotheses = pd.read_csv(OUTPUT / "hypothesis_decisions.csv")
    nulls = pd.read_csv(OUTPUT / "null_summary.csv")
    nulls["q_value"] = bh(nulls["one_sided_p"].to_numpy(dtype=float))
    nulls.to_csv(OUTPUT / "null_summary.csv", index=False)
    matched = nulls.loc[nulls["null"] == "matched_cd4_feature_permutation"].iloc[0]
    rewired = nulls.loc[nulls["null"] == "directed_degree_preserving_rewiring"].iloc[0]
    additional = pd.DataFrame(
        [
            {
                "hypothesis": "Frozen CD4 feature identity exceeds matched target-label permutations",
                "primary_metric": "delta_r2_over_simple_covariates",
                "estimate": matched["observed"],
                "ci_low": matched["null_95_low"],
                "ci_high": matched["null_95_high"],
                "minimally_relevant_effect": 0.0,
                "decision": "passed" if matched["q_value"] < 0.05 and matched["observed"] > matched["null_median"] else "unresolved",
            },
            {
                "hypothesis": "Observed CD4 module-neighbor topology exceeds degree-preserving rewiring",
                "primary_metric": "graph_r2",
                "estimate": rewired["observed"],
                "ci_low": rewired["null_95_low"],
                "ci_high": rewired["null_95_high"],
                "minimally_relevant_effect": 0.0,
                "decision": "passed" if rewired["q_value"] < 0.05 and rewired["observed"] > rewired["null_95_high"] else "unresolved",
            },
        ]
    )
    hypotheses = pd.concat([hypotheses.iloc[:1], additional], ignore_index=True)
    hypotheses.to_csv(OUTPUT / "hypothesis_decisions.csv", index=False)
    audit_path = OUTPUT / "benchmark_results.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["pending_null_analyses"] = []
    audit["hypothesis_decision_counts"] = hypotheses["decision"].value_counts().to_dict()
    audit["null_replicates_complete"] = {
        "matched_permutations": int(matched["replicates"]),
        "degree_preserving_rewirings": int(rewired["replicates"]),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
