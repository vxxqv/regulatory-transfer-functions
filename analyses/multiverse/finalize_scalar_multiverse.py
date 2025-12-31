"""Combine scalar multiverse shards and apply family-wise multiplicity correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.multiverse.run_scalar_multiverse import ROOT, adjust_q


OUTPUT = ROOT / "analyses/multiverse/results"


def main() -> None:
    parts = sorted(OUTPUT.glob("scalar_specifications_part*of*.parquet"))
    if not parts:
        raise FileNotFoundError("No scalar multiverse shards found")
    table = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    unavailable_parts = sorted(OUTPUT.glob("scalar_unavailable_part*of*.csv"))
    if not unavailable_parts:
        raise FileNotFoundError("No scalar unavailable-specification shards found")
    unavailable = pd.concat([pd.read_csv(path) for path in unavailable_parts], ignore_index=True)
    result_ids = set(table["specification_id"].astype(int))
    unavailable_ids = set(unavailable["specification_id"].astype(int))
    expected_ids = set(range(1, 4321))
    if result_ids & unavailable_ids:
        raise ValueError("A scalar base specification appears in both result and unavailable tables")
    if result_ids | unavailable_ids != expected_ids:
        raise ValueError("The combined scalar multiverse does not account for all 4,320 base specifications")
    table = adjust_q(table.drop(columns=["q_value", "supported", "direction_expected"], errors="ignore"))
    table["direction_expected"] = np.where(
        table["expected_direction"] == "positive", table["estimate"] > 0, table["estimate"] < 0
    )
    table["supported"] = table["direction_expected"] & (table["q_value"] < 0.05)
    table.to_parquet(OUTPUT / "scalar_specifications.parquet", index=False)
    unavailable.to_csv(OUTPUT / "scalar_unavailable.csv", index=False)
    family = table.groupby("result_family").agg(
        specifications=("specification_id", "size"),
        expected_direction_fraction=("direction_expected", "mean"),
        q_supported_fraction=("supported", "mean"),
        median_estimate=("estimate", "median"),
        minimum_estimate=("estimate", "min"),
        maximum_estimate=("estimate", "max"),
    ).reset_index()
    family["robustness_label"] = np.select(
        [
            (family["expected_direction_fraction"] >= 0.80) & (family["q_supported_fraction"] >= 0.50),
            family["expected_direction_fraction"] >= 0.60,
        ],
        ["broad", "mixed"],
        default="narrow",
    )
    family.to_csv(OUTPUT / "scalar_family_summary.csv", index=False)
    audit = {
        "base_specifications": 4320,
        "base_specifications_with_results": len(result_ids),
        "base_specifications_unavailable": len(unavailable_ids),
        "result_rows": len(table),
        "unavailable_base_specifications": len(unavailable),
        "all_shards_combined": True,
    }
    (OUTPUT / "scalar_multiverse_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
