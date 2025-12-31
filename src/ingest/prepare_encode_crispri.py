"""Normalize released ENCODE element-quantification tables into one schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "chrPerturbationTarget",
    "startPerturbationTarget",
    "endPerturbationTarget",
    "EffectSize",
    "measuredGeneSymbol",
    "Significant",
    "ValidConnection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("data/raw/encode_crispri/element_quantifications")
    )
    parser.add_argument("--output", type=Path, default=Path("data/interim/encode_crispri"))
    return parser.parse_args()


def as_boolean(values: pd.Series) -> pd.Series:
    return values.astype("string").str.lower().eq("true").fillna(False)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(args.source / "catalog.tsv", sep="\t")
    tables: list[pd.DataFrame] = []
    rejected: list[dict[str, str]] = []
    for item in catalog.itertuples(index=False):
        path = args.source / item.local_name
        table = pd.read_csv(path, sep="\t", low_memory=False)
        missing = REQUIRED.difference(table.columns)
        if missing:
            rejected.append({"file_accession": item.file_accession, "reason": ",".join(sorted(missing))})
            continue
        table = table.copy()
        table["experiment_accession"] = item.experiment_accession
        table["file_accession"] = item.file_accession
        table["biosample"] = item.biosample
        table["readout_target"] = item.readout_target
        table["assembly"] = item.assembly
        table["significant"] = as_boolean(table["Significant"])
        table["valid_connection"] = as_boolean(table["ValidConnection"])
        table["effect_size"] = pd.to_numeric(table["EffectSize"], errors="coerce")
        for optional in ("measuredEnsemblID", "pValue", "pValueAdjusted"):
            if optional not in table.columns:
                table[optional] = np.nan
        if "startTSS" in table.columns:
            table["distance_to_tss"] = np.abs(
                pd.to_numeric(table["startPerturbationTarget"], errors="coerce")
                - pd.to_numeric(table["startTSS"], errors="coerce")
            )
        else:
            table["distance_to_tss"] = np.nan
        table["element_id"] = (
            table["chrPerturbationTarget"].astype(str)
            + ":"
            + table["startPerturbationTarget"].astype(str)
            + "-"
            + table["endPerturbationTarget"].astype(str)
        )
        tables.append(
            table[
                [
                    "experiment_accession",
                    "file_accession",
                    "biosample",
                    "assembly",
                    "readout_target",
                    "measuredGeneSymbol",
                    "measuredEnsemblID",
                    "element_id",
                    "effect_size",
                    "distance_to_tss",
                    "significant",
                    "valid_connection",
                    "pValue",
                    "pValueAdjusted",
                ]
            ]
        )
    if not tables:
        raise ValueError("No ENCODE element tables passed the required schema")
    elements = pd.concat(tables, ignore_index=True)
    valid = elements.loc[elements["valid_connection"] & elements["effect_size"].notna()].copy()
    experiment = (
        valid.groupby(
            ["experiment_accession", "file_accession", "biosample", "assembly", "readout_target"],
            as_index=False,
        )
        .agg(
            elements_tested=("element_id", "nunique"),
            significant_elements=("significant", "sum"),
            median_absolute_effect=("effect_size", lambda x: float(np.median(np.abs(x)))),
            maximum_absolute_effect=("effect_size", lambda x: float(np.max(np.abs(x)))),
            median_distance_to_tss=("distance_to_tss", "median"),
        )
    )
    experiment["significant_fraction"] = (
        experiment["significant_elements"] / experiment["elements_tested"]
    )
    target = (
        experiment.groupby(["biosample", "readout_target"], as_index=False)
        .agg(
            experiments=("experiment_accession", "nunique"),
            elements_tested=("elements_tested", "sum"),
            significant_elements=("significant_elements", "sum"),
            median_absolute_effect=("median_absolute_effect", "median"),
            maximum_absolute_effect=("maximum_absolute_effect", "max"),
        )
    )
    target["significant_fraction"] = target["significant_elements"] / target["elements_tested"]
    elements.to_parquet(args.output / "elements.parquet", index=False)
    experiment.to_parquet(args.output / "experiments.parquet", index=False)
    target.to_parquet(args.output / "targets.parquet", index=False)
    pd.DataFrame(rejected, columns=["file_accession", "reason"]).to_csv(
        args.output / "rejected_files.tsv", sep="\t", index=False
    )
    audit = {
        "catalog_files": int(len(catalog)),
        "accepted_files": int(elements["file_accession"].nunique()),
        "rejected_files": int(len(rejected)),
        "biosamples": int(elements["biosample"].nunique()),
        "readout_targets": int(elements["readout_target"].nunique()),
        "element_rows": int(len(elements)),
        "valid_element_rows": int(len(valid)),
        "significant_valid_rows": int(valid["significant"].sum()),
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
