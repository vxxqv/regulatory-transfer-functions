"""Quality-control summaries for perturbation result tables."""

from __future__ import annotations

import pandas as pd

from src.models.empirical_transfer import quality_filter, validate_summary


def audit_counts(data: pd.DataFrame, minimum_guides: int, minimum_cells: int) -> pd.DataFrame:
    validate_summary(data)
    rules = {
        "all_rows": pd.Series(True, index=data.index),
        "ontarget_significant": data["ontarget_significant"].fillna(False).astype(bool),
        "adequate_target_expression": ~data["low_target_gex"].fillna(True).astype(bool),
        "no_neighboring_gene_knockdown": ~data["neighboring_gene_KD"].fillna(True).astype(bool),
        "no_distal_offtarget": ~data["distal_offtarget_flag"].fillna(True).astype(bool),
        "minimum_guides": data["n_guides"].fillna(0) >= minimum_guides,
        "minimum_cells": data["n_cells_target"].fillna(0) >= minimum_cells,
        "complete_covariates": data[["target_baseMean", "ontarget_effect_size"]].notna().all(axis=1),
        "primary_analysis_set": quality_filter(data, minimum_guides, minimum_cells),
    }
    rows = []
    for name, mask in rules.items():
        rows.append(
            {
                "criterion": name,
                "rows_passing": int(mask.sum()),
                "targets_passing": int(data.loc[mask, "target_contrast"].nunique()),
                "fraction_rows": float(mask.mean()),
            }
        )
    return pd.DataFrame(rows)
