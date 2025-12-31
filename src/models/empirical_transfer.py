"""Cross-fitted empirical transfer phenotypes from perturbation summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = {
    "target_contrast",
    "target_contrast_gene_name",
    "culture_condition",
    "n_cells_target",
    "n_downstream",
    "ontarget_effect_size",
    "ontarget_significant",
    "target_baseMean",
    "neighboring_gene_KD",
    "distal_offtarget_flag",
    "low_target_gex",
    "n_guides",
}


@dataclass(frozen=True)
class TransferFit:
    table: pd.DataFrame
    r2: float
    folds: int


def validate_summary(data: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[["target_contrast", "culture_condition"]].duplicated().any():
        raise ValueError("Target and condition keys must be unique")
    if (pd.to_numeric(data["n_downstream"], errors="coerce") < 0).any():
        raise ValueError("n_downstream must be non-negative")


def quality_filter(
    data: pd.DataFrame,
    minimum_guides: int = 2,
    minimum_cells: int = 200,
) -> pd.Series:
    """Return the prespecified high-confidence inclusion mask."""

    validate_summary(data)
    return (
        data["ontarget_significant"].fillna(False).astype(bool)
        & ~data["low_target_gex"].fillna(True).astype(bool)
        & ~data["neighboring_gene_KD"].fillna(True).astype(bool)
        & ~data["distal_offtarget_flag"].fillna(True).astype(bool)
        & (pd.to_numeric(data["n_guides"], errors="coerce") >= minimum_guides)
        & (pd.to_numeric(data["n_cells_target"], errors="coerce") >= minimum_cells)
        & data["target_baseMean"].notna()
        & data["ontarget_effect_size"].notna()
    )


def _pipeline(seed: int) -> Pipeline:
    numeric = [
        "cis_magnitude",
        "log1p_target_baseMean",
        "log1p_n_cells_target",
        "n_guides",
    ]
    categorical = ["culture_condition"]
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline([("prepare", prep), ("model", model)])


def cross_fitted_transfer(
    data: pd.DataFrame,
    folds: int = 10,
    seed: int = 20260912,
) -> TransferFit:
    """Estimate out-of-fold expected response and empirical transfer residuals."""

    frame = data.copy()
    validate_summary(frame)
    frame["cis_magnitude"] = frame["ontarget_effect_size"].abs()
    frame["log1p_target_baseMean"] = np.log1p(frame["target_baseMean"])
    frame["log1p_n_cells_target"] = np.log1p(frame["n_cells_target"])
    frame["log1p_n_downstream"] = np.log1p(frame["n_downstream"])

    groups = frame["target_contrast"].astype(str)
    n_groups = groups.nunique()
    used_folds = min(folds, n_groups)
    if used_folds < 2:
        raise ValueError("At least two target genes are required for cross-fitting")

    splitter = GroupKFold(n_splits=used_folds)
    predictions = np.full(len(frame), np.nan, dtype=float)
    features = [
        "cis_magnitude",
        "log1p_target_baseMean",
        "log1p_n_cells_target",
        "n_guides",
        "culture_condition",
    ]
    outcome = frame["log1p_n_downstream"].to_numpy(dtype=float)

    for train, test in splitter.split(frame, outcome, groups):
        estimator = _pipeline(seed)
        estimator.fit(frame.iloc[train][features], outcome[train])
        predictions[test] = estimator.predict(frame.iloc[test][features])

    if np.isnan(predictions).any():
        raise RuntimeError("Cross-fitting left observations without predictions")

    frame["expected_log1p_n_downstream"] = predictions
    frame["transfer_residual"] = outcome - predictions
    scale = frame.groupby("culture_condition")["transfer_residual"].transform("std")
    frame["transfer_z"] = frame["transfer_residual"] / scale.replace(0, np.nan)
    low, high = frame["transfer_z"].quantile([0.10, 0.90])
    frame["transfer_class"] = np.select(
        [frame["transfer_z"] <= low, frame["transfer_z"] >= high],
        ["buffered", "amplified"],
        default="intermediate",
    )
    return TransferFit(frame, r2_score(outcome, predictions), used_folds)
