"""Leakage-safe decomposition and comparison of regulatory response vectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold


def row_cosine(left: sparse.spmatrix, right: sparse.spmatrix) -> np.ndarray:
    """Return aligned sparse-row cosine similarities, with zero for empty rows."""
    numerator = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_norm = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).ravel())
    right_norm = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).ravel())
    denominator = left_norm * right_norm
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def module_energy(scores: np.ndarray) -> np.ndarray:
    """Convert orthogonal component scores to per-row energy fractions."""
    squared = np.square(scores)
    totals = squared.sum(axis=1, keepdims=True)
    return np.divide(squared, totals, out=np.zeros_like(squared), where=totals > 0)


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-aligned Jensen-Shannon divergence on base 2, bounded by 0 and 1."""
    values = np.zeros(left.shape[0], dtype=float)
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        if left_row.sum() > 0 and right_row.sum() > 0:
            distance = jensenshannon(left_row, right_row, base=2.0)
            values[index] = float(distance**2)
    return values


def cross_fitted_modules(
    matrix: sparse.csr_matrix,
    groups: np.ndarray,
    n_components: int,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit SVD modules on training targets and score held-out targets once."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two distinct target groups are required")
    effective_folds = min(folds, len(unique_groups))
    splitter = GroupKFold(n_splits=effective_folds)
    scores = np.zeros((matrix.shape[0], n_components), dtype=np.float32)
    reconstruction_cosine = np.zeros(matrix.shape[0], dtype=np.float32)
    fold_records: list[dict[str, int | float]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(matrix.shape[0]), groups=groups), start=1
    ):
        usable_components = min(n_components, matrix[train_index].shape[0] - 1, matrix.shape[1] - 1)
        model = TruncatedSVD(n_components=usable_components, random_state=seed + fold)
        model.fit(matrix[train_index])
        held_out_scores = model.transform(matrix[test_index])
        reconstruction = sparse.csr_matrix(held_out_scores @ model.components_)
        scores[test_index, :usable_components] = held_out_scores.astype(np.float32)
        reconstruction_cosine[test_index] = row_cosine(matrix[test_index], reconstruction)
        fold_records.append(
            {
                "fold": fold,
                "training_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "training_targets": int(pd.Series(groups[train_index]).nunique()),
                "test_targets": int(pd.Series(groups[test_index]).nunique()),
                "components": int(usable_components),
                "training_explained_variance": float(model.explained_variance_ratio_.sum()),
            }
        )
    return scores, reconstruction_cosine, pd.DataFrame(fold_records)


def aligned_state_pairs(rows: pd.DataFrame, states: tuple[str, ...]) -> pd.DataFrame:
    """Enumerate row indices for within-target state comparisons."""
    lookup = rows.reset_index(names="row_index").pivot(
        index="target_contrast", columns="culture_condition", values="row_index"
    )
    records: list[dict[str, object]] = []
    for left_position, left_state in enumerate(states):
        for right_state in states[left_position + 1 :]:
            complete = lookup[[left_state, right_state]].dropna().astype(int)
            for target, pair in complete.iterrows():
                records.append(
                    {
                        "target_contrast": target,
                        "left_state": left_state,
                        "right_state": right_state,
                        "left_index": int(pair[left_state]),
                        "right_index": int(pair[right_state]),
                    }
                )
    return pd.DataFrame(records)
