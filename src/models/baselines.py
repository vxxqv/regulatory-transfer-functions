"""Prespecified linear, hierarchical, nonlinear, and graph-network baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


@dataclass(frozen=True)
class StabilityDiagnostics:
    spectral_radius: float
    condition_number: float
    stable: bool


def stability_diagnostics(operator: np.ndarray, threshold: float = 0.99) -> StabilityDiagnostics:
    eigenvalues = np.linalg.eigvals(operator)
    radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    condition = float(np.linalg.cond(np.eye(operator.shape[0]) - operator))
    return StabilityDiagnostics(radius, condition, radius < threshold)


def stabilize_operator(operator: np.ndarray, maximum_radius: float = 0.95) -> np.ndarray:
    diagnostics = stability_diagnostics(operator, threshold=maximum_radius + np.finfo(float).eps)
    if diagnostics.spectral_radius <= maximum_radius or diagnostics.spectral_radius == 0:
        return operator.copy()
    return operator * (maximum_radius / diagnostics.spectral_radius)


def finite_horizon_transfer(operator: np.ndarray, input_map: np.ndarray, depth: int) -> np.ndarray:
    """Compute sum from k=0 to depth of W^k B without an infinite-series assumption."""
    if depth < 0:
        raise ValueError("Depth must be nonnegative")
    result = input_map.copy()
    current = input_map.copy()
    for _ in range(depth):
        current = operator @ current
        result = result + current
    return result


def hierarchical_gaussian_pool(
    values: np.ndarray, groups: np.ndarray, observation_variance: np.ndarray
) -> np.ndarray:
    """Empirical-Bayes partial pooling under a Gaussian random-intercept model."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    observation_variance = np.asarray(observation_variance, dtype=float)
    global_mean = float(np.average(values, weights=1.0 / np.maximum(observation_variance, 1e-8)))
    group_means = {group: float(values[groups == group].mean()) for group in np.unique(groups)}
    between = float(np.var(list(group_means.values()), ddof=1)) if len(group_means) > 1 else 0.0
    pooled = np.empty_like(values)
    for group in np.unique(groups):
        index = groups == group
        precision = np.sum(1.0 / np.maximum(observation_variance[index], 1e-8))
        shrinkage = between / (between + 1.0 / precision) if between > 0 else 0.0
        pooled[index] = shrinkage * group_means[group] + (1.0 - shrinkage) * global_mean
    return pooled


class SparseNonlinearBaseline:
    def __init__(self, seed: int) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )

    def fit(self, features: np.ndarray, outcome: np.ndarray) -> "SparseNonlinearBaseline":
        self.model.fit(features, outcome)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


class GraphNeuralBaseline:
    """Two-step graph propagation followed by a compact nonlinear readout."""

    def __init__(self, seed: int) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            alpha=1.0,
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        )

    @staticmethod
    def propagate(features: np.ndarray, adjacency: sparse.spmatrix) -> np.ndarray:
        adjacency = sparse.csr_matrix(adjacency, dtype=float)
        augmented = adjacency + sparse.eye(adjacency.shape[0], format="csr")
        degree = np.asarray(augmented.sum(axis=1)).ravel()
        inverse_root = sparse.diags(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
        normalized = inverse_root @ augmented @ inverse_root
        first = normalized @ features
        second = normalized @ first
        return np.hstack([features, np.asarray(first), np.asarray(second)])

    def fit(
        self, features: np.ndarray, adjacency: sparse.spmatrix, outcome: np.ndarray
    ) -> "GraphNeuralBaseline":
        self.model.fit(self.propagate(features, adjacency), outcome)
        return self

    def predict(self, features: np.ndarray, adjacency: sparse.spmatrix) -> np.ndarray:
        return self.model.predict(self.propagate(features, adjacency))
