import numpy as np
from scipy import sparse

from src.models.baselines import (
    GraphNeuralBaseline,
    finite_horizon_transfer,
    hierarchical_gaussian_pool,
    stability_diagnostics,
    stabilize_operator,
)


def test_finite_horizon_and_stability() -> None:
    operator = np.array([[0.0, 0.5], [0.0, 0.0]])
    input_map = np.eye(2)
    expected = input_map + operator + operator @ operator
    assert np.allclose(finite_horizon_transfer(operator, input_map, 2), expected)
    unstable = np.array([[1.2, 0.0], [0.0, 0.2]])
    assert not stability_diagnostics(unstable).stable
    assert stability_diagnostics(stabilize_operator(unstable)).spectral_radius <= 0.9500001


def test_partial_pooling_and_graph_propagation() -> None:
    values = np.array([0.0, 2.0, 10.0, 12.0])
    groups = np.array(["a", "a", "b", "b"])
    pooled = hierarchical_gaussian_pool(values, groups, np.ones(4))
    assert pooled[0] == pooled[1]
    assert pooled[2] == pooled[3]
    assert pooled[0] < pooled[2]
    features = np.eye(3)
    adjacency = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    propagated = GraphNeuralBaseline.propagate(features, adjacency)
    assert propagated.shape == (3, 9)
    assert np.isfinite(propagated).all()
