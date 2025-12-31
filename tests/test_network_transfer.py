import numpy as np
import pandas as pd
from scipy import sparse

from src.models.network_transfer import (
    aligned_state_pairs,
    cross_fitted_modules,
    js_divergence,
    module_energy,
    row_cosine,
)


def test_sparse_cosine_and_module_energy() -> None:
    left = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    right = sparse.csr_matrix([[1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(row_cosine(left, right), [1.0, 0.0])
    energy = module_energy(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(energy[0], [0.36, 0.64])
    assert np.allclose(energy[1], [0.0, 0.0])
    assert np.allclose(js_divergence(energy[:1], energy[:1]), [0.0])


def test_target_group_cross_fitting_and_pairs() -> None:
    matrix = sparse.csr_matrix(
        [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, -1], [0, -1, 1], [-1, 1, 0]],
        dtype=float,
    )
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    scores, cosine, folds = cross_fitted_modules(matrix, groups, 2, 3, 7)
    assert scores.shape == (6, 2)
    assert cosine.shape == (6,)
    assert folds["test_targets"].sum() == 3
    rows = pd.DataFrame(
        {
            "target_contrast": ["a", "a", "b", "b"],
            "culture_condition": ["Rest", "Stim8hr", "Rest", "Stim8hr"],
        }
    )
    pairs = aligned_state_pairs(rows, ("Rest", "Stim8hr"))
    assert len(pairs) == 2
