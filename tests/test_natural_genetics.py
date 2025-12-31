import numpy as np

from src.models.natural_genetics import (
    allele_orientation,
    predicted_genetic_direction,
    signed_cosine,
)


def test_allele_orientation_supports_swap_and_strand() -> None:
    assert allele_orientation("A", "G", "A", "G") == 1
    assert allele_orientation("A", "G", "G", "A") == -1
    assert allele_orientation("A", "G", "T", "C") == 1
    assert allele_orientation("A", "G", "C", "T") == -1
    assert allele_orientation("A", "G", "A", "C") == 0


def test_directional_prediction_and_cosine() -> None:
    knockdown = np.array([-1.0, 2.0])
    assert np.allclose(predicted_genetic_direction(knockdown, 3.0), [1.0, -2.0])
    assert signed_cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
