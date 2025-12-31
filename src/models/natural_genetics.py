"""Allele harmonization and directional validation for cis-to-trans eQTLs."""

from __future__ import annotations

import numpy as np


COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def allele_orientation(
    cis_assessed: str,
    cis_other: str,
    trans_assessed: str,
    trans_other: str,
) -> int:
    """Return 1 for aligned, -1 for swapped, and 0 for unresolved alleles."""
    cis = (cis_assessed.upper(), cis_other.upper())
    trans = (trans_assessed.upper(), trans_other.upper())
    if cis == trans:
        return 1
    if cis == trans[::-1]:
        return -1
    complemented = tuple(COMPLEMENT.get(allele, "") for allele in trans)
    if cis == complemented:
        return 1
    if cis == complemented[::-1]:
        return -1
    return 0


def predicted_genetic_direction(perturbation_effect: np.ndarray, cis_z: float) -> np.ndarray:
    """Orient knockdown effects to the allele that increases or decreases the mediator."""
    return perturbation_effect * -np.sign(cis_z)


def signed_cosine(predicted: np.ndarray, observed: np.ndarray) -> float:
    denominator = np.linalg.norm(predicted) * np.linalg.norm(observed)
    return float(np.dot(predicted, observed) / denominator) if denominator > 0 else np.nan
