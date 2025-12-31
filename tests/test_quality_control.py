from __future__ import annotations

import unittest

import pandas as pd

from src.models.empirical_transfer import quality_filter, validate_summary


class QualityControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "target_contrast": ["ENSG1", "ENSG2"],
                "target_contrast_gene_name": ["A", "B"],
                "culture_condition": ["Rest", "Rest"],
                "n_cells_target": [250, 250],
                "n_downstream": [3, 4],
                "ontarget_effect_size": [-1.0, -1.0],
                "ontarget_significant": [True, True],
                "target_baseMean": [10.0, 10.0],
                "neighboring_gene_KD": [False, False],
                "distal_offtarget_flag": [False, False],
                "low_target_gex": [False, False],
                "n_guides": [2, 1],
            }
        )

    def test_quality_filter_applies_all_rules(self) -> None:
        mask = quality_filter(self.frame, minimum_guides=2, minimum_cells=200)
        self.assertEqual(mask.tolist(), [True, False])

    def test_duplicate_target_condition_is_rejected(self) -> None:
        duplicate = pd.concat([self.frame.iloc[[0]], self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_summary(duplicate)


if __name__ == "__main__":
    unittest.main()
