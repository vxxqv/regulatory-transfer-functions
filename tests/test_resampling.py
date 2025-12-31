from __future__ import annotations

import unittest

import pandas as pd

from src.statistics.resampling import grouped_bootstrap_interval


class ResamplingTests(unittest.TestCase):
    def test_grouped_bootstrap_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            {"target": ["A", "A", "B", "B"], "value": [1.0, 2.0, 3.0, 4.0]}
        )
        first = grouped_bootstrap_interval(frame, "target", "value", replicates=100, seed=7)
        second = grouped_bootstrap_interval(frame, "target", "value", replicates=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 2.5)


if __name__ == "__main__":
    unittest.main()
