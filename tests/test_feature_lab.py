import unittest

import numpy as np
import pandas as pd

from agent.feature_lab import candidate_library, evaluate_feature


class FeatureLabTests(unittest.TestCase):
    def test_library_has_unique_features(self):
        names = [x.name for x in candidate_library()]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 5)

    def test_waits_with_small_sample(self):
        data = pd.DataFrame({
            "paper_trade_id": range(50),
            "fixture_date": pd.date_range("2026-01-01", periods=50).astype(str),
            "result_over25": [0, 1] * 25,
            "sports_probability": np.linspace(.4, .7, 50),
            "market_probability": np.linspace(.4, .6, 50),
            "clv_probability": np.linspace(.4, .7, 50),
            "odd_over25_open": np.repeat(1.9, 50),
            "base": np.linspace(0, 1, 50),
        })
        candidate = candidate_library()[0]
        result = evaluate_feature(data, ["base"], candidate)
        self.assertEqual(result.status, "WAIT")

    def test_unavailable_when_source_columns_missing(self):
        data = pd.DataFrame({
            "paper_trade_id": [1],
            "fixture_date": ["2026-01-01"],
            "result_over25": [1],
            "base": [0.2],
        })
        result = evaluate_feature(data, ["base"], candidate_library()[0])
        self.assertEqual(result.status, "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
