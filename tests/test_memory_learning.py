import unittest

import pandas as pd

from agent.memory_learning import build_memory_summary, standardized_mean_shift


class MemoryLearningTests(unittest.TestCase):
    def test_resolved_rows_drive_roi(self):
        resolved = pd.DataFrame([
            {"result_over25": 1, "odd_over25_open": 2.0, "clv_probability": 0.60, "sports_probability": 0.60, "expected_value": 0.08, "sports_edge_market": 0.04},
            {"result_over25": 0, "odd_over25_open": 2.0, "clv_probability": 0.58, "sports_probability": 0.57, "expected_value": 0.03, "sports_edge_market": 0.02},
        ])
        summary = build_memory_summary(pd.DataFrame([{"x": 1}]), resolved, pd.DataFrame())
        self.assertAlmostEqual(summary.resolved_roi, 0.0)
        self.assertAlmostEqual(summary.resolved_hit_rate, 0.5)

    def test_open_rows_are_unlabelled_context_only(self):
        open_rows = pd.DataFrame([
            {"clv_probability": 0.50, "sports_probability": 0.61, "expected_value": 0.07, "sports_edge_market": 0.03, "odd_over25_open": 1.9},
            {"clv_probability": 0.54, "sports_probability": 0.63, "expected_value": 0.09, "sports_edge_market": 0.04, "odd_over25_open": 1.8},
        ])
        summary = build_memory_summary(pd.DataFrame(), pd.DataFrame(), open_rows)
        self.assertEqual(summary.open_rows, 2)
        self.assertIsNone(summary.resolved_roi)
        self.assertAlmostEqual(summary.open_clv_mean, 0.52)

    def test_drift_detects_large_shift(self):
        resolved = pd.DataFrame([
            {"clv_probability": 0.50 + i * 0.001, "sports_probability": 0.50 + i * 0.001, "expected_value": 0.01 + i * 0.001, "sports_edge_market": 0.01 + i * 0.001, "odd_over25_open": 1.8 + i * 0.002}
            for i in range(30)
        ])
        open_rows = pd.DataFrame([
            {"clv_probability": 0.70, "sports_probability": 0.72, "expected_value": 0.20, "sports_edge_market": 0.15, "odd_over25_open": 2.5}
            for _ in range(10)
        ])
        score = standardized_mean_shift(resolved, open_rows)
        self.assertIsNotNone(score)
        self.assertGreater(score, 1.0)


if __name__ == "__main__":
    unittest.main()
