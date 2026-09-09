import unittest

from agent.broad_behavior_learning import build_learning_rows, segment_rows


class BroadBehaviorLearningTests(unittest.TestCase):
    def test_low_clv_still_enters_learning(self):
        rows = [{
            "id": 1,
            "fixture_date": "2026-08-01",
            "league": "Test",
            "home_team": "A",
            "away_team": "B",
            "odd_over25_open": 2.0,
            "fair_probability_open": 0.50,
            "clv_probability": 0.42,
            "signal_threshold": 0.55,
            "signal": 0,
            "features_json": "{}",
            "result_over25": 1,
        }]

        learned = build_learning_rows(rows, lambda pre: 0.60)
        self.assertEqual(len(learned), 1)
        self.assertEqual(learned[0]["clv_bin"], "0.40-0.45")
        self.assertEqual(learned[0]["result_over25"], 1)
        self.assertAlmostEqual(learned[0]["expected_value"], 0.20)

    def test_outcome_is_not_passed_to_model(self):
        rows = [{
            "id": 2,
            "fixture_date": "2026-08-02",
            "league": "Test",
            "home_team": "A",
            "away_team": "B",
            "odd_over25_open": 1.9,
            "fair_probability_open": 0.52,
            "clv_probability": 0.50,
            "signal_threshold": 0.55,
            "signal": 0,
            "features_json": "{}",
            "result_over25": 0,
        }]

        def sports(pre):
            self.assertNotIn("result_over25", pre)
            self.assertNotIn("profit", pre)
            self.assertNotIn("clv", pre)
            return 0.58

        build_learning_rows(rows, sports)

    def test_segments_include_signal_zero(self):
        rows = [
            {
                "id": 1, "fixture_date": "2026-08-01", "league": "L",
                "home_team": "A", "away_team": "B", "odd_over25_open": 2.0,
                "fair_probability_open": 0.5, "clv_probability": 0.42,
                "signal_threshold": 0.55, "signal": 0, "features_json": "{}",
                "result_over25": 1,
            },
            {
                "id": 2, "fixture_date": "2026-08-02", "league": "L",
                "home_team": "C", "away_team": "D", "odd_over25_open": 2.0,
                "fair_probability_open": 0.5, "clv_probability": 0.62,
                "signal_threshold": 0.55, "signal": 1, "features_json": "{}",
                "result_over25": 0,
            },
        ]
        learned = build_learning_rows(rows, lambda pre: 0.60)
        segments = segment_rows(learned)
        signal0 = next(s for s in segments if s["segment_type"] == "ORIGINAL_SIGNAL" and s["segment_value"] == "SIGNAL_0")
        self.assertEqual(signal0["sample"], 1)
        self.assertAlmostEqual(signal0["roi"], 1.0)


if __name__ == "__main__":
    unittest.main()
