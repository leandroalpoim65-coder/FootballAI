import unittest

from agent.decision_engine import PolicyConfig
from agent.historical_replay import replay_rows


class HistoricalReplayTests(unittest.TestCase):
    def row(self, ident, date, result, signal=1):
        return {
            "id": ident, "fixture_date": date, "fixture_time": "20:00",
            "league": "Test", "home_team": "A", "away_team": "B",
            "odd_over25_open": 2.0, "fair_probability_open": 0.50,
            "clv_probability": 0.62, "signal_threshold": 0.55,
            "signal": signal, "features_json": "{}", "result_over25": result,
            "clv": 0.02, "closing_source": "The Odds API",
        }

    def test_callback_never_receives_outcome(self):
        seen = []
        def sports(pre):
            seen.append(set(pre))
            self.assertNotIn("result_over25", pre)
            self.assertNotIn("profit", pre)
            self.assertNotIn("clv", pre)
            self.assertNotIn("closing_source", pre)
            return 0.60
        replay_rows([self.row(1, "2026-08-01", 1)], sports, PolicyConfig())
        self.assertEqual(len(seen), 1)

    def test_result_changes_evaluation_not_decision(self):
        def sports(pre): return 0.60
        win, _ = replay_rows([self.row(1, "2026-08-01", 1)], sports, PolicyConfig())
        loss, _ = replay_rows([self.row(1, "2026-08-01", 0)], sports, PolicyConfig())
        self.assertEqual(win[0]["decision"], loss[0]["decision"])
        self.assertEqual(win[0]["tier"], loss[0]["tier"])
        self.assertNotEqual(win[0]["counterfactual_profit"], loss[0]["counterfactual_profit"])

    def test_future_result_cannot_change_prior_decision(self):
        def sports(pre): return 0.60
        rows_a = [self.row(1, "2026-08-01", 1), self.row(2, "2026-08-02", 1)]
        rows_b = [self.row(1, "2026-08-01", 1), self.row(2, "2026-08-02", 0)]
        a, _ = replay_rows(rows_a, sports, PolicyConfig())
        b, _ = replay_rows(rows_b, sports, PolicyConfig())
        self.assertEqual(a[0]["decision"], b[0]["decision"])
        self.assertEqual(a[0]["expected_value"], b[0]["expected_value"])


if __name__ == "__main__":
    unittest.main()
