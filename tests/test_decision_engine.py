import unittest

from agent.decision_engine import (
    PerformanceStats,
    evaluate_candidate,
)


class DecisionEngineTests(unittest.TestCase):
    def setUp(self):
        self.global_stats = PerformanceStats(
            sample=100,
            wins=57,
            stake=100.0,
            profit=5.0,
            hit_rate=0.57,
            roi=0.05,
            hq_clv_sample=60,
            hq_clv_mean=0.01,
        )
        self.row = {
            "odd_over25_open": 1.90,
            "fair_probability_open": 0.52,
            "clv_probability": 0.61,
            "signal_threshold": 0.55,
            "signal": 1,
        }

    def test_strong_alignment_qualifies(self):
        league = PerformanceStats(
            sample=20,
            wins=12,
            stake=20.0,
            profit=2.0,
            hit_rate=0.60,
            roi=0.10,
            hq_clv_sample=10,
            hq_clv_mean=0.02,
        )
        result = evaluate_candidate(
            self.row,
            sports_probability=0.58,
            global_stats=self.global_stats,
            league_stats=league,
        )
        self.assertEqual(result.decision, "QUALIFIED_SHADOW")
        self.assertEqual(result.tier, "A")

    def test_model_disagreement_is_watch(self):
        result = evaluate_candidate(
            self.row,
            sports_probability=0.53,
            global_stats=self.global_stats,
            league_stats=PerformanceStats(),
        )
        self.assertEqual(result.decision, "WATCH")
        self.assertEqual(result.tier, "MODEL_DISAGREEMENT")

    def test_missing_original_clv_signal_rejects(self):
        row = dict(self.row)
        row["signal"] = 0
        result = evaluate_candidate(
            row,
            sports_probability=0.58,
            global_stats=self.global_stats,
            league_stats=PerformanceStats(),
        )
        self.assertEqual(result.decision, "REJECT")
        self.assertEqual(result.tier, "CLV_REJECT")

    def test_degraded_league_vetoes_to_watch(self):
        league = PerformanceStats(
            sample=80,
            wins=36,
            stake=80.0,
            profit=-12.0,
            hit_rate=0.45,
            roi=-0.15,
            hq_clv_sample=30,
            hq_clv_mean=-0.01,
        )
        result = evaluate_candidate(
            self.row,
            sports_probability=0.58,
            global_stats=self.global_stats,
            league_stats=league,
        )
        self.assertEqual(result.decision, "WATCH")
        self.assertEqual(result.tier, "LEAGUE_DEGRADED")


if __name__ == "__main__":
    unittest.main()
