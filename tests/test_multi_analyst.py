import unittest

from agent.multi_analyst import score_pre_match, weighted_score


class MultiAnalystTests(unittest.TestCase):
    def base_features(self):
        return {
            "expected_total_goals": 2.8,
            "home_over25_rate_5": 0.5,
            "away_over25_rate_5": 0.5,
            "home_btts_rate_5": 0.5,
            "away_btts_rate_5": 0.5,
            "over25_signal": 0,
            "home_scored_rate_5": 0.7,
            "away_scored_rate_5": 0.7,
            "home_avg_scored_5": 1.4,
            "away_avg_scored_5": 1.4,
            "home_avg_conceded_5": 1.2,
            "away_avg_conceded_5": 1.2,
            "total_expected_scored": 2.6,
            "home_clean_sheet_rate_5": 0.3,
            "away_clean_sheet_rate_5": 0.3,
        }

    def test_stronger_goal_environment_scores_higher(self):
        weak = self.base_features()
        strong = self.base_features()
        strong.update({
            "expected_total_goals": 3.4,
            "home_over25_rate_5": 0.8,
            "away_over25_rate_5": 0.8,
            "home_btts_rate_5": 0.75,
            "away_btts_rate_5": 0.75,
            "over25_signal": 1,
        })

        a = score_pre_match(weak, 0.55, 0.02, 0.01)
        b = score_pre_match(strong, 0.55, 0.02, 0.01)
        self.assertGreater(b.goal_profile, a.goal_profile)

    def test_better_ml_and_value_scores_raise_independent_analysts(self):
        f = self.base_features()
        low = score_pre_match(f, 0.50, -0.02, -0.01)
        high = score_pre_match(f, 0.64, 0.12, 0.06)
        self.assertGreater(high.ml_score, low.ml_score)
        self.assertGreater(high.value_score, low.value_score)
        self.assertEqual(high.goal_profile, low.goal_profile)

    def test_weighted_score_stays_in_range(self):
        s = score_pre_match(self.base_features(), 0.60, 0.08, 0.03)
        value = weighted_score(s, (0.25, 0.25, 0.25, 0.25))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 100.0)


if __name__ == "__main__":
    unittest.main()
