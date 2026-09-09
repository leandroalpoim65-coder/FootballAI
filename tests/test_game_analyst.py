import json
import unittest

from agent.game_analyst import analyze_features


class GameAnalystTests(unittest.TestCase):
    def test_high_goal_profile_scores_higher(self):
        high = {
            "expected_total_goals": 3.4,
            "home_over25_rate_5": .8,
            "away_over25_rate_5": .7,
            "home_btts_rate_5": .8,
            "away_btts_rate_5": .7,
            "home_scored_rate_5": .9,
            "away_scored_rate_5": .9,
            "home_avg_scored_5": 2.0,
            "away_avg_scored_5": 1.8,
            "home_avg_conceded_5": 1.6,
            "away_avg_conceded_5": 1.7,
            "over25_signal": 1,
            "scoring_signal": 1,
        }
        low = {
            "expected_total_goals": 1.9,
            "home_over25_rate_5": .2,
            "away_over25_rate_5": .2,
            "home_btts_rate_5": .2,
            "away_btts_rate_5": .2,
            "home_scored_rate_5": .4,
            "away_scored_rate_5": .4,
            "home_avg_scored_5": .8,
            "away_avg_scored_5": .7,
            "home_avg_conceded_5": .8,
            "away_avg_conceded_5": .7,
            "home_clean_sheet_rate_5": .6,
            "away_clean_sheet_rate_5": .6,
        }
        self.assertGreater(analyze_features(json.dumps(high)).game_score, analyze_features(json.dumps(low)).game_score)

    def test_score_is_bounded(self):
        a = analyze_features({"expected_total_goals": 8, "home_over25_rate_5": 1, "away_over25_rate_5": 1})
        self.assertGreaterEqual(a.game_score, 0)
        self.assertLessEqual(a.game_score, 100)


if __name__ == "__main__":
    unittest.main()
