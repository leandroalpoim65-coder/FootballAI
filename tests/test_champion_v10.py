import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.champion_v10 import evaluate_champion
from agent.multi_analyst import AnalystScores
from agent_champion_v10_shadow_cloud import is_safely_pre_kickoff


class ChampionV10Tests(unittest.TestCase):
    def scores(self, goal=60, attack=60, ml=70, value=65):
        return AnalystScores(
            goal_profile=goal,
            attack_pressure=attack,
            ml_score=ml,
            value_score=value,
            stats_score=(goal + attack) / 2,
        )

    def test_three_votes_and_non_negative_ev_bets(self):
        decision = evaluate_champion(
            self.scores(value=50),
            expected_value=0.01,
        )
        self.assertEqual(decision.vote_count, 3)
        self.assertEqual(decision.decision, "BET_SHADOW")

    def test_negative_ev_blocks_even_four_votes(self):
        decision = evaluate_champion(
            self.scores(),
            expected_value=-0.001,
        )
        self.assertEqual(decision.vote_count, 4)
        self.assertEqual(decision.decision, "PASS")

    def test_two_votes_pass(self):
        decision = evaluate_champion(
            self.scores(ml=50, value=50),
            expected_value=0.10,
        )
        self.assertEqual(decision.vote_count, 2)
        self.assertEqual(decision.decision, "PASS")

    def test_same_day_requires_five_minute_buffer(self):
        tz = ZoneInfo("Europe/Lisbon")
        now = datetime(2026, 9, 11, 15, 0, tzinfo=tz)
        safe = {"fixture_date": "2026-09-11", "fixture_time": "15:10"}
        late = {"fixture_date": "2026-09-11", "fixture_time": "15:04"}
        unknown = {"fixture_date": "2026-09-11", "fixture_time": None}

        self.assertTrue(is_safely_pre_kickoff(safe, now))
        self.assertFalse(is_safely_pre_kickoff(late, now))
        self.assertFalse(is_safely_pre_kickoff(unknown, now))

    def test_future_date_is_safe(self):
        tz = ZoneInfo("Europe/Lisbon")
        now = datetime(2026, 9, 11, 15, 0, tzinfo=tz)
        row = {"fixture_date": "2026-09-12", "fixture_time": None}
        self.assertTrue(is_safely_pre_kickoff(row, now))


if __name__ == "__main__":
    unittest.main()
