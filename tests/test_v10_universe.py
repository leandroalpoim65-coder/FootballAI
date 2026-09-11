import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from agent_v10_universe_cloud import (
    canonical_fixture_key,
    identity_key,
    is_safely_pre_kickoff,
)


LISBON = ZoneInfo("Europe/Lisbon")


class V10UniverseTests(unittest.TestCase):
    def test_canonical_key_is_stable_and_case_insensitive(self):
        a = canonical_fixture_key(
            "2026-09-11",
            "France Ligue 2",
            "Nancy",
            "Reims",
        )
        b = canonical_fixture_key(
            " 2026-09-11 ",
            "FRANCE LIGUE 2",
            "NANCY",
            "reims",
        )
        self.assertEqual(a, b)

    def test_identity_key_ignores_case_and_outer_spaces(self):
        a = identity_key("2026-09-11", "Portugal", "Porto", "Braga")
        b = identity_key("2026-09-11", " PORTUGAL ", "PORTO", "braga")
        self.assertEqual(a, b)

    def test_future_fixture_is_safe_without_time(self):
        now = datetime(2026, 9, 11, 12, 0, tzinfo=LISBON)
        row = {"date": "2026-09-12", "time": None}
        self.assertTrue(is_safely_pre_kickoff(row, now))

    def test_same_day_requires_five_minute_margin(self):
        now = datetime(2026, 9, 11, 18, 54, tzinfo=LISBON)
        row = {"date": "2026-09-11", "time": "19:00"}
        self.assertTrue(is_safely_pre_kickoff(row, now))

        late = datetime(2026, 9, 11, 18, 56, tzinfo=LISBON)
        self.assertFalse(is_safely_pre_kickoff(row, late))

    def test_past_fixture_is_never_safe(self):
        now = datetime(2026, 9, 11, 12, 0, tzinfo=LISBON)
        row = {"date": "2026-09-10", "time": "20:00"}
        self.assertFalse(is_safely_pre_kickoff(row, now))


if __name__ == "__main__":
    unittest.main()
