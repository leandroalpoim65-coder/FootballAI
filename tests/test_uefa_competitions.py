import unittest

from agent.uefa_competitions import (
    UEFA_COMPETITIONS,
    canonical_uefa_league,
    is_uefa_club_competition,
)


class UefaCompetitionTests(unittest.TestCase):
    def test_expected_competition_ids(self):
        self.assertEqual(UEFA_COMPETITIONS[2], "UEFA Champions League")
        self.assertEqual(UEFA_COMPETITIONS[3], "UEFA Europa League")
        self.assertEqual(UEFA_COMPETITIONS[848], "UEFA Conference League")

    def test_aliases_are_normalized(self):
        self.assertEqual(
            canonical_uefa_league("Champions League"),
            "UEFA Champions League",
        )
        self.assertEqual(
            canonical_uefa_league("UEFA Europa Conference League"),
            "UEFA Conference League",
        )

    def test_detects_only_supported_uefa_club_competitions(self):
        self.assertTrue(is_uefa_club_competition("Europa League"))
        self.assertFalse(is_uefa_club_competition("Premier League"))


if __name__ == "__main__":
    unittest.main()
