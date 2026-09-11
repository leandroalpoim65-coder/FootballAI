import unittest

from agent.outcome_learning import (
    CHAMPION_POLICY,
    OutcomePolicy,
    build_patterns,
    challenger_gate,
    metrics,
    qualifies,
    unique_date_split,
)


def row(
    idx,
    date,
    result,
    odd=1.90,
    goal=60,
    attack=60,
    ml=70,
    value=65,
    ev=0.08,
    decision="BET_SHADOW",
):
    return {
        "id": idx,
        "fixture_date": date,
        "fixture_time": "15:00",
        "result_over25": result,
        "odd_over25_open": odd,
        "goal_score": goal,
        "attack_score": attack,
        "ml_score": ml,
        "value_score": value,
        "expected_value": ev,
        "goal_vote": int(goal >= 55),
        "attack_vote": int(attack >= 55),
        "ml_vote": int(ml >= 65),
        "value_vote": int(value >= 60),
        "vote_count": sum((goal >= 55, attack >= 55, ml >= 65, value >= 60)),
        "decision": decision,
    }


class OutcomeLearningTests(unittest.TestCase):
    def test_champion_rule_matches_three_of_four(self):
        r = row(1, "2026-09-01", 1, value=50)
        self.assertTrue(qualifies(r, CHAMPION_POLICY))

        r2 = row(2, "2026-09-01", 1, attack=40, value=50)
        self.assertFalse(qualifies(r2, CHAMPION_POLICY))

    def test_metrics_use_counterfactual_profit_for_passes_too(self):
        policy = OutcomePolicy(0, 0, 0, 0, 1, 0)
        rows = [
            row(1, "2026-09-01", 1, odd=2.0, decision="PASS"),
            row(2, "2026-09-02", 0, odd=2.0, decision="PASS"),
        ]
        m = metrics(rows, policy)
        self.assertEqual(m.n, 2)
        self.assertAlmostEqual(m.profit, 0.0)
        self.assertAlmostEqual(m.roi, 0.0)

    def test_temporal_split_keeps_dates_separate(self):
        rows = [
            row(1, "2026-09-01", 1),
            row(2, "2026-09-01", 0),
            row(3, "2026-09-02", 1),
            row(4, "2026-09-03", 1),
        ]
        train, validation, split_date = unique_date_split(rows, 0.67)
        train_dates = {r["fixture_date"] for r in train}
        valid_dates = {r["fixture_date"] for r in validation}
        self.assertTrue(train_dates.isdisjoint(valid_dates))
        self.assertEqual(split_date, min(valid_dates))

    def test_patterns_include_pass_and_vote_signature(self):
        rows = [
            row(1, "2026-09-01", 1, decision="PASS"),
            row(2, "2026-09-02", 0, value=50, decision="BET_SHADOW"),
        ]
        patterns = build_patterns(rows)
        decisions = {
            p["segment"] for p in patterns if p["dimension"] == "decision"
        }
        signatures = {
            p["segment"] for p in patterns if p["dimension"] == "vote_signature"
        }
        self.assertIn("PASS", decisions)
        self.assertIn("BET_SHADOW", decisions)
        self.assertTrue(any("G" in s and "M" in s for s in signatures))

    def test_challenger_gate_rejects_tiny_sample(self):
        champ = metrics(
            [row(i, f"2026-09-{i:02d}", int(i % 2 == 0)) for i in range(1, 21)],
            CHAMPION_POLICY,
        )
        candidate_policy = OutcomePolicy(0, 0, 0, 0, 1, 0)
        candidate = metrics(
            [row(1, "2026-09-01", 1), row(2, "2026-09-02", 1)],
            candidate_policy,
        )
        passed, reasons = challenger_gate(champ, candidate)
        self.assertFalse(passed)
        self.assertTrue(any("candidate_validation_bets" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
