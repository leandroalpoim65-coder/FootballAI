import unittest

from agent.decision_engine import DEFAULT_POLICY
from agent.policy_learning import (
    MIN_RESOLVED_ROWS,
    candidate_grid,
    learn_policy,
    policy_metrics,
    temporal_split,
)


def make_row(
    index,
    *,
    high_quality,
    result,
):
    if high_quality:
        clv_probability = 0.62
        expected_value = 0.08
        sports_edge_market = 0.04
    else:
        clv_probability = 0.56
        expected_value = 0.035
        sports_edge_market = 0.022

    return {
        "paper_trade_id": index,
        "fixture_date": (
            f"2026-{1 + (index // 28):02d}-"
            f"{1 + (index % 28):02d}"
        ),
        "original_clv_signal": 1,
        "odd_over25_open": 2.0,
        "clv_probability": clv_probability,
        "expected_value": expected_value,
        "sports_edge_market": sports_edge_market,
        "result_over25": result,
        "clv": 0.02 if high_quality else -0.01,
        "closing_source": "The Odds API",
    }


class PolicyLearningTests(unittest.TestCase):
    def test_waits_for_minimum_resolved_sample(self):
        rows = [
            make_row(
                i,
                high_quality=True,
                result=1,
            )
            for i in range(
                MIN_RESOLVED_ROWS - 1
            )
        ]

        result = learn_policy(
            rows,
            DEFAULT_POLICY,
        )

        self.assertEqual(
            result.decision,
            "WAIT",
        )

    def test_temporal_split_keeps_future_in_validation(self):
        rows = [
            make_row(
                i,
                high_quality=True,
                result=1,
            )
            for i in range(140)
        ]

        train, validation, split_date = (
            temporal_split(rows)
        )

        self.assertGreaterEqual(
            len(validation),
            40,
        )
        self.assertLess(
            train[-1]["fixture_date"],
            validation[0]["fixture_date"],
        )
        self.assertEqual(
            split_date,
            validation[0]["fixture_date"],
        )

    def test_grid_never_relaxes_below_default_floor(self):
        candidates = candidate_grid(
            DEFAULT_POLICY
        )

        self.assertTrue(candidates)

        self.assertGreaterEqual(
            min(
                item.min_clv_probability
                for item in candidates
            ),
            DEFAULT_POLICY.min_clv_probability,
        )
        self.assertGreaterEqual(
            min(
                item.min_expected_value
                for item in candidates
            ),
            DEFAULT_POLICY.min_expected_value,
        )
        self.assertGreaterEqual(
            min(
                item.min_edge_market
                for item in candidates
            ),
            DEFAULT_POLICY.min_edge_market,
        )

    def test_promotes_stricter_policy_when_holdout_confirms(self):
        rows = []

        for i in range(160):
            high = (i % 2 == 0)

            if high:
                # 75% wins: strong positive ROI at odd 2.0.
                result = 0 if (i % 8 == 6) else 1
            else:
                # 25% wins: negative ROI, despite passing default gates.
                result = 1 if (i % 8 == 1) else 0

            rows.append(
                make_row(
                    i,
                    high_quality=high,
                    result=result,
                )
            )

        active_metrics = policy_metrics(
            rows,
            DEFAULT_POLICY,
        )

        result = learn_policy(
            rows,
            DEFAULT_POLICY,
        )

        self.assertEqual(
            result.decision,
            "PROMOTE",
        )
        self.assertIsNotNone(
            result.candidate,
        )
        self.assertGreater(
            result.candidate.min_clv_probability,
            DEFAULT_POLICY.min_clv_probability,
        )
        self.assertGreater(
            result.candidate_validation.roi,
            result.active_validation.roi,
        )
        self.assertGreater(
            result.candidate_validation.roi_lcb,
            0.0,
        )
        self.assertGreater(
            result.candidate_train.roi,
            active_metrics.roi,
        )


if __name__ == "__main__":
    unittest.main()
