import unittest

from agent.performance_monitor import (
    build_metrics,
    build_segments,
    filter_diagnostics,
    monitor_state,
)


def row(
    result,
    odd,
    signal,
    decision,
    tier="",
    clv_probability=0.60,
    sports_probability=0.60,
    expected_value=0.10,
    sports_edge_market=0.05,
    league="Portugal",
    clv=0.02,
    closing_source="The Odds API",
):
    return {
        "result_over25": result,
        "odd_over25_open": odd,
        "original_clv_signal": signal,
        "decision": decision,
        "tier": tier,
        "clv_probability": clv_probability,
        "sports_probability": sports_probability,
        "expected_value": expected_value,
        "sports_edge_market": sports_edge_market,
        "league": league,
        "clv": clv,
        "closing_source": closing_source,
    }


class PerformanceMonitorTests(unittest.TestCase):
    def test_build_metrics_uses_unit_counterfactual_profit(self):
        rows = [
            row(
                1,
                2.00,
                1,
                "QUALIFIED_SHADOW",
                "A",
            ),
            row(
                0,
                1.80,
                1,
                "WATCH",
            ),
            row(
                1,
                1.50,
                0,
                "REJECT",
            ),
        ]

        metrics = build_metrics(rows)

        self.assertEqual(metrics.sample, 3)
        self.assertEqual(metrics.wins, 2)
        self.assertEqual(metrics.losses, 1)
        self.assertAlmostEqual(
            metrics.profit_units,
            0.50,
            places=6,
        )
        self.assertAlmostEqual(
            metrics.roi,
            0.50 / 3.0,
            places=6,
        )

    def test_filter_diagnostics_measures_retention_and_rejection(self):
        rows = [
            row(
                1,
                2.00,
                1,
                "QUALIFIED_SHADOW",
                "A",
            ),
            row(
                0,
                2.00,
                1,
                "QUALIFIED_SHADOW",
                "B",
            ),
            row(
                1,
                2.00,
                1,
                "WATCH",
            ),
            row(
                0,
                2.00,
                1,
                "REJECT",
            ),
            row(
                0,
                2.00,
                0,
                "REJECT",
            ),
        ]

        diagnostics = filter_diagnostics(rows)

        self.assertEqual(
            diagnostics["original_sample"],
            4,
        )
        self.assertEqual(
            diagnostics["qualified_sample"],
            2,
        )
        self.assertAlmostEqual(
            diagnostics["winner_retention"],
            0.5,
        )
        self.assertAlmostEqual(
            diagnostics["loser_rejection"],
            0.5,
        )
        self.assertAlmostEqual(
            diagnostics["selection_rate"],
            0.5,
        )

    def test_segments_include_strategy_league_and_bins(self):
        rows = [
            row(
                1,
                1.95,
                1,
                "QUALIFIED_SHADOW",
                "A",
                clv_probability=0.62,
                sports_probability=0.64,
                expected_value=0.08,
            )
        ]

        segments = build_segments(rows)

        self.assertIn(
            ("strategy", "ORIGINAL_SIGNAL"),
            segments,
        )
        self.assertIn(
            ("strategy", "QUALIFIED_SHADOW"),
            segments,
        )
        self.assertIn(
            ("strategy", "QUALIFIED_A"),
            segments,
        )
        self.assertIn(
            ("league_qualified", "Portugal"),
            segments,
        )
        self.assertIn(
            (
                "clv_bin_qualified",
                "0.60-0.65",
            ),
            segments,
        )
        self.assertIn(
            (
                "ev_bin_qualified",
                "0.05-0.10",
            ),
            segments,
        )

    def test_monitor_state_requires_evidence(self):
        self.assertEqual(
            monitor_state(
                {"qualified_sample": 5}
            ),
            "COLD_START",
        )

        self.assertEqual(
            monitor_state(
                {"qualified_sample": 25}
            ),
            "EVIDENCE_BUILDING",
        )

        self.assertEqual(
            monitor_state(
                {
                    "qualified_sample": 60,
                    "original_roi": 0.10,
                    "qualified_roi": 0.00,
                }
            ),
            "REVIEW",
        )

        self.assertEqual(
            monitor_state(
                {
                    "qualified_sample": 60,
                    "original_roi": 0.02,
                    "qualified_roi": 0.08,
                }
            ),
            "TRACKING",
        )


if __name__ == "__main__":
    unittest.main()
