import pandas as pd
import pytest

from dashboard_stats import (
    build_paper_trade_analysis,
    build_over25_support_table,
    build_strategy_fixture_results,
)


def test_fixture_stats_use_prior_venue_matches_and_novig_market():
    historical = pd.DataFrame(
        [
            ("2026-01-01", "League", "Home", f"Opponent H{i}", *goals)
            for i, goals in enumerate([(2, 1), (1, 1), (2, 2), (0, 0), (3, 0)])
        ]
        + [
            ("2026-01-02", "League", f"Opponent A{i}", "Away", *goals)
            for i, goals in enumerate([(1, 2), (0, 3), (1, 0), (2, 1), (2, 2)])
        ]
        + [
            ("2026-01-10", "League", "Same-day home", "Same-day away", 4, 0),
            ("2026-02-01", "League", "Home", "Away", 5, 0),
        ],
        columns=["date", "league", "home_team", "away_team", "home_goals", "away_goals"],
    )
    upcoming = pd.DataFrame(
        [
            {
                "date": "2026-01-10",
                "time": "20:00",
                "league": "League",
                "home_team": "Home",
                "away_team": "Away",
                "odd_over25_current": 2.0,
                "odd_under25_current": 2.0,
            }
        ]
    )

    stats = build_over25_support_table(upcoming, historical).iloc[0]

    assert stats["home_sample"] == 5
    assert stats["home_o25_rate"] == pytest.approx(0.6)
    assert stats["home_avg_total_goals"] == pytest.approx(2.4)
    assert stats["away_sample"] == 5
    assert stats["away_o25_rate"] == pytest.approx(0.8)
    assert stats["market_o25_probability"] == pytest.approx(0.5)
    assert stats["league_sample"] == 10
    assert stats["league_history_latest_date"] == "2026-01-02"
    assert stats["league_history_age_days"] == 8
    assert stats["home_form_scored"] == pytest.approx(1.6)
    assert stats["home_form_conceded"] == pytest.approx(0.8)
    assert stats["away_form_scored"] == pytest.approx(1.6)
    assert stats["away_form_conceded"] == pytest.approx(1.2)
    assert stats["expected_home_goals"] > 0
    assert stats["expected_away_goals"] > 0
    assert stats["most_likely_score"]
    assert 0 <= stats["model_o25_probability"] <= 1
    assert 0 <= stats["model_btts_probability"] <= 1
    assert stats["model_o25_edge_pp"] == pytest.approx(
        (stats["model_o25_probability"] - 0.5) * 100
    )
    assert stats["stats_suggestion"] in {
        "Over 2.5 lean",
        "Under 2.5 lean",
        "No clear totals lean",
    }


def test_missing_history_returns_null_rates_without_failing():
    upcoming = pd.DataFrame(
        [{"date": "2026-01-10", "league": "League", "home_team": "Home", "away_team": "Away"}]
    )
    historical = pd.DataFrame(
        columns=["date", "league", "home_team", "away_team", "home_goals", "away_goals"]
    )

    stats = build_over25_support_table(upcoming, historical).iloc[0]

    assert stats["home_sample"] == 0
    assert pd.isna(stats["home_o25_rate"])
    assert pd.isna(stats["league_o25_rate"])
    assert pd.isna(stats["market_o25_probability"])
    assert stats["stats_suggestion"] == "More pre-match history needed"
    assert pd.isna(stats["model_o25_probability"])


def test_source_league_and_team_aliases_link_to_historical_matches():
    historical = pd.DataFrame(
        [
            ("2026-01-01", "Italy", "Atalanta", "Opponent 1", 2, 1),
            ("2026-01-02", "Italy", "Opponent 2", "Parma", 1, 2),
            ("2026-01-03", "Italy", "Atalanta", "Opponent 3", 1, 0),
            ("2026-01-04", "Italy", "Opponent 4", "Parma", 0, 1),
            ("2026-01-05", "Italy", "Atalanta", "Opponent 5", 3, 1),
            ("2026-01-06", "Italy", "Opponent 6", "Parma", 2, 2),
            ("2026-01-07", "Italy", "Other 1", "Other 2", 1, 1),
            ("2026-01-08", "Italy", "Other 3", "Other 4", 2, 0),
            ("2026-01-09", "Italy", "Other 5", "Other 6", 0, 0),
            ("2026-01-10", "Italy", "Other 7", "Other 8", 3, 2),
        ],
        columns=[
            "date",
            "league",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ],
    )
    upcoming = pd.DataFrame(
        [
            {
                "date": "2026-02-01",
                "league": "Serie A - Italy",
                "home_team": "Atalanta BC",
                "away_team": "Parma",
            }
        ]
    )

    stats = build_over25_support_table(upcoming, historical).iloc[0]

    assert stats["home_form_sample"] == 3
    assert stats["away_form_sample"] == 3
    assert stats["home_sample"] == 3
    assert stats["away_sample"] == 3
    assert stats["league_sample"] == 10
    assert pd.notna(stats["model_o25_probability"])


def test_strategy_fixture_results_show_scores_when_available_and_pending_otherwise():
    fixtures = pd.DataFrame(
        [
            {
                "date": "2026-01-10",
                "time": "20:00",
                "league": "League",
                "home_team": "Home A",
                "away_team": "Away A",
                "home_goals": 2,
                "away_goals": 1,
                "odd_over25_open": 1.9,
                "odd_over25_current": 1.8,
            },
            {
                "date": "2026-01-12",
                "time": "20:00",
                "league": "League",
                "home_team": "Home B",
                "away_team": "Away B",
                "home_goals": None,
                "away_goals": None,
                "odd_over25_open": 2.0,
                "odd_over25_current": 2.1,
            },
            {
                "date": "2026-01-09",
                "time": "20:00",
                "league": "League",
                "home_team": "Home C",
                "away_team": "Away C",
                "home_goals": None,
                "away_goals": None,
                "odd_over25_open": 2.0,
                "odd_over25_current": 2.1,
            },
        ]
    )

    results = build_strategy_fixture_results(
        fixtures,
        now=pd.Timestamp("2026-01-11 12:00"),
    )

    assert results["Score"].tolist() == ["2-1", "—", "—"]
    assert results["Status"].tolist() == ["Completed", "Upcoming", "Awaiting result"]
    assert results["Over 2.5 result"].tolist() == ["OVER", "—", "—"]


def test_paper_trade_analysis_uses_only_settled_signals_and_stake_weighted_roi():
    trades = pd.DataFrame(
        [
            ("2026-01-02", "League A", 1, "RESOLVED", 2.0, 1.0, 1),
            ("2026-01-03", "League A", 1, "RESOLVED", 1.0, -1.0, 0),
            ("2026-02-01", "League B", 0, "RESOLVED", 0.0, 0.0, 1),
            ("2026-02-02", "League B", 1, "OPEN", 1.0, None, None),
            ("2026-02-03", "League B", 1, "VOID", 1.0, 0.0, None),
        ],
        columns=[
            "fixture_date",
            "league",
            "signal",
            "status",
            "planned_stake",
            "profit",
            "result_over25",
        ],
    )

    analysis = build_paper_trade_analysis(trades)

    assert analysis["summary"] == {
        "bets": 2,
        "stake_units": 3.0,
        "profit_units": 0.0,
        "roi_pct": 0.0,
        "hit_rate_pct": 50.0,
        "max_drawdown_units": -1.0,
    }
    assert analysis["trades"]["cumulative_profit_units"].tolist() == [1.0, 0.0]
    assert analysis["trades"]["drawdown_units"].tolist() == [0.0, -1.0]
    assert analysis["monthly"]["month"].tolist() == ["2026-01"]
    assert analysis["leagues"]["league"].tolist() == ["League A"]


def test_paper_trade_analysis_handles_no_settled_signals():
    trades = pd.DataFrame(
        [
            ("2026-01-02", "League", 0, "RESOLVED", 0.0, 0.0),
            ("2026-01-03", "League", 1, "OPEN", 1.0, None),
        ],
        columns=[
            "fixture_date",
            "league",
            "signal",
            "status",
            "planned_stake",
            "profit",
        ],
    )

    analysis = build_paper_trade_analysis(trades)

    assert analysis["summary"]["bets"] == 0
    assert analysis["summary"]["roi_pct"] is None
    assert analysis["monthly"].empty
    assert analysis["leagues"].empty