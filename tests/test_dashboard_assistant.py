import pandas as pd

from dashboard_assistant import match_open_trades, signal_candidates, to_json_records


def test_suggestions_require_future_fixture_and_matching_open_signal():
    upcoming = pd.DataFrame(
        [
            {
                "date": "2026-11-02",
                "time": "18:00",
                "league": "Portugal",
                "home_team": "Home A",
                "away_team": "Away A",
                "minutes_to_kickoff": 90,
                "odd_over25_current": 1.8,
            },
            {
                "date": "2026-11-02",
                "time": "19:00",
                "league": "Portugal",
                "home_team": "Home B",
                "away_team": "Away B",
                "minutes_to_kickoff": 150,
                "odd_over25_current": 1.9,
            },
            {
                "date": "2026-11-02",
                "time": "16:00",
                "league": "Portugal",
                "home_team": "Past Game",
                "away_team": "Past Opponent",
                "minutes_to_kickoff": -30,
                "odd_over25_current": 2.0,
            },
        ]
    )
    paper = pd.DataFrame(
        [
            {
                "fixture_date": "2026-11-02",
                "league": "portugal",
                "home_team": "home a",
                "away_team": "away a",
                "status": "OPEN",
                "signal": 1,
                "clv_probability": 0.58,
                "odd_over25_open": 1.75,
            },
            {
                "fixture_date": "2026-11-02",
                "league": "Portugal",
                "home_team": "Home B",
                "away_team": "Away B",
                "status": "OPEN",
                "signal": 0,
                "clv_probability": 0.44,
                "odd_over25_open": 1.9,
            },
            {
                "fixture_date": "2026-11-02",
                "league": "Portugal",
                "home_team": "Past Game",
                "away_team": "Past Opponent",
                "status": "OPEN",
                "signal": 1,
                "clv_probability": 0.61,
                "odd_over25_open": 2.0,
            },
        ]
    )

    matches = match_open_trades(upcoming, paper)
    candidates = signal_candidates(upcoming, paper)

    assert len(matches) == 2
    assert len(candidates) == 1
    assert candidates.iloc[0]["home_team"] == "Home A"
    assert candidates.iloc[0]["clv_probability"] == 0.58


def test_json_records_convert_missing_values_to_null():
    records = to_json_records(
        pd.DataFrame([{"probability": 0.55, "missing": float("nan")}]),
        ["probability", "missing"],
    )

    assert records == [{"probability": 0.55, "missing": None}]


def test_suggestion_matches_source_fixture_aliases_to_canonical_paper_trade():
    upcoming = pd.DataFrame(
        [
            {
                "date": "2026-11-02",
                "time": "20:45",
                "league": "Serie A - Italy",
                "home_team": "Atalanta BC",
                "away_team": "Parma",
                "minutes_to_kickoff": 90,
            }
        ]
    )
    paper = pd.DataFrame(
        [
            {
                "fixture_date": "2026-11-02",
                "league": "Italy",
                "home_team": "Atalanta",
                "away_team": "Parma",
                "status": "OPEN",
                "signal": 1,
                "clv_probability": 0.58,
                "odd_over25_open": 1.8,
            }
        ]
    )

    candidates = signal_candidates(upcoming, paper)

    assert len(candidates) == 1
    assert candidates.iloc[0]["home_team"] == "Atalanta BC"