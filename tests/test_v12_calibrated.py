from __future__ import annotations

import pandas as pd

from agent.v12_calibrated import devig_market_probability, evaluate_v12
from agent.v12_history import league_context, team_feature_history


def test_devig_market_probability_removes_margin():
    fair, margin = devig_market_probability(1.90, 1.90)
    assert abs(fair - 0.5) < 1e-9
    assert margin > 0.0


def test_v12_decision_requires_edge_and_ev():
    policy = {"min_edge":0.04, "min_ev":0.03, "min_odd":1.2, "max_odd":4.0}
    good = evaluate_v12(0.62, 1.90, 1.90, policy)
    weak = evaluate_v12(0.52, 1.90, 1.90, policy)
    assert good.decision == "BET_SHADOW"
    assert weak.decision == "PASS"


def _matches():
    return pd.DataFrame([
        {"id":1,"date":pd.Timestamp("2026-01-01"),"league":"L","home_team":"A","away_team":"B","home_goals":1,"away_goals":0,"actual_over25":0},
        {"id":2,"date":pd.Timestamp("2026-01-08"),"league":"L","home_team":"B","away_team":"A","home_goals":1,"away_goals":2,"actual_over25":1},
        {"id":3,"date":pd.Timestamp("2026-01-15"),"league":"L","home_team":"A","away_team":"B","home_goals":3,"away_goals":1,"actual_over25":1},
        {"id":4,"date":pd.Timestamp("2026-01-22"),"league":"L","home_team":"A","away_team":"B","home_goals":9,"away_goals":9,"actual_over25":1},
    ])


def test_team_rolling_feature_excludes_current_match():
    history = team_feature_history(_matches())
    row = history[(history.id == 4) & (history.team == "A")].iloc[0]
    # A scored 1, 2 and 3 in its three previous games. Current 9 must not leak.
    assert abs(float(row.gf5) - 2.0) < 1e-9


def test_league_context_excludes_same_day_results():
    games = _matches()
    extra = pd.DataFrame([
        {"id":5,"date":pd.Timestamp("2026-01-22"),"league":"L","home_team":"C","away_team":"D","home_goals":0,"away_goals":0,"actual_over25":0}
    ])
    games = pd.concat([games, extra], ignore_index=True)
    context = league_context(games)
    assert float(context.loc[3,"league_over25_rate_200"]) == float(context.loc[4,"league_over25_rate_200"])


def test_team_form_survives_promotion_with_exact_identity():
    games = pd.DataFrame([
        {"id":1,"date":pd.Timestamp("2026-01-01"),"league":"England Championship","home_team":"A","away_team":"B","home_goals":1,"away_goals":0,"actual_over25":0},
        {"id":2,"date":pd.Timestamp("2026-01-08"),"league":"England Championship","home_team":"B","away_team":"A","home_goals":1,"away_goals":2,"actual_over25":1},
        {"id":3,"date":pd.Timestamp("2026-01-15"),"league":"England Championship","home_team":"A","away_team":"B","home_goals":3,"away_goals":1,"actual_over25":1},
        {"id":4,"date":pd.Timestamp("2026-08-10"),"league":"England","home_team":"A","away_team":"C","home_goals":0,"away_goals":0,"actual_over25":0},
    ])
    history = team_feature_history(games)
    promoted = history[(history.id == 4) & (history.team == "A")].iloc[0]
    assert abs(float(promoted.gf5) - 2.0) < 1e-9
