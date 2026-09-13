from __future__ import annotations

import numpy as np
import pandas as pd

from agent.v12_calibrated import V12_SPORT_FEATURES
from agent.v12_history import league_context, prematch_elo, team_feature_history


def build_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    history = team_feature_history(matches)
    home = history[history.venue == "home"].set_index("id")
    away = history[history.venue == "away"].set_index("id")
    data = matches.copy().set_index("id")

    home_map = {
        "gf5":"home_avg_scored_5", "ga5":"home_avg_conceded_5", "gd5":"home_avg_goal_diff_5",
        "points5":"home_points_avg_5", "win5":"home_win_rate_5", "btts5":"home_btts_rate_5",
        "over255":"home_over25_rate_5", "scored5":"home_scored_rate_5", "cs5":"home_clean_sheet_rate_5",
        "v_gf5":"home_home_avg_scored_5", "v_ga5":"home_home_avg_conceded_5", "v_points5":"home_home_points_avg_5",
        "rest":"home_rest_days", "gf10":"home_avg_scored_10", "ga10":"home_avg_conceded_10",
        "over2510":"home_over25_rate_10", "btts10":"home_btts_rate_10",
    }
    away_map = {
        "gf5":"away_avg_scored_5", "ga5":"away_avg_conceded_5", "gd5":"away_avg_goal_diff_5",
        "points5":"away_points_avg_5", "win5":"away_win_rate_5", "btts5":"away_btts_rate_5",
        "over255":"away_over25_rate_5", "scored5":"away_scored_rate_5", "cs5":"away_clean_sheet_rate_5",
        "v_gf5":"away_away_avg_scored_5", "v_ga5":"away_away_avg_conceded_5", "v_points5":"away_away_points_avg_5",
        "rest":"away_rest_days", "gf10":"away_avg_scored_10", "ga10":"away_avg_conceded_10",
        "over2510":"away_over25_rate_10", "btts10":"away_btts_rate_10",
    }
    for src, dst in home_map.items():
        data[dst] = home[src]
    for src, dst in away_map.items():
        data[dst] = away[src]

    elo = prematch_elo(matches).set_axis(matches.id.values, axis=0)
    data["home_elo"] = elo.home_elo
    data["away_elo"] = elo.away_elo
    data["elo_diff"] = data.home_elo - data.away_elo

    data["expected_home_goals"] = (data.home_avg_scored_5 + data.away_avg_conceded_5) / 2.0
    data["expected_away_goals"] = (data.away_avg_scored_5 + data.home_avg_conceded_5) / 2.0
    data["expected_total_goals"] = data.expected_home_goals + data.expected_away_goals
    data["expected_goal_difference"] = data.expected_home_goals - data.expected_away_goals
    data["total_expected_scored"] = data.home_avg_scored_5 + data.away_avg_scored_5
    data["total_expected_conceded"] = data.home_avg_conceded_5 + data.away_avg_conceded_5
    data["home_power_rating"] = data.home_avg_goal_diff_5 + data.home_points_avg_5
    data["away_power_rating"] = data.away_avg_goal_diff_5 + data.away_points_avg_5
    data["power_rating_diff"] = data.home_power_rating - data.away_power_rating
    data["form_points_diff"] = data.home_points_avg_5 - data.away_points_avg_5
    data["btts_signal"] = (data.home_btts_rate_5 + data.away_btts_rate_5) / 2.0
    data["over25_signal"] = (data.home_over25_rate_5 + data.away_over25_rate_5) / 2.0
    data["scoring_signal"] = (data.home_scored_rate_5 + data.away_scored_rate_5) / 2.0
    data["clean_sheet_signal"] = (data.home_clean_sheet_rate_5 + data.away_clean_sheet_rate_5) / 2.0

    context = league_context(matches).set_axis(matches.id.values, axis=0)
    data["league_over25_rate_200"] = context.league_over25_rate_200
    data["league_avg_goals_200"] = context.league_avg_goals_200

    over = pd.to_numeric(data.odd_over25_open, errors="coerce")
    under = pd.to_numeric(data.odd_under25_open, errors="coerce")
    valid_market = (over > 1.0) & (under > 1.0)
    implied_over = np.where(valid_market, 1.0 / over, np.nan)
    implied_under = np.where(valid_market, 1.0 / under, np.nan)
    total = implied_over + implied_under
    data["market_probability"] = implied_over / total
    data["market_overround"] = total - 1.0
    data["log_odd_over25"] = np.where(valid_market, np.log(over), np.nan)
    data["market_eligible"] = valid_market.astype(int)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["date", "actual_over25", *V12_SPORT_FEATURES])
    return data.sort_values("date").reset_index()
