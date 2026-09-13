from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

HOME_ADVANTAGE_ELO = 65.0
ELO_K = 20.0


def _roll(series: pd.Series, window: int, minimum: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=minimum).mean()


def team_feature_history(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "id": matches.id, "date": matches.date, "league": matches.league,
        "team": matches.home_team, "venue": "home",
        "gf": matches.home_goals, "ga": matches.away_goals,
    })
    away = pd.DataFrame({
        "id": matches.id, "date": matches.date, "league": matches.league,
        "team": matches.away_team, "venue": "away",
        "gf": matches.away_goals, "ga": matches.home_goals,
    })
    h = pd.concat([home, away], ignore_index=True)
    h["gd"] = h.gf - h.ga
    h["win"] = (h.gd > 0).astype(int)
    h["draw"] = (h.gd == 0).astype(int)
    h["points"] = np.where(h.win == 1, 3, np.where(h.draw == 1, 1, 0))
    h["btts"] = ((h.gf > 0) & (h.ga > 0)).astype(int)
    h["over25"] = ((h.gf + h.ga) >= 3).astype(int)
    h["scored"] = (h.gf > 0).astype(int)
    h["cs"] = (h.ga == 0).astype(int)
    h = h.sort_values(["league", "team", "date", "id"]).reset_index(drop=True)

    group = h.groupby(["league", "team"], sort=False)
    for col in ("gf", "ga", "gd", "points", "win", "btts", "over25", "scored", "cs"):
        h[f"{col}5"] = group[col].transform(lambda s: _roll(s, 5, 3))
    for col in ("gf", "ga", "over25", "btts"):
        h[f"{col}10"] = group[col].transform(lambda s: _roll(s, 10, 3))
    previous = group["date"].shift(1)
    h["rest"] = (h.date - previous).dt.days.clip(0, 60)

    venue = h.groupby(["league", "team", "venue"], sort=False)
    for col in ("gf", "ga", "points"):
        values = venue[col].transform(lambda s: _roll(s, 5, 2))
        h[f"v_{col}5"] = values.fillna(h[f"{col}5"])
    return h


def prematch_elo(matches: pd.DataFrame) -> pd.DataFrame:
    ratings = defaultdict(lambda: 1500.0)
    out = pd.DataFrame(index=matches.index, columns=["home_elo", "away_elo"], dtype=float)
    ordered = matches.sort_values(["date", "id"])
    for _, games in ordered.groupby("date", sort=True):
        pending = []
        for row in games.itertuples():
            hk = (str(row.league), str(row.home_team))
            ak = (str(row.league), str(row.away_team))
            hr = float(ratings[hk]); ar = float(ratings[ak])
            out.loc[row.Index] = [hr, ar]
            exp = 1.0 / (1.0 + 10.0 ** ((ar - (hr + HOME_ADVANTAGE_ELO)) / 400.0))
            actual = 1.0 if row.home_goals > row.away_goals else 0.0 if row.home_goals < row.away_goals else 0.5
            pending.append((hk, ak, hr, ar, exp, actual))
        for hk, ak, hr, ar, exp, actual in pending:
            ratings[hk] = hr + ELO_K * (actual - exp)
            ratings[ak] = ar + ELO_K * ((1.0 - actual) - (1.0 - exp))
    return out


def league_context(matches: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=matches.index, columns=["league_over25_rate_200", "league_avg_goals_200"], dtype=float)
    for _, league_games in matches.groupby("league", sort=False):
        overs = deque(maxlen=200)
        goals = deque(maxlen=200)
        ordered = league_games.sort_values(["date", "id"])
        for _, games in ordered.groupby("date", sort=True):
            out.loc[games.index, "league_over25_rate_200"] = float(np.mean(overs)) if overs else 0.50
            out.loc[games.index, "league_avg_goals_200"] = float(np.mean(goals)) if goals else 2.50
            for row in games.itertuples():
                overs.append(float(row.actual_over25))
                goals.append(float(row.home_goals + row.away_goals))
    return out
