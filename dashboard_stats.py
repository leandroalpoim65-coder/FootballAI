from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml.goal_model import estimate_lambdas_from_features, project_match_from_lambdas
from ml.paper_trade import normalize_league, normalize_team

MIN_TEAM_HISTORY = 3
MIN_LEAGUE_HISTORY = 10
OVER_LEAN_THRESHOLD = 0.58
UNDER_LEAN_THRESHOLD = 0.42


def build_over25_support_table(
    upcoming: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    team_window: int = 5,
    league_window: int = 50,
) -> pd.DataFrame:
    if upcoming.empty:
        return pd.DataFrame()

    required = {"date", "league", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(historical.columns)
    if missing:
        raise ValueError(f"Historical matches are missing columns: {', '.join(sorted(missing))}")

    history = historical.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["home_goals"] = pd.to_numeric(history["home_goals"], errors="coerce")
    history["away_goals"] = pd.to_numeric(history["away_goals"], errors="coerce")
    history = history.dropna(subset=["date", "home_goals", "away_goals"]).copy()
    history["total_goals"] = history["home_goals"] + history["away_goals"]
    history["over25"] = history["total_goals"] > 2.5
    history["canonical_league"] = history["league"].map(_canonical_league)
    history["league_key"] = history["canonical_league"].str.casefold()
    history["home_key"] = [
        _canonical_team(league, team).casefold()
        for league, team in zip(
            history["canonical_league"],
            history["home_team"],
        )
    ]
    history["away_key"] = [
        _canonical_team(league, team).casefold()
        for league, team in zip(
            history["canonical_league"],
            history["away_team"],
        )
    ]

    output = []
    for _, fixture in upcoming.iterrows():
        fixture_date = pd.to_datetime(fixture.get("date"), errors="coerce")
        canonical_league = _canonical_league(fixture.get("league"))
        league_key = canonical_league.casefold()
        home_key = _canonical_team(
            canonical_league,
            fixture.get("home_team"),
        ).casefold()
        away_key = _canonical_team(
            canonical_league,
            fixture.get("away_team"),
        ).casefold()
        prior = history.loc[
            (history["date"] < fixture_date)
            & (history["league_key"] == league_key)
        ]
        home_last = (
            prior.loc[prior["home_key"] == home_key]
            .sort_values("date", ascending=False)
            .head(team_window)
        )
        away_last = (
            prior.loc[prior["away_key"] == away_key]
            .sort_values("date", ascending=False)
            .head(team_window)
        )
        home_form = _team_form(prior, home_key, team_window)
        away_form = _team_form(prior, away_key, team_window)
        league_last = prior.sort_values("date", ascending=False).head(league_window)

        over_open = _valid_odd(fixture.get("odd_over25_current")) or _valid_odd(
            fixture.get("odd_over25_open")
        )
        under_open = _valid_odd(fixture.get("odd_under25_current")) or _valid_odd(
            fixture.get("odd_under25_open")
        )
        market_probability = None
        if over_open and under_open:
            implied_over = 1.0 / over_open
            implied_under = 1.0 / under_open
            market_probability = implied_over / (implied_over + implied_under)

        preview = _build_goal_preview(
            home_form=home_form,
            away_form=away_form,
            home_venue=home_last,
            away_venue=away_last,
            league_matches=league_last,
        )
        latest_league_match = league_last["date"].max() if not league_last.empty else None
        history_age_days = (
            int((fixture_date.normalize() - latest_league_match.normalize()).days)
            if latest_league_match is not None and pd.notna(fixture_date)
            else None
        )
        output.append(
            {
                "date": fixture.get("date"),
                "time": fixture.get("time"),
                "league": fixture.get("league"),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "home_o25_rate": _rate(home_last),
                "home_sample": len(home_last),
                "home_form_scored": home_form["scored"],
                "home_form_conceded": home_form["conceded"],
                "home_form_sample": home_form["sample"],
                "home_venue_scored": _average(home_last, "home_goals"),
                "home_venue_conceded": _average(home_last, "away_goals"),
                "home_avg_total_goals": _average(home_last, "total_goals"),
                "away_o25_rate": _rate(away_last),
                "away_sample": len(away_last),
                "away_form_scored": away_form["scored"],
                "away_form_conceded": away_form["conceded"],
                "away_form_sample": away_form["sample"],
                "away_venue_scored": _average(away_last, "away_goals"),
                "away_venue_conceded": _average(away_last, "home_goals"),
                "away_avg_total_goals": _average(away_last, "total_goals"),
                "league_o25_rate": _rate(league_last),
                "league_sample": len(league_last),
                "league_history_latest_date": (
                    latest_league_match.strftime("%Y-%m-%d")
                    if latest_league_match is not None
                    else None
                ),
                "league_history_age_days": history_age_days,
                "market_o25_probability": market_probability,
                "model_o25_edge_pp": (
                    (preview["model_o25_probability"] - market_probability) * 100.0
                    if preview["model_o25_probability"] is not None
                    and market_probability is not None
                    else None
                ),
                **preview,
            }
        )

    return pd.DataFrame(output)


def _canonical_league(value: object) -> str:
    normalized = normalize_league(value)
    if normalized is not None:
        return str(normalized).strip()
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_team(league: str, value: object) -> str:
    normalized = normalize_team(league, value)
    if normalized is not None:
        return str(normalized).strip()
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _team_form(
    prior: pd.DataFrame,
    team_key: str,
    window: int,
) -> dict[str, float | int | None]:
    at_home = prior.loc[prior["home_key"] == team_key].copy()
    at_home["team_scored"] = at_home["home_goals"]
    at_home["team_conceded"] = at_home["away_goals"]

    away = prior.loc[prior["away_key"] == team_key].copy()
    away["team_scored"] = away["away_goals"]
    away["team_conceded"] = away["home_goals"]

    recent = (
        pd.concat([at_home, away], ignore_index=True)
        .sort_values("date", ascending=False)
        .head(window)
    )
    if recent.empty:
        return {"scored": None, "conceded": None, "sample": 0}
    return {
        "scored": float(recent["team_scored"].mean()),
        "conceded": float(recent["team_conceded"].mean()),
        "sample": len(recent),
    }


def _build_goal_preview(
    *,
    home_form: dict[str, float | int | None],
    away_form: dict[str, float | int | None],
    home_venue: pd.DataFrame,
    away_venue: pd.DataFrame,
    league_matches: pd.DataFrame,
) -> dict[str, float | str | None]:
    missing = {
        "expected_home_goals": None,
        "expected_away_goals": None,
        "most_likely_score": None,
        "model_o25_probability": None,
        "model_btts_probability": None,
        "model_home_win_probability": None,
        "model_draw_probability": None,
        "model_away_win_probability": None,
        "stats_suggestion": "More pre-match history needed",
    }
    if (
        min(
            int(home_form["sample"] or 0),
            int(away_form["sample"] or 0),
            len(home_venue),
            len(away_venue),
        )
        < MIN_TEAM_HISTORY
        or len(league_matches) < MIN_LEAGUE_HISTORY
    ):
        return missing

    features = {
        "home_avg_scored_5": home_form["scored"],
        "home_avg_conceded_5": home_form["conceded"],
        "away_avg_scored_5": away_form["scored"],
        "away_avg_conceded_5": away_form["conceded"],
        "home_home_avg_scored_5": float(home_venue["home_goals"].mean()),
        "home_home_avg_conceded_5": float(home_venue["away_goals"].mean()),
        "away_away_avg_scored_5": float(away_venue["away_goals"].mean()),
        "away_away_avg_conceded_5": float(away_venue["home_goals"].mean()),
    }
    lambda_home, lambda_away = estimate_lambdas_from_features(
        features,
        league_home_avg=float(league_matches["home_goals"].mean()),
        league_away_avg=float(league_matches["away_goals"].mean()),
    )
    projection = project_match_from_lambdas(lambda_home, lambda_away)
    over_probability = float(projection["totals"]["2.5"]["over"])
    if over_probability >= OVER_LEAN_THRESHOLD:
        suggestion = "Over 2.5 lean"
    elif over_probability <= UNDER_LEAN_THRESHOLD:
        suggestion = "Under 2.5 lean"
    else:
        suggestion = "No clear totals lean"

    top_score = projection["top_scores"][0]
    return {
        "expected_home_goals": lambda_home,
        "expected_away_goals": lambda_away,
        "most_likely_score": (
            f"{int(top_score['home_goals'])}-{int(top_score['away_goals'])}"
        ),
        "model_o25_probability": over_probability,
        "model_btts_probability": float(projection["btts"]["yes"]),
        "model_home_win_probability": float(projection["match_result"]["home"]),
        "model_draw_probability": float(projection["match_result"]["draw"]),
        "model_away_win_probability": float(projection["match_result"]["away"]),
        "stats_suggestion": suggestion,
    }


def build_strategy_fixture_results(
    fixtures: pd.DataFrame,
    *,
    now: object | None = None,
) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame(
            columns=[
                "Kickoff",
                "League",
                "Game",
                "Score",
                "Status",
                "Over 2.5 result",
                "O2.5 open",
                "O2.5 current",
            ]
        )

    current_time = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    date_values = fixtures.get("date", pd.Series("", index=fixtures.index)).astype(str)
    time_values = (
        fixtures.get("time", pd.Series("00:00", index=fixtures.index))
        .fillna("00:00")
        .astype(str)
    )
    kickoff = pd.to_datetime(date_values + " " + time_values, errors="coerce")
    home_goals = pd.to_numeric(
        fixtures.get("home_goals", pd.Series(np.nan, index=fixtures.index)),
        errors="coerce",
    )
    away_goals = pd.to_numeric(
        fixtures.get("away_goals", pd.Series(np.nan, index=fixtures.index)),
        errors="coerce",
    )
    has_result = home_goals.notna() & away_goals.notna()
    total_goals = home_goals + away_goals

    return pd.DataFrame(
        {
            "Kickoff": kickoff.dt.strftime("%d/%m/%Y %H:%M").fillna("—"),
            "League": fixtures.get("league", pd.Series("", index=fixtures.index)),
            "Game": (
                fixtures.get("home_team", pd.Series("", index=fixtures.index)).astype(str)
                + " vs "
                + fixtures.get("away_team", pd.Series("", index=fixtures.index)).astype(str)
            ),
            "Score": [
                f"{int(home)}-{int(away)}" if complete else "—"
                for home, away, complete in zip(home_goals, away_goals, has_result)
            ],
            "Status": [
                "Completed"
                if complete
                else (
                    "Awaiting result"
                    if pd.notna(fixture_kickoff) and fixture_kickoff <= current_time
                    else "Upcoming"
                )
                for fixture_kickoff, complete in zip(kickoff, has_result)
            ],
            "Over 2.5 result": [
                ("OVER" if goals >= 3 else "UNDER") if complete else "—"
                for goals, complete in zip(total_goals, has_result)
            ],
            "O2.5 open": pd.to_numeric(
                fixtures.get("odd_over25_open", pd.Series(np.nan, index=fixtures.index)),
                errors="coerce",
            ).round(2),
            "O2.5 current": pd.to_numeric(
                fixtures.get("odd_over25_current", pd.Series(np.nan, index=fixtures.index)),
                errors="coerce",
            ).round(2),
        },
        index=fixtures.index,
    )


def build_paper_trade_analysis(
    paper_trades: pd.DataFrame,
) -> dict[str, object]:
    required = {
        "fixture_date",
        "league",
        "signal",
        "status",
        "planned_stake",
        "profit",
    }
    missing = required - set(paper_trades.columns)
    if missing:
        raise ValueError(
            f"Paper trades are missing columns: {', '.join(sorted(missing))}"
        )

    trades = paper_trades.copy()
    trades["status"] = trades["status"].fillna("").astype(str).str.upper()
    trades["signal"] = pd.to_numeric(trades["signal"], errors="coerce")
    trades["planned_stake"] = pd.to_numeric(
        trades["planned_stake"],
        errors="coerce",
    )
    trades["profit"] = pd.to_numeric(trades["profit"], errors="coerce")
    trades["fixture_date"] = pd.to_datetime(
        trades["fixture_date"],
        errors="coerce",
    )
    trades = trades.loc[
        (trades["status"] == "RESOLVED")
        & (trades["signal"] == 1)
        & np.isfinite(trades["planned_stake"])
        & (trades["planned_stake"] > 0)
        & np.isfinite(trades["profit"])
        & trades["fixture_date"].notna()
    ].copy()

    if "fixture_time" in trades.columns:
        trades = trades.sort_values(
            ["fixture_date", "fixture_time"],
            kind="stable",
        )
    else:
        trades = trades.sort_values("fixture_date", kind="stable")
    trades = trades.reset_index(drop=True)
    trades["cumulative_profit_units"] = trades["profit"].cumsum()
    trades["cumulative_roi_pct"] = (
        trades["profit"].cumsum() / trades["planned_stake"].cumsum() * 100
    )
    running_peak = trades["cumulative_profit_units"].cummax().clip(lower=0)
    trades["drawdown_units"] = trades["cumulative_profit_units"] - running_peak

    total_stake = float(trades["planned_stake"].sum())
    total_profit = float(trades["profit"].sum())
    resolved_results = pd.to_numeric(
        trades.get("result_over25", pd.Series(np.nan, index=trades.index)),
        errors="coerce",
    )
    result_values = resolved_results.dropna()
    monthly = trades.assign(
        month=trades["fixture_date"].dt.to_period("M").astype(str)
    ).groupby("month", as_index=False, sort=True).agg(
        bets=("profit", "size"),
        stake_units=("planned_stake", "sum"),
        profit_units=("profit", "sum"),
    )
    monthly["roi_pct"] = (
        monthly["profit_units"] / monthly["stake_units"] * 100
    )

    leagues = trades.groupby("league", as_index=False, dropna=False).agg(
        bets=("profit", "size"),
        stake_units=("planned_stake", "sum"),
        profit_units=("profit", "sum"),
    )
    leagues["roi_pct"] = leagues["profit_units"] / leagues["stake_units"] * 100
    leagues = leagues.sort_values(
        ["profit_units", "bets"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)

    summary = {
        "bets": len(trades),
        "stake_units": total_stake,
        "profit_units": total_profit,
        "roi_pct": total_profit / total_stake * 100 if total_stake else None,
        "hit_rate_pct": (
            float(result_values.mean() * 100) if not result_values.empty else None
        ),
        "max_drawdown_units": (
            float(trades["drawdown_units"].min()) if not trades.empty else 0.0
        ),
    }
    return {
        "summary": summary,
        "trades": trades,
        "monthly": monthly,
        "leagues": leagues,
    }


def _valid_odd(value: object) -> float | None:
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None
    return odd if math.isfinite(odd) and odd > 1.0 else None


def _rate(matches: pd.DataFrame) -> float | None:
    return float(matches["over25"].mean()) if not matches.empty else None


def _average(matches: pd.DataFrame, column: str) -> float | None:
    return float(matches[column].mean()) if not matches.empty else None