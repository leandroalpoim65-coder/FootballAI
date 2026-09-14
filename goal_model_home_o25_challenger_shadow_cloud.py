from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import paper_trade_uefa_cloud as uefa_cloud
import goal_model_v2_multileague_research as v2
import goal_model_v4_structural_research as v4
import goal_model_v6_home_o25_research as v6
import goal_model_v6_home_o25_train as v6train
from ml.paper_trade import calculate_season, create_team_history, normalize_league, normalize_team, parse_date


core = uefa_cloud.core

TABLE = "goal_model_home_o25_shadow"
ODDS_TABLE = "multi_market_odds"
MODEL_PATH = Path("ml") / "goal_model_v6_home_o25_trained.pkl"
MARKET = "team_total_home"
LINE = 2.5
MIN_HISTORY = 5
MAX_HISTORY = 20
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 65.0

ALLOWED_LEAGUES = {
    "England",
    "Germany",
    "Spain",
    "Italy",
    "France",
    "Netherlands",
    "Portugal",
}
MODES = ("STRICT", "WARM")


def env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def safe_odd(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x <= 1.0:
        return None
    return x


def no_vig_over(over_odd: float | None, under_odd: float | None) -> float | None:
    if over_odd is None or under_odd is None:
        return None
    io = 1.0 / over_odd
    iu = 1.0 / under_odd
    return io / (io + iu)


def expected_value(probability: float, odd: float | None) -> float | None:
    if odd is None:
        return None
    return float(probability * odd - 1.0)


def fair_odd(probability: float) -> float | None:
    if probability <= 0.0 or probability >= 1.0:
        return None
    return 1.0 / probability


def table_columns(connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def ensure_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_key TEXT NOT NULL,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            season TEXT NOT NULL,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            mode TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prediction_timestamp TEXT NOT NULL,
            probability_over REAL NOT NULL,
            probability_under REAL NOT NULL,
            fair_odd_over REAL,
            fair_odd_under REAL,
            market_open_odd_over REAL,
            market_open_odd_under REAL,
            odd_over_at_prediction REAL,
            odd_under_at_prediction REAL,
            market_probability_over_at_prediction REAL,
            ev_over_at_prediction REAL,
            ev_under_at_prediction REAL,
            best_side_at_prediction TEXT,
            best_ev_at_prediction REAL,
            best_positive_ev INTEGER NOT NULL DEFAULT 0,
            odd_over_last REAL,
            odd_under_last REAL,
            market_probability_over_last REAL,
            odds_first_observed_at TEXT,
            odds_last_observed_at TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN',
            home_goals INTEGER,
            away_goals INTEGER,
            result_over INTEGER,
            profit_over_at_prediction REAL,
            profit_under_at_prediction REAL,
            best_profit REAL,
            clv_over REAL,
            clv_under REAL,
            metadata_json TEXT NOT NULL,
            settled_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_{TABLE}_fixture_model
        ON {TABLE}(fixture_key, mode, model_version)
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS ix_{TABLE}_status
        ON {TABLE}(status, fixture_date)
        """
    )
    connection.commit()


def season_start(fixture_date: pd.Timestamp) -> pd.Timestamp:
    year = fixture_date.year if fixture_date.month >= 7 else fixture_date.year - 1
    return pd.Timestamp(year=year, month=7, day=1)


def elo_diff(season_matches: pd.DataFrame, home_team: str, away_team: str) -> float:
    ratings: dict[str, float] = {}
    ordered = season_matches.sort_values(["date", "id"])
    for row in ordered.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        rh = ratings.get(home, 1500.0)
        ra = ratings.get(away, 1500.0)
        expected = 1.0 / (1.0 + 10.0 ** (-((rh + ELO_HOME_ADVANTAGE) - ra) / 400.0))
        if row.home_goals > row.away_goals:
            actual = 1.0
        elif row.home_goals < row.away_goals:
            actual = 0.0
        else:
            actual = 0.5
        delta = ELO_K * (actual - expected)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return float(ratings.get(home_team, 1500.0) - ratings.get(away_team, 1500.0) + ELO_HOME_ADVANTAGE)


def team_arrays(history: pd.DataFrame, league: str, team: str, fixture_date: pd.Timestamp, mode: str) -> tuple[list[int], list[int], list[int], list[int]]:
    rows = history[
        (history["league"] == league)
        & (history["team"] == team)
        & (history["date"] < fixture_date)
    ].sort_values(["date", "id"])

    if mode == "STRICT":
        rows = rows[rows["date"] >= season_start(fixture_date)]
    elif mode != "WARM":
        raise ValueError(mode)

    all_rows = rows.tail(MAX_HISTORY)
    return (
        all_rows["goals_for"].astype(int).tolist(),
        all_rows["goals_against"].astype(int).tolist(),
        rows.loc[rows["venue"] == "home"].tail(MAX_HISTORY)["goals_for"].astype(int).tolist(),
        rows.loc[rows["venue"] == "home"].tail(MAX_HISTORY)["goals_against"].astype(int).tolist(),
    )


def away_arrays(history: pd.DataFrame, league: str, team: str, fixture_date: pd.Timestamp, mode: str) -> tuple[list[int], list[int], list[int], list[int]]:
    rows = history[
        (history["league"] == league)
        & (history["team"] == team)
        & (history["date"] < fixture_date)
    ].sort_values(["date", "id"])

    if mode == "STRICT":
        rows = rows[rows["date"] >= season_start(fixture_date)]
    elif mode != "WARM":
        raise ValueError(mode)

    all_rows = rows.tail(MAX_HISTORY)
    return (
        all_rows["goals_for"].astype(int).tolist(),
        all_rows["goals_against"].astype(int).tolist(),
        rows.loc[rows["venue"] == "away"].tail(MAX_HISTORY)["goals_for"].astype(int).tolist(),
        rows.loc[rows["venue"] == "away"].tail(MAX_HISTORY)["goals_against"].astype(int).tolist(),
    )


def build_snapshot(
    matches: pd.DataFrame,
    history: pd.DataFrame,
    *,
    fixture_date: pd.Timestamp,
    league: str,
    home_team: str,
    away_team: str,
    mode: str,
) -> dict:
    before = matches[
        (matches["league"] == league)
        & (matches["date"] < fixture_date)
    ].sort_values(["date", "id"])
    season_rows = before[before["date"] >= season_start(fixture_date)]

    if season_rows.empty:
        league_home_avg = np.nan
        league_away_avg = np.nan
    else:
        league_home_avg = float(season_rows["home_goals"].mean())
        league_away_avg = float(season_rows["away_goals"].mean())

    h_all_s, h_all_c, h_home_s, h_home_c = team_arrays(
        history, league, home_team, fixture_date, mode
    )
    a_all_s, a_all_c, a_away_s, a_away_c = away_arrays(
        history, league, away_team, fixture_date, mode
    )

    counts = {
        "home_all": len(h_all_s),
        "home_venue": len(h_home_s),
        "away_all": len(a_all_s),
        "away_venue": len(a_away_s),
    }
    if min(counts.values()) < MIN_HISTORY:
        raise ValueError(f"historico insuficiente {mode}: {counts}")

    return {
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
        "elo_diff": elo_diff(season_rows, home_team, away_team),
        "h_all_scored": h_all_s,
        "h_all_conceded": h_all_c,
        "a_all_scored": a_all_s,
        "a_all_conceded": a_all_c,
        "h_home_scored": h_home_s,
        "h_home_conceded": h_home_c,
        "a_away_scored": a_away_s,
        "a_away_conceded": a_away_c,
        "history_counts": counts,
    }


def predict_versions(snapshot: dict, artifact: dict) -> dict[str, dict]:
    data = [snapshot]
    lh2, la2 = v2.lambdas_for_config(data, v4.V2_CONFIG)
    p2 = float(v2.probabilities(lh2, la2)["HomeO2.5"][0])

    lh5, _ = v4.v4_lambdas(data, v6.V5_HOME_CONFIG)
    p5 = float(v2.probabilities(lh5, la2)["HomeO2.5"][0])

    p6f = float(v6.specialist_probability(
        np.asarray([p2]),
        np.asarray([p5]),
        v5_weight=0.75,
        slope=1.05,
        intercept=0.0,
    )[0])

    kind = str(artifact["feature_kind"])
    model = artifact["model"]
    x = v6train.feature_matrix(
        np.asarray([p2]), np.asarray([p5]), kind
    )
    p6t = float(model.predict_proba(x)[0, 1])

    common = {
        "lambda_home_v2": float(lh2[0]),
        "lambda_home_v5": float(lh5[0]),
        "p2": p2,
        "p5": p5,
        "p6_frozen": p6f,
        "trained_feature_kind": kind,
        "trained_C": float(artifact["C"]),
        "trained_at": artifact.get("trained_at"),
    }
    return {
        "V2_FROZEN": {"probability": p2, "meta": common},
        "V5_HOME_FROZEN": {"probability": p5, "meta": common},
        "V6_FROZEN": {"probability": p6f, "meta": common},
        "V6_TRAINED": {"probability": p6t, "meta": common},
    }


def load_odds(connection, fixture_key: str) -> dict:
    result = {
        "market_open_over": None,
        "market_open_under": None,
        "current_over": None,
        "current_under": None,
        "first_observed_at": None,
        "current_observed_at": None,
    }
    if not core.table_exists(connection, ODDS_TABLE):
        return result

    rows = core.fetch_dicts(
        connection,
        f"""
        SELECT selection, open_odd, current_odd,
               first_observed_at, current_observed_at
        FROM {ODDS_TABLE}
        WHERE fixture_key = ?
          AND market = 'team_total_home'
          AND ABS(line - 2.5) < 0.000001
        """,
        (fixture_key,),
    )
    for row in rows:
        selection = str(row.get("selection") or "").lower()
        if selection not in {"over", "under"}:
            continue
        result[f"market_open_{selection}"] = safe_odd(row.get("open_odd"))
        result[f"current_{selection}"] = safe_odd(row.get("current_odd"))
        if result["first_observed_at"] is None:
            result["first_observed_at"] = row.get("first_observed_at")
        current_seen = row.get("current_observed_at")
        if current_seen and (
            result["current_observed_at"] is None
            or str(current_seen) > str(result["current_observed_at"])
        ):
            result["current_observed_at"] = current_seen
    return result


def insert_if_new(
    connection,
    *,
    fixture: dict,
    mode: str,
    model_version: str,
    probability: float,
    metadata: dict,
    odds: dict,
    prediction_timestamp: str,
) -> bool:
    existing = connection.execute(
        f"""
        SELECT id FROM {TABLE}
        WHERE fixture_key = ? AND mode = ? AND model_version = ?
        LIMIT 1
        """,
        (fixture["fixture_key"], mode, model_version),
    ).fetchone()
    if existing is not None:
        return False

    p = float(probability)
    q = 1.0 - p
    current_over = odds.get("current_over") or odds.get("market_open_over")
    current_under = odds.get("current_under") or odds.get("market_open_under")
    market_p = no_vig_over(current_over, current_under)
    ev_over = expected_value(p, current_over)
    ev_under = expected_value(q, current_under)

    candidates = [("over", ev_over), ("under", ev_under)]
    candidates = [(side, ev) for side, ev in candidates if ev is not None]
    best_side = None
    best_ev = None
    if candidates:
        best_side, best_ev = max(candidates, key=lambda x: float(x[1]))

    now = prediction_timestamp
    connection.execute(
        f"""
        INSERT INTO {TABLE} (
            fixture_key, fixture_date, fixture_time, season, league,
            home_team, away_team, mode, model_version, prediction_timestamp,
            probability_over, probability_under, fair_odd_over, fair_odd_under,
            market_open_odd_over, market_open_odd_under,
            odd_over_at_prediction, odd_under_at_prediction,
            market_probability_over_at_prediction,
            ev_over_at_prediction, ev_under_at_prediction,
            best_side_at_prediction, best_ev_at_prediction, best_positive_ev,
            odd_over_last, odd_under_last, market_probability_over_last,
            odds_first_observed_at, odds_last_observed_at,
            status, metadata_json, created_at, updated_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            'OPEN', ?, ?, ?
        )
        """,
        (
            fixture["fixture_key"], fixture["fixture_date"], fixture.get("fixture_time"),
            fixture["season"], fixture["league"], fixture["home_team"], fixture["away_team"],
            mode, model_version, prediction_timestamp,
            p, q, fair_odd(p), fair_odd(q),
            odds.get("market_open_over"), odds.get("market_open_under"),
            current_over, current_under, market_p,
            ev_over, ev_under, best_side, best_ev,
            int(best_ev is not None and float(best_ev) > 0.0),
            current_over, current_under, market_p,
            odds.get("first_observed_at"), odds.get("current_observed_at"),
            json.dumps(metadata, ensure_ascii=False, sort_keys=True), now, now,
        ),
    )
    return True


def refresh_odds(connection) -> int:
    rows = core.fetch_dicts(
        connection,
        f"SELECT id, fixture_key, probability_over, probability_under, odd_over_at_prediction, odd_under_at_prediction FROM {TABLE} WHERE status = 'OPEN'",
    )
    updated = 0
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for row in rows:
        odds = load_odds(connection, str(row["fixture_key"]))
        current_over = odds.get("current_over") or odds.get("market_open_over")
        current_under = odds.get("current_under") or odds.get("market_open_under")
        if current_over is None and current_under is None:
            continue

        # If no price existed when the probability was first frozen, freeze the first
        # subsequently observed price as the actionable prediction-time price.
        over_at_prediction = safe_odd(row.get("odd_over_at_prediction")) or current_over
        under_at_prediction = safe_odd(row.get("odd_under_at_prediction")) or current_under
        p = float(row["probability_over"])
        q = float(row["probability_under"])
        market_p_prediction = no_vig_over(over_at_prediction, under_at_prediction)
        market_p_last = no_vig_over(current_over, current_under)
        ev_over = expected_value(p, over_at_prediction)
        ev_under = expected_value(q, under_at_prediction)
        candidates = [("over", ev_over), ("under", ev_under)]
        candidates = [(side, ev) for side, ev in candidates if ev is not None]
        best_side = None
        best_ev = None
        if candidates:
            best_side, best_ev = max(candidates, key=lambda x: float(x[1]))

        connection.execute(
            f"""
            UPDATE {TABLE}
            SET market_open_odd_over = COALESCE(market_open_odd_over, ?),
                market_open_odd_under = COALESCE(market_open_odd_under, ?),
                odd_over_at_prediction = COALESCE(odd_over_at_prediction, ?),
                odd_under_at_prediction = COALESCE(odd_under_at_prediction, ?),
                market_probability_over_at_prediction = COALESCE(market_probability_over_at_prediction, ?),
                ev_over_at_prediction = COALESCE(ev_over_at_prediction, ?),
                ev_under_at_prediction = COALESCE(ev_under_at_prediction, ?),
                best_side_at_prediction = COALESCE(best_side_at_prediction, ?),
                best_ev_at_prediction = COALESCE(best_ev_at_prediction, ?),
                best_positive_ev = CASE WHEN best_ev_at_prediction IS NULL THEN ? ELSE best_positive_ev END,
                odd_over_last = COALESCE(?, odd_over_last),
                odd_under_last = COALESCE(?, odd_under_last),
                market_probability_over_last = COALESCE(?, market_probability_over_last),
                odds_first_observed_at = COALESCE(odds_first_observed_at, ?),
                odds_last_observed_at = COALESCE(?, odds_last_observed_at),
                updated_at = ?
            WHERE id = ?
            """,
            (
                odds.get("market_open_over"), odds.get("market_open_under"),
                over_at_prediction, under_at_prediction, market_p_prediction,
                ev_over, ev_under, best_side, best_ev,
                int(best_ev is not None and float(best_ev) > 0.0),
                current_over, current_under, market_p_last,
                odds.get("first_observed_at"), odds.get("current_observed_at"),
                now, int(row["id"]),
            ),
        )
        updated += 1
    connection.commit()
    return updated


def settle_rows(connection, matches: pd.DataFrame) -> int:
    rows = core.fetch_dicts(
        connection,
        f"""
        SELECT id, fixture_date, league, home_team, away_team,
               odd_over_at_prediction, odd_under_at_prediction,
               odd_over_last, odd_under_last, best_side_at_prediction
        FROM {TABLE}
        WHERE status = 'OPEN'
        """,
    )
    lookup = {}
    for row in matches.itertuples(index=False):
        key = (
            pd.Timestamp(row.date).strftime("%Y-%m-%d"),
            str(row.league), str(row.home_team), str(row.away_team),
        )
        lookup[key] = (int(row.home_goals), int(row.away_goals))

    settled = 0
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for row in rows:
        key = (
            str(row["fixture_date"]), str(row["league"]),
            str(row["home_team"]), str(row["away_team"]),
        )
        result = lookup.get(key)
        if result is None:
            continue
        hg, ag = result
        over = int(hg > 2)
        oo = safe_odd(row.get("odd_over_at_prediction"))
        ou = safe_odd(row.get("odd_under_at_prediction"))
        last_o = safe_odd(row.get("odd_over_last"))
        last_u = safe_odd(row.get("odd_under_last"))
        profit_over = None if oo is None else (oo - 1.0 if over else -1.0)
        profit_under = None if ou is None else (ou - 1.0 if not over else -1.0)
        side = str(row.get("best_side_at_prediction") or "")
        best_profit = profit_over if side == "over" else (profit_under if side == "under" else None)
        clv_over = None if oo is None or last_o is None else oo / last_o - 1.0
        clv_under = None if ou is None or last_u is None else ou / last_u - 1.0

        connection.execute(
            f"""
            UPDATE {TABLE}
            SET status = 'SETTLED', home_goals = ?, away_goals = ?, result_over = ?,
                profit_over_at_prediction = ?, profit_under_at_prediction = ?,
                best_profit = ?, clv_over = ?, clv_under = ?, settled_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                hg, ag, over, profit_over, profit_under, best_profit,
                clv_over, clv_under, now, now, int(row["id"]),
            ),
        )
        settled += 1
    connection.commit()
    return settled


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo treinado nao encontrado: {MODEL_PATH}. Executa primeiro run_goal_model_v6_train_windows.bat"
        )
    artifact = joblib.load(MODEL_PATH)

    horizon_days = env_int("HOME_O25_SHADOW_HORIZON_DAYS", 7, 1)
    today = datetime.now(core.APP_TIMEZONE).date()
    max_date = today + timedelta(days=horizon_days - 1)
    prediction_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    connection = core.connect_turso()
    try:
        ensure_table(connection)
        matches = core.load_historical_matches(connection)
        history = create_team_history(matches)
        settled = settle_rows(connection, matches)
        refreshed_before = refresh_odds(connection)

        upcoming = core.load_upcoming(connection)
        inserted = 0
        skipped = 0
        skip_examples: list[str] = []

        for row in upcoming:
            try:
                fixture_date = parse_date(row.get("date"))
                if fixture_date is None:
                    raise ValueError("data invalida")
                if fixture_date.date() < today or fixture_date.date() > max_date:
                    continue

                league = normalize_league(row.get("league"))
                if league not in ALLOWED_LEAGUES:
                    continue

                home = normalize_team(league, row.get("home_team"))
                away = normalize_team(league, row.get("away_team"))
                if home is None or away is None:
                    raise ValueError("equipas invalidas")

                home = core.resolve_team_to_history(history, league, home)
                away = core.resolve_team_to_history(history, league, away)
                fixture_key = str(row.get("fixture_key") or "").strip()
                if not fixture_key:
                    raise ValueError("fixture_key em falta")

                fixture = {
                    "fixture_key": fixture_key,
                    "fixture_date": fixture_date.strftime("%Y-%m-%d"),
                    "fixture_time": row.get("time"),
                    "season": calculate_season(fixture_date),
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                }
                odds = load_odds(connection, fixture_key)

                for mode in MODES:
                    try:
                        snapshot = build_snapshot(
                            matches,
                            history,
                            fixture_date=fixture_date,
                            league=league,
                            home_team=home,
                            away_team=away,
                            mode=mode,
                        )
                    except Exception as exc:
                        if len(skip_examples) < 30:
                            skip_examples.append(
                                f"{fixture['fixture_date']} {home}-{away} {mode}: {exc}"
                            )
                        continue

                    predictions = predict_versions(snapshot, artifact)
                    for model_version, payload in predictions.items():
                        meta = {
                            **payload["meta"],
                            "mode": mode,
                            "history_counts": snapshot["history_counts"],
                            "league_home_avg": snapshot["league_home_avg"],
                            "league_away_avg": snapshot["league_away_avg"],
                            "elo_diff": snapshot["elo_diff"],
                            "trained_model_version": artifact.get("model_version"),
                        }
                        inserted += int(insert_if_new(
                            connection,
                            fixture=fixture,
                            mode=mode,
                            model_version=model_version,
                            probability=float(payload["probability"]),
                            metadata=meta,
                            odds=odds,
                            prediction_timestamp=prediction_timestamp,
                        ))

            except Exception as exc:
                skipped += 1
                if len(skip_examples) < 30:
                    skip_examples.append(
                        f"{row.get('date')} {row.get('home_team')}-{row.get('away_team')}: {exc}"
                    )

        connection.commit()
        refreshed_after = refresh_odds(connection)

        counts = core.fetch_dicts(
            connection,
            f"""
            SELECT mode, model_version, status,
                   COUNT(*) AS n,
                   SUM(CASE WHEN best_positive_ev = 1 THEN 1 ELSE 0 END) AS positive_ev
            FROM {TABLE}
            GROUP BY mode, model_version, status
            ORDER BY mode, model_version, status
            """,
        )

        print("\n=== HOME O2.5 CHALLENGER SHADOW ===")
        print(f"Horizon days       : {horizon_days}")
        print(f"New frozen rows    : {inserted}")
        print(f"Odds refreshed     : {refreshed_before + refreshed_after}")
        print(f"Rows settled       : {settled}")
        print(f"Fixtures skipped   : {skipped}")
        for item in counts:
            print(
                f"{item['mode']:6s} | {item['model_version']:16s} | {item['status']:7s} | "
                f"n={item['n']} | +EV={item['positive_ev']}"
            )
        if skip_examples:
            print("\nExamples without prediction:")
            for item in skip_examples:
                print(" -", item)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
