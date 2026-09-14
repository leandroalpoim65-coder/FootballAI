from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

import paper_trade_uefa_cloud as uefa_cloud
from ml.goal_model import expected_value, project_match
from ml.paper_trade import (
    build_sports_features,
    calculate_current_elo,
    create_team_history,
    normalize_league,
    normalize_team,
    parse_date,
)


core = uefa_cloud.core

RUN_TABLE = "goal_model_controlled_runs"
METRIC_TABLE = "goal_model_controlled_metrics"
ROI_TABLE = "goal_model_controlled_roi"
BACKTEST_VERSION = "goal-model-walk-forward-v1"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def ensure_tables(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            backtest_version TEXT NOT NULL,
            requested_per_league INTEGER NOT NULL,
            requested_paper_limit INTEGER NOT NULL,
            historical_predictions INTEGER NOT NULL DEFAULT 0,
            historical_skipped INTEGER NOT NULL DEFAULT 0,
            paper_predictions INTEGER NOT NULL DEFAULT 0,
            paper_skipped INTEGER NOT NULL DEFAULT 0,
            first_historical_date TEXT,
            last_historical_date TEXT,
            first_paper_date TEXT,
            last_paper_date TEXT,
            notes TEXT
        )
        """
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {METRIC_TABLE} (
            run_id INTEGER NOT NULL,
            market TEXT NOT NULL,
            observations INTEGER NOT NULL,
            accuracy_pct REAL,
            brier_score REAL,
            log_loss REAL,
            avg_predicted_pct REAL,
            actual_rate_pct REAL,
            calibration_gap_pp REAL,
            PRIMARY KEY (run_id, market)
        )
        """
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROI_TABLE} (
            run_id INTEGER NOT NULL,
            scope TEXT NOT NULL,
            group_key TEXT NOT NULL,
            strategy TEXT NOT NULL,
            threshold REAL NOT NULL,
            bets INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            profit_units REAL NOT NULL,
            roi_pct REAL NOT NULL,
            avg_odd REAL,
            avg_ev_pct REAL,
            max_drawdown_units REAL NOT NULL,
            over_bets INTEGER NOT NULL,
            under_bets INTEGER NOT NULL,
            PRIMARY KEY (run_id, scope, group_key, strategy, threshold)
        )
        """
    )

    connection.commit()


def league_goal_averages(history: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if history.empty:
        return {}

    grouped = history.groupby("league", dropna=True)[["home_goals", "away_goals"]].mean()
    return {
        str(league): (float(row["home_goals"]), float(row["away_goals"]))
        for league, row in grouped.iterrows()
    }


def choose_historical_targets(matches: pd.DataFrame, per_league: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for _, group in matches.groupby("league", dropna=True):
        ordered = group.sort_values(["date", "id"]).copy()
        # Preserve a meaningful warm-up before the evaluation window.
        if len(ordered) < max(30, per_league + 10):
            continue
        parts.append(ordered.tail(per_league))

    if not parts:
        return matches.iloc[0:0].copy()

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "league", "id"])
        .reset_index(drop=True)
    )


def binary_metric_row(rows: list[tuple[float, int]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "observations": 0,
            "accuracy_pct": None,
            "brier_score": None,
            "log_loss": None,
            "avg_predicted_pct": None,
            "actual_rate_pct": None,
            "calibration_gap_pp": None,
        }

    eps = 1e-12
    n = len(rows)
    accuracy = sum((probability >= 0.5) == bool(actual) for probability, actual in rows) / n
    brier = sum((probability - actual) ** 2 for probability, actual in rows) / n
    log_loss = -sum(
        actual * math.log(max(eps, min(1.0 - eps, probability)))
        + (1 - actual) * math.log(max(eps, min(1.0 - eps, 1.0 - probability)))
        for probability, actual in rows
    ) / n
    avg_predicted = sum(probability for probability, _ in rows) / n
    actual_rate = sum(actual for _, actual in rows) / n

    return {
        "observations": n,
        "accuracy_pct": accuracy * 100.0,
        "brier_score": brier,
        "log_loss": log_loss,
        "avg_predicted_pct": avg_predicted * 100.0,
        "actual_rate_pct": actual_rate * 100.0,
        "calibration_gap_pp": (avg_predicted - actual_rate) * 100.0,
    }


def multiclass_1x2_metric(rows: list[tuple[dict[str, float], str]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "observations": 0,
            "accuracy_pct": None,
            "brier_score": None,
            "log_loss": None,
            "avg_predicted_pct": None,
            "actual_rate_pct": None,
            "calibration_gap_pp": None,
        }

    eps = 1e-12
    correct = 0
    brier_total = 0.0
    log_total = 0.0
    confidence_total = 0.0

    for probabilities, actual in rows:
        predicted = max(probabilities, key=probabilities.get)
        correct += int(predicted == actual)
        confidence_total += max(probabilities.values())
        for label in ("home", "draw", "away"):
            target = 1.0 if label == actual else 0.0
            brier_total += (float(probabilities[label]) - target) ** 2
        log_total += -math.log(max(eps, min(1.0, float(probabilities[actual]))))

    n = len(rows)
    return {
        "observations": n,
        "accuracy_pct": correct / n * 100.0,
        # Multiclass Brier is the sum over the three mutually-exclusive outcomes.
        "brier_score": brier_total / n,
        "log_loss": log_total / n,
        "avg_predicted_pct": confidence_total / n * 100.0,
        "actual_rate_pct": None,
        "calibration_gap_pp": None,
    }


def prediction_for_match(
    history: pd.DataFrame,
    team_history: pd.DataFrame,
    elo_ratings: dict,
    league_averages: dict[str, tuple[float, float]],
    *,
    fixture_date: pd.Timestamp,
    league: str,
    home_team: str,
    away_team: str,
) -> dict:
    sports_features = build_sports_features(
        history=team_history,
        elo_ratings=elo_ratings,
        league=league,
        home_team=home_team,
        away_team=away_team,
        fixture_date=fixture_date,
    )

    league_home_avg = None
    league_away_avg = None
    if league in league_averages:
        league_home_avg, league_away_avg = league_averages[league]

    return project_match(
        sports_features,
        league_home_avg=league_home_avg,
        league_away_avg=league_away_avg,
    )


def evaluate_historical_predictions(
    matches: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[dict[str, dict], int, list[str]]:
    binary_rows: dict[str, list[tuple[float, int]]] = defaultdict(list)
    one_x_two_rows: list[tuple[dict[str, float], str]] = []
    skipped = 0
    skipped_examples: list[str] = []

    for fixture_date, day_targets in targets.groupby("date", sort=True):
        fixture_date = pd.Timestamp(fixture_date)
        # Strictly prior dates only. No same-day result can leak into a prediction.
        history = matches[matches["date"] < fixture_date].copy()
        if history.empty:
            skipped += len(day_targets)
            continue

        team_history = create_team_history(history)
        elo_ratings = calculate_current_elo(history)
        averages = league_goal_averages(history)

        for row in day_targets.itertuples(index=False):
            try:
                projection = prediction_for_match(
                    history,
                    team_history,
                    elo_ratings,
                    averages,
                    fixture_date=fixture_date,
                    league=str(row.league),
                    home_team=str(row.home_team),
                    away_team=str(row.away_team),
                )
            except Exception as error:
                skipped += 1
                if len(skipped_examples) < 12:
                    skipped_examples.append(
                        f"{fixture_date.date()} | {row.home_team} vs {row.away_team} | {error}"
                    )
                continue

            home_goals = int(row.home_goals)
            away_goals = int(row.away_goals)
            total_goals = home_goals + away_goals

            if home_goals > away_goals:
                actual_1x2 = "home"
            elif home_goals < away_goals:
                actual_1x2 = "away"
            else:
                actual_1x2 = "draw"

            one_x_two_rows.append(
                (
                    {
                        "home": float(projection["match_result"]["home"]),
                        "draw": float(projection["match_result"]["draw"]),
                        "away": float(projection["match_result"]["away"]),
                    },
                    actual_1x2,
                )
            )

            definitions = {
                "BTTS_YES": (
                    float(projection["btts"]["yes"]),
                    int(home_goals > 0 and away_goals > 0),
                ),
                "OVER_1.5": (
                    float(projection["totals"]["1.5"]["over"]),
                    int(total_goals > 1.5),
                ),
                "OVER_2.5": (
                    float(projection["totals"]["2.5"]["over"]),
                    int(total_goals > 2.5),
                ),
                "OVER_3.5": (
                    float(projection["totals"]["3.5"]["over"]),
                    int(total_goals > 3.5),
                ),
                "HOME_OVER_0.5": (
                    float(projection["team_totals"]["home"]["0.5"]["over"]),
                    int(home_goals > 0.5),
                ),
                "HOME_OVER_1.5": (
                    float(projection["team_totals"]["home"]["1.5"]["over"]),
                    int(home_goals > 1.5),
                ),
                "HOME_OVER_2.5": (
                    float(projection["team_totals"]["home"]["2.5"]["over"]),
                    int(home_goals > 2.5),
                ),
                "AWAY_OVER_0.5": (
                    float(projection["team_totals"]["away"]["0.5"]["over"]),
                    int(away_goals > 0.5),
                ),
                "AWAY_OVER_1.5": (
                    float(projection["team_totals"]["away"]["1.5"]["over"]),
                    int(away_goals > 1.5),
                ),
                "AWAY_OVER_2.5": (
                    float(projection["team_totals"]["away"]["2.5"]["over"]),
                    int(away_goals > 2.5),
                ),
            }

            for market, pair in definitions.items():
                binary_rows[market].append(pair)

    metrics: dict[str, dict] = {
        "1X2": multiclass_1x2_metric(one_x_two_rows),
    }
    for market, rows in sorted(binary_rows.items()):
        metrics[market] = binary_metric_row(rows)

    return metrics, skipped, skipped_examples


def load_resolved_paper_trades(connection, limit: int) -> list[dict]:
    if not core.table_exists(connection, "paper_trades"):
        return []

    rows = core.fetch_dicts(
        connection,
        """
        SELECT
            id,
            fixture_date,
            fixture_time,
            league,
            home_team,
            away_team,
            odd_over25_open,
            odd_under25_open,
            result_over25
        FROM paper_trades
        WHERE UPPER(COALESCE(status, 'OPEN')) = 'RESOLVED'
          AND result_over25 IN (0, 1)
          AND odd_over25_open IS NOT NULL
          AND odd_under25_open IS NOT NULL
        ORDER BY fixture_date DESC, fixture_time DESC, id DESC
        LIMIT ?
        """,
        (int(limit),),
    )

    rows.reverse()
    return rows


def resolve_team(team_history: pd.DataFrame, league: str, team: object) -> str:
    normalized = normalize_team(league, team)
    if normalized is None:
        raise ValueError("equipa inválida")
    return core.resolve_team_to_history(team_history, league, str(normalized))


def evaluate_paper_odds(
    matches: pd.DataFrame,
    paper_rows: list[dict],
) -> tuple[list[dict], int, list[str]]:
    replay: list[dict] = []
    skipped = 0
    skipped_examples: list[str] = []

    rows_by_date: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for row in paper_rows:
        fixture_date = parse_date(row.get("fixture_date"))
        if fixture_date is None:
            skipped += 1
            continue
        rows_by_date[pd.Timestamp(fixture_date).normalize()].append(row)

    for fixture_date in sorted(rows_by_date):
        history = matches[matches["date"] < fixture_date].copy()
        if history.empty:
            skipped += len(rows_by_date[fixture_date])
            continue

        team_history = create_team_history(history)
        elo_ratings = calculate_current_elo(history)
        averages = league_goal_averages(history)

        for row in rows_by_date[fixture_date]:
            try:
                league = normalize_league(row.get("league"))
                if league is None:
                    raise ValueError("liga inválida")
                league = str(league)

                home_team = resolve_team(team_history, league, row.get("home_team"))
                away_team = resolve_team(team_history, league, row.get("away_team"))

                projection = prediction_for_match(
                    history,
                    team_history,
                    elo_ratings,
                    averages,
                    fixture_date=fixture_date,
                    league=league,
                    home_team=home_team,
                    away_team=away_team,
                )

                odd_over = core.valid_odd(row.get("odd_over25_open"))
                odd_under = core.valid_odd(row.get("odd_under25_open"))
                if odd_over is None or odd_under is None:
                    raise ValueError("odds OPEN inválidas")

                probability_over = float(projection["totals"]["2.5"]["over"])
                probability_under = float(projection["totals"]["2.5"]["under"])

                ev_over = expected_value(probability_over, odd_over)
                ev_under = expected_value(probability_under, odd_under)

                if ev_over >= ev_under:
                    side = "OVER"
                    odd = float(odd_over)
                    ev = float(ev_over)
                    won = int(row["result_over25"]) == 1
                else:
                    side = "UNDER"
                    odd = float(odd_under)
                    ev = float(ev_under)
                    won = int(row["result_over25"]) == 0

                replay.append(
                    {
                        "paper_trade_id": int(row["id"]),
                        "fixture_date": fixture_date.strftime("%Y-%m-%d"),
                        "fixture_time": str(row.get("fixture_time") or ""),
                        "league": league,
                        "side": side,
                        "odd": odd,
                        "ev": ev,
                        "won": bool(won),
                        "profit": odd - 1.0 if won else -1.0,
                        "odd_over": float(odd_over),
                        "odd_under": float(odd_under),
                        "result_over25": int(row["result_over25"]),
                    }
                )

            except Exception as error:
                skipped += 1
                if len(skipped_examples) < 12:
                    skipped_examples.append(
                        f"{fixture_date.date()} | {row.get('home_team')} vs {row.get('away_team')} | {error}"
                    )

    return replay, skipped, skipped_examples


def max_drawdown(rows: list[dict]) -> float:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0

    for row in rows:
        cumulative += float(row["profit"])
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)

    return maximum


def roi_metrics(rows: list[dict]) -> dict[str, float | int | None]:
    bets = len(rows)
    if not rows:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "profit_units": 0.0,
            "roi_pct": 0.0,
            "avg_odd": None,
            "avg_ev_pct": None,
            "max_drawdown_units": 0.0,
            "over_bets": 0,
            "under_bets": 0,
        }

    wins = sum(bool(row["won"]) for row in rows)
    profit = sum(float(row["profit"]) for row in rows)

    return {
        "bets": bets,
        "wins": wins,
        "losses": bets - wins,
        "profit_units": profit,
        "roi_pct": profit / bets * 100.0,
        "avg_odd": sum(float(row["odd"]) for row in rows) / bets,
        "avg_ev_pct": sum(float(row["ev"]) for row in rows) / bets * 100.0,
        "max_drawdown_units": max_drawdown(rows),
        "over_bets": sum(str(row["side"]) == "OVER" for row in rows),
        "under_bets": sum(str(row["side"]) == "UNDER" for row in rows),
    }


def baseline_rows(replay: list[dict], side: str) -> list[dict]:
    rows: list[dict] = []

    for row in replay:
        result_over25 = int(row["result_over25"])
        if side == "OVER":
            odd = float(row["odd_over"])
            won = result_over25 == 1
        else:
            odd = float(row["odd_under"])
            won = result_over25 == 0

        copy = dict(row)
        copy["side"] = side
        copy["odd"] = odd
        copy["ev"] = 0.0
        copy["won"] = won
        copy["profit"] = odd - 1.0 if won else -1.0
        rows.append(copy)

    return rows


def write_metric_rows(connection, run_id: int, metrics: dict[str, dict]) -> None:
    for market, values in metrics.items():
        connection.execute(
            f"""
            INSERT OR REPLACE INTO {METRIC_TABLE} (
                run_id,
                market,
                observations,
                accuracy_pct,
                brier_score,
                log_loss,
                avg_predicted_pct,
                actual_rate_pct,
                calibration_gap_pp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                market,
                int(values["observations"]),
                values["accuracy_pct"],
                values["brier_score"],
                values["log_loss"],
                values["avg_predicted_pct"],
                values["actual_rate_pct"],
                values["calibration_gap_pp"],
            ),
        )


def write_roi_row(
    connection,
    run_id: int,
    scope: str,
    group_key: str,
    strategy: str,
    threshold: float,
    rows: list[dict],
) -> None:
    values = roi_metrics(rows)
    connection.execute(
        f"""
        INSERT OR REPLACE INTO {ROI_TABLE} (
            run_id,
            scope,
            group_key,
            strategy,
            threshold,
            bets,
            wins,
            losses,
            profit_units,
            roi_pct,
            avg_odd,
            avg_ev_pct,
            max_drawdown_units,
            over_bets,
            under_bets
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            scope,
            group_key,
            strategy,
            threshold,
            int(values["bets"]),
            int(values["wins"]),
            int(values["losses"]),
            float(values["profit_units"]),
            float(values["roi_pct"]),
            values["avg_odd"],
            values["avg_ev_pct"],
            float(values["max_drawdown_units"]),
            int(values["over_bets"]),
            int(values["under_bets"]),
        ),
    )


def main() -> None:
    print()
    print("================================================")
    print(" FOOTBALLAI — GOAL MODEL CONTROLLED BACKTEST")
    print("================================================")
    print()

    per_league = env_int("GOAL_CONTROL_PER_LEAGUE", 40, 10)
    paper_limit = env_int("GOAL_CONTROL_PAPER_LIMIT", 500, 20)
    thresholds = [-1.0, 0.00, 0.02, 0.05, 0.08, 0.10, 0.15]

    connection = core.connect_turso()

    try:
        ensure_tables(connection)

        if not core.table_exists(connection, "matches"):
            raise RuntimeError("A tabela matches não existe na Turso.")

        # paper_trade_uefa_cloud patches this loader to include UEFA history too.
        matches = core.load_historical_matches(connection)
        matches = matches.sort_values(["date", "id"]).reset_index(drop=True)

        targets = choose_historical_targets(matches, per_league)

        run_at = now_utc()
        connection.execute(
            f"""
            INSERT INTO {RUN_TABLE} (
                run_at,
                backtest_version,
                requested_per_league,
                requested_paper_limit,
                notes
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_at,
                BACKTEST_VERSION,
                per_league,
                paper_limit,
                "Strict walk-forward: each fixture sees only matches from earlier calendar dates.",
            ),
        )
        run_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
        connection.commit()

        metrics, historical_skipped, historical_examples = evaluate_historical_predictions(
            matches,
            targets,
        )
        write_metric_rows(connection, run_id, metrics)

        paper_rows = load_resolved_paper_trades(connection, paper_limit)
        replay, paper_skipped, paper_examples = evaluate_paper_odds(matches, paper_rows)

        # Baselines: what the same resolved sample did if blindly betting one side.
        write_roi_row(
            connection,
            run_id,
            "GLOBAL",
            "ALL",
            "ALWAYS_OVER25",
            -1.0,
            baseline_rows(replay, "OVER"),
        )
        write_roi_row(
            connection,
            run_id,
            "GLOBAL",
            "ALL",
            "ALWAYS_UNDER25",
            -1.0,
            baseline_rows(replay, "UNDER"),
        )

        for threshold in thresholds:
            selected = [
                row
                for row in replay
                if threshold < 0.0 or float(row["ev"]) >= threshold
            ]
            write_roi_row(
                connection,
                run_id,
                "GLOBAL",
                "ALL",
                "MODEL_BEST_SIDE",
                threshold,
                selected,
            )

            if threshold in {0.00, 0.05, 0.08, 0.10}:
                leagues = sorted({str(row["league"]) for row in selected})
                for league in leagues:
                    league_rows = [row for row in selected if str(row["league"]) == league]
                    write_roi_row(
                        connection,
                        run_id,
                        "LEAGUE",
                        league,
                        "MODEL_BEST_SIDE",
                        threshold,
                        league_rows,
                    )

        historical_dates = [pd.Timestamp(value) for value in targets["date"].tolist()] if not targets.empty else []
        paper_dates = [parse_date(row.get("fixture_date")) for row in paper_rows]
        paper_dates = [pd.Timestamp(value) for value in paper_dates if value is not None]

        historical_predictions = int(metrics.get("OVER_2.5", {}).get("observations", 0))

        connection.execute(
            f"""
            UPDATE {RUN_TABLE}
            SET
                historical_predictions = ?,
                historical_skipped = ?,
                paper_predictions = ?,
                paper_skipped = ?,
                first_historical_date = ?,
                last_historical_date = ?,
                first_paper_date = ?,
                last_paper_date = ?
            WHERE id = ?
            """,
            (
                historical_predictions,
                historical_skipped,
                len(replay),
                paper_skipped,
                min(historical_dates).strftime("%Y-%m-%d") if historical_dates else None,
                max(historical_dates).strftime("%Y-%m-%d") if historical_dates else None,
                min(paper_dates).strftime("%Y-%m-%d") if paper_dates else None,
                max(paper_dates).strftime("%Y-%m-%d") if paper_dates else None,
                run_id,
            ),
        )
        connection.commit()

        print(f"Run ID                       : {run_id}")
        print(f"Histórico alvo               : {len(targets)}")
        print(f"Histórico previsto           : {historical_predictions}")
        print(f"Histórico ignorado           : {historical_skipped}")
        print(f"Paper Trades resolvidos lidos: {len(paper_rows)}")
        print(f"Paper Trades reconstruídos   : {len(replay)}")
        print(f"Paper Trades ignorados       : {paper_skipped}")

        print()
        print("QUALIDADE PREDITIVA — WALK FORWARD")
        print("----------------------------------")
        for market in ("1X2", "BTTS_YES", "OVER_1.5", "OVER_2.5", "OVER_3.5", "HOME_OVER_1.5", "AWAY_OVER_1.5"):
            values = metrics.get(market)
            if not values or not values.get("observations"):
                continue
            print(
                f"{market:16s} | N={int(values['observations']):4d} | "
                f"acc={float(values['accuracy_pct']):6.2f}% | "
                f"brier={float(values['brier_score']):.4f} | "
                f"logloss={float(values['log_loss']):.4f}"
            )

        print()
        print("ROI REAL — O/U 2.5 COM ODDS OPEN CONGELADAS")
        print("-------------------------------------------")
        for threshold in thresholds:
            selected = [
                row
                for row in replay
                if threshold < 0.0 or float(row["ev"]) >= threshold
            ]
            values = roi_metrics(selected)
            label = "BEST SIDE / todos" if threshold < 0 else f"EV >= {threshold:.0%}"
            print(
                f"{label:18s} | N={int(values['bets']):4d} | "
                f"ROI={float(values['roi_pct']):+7.2f}% | "
                f"P/L={float(values['profit_units']):+7.2f}u | "
                f"DD={float(values['max_drawdown_units']):6.2f}u"
            )

        if historical_examples or paper_examples:
            print()
            print("Alguns jogos ignorados (esperado quando falta histórico suficiente):")
            for example in (historical_examples + paper_examples)[:12]:
                print(f" - {example}")

        print()
        print("Nota: BTTS/Team Totals sem odds históricas reais são avaliados por calibração/acerto, não por ROI.")
        print("O ROI usa apenas odds OPEN congeladas de paper_trades para O/U 2.5.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
