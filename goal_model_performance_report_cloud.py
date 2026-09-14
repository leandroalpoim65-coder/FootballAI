from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable

import paper_trade_cloud as core
from goal_model_value_shadow_cloud import NO_LINE, settle_best


SHADOW_TABLE = "goal_model_shadow"
OUTCOMES_TABLE = "goal_model_market_outcomes"
SUMMARY_TABLE = "goal_model_performance_summary"


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def ensure_tables(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {OUTCOMES_TABLE} (
            shadow_id INTEGER NOT NULL,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            market TEXT NOT NULL,
            market_group TEXT NOT NULL,
            market_label TEXT NOT NULL,
            selection TEXT NOT NULL,
            line REAL NOT NULL DEFAULT -1.0,
            odd REAL NOT NULL,
            fair_odd REAL,
            model_probability REAL,
            implied_probability REAL,
            edge_pp REAL,
            ev REAL NOT NULL,
            source TEXT,
            bookmakers_count INTEGER,
            result TEXT NOT NULL,
            profit REAL NOT NULL,
            is_best_overall INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (shadow_id, market, line)
        )
        """
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{OUTCOMES_TABLE}_date "
        f"ON {OUTCOMES_TABLE}(fixture_date)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{OUTCOMES_TABLE}_market "
        f"ON {OUTCOMES_TABLE}(market_group, market_label)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{OUTCOMES_TABLE}_ev "
        f"ON {OUTCOMES_TABLE}(ev)"
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SUMMARY_TABLE} (
            scope TEXT NOT NULL,
            group_key TEXT NOT NULL,
            group_label TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            bets INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            half_wins INTEGER NOT NULL,
            pushes INTEGER NOT NULL,
            half_losses INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            profit_units REAL NOT NULL,
            total_staked_units REAL NOT NULL,
            roi_pct REAL NOT NULL,
            yield_pct REAL NOT NULL,
            hit_rate_pct REAL,
            avg_odd REAL,
            avg_fair_odd REAL,
            avg_model_probability REAL,
            avg_ev_pct REAL,
            avg_edge_pp REAL,
            max_drawdown_units REAL NOT NULL,
            max_losing_streak INTEGER NOT NULL,
            first_fixture_date TEXT,
            last_fixture_date TEXT,
            sample_band TEXT NOT NULL,
            PRIMARY KEY (scope, group_key)
        )
        """
    )
    connection.commit()


def normalize_line(value: object) -> float:
    if value is None:
        return NO_LINE
    try:
        return float(value)
    except (TypeError, ValueError):
        return NO_LINE


def market_group(market: str) -> str:
    if market == "h2h":
        return "1X2"
    if market == "btts":
        return "BTTS"
    if market == "totals":
        return "Goals Totals"
    if market == "team_total_home":
        return "Home Team Totals"
    if market == "team_total_away":
        return "Away Team Totals"
    return market


def market_label(market: str, line: float) -> str:
    group = market_group(market)
    if line == NO_LINE:
        return group
    if market == "totals":
        if abs(line % 1.0 - 0.5) < 1e-9:
            return f"O/U {line:g}"
        return f"Asian O/U {line:g}"
    if market == "team_total_home":
        return f"Home Team O/U {line:g}"
    if market == "team_total_away":
        return f"Away Team O/U {line:g}"
    return f"{group} {line:g}"


def parse_opportunities(raw: object) -> list[dict]:
    if raw is None:
        return []
    try:
        data = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def best_side_per_market_line(opportunities: list[dict]) -> list[dict]:
    """Escolhe apenas o lado com maior EV em cada mercado/linha."""
    grouped_rows: dict[tuple[str, float], dict] = {}

    for item in opportunities:
        market = str(item.get("market") or "").strip()
        selection = str(item.get("selection") or "").strip()
        if not market or not selection:
            continue

        line = normalize_line(item.get("line"))
        try:
            ev = float(item.get("ev"))
            odd = float(item.get("odd"))
        except (TypeError, ValueError):
            continue

        if odd <= 1.0:
            continue

        key = (market, line)
        current = grouped_rows.get(key)
        if current is None or ev > float(current.get("ev", float("-inf"))):
            candidate = dict(item)
            candidate["_line_norm"] = line
            grouped_rows[key] = candidate

    return list(grouped_rows.values())


def settled_shadow_rows(connection) -> list[dict]:
    return core.fetch_dicts(
        connection,
        f"""
        SELECT
            id,
            fixture_date,
            fixture_time,
            league,
            home_team,
            away_team,
            market_values_json,
            home_goals,
            away_goals
        FROM {SHADOW_TABLE}
        WHERE status = 'SETTLED'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
          AND market_values_json IS NOT NULL
        ORDER BY fixture_date, fixture_time, id
        """,
    )


def materialize_outcomes(connection) -> tuple[int, int]:
    rows = settled_shadow_rows(connection)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    fixtures_with_markets = 0
    outcomes_written = 0

    for row in rows:
        opportunities = parse_opportunities(row.get("market_values_json"))
        candidates = best_side_per_market_line(opportunities)
        if not candidates:
            continue

        settled_candidates: list[dict] = []
        for item in candidates:
            market = str(item.get("market") or "")
            selection = str(item.get("selection") or "")
            line = float(item["_line_norm"])

            result, profit = settle_best(
                market,
                selection,
                None if line == NO_LINE else line,
                item.get("odd"),
                int(row["home_goals"]),
                int(row["away_goals"]),
            )
            if result is None or profit is None:
                continue

            candidate = dict(item)
            candidate["_result"] = result
            candidate["_profit"] = float(profit)
            settled_candidates.append(candidate)

        if not settled_candidates:
            continue

        fixtures_with_markets += 1
        best_identity = max(
            settled_candidates,
            key=lambda item: float(item.get("ev", float("-inf"))),
        )

        connection.execute(
            f"DELETE FROM {OUTCOMES_TABLE} WHERE shadow_id = ?",
            (int(row["id"]),),
        )

        for item in settled_candidates:
            market = str(item.get("market") or "")
            line = float(item["_line_norm"])
            connection.execute(
                f"""
                INSERT INTO {OUTCOMES_TABLE} (
                    shadow_id,
                    fixture_date,
                    fixture_time,
                    league,
                    home_team,
                    away_team,
                    market,
                    market_group,
                    market_label,
                    selection,
                    line,
                    odd,
                    fair_odd,
                    model_probability,
                    implied_probability,
                    edge_pp,
                    ev,
                    source,
                    bookmakers_count,
                    result,
                    profit,
                    is_best_overall,
                    created_at,
                    updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    int(row["id"]),
                    str(row["fixture_date"]),
                    row.get("fixture_time"),
                    str(row["league"]),
                    str(row["home_team"]),
                    str(row["away_team"]),
                    market,
                    market_group(market),
                    market_label(market, line),
                    str(item.get("selection") or ""),
                    line,
                    float(item["odd"]),
                    None if item.get("fair_odd") is None else float(item["fair_odd"]),
                    None if item.get("model_probability") is None else float(item["model_probability"]),
                    None if item.get("implied_probability") is None else float(item["implied_probability"]),
                    None if item.get("edge_pp") is None else float(item["edge_pp"]),
                    float(item["ev"]),
                    item.get("source"),
                    None if item.get("bookmakers_count") is None else int(item["bookmakers_count"]),
                    str(item["_result"]),
                    float(item["_profit"]),
                    int(item is best_identity),
                    now,
                    now,
                ),
            )
            outcomes_written += 1

    connection.commit()
    return fixtures_with_markets, outcomes_written


def load_outcomes(connection) -> list[dict]:
    return core.fetch_dicts(
        connection,
        f"""
        SELECT
            shadow_id,
            fixture_date,
            fixture_time,
            league,
            home_team,
            away_team,
            market,
            market_group,
            market_label,
            selection,
            line,
            odd,
            fair_odd,
            model_probability,
            implied_probability,
            edge_pp,
            ev,
            result,
            profit,
            is_best_overall
        FROM {OUTCOMES_TABLE}
        ORDER BY fixture_date, fixture_time, shadow_id, market, line
        """,
    )


def mean(values: Iterable[float]) -> float | None:
    data = list(values)
    return None if not data else sum(data) / len(data)


def sample_band(bets: int) -> str:
    if bets < 30:
        return "VERY_LOW_<30"
    if bets < 100:
        return "EARLY_30_99"
    if bets < 300:
        return "USEFUL_100_299"
    return "STRONGER_300_PLUS"


def max_drawdown(rows: list[dict]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for row in rows:
        cumulative += float(row["profit"])
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def max_losing_streak(rows: list[dict]) -> int:
    current = 0
    maximum = 0
    for row in rows:
        profit = float(row["profit"])
        if profit < 0:
            current += 1
            maximum = max(maximum, current)
        elif profit > 0:
            current = 0
    return maximum


def metrics(rows: list[dict]) -> dict:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("fixture_date") or ""),
            str(row.get("fixture_time") or ""),
            int(row.get("shadow_id") or 0),
        ),
    )

    bets = len(ordered)
    wins = sum(str(row.get("result")) == "WIN" for row in ordered)
    half_wins = sum(str(row.get("result")) == "HALF_WIN" for row in ordered)
    pushes = sum(str(row.get("result")) == "PUSH" for row in ordered)
    half_losses = sum(str(row.get("result")) == "HALF_LOSS" for row in ordered)
    losses = sum(str(row.get("result")) == "LOSS" for row in ordered)

    profit = sum(float(row["profit"]) for row in ordered)
    staked = float(bets)
    roi = (profit / staked * 100.0) if staked else 0.0

    positive = sum(float(row["profit"]) > 0 for row in ordered)
    negative = sum(float(row["profit"]) < 0 for row in ordered)
    resolved_non_push = positive + negative
    hit_rate = positive / resolved_non_push * 100.0 if resolved_non_push else None

    return {
        "bets": bets,
        "wins": wins,
        "half_wins": half_wins,
        "pushes": pushes,
        "half_losses": half_losses,
        "losses": losses,
        "profit_units": profit,
        "total_staked_units": staked,
        "roi_pct": roi,
        "yield_pct": roi,
        "hit_rate_pct": hit_rate,
        "avg_odd": mean(float(row["odd"]) for row in ordered),
        "avg_fair_odd": mean(
            float(row["fair_odd"])
            for row in ordered
            if row.get("fair_odd") is not None
        ),
        "avg_model_probability": mean(
            float(row["model_probability"])
            for row in ordered
            if row.get("model_probability") is not None
        ),
        "avg_ev_pct": mean(float(row["ev"]) * 100.0 for row in ordered),
        "avg_edge_pp": mean(
            float(row["edge_pp"])
            for row in ordered
            if row.get("edge_pp") is not None
        ),
        "max_drawdown_units": max_drawdown(ordered),
        "max_losing_streak": max_losing_streak(ordered),
        "first_fixture_date": str(ordered[0]["fixture_date"]) if ordered else None,
        "last_fixture_date": str(ordered[-1]["fixture_date"]) if ordered else None,
        "sample_band": sample_band(bets),
    }


def odd_bucket(odd: float) -> str:
    if odd < 1.50:
        return "<1.50"
    if odd < 1.80:
        return "1.50-1.79"
    if odd < 2.00:
        return "1.80-1.99"
    if odd < 2.50:
        return "2.00-2.49"
    return "2.50+"


def ev_bucket(ev: float) -> str:
    pct = ev * 100.0
    if pct < 0.0:
        return "<0%"
    if pct < 2.0:
        return "0-2%"
    if pct < 5.0:
        return "2-5%"
    if pct < 8.0:
        return "5-8%"
    if pct < 12.0:
        return "8-12%"
    if pct < 20.0:
        return "12-20%"
    return "20%+"


def grouped(rows: list[dict], key_func) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(key_func(row))].append(row)
    return dict(groups)


def add_summary(
    summaries: list[dict],
    scope: str,
    group_key: str,
    group_label: str,
    rows: list[dict],
) -> None:
    if not rows:
        return
    summaries.append(
        {
            "scope": scope,
            "group_key": group_key,
            "group_label": group_label,
            **metrics(rows),
        }
    )


def build_summaries(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    best_rows = [row for row in rows if int(row.get("is_best_overall") or 0) == 1]
    best_positive = [row for row in best_rows if float(row["ev"]) > 0.0]
    positive_rows = [row for row in rows if float(row["ev"]) > 0.0]

    add_summary(summaries, "OVERALL", "best_all", "Best market per fixture — all EV", best_rows)
    add_summary(summaries, "OVERALL", "best_positive", "Best market per fixture — EV > 0%", best_positive)
    add_summary(
        summaries,
        "OVERALL",
        "all_market_lines_positive",
        "Every market/line candidate — EV > 0%",
        positive_rows,
    )

    for label, group_rows in grouped(positive_rows, lambda row: row["market_group"]).items():
        add_summary(summaries, "MARKET", label, label, group_rows)

    for label, group_rows in grouped(positive_rows, lambda row: row["market_label"]).items():
        add_summary(summaries, "MARKET_LINE", label, label, group_rows)

    for label, group_rows in grouped(best_positive, lambda row: row["league"]).items():
        add_summary(summaries, "LEAGUE_BEST", label, label, group_rows)

    for label, group_rows in grouped(best_positive, lambda row: odd_bucket(float(row["odd"]))).items():
        add_summary(summaries, "ODD_BUCKET_BEST", label, label, group_rows)

    for label, group_rows in grouped(best_rows, lambda row: ev_bucket(float(row["ev"]))).items():
        add_summary(summaries, "EV_BUCKET_BEST", label, label, group_rows)

    for threshold in (0.00, 0.02, 0.05, 0.08, 0.10, 0.15):
        threshold_rows = [row for row in best_rows if float(row["ev"]) >= threshold]
        add_summary(
            summaries,
            "EV_THRESHOLD_BEST",
            f"{threshold:.4f}",
            f"EV >= {threshold:.0%}",
            threshold_rows,
        )

    for market_name, group_rows in grouped(rows, lambda row: row["market_group"]).items():
        for threshold in (0.00, 0.05, 0.08, 0.10, 0.15):
            threshold_rows = [row for row in group_rows if float(row["ev"]) >= threshold]
            add_summary(
                summaries,
                "MARKET_EV_THRESHOLD",
                f"{market_name}|{threshold:.4f}",
                f"{market_name} | EV >= {threshold:.0%}",
                threshold_rows,
            )

    return summaries


def persist_summaries(connection, summaries: list[dict]) -> str:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    connection.execute(f"DELETE FROM {SUMMARY_TABLE}")

    columns = [
        "scope",
        "group_key",
        "group_label",
        "generated_at",
        "bets",
        "wins",
        "half_wins",
        "pushes",
        "half_losses",
        "losses",
        "profit_units",
        "total_staked_units",
        "roi_pct",
        "yield_pct",
        "hit_rate_pct",
        "avg_odd",
        "avg_fair_odd",
        "avg_model_probability",
        "avg_ev_pct",
        "avg_edge_pp",
        "max_drawdown_units",
        "max_losing_streak",
        "first_fixture_date",
        "last_fixture_date",
        "sample_band",
    ]
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)

    for summary in summaries:
        record = {**summary, "generated_at": generated_at}
        connection.execute(
            f"INSERT INTO {SUMMARY_TABLE} ({column_sql}) VALUES ({placeholders})",
            tuple(record.get(column) for column in columns),
        )

    connection.commit()
    return generated_at


def print_summary_table(
    title: str,
    summaries: list[dict],
    scope: str,
    min_sample: int,
    limit: int = 15,
) -> None:
    selected = [
        row
        for row in summaries
        if row["scope"] == scope and int(row["bets"]) >= min_sample
    ]
    selected.sort(
        key=lambda row: (float(row["roi_pct"]), int(row["bets"])),
        reverse=True,
    )

    print()
    print(title)
    print("-" * len(title))

    if not selected:
        print(f"Sem grupos com pelo menos {min_sample} apostas liquidadas.")
        return

    for row in selected[:limit]:
        print(
            f"{row['group_label']:<34} | "
            f"N={int(row['bets']):>4} | "
            f"ROI {float(row['roi_pct']):+7.2f}% | "
            f"Lucro {float(row['profit_units']):+7.2f}u | "
            f"Hit {float(row['hit_rate_pct'] or 0.0):5.1f}% | "
            f"DD {float(row['max_drawdown_units']):6.2f}u"
        )


def main() -> None:
    print()
    print("================================================")
    print(" FOOTBALLAI — MULTI-MARKET PERFORMANCE REPORT")
    print("================================================")
    print()

    min_print_sample = env_int("GOAL_REPORT_MIN_SAMPLE", 10, 1)
    connection = core.connect_turso()

    try:
        if not core.table_exists(connection, SHADOW_TABLE):
            print("A tabela goal_model_shadow ainda não existe. Sem dados para relatório.")
            return

        ensure_tables(connection)
        fixtures, outcomes_written = materialize_outcomes(connection)
        outcomes = load_outcomes(connection)
        summaries = build_summaries(outcomes)
        generated_at = persist_summaries(connection, summaries)

        settled_fixtures = len({int(row["shadow_id"]) for row in outcomes})
        positive_outcomes = sum(float(row["ev"]) > 0.0 for row in outcomes)
        best_positive = sum(
            int(row.get("is_best_overall") or 0) == 1 and float(row["ev"]) > 0.0
            for row in outcomes
        )

        print(f"Gerado em UTC                 : {generated_at}")
        print(f"Fixtures SETTLED com mercados : {settled_fixtures}")
        print(f"Fixtures materializados agora : {fixtures}")
        print(f"Market-lines materializados   : {outcomes_written}")
        print(f"Market-lines EV > 0%          : {positive_outcomes}")
        print(f"Best-per-game EV > 0%         : {best_positive}")
        print(f"Linhas resumo gravadas        : {len(summaries)}")

        overall = {
            row["group_key"]: row
            for row in summaries
            if row["scope"] == "OVERALL"
        }
        best = overall.get("best_positive")
        if best:
            print()
            print("BEST MARKET / FIXTURE — EV > 0%")
            print("--------------------------------")
            print(
                f"Apostas  : {int(best['bets'])}\n"
                f"Lucro    : {float(best['profit_units']):+.2f}u\n"
                f"ROI/Yield: {float(best['roi_pct']):+.2f}%\n"
                f"Hit rate : {float(best['hit_rate_pct'] or 0.0):.1f}%\n"
                f"Drawdown : {float(best['max_drawdown_units']):.2f}u\n"
                f"Amostra  : {best['sample_band']}"
            )

        print_summary_table("ROI por mercado — EV > 0%", summaries, "MARKET", min_print_sample)
        print_summary_table("ROI por linha — EV > 0%", summaries, "MARKET_LINE", min_print_sample)
        print_summary_table(
            "Thresholds EV — melhor mercado por jogo",
            summaries,
            "EV_THRESHOLD_BEST",
            min_print_sample,
        )
        print_summary_table(
            "ROI por liga — melhor mercado EV > 0%",
            summaries,
            "LEAGUE_BEST",
            min_print_sample,
        )
        print_summary_table(
            "ROI por faixa de odd — melhor mercado EV > 0%",
            summaries,
            "ODD_BUCKET_BEST",
            min_print_sample,
        )

        print()
        print(
            "Nota: ROI/Yield = lucro / stake total com stake fixa de 1u. "
            "Pushes não contam no denominador do hit rate."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
