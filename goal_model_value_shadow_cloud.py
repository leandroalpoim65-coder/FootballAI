from __future__ import annotations

import json
from datetime import datetime, timezone

import paper_trade_cloud as core
from ml.goal_model import asian_expected_value, expected_value


SHADOW_TABLE = "goal_model_shadow"
ODDS_TABLE = "multi_market_odds"
NO_LINE = -1.0


def table_columns(connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def add_column(connection, table: str, columns: set[str], name: str, sql_type: str) -> None:
    if name in columns:
        return
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    columns.add(name)


def ensure_value_columns(connection) -> None:
    columns = table_columns(connection, SHADOW_TABLE)
    additions = {
        "market_odds_json": "TEXT",
        "market_values_json": "TEXT",
        "market_odds_count": "INTEGER",
        "best_market": "TEXT",
        "best_selection": "TEXT",
        "best_line": "REAL",
        "best_odd": "REAL",
        "best_fair_odd": "REAL",
        "best_probability": "REAL",
        "best_implied_probability": "REAL",
        "best_edge_pp": "REAL",
        "best_ev": "REAL",
        "best_positive_ev": "INTEGER",
        "best_result": "TEXT",
        "best_profit": "REAL",
    }
    for name, sql_type in additions.items():
        add_column(connection, SHADOW_TABLE, columns, name, sql_type)

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS ix_goal_model_shadow_best_ev
        ON {SHADOW_TABLE}(best_ev)
        """
    )
    connection.commit()


def load_market_odds(connection, fixture_key: object) -> list[dict]:
    if not fixture_key or not core.table_exists(connection, ODDS_TABLE):
        return []
    return core.fetch_dicts(
        connection,
        f"""
        SELECT
            market,
            selection,
            line,
            open_odd,
            first_observed_at,
            open_source,
            open_bookmakers_count
        FROM {ODDS_TABLE}
        WHERE fixture_key = ?
        ORDER BY market, line, selection
        """,
        (str(fixture_key),),
    )


def append_legacy_over25(odds_rows: list[dict], shadow_row: dict) -> None:
    existing = {
        (str(item.get("market")), str(item.get("selection")), float(item.get("line", NO_LINE)))
        for item in odds_rows
        if item.get("line") is not None
    }
    for selection, column in (
        ("over", "odd_over25_open"),
        ("under", "odd_under25_open"),
    ):
        odd = core.valid_odd(shadow_row.get(column))
        key = ("totals", selection, 2.5)
        if odd is None or key in existing:
            continue
        odds_rows.append(
            {
                "market": "totals",
                "selection": selection,
                "line": 2.5,
                "open_odd": odd,
                "first_observed_at": None,
                "open_source": "goal_model_shadow / legacy totals 2.5",
                "open_bookmakers_count": None,
            }
        )


def market_model_values(
    projection: dict,
    market: str,
    selection: str,
    line: float,
    odd: float,
) -> tuple[float, float | None, float] | None:
    if market == "h2h" and selection in {"home", "draw", "away"}:
        block = projection["match_result"]
        probability = float(block[selection])
        fair_odd = block.get(f"fair_odd_{selection}")
        return probability, fair_odd, expected_value(probability, odd)

    if market == "btts" and selection in {"yes", "no"}:
        block = projection["btts"]
        probability = float(block[selection])
        fair_odd = block.get(f"fair_odd_{selection}")
        return probability, fair_odd, expected_value(probability, odd)

    if market == "totals" and selection in {"over", "under"}:
        if abs(line % 1.0 - 0.5) < 1e-9:
            key = f"{line:.1f}"
            if key in projection["totals"]:
                block = projection["totals"][key]
                probability = float(block[selection])
                fair_odd = block.get(f"fair_odd_{selection}")
                return probability, fair_odd, expected_value(probability, odd)

        asian_key = f"{line:.2f}"
        if asian_key in projection["asian_totals"]:
            settlement = projection["asian_totals"][asian_key][selection]
            fair_odd = settlement.get("fair_odd")
            probability = 0.0 if not fair_odd else 1.0 / float(fair_odd)
            return probability, fair_odd, asian_expected_value(settlement, odd)

    if market in {"team_total_home", "team_total_away"} and selection in {"over", "under"}:
        team = "home" if market.endswith("home") else "away"
        key = f"{line:.1f}"
        block = projection["team_totals"].get(team, {}).get(key)
        if block:
            probability = float(block[selection])
            fair_odd = block.get(f"fair_odd_{selection}")
            return probability, fair_odd, expected_value(probability, odd)

    return None


def build_opportunities(projection: dict, odds_rows: list[dict]) -> list[dict]:
    opportunities: list[dict] = []
    seen: set[tuple[str, str, float]] = set()

    for item in odds_rows:
        market = str(item.get("market") or "")
        selection = str(item.get("selection") or "")
        try:
            line = float(item.get("line", NO_LINE))
        except (TypeError, ValueError):
            continue
        odd = core.valid_odd(item.get("open_odd"))
        if odd is None:
            continue

        identity = (market, selection, line)
        if identity in seen:
            continue
        seen.add(identity)

        values = market_model_values(
            projection,
            market,
            selection,
            line,
            odd,
        )
        if values is None:
            continue

        probability, fair_odd, ev = values
        implied = 1.0 / odd
        opportunities.append(
            {
                "market": market,
                "selection": selection,
                "line": None if line == NO_LINE else line,
                "odd": float(odd),
                "fair_odd": None if fair_odd is None else float(fair_odd),
                "model_probability": float(probability),
                "implied_probability": float(implied),
                "edge_pp": (float(probability) - float(implied)) * 100.0,
                "ev": float(ev),
                "source": item.get("open_source"),
                "first_observed_at": item.get("first_observed_at"),
                "bookmakers_count": item.get("open_bookmakers_count"),
            }
        )

    opportunities.sort(key=lambda row: float(row["ev"]), reverse=True)
    return opportunities


def update_open_rows(connection) -> int:
    rows = core.fetch_dicts(
        connection,
        f"""
        SELECT
            id,
            fixture_key,
            projection_json,
            odd_over25_open,
            odd_under25_open
        FROM {SHADOW_TABLE}
        WHERE status = 'OPEN'
          AND fixture_key IS NOT NULL
        """,
    )
    updated = 0
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for row in rows:
        try:
            projection = json.loads(str(row["projection_json"]))
        except (TypeError, json.JSONDecodeError):
            continue

        odds_rows = load_market_odds(connection, row.get("fixture_key"))
        append_legacy_over25(odds_rows, row)
        opportunities = build_opportunities(projection, odds_rows)
        best = opportunities[0] if opportunities else None

        fields = {
            "market_odds_json": json.dumps(odds_rows, ensure_ascii=False, sort_keys=True),
            "market_values_json": json.dumps(opportunities, ensure_ascii=False, sort_keys=True),
            "market_odds_count": len(opportunities),
            "best_market": best.get("market") if best else None,
            "best_selection": best.get("selection") if best else None,
            "best_line": best.get("line") if best else None,
            "best_odd": best.get("odd") if best else None,
            "best_fair_odd": best.get("fair_odd") if best else None,
            "best_probability": best.get("model_probability") if best else None,
            "best_implied_probability": best.get("implied_probability") if best else None,
            "best_edge_pp": best.get("edge_pp") if best else None,
            "best_ev": best.get("ev") if best else None,
            "best_positive_ev": int(best is not None and float(best["ev"]) > 0.0),
        }

        assignments = ", ".join(f"{name} = ?" for name in fields)
        connection.execute(
            f"UPDATE {SHADOW_TABLE} SET {assignments}, updated_at = ? WHERE id = ?",
            tuple(fields.values()) + (now, int(row["id"])),
        )
        updated += 1

    connection.commit()
    return updated


def split_line(line: float) -> tuple[float, ...]:
    scaled = round(float(line) * 4)
    normalized = scaled / 4.0
    if scaled % 4 in {0, 2}:
        return (normalized,)
    return (normalized - 0.25, normalized + 0.25)


def settlement_coefficient(goals: int, line: float, selection: str) -> float:
    results: list[float] = []
    for part in split_line(line):
        if selection == "over":
            result = 1.0 if goals > part else (0.0 if goals == part else -1.0)
        elif selection == "under":
            result = 1.0 if goals < part else (0.0 if goals == part else -1.0)
        else:
            raise ValueError("seleção inválida")
        results.append(result)
    return sum(results) / len(results)


def settle_best(
    market: object,
    selection: object,
    line: object,
    odd: object,
    home_goals: int,
    away_goals: int,
) -> tuple[str | None, float | None]:
    market_text = str(market or "")
    selection_text = str(selection or "")
    odd_value = core.valid_odd(odd)
    if not market_text or not selection_text or odd_value is None:
        return None, None

    if market_text == "h2h":
        win = (
            (selection_text == "home" and home_goals > away_goals)
            or (selection_text == "draw" and home_goals == away_goals)
            or (selection_text == "away" and home_goals < away_goals)
        )
        return ("WIN", odd_value - 1.0) if win else ("LOSS", -1.0)

    if market_text == "btts":
        btts = home_goals > 0 and away_goals > 0
        win = btts if selection_text == "yes" else not btts
        return ("WIN", odd_value - 1.0) if win else ("LOSS", -1.0)

    if market_text not in {"totals", "team_total_home", "team_total_away"}:
        return None, None

    try:
        line_value = float(line)
    except (TypeError, ValueError):
        return None, None

    goals = home_goals + away_goals
    if market_text == "team_total_home":
        goals = home_goals
    elif market_text == "team_total_away":
        goals = away_goals

    coefficient = round(
        settlement_coefficient(goals, line_value, selection_text) * 2.0
    ) / 2.0
    names = {
        1.0: "WIN",
        0.5: "HALF_WIN",
        0.0: "PUSH",
        -0.5: "HALF_LOSS",
        -1.0: "LOSS",
    }
    if coefficient > 0:
        profit = coefficient * (odd_value - 1.0)
    else:
        profit = coefficient
    return names.get(coefficient), float(profit)


def settle_ranked_rows(connection) -> int:
    rows = core.fetch_dicts(
        connection,
        f"""
        SELECT
            id,
            best_market,
            best_selection,
            best_line,
            best_odd,
            home_goals,
            away_goals
        FROM {SHADOW_TABLE}
        WHERE status = 'SETTLED'
          AND best_market IS NOT NULL
          AND best_profit IS NULL
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        """,
    )
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    settled = 0

    for row in rows:
        result, profit = settle_best(
            row.get("best_market"),
            row.get("best_selection"),
            row.get("best_line"),
            row.get("best_odd"),
            int(row["home_goals"]),
            int(row["away_goals"]),
        )
        if result is None:
            continue
        connection.execute(
            f"""
            UPDATE {SHADOW_TABLE}
            SET best_result = ?, best_profit = ?, updated_at = ?
            WHERE id = ?
            """,
            (result, profit, now, int(row["id"])),
        )
        settled += 1

    connection.commit()
    return settled


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — GOAL MODEL VALUE SHADOW")
    print("==========================================")
    print()

    connection = core.connect_turso()
    try:
        if not core.table_exists(connection, SHADOW_TABLE):
            raise RuntimeError("A tabela goal_model_shadow ainda não existe.")

        ensure_value_columns(connection)
        updated = update_open_rows(connection)
        settled = settle_ranked_rows(connection)

        open_total = connection.execute(
            f"SELECT COUNT(*) FROM {SHADOW_TABLE} WHERE status = 'OPEN'"
        ).fetchone()[0]
        positive = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {SHADOW_TABLE}
            WHERE status = 'OPEN'
              AND best_positive_ev = 1
            """
        ).fetchone()[0]

        top_rows = core.fetch_dicts(
            connection,
            f"""
            SELECT
                fixture_date,
                home_team,
                away_team,
                best_market,
                best_selection,
                best_line,
                best_odd,
                best_fair_odd,
                best_ev
            FROM {SHADOW_TABLE}
            WHERE status = 'OPEN'
              AND best_ev IS NOT NULL
            ORDER BY best_ev DESC
            LIMIT 10
            """,
        )

        print(f"Shadow OPEN analisados      : {updated}/{open_total}")
        print(f"Melhor mercado EV positivo  : {positive}")
        print(f"Resultados best liquidados  : {settled}")

        if top_rows:
            print()
            print("TOP VALUE (shadow, sem threshold):")
            for row in top_rows:
                line = row.get("best_line")
                line_text = "" if line is None else f" {float(line):g}"
                print(
                    f" - {row['fixture_date']} | {row['home_team']} vs {row['away_team']} | "
                    f"{row.get('best_market')} {row.get('best_selection')}{line_text} | "
                    f"odd {float(row['best_odd']):.3f} | "
                    f"fair {float(row['best_fair_odd']):.3f} | "
                    f"EV {float(row['best_ev']):+.2%}"
                )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
