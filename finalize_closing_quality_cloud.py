from __future__ import annotations

from datetime import date, datetime, timezone

from ml.paper_trade import normalize_league

from update_results_cloud import (
    FOOTBALL_DATA_DIVISION,
    calculate_clv,
    connect_turso,
    fetch_dicts,
    fetch_football_data_league,
    find_paper_table,
    find_simple_result,
    football_data_season_code,
    parse_db_date,
    safe_odd,
    season_start_for,
    table_columns,
)


FINAL_SOURCE_MARKERS = (
    "football-data",
    "the odds api",
)


def is_provisional_source(value: object) -> bool:
    source = str(value or "").casefold()
    return "api-football" in source or "provisional" in source


def is_final_source(value: object) -> bool:
    source = str(value or "").casefold()
    return any(marker in source for marker in FINAL_SOURCE_MARKERS)


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — FINALIZE CLOSING QUALITY")
    print("==========================================")
    print()

    today = datetime.now().date()
    season_start = season_start_for(today)
    season_code = football_data_season_code(today)

    connection = connect_turso()
    try:
        table = find_paper_table(connection)
        if table is None:
            raise RuntimeError("Tabela de Paper Trade não encontrada.")

        columns = table_columns(connection, table)
        required = {
            "id",
            "league",
            "home_team",
            "away_team",
            "odd_over25_open",
            "odd_over25_close",
            "odd_under25_close",
            "closing_source",
        }
        missing = required - columns
        if missing:
            raise RuntimeError(
                "Colunas em falta: " + ", ".join(sorted(missing))
            )

        fixture_date_column = (
            "fixture_date"
            if "fixture_date" in columns
            else "date"
            if "date" in columns
            else None
        )
        if fixture_date_column is None:
            raise RuntimeError("A tabela não tem fixture_date/date.")

        rows = fetch_dicts(
            connection,
            f"""
            SELECT *
            FROM {table}
            WHERE date({fixture_date_column}) <= date(?)
              AND date({fixture_date_column}) >= date(?)
            ORDER BY {fixture_date_column}, id
            """,
            (
                today.isoformat(),
                season_start.isoformat(),
            ),
        )

        candidates = []
        needed: dict[str, str] = {}

        for row in rows:
            over = safe_odd(row.get("odd_over25_close"))
            under = safe_odd(row.get("odd_under25_close"))
            source = row.get("closing_source")

            needs_close = over is None or under is None
            provisional = is_provisional_source(source)

            if not needs_close and not provisional:
                continue

            league = normalize_league(row.get("league"))
            fixture_date = parse_db_date(row.get(fixture_date_column))
            if not isinstance(league, str) or not isinstance(fixture_date, date):
                continue

            division = FOOTBALL_DATA_DIVISION.get(league)
            if not division:
                continue

            candidates.append(
                {
                    "row": row,
                    "league": league,
                    "date": fixture_date,
                    "division": division,
                    "provisional": provisional,
                }
            )
            needed[league] = division

        print(f"Trades da época            : {len(rows)}")
        print(f"A finalizar                : {len(candidates)}")

        if not candidates:
            print("✅ Nada para finalizar.")
            return

        results_by_league: dict[str, list[dict]] = {}
        for league, division in sorted(needed.items()):
            results, error = fetch_football_data_league(
                league,
                division,
                season_code,
            )
            if error:
                print(f"⚠ {division} {league}: {error[:180]}")
                continue
            results_by_league[league] = results
            with_close = sum(
                1
                for item in results
                if safe_odd(item.get("odd_over25_close")) is not None
                and safe_odd(item.get("odd_under25_close")) is not None
            )
            print(
                f"✅ {division} {league}: "
                f"{len(results)} resultados | {with_close} com CLOSE"
            )

        replaced_provisional = 0
        filled_missing = 0
        recalculated_clv = 0
        unmatched = 0

        for item in candidates:
            row = item["row"]
            league = item["league"]
            fixture_date = item["date"]

            result = find_simple_result(
                results_by_league.get(league, []),
                league,
                fixture_date,
                str(row.get("home_team") or ""),
                str(row.get("away_team") or ""),
            )
            if result is None:
                unmatched += 1
                continue

            over = safe_odd(result.get("odd_over25_close"))
            under = safe_odd(result.get("odd_under25_close"))
            source = result.get("closing_source")

            if over is None or under is None or not source:
                unmatched += 1
                continue

            old_over = safe_odd(row.get("odd_over25_close"))
            old_under = safe_odd(row.get("odd_under25_close"))
            old_source = row.get("closing_source")

            if (
                old_over is not None
                and old_under is not None
                and is_final_source(old_source)
            ):
                continue

            clv = calculate_clv(
                row.get("odd_over25_open"),
                over,
            )
            observed_at = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )

            connection.execute(
                f"""
                UPDATE {table}
                SET
                    odd_over25_close = ?,
                    odd_under25_close = ?,
                    closing_observed_at = ?,
                    closing_minutes_before_kickoff = NULL,
                    closing_source = ?,
                    clv = ?
                WHERE id = ?
                """,
                (
                    over,
                    under,
                    observed_at,
                    str(source),
                    clv,
                    int(row["id"]),
                ),
            )

            if item["provisional"]:
                replaced_provisional += 1
                action = "substituída snapshot API-Football"
            else:
                filled_missing += 1
                action = "preenchida CLOSE em falta"

            if clv is not None:
                recalculated_clv += 1

            print(
                f"✅ {fixture_date} | "
                f"{row.get('home_team')} vs {row.get('away_team')} "
                f"| {action} | Over {over:.3f}"
            )

        connection.commit()

        print()
        print("==========================================")
        print(" RESUMO")
        print("==========================================")
        print(
            f"Snapshots provisórias substituídas: "
            f"{replaced_provisional}"
        )
        print(f"CLOSE em falta preenchidas          : {filled_missing}")
        print(f"CLV recalculados                    : {recalculated_clv}")
        print(f"Sem fecho Football-Data ainda       : {unmatched}")
        print()
        print(
            "Prioridade final: The Odds API pré-kickoff "
            "> Football-Data AvgC pós-jogo "
            "> API-Football snapshot provisória."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
