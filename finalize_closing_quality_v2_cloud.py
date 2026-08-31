from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

from ml.paper_trade import normalize_league

from closing_odds_v3_cloud import (
    API_FOOTBALL_PREMATCH_SOURCE,
    APP_TIMEZONE,
    OBS_TABLE,
    ensure_observation_table,
    fetch_api_football_fixtures,
    fetch_api_football_prematch_odds,
    find_api_football_fixture,
    fixture_details,
    parse_api_datetime,
    parse_fixture_datetime,
    record_observation,
    safe_int,
)
from update_results_cloud import (
    FOOTBALL_DATA_DIVISION,
    add_column_if_missing,
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


API_FOOTBALL_RECOVERED_SOURCE = (
    "API-Football / recovered latest pre-match snapshot / post-game backfill"
)
API_FOOTBALL_HISTORY_DAYS = 6
API_FOOTBALL_RECHECK_HOURS = 24
API_FOOTBALL_BACKFILL_MAX_REQUESTS = 35


def is_api_football_live_source(value: object) -> bool:
    source = str(value or "").casefold()
    return "api-football" in source and "live pre-kickoff" in source


def is_api_football_provisional_source(value: object) -> bool:
    source = str(value or "").casefold()
    return "api-football" in source and not is_api_football_live_source(source)


def is_final_prospective_source(value: object) -> bool:
    source = str(value or "").casefold()
    return "the odds api" in source or is_api_football_live_source(source)


def has_valid_close(row: dict) -> bool:
    return (
        safe_odd(row.get("odd_over25_close")) is not None
        and safe_odd(row.get("odd_under25_close")) is not None
    )


def update_reference_close(
    connection,
    table: str,
    row: dict,
    over: float,
    under: float,
    source: str,
    observed_at: str,
    minutes_before_kickoff: float | None,
) -> bool:
    clv = calculate_clv(row.get("odd_over25_open"), over)

    connection.execute(
        f"""
        UPDATE {table}
        SET
            odd_over25_close = ?,
            odd_under25_close = ?,
            closing_observed_at = ?,
            closing_minutes_before_kickoff = ?,
            closing_source = ?,
            clv = ?
        WHERE id = ?
        """,
        (
            over,
            under,
            observed_at,
            minutes_before_kickoff,
            source,
            clv,
            int(row["id"]),
        ),
    )

    row["odd_over25_close"] = over
    row["odd_under25_close"] = under
    row["closing_observed_at"] = observed_at
    row["closing_minutes_before_kickoff"] = minutes_before_kickoff
    row["closing_source"] = source
    row["clv"] = clv
    return True


def mark_backfill_checked(
    connection,
    table: str,
    row_id: int,
    when: datetime,
) -> None:
    connection.execute(
        f"""
        UPDATE {table}
        SET closing_api_football_backfill_checked_at = ?
        WHERE id = ?
        """,
        (when.isoformat(timespec="seconds"), row_id),
    )


def backfill_due(row: dict, now: datetime) -> bool:
    checked = parse_api_datetime(
        row.get("closing_api_football_backfill_checked_at")
    )
    if checked is None:
        return True
    return (now - checked).total_seconds() >= API_FOOTBALL_RECHECK_HOURS * 3600


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — FINALIZE CLOSING QUALITY V2")
    print("==========================================")
    print()

    now = datetime.now(APP_TIMEZONE)
    today = now.date()
    season_start = season_start_for(today)
    season_code = football_data_season_code(today)
    api_football_key = os.getenv("API_FOOTBALL_KEY", "").strip()

    connection = connect_turso()
    try:
        table = find_paper_table(connection)
        if table is None:
            raise RuntimeError("Tabela de Paper Trade não encontrada.")

        columns = table_columns(connection, table)
        for name, sql_type in (
            ("odd_over25_close", "REAL"),
            ("odd_under25_close", "REAL"),
            ("closing_observed_at", "TEXT"),
            ("closing_minutes_before_kickoff", "REAL"),
            ("closing_source", "TEXT"),
            ("clv", "REAL"),
            ("api_football_fixture_id", "INTEGER"),
            ("closing_api_football_backfill_checked_at", "TEXT"),
        ):
            add_column_if_missing(connection, table, columns, name, sql_type)

        ensure_observation_table(connection)
        connection.commit()

        fixture_date_column = (
            "fixture_date"
            if "fixture_date" in columns
            else "date"
            if "date" in columns
            else None
        )
        fixture_time_column = (
            "fixture_time"
            if "fixture_time" in columns
            else "time"
            if "time" in columns
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
            (today.isoformat(), season_start.isoformat()),
        )

        # ====================================================
        # STEP 1 — Football-Data final closing odds
        # ====================================================
        fd_candidates: list[dict] = []
        needed_fd: dict[str, str] = {}

        for row in rows:
            source = row.get("closing_source")
            if is_final_prospective_source(source):
                # Never replace an actually observed close near kickoff.
                continue

            needs_close = not has_valid_close(row)
            provisional = is_api_football_provisional_source(source)
            if not needs_close and not provisional:
                continue

            league = normalize_league(row.get("league"))
            fixture_date = parse_db_date(row.get(fixture_date_column))
            if not isinstance(league, str) or not isinstance(fixture_date, date):
                continue

            division = FOOTBALL_DATA_DIVISION.get(league)
            if not division:
                continue

            fd_candidates.append(
                {
                    "row": row,
                    "league": league,
                    "date": fixture_date,
                    "division": division,
                    "provisional": provisional,
                }
            )
            needed_fd[league] = division

        print(f"Trades da época            : {len(rows)}")
        print(f"Candidatos Football-Data   : {len(fd_candidates)}")

        results_by_league: dict[str, list[dict]] = {}
        for league, division in sorted(needed_fd.items()):
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

        fd_filled = 0
        fd_replaced_provisional = 0
        fd_unmatched = 0
        fd_finalized_ids: set[int] = set()

        for item in fd_candidates:
            row = item["row"]
            result = find_simple_result(
                results_by_league.get(item["league"], []),
                item["league"],
                item["date"],
                str(row.get("home_team") or ""),
                str(row.get("away_team") or ""),
            )
            if result is None:
                fd_unmatched += 1
                continue

            over = safe_odd(result.get("odd_over25_close"))
            under = safe_odd(result.get("odd_under25_close"))
            source = result.get("closing_source")
            if over is None or under is None or not source:
                fd_unmatched += 1
                continue

            observed_at_dt = datetime.now(timezone.utc).replace(microsecond=0)
            record_observation(
                connection,
                row,
                str(source),
                "POST_GAME_FINAL",
                observed_at_dt.astimezone(APP_TIMEZONE),
                None,
                over,
                under,
            )

            update_reference_close(
                connection,
                table,
                row,
                over,
                under,
                str(source),
                observed_at_dt.isoformat(),
                None,
            )
            fd_finalized_ids.add(int(row["id"]))

            if item["provisional"]:
                fd_replaced_provisional += 1
                action = "substituiu snapshot provisória"
            else:
                fd_filled += 1
                action = "preencheu CLOSE em falta"

            print(
                f"✅ Football-Data {item['date']} | "
                f"{row.get('home_team')} vs {row.get('away_team')} "
                f"| {action} | Over {over:.3f}"
            )

        connection.commit()

        # ====================================================
        # STEP 2 — API-Football 7-day historical fallback
        # ====================================================
        recent_cutoff = today - timedelta(days=API_FOOTBALL_HISTORY_DAYS)
        api_candidates: list[dict] = []

        for row in rows:
            row_id = int(row["id"])
            if row_id in fd_finalized_ids:
                continue
            if has_valid_close(row):
                # Keep any valid pre-match snapshot. This fallback is only to
                # turn None into data; Football-Data will replace provisional
                # values later if a true closing price becomes available.
                continue

            fixture_date = parse_db_date(row.get(fixture_date_column))
            league = normalize_league(row.get("league"))
            if not isinstance(fixture_date, date) or not isinstance(league, str):
                continue
            if fixture_date < recent_cutoff or fixture_date > today:
                continue
            if not backfill_due(row, now):
                continue

            fixture_datetime = parse_fixture_datetime(
                row.get(fixture_date_column),
                row.get(fixture_time_column) if fixture_time_column else None,
            )
            if fixture_datetime is None:
                continue

            api_candidates.append(
                {
                    "row": row,
                    "league": league,
                    "date": fixture_date,
                    "fixture_datetime": fixture_datetime,
                }
            )

        print()
        print(f"Candidatos API-Football   : {len(api_candidates)}")
        print(
            "API-Football key          : "
            + ("✅ configurada" if api_football_key else "⚪ sem chave")
        )

        api_requests = 0
        api_remaining = None
        api_filled = 0
        api_no_odds = 0
        api_fixture_unmatched = 0
        budget_stopped = 0

        if api_candidates and api_football_key:
            # Only dates with candidates lacking a known fixture id need the
            # relatively expensive fixture-list call.
            need_lookup = [
                item
                for item in api_candidates
                if safe_int(item["row"].get("api_football_fixture_id")) is None
            ]
            dates_needed = sorted(
                {item["date"].isoformat() for item in need_lookup}
            )
            fixtures_by_date: dict[str, list[dict]] = {}

            for fixture_date_text in dates_needed:
                if api_requests >= API_FOOTBALL_BACKFILL_MAX_REQUESTS:
                    budget_stopped += 1
                    break
                try:
                    fixtures, used, remaining = fetch_api_football_fixtures(
                        fixture_date_text,
                        api_football_key,
                    )
                except Exception as error:
                    print(
                        f"⚠ API-Football fixtures {fixture_date_text}: {error}"
                    )
                    continue
                api_requests += used
                api_remaining = remaining or api_remaining
                fixtures_by_date[fixture_date_text] = fixtures

            for item in api_candidates:
                if api_requests >= API_FOOTBALL_BACKFILL_MAX_REQUESTS:
                    budget_stopped += 1
                    break

                row = item["row"]
                row_id = int(row["id"])
                fixture_id = safe_int(row.get("api_football_fixture_id"))

                if fixture_id is None:
                    match = find_api_football_fixture(
                        fixtures_by_date.get(item["date"].isoformat(), []),
                        item["league"],
                        item["fixture_datetime"],
                        row.get("home_team"),
                        row.get("away_team"),
                    )
                    if match is not None:
                        fixture_id, _, _, _ = fixture_details(match)
                        if fixture_id is not None:
                            connection.execute(
                                f"""
                                UPDATE {table}
                                SET api_football_fixture_id = ?
                                WHERE id = ?
                                """,
                                (fixture_id, row_id),
                            )
                            row["api_football_fixture_id"] = fixture_id

                mark_backfill_checked(connection, table, row_id, now)

                if fixture_id is None:
                    api_fixture_unmatched += 1
                    continue

                try:
                    over, under, snapshot, used, remaining = (
                        fetch_api_football_prematch_odds(
                            fixture_id,
                            api_football_key,
                        )
                    )
                except Exception as error:
                    print(
                        f"⚠ API-Football historical fixture {fixture_id}: {error}"
                    )
                    continue

                api_requests += used
                api_remaining = remaining or api_remaining

                if over is None or under is None or snapshot is None:
                    api_no_odds += 1
                    continue

                minutes_before = (
                    item["fixture_datetime"] - snapshot
                ).total_seconds() / 60.0
                if minutes_before < 0:
                    api_no_odds += 1
                    continue

                record_observation(
                    connection,
                    row,
                    API_FOOTBALL_RECOVERED_SOURCE,
                    "PROVISIONAL_RECOVERED",
                    snapshot,
                    minutes_before,
                    over,
                    under,
                )

                update_reference_close(
                    connection,
                    table,
                    row,
                    over,
                    under,
                    API_FOOTBALL_RECOVERED_SOURCE,
                    snapshot.isoformat(timespec="seconds"),
                    minutes_before,
                )
                api_filled += 1

                print(
                    f"🟡 API-Football recuperada {item['date']} | "
                    f"{row.get('home_team')} vs {row.get('away_team')} "
                    f"| {minutes_before:.0f} min antes | Over {over:.3f}"
                )

            connection.commit()

        # ====================================================
        # COVERAGE SUMMARY
        # ====================================================
        coverage = connection.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN odd_over25_close > 1 AND odd_under25_close > 1
                         THEN 1 ELSE 0 END) AS with_close,
                SUM(CASE WHEN odd_over25_close IS NULL OR odd_over25_close <= 1
                           OR odd_under25_close IS NULL OR odd_under25_close <= 1
                         THEN 1 ELSE 0 END) AS without_close
            FROM {table}
            WHERE date({fixture_date_column}) <= date(?)
              AND date({fixture_date_column}) >= date(?)
            """,
            (today.isoformat(), season_start.isoformat()),
        ).fetchone()

        total = int(coverage[0] or 0)
        with_close = int(coverage[1] or 0)
        without_close = int(coverage[2] or 0)
        coverage_pct = 100.0 * with_close / total if total else 0.0

        print()
        print("==========================================")
        print(" RESUMO")
        print("==========================================")
        print(f"Football-Data preenchidas           : {fd_filled}")
        print(
            f"Football-Data substituiu provisórias: {fd_replaced_provisional}"
        )
        print(f"Football-Data ainda sem match/close : {fd_unmatched}")
        print(f"API-Football recuperadas            : {api_filled}")
        print(f"API-Football sem odds 2.5           : {api_no_odds}")
        print(f"API-Football fixture não encontrada : {api_fixture_unmatched}")
        print(f"Pedidos API-Football backfill       : {api_requests}")
        if api_remaining is not None:
            print(f"API-Football restantes              : {api_remaining}")
        if budget_stopped:
            print(
                f"Backfill parado por orçamento        : "
                f"{API_FOOTBALL_BACKFILL_MAX_REQUESTS} pedidos/run"
            )
        print(f"Paper Trades época                  : {total}")
        print(f"Com alguma CLOSE                    : {with_close}")
        print(f"Sem qualquer CLOSE                  : {without_close}")
        print(f"Cobertura CLOSE                     : {coverage_pct:.1f}%")
        print()
        print("Prioridade dos dados:")
        print("1. The Odds API / API-Football LIVE observada pré-kickoff.")
        print("2. Football-Data AvgC closing pós-jogo.")
        print("3. API-Football pre-match snapshot provisória/recuperada.")
        print(
            f"Todas as observações ficam auditadas em {OBS_TABLE}."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
