from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from agent.champion_v10 import CHAMPION_VERSION, evaluate_champion
from agent.multi_analyst import score_pre_match
from footballai_agent_cloud import (
    connect_turso,
    fetch_dicts,
    load_sports_model,
    predict_sports_probability,
    table_exists,
)
from ml.paper_trade import (
    build_market_features,
    build_sports_features,
    calculate_current_elo,
    create_team_history,
    normalize_league,
    normalize_team,
    parse_date,
)
from paper_trade_cloud import (
    load_historical_matches,
    resolve_team_to_history,
    valid_odd,
)


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
UPCOMING_TABLE = "upcoming_matches"
PAPER_TABLE = "paper_trades"
LEGACY_TABLE = "agent_v10_shadow_decisions"
UNIVERSE_TABLE = "agent_v10_universe_decisions"
RUN_TABLE = "agent_v10_universe_runs"


def ensure_tables(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {UNIVERSE_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_key TEXT NOT NULL UNIQUE,
            paper_trade_id INTEGER,
            evaluated_at TEXT NOT NULL,
            champion_version TEXT NOT NULL,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            source TEXT,
            odd_over25_open REAL NOT NULL,
            odd_under25_open REAL,
            market_probability REAL NOT NULL,
            sports_probability REAL NOT NULL,
            expected_value REAL NOT NULL,
            sports_edge_market REAL NOT NULL,
            clv_probability REAL,
            features_json TEXT NOT NULL,
            goal_score REAL NOT NULL,
            attack_score REAL NOT NULL,
            ml_score REAL NOT NULL,
            value_score REAL NOT NULL,
            stats_score REAL NOT NULL,
            goal_vote INTEGER NOT NULL,
            attack_vote INTEGER NOT NULL,
            ml_vote INTEGER NOT NULL,
            value_vote INTEGER NOT NULL,
            vote_count INTEGER NOT NULL,
            decision TEXT NOT NULL,
            result_over25 INTEGER,
            profit REAL,
            resolution_source TEXT,
            resolved_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{UNIVERSE_TABLE}_fixture
        ON {UNIVERSE_TABLE}(fixture_date, decision)
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{UNIVERSE_TABLE}_paper
        ON {UNIVERSE_TABLE}(paper_trade_id)
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            champion_version TEXT NOT NULL,
            upcoming_seen INTEGER NOT NULL,
            already_frozen INTEGER NOT NULL,
            inserted INTEGER NOT NULL,
            bets INTEGER NOT NULL,
            passes INTEGER NOT NULL,
            skipped_timing INTEGER NOT NULL,
            skipped_data INTEGER NOT NULL,
            resolved_updated INTEGER NOT NULL,
            linked_paper INTEGER NOT NULL,
            migrated_legacy INTEGER NOT NULL
        )
        """
    )
    connection.commit()


def canonical_fixture_key(
    fixture_date: object,
    league: object,
    home_team: object,
    away_team: object,
) -> str:
    return "|".join(
        str(value or "").strip().casefold()
        for value in (fixture_date, league, home_team, away_team)
    )


def _parse_same_day_kickoff(fixture_time: object, now_local: datetime) -> datetime | None:
    raw = str(fixture_time or "").strip()
    if not raw:
        return None
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?", raw)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return None
    return now_local.replace(
        hour=hour,
        minute=minute,
        second=second,
        microsecond=0,
    )


def is_safely_pre_kickoff(row: dict, now_local: datetime) -> bool:
    fixture_date = str(row.get("date") or row.get("fixture_date") or "").strip()
    if not fixture_date:
        return False
    today = now_local.date().isoformat()
    if fixture_date > today:
        return True
    if fixture_date < today:
        return False
    kickoff = _parse_same_day_kickoff(
        row.get("time") if "time" in row else row.get("fixture_time"),
        now_local,
    )
    if kickoff is None:
        return False
    return now_local <= kickoff - timedelta(minutes=5)


def load_upcoming_universe(connection) -> list[dict]:
    today = datetime.now(APP_TIMEZONE).date().isoformat()
    return fetch_dicts(
        connection,
        f"""
        SELECT fixture_key, date, time, league, home_team, away_team,
               odd_over25_open, odd_under25_open,
               odd_over25_current, odd_under25_current,
               first_observed_at, current_observed_at, source, active
        FROM {UPCOMING_TABLE}
        WHERE active = 1
          AND date >= ?
        ORDER BY date, time, league, home_team, away_team
        """,
        (today,),
    )


def existing_fixture_keys(connection) -> set[str]:
    rows = connection.execute(
        f"SELECT fixture_key FROM {UNIVERSE_TABLE}"
    ).fetchall()
    return {str(row[0]) for row in rows}


def find_paper_trade(connection, fixture_date: str, league: str, home: str, away: str):
    if not table_exists(connection, PAPER_TABLE):
        return None
    row = connection.execute(
        f"""
        SELECT id, clv_probability
        FROM {PAPER_TABLE}
        WHERE fixture_date = ?
          AND league = ?
          AND home_team = ?
          AND away_team = ?
          AND market = 'over25'
        ORDER BY id
        LIMIT 1
        """,
        (fixture_date, league, home, away),
    ).fetchone()
    return row


def migrate_legacy(connection, now_utc: str) -> int:
    if not table_exists(connection, LEGACY_TABLE):
        return 0

    rows = fetch_dicts(
        connection,
        f"""
        SELECT d.*, p.features_json AS legacy_features_json
        FROM {LEGACY_TABLE} d
        LEFT JOIN {PAPER_TABLE} p ON p.id = d.paper_trade_id
        ORDER BY d.id
        """ if table_exists(connection, PAPER_TABLE) else f"SELECT d.*, NULL AS legacy_features_json FROM {LEGACY_TABLE} d ORDER BY d.id",
    )
    migrated = 0
    for row in rows:
        key_row = None
        if table_exists(connection, UPCOMING_TABLE):
            key_row = connection.execute(
                f"""
                SELECT fixture_key
                FROM {UPCOMING_TABLE}
                WHERE date = ? AND league = ? AND home_team = ? AND away_team = ?
                LIMIT 1
                """,
                (
                    str(row["fixture_date"]),
                    str(row["league"]),
                    str(row["home_team"]),
                    str(row["away_team"]),
                ),
            ).fetchone()
        fixture_key = (
            str(key_row[0])
            if key_row is not None
            else canonical_fixture_key(
                row["fixture_date"], row["league"], row["home_team"], row["away_team"]
            )
        )
        before = connection.total_changes
        connection.execute(
            f"""
            INSERT OR IGNORE INTO {UNIVERSE_TABLE} (
                fixture_key, paper_trade_id, evaluated_at, champion_version,
                fixture_date, fixture_time, league, home_team, away_team, source,
                odd_over25_open, odd_under25_open, market_probability,
                sports_probability, expected_value, sports_edge_market, clv_probability,
                features_json, goal_score, attack_score, ml_score, value_score, stats_score,
                goal_vote, attack_vote, ml_vote, value_vote, vote_count, decision,
                result_over25, profit, resolution_source, resolved_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fixture_key,
                row.get("paper_trade_id"),
                str(row["evaluated_at"]),
                str(row["champion_version"]),
                str(row["fixture_date"]),
                row.get("fixture_time"),
                str(row["league"]),
                str(row["home_team"]),
                str(row["away_team"]),
                "legacy-paper-trade",
                float(row["odd_over25_open"]),
                None,
                float(row["market_probability"]),
                float(row["sports_probability"]),
                float(row["expected_value"]),
                float(row["sports_edge_market"]),
                float(row["clv_probability"]) if row.get("clv_probability") is not None else None,
                str(row.get("legacy_features_json") or "{}"),
                float(row["goal_score"]),
                float(row["attack_score"]),
                float(row["ml_score"]),
                float(row["value_score"]),
                float(row["stats_score"]),
                int(row["goal_vote"]),
                int(row["attack_vote"]),
                int(row["ml_vote"]),
                int(row["value_vote"]),
                int(row["vote_count"]),
                str(row["decision"]),
                row.get("result_over25"),
                row.get("profit"),
                "legacy-paper-trade" if row.get("result_over25") is not None else None,
                row.get("resolved_at"),
                str(row.get("created_at") or row["evaluated_at"]),
                str(row.get("updated_at") or now_utc),
            ),
        )
        if connection.total_changes > before:
            migrated += 1
    if migrated:
        connection.commit()
    return migrated


def attach_paper_links(connection, now_utc: str) -> int:
    if not table_exists(connection, PAPER_TABLE):
        return 0
    rows = fetch_dicts(
        connection,
        f"""
        SELECT u.fixture_key, p.id AS paper_trade_id, p.clv_probability
        FROM {UNIVERSE_TABLE} u
        JOIN {PAPER_TABLE} p
          ON p.fixture_date = u.fixture_date
         AND p.league = u.league
         AND p.home_team = u.home_team
         AND p.away_team = u.away_team
         AND p.market = 'over25'
        WHERE u.paper_trade_id IS NULL
        ORDER BY u.id
        """,
    )
    for row in rows:
        connection.execute(
            f"""
            UPDATE {UNIVERSE_TABLE}
            SET paper_trade_id = ?,
                clv_probability = COALESCE(clv_probability, ?),
                updated_at = ?
            WHERE fixture_key = ? AND paper_trade_id IS NULL
            """,
            (
                int(row["paper_trade_id"]),
                row.get("clv_probability"),
                now_utc,
                str(row["fixture_key"]),
            ),
        )
    if rows:
        connection.commit()
    return len(rows)


def _apply_result(connection, fixture_key: str, result: int, odd: float, source: str, now_utc: str) -> bool:
    profit = odd - 1.0 if int(result) == 1 else -1.0
    before = connection.total_changes
    connection.execute(
        f"""
        UPDATE {UNIVERSE_TABLE}
        SET result_over25 = ?, profit = ?, resolution_source = ?,
            resolved_at = ?, updated_at = ?
        WHERE fixture_key = ? AND result_over25 IS NULL
        """,
        (int(result), profit, source, now_utc, now_utc, fixture_key),
    )
    return connection.total_changes > before


def resolve_existing(connection, now_utc: str) -> int:
    updated = 0
    if table_exists(connection, PAPER_TABLE):
        paper_rows = fetch_dicts(
            connection,
            f"""
            SELECT u.fixture_key, u.odd_over25_open, p.result_over25
            FROM {UNIVERSE_TABLE} u
            JOIN {PAPER_TABLE} p ON p.id = u.paper_trade_id
            WHERE u.result_over25 IS NULL AND p.result_over25 IN (0,1)
            ORDER BY u.id
            """,
        )
        for row in paper_rows:
            updated += int(
                _apply_result(
                    connection,
                    str(row["fixture_key"]),
                    int(row["result_over25"]),
                    float(row["odd_over25_open"]),
                    "paper_trades",
                    now_utc,
                )
            )

    if table_exists(connection, "matches"):
        match_rows = fetch_dicts(
            connection,
            f"""
            SELECT u.fixture_key, u.odd_over25_open, m.home_goals, m.away_goals
            FROM {UNIVERSE_TABLE} u
            JOIN matches m
              ON m.date = u.fixture_date
             AND m.league = u.league
             AND m.home_team = u.home_team
             AND m.away_team = u.away_team
            WHERE u.result_over25 IS NULL
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
            ORDER BY u.id, m.id
            """,
        )
        seen: set[str] = set()
        for row in match_rows:
            key = str(row["fixture_key"])
            if key in seen:
                continue
            seen.add(key)
            result = int(float(row["home_goals"]) + float(row["away_goals"]) > 2.5)
            updated += int(
                _apply_result(
                    connection,
                    key,
                    result,
                    float(row["odd_over25_open"]),
                    "matches",
                    now_utc,
                )
            )

    if updated:
        connection.commit()
    return updated


def insert_decision(
    connection,
    row: dict,
    fixture_key: str,
    fixture_date: pd.Timestamp,
    league: str,
    home_team: str,
    away_team: str,
    sports_features: dict[str, float],
    sports_probability: float,
    market_probability: float,
    odd_over: float,
    odd_under: float,
    paper_trade_id: int | None,
    clv_probability: float | None,
    evaluated_at: str,
) -> str:
    expected_value = sports_probability * odd_over - 1.0
    sports_edge_market = sports_probability - market_probability
    scores = score_pre_match(
        sports_features,
        sports_probability,
        expected_value,
        sports_edge_market,
    )
    champion = evaluate_champion(scores, expected_value)
    connection.execute(
        f"""
        INSERT OR IGNORE INTO {UNIVERSE_TABLE} (
            fixture_key, paper_trade_id, evaluated_at, champion_version,
            fixture_date, fixture_time, league, home_team, away_team, source,
            odd_over25_open, odd_under25_open, market_probability,
            sports_probability, expected_value, sports_edge_market, clv_probability,
            features_json, goal_score, attack_score, ml_score, value_score, stats_score,
            goal_vote, attack_vote, ml_vote, value_vote, vote_count, decision,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fixture_key,
            paper_trade_id,
            evaluated_at,
            CHAMPION_VERSION,
            fixture_date.strftime("%Y-%m-%d"),
            row.get("time"),
            league,
            home_team,
            away_team,
            str(row.get("source") or "upcoming_matches"),
            odd_over,
            odd_under,
            market_probability,
            sports_probability,
            expected_value,
            sports_edge_market,
            clv_probability,
            json.dumps(sports_features, ensure_ascii=False, sort_keys=True),
            scores.goal_profile,
            scores.attack_pressure,
            scores.ml_score,
            scores.value_score,
            scores.stats_score,
            int(champion.goal_vote),
            int(champion.attack_vote),
            int(champion.ml_vote),
            int(champion.value_vote),
            champion.vote_count,
            champion.decision,
            evaluated_at,
            evaluated_at,
        ),
    )
    return champion.decision


def print_performance(connection) -> None:
    rows = fetch_dicts(
        connection,
        f"""
        SELECT decision, result_over25, profit, clv_probability
        FROM {UNIVERSE_TABLE}
        WHERE result_over25 IN (0,1)
        ORDER BY fixture_date, id
        """,
    )
    bets = [row for row in rows if row["decision"] == "BET_SHADOW"]
    if not bets:
        print("BET V10 resolvidas          : 0")
        return
    wins = sum(int(row["result_over25"]) for row in bets)
    profit = sum(float(row["profit"]) for row in bets)
    print(f"BET V10 resolvidas          : {len(bets)}")
    print(f"Hit rate V10               : {wins / len(bets):.2%}")
    print(f"Lucro V10                  : {profit:.2f}u")
    print(f"ROI V10                    : {profit / len(bets):.2%}")


def main() -> None:
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    now_local = datetime.now(APP_TIMEZONE)

    print("\n=================================================")
    print(" FOOTBALLAI — V10 FULL UPCOMING UNIVERSE SHADOW")
    print("=================================================\n")
    print(f"Champion                   : {CHAMPION_VERSION}")
    print("Fonte principal             : upcoming_matches")
    print("Dependente de paper_trades? : NÃO")
    print("CLV usado como gate?        : NÃO")

    connection = connect_turso()
    try:
        if not table_exists(connection, UPCOMING_TABLE):
            raise RuntimeError("upcoming_matches não existe na Turso")
        if not table_exists(connection, "matches"):
            raise RuntimeError("matches não existe na Turso")

        ensure_tables(connection)
        migrated = migrate_legacy(connection, started_at)
        linked_before = attach_paper_links(connection, started_at)
        resolved = resolve_existing(connection, started_at)

        upcoming = load_upcoming_universe(connection)
        existing = existing_fixture_keys(connection)
        model, model_features, model_generated_at = load_sports_model()
        matches = load_historical_matches(connection)
        team_history = create_team_history(matches)
        elo_ratings = calculate_current_elo(matches)

        inserted = 0
        bets = 0
        passes = 0
        skipped_timing = 0
        skipped_data = 0
        already = 0
        examples: list[str] = []

        for row in upcoming:
            fixture_key = str(
                row.get("fixture_key")
                or canonical_fixture_key(
                    row.get("date"), row.get("league"), row.get("home_team"), row.get("away_team")
                )
            )
            if fixture_key in existing:
                already += 1
                continue
            if not is_safely_pre_kickoff(row, now_local):
                skipped_timing += 1
                continue

            try:
                fixture_date = parse_date(row.get("date"))
                if fixture_date is None:
                    raise ValueError("data inválida")

                league = normalize_league(row.get("league"))
                if league is None:
                    raise ValueError("liga inválida")
                home_team = normalize_team(league, row.get("home_team"))
                away_team = normalize_team(league, row.get("away_team"))
                if home_team is None or away_team is None:
                    raise ValueError("equipa inválida")

                home_team = resolve_team_to_history(team_history, str(league), str(home_team))
                away_team = resolve_team_to_history(team_history, str(league), str(away_team))

                odd_over = valid_odd(row.get("odd_over25_open"))
                odd_under = valid_odd(row.get("odd_under25_open"))
                if odd_over is None or odd_under is None:
                    raise ValueError("odds Over/Under de abertura inválidas")

                sports_features = build_sports_features(
                    history=team_history,
                    elo_ratings=elo_ratings,
                    league=str(league),
                    home_team=str(home_team),
                    away_team=str(away_team),
                    fixture_date=fixture_date,
                )
                market = build_market_features(odd_over, odd_under)
                market_probability = float(market["fair_probability_open"])
                model_row = {
                    "features_json": json.dumps(
                        {key: float(value) for key, value in sports_features.items()},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                }
                sports_probability = predict_sports_probability(
                    model,
                    model_features,
                    model_row,
                )

                fixture_date_text = fixture_date.strftime("%Y-%m-%d")
                paper = find_paper_trade(
                    connection,
                    fixture_date_text,
                    str(league),
                    str(home_team),
                    str(away_team),
                )
                paper_id = int(paper[0]) if paper is not None else None
                clv_probability = (
                    float(paper[1])
                    if paper is not None and paper[1] is not None
                    else None
                )

                decision = insert_decision(
                    connection,
                    row,
                    fixture_key,
                    fixture_date,
                    str(league),
                    str(home_team),
                    str(away_team),
                    sports_features,
                    sports_probability,
                    market_probability,
                    odd_over,
                    odd_under,
                    paper_id,
                    clv_probability,
                    started_at,
                )
                connection.commit()
                existing.add(fixture_key)
                inserted += 1
                if decision == "BET_SHADOW":
                    bets += 1
                else:
                    passes += 1
                if len(examples) < 12:
                    examples.append(
                        f"{fixture_date_text} | {home_team} vs {away_team} | {decision}"
                    )
            except Exception as error:
                skipped_data += 1
                if skipped_data <= 15:
                    print(
                        f"SKIP {row.get('date')} | {row.get('home_team')} vs "
                        f"{row.get('away_team')} | {type(error).__name__}: {error}"
                    )

        linked_after = attach_paper_links(connection, started_at)
        finished_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        connection.execute(
            f"""
            INSERT INTO {RUN_TABLE} (
                started_at, finished_at, champion_version, upcoming_seen,
                already_frozen, inserted, bets, passes, skipped_timing,
                skipped_data, resolved_updated, linked_paper, migrated_legacy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                started_at,
                finished_at,
                CHAMPION_VERSION,
                len(upcoming),
                already,
                inserted,
                bets,
                passes,
                skipped_timing,
                skipped_data,
                resolved,
                linked_before + linked_after,
                migrated,
            ),
        )
        connection.commit()

        print("\n-----------------------------------------------")
        print(f"Upcoming ativos vistos      : {len(upcoming)}")
        print(f"Já congelados               : {already}")
        print(f"Novas decisões V10          : {inserted}")
        print(f"BET_SHADOW novas            : {bets}")
        print(f"PASS novas                  : {passes}")
        print(f"Ignorados por timing        : {skipped_timing}")
        print(f"Sem dados/features          : {skipped_data}")
        print(f"Resultados atualizados      : {resolved}")
        print(f"Links paper associados      : {linked_before + linked_after}")
        print(f"Legado migrado              : {migrated}")
        print(f"Modelo Over gerado em       : {model_generated_at or 'desconhecido'}")

        if examples:
            print("\nExemplos:")
            for example in examples:
                print(f"- {example}")

        print("\nPerformance prospectiva acumulada:")
        print_performance(connection)
        print("\nNota: o V10 congela a decisão a partir de upcoming_matches antes do Paper Trade.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
