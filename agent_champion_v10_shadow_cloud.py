from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agent.champion_v10 import CHAMPION_VERSION, evaluate_champion
from agent.multi_analyst import score_pre_match
from footballai_agent_cloud import (
    connect_turso,
    fetch_dicts,
    load_sports_model,
    predict_sports_probability,
)


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
PAPER_TABLE = "paper_trades"
SHADOW_TABLE = "agent_v10_shadow_decisions"


def ensure_shadow_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SHADOW_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_trade_id INTEGER NOT NULL UNIQUE,
            evaluated_at TEXT NOT NULL,
            champion_version TEXT NOT NULL,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            odd_over25_open REAL NOT NULL,
            market_probability REAL NOT NULL,
            sports_probability REAL NOT NULL,
            expected_value REAL NOT NULL,
            sports_edge_market REAL NOT NULL,
            clv_probability REAL,
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
            resolved_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{SHADOW_TABLE}_fixture
        ON {SHADOW_TABLE}(fixture_date, decision)
        """
    )
    connection.commit()


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
    fixture_date = str(row.get("fixture_date") or "").strip()
    if not fixture_date:
        return False

    today = now_local.date().isoformat()
    if fixture_date > today:
        return True
    if fixture_date < today:
        return False

    kickoff = _parse_same_day_kickoff(row.get("fixture_time"), now_local)
    if kickoff is None:
        return False

    return now_local <= kickoff - timedelta(minutes=5)


def load_open_candidates(connection) -> list[dict]:
    today = datetime.now(APP_TIMEZONE).date().isoformat()
    return fetch_dicts(
        connection,
        f"""
        SELECT
            p.id,
            p.fixture_date,
            p.fixture_time,
            p.league,
            p.home_team,
            p.away_team,
            p.odd_over25_open,
            p.fair_probability_open,
            p.clv_probability,
            p.features_json
        FROM {PAPER_TABLE} p
        LEFT JOIN {SHADOW_TABLE} d
          ON d.paper_trade_id = p.id
        WHERE d.paper_trade_id IS NULL
          AND UPPER(COALESCE(p.status, 'OPEN')) = 'OPEN'
          AND date(p.fixture_date) >= date(?)
          AND p.odd_over25_open IS NOT NULL
          AND p.fair_probability_open IS NOT NULL
          AND p.features_json IS NOT NULL
        ORDER BY p.fixture_date, p.fixture_time, p.league, p.id
        """,
        (today,),
    )


def resolve_existing_decisions(connection, resolved_at: str) -> int:
    rows = fetch_dicts(
        connection,
        f"""
        SELECT
            d.paper_trade_id,
            d.odd_over25_open,
            p.result_over25
        FROM {SHADOW_TABLE} d
        JOIN {PAPER_TABLE} p
          ON p.id = d.paper_trade_id
        WHERE d.result_over25 IS NULL
          AND UPPER(COALESCE(p.status, 'OPEN')) = 'RESOLVED'
          AND p.result_over25 IN (0, 1)
        ORDER BY d.paper_trade_id
        """,
    )

    for row in rows:
        result = int(row["result_over25"])
        odd = float(row["odd_over25_open"])
        profit = odd - 1.0 if result == 1 else -1.0
        connection.execute(
            f"""
            UPDATE {SHADOW_TABLE}
            SET result_over25 = ?,
                profit = ?,
                resolved_at = ?,
                updated_at = ?
            WHERE paper_trade_id = ?
              AND result_over25 IS NULL
            """,
            (
                result,
                profit,
                resolved_at,
                resolved_at,
                int(row["paper_trade_id"]),
            ),
        )

    if rows:
        connection.commit()
    return len(rows)


def insert_shadow_decision(
    connection,
    row: dict,
    sports_probability: float,
    evaluated_at: str,
) -> str:
    odd = float(row["odd_over25_open"])
    market_probability = float(row["fair_probability_open"])
    expected_value = sports_probability * odd - 1.0
    sports_edge_market = sports_probability - market_probability

    scores = score_pre_match(
        row["features_json"],
        sports_probability,
        expected_value,
        sports_edge_market,
    )
    champion = evaluate_champion(scores, expected_value)

    connection.execute(
        f"""
        INSERT OR IGNORE INTO {SHADOW_TABLE} (
            paper_trade_id,
            evaluated_at,
            champion_version,
            fixture_date,
            fixture_time,
            league,
            home_team,
            away_team,
            odd_over25_open,
            market_probability,
            sports_probability,
            expected_value,
            sports_edge_market,
            clv_probability,
            goal_score,
            attack_score,
            ml_score,
            value_score,
            stats_score,
            goal_vote,
            attack_vote,
            ml_vote,
            value_vote,
            vote_count,
            decision,
            created_at,
            updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            int(row["id"]),
            evaluated_at,
            CHAMPION_VERSION,
            str(row["fixture_date"]),
            row.get("fixture_time"),
            str(row["league"]),
            str(row["home_team"]),
            str(row["away_team"]),
            odd,
            market_probability,
            sports_probability,
            expected_value,
            sports_edge_market,
            (
                float(row["clv_probability"])
                if row.get("clv_probability") is not None
                else None
            ),
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
        FROM {SHADOW_TABLE}
        WHERE result_over25 IN (0, 1)
        ORDER BY fixture_date, id
        """,
    )
    bets = [r for r in rows if r["decision"] == "BET_SHADOW"]
    if not bets:
        print("BET V10 resolvidas          : 0")
        return

    wins = sum(int(r["result_over25"]) for r in bets)
    profit = sum(float(r["profit"]) for r in bets)
    roi = profit / len(bets)
    low_clv = [
        r for r in bets
        if r.get("clv_probability") is not None
        and float(r["clv_probability"]) < 0.55
    ]

    print(f"BET V10 resolvidas          : {len(bets)}")
    print(f"Hit rate V10               : {wins / len(bets):.2%}")
    print(f"Lucro V10                  : {profit:.2f}u")
    print(f"ROI V10                    : {roi:.2%}")
    print(f"BET V10 com CLV<55         : {len(low_clv)}")


def main() -> None:
    now_local = datetime.now(APP_TIMEZONE)
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    print()
    print("==============================================")
    print(" FOOTBALLAI — V10 CHAMPION PROSPECTIVE SHADOW")
    print("==============================================")
    print()
    print(f"Champion                   : {CHAMPION_VERSION}")
    print("Regra                      : 3/4 votos + EV >= 0")
    print("CLV usado como gate?       : NÃO")
    print("Margem mínima pré-jogo     : 5 minutos")

    model, sports_features, model_generated_at = load_sports_model()
    connection = connect_turso()

    try:
        ensure_shadow_table(connection)

        resolved = resolve_existing_decisions(connection, now_utc)
        candidates = load_open_candidates(connection)

        inserted = 0
        bets = 0
        passes = 0
        skipped_late = 0
        errors = 0
        examples: list[str] = []

        for row in candidates:
            if not is_safely_pre_kickoff(row, now_local):
                skipped_late += 1
                continue

            try:
                sports_probability = predict_sports_probability(
                    model,
                    sports_features,
                    row,
                )
                decision = insert_shadow_decision(
                    connection,
                    row,
                    sports_probability,
                    now_utc,
                )
                inserted += 1
                if decision == "BET_SHADOW":
                    bets += 1
                else:
                    passes += 1

                if len(examples) < 10:
                    examples.append(
                        f"{row['fixture_date']} | {row['home_team']} vs "
                        f"{row['away_team']} | {decision}"
                    )
            except Exception as error:
                errors += 1
                print(
                    f"ERRO paper_trade_id={row.get('id')}: "
                    f"{type(error).__name__}: {error}"
                )

        connection.commit()

        print()
        print("----------------------------------------------")
        print(f"Resultados atualizados      : {resolved}")
        print(f"Candidatos novos            : {len(candidates)}")
        print(f"Decisões V10 congeladas     : {inserted}")
        print(f"BET_SHADOW novas            : {bets}")
        print(f"PASS novas                  : {passes}")
        print(f"Ignorados por timing        : {skipped_late}")
        print(f"Erros                       : {errors}")
        print(f"Modelo Over gerado em       : {model_generated_at or 'desconhecido'}")

        if examples:
            print("\nExemplos:")
            for example in examples:
                print(f"- {example}")

        print("\nPerformance prospectiva acumulada:")
        print_performance(connection)
        print()
        print("Nota: decisões V10 são congeladas antes do jogo e nunca são reescritas.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
