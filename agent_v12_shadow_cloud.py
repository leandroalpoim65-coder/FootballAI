from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from agent.v12_calibrated import build_v12_feature_row, evaluate_v12, predict_v12_probability, valid_odd
from agent_v10_universe_cloud import canonical_fixture_key, is_safely_pre_kickoff, load_upcoming_universe
from agent_v12_train_cloud import META_PATH, MODEL_PATH, main as train_v12
from footballai_agent_cloud import connect_turso, fetch_dicts, table_exists
from ml.paper_trade import (
    build_sports_features, calculate_current_elo, create_team_history,
    normalize_league, normalize_team, parse_date,
)
from paper_trade_cloud import load_historical_matches, resolve_team_to_history

TABLE = "agent_v12_shadow_decisions"
RUN_TABLE = "agent_v12_shadow_runs"


def ensure_tables(connection) -> None:
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_key TEXT NOT NULL UNIQUE,
            evaluated_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            source TEXT,
            odd_over25_open REAL NOT NULL,
            odd_under25_open REAL NOT NULL,
            market_probability REAL NOT NULL,
            v12_probability REAL NOT NULL,
            v12_edge REAL NOT NULL,
            expected_value REAL NOT NULL,
            policy_min_edge REAL NOT NULL,
            policy_min_ev REAL NOT NULL,
            decision TEXT NOT NULL,
            confidence TEXT NOT NULL,
            features_json TEXT NOT NULL,
            result_over25 INTEGER,
            profit REAL,
            resolution_source TEXT,
            resolved_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    connection.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_fixture ON {TABLE}(fixture_date, decision)")
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            historical_matches INTEGER NOT NULL,
            sport_eligible_matches INTEGER NOT NULL,
            market_eligible_matches INTEGER NOT NULL,
            upcoming_seen INTEGER NOT NULL,
            already_frozen INTEGER NOT NULL,
            inserted INTEGER NOT NULL,
            bets INTEGER NOT NULL,
            passes INTEGER NOT NULL,
            skipped_timing INTEGER NOT NULL,
            skipped_data INTEGER NOT NULL,
            resolved_updated INTEGER NOT NULL,
            test_brier REAL,
            test_log_loss REAL,
            test_auc REAL,
            test_ece REAL,
            test_bets INTEGER,
            test_roi REAL
        )
    """)
    connection.commit()


def resolve_existing(connection, now_utc: str) -> int:
    if not table_exists(connection, "matches"):
        return 0
    rows = fetch_dicts(connection, f"""
        SELECT d.fixture_key, d.odd_over25_open, m.home_goals, m.away_goals
        FROM {TABLE} d
        JOIN matches m
          ON m.date = d.fixture_date
         AND m.league = d.league
         AND m.home_team = d.home_team
         AND m.away_team = d.away_team
        WHERE d.result_over25 IS NULL
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
        ORDER BY d.id, m.id
    """)
    updated = 0; seen: set[str] = set()
    for row in rows:
        key = str(row["fixture_key"])
        if key in seen:
            continue
        seen.add(key)
        result = int(float(row["home_goals"]) + float(row["away_goals"]) >= 3.0)
        odd = float(row["odd_over25_open"])
        profit = odd - 1.0 if result == 1 else -1.0
        cursor = connection.execute(f"""
            UPDATE {TABLE}
            SET result_over25=?, profit=?, resolution_source='matches',
                resolved_at=?, updated_at=?
            WHERE fixture_key=? AND result_over25 IS NULL
        """, (result, profit, now_utc, now_utc, key))
        updated += int(cursor.rowcount or 0)
    if updated:
        connection.commit()
    return updated


def print_performance(connection) -> None:
    rows = fetch_dicts(connection, f"""
        SELECT decision, result_over25, profit, odd_over25_open,
               v12_probability, market_probability, v12_edge
        FROM {TABLE}
        WHERE result_over25 IN (0,1)
        ORDER BY fixture_date, id
    """)
    bets = [row for row in rows if row["decision"] == "BET_SHADOW"]
    passes = [row for row in rows if row["decision"] == "PASS"]
    print(f"V12 resolvidas               : {len(rows)}")
    if bets:
        wins = sum(int(row["result_over25"]) for row in bets)
        profit = sum(float(row["profit"]) for row in bets)
        print(f"BET_SHADOW resolvidas        : {len(bets)}")
        print(f"Hit rate V12                 : {wins/len(bets):.2%}")
        print(f"Lucro V12                    : {profit:+.2f}u")
        print(f"ROI V12                      : {profit/len(bets):+.2%}")
    else:
        print("BET_SHADOW resolvidas        : 0")
    if passes:
        pass_profit = sum(float(row["profit"]) for row in passes)
        pass_wins = sum(int(row["result_over25"]) for row in passes)
        print(f"PASS resolvidos              : {len(passes)}")
        print(f"Over rate PASS               : {pass_wins/len(passes):.2%}")
        print(f"ROI contra factual PASS      : {pass_profit/len(passes):+.2%}")


def main() -> None:
    # Each cloud runner is ephemeral: train/evaluate first, then use the exact
    # generated challenger in the same run. The Champion V10 is untouched.
    train_v12()
    bundle = joblib.load(MODEL_PATH)
    metadata = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
    version = str(bundle["version"])
    policy = dict(bundle["policy"])

    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    print("=" * 72)
    print("FOOTBALLAI V12 CALIBRATED SHADOW")
    print("=" * 72)
    print(f"Modelo                       : {version}")
    print(f"Policy                       : {policy}")
    print("Champion V10 alterado?       : NÃO")

    connection = connect_turso()
    try:
        if not table_exists(connection, "upcoming_matches"):
            raise RuntimeError("upcoming_matches não existe na Turso")
        ensure_tables(connection)
        resolved = resolve_existing(connection, started_at)
        upcoming = load_upcoming_universe(connection)
        existing = {str(row[0]) for row in connection.execute(f"SELECT fixture_key FROM {TABLE}").fetchall()}

        matches = load_historical_matches(connection)
        team_history = create_team_history(matches)
        elo_ratings = calculate_current_elo(matches)

        from zoneinfo import ZoneInfo
        now_local = datetime.now(ZoneInfo("Europe/Lisbon"))
        inserted = bets = passes = skipped_timing = skipped_data = already = 0

        for row in upcoming:
            fixture_key = str(row.get("fixture_key") or canonical_fixture_key(row.get("date"), row.get("league"), row.get("home_team"), row.get("away_team")))
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
                home = normalize_team(league, row.get("home_team")); away = normalize_team(league, row.get("away_team"))
                if home is None or away is None:
                    raise ValueError("equipa inválida")
                home = resolve_team_to_history(team_history, str(league), str(home))
                away = resolve_team_to_history(team_history, str(league), str(away))
                odd_over = valid_odd(row.get("odd_over25_open")); odd_under = valid_odd(row.get("odd_under25_open"))
                if odd_over is None or odd_under is None:
                    raise ValueError("odds de abertura inválidas")

                sports = build_sports_features(
                    history=team_history, elo_ratings=elo_ratings, league=str(league),
                    home_team=str(home), away_team=str(away), fixture_date=fixture_date,
                )
                features = build_v12_feature_row(
                    sports_features=sports, history=team_history, league=str(league),
                    home_team=str(home), away_team=str(away), fixture_date=fixture_date,
                    odd_over25=odd_over, odd_under25=odd_under,
                )
                probability = predict_v12_probability(bundle, features)
                decision = evaluate_v12(probability, odd_over, odd_under, policy)
                now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                connection.execute(f"""
                    INSERT INTO {TABLE} (
                        fixture_key,evaluated_at,model_version,fixture_date,fixture_time,
                        league,home_team,away_team,source,odd_over25_open,odd_under25_open,
                        market_probability,v12_probability,v12_edge,expected_value,
                        policy_min_edge,policy_min_ev,decision,confidence,features_json,
                        created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    fixture_key,now,version,fixture_date.strftime("%Y-%m-%d"),row.get("time"),
                    str(league),str(home),str(away),str(row.get("source") or "upcoming_matches"),
                    odd_over,odd_under,decision.market_probability,decision.probability,
                    decision.edge,decision.expected_value,decision.min_edge,decision.min_ev,
                    decision.decision,decision.confidence,json.dumps(features,ensure_ascii=False,sort_keys=True),now,now,
                ))
                connection.commit()
                existing.add(fixture_key); inserted += 1
                if decision.decision == "BET_SHADOW": bets += 1
                else: passes += 1
            except Exception as error:
                skipped_data += 1
                if skipped_data <= 20:
                    print(f"SKIP {row.get('date')} | {row.get('home_team')} vs {row.get('away_team')} | {type(error).__name__}: {error}")

        finished_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        test = metadata.get("test_probability_metrics") or {}
        test_bets = metadata.get("test_betting_metrics") or {}
        connection.execute(f"""
            INSERT INTO {RUN_TABLE} (
                started_at,finished_at,model_version,historical_matches,sport_eligible_matches,
                market_eligible_matches,upcoming_seen,already_frozen,inserted,bets,passes,
                skipped_timing,skipped_data,resolved_updated,test_brier,test_log_loss,
                test_auc,test_ece,test_bets,test_roi
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            started_at,finished_at,version,int(metadata.get("raw_matches") or 0),
            int(metadata.get("sport_eligible_matches") or 0),int(metadata.get("market_eligible_matches") or 0),
            len(upcoming),already,inserted,bets,passes,skipped_timing,skipped_data,resolved,
            test.get("brier"),test.get("log_loss"),test.get("auc"),metadata.get("test_ece"),
            int(test_bets.get("bets") or 0),test_bets.get("roi"),
        ))
        connection.commit()

        print(f"Upcoming vistos               : {len(upcoming)}")
        print(f"Já congelados V12             : {already}")
        print(f"Novas decisões                : {inserted}")
        print(f"BET_SHADOW novas              : {bets}")
        print(f"PASS novas                    : {passes}")
        print(f"Skip timing                   : {skipped_timing}")
        print(f"Skip dados                    : {skipped_data}")
        print(f"Resultados atualizados        : {resolved}")
        print_performance(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
