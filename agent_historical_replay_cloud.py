from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.historical_replay import replay_rows
from agent.policy_store import ensure_policy_table, load_active_policy
from footballai_agent_cloud import (
    connect_turso, fetch_dicts, load_sports_model, predict_sports_probability,
    table_exists,
)

REPLAY_TABLE = "agent_historical_replay"
RUN_TABLE = "agent_historical_replay_runs"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_tables(c):
    c.execute(f"""CREATE TABLE IF NOT EXISTS {REPLAY_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        replay_version TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        paper_trade_id INTEGER NOT NULL,
        fixture_date TEXT NOT NULL,
        league TEXT NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        decision TEXT NOT NULL,
        tier TEXT NOT NULL,
        sports_probability REAL NOT NULL,
        clv_probability REAL NOT NULL,
        odd_over25_open REAL NOT NULL,
        expected_value REAL NOT NULL,
        sports_edge_market REAL NOT NULL,
        result_over25 INTEGER NOT NULL,
        counterfactual_profit REAL NOT NULL,
        reasons_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(replay_version, policy_version, paper_trade_id)
    )""")
    c.execute(f"""CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT NOT NULL, replay_version TEXT NOT NULL, policy_version TEXT NOT NULL,
        eligible INTEGER NOT NULL, qualified INTEGER NOT NULL, watch INTEGER NOT NULL,
        rejected INTEGER NOT NULL, qualified_wins INTEGER NOT NULL,
        qualified_profit REAL NOT NULL, qualified_roi REAL,
        sports_model_generated_at TEXT
    )""")
    c.commit()


def load_resolved(c, model_generated_at: str | None):
    # Critical anti-leakage guard: never replay a fixture dated before the
    # production sports model itself existed. Frozen pre-match features are
    # required, and outcomes are fetched only for the reveal/evaluation stage.
    model_date = (model_generated_at or "")[:10]
    return fetch_dicts(c, """
        SELECT id, prediction_timestamp, fixture_date, fixture_time, league,
               home_team, away_team, odd_over25_open, fair_probability_open,
               clv_probability, signal_threshold, signal, features_json,
               result_over25, clv, closing_source
        FROM paper_trades
        WHERE UPPER(COALESCE(status,'OPEN'))='RESOLVED'
          AND result_over25 IN (0,1)
          AND odd_over25_open IS NOT NULL
          AND fair_probability_open IS NOT NULL
          AND clv_probability IS NOT NULL
          AND features_json IS NOT NULL
          AND date(fixture_date) >= date(?)
        ORDER BY fixture_date, fixture_time, id
    """, (model_date,))


def main():
    print("\n==========================================")
    print(" FOOTBALLAI — HISTORICAL BLIND REPLAY V6")
    print("==========================================\n")
    c = connect_turso()
    try:
        if not table_exists(c, "paper_trades"):
            raise RuntimeError("paper_trades não existe")
        ensure_tables(c)
        ensure_policy_table(c)
        policy = load_active_policy(c)
        model, features, generated_at = load_sports_model()
        rows = load_resolved(c, generated_at)
        replay_version = "blind-replay-v6-model-safe"

        def sports(pre):
            return predict_sports_probability(model, features, pre)

        decisions, summary = replay_rows(rows, sports, policy)
        stamp = now()
        for d in decisions:
            c.execute(f"""INSERT OR REPLACE INTO {REPLAY_TABLE} (
                replay_version, policy_version, paper_trade_id, fixture_date, league,
                home_team, away_team, decision, tier, sports_probability,
                clv_probability, odd_over25_open, expected_value, sports_edge_market,
                result_over25, counterfactual_profit, reasons_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                replay_version, policy.version, d["paper_trade_id"], d["fixture_date"],
                d["league"], str(d["home_team"]), str(d["away_team"]), d["decision"],
                d["tier"], d["sports_probability"], d["clv_probability"],
                d["odd_over25_open"], d["expected_value"], d["sports_edge_market"],
                d["result_over25"], d["counterfactual_profit"],
                json.dumps(d["reasons"], ensure_ascii=False), stamp,
            ))
        c.execute(f"""INSERT INTO {RUN_TABLE} (
            run_at,replay_version,policy_version,eligible,qualified,watch,rejected,
            qualified_wins,qualified_profit,qualified_roi,sports_model_generated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (
            stamp,replay_version,policy.version,summary.eligible,summary.qualified,
            summary.watch,summary.rejected,summary.qualified_wins,
            summary.qualified_profit,summary.qualified_roi,generated_at,
        ))
        c.commit()
        print(f"Modelo disponível desde    : {generated_at}")
        print(f"Jogos elegíveis            : {summary.eligible}")
        print(f"QUALIFIED_SHADOW           : {summary.qualified}")
        print(f"WATCH                      : {summary.watch}")
        print(f"REJECT                     : {summary.rejected}")
        if summary.qualified_roi is not None:
            print(f"ROI replay qualified       : {summary.qualified_roi:.2%}")
        print("\nResultados foram revelados apenas após cada decisão ser congelada.")
        print("Historical Replay continua separado do Shadow Prospective.")
    finally:
        c.close()


if __name__ == "__main__":
    main()
