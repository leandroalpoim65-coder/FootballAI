from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import libsql
import pandas as pd

from agent.memory_learning import MEMORY_VERSION, build_memory_summary

RUN_TABLE = "agent_memory_runs"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect():
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("Turso não configurada.")
    return libsql.connect(database=url, auth_token=token)


def table_exists(connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def frame(connection, sql: str) -> pd.DataFrame:
    cursor = connection.execute(sql)
    return pd.DataFrame(cursor.fetchall(), columns=[x[0] for x in (cursor.description or [])])


def ensure_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at TEXT NOT NULL,
            memory_version TEXT NOT NULL,
            historical_rows INTEGER NOT NULL,
            resolved_rows INTEGER NOT NULL,
            open_rows INTEGER NOT NULL,
            resolved_roi REAL,
            resolved_hit_rate REAL,
            open_clv_mean REAL,
            open_sports_mean REAL,
            open_ev_mean REAL,
            drift_score REAL,
            state TEXT NOT NULL,
            details_json TEXT NOT NULL
        )
        """
    )
    connection.commit()


def load_historical(connection) -> pd.DataFrame:
    frames = []
    if table_exists(connection, "matches"):
        frames.append(frame(connection, "SELECT date, league, home_team, away_team, home_goals, away_goals FROM matches"))
    if table_exists(connection, "uefa_matches"):
        frames.append(frame(connection, "SELECT date, league, home_team, away_team, home_goals, away_goals FROM uefa_matches"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_agent_memory(connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not table_exists(connection, "agent_decisions") or not table_exists(connection, "paper_trades"):
        return pd.DataFrame(), pd.DataFrame()
    base = """
        SELECT
            d.paper_trade_id,
            d.fixture_date,
            d.league,
            d.odd_over25_open,
            d.market_probability,
            d.sports_probability,
            d.clv_probability,
            d.expected_value,
            d.sports_edge_market,
            d.decision,
            d.tier,
            p.status,
            p.result_over25,
            p.clv
        FROM agent_decisions d
        JOIN paper_trades p ON p.id=d.paper_trade_id
    """
    resolved = frame(connection, base + " WHERE UPPER(COALESCE(p.status,'OPEN'))='RESOLVED' AND p.result_over25 IN (0,1)")
    open_rows = frame(connection, base + " WHERE UPPER(COALESCE(p.status,'OPEN'))='OPEN'")
    return resolved, open_rows


def main() -> None:
    print("\n==========================================")
    print(" FOOTBALLAI — AGENT LEARNING V5 MEMORY")
    print("==========================================\n")

    connection = connect()
    try:
        ensure_table(connection)
        historical = load_historical(connection)
        resolved, open_rows = load_agent_memory(connection)
        summary = build_memory_summary(historical, resolved, open_rows)

        details = {
            "principle": "historical=context; resolved=supervised evidence; open=unlabelled drift only",
            "no_pseudo_labels": True,
            "promotion_allowed_from_open_rows": False,
        }
        connection.execute(
            f"""
            INSERT INTO {RUN_TABLE} (
                evaluated_at,memory_version,historical_rows,resolved_rows,open_rows,
                resolved_roi,resolved_hit_rate,open_clv_mean,open_sports_mean,
                open_ev_mean,drift_score,state,details_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now_iso(), MEMORY_VERSION, summary.historical_rows, summary.resolved_rows,
                summary.open_rows, summary.resolved_roi, summary.resolved_hit_rate,
                summary.open_clv_mean, summary.open_sports_mean, summary.open_ev_mean,
                summary.drift_score, summary.state, json.dumps(details, ensure_ascii=False),
            ),
        )
        connection.commit()

        print(f"Histórico contexto         : {summary.historical_rows}")
        print(f"Shadow resolvido           : {summary.resolved_rows}")
        print(f"Live OPEN sem label        : {summary.open_rows}")
        print(f"ROI resolvido              : {summary.resolved_roi}")
        print(f"Drift score                : {summary.drift_score}")
        print(f"Estado memória             : {summary.state}")
        print("OPEN é usado apenas para drift/padrões, nunca como resultado inventado.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
