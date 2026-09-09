from __future__ import annotations

from datetime import datetime, timezone

from agent.broad_behavior_learning import LEARNING_VERSION, build_learning_rows, segment_rows
from footballai_agent_cloud import connect_turso, fetch_dicts, load_sports_model, predict_sports_probability, table_exists

RUN_TABLE = "agent_behavior_runs"
SEGMENT_TABLE = "agent_behavior_segments"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_tables(c) -> None:
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            learning_version TEXT NOT NULL,
            eligible INTEGER NOT NULL,
            sports_model_generated_at TEXT
        )
    """)
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {SEGMENT_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            learning_version TEXT NOT NULL,
            segment_type TEXT NOT NULL,
            segment_value TEXT NOT NULL,
            sample INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            profit REAL NOT NULL,
            roi REAL,
            hit_rate REAL,
            avg_odd REAL,
            avg_clv_probability REAL,
            avg_sports_probability REAL,
            avg_expected_value REAL,
            avg_sports_edge_market REAL
        )
    """)
    c.commit()


def load_resolved(c, model_generated_at: str | None):
    model_date = (model_generated_at or "")[:10]
    return fetch_dicts(c, """
        SELECT id, prediction_timestamp, fixture_date, fixture_time, league,
               home_team, away_team, odd_over25_open, fair_probability_open,
               clv_probability, signal_threshold, signal, features_json,
               result_over25
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


def main() -> None:
    print("\n==========================================")
    print(" FOOTBALLAI — BROAD BEHAVIOR LEARNING V7")
    print("==========================================\n")
    c = connect_turso()
    try:
        if not table_exists(c, "paper_trades"):
            raise RuntimeError("paper_trades não existe")
        ensure_tables(c)
        model, features, generated_at = load_sports_model()
        rows = load_resolved(c, generated_at)

        def sports(pre):
            return predict_sports_probability(model, features, pre)

        learning_rows = build_learning_rows(rows, sports)
        segments = segment_rows(learning_rows)
        stamp = now()
        cur = c.execute(
            f"INSERT INTO {RUN_TABLE} (run_at,learning_version,eligible,sports_model_generated_at) VALUES (?,?,?,?)",
            (stamp, LEARNING_VERSION, len(learning_rows), generated_at),
        )
        run_id = int(cur.lastrowid)

        for s in segments:
            c.execute(f"""
                INSERT INTO {SEGMENT_TABLE} (
                    run_id, learning_version, segment_type, segment_value,
                    sample, wins, profit, roi, hit_rate, avg_odd,
                    avg_clv_probability, avg_sports_probability,
                    avg_expected_value, avg_sports_edge_market
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id, LEARNING_VERSION, s["segment_type"], s["segment_value"],
                s["sample"], s["wins"], s["profit"], s["roi"], s["hit_rate"],
                s["avg_odd"], s["avg_clv_probability"], s["avg_sports_probability"],
                s["avg_expected_value"], s["avg_sports_edge_market"],
            ))
        c.commit()

        print(f"Jogos analisados            : {len(learning_rows)}")
        print("CLV <55% bloqueia aposta    : SIM")
        print("CLV <55% bloqueia aprender  : NÃO")
        print(f"Segmentos guardados         : {len(segments)}")
        print("\nTodos os jogos elegíveis entram na memória comportamental após o resultado ser revelado.")
    finally:
        c.close()


if __name__ == "__main__":
    main()
