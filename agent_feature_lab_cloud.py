from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import joblib
import libsql
import pandas as pd

from agent.feature_lab import candidate_library, evaluate_feature


TABLE = "agent_feature_experiments"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect():
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("Turso não configurada.")
    return libsql.connect(database=url, auth_token=token)


def ensure_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            status TEXT NOT NULL,
            train_rows INTEGER NOT NULL,
            validation_rows INTEGER NOT NULL,
            baseline_auc REAL,
            candidate_auc REAL,
            auc_lift REAL,
            baseline_brier REAL,
            candidate_brier REAL,
            brier_improvement REAL,
            reason TEXT NOT NULL,
            details_json TEXT NOT NULL
        )
        """
    )
    connection.commit()


def load_rows(connection) -> pd.DataFrame:
    cursor = connection.execute(
        """
        SELECT
            d.paper_trade_id,
            d.fixture_date,
            d.odd_over25_open,
            d.market_probability,
            d.sports_probability,
            d.clv_probability,
            d.expected_value,
            d.sports_edge_market,
            d.features_json,
            p.result_over25
        FROM agent_decisions d
        JOIN paper_trades p ON p.id=d.paper_trade_id
        WHERE UPPER(COALESCE(p.status, 'OPEN'))='RESOLVED'
          AND p.result_over25 IN (0,1)
        ORDER BY d.fixture_date, d.paper_trade_id
        """
    )
    names = [x[0] for x in cursor.description]
    rows = [dict(zip(names, row)) for row in cursor.fetchall()]
    expanded: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            frozen = json.loads(item.pop("features_json") or "{}")
        except Exception:
            frozen = {}
        if isinstance(frozen, dict):
            for key, value in frozen.items():
                item.setdefault(str(key), value)
        expanded.append(item)
    return pd.DataFrame(expanded)


def main() -> None:
    print("\n==========================================")
    print(" FOOTBALLAI — AUTONOMOUS FEATURE LAB V4")
    print("==========================================\n")

    connection = connect()
    try:
        ensure_table(connection)
        data = load_rows(connection)
        base_features = list(joblib.load("ml/features_over25.pkl"))
        print(f"Features champion atuais : {len(base_features)}")
        print(f"Decisões resolvidas      : {len(data)}")

        for candidate in candidate_library():
            result = evaluate_feature(data, base_features, candidate)
            connection.execute(
                f"""
                INSERT INTO {TABLE} (
                    evaluated_at, feature_name, status, train_rows,
                    validation_rows, baseline_auc, candidate_auc, auc_lift,
                    baseline_brier, candidate_brier, brier_improvement,
                    reason, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now(), result.feature, result.status, result.train_rows,
                    result.validation_rows, result.baseline_auc, result.candidate_auc,
                    result.auc_lift, result.baseline_brier, result.candidate_brier,
                    result.brier_improvement, result.reason,
                    json.dumps(asdict(result), ensure_ascii=False),
                ),
            )
            print(
                f"{result.feature:24s} | {result.status:15s} | "
                f"AUC lift={result.auc_lift if result.auc_lift is not None else 'n/a'} | "
                f"Brier imp={result.brier_improvement if result.brier_improvement is not None else 'n/a'}"
            )
        connection.commit()
        print("\nO laboratório apenas cria challengers. Não substitui o champion automaticamente nesta fase.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
