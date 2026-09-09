from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import libsql

from agent.decision_engine import PolicyConfig
from agent.policy_learning import (
    learn_policy,
    metrics_dict,
)
from agent.policy_store import (
    activate_policy,
    ensure_policy_table,
    load_active_policy,
)


DECISION_TABLE = "agent_decisions"
PAPER_TABLE = "paper_trades"
LEARNING_TABLE = "agent_learning_runs"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def connect_turso():
    url = os.getenv(
        "TURSO_DATABASE_URL",
        "",
    ).strip()

    token = os.getenv(
        "TURSO_AUTH_TOKEN",
        "",
    ).strip()

    if not url:
        raise RuntimeError(
            "TURSO_DATABASE_URL não configurada."
        )

    if not token:
        raise RuntimeError(
            "TURSO_AUTH_TOKEN não configurado."
        )

    return libsql.connect(
        database=url,
        auth_token=token,
    )


def table_exists(
    connection,
    table: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table'
          AND name=?
        LIMIT 1
        """,
        (table,),
    ).fetchone()

    return row is not None


def table_columns(
    connection,
    table: str,
) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def fetch_dicts(
    connection,
    sql: str,
    params: tuple = (),
) -> list[dict[str, Any]]:
    cursor = connection.execute(
        sql,
        params,
    )

    rows = cursor.fetchall()
    names = [
        item[0]
        for item in (cursor.description or [])
    ]

    return [
        dict(zip(names, row))
        for row in rows
    ]


def ensure_learning_table(
    connection,
) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LEARNING_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            active_policy_version TEXT NOT NULL,
            decision TEXT NOT NULL,
            candidate_policy_json TEXT,
            resolved_rows INTEGER NOT NULL,
            train_rows INTEGER NOT NULL,
            validation_rows INTEGER NOT NULL,
            split_date TEXT,
            active_train_json TEXT NOT NULL,
            candidate_train_json TEXT NOT NULL,
            active_validation_json TEXT NOT NULL,
            candidate_validation_json TEXT NOT NULL,
            winner_retention REAL,
            loser_rejection REAL,
            reasons_json TEXT NOT NULL,
            promoted_policy_version TEXT
        )
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{LEARNING_TABLE}_decision
        ON {LEARNING_TABLE}(
            decision,
            finished_at
        )
        """
    )

    connection.commit()


def load_learning_rows(
    connection,
) -> list[dict[str, Any]]:
    if not table_exists(
        connection,
        DECISION_TABLE,
    ):
        return []

    if not table_exists(
        connection,
        PAPER_TABLE,
    ):
        return []

    paper_columns = table_columns(
        connection,
        PAPER_TABLE,
    )

    optional = {
        "clv": (
            "p.clv AS clv"
            if "clv" in paper_columns
            else "NULL AS clv"
        ),
        "closing_source": (
            "p.closing_source AS closing_source"
            if "closing_source" in paper_columns
            else "NULL AS closing_source"
        ),
    }

    return fetch_dicts(
        connection,
        f"""
        SELECT
            d.paper_trade_id,
            d.fixture_date,
            d.policy_version,
            d.odd_over25_open,
            d.market_probability,
            d.sports_probability,
            d.clv_probability,
            d.original_clv_signal,
            d.expected_value,
            d.sports_edge_market,
            p.result_over25,
            {optional["clv"]},
            {optional["closing_source"]}
        FROM {DECISION_TABLE} d
        JOIN {PAPER_TABLE} p
          ON p.id = d.paper_trade_id
        WHERE
            UPPER(
                COALESCE(
                    p.status,
                    'OPEN'
                )
            ) = 'RESOLVED'
          AND p.result_over25 IN (0, 1)
        ORDER BY
            d.fixture_date,
            d.paper_trade_id
        """,
    )


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def candidate_with_version(
    candidate: PolicyConfig,
    version: str,
) -> PolicyConfig:
    return PolicyConfig(
        version=version,
        min_clv_probability=candidate.min_clv_probability,
        min_expected_value=candidate.min_expected_value,
        min_edge_market=candidate.min_edge_market,
        min_odd=candidate.min_odd,
        max_odd=candidate.max_odd,
        strong_clv_probability=candidate.strong_clv_probability,
        strong_expected_value=candidate.strong_expected_value,
        strong_edge_market=candidate.strong_edge_market,
    )


def main() -> None:
    print()
    print(
        "=========================================="
    )
    print(
        " FOOTBALLAI — AUTONOMOUS POLICY LEARNER V3"
    )
    print(
        "=========================================="
    )
    print()

    started_at = utc_now()
    connection = connect_turso()

    try:
        ensure_policy_table(
            connection
        )
        ensure_learning_table(
            connection
        )

        active = load_active_policy(
            connection
        )

        rows = load_learning_rows(
            connection
        )

        result = learn_policy(
            rows,
            active,
        )

        promoted_version = None
        promoted_policy = None

        if (
            result.decision == "PROMOTE"
            and result.candidate is not None
        ):
            stamp = (
                datetime.now(timezone.utc)
                .strftime("%Y%m%dT%H%M%SZ")
            )

            promoted_version = (
                f"agent-auto-{stamp}"
            )

            promoted_policy = (
                candidate_with_version(
                    result.candidate,
                    promoted_version,
                )
            )

            evidence = {
                "learner": (
                    "temporal-holdout-v3"
                ),
                "active_policy": (
                    active.version
                ),
                "candidate": asdict(
                    promoted_policy
                ),
                "train": {
                    "active": metrics_dict(
                        result.active_train
                    ),
                    "candidate": metrics_dict(
                        result.candidate_train
                    ),
                },
                "validation": {
                    "active": metrics_dict(
                        result.active_validation
                    ),
                    "candidate": metrics_dict(
                        result.candidate_validation
                    ),
                },
                "winner_retention": (
                    result.winner_retention
                ),
                "loser_rejection": (
                    result.loser_rejection
                ),
                "split_date": (
                    result.split_date
                ),
                "reasons": list(
                    result.reasons
                ),
            }

            activate_policy(
                connection,
                promoted_policy,
                source=(
                    "autonomous-temporal-holdout"
                ),
                parent_version=active.version,
                evidence=evidence,
            )

        finished_at = utc_now()

        candidate_json = (
            json.dumps(
                asdict(
                    promoted_policy
                    or result.candidate
                ),
                ensure_ascii=False,
            )
            if (
                promoted_policy is not None
                or result.candidate is not None
            )
            else None
        )

        connection.execute(
            f"""
            INSERT INTO {LEARNING_TABLE} (
                started_at,
                finished_at,
                active_policy_version,
                decision,
                candidate_policy_json,
                resolved_rows,
                train_rows,
                validation_rows,
                split_date,
                active_train_json,
                candidate_train_json,
                active_validation_json,
                candidate_validation_json,
                winner_retention,
                loser_rejection,
                reasons_json,
                promoted_policy_version
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                started_at,
                finished_at,
                active.version,
                result.decision,
                candidate_json,
                len(rows),
                result.train_rows,
                result.validation_rows,
                result.split_date,
                json.dumps(
                    metrics_dict(
                        result.active_train
                    ),
                    ensure_ascii=False,
                ),
                json.dumps(
                    metrics_dict(
                        result.candidate_train
                    ),
                    ensure_ascii=False,
                ),
                json.dumps(
                    metrics_dict(
                        result.active_validation
                    ),
                    ensure_ascii=False,
                ),
                json.dumps(
                    metrics_dict(
                        result.candidate_validation
                    ),
                    ensure_ascii=False,
                ),
                result.winner_retention,
                result.loser_rejection,
                json.dumps(
                    list(result.reasons),
                    ensure_ascii=False,
                ),
                promoted_version,
            ),
        )

        connection.commit()

        print(
            f"Política ativa             : "
            f"{active.version}"
        )
        print(
            f"Jogos resolvidos avaliados: "
            f"{len(rows)}"
        )
        print(
            f"Treino temporal            : "
            f"{result.train_rows}"
        )
        print(
            f"Validação temporal         : "
            f"{result.validation_rows}"
        )
        print(
            f"Decisão do learner         : "
            f"{result.decision}"
        )

        if result.candidate is not None:
            print()
            print(
                "Melhor candidato:"
            )
            print(
                f"  CLV mínimo              : "
                f"{result.candidate.min_clv_probability:.1%}"
            )
            print(
                f"  EV mínimo               : "
                f"{result.candidate.min_expected_value:.1%}"
            )
            print(
                f"  Edge mínimo             : "
                f"{result.candidate.min_edge_market:.1%}"
            )
            print(
                f"  ROI treino              : "
                f"{fmt_pct(result.candidate_train.roi)}"
            )
            print(
                f"  ROI validação           : "
                f"{fmt_pct(result.candidate_validation.roi)}"
            )
            print(
                f"  ROI LCB validação       : "
                f"{fmt_pct(result.candidate_validation.roi_lcb)}"
            )

        if promoted_version is not None:
            print()
            print(
                "✅ NOVA POLÍTICA PROMOVIDA AUTOMATICAMENTE"
            )
            print(
                f"Versão                    : "
                f"{promoted_version}"
            )
        else:
            print()
            print(
                "Nenhuma mudança de política foi ativada."
            )

        if result.reasons:
            print()
            print(
                "Razões:"
            )
            for reason in result.reasons:
                print(
                    f"  - {reason}"
                )

        print()
        print(
            "O learner pode alterar apenas a política "
            "de shadow validada. Não executa apostas reais."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
