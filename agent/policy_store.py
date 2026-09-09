from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agent.decision_engine import (
    DEFAULT_POLICY,
    PolicyConfig,
)


POLICY_TABLE = "agent_policies"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def ensure_policy_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {POLICY_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            policy_version TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            min_clv_probability REAL NOT NULL,
            min_expected_value REAL NOT NULL,
            min_edge_market REAL NOT NULL,
            min_odd REAL NOT NULL,
            max_odd REAL NOT NULL,
            strong_clv_probability REAL NOT NULL,
            strong_expected_value REAL NOT NULL,
            strong_edge_market REAL NOT NULL,
            source TEXT NOT NULL,
            parent_version TEXT,
            evidence_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            activated_at TEXT,
            retired_at TEXT
        )
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{POLICY_TABLE}_status
        ON {POLICY_TABLE}(status, activated_at)
        """
    )

    connection.commit()

    existing = connection.execute(
        f"""
        SELECT policy_version
        FROM {POLICY_TABLE}
        WHERE status = 'ACTIVE'
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()

    if existing is None:
        now = utc_now()

        connection.execute(
            f"""
            INSERT OR IGNORE INTO {POLICY_TABLE} (
                policy_version,
                status,
                min_clv_probability,
                min_expected_value,
                min_edge_market,
                min_odd,
                max_odd,
                strong_clv_probability,
                strong_expected_value,
                strong_edge_market,
                source,
                parent_version,
                evidence_json,
                created_at,
                activated_at,
                retired_at
            )
            VALUES (
                ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?,
                'bootstrap-default', NULL, ?, ?, ?, NULL
            )
            """,
            (
                DEFAULT_POLICY.version,
                DEFAULT_POLICY.min_clv_probability,
                DEFAULT_POLICY.min_expected_value,
                DEFAULT_POLICY.min_edge_market,
                DEFAULT_POLICY.min_odd,
                DEFAULT_POLICY.max_odd,
                DEFAULT_POLICY.strong_clv_probability,
                DEFAULT_POLICY.strong_expected_value,
                DEFAULT_POLICY.strong_edge_market,
                json.dumps(
                    {
                        "reason": (
                            "Initial frozen policy before "
                            "autonomous learning has enough evidence."
                        )
                    },
                    ensure_ascii=False,
                ),
                now,
                now,
            ),
        )

        connection.execute(
            f"""
            UPDATE {POLICY_TABLE}
            SET
                status = 'ACTIVE',
                activated_at = ?,
                retired_at = NULL
            WHERE policy_version = ?
              AND NOT EXISTS (
                    SELECT 1
                    FROM {POLICY_TABLE}
                    WHERE status = 'ACTIVE'
                      AND policy_version <> ?
              )
            """,
            (
                now,
                DEFAULT_POLICY.version,
                DEFAULT_POLICY.version,
            ),
        )

        connection.commit()


def _policy_from_row(row: tuple[Any, ...]) -> PolicyConfig:
    return PolicyConfig(
        version=str(row[0]),
        min_clv_probability=float(row[1]),
        min_expected_value=float(row[2]),
        min_edge_market=float(row[3]),
        min_odd=float(row[4]),
        max_odd=float(row[5]),
        strong_clv_probability=float(row[6]),
        strong_expected_value=float(row[7]),
        strong_edge_market=float(row[8]),
    )


def load_active_policy(connection) -> PolicyConfig:
    ensure_policy_table(connection)

    row = connection.execute(
        f"""
        SELECT
            policy_version,
            min_clv_probability,
            min_expected_value,
            min_edge_market,
            min_odd,
            max_odd,
            strong_clv_probability,
            strong_expected_value,
            strong_edge_market
        FROM {POLICY_TABLE}
        WHERE status = 'ACTIVE'
        ORDER BY
            COALESCE(activated_at, created_at) DESC,
            id DESC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        return DEFAULT_POLICY

    return _policy_from_row(row)


def activate_policy(
    connection,
    policy: PolicyConfig,
    *,
    source: str,
    parent_version: str | None,
    evidence: dict[str, Any],
) -> None:
    now = utc_now()

    connection.execute(
        f"""
        UPDATE {POLICY_TABLE}
        SET
            status = 'RETIRED',
            retired_at = ?
        WHERE status = 'ACTIVE'
          AND policy_version <> ?
        """,
        (
            now,
            policy.version,
        ),
    )

    connection.execute(
        f"""
        INSERT INTO {POLICY_TABLE} (
            policy_version,
            status,
            min_clv_probability,
            min_expected_value,
            min_edge_market,
            min_odd,
            max_odd,
            strong_clv_probability,
            strong_expected_value,
            strong_edge_market,
            source,
            parent_version,
            evidence_json,
            created_at,
            activated_at,
            retired_at
        )
        VALUES (
            ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
        )
        ON CONFLICT(policy_version)
        DO UPDATE SET
            status = 'ACTIVE',
            min_clv_probability = excluded.min_clv_probability,
            min_expected_value = excluded.min_expected_value,
            min_edge_market = excluded.min_edge_market,
            min_odd = excluded.min_odd,
            max_odd = excluded.max_odd,
            strong_clv_probability = excluded.strong_clv_probability,
            strong_expected_value = excluded.strong_expected_value,
            strong_edge_market = excluded.strong_edge_market,
            source = excluded.source,
            parent_version = excluded.parent_version,
            evidence_json = excluded.evidence_json,
            activated_at = excluded.activated_at,
            retired_at = NULL
        """,
        (
            policy.version,
            policy.min_clv_probability,
            policy.min_expected_value,
            policy.min_edge_market,
            policy.min_odd,
            policy.max_odd,
            policy.strong_clv_probability,
            policy.strong_expected_value,
            policy.strong_edge_market,
            source,
            parent_version,
            json.dumps(
                evidence,
                ensure_ascii=False,
            ),
            now,
            now,
        ),
    )

    connection.commit()
