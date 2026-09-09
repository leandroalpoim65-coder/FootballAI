from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import libsql

from agent.performance_monitor import (
    MONITOR_VERSION,
    PerformanceMetrics,
    build_metrics,
    build_segments,
    filter_diagnostics,
    monitor_state,
)


PAPER_TABLE = "paper_trades"
DECISION_TABLE = "agent_decisions"
RUN_TABLE = "agent_performance_runs"
SEGMENT_TABLE = "agent_performance_segments"


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


def ensure_monitor_tables(
    connection,
) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            monitor_version TEXT NOT NULL,
            policy_versions_json TEXT NOT NULL,
            data_max_fixture_date TEXT,
            resolved_agent_sample INTEGER NOT NULL,
            original_signal_sample INTEGER NOT NULL,
            qualified_sample INTEGER NOT NULL,
            watch_sample INTEGER NOT NULL,
            reject_sample INTEGER NOT NULL,
            original_roi REAL,
            qualified_roi REAL,
            roi_lift REAL,
            winner_retention REAL,
            loser_rejection REAL,
            selection_rate REAL,
            monitor_state TEXT NOT NULL,
            notes_json TEXT NOT NULL
        )
        """
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SEGMENT_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            segment_type TEXT NOT NULL,
            segment_key TEXT NOT NULL,
            sample INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            profit_units REAL NOT NULL,
            roi REAL,
            hit_rate REAL,
            avg_open_odd REAL,
            avg_clv REAL,
            hq_clv_sample INTEGER NOT NULL,
            hq_clv_mean REAL,
            avg_expected_value REAL,
            avg_sports_edge REAL,
            sports_brier REAL,
            created_at TEXT NOT NULL,
            UNIQUE(
                run_id,
                segment_type,
                segment_key
            )
        )
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{SEGMENT_TABLE}_type_key
        ON {SEGMENT_TABLE}(
            segment_type,
            segment_key,
            run_id
        )
        """
    )

    connection.commit()


def load_resolved_agent_rows(
    connection,
) -> list[dict[str, Any]]:
    if not table_exists(
        connection,
        PAPER_TABLE,
    ):
        raise RuntimeError(
            f"A tabela {PAPER_TABLE} não existe."
        )

    if not table_exists(
        connection,
        DECISION_TABLE,
    ):
        raise RuntimeError(
            f"A tabela {DECISION_TABLE} não existe."
        )

    paper_columns = table_columns(
        connection,
        PAPER_TABLE,
    )

    decision_columns = table_columns(
        connection,
        DECISION_TABLE,
    )

    required_paper = {
        "id",
        "status",
        "result_over25",
        "odd_over25_open",
    }

    required_decision = {
        "paper_trade_id",
        "evaluated_at",
        "policy_version",
        "fixture_date",
        "league",
        "home_team",
        "away_team",
        "sports_probability",
        "clv_probability",
        "original_clv_signal",
        "expected_value",
        "sports_edge_market",
        "decision",
        "tier",
    }

    missing_paper = sorted(
        required_paper - paper_columns
    )

    missing_decision = sorted(
        required_decision - decision_columns
    )

    if missing_paper:
        raise RuntimeError(
            "paper_trades sem colunas necessárias: "
            + ", ".join(missing_paper)
        )

    if missing_decision:
        raise RuntimeError(
            "agent_decisions sem colunas necessárias: "
            + ", ".join(missing_decision)
        )

    def paper_expr(
        column: str,
    ) -> str:
        if column in paper_columns:
            return f"p.{column} AS {column}"

        return f"NULL AS {column}"

    return fetch_dicts(
        connection,
        f"""
        SELECT
            d.paper_trade_id,
            d.evaluated_at,
            d.policy_version,
            d.fixture_date,
            d.fixture_time,
            d.league,
            d.home_team,
            d.away_team,
            d.odd_over25_open,
            d.market_probability,
            d.sports_probability,
            d.clv_probability,
            d.original_clv_signal,
            d.expected_value,
            d.sports_edge_market,
            d.decision,
            d.tier,
            {paper_expr("planned_stake")},
            {paper_expr("profit")},
            {paper_expr("result_over25")},
            {paper_expr("clv")},
            {paper_expr("closing_source")},
            {paper_expr("odd_over25_close")},
            {paper_expr("updated_at")}
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
            d.league,
            d.paper_trade_id
        """,
    )


def insert_segment(
    connection,
    run_id: int,
    segment_type: str,
    segment_key: str,
    metrics: PerformanceMetrics,
    created_at: str,
) -> None:
    connection.execute(
        f"""
        INSERT OR REPLACE INTO {SEGMENT_TABLE} (
            run_id,
            segment_type,
            segment_key,
            sample,
            wins,
            losses,
            profit_units,
            roi,
            hit_rate,
            avg_open_odd,
            avg_clv,
            hq_clv_sample,
            hq_clv_mean,
            avg_expected_value,
            avg_sports_edge,
            sports_brier,
            created_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            run_id,
            segment_type,
            segment_key,
            metrics.sample,
            metrics.wins,
            metrics.losses,
            metrics.profit_units,
            metrics.roi,
            metrics.hit_rate,
            metrics.avg_open_odd,
            metrics.avg_clv,
            metrics.hq_clv_sample,
            metrics.hq_clv_mean,
            metrics.avg_expected_value,
            metrics.avg_sports_edge,
            metrics.sports_brier,
            created_at,
        ),
    )


def fmt_pct(
    value: Any,
) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_num(
    value: Any,
) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def print_strategy_line(
    label: str,
    metrics: PerformanceMetrics,
) -> None:
    print(
        f"{label:<24} "
        f"n={metrics.sample:<4} "
        f"ROI={fmt_pct(metrics.roi):>8} "
        f"HR={fmt_pct(metrics.hit_rate):>8} "
        f"CLV_HQ={fmt_pct(metrics.hq_clv_mean):>8} "
        f"Brier={fmt_num(metrics.sports_brier):>6}"
    )


def main() -> None:
    print()
    print(
        "=========================================="
    )
    print(
        " FOOTBALLAI — AGENT PERFORMANCE MONITOR V2"
    )
    print(
        "=========================================="
    )
    print()

    started_at = (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )

    connection = connect_turso()

    try:
        ensure_monitor_tables(
            connection
        )

        rows = load_resolved_agent_rows(
            connection
        )

        diagnostics = filter_diagnostics(
            rows
        )

        state = monitor_state(
            diagnostics
        )

        segments = build_segments(
            rows
        )

        strategy_metrics = {
            key: build_metrics(items)
            for (
                segment_type,
                key,
            ), items in segments.items()
            if segment_type == "strategy"
        }

        resolved_agent_sample = (
            build_metrics(rows).sample
        )

        original_sample = int(
            diagnostics[
                "original_sample"
            ] or 0
        )

        qualified_sample = int(
            diagnostics[
                "qualified_sample"
            ] or 0
        )

        watch_sample = (
            strategy_metrics
            .get(
                "WATCH",
                PerformanceMetrics(),
            )
            .sample
        )

        reject_sample = (
            strategy_metrics
            .get(
                "REJECT",
                PerformanceMetrics(),
            )
            .sample
        )

        policy_versions = sorted(
            {
                str(
                    row.get(
                        "policy_version"
                    )
                    or ""
                ).strip()
                for row in rows
                if str(
                    row.get(
                        "policy_version"
                    )
                    or ""
                ).strip()
            }
        )

        data_max_fixture_date = max(
            (
                str(
                    row.get(
                        "fixture_date"
                    )
                    or ""
                )
                for row in rows
                if str(
                    row.get(
                        "fixture_date"
                    )
                    or ""
                )
            ),
            default=None,
        )

        finished_at = (
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        )

        notes = {
            "profit_method": (
                "counterfactual flat 1-unit profit "
                "using frozen opening odd and final "
                "Over 2.5 result"
            ),
            "threshold_policy": (
                "monitor-only; does not change agent "
                "thresholds automatically"
            ),
            "comparison_focus": (
                "original CLV signal vs frozen agent "
                "QUALIFIED/WATCH/REJECT"
            ),
        }

        connection.execute(
            f"""
            INSERT INTO {RUN_TABLE} (
                started_at,
                finished_at,
                monitor_version,
                policy_versions_json,
                data_max_fixture_date,
                resolved_agent_sample,
                original_signal_sample,
                qualified_sample,
                watch_sample,
                reject_sample,
                original_roi,
                qualified_roi,
                roi_lift,
                winner_retention,
                loser_rejection,
                selection_rate,
                monitor_state,
                notes_json
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                started_at,
                finished_at,
                MONITOR_VERSION,
                json.dumps(
                    policy_versions,
                    ensure_ascii=False,
                ),
                data_max_fixture_date,
                resolved_agent_sample,
                original_sample,
                qualified_sample,
                watch_sample,
                reject_sample,
                diagnostics.get(
                    "original_roi"
                ),
                diagnostics.get(
                    "qualified_roi"
                ),
                diagnostics.get(
                    "roi_lift"
                ),
                diagnostics.get(
                    "winner_retention"
                ),
                diagnostics.get(
                    "loser_rejection"
                ),
                diagnostics.get(
                    "selection_rate"
                ),
                state,
                json.dumps(
                    notes,
                    ensure_ascii=False,
                ),
            ),
        )

        connection.commit()

        run_id_row = (
            connection.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()
        )

        run_id = int(
            run_id_row[0]
        )

        created_at = finished_at

        segment_metrics: dict[
            tuple[str, str],
            PerformanceMetrics,
        ] = {}

        for (
            segment_type,
            segment_key,
        ), items in sorted(
            segments.items()
        ):
            metrics = build_metrics(
                items
            )

            segment_metrics[
                (
                    segment_type,
                    segment_key,
                )
            ] = metrics

            insert_segment(
                connection,
                run_id,
                segment_type,
                segment_key,
                metrics,
                created_at,
            )

        connection.commit()

        print(
            f"Run monitor                : "
            f"{run_id}"
        )

        print(
            f"Decisões resolvidas        : "
            f"{resolved_agent_sample}"
        )

        print(
            f"Sinais originais resolvidos: "
            f"{original_sample}"
        )

        print(
            f"QUALIFIED resolvidos       : "
            f"{qualified_sample}"
        )

        print(
            f"Estado                     : "
            f"{state}"
        )

        print()
        print(
            "COMPARAÇÃO PRINCIPAL"
        )
        print(
            "------------------------------------------"
        )

        for key in (
            "ORIGINAL_SIGNAL",
            "QUALIFIED_SHADOW",
            "QUALIFIED_A",
            "QUALIFIED_B",
            "WATCH",
            "REJECT",
        ):
            print_strategy_line(
                key,
                strategy_metrics.get(
                    key,
                    PerformanceMetrics(),
                ),
            )

        print()
        print(
            "EFICIÊNCIA DO FILTRO"
        )
        print(
            "------------------------------------------"
        )

        print(
            "Selection rate             : "
            + fmt_pct(
                diagnostics.get(
                    "selection_rate"
                )
            )
        )

        print(
            "Winner retention           : "
            + fmt_pct(
                diagnostics.get(
                    "winner_retention"
                )
            )
        )

        print(
            "Loser rejection            : "
            + fmt_pct(
                diagnostics.get(
                    "loser_rejection"
                )
            )
        )

        print(
            "ROI lift QUALIFIED-original: "
            + fmt_pct(
                diagnostics.get(
                    "roi_lift"
                )
            )
        )

        print(
            "ROI sinais não qualificados: "
            + fmt_pct(
                diagnostics.get(
                    "not_qualified_roi"
                )
            )
        )

        league_rows = [
            (
                key,
                metrics,
            )
            for (
                segment_type,
                key,
            ), metrics in (
                segment_metrics.items()
            )
            if (
                segment_type
                == "league_qualified"
                and metrics.sample >= 5
                and metrics.roi is not None
            )
        ]

        if league_rows:
            print()
            print(
                "LIGAS QUALIFIED "
                "(mín. 5 resolvidos)"
            )
            print(
                "------------------------------------------"
            )

            for (
                league,
                metrics,
            ) in sorted(
                league_rows,
                key=lambda item: (
                    item[1].roi
                    if item[1].roi is not None
                    else -999.0
                ),
                reverse=True,
            ):
                print(
                    f"{league:<30} "
                    f"n={metrics.sample:<3} "
                    f"ROI={fmt_pct(metrics.roi):>8} "
                    f"HR={fmt_pct(metrics.hit_rate):>8}"
                )

        print()
        print(
            "Nota: o ROI do monitor é contrafactual "
            "a 1 unidade por jogo. O V2 observa; "
            "não altera thresholds nem executa apostas."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
