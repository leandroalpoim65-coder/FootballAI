from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import libsql
import pandas as pd

from agent.decision_engine import (
    POLICY_VERSION,
    PerformanceStats,
    build_stats,
    evaluate_candidate,
)


ROOT_DIR = Path(__file__).resolve().parent
APP_TIMEZONE = ZoneInfo("Europe/Lisbon")

PAPER_TABLE = "paper_trades"
DECISION_TABLE = "agent_decisions"
RUN_TABLE = "agent_runs"

SPORT_MODEL_PATH = ROOT_DIR / "ml" / "modelo_over25.pkl"
SPORT_FEATURES_PATH = ROOT_DIR / "ml" / "features_over25.pkl"
SPORT_METADATA_PATH = ROOT_DIR / "ml" / "metadata_modelo_over25.json"


def connect_turso():
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()

    if not url:
        raise RuntimeError("TURSO_DATABASE_URL não configurada.")
    if not token:
        raise RuntimeError("TURSO_AUTH_TOKEN não configurado.")

    return libsql.connect(
        database=url,
        auth_token=token,
    )


def fetch_dicts(
    connection,
    sql: str,
    params: tuple = (),
) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, params)
    rows = cursor.fetchall()
    names = [item[0] for item in (cursor.description or [])]
    return [dict(zip(names, row)) for row in rows]


def table_exists(connection, table: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (table,),
    ).fetchone()
    return row is not None


def table_columns(connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def ensure_agent_tables(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DECISION_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_trade_id INTEGER NOT NULL UNIQUE,
            evaluated_at TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            odd_over25_open REAL NOT NULL,
            market_probability REAL NOT NULL,
            sports_probability REAL NOT NULL,
            clv_probability REAL NOT NULL,
            original_clv_signal INTEGER NOT NULL,
            expected_value REAL NOT NULL,
            sports_edge_market REAL NOT NULL,
            league_sample INTEGER NOT NULL,
            league_roi REAL,
            league_roi_shrunk REAL,
            league_hit_rate REAL,
            league_hit_rate_shrunk REAL,
            league_hq_clv_sample INTEGER NOT NULL,
            league_hq_clv_mean REAL,
            global_sample INTEGER NOT NULL,
            global_roi REAL,
            global_hit_rate REAL,
            global_hq_clv_sample INTEGER NOT NULL,
            global_hq_clv_mean REAL,
            decision TEXT NOT NULL,
            tier TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            sports_model_generated_at TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{DECISION_TABLE}_fixture
        ON {DECISION_TABLE}(fixture_date, league, decision)
        """
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            open_trades_seen INTEGER NOT NULL,
            inserted INTEGER NOT NULL,
            already_evaluated INTEGER NOT NULL,
            errors INTEGER NOT NULL,
            qualified INTEGER NOT NULL,
            watch INTEGER NOT NULL,
            rejected INTEGER NOT NULL,
            global_resolved_signal_sample INTEGER NOT NULL,
            global_roi REAL,
            global_hit_rate REAL,
            global_hq_clv_sample INTEGER NOT NULL,
            global_hq_clv_mean REAL,
            health_state TEXT NOT NULL,
            health_reasons_json TEXT NOT NULL
        )
        """
    )

    connection.commit()


def load_sports_model():
    if not SPORT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo Over 2.5 não encontrado: {SPORT_MODEL_PATH}"
        )
    if not SPORT_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features Over 2.5 não encontradas: {SPORT_FEATURES_PATH}"
        )

    model = joblib.load(SPORT_MODEL_PATH)
    features = list(joblib.load(SPORT_FEATURES_PATH))

    generated_at = None
    if SPORT_METADATA_PATH.exists():
        try:
            metadata = json.loads(
                SPORT_METADATA_PATH.read_text(encoding="utf-8")
            )
            generated_at = str(metadata.get("generated_at") or "") or None
        except (OSError, json.JSONDecodeError):
            generated_at = None

    return model, features, generated_at


def positive_probability(model, frame: pd.DataFrame) -> float:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)

    if 1 not in classes:
        raise RuntimeError(
            "O modelo Over 2.5 não contém a classe positiva 1."
        )

    positive_index = classes.index(1)
    return float(probabilities[0, positive_index])


def load_resolved_signal_rows(connection) -> list[dict[str, Any]]:
    columns = table_columns(connection, PAPER_TABLE)

    wanted = [
        "fixture_date",
        "league",
        "signal",
        "status",
        "planned_stake",
        "profit",
        "result_over25",
        "clv",
        "closing_source",
    ]
    selected = [column for column in wanted if column in columns]

    required = {
        "fixture_date",
        "league",
        "signal",
        "status",
        "profit",
        "result_over25",
    }
    if not required.issubset(columns):
        missing = sorted(required - columns)
        raise RuntimeError(
            "paper_trades não tem as colunas necessárias: "
            + ", ".join(missing)
        )

    sql = (
        f"SELECT {', '.join(selected)} "
        f"FROM {PAPER_TABLE} "
        "WHERE signal = 1 "
        "AND UPPER(COALESCE(status, 'OPEN')) = 'RESOLVED' "
        "AND result_over25 IN (0, 1)"
    )

    return fetch_dicts(connection, sql)


def build_performance_context(
    rows: list[dict[str, Any]],
) -> tuple[PerformanceStats, dict[str, PerformanceStats]]:
    global_stats = build_stats(rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        league = str(row.get("league") or "").strip()
        if not league:
            continue
        grouped.setdefault(league, []).append(row)

    league_stats = {
        league: build_stats(items)
        for league, items in grouped.items()
    }
    return global_stats, league_stats


def load_open_trades(connection) -> list[dict[str, Any]]:
    today = datetime.now(APP_TIMEZONE).date().isoformat()

    return fetch_dicts(
        connection,
        f"""
        SELECT
            id,
            prediction_timestamp,
            fixture_date,
            fixture_time,
            league,
            home_team,
            away_team,
            odd_over25_open,
            fair_probability_open,
            clv_probability,
            signal_threshold,
            signal,
            features_json
        FROM {PAPER_TABLE}
        WHERE UPPER(COALESCE(status, 'OPEN')) = 'OPEN'
          AND date(fixture_date) >= date(?)
        ORDER BY fixture_date, fixture_time, league, id
        """,
        (today,),
    )


def existing_decision_ids(connection) -> set[int]:
    rows = connection.execute(
        f"SELECT paper_trade_id FROM {DECISION_TABLE}"
    ).fetchall()
    return {int(row[0]) for row in rows}


def predict_sports_probability(
    model,
    sports_features: list[str],
    row: dict[str, Any],
) -> float:
    raw = row.get("features_json")
    if not raw:
        raise ValueError("features_json vazio")

    try:
        stored = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise ValueError("features_json inválido") from error

    missing = [
        feature
        for feature in sports_features
        if feature not in stored
    ]
    if missing:
        raise ValueError(
            "features desportivas em falta: "
            + ", ".join(missing[:10])
        )

    values: dict[str, float] = {}
    for feature in sports_features:
        try:
            values[feature] = float(stored[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"feature inválida: {feature}"
            ) from error

    frame = pd.DataFrame([values], columns=sports_features)
    return positive_probability(model, frame[sports_features])


def insert_decision(
    connection,
    row: dict[str, Any],
    sports_probability: float,
    global_stats: PerformanceStats,
    league_stats: PerformanceStats,
    result,
    sports_model_generated_at: str | None,
    evaluated_at: str,
) -> None:
    reasons_json = json.dumps(
        list(result.reasons),
        ensure_ascii=False,
    )

    connection.execute(
        f"""
        INSERT OR IGNORE INTO {DECISION_TABLE} (
            paper_trade_id,
            evaluated_at,
            policy_version,
            fixture_date,
            fixture_time,
            league,
            home_team,
            away_team,
            odd_over25_open,
            market_probability,
            sports_probability,
            clv_probability,
            original_clv_signal,
            expected_value,
            sports_edge_market,
            league_sample,
            league_roi,
            league_roi_shrunk,
            league_hit_rate,
            league_hit_rate_shrunk,
            league_hq_clv_sample,
            league_hq_clv_mean,
            global_sample,
            global_roi,
            global_hit_rate,
            global_hq_clv_sample,
            global_hq_clv_mean,
            decision,
            tier,
            reasons_json,
            sports_model_generated_at,
            created_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            int(row["id"]),
            evaluated_at,
            POLICY_VERSION,
            str(row["fixture_date"]),
            row.get("fixture_time"),
            str(row["league"]),
            str(row["home_team"]),
            str(row["away_team"]),
            float(row["odd_over25_open"]),
            float(row["fair_probability_open"]),
            sports_probability,
            float(row["clv_probability"]),
            int(row.get("signal") or 0),
            result.expected_value,
            result.sports_edge_market,
            league_stats.sample,
            league_stats.roi,
            result.league_roi_shrunk,
            league_stats.hit_rate,
            result.league_hit_rate_shrunk,
            league_stats.hq_clv_sample,
            league_stats.hq_clv_mean,
            global_stats.sample,
            global_stats.roi,
            global_stats.hit_rate,
            global_stats.hq_clv_sample,
            global_stats.hq_clv_mean,
            result.decision,
            result.tier,
            reasons_json,
            sports_model_generated_at,
            evaluated_at,
        ),
    )


def health_state(
    global_stats: PerformanceStats,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if global_stats.sample < 30:
        reasons.append(
            f"amostra shadow ainda curta ({global_stats.sample})"
        )
        return "COLD_START", reasons

    if (
        global_stats.sample >= 80
        and global_stats.roi is not None
        and global_stats.roi < -0.05
        and global_stats.hq_clv_sample >= 30
        and global_stats.hq_clv_mean is not None
        and global_stats.hq_clv_mean <= 0.0
    ):
        reasons.append(
            "ROI shadow abaixo de -5% e CLV de alta qualidade não positivo"
        )
        return "DEGRADED", reasons

    if (
        global_stats.sample >= 50
        and global_stats.roi is not None
        and global_stats.roi < 0.0
    ):
        reasons.append("ROI shadow global negativo")
        return "REVIEW", reasons

    reasons.append("sem degradação estatística forte detetada")
    return "HEALTHY", reasons


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — AUTONOMOUS AGENT V1")
    print("==========================================")
    print()

    started_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    model, sports_features, sports_model_generated_at = (
        load_sports_model()
    )

    connection = connect_turso()

    try:
        if not table_exists(connection, PAPER_TABLE):
            raise RuntimeError(
                "A tabela paper_trades ainda não existe."
            )

        ensure_agent_tables(connection)

        resolved_rows = load_resolved_signal_rows(connection)
        global_stats, per_league = build_performance_context(
            resolved_rows
        )

        open_trades = load_open_trades(connection)
        evaluated_ids = existing_decision_ids(connection)

        print(f"Trades OPEN vistos         : {len(open_trades)}")
        print(f"Sinais resolvidos shadow  : {global_stats.sample}")

        if global_stats.roi is not None:
            print(f"ROI shadow global          : {global_stats.roi:.2%}")
        if global_stats.hq_clv_mean is not None:
            print(
                f"CLV HQ médio               : "
                f"{global_stats.hq_clv_mean:.2%} "
                f"(n={global_stats.hq_clv_sample})"
            )

        inserted = 0
        already = 0
        errors = 0
        qualified = 0
        watch = 0
        rejected = 0

        examples: list[str] = []

        for row in open_trades:
            paper_trade_id = int(row["id"])

            if paper_trade_id in evaluated_ids:
                already += 1
                continue

            try:
                sports_probability = predict_sports_probability(
                    model,
                    sports_features,
                    row,
                )

                league = str(row.get("league") or "").strip()
                league_stats = per_league.get(
                    league,
                    PerformanceStats(),
                )

                result = evaluate_candidate(
                    row=row,
                    sports_probability=sports_probability,
                    global_stats=global_stats,
                    league_stats=league_stats,
                )

                evaluated_at = (
                    datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                )

                insert_decision(
                    connection=connection,
                    row=row,
                    sports_probability=sports_probability,
                    global_stats=global_stats,
                    league_stats=league_stats,
                    result=result,
                    sports_model_generated_at=sports_model_generated_at,
                    evaluated_at=evaluated_at,
                )

                evaluated_ids.add(paper_trade_id)
                inserted += 1

                if result.decision == "QUALIFIED_SHADOW":
                    qualified += 1
                elif result.decision == "WATCH":
                    watch += 1
                else:
                    rejected += 1

                if len(examples) < 12:
                    examples.append(
                        f"{row['fixture_date']} | "
                        f"{row['home_team']} vs {row['away_team']} | "
                        f"{result.decision} {result.tier} | "
                        f"sports={sports_probability:.1%} | "
                        f"clv={float(row['clv_probability']):.1%} | "
                        f"EV={result.expected_value:.1%}"
                    )

            except Exception as error:
                errors += 1
                print(
                    f"⚠ Trade {paper_trade_id}: {error}"
                )

        connection.commit()

        state, health_reasons = health_state(global_stats)

        finished_at = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )

        connection.execute(
            f"""
            INSERT INTO {RUN_TABLE} (
                started_at,
                finished_at,
                policy_version,
                open_trades_seen,
                inserted,
                already_evaluated,
                errors,
                qualified,
                watch,
                rejected,
                global_resolved_signal_sample,
                global_roi,
                global_hit_rate,
                global_hq_clv_sample,
                global_hq_clv_mean,
                health_state,
                health_reasons_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                started_at,
                finished_at,
                POLICY_VERSION,
                len(open_trades),
                inserted,
                already,
                errors,
                qualified,
                watch,
                rejected,
                global_stats.sample,
                global_stats.roi,
                global_stats.hit_rate,
                global_stats.hq_clv_sample,
                global_stats.hq_clv_mean,
                state,
                json.dumps(
                    health_reasons,
                    ensure_ascii=False,
                ),
            ),
        )
        connection.commit()

        print()
        print("------------------------------------------")
        print(f"Novas decisões             : {inserted}")
        print(f"Já avaliadas               : {already}")
        print(f"Erros                      : {errors}")
        print(f"QUALIFIED_SHADOW           : {qualified}")
        print(f"WATCH                      : {watch}")
        print(f"REJECT                     : {rejected}")
        print(f"Estado do agente           : {state}")

        if examples:
            print()
            print("Exemplos:")
            for example in examples:
                print(f"  {example}")

        print()
        print(
            "Nota: V1 é apenas shadow. "
            "Não executa apostas nem altera thresholds automaticamente."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
