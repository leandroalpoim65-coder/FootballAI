from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import libsql
import pandas as pd

from ml.paper_trade import (
    MARKET,
    build_market_features,
    build_sports_features,
    calculate_current_elo,
    calculate_season,
    create_team_history,
    normalize_league,
    normalize_team,
    parse_date,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "ml" / "modelo_clv_over25.pkl"

APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
UPCOMING_TABLE = "upcoming_matches"
PAPER_TABLE = "paper_trades"


# ============================================================
# TURSO
# ============================================================

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
) -> list[dict]:
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
    table_name: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()

    return row is not None


# ============================================================
# PAPER TRADES
# ============================================================

def ensure_paper_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PAPER_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_timestamp TEXT NOT NULL,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            season TEXT NOT NULL,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            market TEXT NOT NULL,
            odd_over25_open REAL NOT NULL,
            odd_under25_open REAL NOT NULL,
            fair_probability_open REAL NOT NULL,
            clv_probability REAL NOT NULL,
            training_base_rate REAL NOT NULL,
            signal_threshold REAL NOT NULL,
            edge_over_base_rate REAL NOT NULL,
            signal INTEGER NOT NULL,
            planned_stake REAL NOT NULL,
            status TEXT NOT NULL,
            odd_over25_close REAL,
            odd_under25_close REAL,
            home_goals INTEGER,
            away_goals INTEGER,
            result_over25 INTEGER,
            clv REAL,
            profit REAL,
            model_generated_at TEXT,
            features_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        ux_paper_trades_fixture_market
        ON {PAPER_TABLE} (
            fixture_date,
            league,
            home_team,
            away_team,
            market
        )
        """
    )

    connection.commit()


def existing_paper_keys(connection) -> set[tuple[str, str, str, str, str]]:
    rows = connection.execute(
        f"""
        SELECT
            fixture_date,
            league,
            home_team,
            away_team,
            market
        FROM {PAPER_TABLE}
        """
    ).fetchall()

    return {
        (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
        )
        for row in rows
    }


def insert_paper_trade(
    connection,
    record: dict,
) -> None:
    connection.execute(
        f"""
        INSERT INTO {PAPER_TABLE} (
            prediction_timestamp,
            fixture_date,
            fixture_time,
            season,
            league,
            home_team,
            away_team,
            market,
            odd_over25_open,
            odd_under25_open,
            fair_probability_open,
            clv_probability,
            training_base_rate,
            signal_threshold,
            edge_over_base_rate,
            signal,
            planned_stake,
            status,
            model_generated_at,
            features_json,
            created_at,
            updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            record["prediction_timestamp"],
            record["fixture_date"],
            record["fixture_time"],
            record["season"],
            record["league"],
            record["home_team"],
            record["away_team"],
            record["market"],
            record["odd_over25_open"],
            record["odd_under25_open"],
            record["fair_probability_open"],
            record["clv_probability"],
            record["training_base_rate"],
            record["signal_threshold"],
            record["edge_over_base_rate"],
            record["signal"],
            record["planned_stake"],
            record["status"],
            record["model_generated_at"],
            record["features_json"],
            record["created_at"],
            record["updated_at"],
        ),
    )


# ============================================================
# HISTÓRICO
# ============================================================

def load_historical_matches(connection) -> pd.DataFrame:
    rows = fetch_dicts(
        connection,
        """
        SELECT
            id,
            date,
            league,
            home_team,
            away_team,
            home_goals,
            away_goals
        FROM matches
        ORDER BY date, id
        """,
    )

    if not rows:
        raise RuntimeError(
            "A tabela matches está vazia."
        )

    matches = pd.DataFrame(rows)

    required = [
        "id",
        "date",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    missing = [
        column
        for column in required
        if column not in matches.columns
    ]

    if missing:
        raise RuntimeError(
            "Faltam colunas na tabela matches: "
            + ", ".join(missing)
        )

    matches["date"] = matches["date"].apply(
        parse_date
    )

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="coerce",
    )

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="coerce",
    )

    for column in (
        "league",
        "home_team",
        "away_team",
    ):
        matches[column] = (
            matches[column]
            .astype("string")
            .str.strip()
        )

    matches = matches.dropna(
        subset=required
    ).copy()

    matches = matches.sort_values(
        ["date", "id"]
    ).reset_index(drop=True)

    if matches.empty:
        raise RuntimeError(
            "Não existem jogos históricos válidos."
        )

    return matches


# ============================================================
# UPCOMING
# ============================================================

def load_upcoming(connection) -> list[dict]:
    today = datetime.now(
        APP_TIMEZONE
    ).date().isoformat()

    return fetch_dicts(
        connection,
        f"""
        SELECT
            fixture_key,
            date,
            time,
            league,
            home_team,
            away_team,
            odd_over25_open,
            odd_under25_open,
            odd_over25_current,
            odd_under25_current,
            first_observed_at,
            current_observed_at,
            source,
            active
        FROM {UPCOMING_TABLE}
        WHERE active = 1
          AND date >= ?
        ORDER BY
            date,
            time,
            league,
            home_team
        """,
        (today,),
    )


def valid_odd(value) -> float | None:
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None

    if odd <= 1.0:
        return None

    return odd


# ============================================================
# MODELO
# ============================================================

def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo CLV não encontrado: {MODEL_PATH}"
        )

    bundle = joblib.load(
        MODEL_PATH
    )

    required = [
        "model",
        "base_features",
        "market_features",
        "model_features",
        "training_base_rate",
        "signal_threshold",
        "min_odd",
        "max_odd",
        "generated_at",
    ]

    missing = [
        key
        for key in required
        if key not in bundle
    ]

    if missing:
        raise RuntimeError(
            "O bundle CLV está incompleto: "
            + ", ".join(missing)
        )

    return bundle


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — PAPER TRADE CLOUD")
    print("==========================================")
    print()

    bundle = load_model_bundle()

    model = bundle["model"]
    base_features = list(
        bundle["base_features"]
    )
    market_features = list(
        bundle["market_features"]
    )
    model_features = list(
        bundle["model_features"]
    )

    training_base_rate = float(
        bundle["training_base_rate"]
    )
    signal_threshold = float(
        bundle["signal_threshold"]
    )
    min_odd = float(
        bundle["min_odd"]
    )
    max_odd = float(
        bundle["max_odd"]
    )
    model_generated_at = str(
        bundle["generated_at"]
    )

    prediction_timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    today = datetime.now(
        APP_TIMEZONE
    ).date()

    connection = connect_turso()

    try:
        if not table_exists(
            connection,
            UPCOMING_TABLE,
        ):
            raise RuntimeError(
                "A tabela upcoming_matches ainda não existe. "
                "Executa primeiro o FootballAI Upcoming Cloud."
            )

        if not table_exists(
            connection,
            "matches",
        ):
            raise RuntimeError(
                "A tabela matches não existe na Turso."
            )

        ensure_paper_table(
            connection
        )

        upcoming = load_upcoming(
            connection
        )

        print(
            f"Jogos upcoming ativos     : {len(upcoming)}"
        )

        if not upcoming:
            print(
                "Nenhum jogo futuro ativo para analisar."
            )
            return

        matches = load_historical_matches(
            connection
        )

        team_history = create_team_history(
            matches
        )

        elo_ratings = calculate_current_elo(
            matches
        )

        latest_history_date = pd.Timestamp(
            matches["date"].max()
        )

        print(
            f"Jogos históricos          : {len(matches)}"
        )
        print(
            f"Último jogo histórico     : "
            f"{latest_history_date.date()}"
        )

        existing_keys = existing_paper_keys(
            connection
        )

        inserted = 0
        already_exists = 0
        skipped = 0
        signals = 0

        skipped_examples: list[str] = []
        signal_examples: list[str] = []

        for row in upcoming:
            try:
                fixture_date = parse_date(
                    row.get("date")
                )

                if fixture_date is None:
                    raise ValueError(
                        "data inválida"
                    )

                if fixture_date.date() < today:
                    raise ValueError(
                        "jogo no passado"
                    )

                if (
                    fixture_date.normalize()
                    <= latest_history_date.normalize()
                ):
                    raise ValueError(
                        "data não posterior ao último "
                        "jogo histórico"
                    )

                league = normalize_league(
                    row.get("league")
                )

                if league is None:
                    raise ValueError(
                        "liga inválida"
                    )

                home_team = normalize_team(
                    league,
                    row.get("home_team"),
                )

                away_team = normalize_team(
                    league,
                    row.get("away_team"),
                )

                if home_team is None:
                    raise ValueError(
                        "equipa da casa inválida"
                    )

                if away_team is None:
                    raise ValueError(
                        "equipa visitante inválida"
                    )

                odd_over_open = valid_odd(
                    row.get("odd_over25_open")
                )

                odd_under_open = valid_odd(
                    row.get("odd_under25_open")
                )

                if odd_over_open is None:
                    raise ValueError(
                        "odd Over 2.5 OPEN inválida"
                    )

                if odd_under_open is None:
                    raise ValueError(
                        "odd Under 2.5 OPEN inválida"
                    )

                fixture_date_text = (
                    fixture_date.strftime("%Y-%m-%d")
                )

                paper_key = (
                    fixture_date_text,
                    str(league),
                    str(home_team),
                    str(away_team),
                    MARKET,
                )

                # Regra principal: a primeira previsão fica congelada.
                if paper_key in existing_keys:
                    already_exists += 1
                    continue

                sports_features = build_sports_features(
                    history=team_history,
                    elo_ratings=elo_ratings,
                    league=str(league),
                    home_team=str(home_team),
                    away_team=str(away_team),
                    fixture_date=fixture_date,
                )

                missing_sports = [
                    feature
                    for feature in base_features
                    if feature not in sports_features
                ]

                if missing_sports:
                    raise ValueError(
                        "features desportivas em falta: "
                        + ", ".join(missing_sports)
                    )

                market_values = build_market_features(
                    odd_over_open,
                    odd_under_open,
                )

                missing_market = [
                    feature
                    for feature in market_features
                    if feature not in market_values
                ]

                if missing_market:
                    raise ValueError(
                        "features de mercado em falta: "
                        + ", ".join(missing_market)
                    )

                prediction_values = {
                    **sports_features,
                    **market_values,
                    "league": str(league),
                }

                frame = pd.DataFrame(
                    [prediction_values]
                )

                missing_model = [
                    feature
                    for feature in model_features
                    if feature not in frame.columns
                ]

                if missing_model:
                    raise ValueError(
                        "features do modelo em falta: "
                        + ", ".join(missing_model)
                    )

                probabilities = model.predict_proba(
                    frame[model_features]
                )

                classes = list(
                    model.classes_
                )

                if 1 not in classes:
                    raise ValueError(
                        "o modelo não contém a classe 1"
                    )

                positive_index = classes.index(1)

                clv_probability = float(
                    probabilities[0, positive_index]
                )

                edge_over_base_rate = (
                    clv_probability
                    - training_base_rate
                )

                valid_signal_odd = (
                    min_odd
                    <= odd_over_open
                    <= max_odd
                )

                signal = int(
                    valid_signal_odd
                    and clv_probability >= signal_threshold
                )

                stored_features: dict[str, object] = {}

                for feature in model_features:
                    value = prediction_values[feature]

                    if feature == "league":
                        stored_features[feature] = str(value)
                    else:
                        stored_features[feature] = float(value)

                record = {
                    "prediction_timestamp":
                        prediction_timestamp,
                    "fixture_date":
                        fixture_date_text,
                    "fixture_time":
                        row.get("time"),
                    "season":
                        calculate_season(fixture_date),
                    "league":
                        str(league),
                    "home_team":
                        str(home_team),
                    "away_team":
                        str(away_team),
                    "market":
                        MARKET,
                    "odd_over25_open":
                        odd_over_open,
                    "odd_under25_open":
                        odd_under_open,
                    "fair_probability_open":
                        float(
                            market_values[
                                "fair_probability_open"
                            ]
                        ),
                    "clv_probability":
                        clv_probability,
                    "training_base_rate":
                        training_base_rate,
                    "signal_threshold":
                        signal_threshold,
                    "edge_over_base_rate":
                        edge_over_base_rate,
                    "signal":
                        signal,
                    "planned_stake":
                        1.0 if signal else 0.0,
                    "status":
                        "OPEN",
                    "model_generated_at":
                        model_generated_at,
                    "features_json":
                        json.dumps(
                            stored_features,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    "created_at":
                        prediction_timestamp,
                    "updated_at":
                        prediction_timestamp,
                }

                insert_paper_trade(
                    connection,
                    record,
                )

                existing_keys.add(
                    paper_key
                )

                inserted += 1
                signals += signal

                if signal:
                    signal_examples.append(
                        f"{fixture_date_text} | "
                        f"{home_team} vs {away_team} | "
                        f"odd {odd_over_open:.3f} | "
                        f"CLV prob {clv_probability:.2%}"
                    )

            except Exception as error:
                skipped += 1

                if len(skipped_examples) < 20:
                    skipped_examples.append(
                        f"{row.get('date')} | "
                        f"{row.get('home_team')} vs "
                        f"{row.get('away_team')} | "
                        f"{error}"
                    )

        connection.commit()

        total_paper = connection.execute(
            f"SELECT COUNT(*) FROM {PAPER_TABLE}"
        ).fetchone()[0]

        total_open = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {PAPER_TABLE}
            WHERE COALESCE(status, 'OPEN') = 'OPEN'
            """
        ).fetchone()[0]

        total_signals = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {PAPER_TABLE}
            WHERE signal = 1
            """
        ).fetchone()[0]

        print()
        print("==========================================")
        print(" RESUMO")
        print("==========================================")
        print(f"Novos registos           : {inserted}")
        print(f"Já existentes            : {already_exists}")
        print(f"Linhas ignoradas         : {skipped}")
        print(f"Novos sinais CLV         : {signals}")
        print(f"Paper Trades total       : {total_paper}")
        print(f"Paper Trades OPEN        : {total_open}")
        print(f"Sinais total             : {total_signals}")
        print(
            f"Taxa-base treino         : "
            f"{training_base_rate:.2%}"
        )
        print(
            f"Limiar congelado         : "
            f"{signal_threshold:.2%}"
        )

        if signal_examples:
            print()
            print("Novos sinais:")

            for item in signal_examples:
                print(f"  ✅ {item}")

        if skipped_examples:
            print()
            print("Exemplos ignorados:")

            for item in skipped_examples:
                print(f"  - {item}")

        print()
        print(
            "✅ A primeira previsão de cada jogo fica congelada."
        )
        print(
            "✅ O script cloud não altera previsões existentes."
        )
        print(
            "⚠ CLV probability continua a prever CLV positivo, "
            "não o resultado Over 2.5."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
