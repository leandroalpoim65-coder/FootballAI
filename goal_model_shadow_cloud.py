from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

import paper_trade_uefa_cloud as uefa_cloud
from ml.goal_model import DEFAULT_CONFIG, expected_value, flatten_projection, project_match
from ml.paper_trade import (
    calculate_current_elo,
    calculate_season,
    create_team_history,
    normalize_league,
    normalize_team,
    parse_date,
)


core = uefa_cloud.core

GOAL_SHADOW_TABLE = "goal_model_shadow"
GOAL_MODEL_VERSION = "goal_model_v1_poisson"


def ensure_goal_shadow_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {GOAL_SHADOW_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_timestamp TEXT NOT NULL,
            fixture_key TEXT,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            season TEXT NOT NULL,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            model_version TEXT NOT NULL,
            model_config_json TEXT NOT NULL,
            league_home_avg REAL,
            league_away_avg REAL,

            lambda_home REAL NOT NULL,
            lambda_away REAL NOT NULL,
            expected_total_goals REAL NOT NULL,

            prob_home_win REAL NOT NULL,
            prob_draw REAL NOT NULL,
            prob_away_win REAL NOT NULL,
            prob_btts_yes REAL NOT NULL,
            prob_btts_no REAL NOT NULL,

            prob_over_15 REAL NOT NULL,
            prob_under_15 REAL NOT NULL,
            prob_over_25 REAL NOT NULL,
            prob_under_25 REAL NOT NULL,
            prob_over_35 REAL NOT NULL,
            prob_under_35 REAL NOT NULL,

            prob_home_over_05 REAL NOT NULL,
            prob_home_over_15 REAL NOT NULL,
            prob_home_over_25 REAL NOT NULL,
            prob_away_over_05 REAL NOT NULL,
            prob_away_over_15 REAL NOT NULL,
            prob_away_over_25 REAL NOT NULL,

            fair_odd_home REAL,
            fair_odd_draw REAL,
            fair_odd_away REAL,
            fair_odd_btts_yes REAL,
            fair_odd_btts_no REAL,
            fair_odd_over_15 REAL,
            fair_odd_under_15 REAL,
            fair_odd_over_25 REAL,
            fair_odd_under_25 REAL,
            fair_odd_over_35 REAL,
            fair_odd_under_35 REAL,

            fair_odd_asian_over_200 REAL,
            fair_odd_asian_under_200 REAL,
            fair_odd_asian_over_225 REAL,
            fair_odd_asian_under_225 REAL,
            fair_odd_asian_over_250 REAL,
            fair_odd_asian_under_250 REAL,
            fair_odd_asian_over_275 REAL,
            fair_odd_asian_under_275 REAL,
            fair_odd_asian_over_300 REAL,
            fair_odd_asian_under_300 REAL,

            odd_over25_open REAL,
            odd_under25_open REAL,
            ev_over25_open REAL,
            ev_under25_open REAL,

            top_scores_json TEXT NOT NULL,
            projection_json TEXT NOT NULL,

            status TEXT NOT NULL DEFAULT 'OPEN',
            home_goals INTEGER,
            away_goals INTEGER,
            total_goals INTEGER,
            result_home_win INTEGER,
            result_draw INTEGER,
            result_away_win INTEGER,
            result_btts_yes INTEGER,
            result_over_15 INTEGER,
            result_over_25 INTEGER,
            result_over_35 INTEGER,
            result_home_over_05 INTEGER,
            result_home_over_15 INTEGER,
            result_home_over_25 INTEGER,
            result_away_over_05 INTEGER,
            result_away_over_15 INTEGER,
            result_away_over_25 INTEGER,

            settled_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        ux_goal_model_shadow_fixture
        ON {GOAL_SHADOW_TABLE} (
            fixture_date,
            league,
            home_team,
            away_team
        )
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        ix_goal_model_shadow_status
        ON {GOAL_SHADOW_TABLE} (status)
        """
    )

    connection.commit()


def existing_shadow_keys(connection) -> set[tuple[str, str, str, str]]:
    rows = connection.execute(
        f"""
        SELECT
            fixture_date,
            league,
            home_team,
            away_team
        FROM {GOAL_SHADOW_TABLE}
        """
    ).fetchall()

    return {
        (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
        )
        for row in rows
    }


def build_league_goal_averages(
    matches: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    grouped = (
        matches.groupby("league", dropna=True)[["home_goals", "away_goals"]]
        .mean()
    )

    return {
        str(league): (
            float(row["home_goals"]),
            float(row["away_goals"]),
        )
        for league, row in grouped.iterrows()
    }


def settle_existing_shadow(
    connection,
    matches: pd.DataFrame,
) -> int:
    open_rows = core.fetch_dicts(
        connection,
        f"""
        SELECT
            id,
            fixture_date,
            league,
            home_team,
            away_team
        FROM {GOAL_SHADOW_TABLE}
        WHERE status = 'OPEN'
        """,
    )

    if not open_rows:
        return 0

    lookup: dict[tuple[str, str, str, str], tuple[int, int]] = {}

    for match in matches.itertuples(index=False):
        match_date = pd.Timestamp(match.date).strftime("%Y-%m-%d")
        key = (
            match_date,
            str(match.league),
            str(match.home_team),
            str(match.away_team),
        )
        lookup[key] = (
            int(match.home_goals),
            int(match.away_goals),
        )

    settled_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    settled = 0

    for row in open_rows:
        key = (
            str(row["fixture_date"]),
            str(row["league"]),
            str(row["home_team"]),
            str(row["away_team"]),
        )

        result = lookup.get(key)
        if result is None:
            continue

        home_goals, away_goals = result
        total_goals = home_goals + away_goals

        connection.execute(
            f"""
            UPDATE {GOAL_SHADOW_TABLE}
            SET
                status = 'SETTLED',
                home_goals = ?,
                away_goals = ?,
                total_goals = ?,
                result_home_win = ?,
                result_draw = ?,
                result_away_win = ?,
                result_btts_yes = ?,
                result_over_15 = ?,
                result_over_25 = ?,
                result_over_35 = ?,
                result_home_over_05 = ?,
                result_home_over_15 = ?,
                result_home_over_25 = ?,
                result_away_over_05 = ?,
                result_away_over_15 = ?,
                result_away_over_25 = ?,
                settled_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                home_goals,
                away_goals,
                total_goals,
                int(home_goals > away_goals),
                int(home_goals == away_goals),
                int(home_goals < away_goals),
                int(home_goals > 0 and away_goals > 0),
                int(total_goals > 1.5),
                int(total_goals > 2.5),
                int(total_goals > 3.5),
                int(home_goals > 0.5),
                int(home_goals > 1.5),
                int(home_goals > 2.5),
                int(away_goals > 0.5),
                int(away_goals > 1.5),
                int(away_goals > 2.5),
                settled_at,
                settled_at,
                int(row["id"]),
            ),
        )
        settled += 1

    connection.commit()
    return settled


def insert_shadow_projection(
    connection,
    *,
    row: dict,
    fixture_date: pd.Timestamp,
    league: str,
    home_team: str,
    away_team: str,
    projection: dict,
    prediction_timestamp: str,
    league_home_avg: float | None,
    league_away_avg: float | None,
) -> None:
    flat = flatten_projection(projection)

    odd_over25_open = core.valid_odd(
        row.get("odd_over25_open")
    )
    odd_under25_open = core.valid_odd(
        row.get("odd_under25_open")
    )

    prob_over25 = float(
        projection["totals"]["2.5"]["over"]
    )
    prob_under25 = float(
        projection["totals"]["2.5"]["under"]
    )

    ev_over25_open = (
        expected_value(prob_over25, odd_over25_open)
        if odd_over25_open is not None
        else None
    )
    ev_under25_open = (
        expected_value(prob_under25, odd_under25_open)
        if odd_under25_open is not None
        else None
    )

    fixture_date_text = fixture_date.strftime("%Y-%m-%d")
    match_result = projection["match_result"]
    btts = projection["btts"]
    totals = projection["totals"]
    asian = projection["asian_totals"]

    connection.execute(
        f"""
        INSERT INTO {GOAL_SHADOW_TABLE} (
            prediction_timestamp,
            fixture_key,
            fixture_date,
            fixture_time,
            season,
            league,
            home_team,
            away_team,
            model_version,
            model_config_json,
            league_home_avg,
            league_away_avg,

            lambda_home,
            lambda_away,
            expected_total_goals,

            prob_home_win,
            prob_draw,
            prob_away_win,
            prob_btts_yes,
            prob_btts_no,

            prob_over_15,
            prob_under_15,
            prob_over_25,
            prob_under_25,
            prob_over_35,
            prob_under_35,

            prob_home_over_05,
            prob_home_over_15,
            prob_home_over_25,
            prob_away_over_05,
            prob_away_over_15,
            prob_away_over_25,

            fair_odd_home,
            fair_odd_draw,
            fair_odd_away,
            fair_odd_btts_yes,
            fair_odd_btts_no,
            fair_odd_over_15,
            fair_odd_under_15,
            fair_odd_over_25,
            fair_odd_under_25,
            fair_odd_over_35,
            fair_odd_under_35,

            fair_odd_asian_over_200,
            fair_odd_asian_under_200,
            fair_odd_asian_over_225,
            fair_odd_asian_under_225,
            fair_odd_asian_over_250,
            fair_odd_asian_under_250,
            fair_odd_asian_over_275,
            fair_odd_asian_under_275,
            fair_odd_asian_over_300,
            fair_odd_asian_under_300,

            odd_over25_open,
            odd_under25_open,
            ev_over25_open,
            ev_under25_open,

            top_scores_json,
            projection_json,
            status,
            created_at,
            updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, 'OPEN', ?, ?
        )
        """,
        (
            prediction_timestamp,
            row.get("fixture_key"),
            fixture_date_text,
            row.get("time"),
            calculate_season(fixture_date),
            league,
            home_team,
            away_team,
            GOAL_MODEL_VERSION,
            json.dumps(
                asdict(DEFAULT_CONFIG),
                ensure_ascii=False,
                sort_keys=True,
            ),
            league_home_avg,
            league_away_avg,

            float(flat["goal_lambda_home"]),
            float(flat["goal_lambda_away"]),
            float(flat["goal_expected_total"]),

            float(flat["goal_prob_home_win"]),
            float(flat["goal_prob_draw"]),
            float(flat["goal_prob_away_win"]),
            float(flat["goal_prob_btts_yes"]),
            float(flat["goal_prob_btts_no"]),

            float(flat["goal_prob_over_15"]),
            float(flat["goal_prob_under_15"]),
            float(flat["goal_prob_over_25"]),
            float(flat["goal_prob_under_25"]),
            float(flat["goal_prob_over_35"]),
            float(flat["goal_prob_under_35"]),

            float(flat["goal_prob_home_over_05"]),
            float(flat["goal_prob_home_over_15"]),
            float(flat["goal_prob_home_over_25"]),
            float(flat["goal_prob_away_over_05"]),
            float(flat["goal_prob_away_over_15"]),
            float(flat["goal_prob_away_over_25"]),

            match_result["fair_odd_home"],
            match_result["fair_odd_draw"],
            match_result["fair_odd_away"],
            btts["fair_odd_yes"],
            btts["fair_odd_no"],
            totals["1.5"]["fair_odd_over"],
            totals["1.5"]["fair_odd_under"],
            totals["2.5"]["fair_odd_over"],
            totals["2.5"]["fair_odd_under"],
            totals["3.5"]["fair_odd_over"],
            totals["3.5"]["fair_odd_under"],

            asian["2.00"]["over"]["fair_odd"],
            asian["2.00"]["under"]["fair_odd"],
            asian["2.25"]["over"]["fair_odd"],
            asian["2.25"]["under"]["fair_odd"],
            asian["2.50"]["over"]["fair_odd"],
            asian["2.50"]["under"]["fair_odd"],
            asian["2.75"]["over"]["fair_odd"],
            asian["2.75"]["under"]["fair_odd"],
            asian["3.00"]["over"]["fair_odd"],
            asian["3.00"]["under"]["fair_odd"],

            odd_over25_open,
            odd_under25_open,
            ev_over25_open,
            ev_under25_open,

            json.dumps(
                projection["top_scores"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
            ),
            prediction_timestamp,
            prediction_timestamp,
        ),
    )


def main() -> None:
    print()
    print("==============================================")
    print(" FOOTBALLAI — GOAL MODEL MULTI-MARKET SHADOW")
    print("==============================================")
    print()

    prediction_timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )
    today = datetime.now(
        core.APP_TIMEZONE
    ).date()

    connection = core.connect_turso()

    try:
        if not core.table_exists(connection, "matches"):
            raise RuntimeError(
                "A tabela matches não existe na Turso."
            )

        if not core.table_exists(
            connection,
            core.UPCOMING_TABLE,
        ):
            raise RuntimeError(
                "A tabela upcoming_matches não existe na Turso."
            )

        ensure_goal_shadow_table(connection)

        # paper_trade_uefa_cloud já substituiu esta função por uma versão
        # que combina histórico doméstico + UEFA quando disponível.
        matches = core.load_historical_matches(
            connection
        )

        settled = settle_existing_shadow(
            connection,
            matches,
        )

        team_history = create_team_history(
            matches
        )
        elo_ratings = calculate_current_elo(
            matches
        )
        league_averages = build_league_goal_averages(
            matches
        )

        latest_history_date = pd.Timestamp(
            matches["date"].max()
        )

        upcoming = core.load_upcoming(
            connection
        )
        existing_keys = existing_shadow_keys(
            connection
        )

        inserted = 0
        already_exists = 0
        skipped = 0
        resolved_names = 0
        skipped_examples: list[str] = []

        for row in upcoming:
            try:
                fixture_date = parse_date(
                    row.get("date")
                )
                if fixture_date is None:
                    raise ValueError("data inválida")

                if fixture_date.date() < today:
                    raise ValueError("jogo no passado")

                if (
                    fixture_date.normalize()
                    <= latest_history_date.normalize()
                ):
                    raise ValueError(
                        "data não posterior ao último jogo histórico"
                    )

                league = normalize_league(
                    row.get("league")
                )
                if league is None:
                    raise ValueError("liga inválida")

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

                resolved_home = core.resolve_team_to_history(
                    team_history,
                    str(league),
                    str(home_team),
                )
                if resolved_home != home_team:
                    resolved_names += 1
                    home_team = resolved_home

                resolved_away = core.resolve_team_to_history(
                    team_history,
                    str(league),
                    str(away_team),
                )
                if resolved_away != away_team:
                    resolved_names += 1
                    away_team = resolved_away

                fixture_date_text = fixture_date.strftime(
                    "%Y-%m-%d"
                )
                key = (
                    fixture_date_text,
                    str(league),
                    str(home_team),
                    str(away_team),
                )

                # Primeira previsão fica congelada para evitar leakage.
                if key in existing_keys:
                    already_exists += 1
                    continue

                sports_features = core.build_sports_features(
                    history=team_history,
                    elo_ratings=elo_ratings,
                    league=str(league),
                    home_team=str(home_team),
                    away_team=str(away_team),
                    fixture_date=fixture_date,
                )

                league_home_avg = None
                league_away_avg = None
                if str(league) in league_averages:
                    (
                        league_home_avg,
                        league_away_avg,
                    ) = league_averages[str(league)]

                projection = project_match(
                    sports_features,
                    league_home_avg=league_home_avg,
                    league_away_avg=league_away_avg,
                )

                insert_shadow_projection(
                    connection,
                    row=row,
                    fixture_date=fixture_date,
                    league=str(league),
                    home_team=str(home_team),
                    away_team=str(away_team),
                    projection=projection,
                    prediction_timestamp=prediction_timestamp,
                    league_home_avg=league_home_avg,
                    league_away_avg=league_away_avg,
                )

                existing_keys.add(key)
                inserted += 1

            except Exception as error:
                skipped += 1
                if len(skipped_examples) < 20:
                    skipped_examples.append(
                        f"{row.get('date')} | "
                        f"{row.get('home_team')} vs "
                        f"{row.get('away_team')} | {error}"
                    )

        connection.commit()

        open_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {GOAL_SHADOW_TABLE}
            WHERE status = 'OPEN'
            """
        ).fetchone()[0]

        settled_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {GOAL_SHADOW_TABLE}
            WHERE status = 'SETTLED'
            """
        ).fetchone()[0]

        print(f"Jogos históricos          : {len(matches)}")
        print(f"Upcoming ativos           : {len(upcoming)}")
        print(f"Novas previsões shadow    : {inserted}")
        print(f"Já existentes             : {already_exists}")
        print(f"Resultados liquidados     : {settled}")
        print(f"Nomes resolvidos          : {resolved_names}")
        print(f"Ignorados                 : {skipped}")
        print(f"Shadow OPEN total         : {open_count}")
        print(f"Shadow SETTLED total      : {settled_count}")

        if skipped_examples:
            print()
            print("Exemplos ignorados:")
            for example in skipped_examples:
                print(f" - {example}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
