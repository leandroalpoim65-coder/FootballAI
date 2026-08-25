from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import libsql
import pandas as pd

from ml.paper_trade import normalize_league, normalize_team


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")

FOOTBALL_DATA_BASE_URL = (
    "https://www.football-data.co.uk/mmz4281"
)

FOOTBALL_DATA_DIVISION = {
    "Belgium First Division A": "B1",
    "England": "E0",
    "England Championship": "E1",
    "England League One": "E2",
    "England League Two": "E3",
    "France": "F1",
    "France Ligue 2": "F2",
    "Germany": "D1",
    "Germany 2 Bundesliga": "D2",
    "Greece Super League": "G1",
    "Italy": "I1",
    "Italy Serie B": "I2",
    "Netherlands": "N1",
    "Portugal": "P1",
    "Scotland": "SC0",
    "Spain": "SP1",
    "Spain Segunda Division": "SP2",
    "Turkey Super Lig": "T1",
}


def connect_turso():
    import os

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


def find_paper_table(
    connection,
) -> str | None:
    existing = {
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
    }

    for candidate in (
        "paper_trades",
        "paper_trade",
        "paper_trading",
        "historico_paper_trade",
    ):
        if candidate in existing:
            return candidate

    return None


def table_columns(
    connection,
    table: str,
) -> set[str]:
    return {
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def add_column_if_missing(
    connection,
    table: str,
    columns: set[str],
    name: str,
    sql_type: str,
) -> None:
    if name in columns:
        return

    connection.execute(
        f"ALTER TABLE {table} "
        f"ADD COLUMN {name} {sql_type}"
    )

    columns.add(name)


def normalize_name(
    value: object,
) -> str:
    text = (
        ""
        if value is None
        else str(value)
    )

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.lower()
    text = text.replace("&", " and ")

    text = re.sub(
        r"\bsp\b",
        " sporting ",
        text,
    )

    text = re.sub(
        r"\butd\b",
        " united ",
        text,
    )

    text = re.sub(
        r"\bst\b",
        " saint ",
        text,
    )

    text = re.sub(
        r"\b(fc|cf|afc|sc|fk|ac|sv|rcd|cd)\b",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def canonical_team(
    league: str | None,
    value: object,
) -> str:
    normalized = normalize_team(
        league,
        value,
    )

    return normalize_name(normalized)


def team_names_match(
    league: str | None,
    left: object,
    right: object,
) -> bool:
    a = canonical_team(
        league,
        left,
    )

    b = canonical_team(
        league,
        right,
    )

    return bool(
        a
        and b
        and a == b
    )


def parse_db_date(
    value: object,
) -> date | None:
    if value is None:
        return None

    text = str(value).strip()

    if (
        not text
        or text.casefold() in {
            "nan",
            "none",
            "<na>",
        }
    ):
        return None

    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_football_data_date(
    value: object,
) -> date | None:
    if value is None:
        return None

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        dayfirst=True,
    )

    if pd.isna(parsed):
        return None

    return parsed.date()


def season_start_for(
    reference_date: date,
) -> date:
    if reference_date.month >= 7:
        return date(
            reference_date.year,
            7,
            1,
        )

    return date(
        reference_date.year - 1,
        7,
        1,
    )


def football_data_season_code(
    reference_date: date,
) -> str:
    start = season_start_for(reference_date)
    end_year = start.year + 1

    return (
        f"{start.year % 100:02d}"
        f"{end_year % 100:02d}"
    )


def safe_float(
    value: object,
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number != number:
        return None

    return number


def safe_int(
    value: object,
) -> int | None:
    number = safe_float(value)

    if number is None:
        return None

    rounded = int(number)

    if abs(number - rounded) > 1e-9:
        return None

    return rounded


def safe_odd(
    value: object,
) -> float | None:
    number = safe_float(value)

    if (
        number is None
        or number <= 1.0
    ):
        return None

    return number


def get_bytes(
    url: str,
) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 FootballAI/Cloud"
        },
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:
            return response.read()

    except HTTPError as error:
        try:
            detail = (
                error.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )
        except Exception:
            detail = ""

        raise RuntimeError(
            f"HTTP {error.code}: {detail[:300]}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Erro de ligação: {error.reason}"
        ) from error


def read_football_data_csv(
    content: bytes,
) -> pd.DataFrame:
    errors: list[str] = []

    for encoding in (
        "utf-8-sig",
        "latin-1",
    ):
        try:
            return pd.read_csv(
                BytesIO(content),
                encoding=encoding,
                low_memory=False,
                on_bad_lines="skip",
            )
        except Exception as error:
            errors.append(
                f"{encoding}: {error}"
            )

    raise RuntimeError(
        "Não foi possível ler CSV Football-Data. "
        + " | ".join(errors)
    )


def fetch_football_data_league(
    canonical_league: str,
    division: str,
    season_code: str,
) -> tuple[list[dict], str | None]:
    url = (
        f"{FOOTBALL_DATA_BASE_URL}/"
        f"{season_code}/"
        f"{division}.csv"
    )

    try:
        content = get_bytes(url)
        raw = read_football_data_csv(content)
    except Exception as error:
        return [], str(error)

    required = {
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    }

    missing = required - set(raw.columns)

    if missing:
        return (
            [],
            "colunas em falta: "
            + ", ".join(sorted(missing)),
        )

    results: list[dict] = []

    for _, row in raw.iterrows():
        match_date = parse_football_data_date(
            row.get("Date")
        )

        home_goals = safe_int(
            row.get("FTHG")
        )

        away_goals = safe_int(
            row.get("FTAG")
        )

        if (
            match_date is None
            or home_goals is None
            or away_goals is None
        ):
            continue

        home_team = row.get("HomeTeam")
        away_team = row.get("AwayTeam")

        if (
            home_team is None
            or away_team is None
        ):
            continue

        results.append(
            {
                "date": match_date,
                "league": canonical_league,
                "home_team": str(home_team),
                "away_team": str(away_team),
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    return results, None


def find_football_data_result(
    results: list[dict],
    canonical_league: str,
    fixture_date: date,
    home_team: str,
    away_team: str,
) -> dict | None:
    matches: list[dict] = []

    for result in results:
        if result.get("date") != fixture_date:
            continue

        if result.get("league") != canonical_league:
            continue

        if not team_names_match(
            canonical_league,
            result.get("home_team"),
            home_team,
        ):
            continue

        if not team_names_match(
            canonical_league,
            result.get("away_team"),
            away_team,
        ):
            continue

        matches.append(result)

    if len(matches) == 1:
        return matches[0]

    return None


def calculate_trade_values(
    row: dict,
    columns: set[str],
    home_goals: int,
    away_goals: int,
) -> tuple[int, float, float | None]:
    result_over25 = int(
        home_goals + away_goals > 2
    )

    signal = 0
    planned_stake = 0.0

    if "signal" in columns:
        try:
            signal = int(
                row.get("signal") or 0
            )
        except (TypeError, ValueError):
            signal = 0

    if "planned_stake" in columns:
        try:
            planned_stake = float(
                row.get("planned_stake") or 0
            )
        except (TypeError, ValueError):
            planned_stake = 0.0

    open_odd = (
        safe_odd(
            row.get("odd_over25_open")
        )
        if "odd_over25_open" in columns
        else None
    )

    close_odd = (
        safe_odd(
            row.get("odd_over25_close")
        )
        if "odd_over25_close" in columns
        else None
    )

    profit = 0.0

    if (
        signal == 1
        and planned_stake > 0
        and open_odd is not None
    ):
        if result_over25 == 1:
            profit = (
                planned_stake
                * (open_odd - 1.0)
            )
        else:
            profit = -planned_stake

    clv = None

    if (
        open_odd is not None
        and close_odd is not None
    ):
        clv = (
            open_odd
            / close_odd
            - 1.0
        )

    return (
        result_over25,
        profit,
        clv,
    )


def update_trade(
    connection,
    table: str,
    row: dict,
    columns: set[str],
    home_goals: int,
    away_goals: int,
) -> float | None:
    (
        result_over25,
        profit,
        clv,
    ) = calculate_trade_values(
        row,
        columns,
        home_goals,
        away_goals,
    )

    connection.execute(
        f"""
        UPDATE {table}
        SET
            status = 'RESOLVED',
            home_goals = ?,
            away_goals = ?,
            result_over25 = ?,
            profit = ?,
            clv = ?
        WHERE id = ?
        """,
        (
            home_goals,
            away_goals,
            result_over25,
            profit,
            clv,
            int(row["id"]),
        ),
    )

    return clv


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — RESULTADOS CLOUD")
    print("==========================================")
    print()

    today = datetime.now(
        APP_TIMEZONE
    ).date()

    season_start = season_start_for(today)
    season_code = football_data_season_code(today)

    try:
        connection = connect_turso()
    except Exception as error:
        print(
            f"❌ Não foi possível ligar à Turso: {error}"
        )
        raise SystemExit(1) from error

    print("Base de dados             : Turso Cloud")
    print("Timezone                  : Europe/Lisbon")
    print(
        f"Época Football-Data       : {season_code}"
    )
    print(
        f"Desde                     : "
        f"{season_start.isoformat()}"
    )

    try:
        table = find_paper_table(connection)

        if table is None:
            print(
                "❌ Nenhuma tabela de paper trading encontrada."
            )
            raise SystemExit(1)

        columns = table_columns(
            connection,
            table,
        )

        for name, sql_type in (
            ("status", "TEXT"),
            ("home_goals", "INTEGER"),
            ("away_goals", "INTEGER"),
            ("result_over25", "INTEGER"),
            ("profit", "REAL"),
            ("clv", "REAL"),
            ("odd_over25_close", "REAL"),
            ("odd_under25_close", "REAL"),
        ):
            add_column_if_missing(
                connection,
                table,
                columns,
                name,
                sql_type,
            )

        connection.commit()

        fixture_date_column = (
            "fixture_date"
            if "fixture_date" in columns
            else "date"
            if "date" in columns
            else None
        )

        if fixture_date_column is None:
            print(
                "❌ A tabela não tem fixture_date/date."
            )
            raise SystemExit(1)

        required_columns = {
            "id",
            "status",
            "league",
            "home_team",
            "away_team",
            fixture_date_column,
        }

        missing_columns = (
            required_columns
            - columns
        )

        if missing_columns:
            print(
                "❌ Colunas em falta: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )
            raise SystemExit(1)

        rows = fetch_dicts(
            connection,
            f"""
            SELECT *
            FROM {table}
            WHERE COALESCE(status, 'OPEN') != 'RESOLVED'
              AND date({fixture_date_column}) <= date(?)
              AND date({fixture_date_column}) >= date(?)
            ORDER BY
                {fixture_date_column},
                id
            """,
            (
                today.isoformat(),
                season_start.isoformat(),
            ),
        )

        print(
            f"Registos candidatos       : {len(rows)}"
        )

        if not rows:
            print(
                "✅ Nada para resolver neste momento."
            )
            return

        metadata_by_id: dict[
            int,
            dict[str, object],
        ] = {}

        required_football_data: dict[
            str,
            str,
        ] = {}

        for row in rows:
            row_id = int(row["id"])

            canonical_league = normalize_league(
                row.get("league")
            )

            fixture_date = parse_db_date(
                row.get(fixture_date_column)
            )

            division = FOOTBALL_DATA_DIVISION.get(
                canonical_league
            )

            metadata_by_id[row_id] = {
                "league": canonical_league,
                "fixture_date": fixture_date,
                "division": division,
            }

            if (
                isinstance(canonical_league, str)
                and division
            ):
                required_football_data[
                    canonical_league
                ] = division

        print()
        print("Football-Data:")

        football_results: list[dict] = []
        football_errors: list[str] = []

        for canonical_league, division in sorted(
            required_football_data.items()
        ):
            results, error = fetch_football_data_league(
                canonical_league,
                division,
                season_code,
            )

            if error is not None:
                football_errors.append(
                    f"{division} / "
                    f"{canonical_league}: {error}"
                )

                print(
                    f"  ⚠ {division} "
                    f"{canonical_league}: falhou"
                )
                continue

            football_results.extend(results)

            print(
                f"  ✅ {division} "
                f"{canonical_league}: "
                f"{len(results)} resultados"
            )

        print(
            f"Resultados Football-Data  : "
            f"{len(football_results)}"
        )

        resolved_ids: set[int] = set()
        resolved_football = 0
        with_clv = 0
        without_clv = 0

        for row in rows:
            row_id = int(row["id"])
            metadata = metadata_by_id[row_id]

            canonical_league = metadata["league"]
            fixture_date = metadata["fixture_date"]

            if (
                not isinstance(canonical_league, str)
                or not isinstance(fixture_date, date)
            ):
                continue

            result = find_football_data_result(
                football_results,
                canonical_league,
                fixture_date,
                str(row.get("home_team", "")),
                str(row.get("away_team", "")),
            )

            if result is None:
                continue

            home_goals = int(
                result["home_goals"]
            )

            away_goals = int(
                result["away_goals"]
            )

            clv = update_trade(
                connection,
                table,
                row,
                columns,
                home_goals,
                away_goals,
            )

            resolved_ids.add(row_id)
            resolved_football += 1

            if clv is None:
                without_clv += 1
            else:
                with_clv += 1

            print(
                f"✅ {fixture_date.isoformat()} "
                f"{row.get('home_team')} "
                f"{home_goals}-{away_goals} "
                f"{row.get('away_team')}"
            )

        connection.commit()

        remaining = [
            row
            for row in rows
            if int(row["id"]) not in resolved_ids
        ]

        print()
        print("==========================================")
        print(" RESUMO")
        print("==========================================")
        print(
            f"Resolvidos Football-Data  : "
            f"{resolved_football}"
        )
        print(
            f"Com CLV válido            : {with_clv}"
        )
        print(
            f"Sem CLOSE / CLV           : {without_clv}"
        )
        print(
            f"Ainda por resolver        : {len(remaining)}"
        )

        if football_errors:
            print()
            print("Avisos Football-Data:")

            for error in football_errors[:10]:
                print(f"  - {error}")

        if remaining:
            print()
            print("Ainda por resolver:")

            for row in remaining[:30]:
                print(
                    f"  - {row.get(fixture_date_column)} | "
                    f"{row.get('league')} | "
                    f"{row.get('home_team')} vs "
                    f"{row.get('away_team')}"
                )

        print()
        print(
            "CLV é calculado apenas quando existem "
            "OPEN e CLOSE válidas."
        )
        print(
            "Este script nunca altera as odds OPEN "
            "ou CLOSE."
        )
        print(
            "A versão cloud usa Football-Data como "
            "fonte de resultados para não gastar "
            "créditos da The Odds API."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
