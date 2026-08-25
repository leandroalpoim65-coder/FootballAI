from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from ml.paper_trade import normalize_league, normalize_team


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATABASE_PATH = (
    ROOT_DIR
    / "database"
    / "football.db"
)

ENV_PATH = (
    ROOT_DIR
    / ".env"
)


# ============================================================
# FONTES DE RESULTADOS
# ============================================================

ODDS_API_BASE_URL = (
    "https://api.the-odds-api.com/v4"
)

FOOTBALL_DATA_BASE_URL = (
    "https://www.football-data.co.uk/mmz4281"
)


# The Odds API só permite resultados concluídos até 3 dias atrás.
ODDS_API_DAYS_FROM = 3


# ============================================================
# MAPAS DE LIGAS
# ============================================================

# Nomes CANÓNICOS -> sport_key da The Odds API.
LEAGUE_SPORT_KEY = {
    "Belgium First Division A":
        "soccer_belgium_first_div",

    "England":
        "soccer_epl",

    "England Championship":
        "soccer_efl_champ",

    "England League One":
        "soccer_england_league1",

    "England League Two":
        "soccer_england_league2",

    "France":
        "soccer_france_ligue_one",

    "France Ligue 2":
        "soccer_france_ligue_two",

    "Germany":
        "soccer_germany_bundesliga",

    "Germany 2 Bundesliga":
        "soccer_germany_bundesliga2",

    "Greece Super League":
        "soccer_greece_super_league",

    "Italy":
        "soccer_italy_serie_a",

    "Italy Serie B":
        "soccer_italy_serie_b",

    "Netherlands":
        "soccer_netherlands_eredivisie",

    "Portugal":
        "soccer_portugal_primeira_liga",

    "Scotland":
        "soccer_spl",

    "Spain":
        "soccer_spain_la_liga",

    "Spain Segunda Division":
        "soccer_spain_segunda_division",

    "Turkey Super Lig":
        "soccer_turkey_super_league",
}


# Nomes CANÓNICOS -> divisão Football-Data.co.uk.
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


# ============================================================
# .ENV
# ============================================================

def load_simple_env(
    path: Path,
) -> None:

    if not path.exists():
        return

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        key = key.strip()

        value = (
            value
            .strip()
            .strip('"')
            .strip("'")
        )

        if key:
            os.environ.setdefault(
                key,
                value,
            )


# ============================================================
# NORMALIZAÇÃO
# ============================================================

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
        if not unicodedata.combining(
            char
        )
    )

    text = text.lower()

    text = text.replace(
        "&",
        " and ",
    )

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

    return normalize_name(
        normalized
    )


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

    if (
        not a
        or not b
    ):
        return False

    return a == b


# ============================================================
# DATAS
# ============================================================

def parse_db_date(
    value: object,
) -> date | None:

    if value is None:
        return None

    text = str(
        value
    ).strip()

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
        return date.fromisoformat(
            text[:10]
        )

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

    if pd.isna(
        parsed
    ):
        return None

    return parsed.date()


def event_local_date(
    event: dict,
) -> date | None:

    raw_value = event.get(
        "commence_time"
    )

    if raw_value is None:
        return None

    text = str(
        raw_value
    ).strip()

    if not text:
        return None

    try:

        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:

        return None

    if parsed.tzinfo is not None:

        parsed = parsed.astimezone()

    return parsed.date()


def season_start_for(
    reference_date: date,
) -> date:
    """
    Época europeia:
      julho -> dezembro: começa em julho do próprio ano
      janeiro -> junho:  começa em julho do ano anterior
    """

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
    """
    2026/2027 -> 2627
    2027/2028 -> 2728
    """

    start = season_start_for(
        reference_date
    )

    end_year = (
        start.year
        + 1
    )

    return (
        f"{start.year % 100:02d}"
        f"{end_year % 100:02d}"
    )


# ============================================================
# NÚMEROS / ODDS
# ============================================================

def safe_float(
    value: object,
) -> float | None:

    try:

        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if number != number:
        return None

    return number


def safe_int(
    value: object,
) -> int | None:

    number = safe_float(
        value
    )

    if number is None:
        return None

    rounded = int(
        number
    )

    if abs(
        number - rounded
    ) > 1e-9:
        return None

    return rounded


def safe_odd(
    value: object,
) -> float | None:

    number = safe_float(
        value
    )

    if (
        number is None
        or number <= 1.0
    ):
        return None

    return number


# ============================================================
# HTTP
# ============================================================

def get_bytes(
    url: str,
) -> tuple[
    bytes,
    dict[str, str],
]:

    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 FootballAI/2.0"
        },
    )

    try:

        with urlopen(
            request,
            timeout=30,
        ) as response:

            content = response.read()

            headers = {
                key.lower(): value
                for key, value
                in response.headers.items()
            }

    except HTTPError as error:

        try:

            detail = (
                error
                .read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

        except Exception:

            detail = ""

        raise RuntimeError(
            f"HTTP {error.code}: "
            f"{detail[:300]}"
        ) from error

    except URLError as error:

        raise RuntimeError(
            "Erro de ligação: "
            f"{error.reason}"
        ) from error

    return (
        content,
        headers,
    )


def get_json(
    url: str,
) -> tuple[
    object,
    dict[str, str],
]:

    content, headers = get_bytes(
        url
    )

    try:

        payload = json.loads(
            content.decode(
                "utf-8"
            )
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:

        raise RuntimeError(
            "Resposta JSON inválida."
        ) from error

    return (
        payload,
        headers,
    )


def is_quota_error(
    error: Exception,
) -> bool:

    text = str(
        error
    ).casefold()

    return (
        "out_of_usage_credits" in text
        or "usage quota has been reached" in text
        or "usage quota" in text
    )


# ============================================================
# BASE DE DADOS
# ============================================================

def find_paper_table(
    connection: sqlite3.Connection,
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
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:

    return {
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def add_column_if_missing(
    connection: sqlite3.Connection,
    table: str,
    columns: set[str],
    name: str,
    sql_type: str,
) -> None:

    if name in columns:
        return

    connection.execute(
        f"""
        ALTER TABLE {table}
        ADD COLUMN {name} {sql_type}
        """
    )

    columns.add(
        name
    )


# ============================================================
# FOOTBALL-DATA.CO.UK
# ============================================================

def read_football_data_csv(
    content: bytes,
) -> pd.DataFrame:

    errors: list[
        str
    ] = []

    for encoding in (
        "utf-8-sig",
        "latin-1",
    ):

        try:

            dataframe = pd.read_csv(
                BytesIO(
                    content
                ),
                encoding=encoding,
                low_memory=False,
                on_bad_lines="skip",
            )

            return dataframe

        except Exception as error:

            errors.append(
                f"{encoding}: {error}"
            )

    raise RuntimeError(
        "Não foi possível ler CSV Football-Data. "
        + " | ".join(
            errors
        )
    )


def fetch_football_data_league(
    canonical_league: str,
    division: str,
    season_code: str,
) -> tuple[
    list[dict],
    str | None,
]:

    url = (
        f"{FOOTBALL_DATA_BASE_URL}/"
        f"{season_code}/"
        f"{division}.csv"
    )

    try:

        content, _ = get_bytes(
            url
        )

        raw = read_football_data_csv(
            content
        )

    except Exception as error:

        return (
            [],
            str(
                error
            ),
        )

    required = {
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    }

    missing = (
        required
        - set(
            raw.columns
        )
    )

    if missing:

        return (
            [],
            "colunas em falta: "
            + ", ".join(
                sorted(
                    missing
                )
            ),
        )

    results: list[
        dict
    ] = []

    for _, row in raw.iterrows():

        match_date = parse_football_data_date(
            row.get(
                "Date"
            )
        )

        home_goals = safe_int(
            row.get(
                "FTHG"
            )
        )

        away_goals = safe_int(
            row.get(
                "FTAG"
            )
        )

        if (
            match_date is None
            or home_goals is None
            or away_goals is None
        ):
            continue

        home_team = row.get(
            "HomeTeam"
        )

        away_team = row.get(
            "AwayTeam"
        )

        if (
            home_team is None
            or away_team is None
        ):
            continue

        results.append(
            {
                "date":
                    match_date,

                "league":
                    canonical_league,

                "home_team":
                    str(
                        home_team
                    ),

                "away_team":
                    str(
                        away_team
                    ),

                "home_goals":
                    home_goals,

                "away_goals":
                    away_goals,

                "source":
                    "Football-Data",
            }
        )

    return (
        results,
        None,
    )


def find_football_data_result(
    results: list[dict],
    canonical_league: str,
    fixture_date: date,
    home_team: str,
    away_team: str,
) -> dict | None:

    matches: list[
        dict
    ] = []

    for result in results:

        if (
            result.get(
                "date"
            )
            != fixture_date
        ):
            continue

        if (
            result.get(
                "league"
            )
            != canonical_league
        ):
            continue

        if not team_names_match(
            canonical_league,
            result.get(
                "home_team"
            ),
            home_team,
        ):
            continue

        if not team_names_match(
            canonical_league,
            result.get(
                "away_team"
            ),
            away_team,
        ):
            continue

        matches.append(
            result
        )

    if len(
        matches
    ) == 1:
        return matches[
            0
        ]

    return None


# ============================================================
# THE ODDS API
# ============================================================

def find_odds_api_event(
    events: list[dict],
    league: str,
    fixture_date: date,
    home_team: str,
    away_team: str,
) -> dict | None:

    matches: list[
        dict
    ] = []

    for event in events:

        if not isinstance(
            event,
            dict,
        ):
            continue

        if not event.get(
            "completed"
        ):
            continue

        api_date = event_local_date(
            event
        )

        if (
            api_date is None
            or api_date != fixture_date
        ):
            continue

        if not team_names_match(
            league,
            event.get(
                "home_team"
            ),
            home_team,
        ):
            continue

        if not team_names_match(
            league,
            event.get(
                "away_team"
            ),
            away_team,
        ):
            continue

        matches.append(
            event
        )

    if len(
        matches
    ) == 1:
        return matches[
            0
        ]

    return None


def extract_odds_api_score(
    event: dict,
    league: str,
    team_name: str,
) -> int | None:

    scores = event.get(
        "scores"
    )

    if not isinstance(
        scores,
        list,
    ):
        return None

    values: list[
        int
    ] = []

    for item in scores:

        if not isinstance(
            item,
            dict,
        ):
            continue

        if not team_names_match(
            league,
            item.get(
                "name"
            ),
            team_name,
        ):
            continue

        score = safe_int(
            item.get(
                "score"
            )
        )

        if score is None:
            continue

        values.append(
            score
        )

    if len(
        values
    ) == 1:
        return values[
            0
        ]

    return None


# ============================================================
# RESOLUÇÃO DE UMA LINHA
# ============================================================

def calculate_trade_values(
    row: sqlite3.Row,
    columns: set[str],
    home_goals: int,
    away_goals: int,
) -> tuple[
    int,
    float,
    float | None,
]:

    result_over25 = int(
        (
            home_goals
            + away_goals
        )
        > 2
    )

    signal = 0
    planned_stake = 0.0

    if "signal" in columns:

        try:

            signal = int(
                row[
                    "signal"
                ]
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):

            signal = 0

    if "planned_stake" in columns:

        try:

            planned_stake = float(
                row[
                    "planned_stake"
                ]
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):

            planned_stake = 0.0

    open_odd = (
        safe_odd(
            row[
                "odd_over25_open"
            ]
        )
        if "odd_over25_open" in columns
        else None
    )

    close_odd = (
        safe_odd(
            row[
                "odd_over25_close"
            ]
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
                * (
                    open_odd
                    - 1.0
                )
            )

        else:

            profit = (
                -planned_stake
            )

    # CLV só existe com OPEN e CLOSE válidas.
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
    connection: sqlite3.Connection,
    table: str,
    row: sqlite3.Row,
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
            int(
                row[
                    "id"
                ]
            ),
        ),
    )

    return clv


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=========================================="
    )

    print(
        " FOOTBALLAI — ATUALIZAR RESULTADOS"
    )

    print(
        "=========================================="
    )

    print()

    load_simple_env(
        ENV_PATH
    )

    if not DATABASE_PATH.exists():

        print(
            "❌ Base de dados não encontrada:"
        )

        print(
            DATABASE_PATH
        )

        return

    today = date.today()

    season_start = season_start_for(
        today
    )

    season_code = football_data_season_code(
        today
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    table = find_paper_table(
        connection
    )

    if table is None:

        connection.close()

        print(
            "Nenhuma tabela de paper trading encontrada."
        )

        return

    columns = table_columns(
        connection,
        table,
    )

    # ========================================================
    # GARANTIR COLUNAS
    # ========================================================

    for name, sql_type in (
        (
            "status",
            "TEXT",
        ),
        (
            "home_goals",
            "INTEGER",
        ),
        (
            "away_goals",
            "INTEGER",
        ),
        (
            "result_over25",
            "INTEGER",
        ),
        (
            "profit",
            "REAL",
        ),
        (
            "clv",
            "REAL",
        ),
        (
            "odd_over25_close",
            "REAL",
        ),
        (
            "odd_under25_close",
            "REAL",
        ),
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
        else
        "date"
        if "date" in columns
        else
        None
    )

    if fixture_date_column is None:

        connection.close()

        print(
            "❌ A tabela não tem fixture_date/date."
        )

        return

    # ========================================================
    # CANDIDATOS: TODA A ÉPOCA ATUAL
    # ========================================================
    #
    # Antes eram apenas 3 dias.
    # Isso fazia desaparecer jogos antigos que ficaram por
    # resolver quando a quota da Odds API acabou.
    # ========================================================

    rows = connection.execute(
        f"""
        SELECT *
        FROM {table}
        WHERE COALESCE(
            status,
            'OPEN'
        ) != 'RESOLVED'
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
    ).fetchall()

    print(
        f"Época Football-Data       : "
        f"{season_code}"
    )

    print(
        f"Desde                     : "
        f"{season_start.isoformat()}"
    )

    print(
        f"Registos candidatos       : "
        f"{len(rows)}"
    )

    if not rows:

        connection.close()

        print(
            "Nada para resolver neste momento."
        )

        return

    # ========================================================
    # METADADOS CANÓNICOS
    # ========================================================

    metadata_by_id: dict[
        int,
        dict[str, object],
    ] = {}

    required_football_data: dict[
        str,
        str,
    ] = {}

    for row in rows:

        row_id = int(
            row[
                "id"
            ]
        )

        canonical_league = normalize_league(
            row[
                "league"
            ]
        )

        fixture_date = parse_db_date(
            row[
                fixture_date_column
            ]
        )

        division = FOOTBALL_DATA_DIVISION.get(
            canonical_league
        )

        sport_key = LEAGUE_SPORT_KEY.get(
            canonical_league
        )

        metadata_by_id[
            row_id
        ] = {
            "league":
                canonical_league,

            "fixture_date":
                fixture_date,

            "division":
                division,

            "sport_key":
                sport_key,
        }

        if (
            isinstance(
                canonical_league,
                str,
            )
            and division
        ):

            required_football_data[
                canonical_league
            ] = division

    # ========================================================
    # 1) FOOTBALL-DATA.CO.UK — FONTE PRINCIPAL
    # ========================================================

    print()
    print(
        "Football-Data:"
    )

    football_results: list[
        dict
    ] = []

    football_errors: list[
        str
    ] = []

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
                f"{canonical_league}: "
                f"{error}"
            )

            print(
                f"  ⚠ {division} "
                f"{canonical_league}: falhou"
            )

            continue

        football_results.extend(
            results
        )

        print(
            f"  ✅ {division} "
            f"{canonical_league}: "
            f"{len(results)} resultados"
        )

    print(
        f"Resultados Football-Data  : "
        f"{len(football_results)}"
    )

    # ========================================================
    # RESOLVER PRIMEIRO COM FOOTBALL-DATA
    # ========================================================

    resolved_ids: set[
        int
    ] = set()

    resolved_football = 0
    resolved_odds_api = 0
    with_clv = 0
    without_clv = 0

    for row in rows:

        row_id = int(
            row[
                "id"
            ]
        )

        metadata = metadata_by_id[
            row_id
        ]

        canonical_league = metadata[
            "league"
        ]

        fixture_date = metadata[
            "fixture_date"
        ]

        if (
            not isinstance(
                canonical_league,
                str,
            )
            or not isinstance(
                fixture_date,
                date,
            )
        ):

            continue

        result = find_football_data_result(
            football_results,
            canonical_league,
            fixture_date,
            str(
                row[
                    "home_team"
                ]
            ),
            str(
                row[
                    "away_team"
                ]
            ),
        )

        if result is None:
            continue

        home_goals = int(
            result[
                "home_goals"
            ]
        )

        away_goals = int(
            result[
                "away_goals"
            ]
        )

        clv = update_trade(
            connection,
            table,
            row,
            columns,
            home_goals,
            away_goals,
        )

        resolved_ids.add(
            row_id
        )

        resolved_football += 1

        if clv is None:
            without_clv += 1
        else:
            with_clv += 1

        print(
            f"✅ [Football-Data] "
            f"{row['fixture_date']} "
            f"{row['home_team']} "
            f"{home_goals}-{away_goals} "
            f"{row['away_team']}"
        )

    # ========================================================
    # 2) THE ODDS API — FALLBACK PARA OS ÚLTIMOS 3 DIAS
    # ========================================================

    api_key = os.getenv(
        "ODDS_API_KEY",
        "",
    ).strip()

    recent_cutoff = (
        today
        - timedelta(
            days=ODDS_API_DAYS_FROM
        )
    )

    recent_remaining_rows: list[
        sqlite3.Row
    ] = []

    for row in rows:

        row_id = int(
            row[
                "id"
            ]
        )

        if row_id in resolved_ids:
            continue

        fixture_date = metadata_by_id[
            row_id
        ][
            "fixture_date"
        ]

        if (
            isinstance(
                fixture_date,
                date,
            )
            and fixture_date >= recent_cutoff
        ):

            recent_remaining_rows.append(
                row
            )

    api_quota_exhausted = False
    api_errors: list[
        str
    ] = []

    credits_remaining: (
        str | None
    ) = None

    if (
        api_key
        and recent_remaining_rows
    ):

        print()
        print(
            "The Odds API fallback:"
        )

        needed_sport_keys = {
            metadata_by_id[
                int(
                    row[
                        "id"
                    ]
                )
            ][
                "sport_key"
            ]
            for row in recent_remaining_rows
        }

        needed_sport_keys.discard(
            None
        )

        events_by_sport: dict[
            str,
            list[dict],
        ] = {}

        for sport_key in sorted(
            str(
                value
            )
            for value in needed_sport_keys
        ):

            query = urlencode(
                {
                    "apiKey":
                        api_key,

                    "daysFrom":
                        ODDS_API_DAYS_FROM,

                    "dateFormat":
                        "iso",
                }
            )

            url = (
                f"{ODDS_API_BASE_URL}"
                f"/sports/"
                f"{sport_key}"
                f"/scores/?{query}"
            )

            try:

                payload, headers = get_json(
                    url
                )

            except Exception as error:

                api_errors.append(
                    f"{sport_key}: {error}"
                )

                if is_quota_error(
                    error
                ):

                    api_quota_exhausted = True

                    print(
                        "  ⚠ quota esgotada; "
                        "fallback API interrompido"
                    )

                    break

                print(
                    f"  ⚠ {sport_key}: falhou"
                )

                continue

            events_by_sport[
                sport_key
            ] = (
                payload
                if isinstance(
                    payload,
                    list,
                )
                else []
            )

            credits_remaining = headers.get(
                "x-requests-remaining",
                credits_remaining,
            )

            print(
                f"  ✅ {sport_key}: "
                f"{len(events_by_sport[sport_key])} eventos"
            )

        for row in recent_remaining_rows:

            row_id = int(
                row[
                    "id"
                ]
            )

            if row_id in resolved_ids:
                continue

            metadata = metadata_by_id[
                row_id
            ]

            canonical_league = metadata[
                "league"
            ]

            sport_key = metadata[
                "sport_key"
            ]

            fixture_date = metadata[
                "fixture_date"
            ]

            if (
                not isinstance(
                    canonical_league,
                    str,
                )
                or not sport_key
                or not isinstance(
                    fixture_date,
                    date,
                )
            ):

                continue

            event = find_odds_api_event(
                events_by_sport.get(
                    str(
                        sport_key
                    ),
                    [],
                ),
                canonical_league,
                fixture_date,
                str(
                    row[
                        "home_team"
                    ]
                ),
                str(
                    row[
                        "away_team"
                    ]
                ),
            )

            if event is None:
                continue

            home_goals = extract_odds_api_score(
                event,
                canonical_league,
                str(
                    row[
                        "home_team"
                    ]
                ),
            )

            away_goals = extract_odds_api_score(
                event,
                canonical_league,
                str(
                    row[
                        "away_team"
                    ]
                ),
            )

            if (
                home_goals is None
                or away_goals is None
            ):
                continue

            clv = update_trade(
                connection,
                table,
                row,
                columns,
                home_goals,
                away_goals,
            )

            resolved_ids.add(
                row_id
            )

            resolved_odds_api += 1

            if clv is None:
                without_clv += 1
            else:
                with_clv += 1

            print(
                f"✅ [Odds API] "
                f"{row['fixture_date']} "
                f"{row['home_team']} "
                f"{home_goals}-{away_goals} "
                f"{row['away_team']}"
            )

    elif not api_key:

        print()
        print(
            "The Odds API fallback: "
            "sem ODDS_API_KEY; ignorado."
        )

    elif not recent_remaining_rows:

        print()
        print(
            "The Odds API fallback: "
            "não é necessário."
        )

    # ========================================================
    # COMMIT ÚNICO
    # ========================================================

    connection.commit()

    # ========================================================
    # O QUE FICOU POR RESOLVER
    # ========================================================

    remaining: list[
        sqlite3.Row
    ] = []

    for row in rows:

        row_id = int(
            row[
                "id"
            ]
        )

        if row_id not in resolved_ids:

            remaining.append(
                row
            )

    connection.close()

    # ========================================================
    # RESUMO
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        " RESUMO"
    )

    print(
        "=========================================="
    )

    print(
        f"Resolvidos Football-Data  : "
        f"{resolved_football}"
    )

    print(
        f"Resolvidos The Odds API   : "
        f"{resolved_odds_api}"
    )

    print(
        f"Resolvidos total          : "
        f"{len(resolved_ids)}"
    )

    print(
        f"Com CLV válido            : "
        f"{with_clv}"
    )

    print(
        f"Sem CLOSE / CLV           : "
        f"{without_clv}"
    )

    print(
        f"Ainda por resolver        : "
        f"{len(remaining)}"
    )

    if credits_remaining is not None:

        print(
            f"Créditos API restantes    : "
            f"{credits_remaining}"
        )

    if api_quota_exhausted:

        print(
            "Quota Odds API            : ESGOTADA"
        )

    if football_errors:

        print()
        print(
            "Avisos Football-Data:"
        )

        for error in football_errors[
            :10
        ]:

            print(
                f"  - {error}"
            )

    if api_errors:

        print()
        print(
            "Avisos The Odds API:"
        )

        for error in api_errors[
            :10
        ]:

            print(
                f"  - {error}"
            )

    if remaining:

        print()
        print(
            "Ainda por resolver:"
        )

        for row in remaining[
            :30
        ]:

            print(
                f"  - "
                f"{row[fixture_date_column]} | "
                f"{row['league']} | "
                f"{row['home_team']} vs "
                f"{row['away_team']}"
            )

    print()
    print(
        "CLV continua a ser calculado apenas "
        "quando OPEN e CLOSE válidas existem."
    )

    print(
        "update_results.py nunca altera "
        "as odds OPEN ou CLOSE."
    )


if __name__ == "__main__":
    main()