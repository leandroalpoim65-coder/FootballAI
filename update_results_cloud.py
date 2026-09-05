from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import libsql
import pandas as pd

from ml.paper_trade import normalize_league, normalize_team


# ============================================================
# CONFIGURAÇÃO
# ============================================================

APP_TIMEZONE = ZoneInfo("Europe/Lisbon")

FOOTBALL_DATA_BASE_URL = (
    "https://www.football-data.co.uk/mmz4281"
)

ESPN_BASE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer"
)

ODDS_API_BASE_URL = (
    "https://api.the-odds-api.com/v4"
)

ESPN_RECENT_DAYS = 7
ODDS_API_DAYS_FROM = 3

RECOVERED_CLOSE_SOURCE_AVG = (
    "Football-Data / AvgC closing totals 2.5 / pós-jogo"
)

RECOVERED_CLOSE_SOURCE_B365 = (
    "Football-Data / Bet365 closing totals 2.5 / pós-jogo"
)


# ============================================================
# MAPAS DE LIGAS
# ============================================================

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

ESPN_LEAGUE_MAP = {
    "Belgium First Division A": "bel.1",
    "England": "eng.1",
    "England Championship": "eng.2",
    "England League One": "eng.3",
    "England League Two": "eng.4",
    "France": "fra.1",
    "France Ligue 2": "fra.2",
    "Germany": "ger.1",
    "Germany 2 Bundesliga": "ger.2",
    "Greece Super League": "gre.1",
    "Italy": "ita.1",
    "Italy Serie B": "ita.2",
    "Netherlands": "ned.1",
    "Portugal": "por.1",
    "Scotland": "sco.1",
    "Spain": "esp.1",
    "Spain Segunda Division": "esp.2",
    "Turkey Super Lig": "tur.1",
}

LEAGUE_SPORT_KEY = {
    "Belgium First Division A": "soccer_belgium_first_div",
    "England": "soccer_epl",
    "England Championship": "soccer_efl_champ",
    "England League One": "soccer_england_league1",
    "England League Two": "soccer_england_league2",
    "France": "soccer_france_ligue_one",
    "France Ligue 2": "soccer_france_ligue_two",
    "Germany": "soccer_germany_bundesliga",
    "Germany 2 Bundesliga": "soccer_germany_bundesliga2",
    "Greece Super League": "soccer_greece_super_league",
    "Italy": "soccer_italy_serie_a",
    "Italy Serie B": "soccer_italy_serie_b",
    "Netherlands": "soccer_netherlands_eredivisie",
    "Portugal": "soccer_portugal_primeira_liga",
    "Scotland": "soccer_spl",
    "Spain": "soccer_spain_la_liga",
    "Spain Segunda Division": "soccer_spain_segunda_division",
    "Turkey Super Lig": "soccer_turkey_super_league",
}

QUOTA_ERROR_MARKERS = (
    "OUT_OF_USAGE_CREDITS",
    "Usage quota has been reached",
    "usage quota",
)


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


def find_paper_table(connection) -> str | None:
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
        f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"
    )
    columns.add(name)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalize_name(value: object) -> str:
    text = "" if value is None else str(value)

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

    text = re.sub(r"\butd\b", " united ", text)
    text = re.sub(r"\bst\b", " saint ", text)

    text = re.sub(
        r"\b(fc|cf|afc|sc|fk|ac|sv|rcd|cd|club)\b",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


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
    a = canonical_team(league, left)
    b = canonical_team(league, right)

    if not a or not b:
        return False

    if a == b:
        return True

    aliases = {
        "bayern munich": "bayern munich",
        "bayern munchen": "bayern munich",
        "bayern munich munich": "bayern munich",
        "vfb stuttgart": "stuttgart",
        "stuttgart": "stuttgart",
        "fc koln": "koln",
        "1 koln": "koln",
        "cologne": "koln",
        "borussia monchengladbach": "m gladbach",
        "monchengladbach": "m gladbach",
        "m gladbach": "m gladbach",
        "sporting cp": "sporting lisbon",
        "sporting clube de portugal": "sporting lisbon",
        "sporting lisbon": "sporting lisbon",
        "sp lisbon": "sporting lisbon",
        "sporting gijon": "sporting gijon",
        "sporting de gijon": "sporting gijon",
        "sp gijon": "sporting gijon",
        "buyuksehyr": "basaksehir",
        "buyuksehir": "basaksehir",
        "buyuksehir belediyespor": "basaksehir",
        "istanbul buyuksehir": "basaksehir",
        "istanbul buyuksehir belediyespor": "basaksehir",
        "istanbul basaksehir": "basaksehir",
        "istanbul basaksehir fk": "basaksehir",
        "basaksehir": "basaksehir",
        "basaksehir fk": "basaksehir",
        "stade lavallois": "laval",
        "stade laval": "laval",
        "laval": "laval",
        "red star fc 93": "red star",
        "red star fc": "red star",
        "red star": "red star",
        "rodez aveyron": "rodez",
        "rodez aveyron football": "rodez",
        "rodez": "rodez",
        "real sociedad b": "sociedad b",
        "real sociedad ii": "sociedad b",
        "real sociedad san sebastian b": "sociedad b",
        "sociedad b": "sociedad b",
        "cd tenerife": "tenerife",
        "tenerife": "tenerife",
    }

    return aliases.get(a, a) == aliases.get(b, b)


# ============================================================
# DATAS / NÚMEROS
# ============================================================

def parse_db_date(value: object) -> date | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.casefold() in {
        "nan",
        "none",
        "<na>",
    }:
        return None

    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_football_data_date(value: object) -> date | None:
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


def parse_iso_local_date(value: object) -> date | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        APP_TIMEZONE
    ).date()


def season_start_for(reference_date: date) -> date:
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


def safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number != number:
        return None

    return number


def safe_int(value: object) -> int | None:
    number = safe_float(value)
    if number is None:
        return None

    rounded = int(number)
    if abs(number - rounded) > 1e-9:
        return None

    return rounded


def safe_odd(value: object) -> float | None:
    number = safe_float(value)
    if number is None or number <= 1.0:
        return None
    return number


def calculate_clv(
    open_odd: object,
    close_odd: object,
) -> float | None:
    open_value = safe_odd(open_odd)
    close_value = safe_odd(close_odd)

    if open_value is None or close_value is None:
        return None

    return open_value / close_value - 1.0


# ============================================================
# HTTP
# ============================================================

def get_bytes(
    url: str,
) -> tuple[bytes, dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 FootballAI/Cloud",
            "Accept": "application/json,text/csv,*/*",
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
                for key, value in response.headers.items()
            }
            return content, headers

    except HTTPError as error:
        try:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            detail = ""

        raise RuntimeError(
            f"HTTP {error.code}: {detail[:400]}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Erro de ligação: {error.reason}"
        ) from error


def get_json(
    url: str,
) -> tuple[object, dict[str, str]]:
    content, headers = get_bytes(url)

    try:
        payload = json.loads(
            content.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            "Resposta JSON inválida."
        ) from error

    return payload, headers


def is_quota_error(error: Exception) -> bool:
    text = str(error).casefold()
    return any(
        marker.casefold() in text
        for marker in QUOTA_ERROR_MARKERS
    )


# ============================================================
# FOOTBALL-DATA.CO.UK
# ============================================================

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


def extract_football_data_closing_odds(
    row: pd.Series,
) -> tuple[float | None, float | None, str | None]:
    pairs = (
        (
            "AvgC>2.5",
            "AvgC<2.5",
            RECOVERED_CLOSE_SOURCE_AVG,
        ),
        (
            "B365C>2.5",
            "B365C<2.5",
            RECOVERED_CLOSE_SOURCE_B365,
        ),
    )

    for over_column, under_column, source in pairs:
        over_value = safe_odd(
            row.get(over_column)
        )
        under_value = safe_odd(
            row.get(under_column)
        )

        if (
            over_value is not None
            and under_value is not None
        ):
            return (
                over_value,
                under_value,
                source,
            )

    return None, None, None


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
        content, _ = get_bytes(url)
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

        if home_team is None or away_team is None:
            continue

        (
            close_over,
            close_under,
            closing_source,
        ) = extract_football_data_closing_odds(row)

        results.append(
            {
                "date": match_date,
                "league": canonical_league,
                "home_team": str(home_team),
                "away_team": str(away_team),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "odd_over25_close": close_over,
                "odd_under25_close": close_under,
                "closing_source": closing_source,
            }
        )

    return results, None


def find_simple_result(
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

    return matches[0] if len(matches) == 1 else None


# ============================================================
# RECUPERAR CLOSE PÓS-JOGO
# ============================================================

def backfill_existing_clv(
    connection,
    table: str,
    row: dict,
) -> bool:
    if safe_float(row.get("clv")) is not None:
        return False

    clv = calculate_clv(
        row.get("odd_over25_open"),
        row.get("odd_over25_close"),
    )

    if clv is None:
        return False

    connection.execute(
        f"""
        UPDATE {table}
        SET clv = ?
        WHERE id = ?
        """,
        (
            clv,
            int(row["id"]),
        ),
    )

    row["clv"] = clv
    return True


def recover_closing_from_football_data(
    connection,
    table: str,
    row: dict,
    result: dict,
) -> tuple[bool, bool]:
    existing_over = safe_odd(
        row.get("odd_over25_close")
    )
    existing_under = safe_odd(
        row.get("odd_under25_close")
    )

    if (
        existing_over is not None
        and existing_under is not None
    ):
        clv_backfilled = backfill_existing_clv(
            connection,
            table,
            row,
        )
        return False, clv_backfilled

    recovered_over = safe_odd(
        result.get("odd_over25_close")
    )
    recovered_under = safe_odd(
        result.get("odd_under25_close")
    )
    source = result.get("closing_source")

    if (
        recovered_over is None
        or recovered_under is None
        or not source
    ):
        return False, False

    clv = calculate_clv(
        row.get("odd_over25_open"),
        recovered_over,
    )

    observed_at = datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()

    connection.execute(
        f"""
        UPDATE {table}
        SET
            odd_over25_close = ?,
            odd_under25_close = ?,
            closing_observed_at = ?,
            closing_minutes_before_kickoff = NULL,
            closing_source = ?,
            clv = ?
        WHERE id = ?
          AND (
                odd_over25_close IS NULL
             OR odd_over25_close <= 1.0
             OR odd_under25_close IS NULL
             OR odd_under25_close <= 1.0
          )
        """,
        (
            recovered_over,
            recovered_under,
            observed_at,
            str(source),
            clv,
            int(row["id"]),
        ),
    )

    row["odd_over25_close"] = recovered_over
    row["odd_under25_close"] = recovered_under
    row["closing_observed_at"] = observed_at
    row["closing_minutes_before_kickoff"] = None
    row["closing_source"] = str(source)
    row["clv"] = clv

    return True, clv is not None


# ============================================================
# ESPN — FALLBACK SEM API KEY
# ============================================================

def fetch_espn_scoreboard(
    league_slug: str,
    fixture_date: date,
) -> tuple[list[dict], str | None]:
    date_code = fixture_date.strftime("%Y%m%d")
    url = (
        f"{ESPN_BASE_URL}/"
        f"{league_slug}/scoreboard?dates={date_code}"
    )

    try:
        payload, _ = get_json(url)
    except Exception as error:
        return [], str(error)

    if not isinstance(payload, dict):
        return [], "resposta inesperada"

    events = payload.get("events", [])
    if not isinstance(events, list):
        return [], "campo events inválido"

    return [
        event
        for event in events
        if isinstance(event, dict)
    ], None


def espn_event_completed(event: dict) -> bool:
    status = event.get("status")
    if not isinstance(status, dict):
        return False

    status_type = status.get("type")
    if not isinstance(status_type, dict):
        return False

    return bool(status_type.get("completed"))


def espn_competitors(event: dict) -> list[dict]:
    competitions = event.get("competitions", [])
    if not isinstance(competitions, list) or not competitions:
        return []

    competition = competitions[0]
    if not isinstance(competition, dict):
        return []

    competitors = competition.get("competitors", [])
    if not isinstance(competitors, list):
        return []

    return [
        item
        for item in competitors
        if isinstance(item, dict)
    ]


def espn_team_name(competitor: dict) -> str | None:
    team = competitor.get("team")
    if not isinstance(team, dict):
        return None

    for key in (
        "displayName",
        "shortDisplayName",
        "name",
        "location",
    ):
        value = team.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return None


def find_espn_event(
    events: list[dict],
    canonical_league: str,
    fixture_date: date,
    home_team: str,
    away_team: str,
) -> dict | None:
    matches: list[dict] = []

    for event in events:
        if not espn_event_completed(event):
            continue

        event_date = parse_iso_local_date(
            event.get("date")
        )
        if event_date != fixture_date:
            continue

        competitors = espn_competitors(event)
        home_candidates = [
            item
            for item in competitors
            if str(item.get("homeAway", "")).casefold()
            == "home"
        ]
        away_candidates = [
            item
            for item in competitors
            if str(item.get("homeAway", "")).casefold()
            == "away"
        ]

        if (
            len(home_candidates) != 1
            or len(away_candidates) != 1
        ):
            continue

        api_home = espn_team_name(
            home_candidates[0]
        )
        api_away = espn_team_name(
            away_candidates[0]
        )

        if not team_names_match(
            canonical_league,
            api_home,
            home_team,
        ):
            continue

        if not team_names_match(
            canonical_league,
            api_away,
            away_team,
        ):
            continue

        matches.append(event)

    return matches[0] if len(matches) == 1 else None


def extract_espn_score(
    event: dict,
    side: str,
) -> int | None:
    candidates = [
        item
        for item in espn_competitors(event)
        if str(item.get("homeAway", "")).casefold()
        == side.casefold()
    ]

    if len(candidates) != 1:
        return None

    return safe_int(
        candidates[0].get("score")
    )


# ============================================================
# THE ODDS API — ÚLTIMO FALLBACK
# ============================================================

def odds_event_local_date(event: dict) -> date | None:
    return parse_iso_local_date(
        event.get("commence_time")
    )


def find_odds_api_event(
    events: list[dict],
    league: str,
    fixture_date: date,
    home_team: str,
    away_team: str,
) -> dict | None:
    matches: list[dict] = []

    for event in events:
        if not isinstance(event, dict):
            continue

        if not event.get("completed"):
            continue

        if odds_event_local_date(event) != fixture_date:
            continue

        if not team_names_match(
            league,
            event.get("home_team"),
            home_team,
        ):
            continue

        if not team_names_match(
            league,
            event.get("away_team"),
            away_team,
        ):
            continue

        matches.append(event)

    return matches[0] if len(matches) == 1 else None


def extract_odds_api_score(
    event: dict,
    league: str,
    team_name: str,
) -> int | None:
    scores = event.get("scores")
    if not isinstance(scores, list):
        return None

    values: list[int] = []

    for item in scores:
        if not isinstance(item, dict):
            continue

        if not team_names_match(
            league,
            item.get("name"),
            team_name,
        ):
            continue

        score = safe_int(
            item.get("score")
        )
        if score is not None:
            values.append(score)

    return values[0] if len(values) == 1 else None


# ============================================================
# RESOLVER TRADE
# ============================================================

def calculate_trade_values(
    row: dict,
    home_goals: int,
    away_goals: int,
) -> tuple[int, float, float | None]:
    result_over25 = int(
        home_goals + away_goals > 2
    )

    try:
        signal = int(
            row.get("signal") or 0
        )
    except (TypeError, ValueError):
        signal = 0

    try:
        planned_stake = float(
            row.get("planned_stake") or 0
        )
    except (TypeError, ValueError):
        planned_stake = 0.0

    open_odd = safe_odd(
        row.get("odd_over25_open")
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

    clv = calculate_clv(
        row.get("odd_over25_open"),
        row.get("odd_over25_close"),
    )

    return result_over25, profit, clv


def update_trade(
    connection,
    table: str,
    row: dict,
    columns: set[str],
    home_goals: int,
    away_goals: int,
) -> float | None:
    result_over25, profit, clv = (
        calculate_trade_values(
            row,
            home_goals,
            away_goals,
        )
    )

    timestamp = datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()

    if "updated_at" in columns:
        connection.execute(
            f"""
            UPDATE {table}
            SET
                status = 'RESOLVED',
                home_goals = ?,
                away_goals = ?,
                result_over25 = ?,
                profit = ?,
                clv = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                home_goals,
                away_goals,
                result_over25,
                profit,
                clv,
                timestamp,
                int(row["id"]),
            ),
        )
    else:
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

    row["status"] = "RESOLVED"
    row["home_goals"] = home_goals
    row["away_goals"] = away_goals
    row["result_over25"] = result_over25
    row["profit"] = profit
    row["clv"] = clv

    return clv


# ============================================================
# MAIN
# ============================================================

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
    print(f"Época Football-Data       : {season_code}")
    print(f"Desde                     : {season_start.isoformat()}")

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
            ("closing_observed_at", "TEXT"),
            ("closing_minutes_before_kickoff", "REAL"),
            ("closing_source", "TEXT"),
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
            required_columns - columns
        )

        if missing_columns:
            print(
                "❌ Colunas em falta: "
                + ", ".join(sorted(missing_columns))
            )
            raise SystemExit(1)

        season_rows = fetch_dicts(
            connection,
            f"""
            SELECT *
            FROM {table}
            WHERE date({fixture_date_column}) <= date(?)
              AND date({fixture_date_column}) >= date(?)
            ORDER BY {fixture_date_column}, id
            """,
            (
                today.isoformat(),
                season_start.isoformat(),
            ),
        )

        rows = [
            row
            for row in season_rows
            if str(
                row.get("status") or "OPEN"
            ).strip().upper()
            != "RESOLVED"
        ]

        print(f"Registos da época         : {len(season_rows)}")
        print(f"Registos candidatos       : {len(rows)}")

        metadata_by_id: dict[int, dict[str, object]] = {}
        required_football_data: dict[str, str] = {}

        for row in season_rows:
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
            espn_slug = ESPN_LEAGUE_MAP.get(
                canonical_league
            )
            sport_key = LEAGUE_SPORT_KEY.get(
                canonical_league
            )

            metadata_by_id[row_id] = {
                "league": canonical_league,
                "fixture_date": fixture_date,
                "division": division,
                "espn_slug": espn_slug,
                "sport_key": sport_key,
            }

            if (
                isinstance(canonical_league, str)
                and division
            ):
                required_football_data[
                    canonical_league
                ] = division

        resolved_ids: set[int] = set()
        resolved_football = 0
        resolved_espn = 0
        resolved_odds_api = 0
        with_clv = 0
        without_clv = 0
        recovered_closes = 0
        clv_backfilled = 0

        # ====================================================
        # 1) FOOTBALL-DATA — RESULTADOS + CLOSE RECUPERADA
        # ====================================================

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
                    f"{division} / {canonical_league}: {error}"
                )
                print(
                    f"  ⚠ {division} {canonical_league}: falhou"
                )
                continue

            football_results.extend(results)
            with_close = sum(
                1
                for result in results
                if safe_odd(
                    result.get("odd_over25_close")
                ) is not None
                and safe_odd(
                    result.get("odd_under25_close")
                ) is not None
            )

            print(
                f"  ✅ {division} {canonical_league}: "
                f"{len(results)} resultados | "
                f"{with_close} com closing 2.5"
            )

        print(
            f"Resultados Football-Data  : {len(football_results)}"
        )

        # Primeiro recuperamos CLOSE em qualquer trade da época,
        # inclusive trades já RESOLVED. Nunca substituímos uma CLOSE
        # válida que tenha sido capturada pré-jogo.
        for row in season_rows:
            row_id = int(row["id"])
            metadata = metadata_by_id[row_id]
            canonical_league = metadata["league"]
            fixture_date = metadata["fixture_date"]

            if (
                not isinstance(canonical_league, str)
                or not isinstance(fixture_date, date)
            ):
                continue

            result = find_simple_result(
                football_results,
                canonical_league,
                fixture_date,
                str(row.get("home_team", "")),
                str(row.get("away_team", "")),
            )

            if result is None:
                if backfill_existing_clv(
                    connection,
                    table,
                    row,
                ):
                    clv_backfilled += 1
                continue

            recovered, got_clv = (
                recover_closing_from_football_data(
                    connection,
                    table,
                    row,
                    result,
                )
            )

            if recovered:
                recovered_closes += 1
                clv_backfilled += int(got_clv)

                print(
                    f"🟦 [CLOSE recuperada] "
                    f"{fixture_date.isoformat()} "
                    f"{row.get('home_team')} vs "
                    f"{row.get('away_team')} "
                    f"| Over {row.get('odd_over25_close'):.3f} "
                    f"| {row.get('closing_source')}"
                )

        connection.commit()

        # Agora resolvemos os trades ainda OPEN através do Football-Data.
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

            result = find_simple_result(
                football_results,
                canonical_league,
                fixture_date,
                str(row.get("home_team", "")),
                str(row.get("away_team", "")),
            )

            if result is None:
                continue

            clv = update_trade(
                connection,
                table,
                row,
                columns,
                int(result["home_goals"]),
                int(result["away_goals"]),
            )

            resolved_ids.add(row_id)
            resolved_football += 1
            with_clv += int(clv is not None)
            without_clv += int(clv is None)

            print(
                f"✅ [Football-Data] {fixture_date.isoformat()} "
                f"{row.get('home_team')} "
                f"{result['home_goals']}-{result['away_goals']} "
                f"{row.get('away_team')}"
            )

        connection.commit()

        # ====================================================
        # 2) ESPN — FALLBACK GRATUITO E SEM QUOTA
        # ====================================================

        espn_cutoff = today - timedelta(
            days=ESPN_RECENT_DAYS
        )

        espn_remaining_rows = [
            row
            for row in rows
            if int(row["id"]) not in resolved_ids
            and isinstance(
                metadata_by_id[int(row["id"])]["fixture_date"],
                date,
            )
            and metadata_by_id[int(row["id"])]["fixture_date"]
            >= espn_cutoff
        ]

        espn_errors: list[str] = []

        if espn_remaining_rows:
            print()
            print("ESPN fallback gratuito:")
            print(
                f"  Jogos recentes por resolver: {len(espn_remaining_rows)}"
            )

            needed_espn: set[tuple[str, date]] = set()

            for row in espn_remaining_rows:
                metadata = metadata_by_id[
                    int(row["id"])
                ]
                slug = metadata["espn_slug"]
                fixture_date = metadata["fixture_date"]

                if (
                    isinstance(slug, str)
                    and isinstance(fixture_date, date)
                ):
                    needed_espn.add(
                        (slug, fixture_date)
                    )

            events_by_key: dict[
                tuple[str, date],
                list[dict],
            ] = {}

            for slug, fixture_date in sorted(
                needed_espn,
                key=lambda item: (
                    item[1],
                    item[0],
                ),
            ):
                events, error = fetch_espn_scoreboard(
                    slug,
                    fixture_date,
                )

                if error is not None:
                    espn_errors.append(
                        f"{slug} {fixture_date}: {error}"
                    )
                    print(
                        f"  ⚠ {slug} {fixture_date}: falhou"
                    )
                    continue

                events_by_key[
                    (slug, fixture_date)
                ] = events
                print(
                    f"  ✅ {slug} {fixture_date}: "
                    f"{len(events)} eventos"
                )

            for row in espn_remaining_rows:
                row_id = int(row["id"])
                if row_id in resolved_ids:
                    continue

                metadata = metadata_by_id[row_id]
                canonical_league = metadata["league"]
                fixture_date = metadata["fixture_date"]
                slug = metadata["espn_slug"]

                if (
                    not isinstance(canonical_league, str)
                    or not isinstance(fixture_date, date)
                    or not isinstance(slug, str)
                ):
                    continue

                event = find_espn_event(
                    events_by_key.get(
                        (slug, fixture_date),
                        [],
                    ),
                    canonical_league,
                    fixture_date,
                    str(row.get("home_team", "")),
                    str(row.get("away_team", "")),
                )

                if event is None:
                    continue

                home_goals = extract_espn_score(
                    event,
                    "home",
                )
                away_goals = extract_espn_score(
                    event,
                    "away",
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

                resolved_ids.add(row_id)
                resolved_espn += 1
                with_clv += int(clv is not None)
                without_clv += int(clv is None)

                print(
                    f"✅ [ESPN] {fixture_date.isoformat()} "
                    f"{row.get('home_team')} "
                    f"{home_goals}-{away_goals} "
                    f"{row.get('away_team')}"
                )

            connection.commit()

        # ====================================================
        # 3) ODDS API — SÓ SE AINDA HOUVER CRÉDITOS
        # ====================================================

        odds_cutoff = today - timedelta(
            days=ODDS_API_DAYS_FROM
        )

        odds_remaining_rows = [
            row
            for row in rows
            if int(row["id"]) not in resolved_ids
            and isinstance(
                metadata_by_id[int(row["id"])]["fixture_date"],
                date,
            )
            and metadata_by_id[int(row["id"])]["fixture_date"]
            >= odds_cutoff
        ]

        api_key = os.getenv(
            "ODDS_API_KEY",
            "",
        ).strip()

        api_quota_exhausted = False
        api_errors: list[str] = []
        credits_remaining: str | None = None

        if api_key and odds_remaining_rows:
            print()
            print("The Odds API — último fallback:")
            print(
                f"  Jogos recentes por resolver: {len(odds_remaining_rows)}"
            )

            needed_sport_keys = {
                metadata_by_id[int(row["id"])]["sport_key"]
                for row in odds_remaining_rows
            }
            needed_sport_keys.discard(None)

            events_by_sport: dict[str, list[dict]] = {}

            for sport_key in sorted(
                str(value)
                for value in needed_sport_keys
            ):
                query = urlencode(
                    {
                        "apiKey": api_key,
                        "daysFrom": ODDS_API_DAYS_FROM,
                        "dateFormat": "iso",
                    }
                )

                url = (
                    f"{ODDS_API_BASE_URL}"
                    f"/sports/{sport_key}"
                    f"/scores/?{query}"
                )

                try:
                    payload, headers = get_json(url)
                except Exception as error:
                    api_errors.append(
                        f"{sport_key}: {error}"
                    )

                    if is_quota_error(error):
                        api_quota_exhausted = True
                        print(
                            "  ⚠ quota esgotada; "
                            "fallback interrompido"
                        )
                        break

                    print(
                        f"  ⚠ {sport_key}: falhou"
                    )
                    continue

                events_by_sport[sport_key] = (
                    payload
                    if isinstance(payload, list)
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

            for row in odds_remaining_rows:
                row_id = int(row["id"])
                if row_id in resolved_ids:
                    continue

                metadata = metadata_by_id[row_id]
                canonical_league = metadata["league"]
                fixture_date = metadata["fixture_date"]
                sport_key = metadata["sport_key"]

                if (
                    not isinstance(canonical_league, str)
                    or not isinstance(fixture_date, date)
                    or not sport_key
                ):
                    continue

                event = find_odds_api_event(
                    events_by_sport.get(
                        str(sport_key),
                        [],
                    ),
                    canonical_league,
                    fixture_date,
                    str(row.get("home_team", "")),
                    str(row.get("away_team", "")),
                )

                if event is None:
                    continue

                home_goals = extract_odds_api_score(
                    event,
                    canonical_league,
                    str(row.get("home_team", "")),
                )
                away_goals = extract_odds_api_score(
                    event,
                    canonical_league,
                    str(row.get("away_team", "")),
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

                resolved_ids.add(row_id)
                resolved_odds_api += 1
                with_clv += int(clv is not None)
                without_clv += int(clv is None)

                print(
                    f"✅ [Odds API] {fixture_date.isoformat()} "
                    f"{row.get('home_team')} "
                    f"{home_goals}-{away_goals} "
                    f"{row.get('away_team')}"
                )

            connection.commit()

        elif not api_key:
            print()
            print(
                "The Odds API: ODDS_API_KEY não configurada."
            )

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
            f"CLOSE recuperadas pós-jogo: {recovered_closes}"
        )
        print(
            f"CLV preenchidos/recalculados: {clv_backfilled}"
        )
        print(
            f"Resolvidos Football-Data  : {resolved_football}"
        )
        print(
            f"Resolvidos ESPN           : {resolved_espn}"
        )
        print(
            f"Resolvidos The Odds API   : {resolved_odds_api}"
        )
        print(
            f"Resolvidos total          : {len(resolved_ids)}"
        )
        print(f"Com CLV válido            : {with_clv}")
        print(f"Sem CLOSE / CLV           : {without_clv}")
        print(f"Ainda por resolver        : {len(remaining)}")

        if credits_remaining is not None:
            print(
                f"Créditos API restantes    : {credits_remaining}"
            )

        if api_quota_exhausted:
            print(
                "Quota Odds API            : ESGOTADA"
            )

        if football_errors:
            print()
            print("Avisos Football-Data:")
            for error in football_errors[:10]:
                print(f"  - {error}")

        if espn_errors:
            print()
            print("Avisos ESPN:")
            for error in espn_errors[:10]:
                print(f"  - {error}")

        if api_errors:
            print()
            print("Avisos The Odds API:")
            for error in api_errors[:10]:
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
        print("Qualidade das closing odds:")
        print(
            "  - The Odds API = captura pré-jogo, quando disponível."
        )
        print(
            "  - Football-Data = CLOSE recuperada pós-jogo; "
            "fica identificada em closing_source e nunca substitui "
            "uma CLOSE pré-jogo válida."
        )
        print()
        print(
            "CLV = odd_over25_open / odd_over25_close - 1."
        )
        print(
            "Ordem das fontes de resultados: Football-Data → ESPN → "
            "The Odds API."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
