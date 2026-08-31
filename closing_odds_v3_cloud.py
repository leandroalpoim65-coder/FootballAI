from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import libsql

from ml.paper_trade import normalize_league, normalize_team


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

# The Odds API is the preferred prospective close. Query it only very near
# kickoff to protect credits. API-Football is used twice at most in normal
# operation: one provisional pre-match snapshot and one live pre-kickoff try.
API_FOOTBALL_CAPTURE_WINDOW_MINUTES = 240
ODDS_API_WINDOW_MINUTES = 20
API_FOOTBALL_LIVE_WINDOW_MINUTES = 12
EXCELLENT_CLOSE_MINUTES = 8
API_FOOTBALL_RETRY_NO_DATA_MINUTES = 180
API_FOOTBALL_FINAL_ACCEPTABLE_MINUTES = 60

ODDS_API_SOURCE = "The Odds API / EU average totals 2.5 / pre-kickoff"
API_FOOTBALL_LIVE_SOURCE = "API-Football / live pre-kickoff totals 2.5"
API_FOOTBALL_PREMATCH_SOURCE = "API-Football / provisional pre-match snapshot"

OBS_TABLE = "closing_odds_observations"

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

TEAM_ALIASES = {
    "ein frankfurt": "eintracht frankfurt",
    "eintracht frankfurt": "eintracht frankfurt",
    "1 nurnberg": "nurnberg",
    "nurnberg": "nurnberg",
    "arminia bielefeld": "bielefeld",
    "bielefeld": "bielefeld",
    "nec nijmegen": "nijmegen",
    "nijmegen": "nijmegen",
    "pec zwolle": "zwolle",
    "zwolle": "zwolle",
    "real sociedad": "sociedad",
    "sociedad": "sociedad",
    "rcd espanyol": "espanol",
    "espanyol": "espanol",
    "espanol": "espanol",
    "standard liege": "standard",
    "standard de liege": "standard",
    "standard": "standard",
    "oh leuven": "oud heverlee leuven",
    "oud heverlee leuven": "oud heverlee leuven",
    "caykur rizespor": "rizespor",
    "rizespor": "rizespor",
    "goztepe sk": "goztepe",
    "goztepe": "goztepe",
    "paris saint germain": "psg",
    "paris sg": "psg",
    "psg": "psg",
    "sporting cp": "sporting lisbon",
    "sporting clube de portugal": "sporting lisbon",
    "sporting lisbon": "sporting lisbon",
    "sp lisbon": "sporting lisbon",
    "sporting gijon": "sporting gijon",
    "sporting de gijon": "sporting gijon",
    "sp gijon": "sporting gijon",
    "levadiakos": "levadeiakos",
    "levadeiakos": "levadeiakos",
}


# ============================================================
# DATABASE
# ============================================================

def connect_turso():
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    if not url:
        raise RuntimeError("TURSO_DATABASE_URL não configurada.")
    if not token:
        raise RuntimeError("TURSO_AUTH_TOKEN não configurado.")
    return libsql.connect(database=url, auth_token=token)


def fetch_dicts(connection, sql: str, params: tuple = ()) -> list[dict]:
    cursor = connection.execute(sql, params)
    rows = cursor.fetchall()
    names = [item[0] for item in (cursor.description or [])]
    return [dict(zip(names, row)) for row in rows]


def find_paper_table(connection) -> str | None:
    existing = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
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


def table_columns(connection, table: str) -> set[str]:
    return {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
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
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    columns.add(name)


def ensure_observation_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {OBS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_trade_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            quality TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            minutes_before_kickoff REAL,
            odd_over25 REAL NOT NULL,
            odd_under25 REAL NOT NULL,
            fixture_date TEXT,
            fixture_time TEXT,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_{OBS_TABLE}_dedupe
        ON {OBS_TABLE} (
            paper_trade_id,
            source,
            observed_at,
            odd_over25,
            odd_under25
        )
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{OBS_TABLE}_paper
        ON {OBS_TABLE}(paper_trade_id, observed_at)
        """
    )


def record_observation(
    connection,
    row: dict,
    source: str,
    quality: str,
    observed_at: datetime,
    minutes_before_kickoff: float | None,
    over: float,
    under: float,
) -> None:
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    connection.execute(
        f"""
        INSERT OR IGNORE INTO {OBS_TABLE} (
            paper_trade_id,
            source,
            quality,
            observed_at,
            minutes_before_kickoff,
            odd_over25,
            odd_under25,
            fixture_date,
            fixture_time,
            league,
            home_team,
            away_team,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(row["id"]),
            source,
            quality,
            observed_at.isoformat(timespec="seconds"),
            minutes_before_kickoff,
            over,
            under,
            row.get("fixture_date") or row.get("date"),
            row.get("fixture_time") or row.get("time"),
            row.get("league"),
            row.get("home_team"),
            row.get("away_team"),
            created_at,
        ),
    )


# ============================================================
# GENERIC HELPERS
# ============================================================

def parse_fixture_datetime(date_value, time_value) -> datetime | None:
    if date_value is None:
        return None
    date_text = str(date_value).strip()
    if not date_text:
        return None
    time_text = "00:00" if time_value is None else str(time_value).strip()
    if not time_text:
        time_text = "00:00"

    for candidate in (
        f"{date_text[:10]} {time_text}",
        f"{date_text[:10]} {time_text[:5]}",
    ):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=APP_TIMEZONE)
        return parsed.astimezone(APP_TIMEZONE)
    return None


def parse_api_datetime(value) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(APP_TIMEZONE)


def safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def safe_int(value) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    rounded = int(number)
    return rounded if abs(number - rounded) < 1e-9 else None


def valid_odd(value) -> float | None:
    number = safe_float(value)
    if number is None or number <= 1.0:
        return None
    return number


def http_get_json(
    url: str,
    extra_headers: dict[str, str] | None = None,
) -> tuple[object, dict[str, str]]:
    headers = {
        "User-Agent": "Mozilla/5.0 FootballAI/Cloud",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
            response_headers = {
                key.lower(): value for key, value in response.headers.items()
            }
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {error.code}: {detail[:400]}") from error
    except URLError as error:
        raise RuntimeError(f"Erro de ligação: {error.reason}") from error

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Resposta JSON inválida.") from error
    return payload, response_headers


def is_quota_error(error: Exception) -> bool:
    text = str(error).casefold()
    return any(marker.casefold() in text for marker in QUOTA_ERROR_MARKERS)


def normalize_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text if not unicodedata.combining(char)
    )
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\butd\b", " united ", text)
    text = re.sub(
        r"\b(fc|cf|afc|sc|fk|ac|sv|rcd|cd|club)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_team(league: str, value) -> str:
    normalized = normalize_team(league, value)
    name = normalize_name(normalized)
    return TEAM_ALIASES.get(name, name)


def team_similarity(league: str, left, right) -> float:
    a = canonical_team(league, left)
    b = canonical_team(league, right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def source_rank(source: object) -> int:
    text = str(source or "").casefold()
    if "football-data" in text:
        return 40
    if "the odds api" in text:
        return 30
    if "api-football" in text and "live pre-kickoff" in text:
        return 25
    if "api-football" in text:
        return 10
    return 0


def has_valid_close(row: dict) -> bool:
    return (
        valid_odd(row.get("odd_over25_close")) is not None
        and valid_odd(row.get("odd_under25_close")) is not None
    )


def has_high_quality_close(row: dict) -> bool:
    if not has_valid_close(row):
        return False
    return source_rank(row.get("closing_source")) >= 25


def calculate_clv(open_odd, close_odd) -> float | None:
    open_value = valid_odd(open_odd)
    close_value = valid_odd(close_odd)
    if open_value is None or close_value is None:
        return None
    return open_value / close_value - 1.0


def maybe_set_preferred_close(
    connection,
    table: str,
    columns: set[str],
    row: dict,
    over: float,
    under: float,
    observed_at: datetime,
    minutes_before_kickoff: float,
    source: str,
) -> bool:
    old_valid = has_valid_close(row)
    old_source = row.get("closing_source")
    old_rank = source_rank(old_source) if old_valid else -1
    new_rank = source_rank(source)
    old_minutes = safe_float(row.get("closing_minutes_before_kickoff"))

    replace = not old_valid
    if old_valid:
        if new_rank > old_rank:
            replace = True
        elif new_rank < old_rank:
            replace = False
        elif old_minutes is None:
            replace = True
        else:
            replace = minutes_before_kickoff < old_minutes

    if not replace:
        return False

    clv = calculate_clv(row.get("odd_over25_open"), over)

    assignments = [
        "odd_over25_close = ?",
        "odd_under25_close = ?",
        "closing_observed_at = ?",
        "closing_minutes_before_kickoff = ?",
        "closing_source = ?",
    ]
    params: list[object] = [
        over,
        under,
        observed_at.isoformat(timespec="seconds"),
        minutes_before_kickoff,
        source,
    ]

    if "clv" in columns:
        assignments.append("clv = ?")
        params.append(clv)

    params.append(int(row["id"]))
    connection.execute(
        f"UPDATE {table} SET {', '.join(assignments)} WHERE id = ?",
        tuple(params),
    )

    row["odd_over25_close"] = over
    row["odd_under25_close"] = under
    row["closing_observed_at"] = observed_at.isoformat(timespec="seconds")
    row["closing_minutes_before_kickoff"] = minutes_before_kickoff
    row["closing_source"] = source
    if "clv" in columns:
        row["clv"] = clv
    return True


# ============================================================
# THE ODDS API
# ============================================================

def find_unique_odds_api_event(
    events: list[dict],
    league: str,
    fixture_datetime: datetime,
    home_team,
    away_team,
) -> dict | None:
    candidates: list[tuple[float, dict]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        kickoff = parse_api_datetime(event.get("commence_time"))
        if kickoff is None or kickoff.date() != fixture_datetime.date():
            continue
        if abs((kickoff - fixture_datetime).total_seconds()) > 3 * 3600:
            continue

        hs = team_similarity(league, event.get("home_team"), home_team)
        as_ = team_similarity(league, event.get("away_team"), away_team)
        if hs < 0.78 or as_ < 0.78:
            continue
        candidates.append((hs + as_, event))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) >= 2 and candidates[0][0] - candidates[1][0] < 0.08:
        return None
    return candidates[0][1]


def extract_odds_api_totals(event: dict) -> tuple[float | None, float | None]:
    over_prices: list[float] = []
    under_prices: list[float] = []

    for bookmaker in event.get("bookmakers", []) or []:
        if not isinstance(bookmaker, dict):
            continue
        for market in bookmaker.get("markets", []) or []:
            if not isinstance(market, dict) or market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []) or []:
                if not isinstance(outcome, dict):
                    continue
                name = str(outcome.get("name", "")).strip().casefold()
                try:
                    point = float(outcome.get("point"))
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                if abs(point - 2.5) > 1e-9 or price <= 1.0:
                    continue
                if name == "over":
                    over_prices.append(price)
                elif name == "under":
                    under_prices.append(price)

    if not over_prices or not under_prices:
        return None, None
    return (
        float(sum(over_prices) / len(over_prices)),
        float(sum(under_prices) / len(under_prices)),
    )


def fetch_odds_api(candidates: list[dict], api_key: str):
    sport_keys = sorted(
        {str(item["sport_key"]) for item in candidates if item.get("sport_key")}
    )
    events_by_sport: dict[str, list[dict]] = {}
    successful: set[str] = set()
    used = 0
    remaining = None
    quota_exhausted = False

    for sport_key in sport_keys:
        query = urlencode(
            {
                "apiKey": api_key,
                "regions": "eu",
                "markets": "totals",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            }
        )
        url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds/?{query}"
        try:
            payload, headers = http_get_json(url)
        except Exception as error:
            print(f"⚠ The Odds API {sport_key}: {error}")
            if is_quota_error(error):
                quota_exhausted = True
                print("⚠ The Odds API sem créditos; passa às fontes API-Football.")
                break
            continue

        used += 1
        if not isinstance(payload, list):
            continue
        events_by_sport[sport_key] = [
            event for event in payload if isinstance(event, dict)
        ]
        successful.add(sport_key)
        remaining = headers.get("x-requests-remaining", remaining)
        print(
            f"✅ The Odds API {sport_key}: "
            f"{len(events_by_sport[sport_key])} eventos"
        )

    return events_by_sport, successful, used, remaining, quota_exhausted


# ============================================================
# API-FOOTBALL
# ============================================================

def api_football_get_json(
    path: str,
    params: dict[str, object],
    api_key: str,
) -> tuple[dict, dict[str, str]]:
    url = f"{API_FOOTBALL_BASE_URL}{path}?{urlencode(params)}"
    payload, headers = http_get_json(
        url,
        {"x-apisports-key": api_key},
    )
    if not isinstance(payload, dict):
        raise RuntimeError("API-Football devolveu resposta inesperada.")
    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"API-Football: {errors}")
    return payload, headers


def fetch_api_football_fixtures(
    fixture_date: str,
    api_key: str,
) -> tuple[list[dict], int, str | None]:
    payload, headers = api_football_get_json(
        "/fixtures",
        {"date": fixture_date, "timezone": "Europe/Lisbon"},
        api_key,
    )
    response = payload.get("response", [])
    fixtures = (
        [item for item in response if isinstance(item, dict)]
        if isinstance(response, list)
        else []
    )
    remaining = headers.get(
        "x-ratelimit-requests-remaining",
        headers.get("x-ratelimit-remaining"),
    )
    return fixtures, 1, remaining


def fixture_details(
    item: dict,
) -> tuple[int | None, datetime | None, str | None, str | None]:
    fixture = item.get("fixture")
    teams = item.get("teams")
    if not isinstance(fixture, dict) or not isinstance(teams, dict):
        return None, None, None, None

    fixture_id = safe_int(fixture.get("id"))
    kickoff = parse_api_datetime(fixture.get("date"))
    home = teams.get("home")
    away = teams.get("away")
    home_name = (
        str(home.get("name")).strip()
        if isinstance(home, dict) and home.get("name")
        else None
    )
    away_name = (
        str(away.get("name")).strip()
        if isinstance(away, dict) and away.get("name")
        else None
    )
    return fixture_id, kickoff, home_name, away_name


def find_api_football_fixture(
    fixtures: list[dict],
    league: str,
    fixture_datetime: datetime,
    home_team,
    away_team,
) -> dict | None:
    candidates: list[tuple[float, dict]] = []
    for item in fixtures:
        fixture_id, kickoff, api_home, api_away = fixture_details(item)
        if fixture_id is None or kickoff is None:
            continue
        if kickoff.date() != fixture_datetime.date():
            continue
        if abs((kickoff - fixture_datetime).total_seconds()) > 3 * 3600:
            continue

        hs = team_similarity(league, api_home, home_team)
        as_ = team_similarity(league, api_away, away_team)
        if hs < 0.76 or as_ < 0.76:
            continue
        candidates.append((hs + as_, item))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) >= 2 and candidates[0][0] - candidates[1][0] < 0.08:
        return None
    return candidates[0][1]


def resolve_api_football_fixture_ids(
    connection,
    table: str,
    candidates: list[dict],
    api_key: str,
) -> tuple[int, str | None]:
    need_lookup = [
        item
        for item in candidates
        if safe_int(item["row"].get("api_football_fixture_id")) is None
    ]
    if not need_lookup:
        return 0, None

    used = 0
    remaining = None
    dates_needed = sorted(
        {
            item["fixture_datetime"].strftime("%Y-%m-%d")
            for item in need_lookup
        }
    )
    fixtures_by_date: dict[str, list[dict]] = {}

    for fixture_date in dates_needed:
        try:
            fixtures, calls, quota_remaining = fetch_api_football_fixtures(
                fixture_date,
                api_key,
            )
        except Exception as error:
            print(f"⚠ API-Football fixtures {fixture_date}: {error}")
            continue
        used += calls
        remaining = quota_remaining or remaining
        fixtures_by_date[fixture_date] = fixtures
        print(
            f"✅ API-Football fixtures {fixture_date}: "
            f"{len(fixtures)} jogos"
        )

    for item in need_lookup:
        row = item["row"]
        fixture_date = item["fixture_datetime"].strftime("%Y-%m-%d")
        match = find_api_football_fixture(
            fixtures_by_date.get(fixture_date, []),
            item["league"],
            item["fixture_datetime"],
            row.get("home_team"),
            row.get("away_team"),
        )
        if match is None:
            continue
        fixture_id, _, _, _ = fixture_details(match)
        if fixture_id is None:
            continue
        connection.execute(
            f"UPDATE {table} SET api_football_fixture_id = ? WHERE id = ?",
            (fixture_id, int(row["id"])),
        )
        row["api_football_fixture_id"] = fixture_id

    connection.commit()
    return used, remaining


def is_over_25_text(value: object) -> bool:
    text = str(value or "").strip().casefold().replace(",", ".")
    return bool(re.search(r"\bover\s*2\.?5\b", text))


def is_under_25_text(value: object) -> bool:
    text = str(value or "").strip().casefold().replace(",", ".")
    return bool(re.search(r"\bunder\s*2\.?5\b", text))


def extract_api_football_prematch_totals(
    payload: dict,
) -> tuple[float | None, float | None, datetime | None]:
    response = payload.get("response", [])
    if not isinstance(response, list):
        return None, None, None

    over_prices: list[float] = []
    under_prices: list[float] = []
    update_times: list[datetime] = []

    for entry in response:
        if not isinstance(entry, dict):
            continue
        update_time = parse_api_datetime(entry.get("update"))
        if update_time is not None:
            update_times.append(update_time)

        for bookmaker in entry.get("bookmakers", []) or []:
            if not isinstance(bookmaker, dict):
                continue
            bookmaker_over = None
            bookmaker_under = None
            for bet in bookmaker.get("bets", []) or []:
                if not isinstance(bet, dict):
                    continue
                for value in bet.get("values", []) or []:
                    if not isinstance(value, dict):
                        continue
                    odd = valid_odd(value.get("odd"))
                    if odd is None:
                        continue
                    if bookmaker_over is None and is_over_25_text(value.get("value")):
                        bookmaker_over = odd
                    elif bookmaker_under is None and is_under_25_text(value.get("value")):
                        bookmaker_under = odd
            if bookmaker_over is not None:
                over_prices.append(bookmaker_over)
            if bookmaker_under is not None:
                under_prices.append(bookmaker_under)

    if not over_prices or not under_prices:
        return None, None, max(update_times) if update_times else None
    return (
        float(sum(over_prices) / len(over_prices)),
        float(sum(under_prices) / len(under_prices)),
        max(update_times) if update_times else None,
    )


def fetch_api_football_prematch_odds(
    fixture_id: int,
    api_key: str,
) -> tuple[float | None, float | None, datetime | None, int, str | None]:
    payload, headers = api_football_get_json(
        "/odds",
        {"fixture": fixture_id},
        api_key,
    )
    over, under, snapshot = extract_api_football_prematch_totals(payload)
    remaining = headers.get(
        "x-ratelimit-requests-remaining",
        headers.get("x-ratelimit-remaining"),
    )
    return over, under, snapshot, 1, remaining


def is_goal_total_live_bet(name: object) -> bool:
    text = normalize_name(name)
    if not text:
        return False
    banned = (
        "corner",
        "card",
        "booking",
        "offside",
        "shot",
        "throw",
        "team",
        "home",
        "away",
        "half",
        "period",
        "player",
    )
    if any(word in text for word in banned):
        return False
    return "over under" in text or text in {"goals", "total goals", "over under line"}


def extract_api_football_live_totals(
    payload: dict,
) -> tuple[float | None, float | None, datetime | None]:
    response = payload.get("response", [])
    if not isinstance(response, list):
        return None, None, None

    over_candidates: list[tuple[int, float]] = []
    under_candidates: list[tuple[int, float]] = []
    update_times: list[datetime] = []

    for entry in response:
        if not isinstance(entry, dict):
            continue
        update_time = parse_api_datetime(entry.get("update"))
        if update_time is not None:
            update_times.append(update_time)

        status = entry.get("status")
        if isinstance(status, dict) and status.get("blocked"):
            continue

        for bet in entry.get("odds", []) or []:
            if not isinstance(bet, dict) or not is_goal_total_live_bet(bet.get("name")):
                continue
            for value in bet.get("values", []) or []:
                if not isinstance(value, dict) or value.get("suspended"):
                    continue
                odd = valid_odd(value.get("odd"))
                if odd is None:
                    continue

                value_text = str(value.get("value") or "").strip().casefold()
                handicap = safe_float(value.get("handicap"))
                side = None
                point = handicap

                if value_text == "over":
                    side = "over"
                elif value_text == "under":
                    side = "under"
                elif is_over_25_text(value_text):
                    side = "over"
                    point = 2.5
                elif is_under_25_text(value_text):
                    side = "under"
                    point = 2.5

                if side is None or point is None or abs(point - 2.5) > 1e-9:
                    continue

                priority = 2 if value.get("main") is True else 1
                if side == "over":
                    over_candidates.append((priority, odd))
                else:
                    under_candidates.append((priority, odd))

    if not over_candidates or not under_candidates:
        return None, None, max(update_times) if update_times else None

    max_over_priority = max(item[0] for item in over_candidates)
    max_under_priority = max(item[0] for item in under_candidates)
    overs = [item[1] for item in over_candidates if item[0] == max_over_priority]
    unders = [item[1] for item in under_candidates if item[0] == max_under_priority]

    return (
        float(sum(overs) / len(overs)),
        float(sum(unders) / len(unders)),
        max(update_times) if update_times else None,
    )


def fetch_api_football_live_odds(
    fixture_id: int,
    api_key: str,
) -> tuple[float | None, float | None, datetime | None, int, str | None]:
    payload, headers = api_football_get_json(
        "/odds/live",
        {"fixture": fixture_id},
        api_key,
    )
    over, under, snapshot = extract_api_football_live_totals(payload)
    remaining = headers.get(
        "x-ratelimit-requests-remaining",
        headers.get("x-ratelimit-remaining"),
    )
    return over, under, snapshot, 1, remaining


def mark_checked(
    connection,
    table: str,
    row_id: int,
    column: str,
    when: datetime,
) -> None:
    connection.execute(
        f"UPDATE {table} SET {column} = ? WHERE id = ?",
        (when.isoformat(timespec="seconds"), row_id),
    )


def should_query_prematch(row: dict, now: datetime, minutes_to_start: float) -> bool:
    if has_high_quality_close(row):
        return False

    source = str(row.get("closing_source") or "").casefold()
    has_provisional = has_valid_close(row) and "api-football" in source
    if has_provisional:
        old_minutes = safe_float(row.get("closing_minutes_before_kickoff"))
        # A valid provisional snapshot is enough as insurance. Near kickoff we
        # use /odds/live instead of repeatedly calling /odds.
        if old_minutes is not None:
            return False

    checked = parse_api_datetime(row.get("closing_api_football_checked_at"))
    if checked is None:
        return True
    return (now - checked).total_seconds() / 60.0 >= API_FOOTBALL_RETRY_NO_DATA_MINUTES


def should_query_live(row: dict, now: datetime, minutes_to_start: float) -> bool:
    if minutes_to_start < 0 or minutes_to_start > API_FOOTBALL_LIVE_WINDOW_MINUTES:
        return False

    source = str(row.get("closing_source") or "").casefold()
    old_minutes = safe_float(row.get("closing_minutes_before_kickoff"))

    if "the odds api" in source and old_minutes is not None and old_minutes <= EXCELLENT_CLOSE_MINUTES:
        return False
    if "api-football" in source and "live pre-kickoff" in source:
        if old_minutes is not None and old_minutes <= EXCELLENT_CLOSE_MINUTES:
            return False

    checked = parse_api_datetime(row.get("closing_api_football_live_checked_at"))
    if checked is None:
        return True
    return (now - checked).total_seconds() / 60.0 >= 8


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — CLOSING ODDS V3")
    print("==========================================")
    print()

    odds_api_key = os.getenv("ODDS_API_KEY", "").strip()
    api_football_key = os.getenv("API_FOOTBALL_KEY", "").strip()

    connection = connect_turso()
    try:
        table = find_paper_table(connection)
        if table is None:
            raise RuntimeError("Tabela de Paper Trade não encontrada.")

        columns = table_columns(connection, table)
        for name, sql_type in (
            ("odd_over25_close", "REAL"),
            ("odd_under25_close", "REAL"),
            ("closing_observed_at", "TEXT"),
            ("closing_minutes_before_kickoff", "REAL"),
            ("closing_source", "TEXT"),
            ("closing_api_football_checked_at", "TEXT"),
            ("closing_api_football_live_checked_at", "TEXT"),
            ("api_football_fixture_id", "INTEGER"),
        ):
            add_column_if_missing(connection, table, columns, name, sql_type)

        ensure_observation_table(connection)
        connection.commit()

        date_col = (
            "fixture_date"
            if "fixture_date" in columns
            else "date"
            if "date" in columns
            else None
        )
        time_col = (
            "fixture_time"
            if "fixture_time" in columns
            else "time"
            if "time" in columns
            else None
        )
        if date_col is None or time_col is None:
            raise RuntimeError("A tabela não tem data/hora de fixture.")

        now = datetime.now(APP_TIMEZONE)
        rows = fetch_dicts(
            connection,
            f"""
            SELECT *
            FROM {table}
            WHERE COALESCE(status, 'OPEN') = 'OPEN'
            ORDER BY {date_col}, {time_col}, id
            """,
        )

        candidates: list[dict] = []
        skipped_past = 0
        skipped_far = 0
        skipped_excellent = 0
        skipped_invalid = 0

        for row in rows:
            fixture_datetime = parse_fixture_datetime(
                row.get(date_col),
                row.get(time_col),
            )
            if fixture_datetime is None:
                skipped_invalid += 1
                continue

            minutes_to_start = (
                fixture_datetime - now
            ).total_seconds() / 60.0

            if minutes_to_start < 0:
                skipped_past += 1
                continue
            if minutes_to_start > API_FOOTBALL_CAPTURE_WINDOW_MINUTES:
                skipped_far += 1
                continue

            if has_high_quality_close(row):
                old_minutes = safe_float(row.get("closing_minutes_before_kickoff"))
                if old_minutes is not None and old_minutes <= EXCELLENT_CLOSE_MINUTES:
                    skipped_excellent += 1
                    continue

            league = normalize_league(row.get("league"))
            if not league:
                skipped_invalid += 1
                continue

            candidates.append(
                {
                    "row": row,
                    "fixture_datetime": fixture_datetime,
                    "minutes_to_start": float(minutes_to_start),
                    "league": str(league),
                    "sport_key": LEAGUE_SPORT_KEY.get(str(league)),
                }
            )

        print("Base de dados             : Turso Cloud")
        print("Timezone                  : Europe/Lisbon")
        print(
            f"Janela seguro API-Football: 0-{API_FOOTBALL_CAPTURE_WINDOW_MINUTES} min"
        )
        print(f"Janela The Odds API       : 0-{ODDS_API_WINDOW_MINUTES} min")
        print(
            f"Janela API-Football live  : 0-{API_FOOTBALL_LIVE_WINDOW_MINUTES} min"
        )
        print(
            "The Odds API              : "
            + ("✅ configurada" if odds_api_key else "⚪ sem chave")
        )
        print(
            "API-Football              : "
            + ("✅ configurada" if api_football_key else "⚪ sem chave")
        )
        print(f"Paper Trades OPEN         : {len(rows)}")
        print(f"Candidatos                : {len(candidates)}")
        print(f"Kickoff já passou         : {skipped_past}")
        print(f"Ainda longe               : {skipped_far}")
        print(f"CLOSE excelente preservada: {skipped_excellent}")
        if skipped_invalid:
            print(f"Data/liga inválida        : {skipped_invalid}")

        if not candidates:
            print("✅ Nenhum jogo precisa de refresh agora.")
            return

        print()
        print("Jogos candidatos:")
        for item in candidates:
            row = item["row"]
            print(
                f"  - {row.get('home_team')} vs {row.get('away_team')} "
                f"| {item['league']} | {item['minutes_to_start']:.0f} min"
            )

        api_used = 0
        api_remaining = None
        api_errors: list[str] = []

        # Resolve and persist API-Football fixture ids early. This lets the
        # final live query avoid an extra fixture lookup at kickoff.
        if api_football_key:
            used, remaining = resolve_api_football_fixture_ids(
                connection,
                table,
                candidates,
                api_football_key,
            )
            api_used += used
            api_remaining = remaining or api_remaining

        preferred_updates = 0
        recorded_observations = 0
        odds_used = 0
        odds_remaining = None
        odds_quota_exhausted = False

        # ----------------------------------------------------
        # 1) The Odds API: only near kickoff
        # ----------------------------------------------------
        odds_candidates = [
            item
            for item in candidates
            if item["minutes_to_start"] <= ODDS_API_WINDOW_MINUTES
            and item.get("sport_key")
        ]

        if odds_candidates and odds_api_key:
            print()
            print("1) The Odds API — close preferida:")
            (
                events_by_sport,
                successful_sports,
                odds_used,
                odds_remaining,
                odds_quota_exhausted,
            ) = fetch_odds_api(odds_candidates, odds_api_key)

            for item in odds_candidates:
                row = item["row"]
                sport_key = item.get("sport_key")
                if not sport_key or sport_key not in successful_sports:
                    continue

                event = find_unique_odds_api_event(
                    events_by_sport.get(str(sport_key), []),
                    item["league"],
                    item["fixture_datetime"],
                    row.get("home_team"),
                    row.get("away_team"),
                )
                if event is None:
                    continue

                over, under = extract_odds_api_totals(event)
                if over is None or under is None:
                    continue

                record_observation(
                    connection,
                    row,
                    ODDS_API_SOURCE,
                    "PRE_GAME_HIGH",
                    now,
                    item["minutes_to_start"],
                    over,
                    under,
                )
                recorded_observations += 1
                if maybe_set_preferred_close(
                    connection,
                    table,
                    columns,
                    row,
                    over,
                    under,
                    now,
                    item["minutes_to_start"],
                    ODDS_API_SOURCE,
                ):
                    preferred_updates += 1
                    print(
                        f"✅ CLOSE [The Odds API] "
                        f"{row.get('home_team')} vs {row.get('away_team')} "
                        f"| {item['minutes_to_start']:.0f} min antes "
                        f"| Over {over:.3f} | Under {under:.3f}"
                    )
            connection.commit()
        elif odds_candidates:
            print()
            print("1) The Odds API: sem chave.")

        # ----------------------------------------------------
        # 2) API-Football live: near-kickoff fallback
        # ----------------------------------------------------
        live_candidates = [
            item
            for item in candidates
            if should_query_live(
                item["row"],
                now,
                item["minutes_to_start"],
            )
            and safe_int(item["row"].get("api_football_fixture_id")) is not None
        ]

        if live_candidates and api_football_key:
            print()
            print("2) API-Football LIVE — close perto do kickoff:")
            for item in live_candidates:
                row = item["row"]
                row_id = int(row["id"])
                fixture_id = safe_int(row.get("api_football_fixture_id"))
                if fixture_id is None:
                    continue

                mark_checked(
                    connection,
                    table,
                    row_id,
                    "closing_api_football_live_checked_at",
                    now,
                )

                try:
                    over, under, snapshot, used, remaining = (
                        fetch_api_football_live_odds(
                            fixture_id,
                            api_football_key,
                        )
                    )
                except Exception as error:
                    api_errors.append(f"live fixture {fixture_id}: {error}")
                    print(f"⚠ API-Football live fixture {fixture_id}: {error}")
                    continue

                api_used += used
                api_remaining = remaining or api_remaining
                if over is None or under is None:
                    print(
                        f"⚠ API-Football live sem total 2.5: "
                        f"{row.get('home_team')} vs {row.get('away_team')}"
                    )
                    continue

                observed_at = snapshot or now
                live_minutes = (
                    item["fixture_datetime"] - observed_at
                ).total_seconds() / 60.0
                if live_minutes < -1 or live_minutes > 30:
                    print(
                        f"⚠ API-Football live timestamp fora da janela: "
                        f"{row.get('home_team')} vs {row.get('away_team')} "
                        f"| {live_minutes:.0f} min"
                    )
                    continue

                live_minutes = max(0.0, live_minutes)
                record_observation(
                    connection,
                    row,
                    API_FOOTBALL_LIVE_SOURCE,
                    "PRE_GAME_HIGH",
                    observed_at,
                    live_minutes,
                    over,
                    under,
                )
                recorded_observations += 1
                if maybe_set_preferred_close(
                    connection,
                    table,
                    columns,
                    row,
                    over,
                    under,
                    observed_at,
                    live_minutes,
                    API_FOOTBALL_LIVE_SOURCE,
                ):
                    preferred_updates += 1
                    print(
                        f"✅ CLOSE [API-Football LIVE] "
                        f"{row.get('home_team')} vs {row.get('away_team')} "
                        f"| {live_minutes:.0f} min antes "
                        f"| Over {over:.3f} | Under {under:.3f}"
                    )
            connection.commit()
        elif live_candidates:
            print()
            print("2) API-Football LIVE: sem chave.")

        # ----------------------------------------------------
        # 3) API-Football pre-match: insurance snapshot
        # ----------------------------------------------------
        prematch_candidates = [
            item
            for item in candidates
            if safe_int(item["row"].get("api_football_fixture_id")) is not None
            and should_query_prematch(
                item["row"],
                now,
                item["minutes_to_start"],
            )
        ]

        if prematch_candidates and api_football_key:
            print()
            print("3) API-Football PRE-MATCH — snapshot de segurança:")
            for item in prematch_candidates:
                row = item["row"]
                row_id = int(row["id"])
                fixture_id = safe_int(row.get("api_football_fixture_id"))
                if fixture_id is None:
                    continue

                mark_checked(
                    connection,
                    table,
                    row_id,
                    "closing_api_football_checked_at",
                    now,
                )

                try:
                    over, under, snapshot, used, remaining = (
                        fetch_api_football_prematch_odds(
                            fixture_id,
                            api_football_key,
                        )
                    )
                except Exception as error:
                    api_errors.append(f"prematch fixture {fixture_id}: {error}")
                    print(f"⚠ API-Football prematch fixture {fixture_id}: {error}")
                    continue

                api_used += used
                api_remaining = remaining or api_remaining
                if over is None or under is None:
                    print(
                        f"⚠ API-Football sem Over/Under 2.5: "
                        f"{row.get('home_team')} vs {row.get('away_team')}"
                    )
                    continue

                observed_at = snapshot or now
                snapshot_minutes = (
                    item["fixture_datetime"] - observed_at
                ).total_seconds() / 60.0
                if snapshot_minutes < 0:
                    print(
                        f"⚠ API-Football snapshot pós-kickoff rejeitada: "
                        f"{row.get('home_team')} vs {row.get('away_team')}"
                    )
                    continue

                record_observation(
                    connection,
                    row,
                    API_FOOTBALL_PREMATCH_SOURCE,
                    "PROVISIONAL",
                    observed_at,
                    snapshot_minutes,
                    over,
                    under,
                )
                recorded_observations += 1
                if maybe_set_preferred_close(
                    connection,
                    table,
                    columns,
                    row,
                    over,
                    under,
                    observed_at,
                    snapshot_minutes,
                    API_FOOTBALL_PREMATCH_SOURCE,
                ):
                    preferred_updates += 1
                    print(
                        f"🟡 SNAPSHOT [API-Football] "
                        f"{row.get('home_team')} vs {row.get('away_team')} "
                        f"| {snapshot_minutes:.0f} min antes "
                        f"| Over {over:.3f} | Under {under:.3f}"
                    )
            connection.commit()
        elif prematch_candidates:
            print()
            print("3) API-Football PRE-MATCH: sem chave.")

        print()
        print("==========================================")
        print(" RESUMO")
        print("==========================================")
        print(f"Observações gravadas      : {recorded_observations}")
        print(f"CLOSE preferida atualizada: {preferred_updates}")
        print(f"Pedidos The Odds API      : {odds_used}")
        if odds_remaining is not None:
            print(f"Créditos Odds API restantes: {odds_remaining}")
        if odds_quota_exhausted:
            print("Quota Odds API            : ESGOTADA")
        print(f"Pedidos API-Football      : {api_used}")
        if api_remaining is not None:
            print(f"API-Football restantes    : {api_remaining}")

        if api_errors:
            print()
            print("Avisos API-Football:")
            for error in api_errors[:10]:
                print(f"  - {error}")

        print()
        print("Estratégia de cobertura:")
        print("- The Odds API: média EU muito perto do kickoff, quando há créditos.")
        print("- API-Football LIVE: tenta capturar total 2.5 nos últimos minutos.")
        print("- API-Football PRE-MATCH: snapshot provisória para não ficar sem nada.")
        print("- Todas as observações ficam guardadas em closing_odds_observations.")
        print("- Football-Data AvgC continua a finalizar/repor CLOSE após o jogo.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
