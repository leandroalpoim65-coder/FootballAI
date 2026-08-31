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
ODDS_API_REGION = "eu"
ODDS_API_MARKET = "totals"

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

REFRESH_WINDOW_MINUTES = 30
MIN_REQUERY_IMPROVEMENT_MINUTES = 12
EXCELLENT_CLOSE_MINUTES = 8

ODDS_API_CLOSING_SOURCE = "The Odds API / EU average totals 2.5"
API_FOOTBALL_CLOSING_SOURCE = (
    "API-Football / average bookmaker pre-match snapshot"
)

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
    "nurnberg": "nurnberg",
    "1 nurnberg": "nurnberg",
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
}


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
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    existing = {row[0] for row in rows}

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
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def add_column_if_missing(connection, table: str, columns: set[str], name: str, sql_type: str) -> None:
    if name in columns:
        return
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    columns.add(name)


def parse_fixture_datetime(date_value, time_value) -> datetime | None:
    if date_value is None:
        return None
    date_text = str(date_value).strip()
    if not date_text:
        return None
    time_text = "00:00" if time_value is None else str(time_value).strip()
    if not time_text:
        time_text = "00:00"

    for candidate in (f"{date_text[:10]} {time_text}", f"{date_text[:10]} {time_text[:5]}"):
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


def http_get_json(url: str, extra_headers: dict[str, str] | None = None) -> tuple[object, dict[str, str]]:
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
            response_headers = {key.lower(): value for key, value in response.headers.items()}
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
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\butd\b", " united ", text)
    text = re.sub(r"\b(fc|cf|afc|sc|fk|ac|sv|rcd|cd|club)\b", " ", text)
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


def safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def should_query_row(row: dict, minutes_to_start: float) -> bool:
    old_minutes = safe_float(row.get("closing_minutes_before_kickoff"))
    if old_minutes is None:
        return True
    if old_minutes <= EXCELLENT_CLOSE_MINUTES:
        return False
    return (old_minutes - minutes_to_start) >= MIN_REQUERY_IMPROVEMENT_MINUTES


def find_unique_odds_api_event(events: list[dict], canonical_league: str, fixture_datetime: datetime, home_team, away_team) -> dict | None:
    candidates: list[tuple[float, dict]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        kickoff_local = parse_api_datetime(event.get("commence_time"))
        if kickoff_local is None or kickoff_local.date() != fixture_datetime.date():
            continue
        if abs((kickoff_local - fixture_datetime).total_seconds()) > 3 * 3600:
            continue
        hs = team_similarity(canonical_league, event.get("home_team"), home_team)
        as_ = team_similarity(canonical_league, event.get("away_team"), away_team)
        if hs < 0.78 or as_ < 0.78:
            continue
        candidates.append((hs + as_, event))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) >= 2 and candidates[0][0] - candidates[1][0] < 0.08:
        return None
    return candidates[0][1]


def extract_odds_api_average_totals_25(event: dict) -> tuple[float | None, float | None]:
    over_prices: list[float] = []
    under_prices: list[float] = []
    bookmakers = event.get("bookmakers", [])
    if not isinstance(bookmakers, list):
        return None, None
    for bookmaker in bookmakers:
        if not isinstance(bookmaker, dict):
            continue
        markets = bookmaker.get("markets", [])
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict) or market.get("key") != "totals":
                continue
            outcomes = market.get("outcomes", [])
            if not isinstance(outcomes, list):
                continue
            for outcome in outcomes:
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
    return float(sum(over_prices) / len(over_prices)), float(sum(under_prices) / len(under_prices))


def fetch_odds_api_events(candidates: list[dict], api_key: str):
    required_sport_keys = sorted({str(item["sport_key"]) for item in candidates if item.get("sport_key")})
    events_by_sport: dict[str, list[dict]] = {}
    successful_sports: set[str] = set()
    failed_sports: list[str] = []
    requests_used = 0
    requests_remaining: str | None = None
    quota_exhausted = False

    for sport_key in required_sport_keys:
        query = urlencode({
            "apiKey": api_key,
            "regions": ODDS_API_REGION,
            "markets": ODDS_API_MARKET,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        })
        url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds/?{query}"
        try:
            payload, headers = http_get_json(url)
        except Exception as error:
            failed_sports.append(sport_key)
            print(f"⚠ The Odds API {sport_key}: {error}")
            if is_quota_error(error):
                quota_exhausted = True
                print("⚠ The Odds API sem créditos; passa ao fallback.")
                break
            continue
        requests_used += 1
        if not isinstance(payload, list):
            failed_sports.append(sport_key)
            continue
        events_by_sport[sport_key] = [event for event in payload if isinstance(event, dict)]
        successful_sports.add(sport_key)
        requests_remaining = headers.get("x-requests-remaining", requests_remaining)
        print(f"✅ The Odds API {sport_key}: {len(events_by_sport[sport_key])} eventos")

    return events_by_sport, successful_sports, requests_used, requests_remaining, quota_exhausted, failed_sports


def api_football_get_json(path: str, params: dict[str, object], api_key: str) -> tuple[dict, dict[str, str]]:
    url = f"{API_FOOTBALL_BASE_URL}{path}?{urlencode(params)}"
    payload, headers = http_get_json(url, {"x-apisports-key": api_key})
    if not isinstance(payload, dict):
        raise RuntimeError("API-Football devolveu resposta inesperada.")
    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"API-Football: {errors}")
    return payload, headers


def fetch_api_football_fixtures(fixture_date: str, api_key: str) -> tuple[list[dict], int, str | None]:
    payload, headers = api_football_get_json(
        "/fixtures",
        {"date": fixture_date, "timezone": "Europe/Lisbon"},
        api_key,
    )
    response = payload.get("response", [])
    fixtures = [item for item in response if isinstance(item, dict)] if isinstance(response, list) else []
    remaining = headers.get("x-ratelimit-requests-remaining", headers.get("x-ratelimit-remaining"))
    return fixtures, 1, remaining


def api_football_fixture_details(item: dict) -> tuple[int | None, datetime | None, str | None, str | None]:
    fixture = item.get("fixture")
    teams = item.get("teams")
    if not isinstance(fixture, dict) or not isinstance(teams, dict):
        return None, None, None, None
    try:
        fixture_id = int(fixture.get("id"))
    except (TypeError, ValueError):
        fixture_id = None
    kickoff = parse_api_datetime(fixture.get("date"))
    home = teams.get("home")
    away = teams.get("away")
    home_name = str(home.get("name")).strip() if isinstance(home, dict) and home.get("name") else None
    away_name = str(away.get("name")).strip() if isinstance(away, dict) and away.get("name") else None
    return fixture_id, kickoff, home_name, away_name


def find_unique_api_football_fixture(fixtures: list[dict], canonical_league: str, fixture_datetime: datetime, home_team, away_team) -> dict | None:
    candidates: list[tuple[float, dict]] = []
    for item in fixtures:
        fixture_id, kickoff, api_home, api_away = api_football_fixture_details(item)
        if fixture_id is None or kickoff is None or kickoff.date() != fixture_datetime.date():
            continue
        if abs((kickoff - fixture_datetime).total_seconds()) > 3 * 3600:
            continue
        hs = team_similarity(canonical_league, api_home, home_team)
        as_ = team_similarity(canonical_league, api_away, away_team)
        if hs < 0.78 or as_ < 0.78:
            continue
        candidates.append((hs + as_, item))
    if not candidates:
        return None
    candidates.sort(key=lambda value: value[0], reverse=True)
    if len(candidates) >= 2 and candidates[0][0] - candidates[1][0] < 0.08:
        return None
    return candidates[0][1]


def is_over_25_text(value: object) -> bool:
    text = str(value or "").strip().casefold().replace(",", ".")
    return bool(re.search(r"\bover\s*2\.?5\b", text) or re.search(r"\bover\s*2\s*5\b", text))


def is_under_25_text(value: object) -> bool:
    text = str(value or "").strip().casefold().replace(",", ".")
    return bool(re.search(r"\bunder\s*2\.?5\b", text) or re.search(r"\bunder\s*2\s*5\b", text))


def extract_api_football_average_totals_25(payload: dict) -> tuple[float | None, float | None, datetime | None]:
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
        bookmakers = entry.get("bookmakers", [])
        if not isinstance(bookmakers, list):
            continue
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue
            bets = bookmaker.get("bets", [])
            if not isinstance(bets, list):
                continue
            bookmaker_over = None
            bookmaker_under = None
            for bet in bets:
                if not isinstance(bet, dict):
                    continue
                values = bet.get("values", [])
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, dict):
                        continue
                    odd = safe_float(value.get("odd"))
                    if odd is None or odd <= 1.0:
                        continue
                    value_text = value.get("value")
                    if bookmaker_over is None and is_over_25_text(value_text):
                        bookmaker_over = odd
                    elif bookmaker_under is None and is_under_25_text(value_text):
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


def fetch_api_football_odds(fixture_id: int, api_key: str):
    payload, headers = api_football_get_json("/odds", {"fixture": fixture_id}, api_key)
    over, under, snapshot_time = extract_api_football_average_totals_25(payload)
    remaining = headers.get("x-ratelimit-requests-remaining", headers.get("x-ratelimit-remaining"))
    return over, under, snapshot_time, 1, remaining


def save_close(connection, table: str, row: dict, over_close: float, under_close: float, observed_at: datetime, minutes_before_kickoff: float, source: str) -> bool:
    old_minutes = safe_float(row.get("closing_minutes_before_kickoff"))
    if old_minutes is not None and minutes_before_kickoff >= old_minutes:
        return False
    connection.execute(
        f"""
        UPDATE {table}
        SET odd_over25_close = ?, odd_under25_close = ?, closing_observed_at = ?,
            closing_minutes_before_kickoff = ?, closing_source = ?
        WHERE id = ?
        """,
        (
            over_close,
            under_close,
            observed_at.isoformat(timespec="seconds"),
            minutes_before_kickoff,
            source,
            int(row["id"]),
        ),
    )
    row["odd_over25_close"] = over_close
    row["odd_under25_close"] = under_close
    row["closing_observed_at"] = observed_at.isoformat(timespec="seconds")
    row["closing_minutes_before_kickoff"] = minutes_before_kickoff
    row["closing_source"] = source
    return True


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — CLOSING ODDS CLOUD")
    print("==========================================")
    print()

    odds_api_key = os.getenv("ODDS_API_KEY", "").strip()
    api_football_key = os.getenv("API_FOOTBALL_KEY", "").strip()

    try:
        connection = connect_turso()
    except Exception as error:
        print(f"❌ Não foi possível ligar à Turso: {error}")
        raise SystemExit(1) from error

    print("Base de dados             : Turso Cloud")
    print("Timezone                  : Europe/Lisbon")
    print(f"Janela CLOSE pré-jogo     : 0-{REFRESH_WINDOW_MINUTES} min")
    print("Fonte 1                   : The Odds API" + (" ✅" if odds_api_key else " ⚪ sem chave"))
    print("Fonte 2                   : API-Football fallback" + (" ✅" if api_football_key else " ⚪ sem chave"))

    try:
        table = find_paper_table(connection)
        if table is None:
            print("❌ Tabela de Paper Trade não encontrada.")
            raise SystemExit(1)

        columns = table_columns(connection, table)
        for name, sql_type in (
            ("odd_over25_close", "REAL"),
            ("odd_under25_close", "REAL"),
            ("closing_observed_at", "TEXT"),
            ("closing_minutes_before_kickoff", "REAL"),
            ("closing_source", "TEXT"),
        ):
            add_column_if_missing(connection, table, columns, name, sql_type)
        connection.commit()

        fixture_date_column = "fixture_date" if "fixture_date" in columns else "date" if "date" in columns else None
        fixture_time_column = "fixture_time" if "fixture_time" in columns else "time" if "time" in columns else None
        if fixture_date_column is None or fixture_time_column is None:
            print("❌ A tabela não tem data/hora de fixture.")
            raise SystemExit(1)

        now = datetime.now(APP_TIMEZONE)
        rows = fetch_dicts(
            connection,
            f"""
            SELECT * FROM {table}
            WHERE COALESCE(status, 'OPEN') = 'OPEN'
            ORDER BY {fixture_date_column}, {fixture_time_column}, id
            """,
        )

        candidates = []
        skipped_past = skipped_far = skipped_good_close = skipped_invalid_datetime = skipped_invalid_league = 0

        for row in rows:
            fixture_datetime = parse_fixture_datetime(row.get(fixture_date_column), row.get(fixture_time_column))
            if fixture_datetime is None:
                skipped_invalid_datetime += 1
                continue
            minutes_to_start = (fixture_datetime - now).total_seconds() / 60.0
            if minutes_to_start < 0:
                skipped_past += 1
                continue
            if minutes_to_start > REFRESH_WINDOW_MINUTES:
                skipped_far += 1
                continue
            if not should_query_row(row, minutes_to_start):
                skipped_good_close += 1
                continue
            canonical_league = normalize_league(row.get("league"))
            if not canonical_league:
                skipped_invalid_league += 1
                continue
            candidates.append({
                "row": row,
                "fixture_datetime": fixture_datetime,
                "minutes_to_start": float(minutes_to_start),
                "canonical_league": str(canonical_league),
                "sport_key": LEAGUE_SPORT_KEY.get(str(canonical_league)),
            })

        print(f"Paper Trades OPEN         : {len(rows)}")
        print(f"Candidatos a consultar    : {len(candidates)}")
        print(f"Kickoff já passou         : {skipped_past}")
        print(f"Ainda longe do kickoff    : {skipped_far}")
        print(f"CLOSE já suficientemente boa: {skipped_good_close}")
        if skipped_invalid_datetime:
            print(f"Data/hora inválida        : {skipped_invalid_datetime}")
        if skipped_invalid_league:
            print(f"Liga inválida             : {skipped_invalid_league}")

        if not candidates:
            print("✅ Nenhum jogo precisa de pedido a uma API agora.")
            return

        print("\nJogos candidatos:")
        for item in candidates:
            row = item["row"]
            print(f"  - {row.get('home_team')} vs {row.get('away_team')} | {item['canonical_league']} | {item['minutes_to_start']:.0f} min")

        captured_ids: set[int] = set()
        odds_requests_used = 0
        odds_requests_remaining = None
        odds_quota_exhausted = False

        if odds_api_key:
            odds_candidates = [item for item in candidates if item.get("sport_key")]
            if odds_candidates:
                print("\n1) The Odds API:")
                events_by_sport, successful_sports, odds_requests_used, odds_requests_remaining, odds_quota_exhausted, _ = fetch_odds_api_events(odds_candidates, odds_api_key)
                for item in odds_candidates:
                    row = item["row"]
                    row_id = int(row["id"])
                    sport_key = item.get("sport_key")
                    if not sport_key or sport_key not in successful_sports:
                        continue
                    event = find_unique_odds_api_event(
                        events_by_sport.get(str(sport_key), []),
                        str(item["canonical_league"]),
                        item["fixture_datetime"],
                        row.get("home_team"),
                        row.get("away_team"),
                    )
                    if event is None:
                        continue
                    over_close, under_close = extract_odds_api_average_totals_25(event)
                    if over_close is None or under_close is None:
                        continue
                    if save_close(connection, table, row, over_close, under_close, now, float(item["minutes_to_start"]), ODDS_API_CLOSING_SOURCE):
                        captured_ids.add(row_id)
                        print(f"✅ CLOSE [The Odds API] {row.get('home_team')} vs {row.get('away_team')} | Over {over_close:.3f} | Under {under_close:.3f}")
                connection.commit()
        else:
            print("\n1) The Odds API: sem ODDS_API_KEY.")

        remaining = [item for item in candidates if int(item["row"]["id"]) not in captured_ids]
        api_football_requests_used = 0
        api_football_remaining = None
        api_football_errors: list[str] = []

        if remaining and api_football_key:
            print("\n2) API-Football fallback:")
            dates_needed = sorted({item["fixture_datetime"].strftime("%Y-%m-%d") for item in remaining})
            fixtures_by_date: dict[str, list[dict]] = {}
            for fixture_date in dates_needed:
                try:
                    fixtures, used, quota_remaining = fetch_api_football_fixtures(fixture_date, api_football_key)
                except Exception as error:
                    api_football_errors.append(f"fixtures {fixture_date}: {error}")
                    print(f"⚠ API-Football fixtures {fixture_date}: {error}")
                    continue
                api_football_requests_used += used
                api_football_remaining = quota_remaining or api_football_remaining
                fixtures_by_date[fixture_date] = fixtures
                print(f"✅ API-Football fixtures {fixture_date}: {len(fixtures)} jogos")

            for item in remaining:
                row = item["row"]
                row_id = int(row["id"])
                fixture_datetime = item["fixture_datetime"]
                canonical_league = str(item["canonical_league"])
                fixture_date = fixture_datetime.strftime("%Y-%m-%d")
                fixture_item = find_unique_api_football_fixture(
                    fixtures_by_date.get(fixture_date, []),
                    canonical_league,
                    fixture_datetime,
                    row.get("home_team"),
                    row.get("away_team"),
                )
                if fixture_item is None:
                    continue
                fixture_id, _, _, _ = api_football_fixture_details(fixture_item)
                if fixture_id is None:
                    continue
                try:
                    over_close, under_close, snapshot_time, used, quota_remaining = fetch_api_football_odds(fixture_id, api_football_key)
                except Exception as error:
                    api_football_errors.append(f"odds fixture {fixture_id}: {error}")
                    print(f"⚠ API-Football odds fixture {fixture_id}: {error}")
                    continue
                api_football_requests_used += used
                api_football_remaining = quota_remaining or api_football_remaining
                if over_close is None or under_close is None:
                    print(f"⚠ API-Football sem Over/Under 2.5: {row.get('home_team')} vs {row.get('away_team')}")
                    continue
                effective_observed_at = snapshot_time or now
                effective_minutes = (fixture_datetime - effective_observed_at).total_seconds() / 60.0
                if effective_minutes < 0:
                    print(f"⚠ API-Football snapshot pós-kickoff: {row.get('home_team')} vs {row.get('away_team')}")
                    continue
                if save_close(connection, table, row, over_close, under_close, effective_observed_at, effective_minutes, API_FOOTBALL_CLOSING_SOURCE):
                    captured_ids.add(row_id)
                    print(f"✅ CLOSE [API-Football] {row.get('home_team')} vs {row.get('away_team')} | snapshot {effective_minutes:.0f} min antes | Over {over_close:.3f} | Under {under_close:.3f}")
            connection.commit()
        elif remaining:
            print("\n2) API-Football fallback: sem API_FOOTBALL_KEY.")

        print("\n==========================================")
        print(" RESUMO")
        print("==========================================")
        print(f"Closing odds atualizadas  : {len(captured_ids)}")
        print(f"Pedidos The Odds API      : {odds_requests_used}")
        if odds_requests_remaining is not None:
            print(f"Créditos Odds API restantes: {odds_requests_remaining}")
        if odds_quota_exhausted:
            print("Quota Odds API            : ESGOTADA")
        print(f"Pedidos API-Football      : {api_football_requests_used}")
        if api_football_remaining is not None:
            print(f"API-Football restantes    : {api_football_remaining}")
        print(f"Sem CLOSE nesta execução  : {len(candidates) - len(captured_ids)}")
        if api_football_errors:
            print("\nAvisos API-Football:")
            for error in api_football_errors[:10]:
                print(f"  - {error}")

        print("\nRegra de segurança:")
        print("- OPEN nunca é alterada por este script.")
        print("- The Odds API continua a ser a fonte preferida para CLOSE pré-jogo.")
        print("- API-Football é apenas fallback e fica identificado em closing_source.")
        print("- Football-Data pós-jogo continua a recuperar closing odds quando disponível.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
