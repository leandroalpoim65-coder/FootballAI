from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from statistics import median
from urllib.parse import urlencode

import paper_trade_cloud as core
from closing_odds_v3_cloud import (
    APP_TIMEZONE,
    LEAGUE_SPORT_KEY,
    ODDS_API_BASE_URL,
    api_football_get_json,
    http_get_json,
    parse_api_datetime,
    team_similarity,
)
from ml.paper_trade import normalize_league
from upcoming_fallback_sources import API_FOOTBALL_LEAGUE_ID_MAP


TABLE_NAME = "multi_market_odds"
UPCOMING_TABLE = "upcoming_matches"

TOTAL_LINES = {1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5}
TEAM_TOTAL_LINES = {0.5, 1.5, 2.5}
NO_LINE = -1.0

API_FOOTBALL_LEAGUES = {
    **API_FOOTBALL_LEAGUE_ID_MAP,
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    848: "UEFA Conference League",
}

ODDS_API_SPORT_KEYS = {
    **LEAGUE_SPORT_KEY,
    "UEFA Champions League": "soccer_uefa_champs_league",
    "UEFA Europa League": "soccer_uefa_europa_league",
    "UEFA Conference League": "soccer_uefa_europa_conference_league",
}


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def clean_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def valid_odd(value: object) -> float | None:
    try:
        odd = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    if odd != odd or odd <= 1.0:
        return None
    return odd


def valid_line(value: object) -> float | None:
    try:
        line = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return line if line == line else None


def parse_side_line(value: object) -> tuple[str, float] | None:
    text = clean_name(value)
    match = re.search(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if not match:
        return None
    return match.group(1), float(match.group(2))


def ensure_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_key TEXT NOT NULL,
            fixture_date TEXT NOT NULL,
            fixture_time TEXT,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            market TEXT NOT NULL,
            selection TEXT NOT NULL,
            line REAL NOT NULL DEFAULT -1.0,
            open_odd REAL NOT NULL,
            current_odd REAL NOT NULL,
            first_observed_at TEXT NOT NULL,
            current_observed_at TEXT NOT NULL,
            open_source TEXT NOT NULL,
            current_source TEXT NOT NULL,
            open_bookmakers_count INTEGER,
            current_bookmakers_count INTEGER,
            source_event_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_{TABLE_NAME}_market
        ON {TABLE_NAME}(fixture_key, market, selection, line)
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_fixture
        ON {TABLE_NAME}(fixture_key, market)
        """
    )
    connection.commit()


def load_upcoming(connection, horizon_days: int) -> list[dict]:
    today = datetime.now(APP_TIMEZONE).date()
    end_date = today + timedelta(days=max(1, horizon_days) - 1)
    return core.fetch_dicts(
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
            current_observed_at
        FROM {UPCOMING_TABLE}
        WHERE active = 1
          AND date >= ?
          AND date <= ?
        ORDER BY date, time, league, home_team
        """,
        (today.isoformat(), end_date.isoformat()),
    )


def persist_market(
    connection,
    fixture: dict,
    market: str,
    selection: str,
    line: float,
    odd: float,
    observed_at: str,
    source: str,
    bookmakers_count: int | None,
    source_event_id: object = None,
) -> bool:
    odd = float(odd)
    line = float(line)
    existing = connection.execute(
        f"""
        SELECT id
        FROM {TABLE_NAME}
        WHERE fixture_key = ?
          AND market = ?
          AND selection = ?
          AND line = ?
        LIMIT 1
        """,
        (
            fixture["fixture_key"],
            market,
            selection,
            line,
        ),
    ).fetchone()

    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if existing is None:
        connection.execute(
            f"""
            INSERT INTO {TABLE_NAME} (
                fixture_key, fixture_date, fixture_time, league,
                home_team, away_team, market, selection, line,
                open_odd, current_odd,
                first_observed_at, current_observed_at,
                open_source, current_source,
                open_bookmakers_count, current_bookmakers_count,
                source_event_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fixture["fixture_key"], fixture["date"], fixture.get("time"),
                fixture["league"], fixture["home_team"], fixture["away_team"],
                market, selection, line, odd, odd, observed_at, observed_at,
                source, source, bookmakers_count, bookmakers_count,
                None if source_event_id is None else str(source_event_id),
                now_utc, now_utc,
            ),
        )
        return True

    connection.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET current_odd = ?,
            current_observed_at = ?,
            current_source = ?,
            current_bookmakers_count = ?,
            source_event_id = COALESCE(?, source_event_id),
            updated_at = ?
        WHERE id = ?
        """,
        (
            odd,
            observed_at,
            source,
            bookmakers_count,
            None if source_event_id is None else str(source_event_id),
            now_utc,
            int(existing[0]),
        ),
    )
    return False


def seed_legacy_over25(connection, upcoming: list[dict]) -> int:
    inserted = 0
    for fixture in upcoming:
        first_seen = str(
            fixture.get("first_observed_at")
            or datetime.now(APP_TIMEZONE).isoformat(timespec="seconds")
        )
        current_seen = str(
            fixture.get("current_observed_at")
            or first_seen
        )
        for selection, open_key, current_key in (
            ("over", "odd_over25_open", "odd_over25_current"),
            ("under", "odd_under25_open", "odd_under25_current"),
        ):
            open_odd = valid_odd(fixture.get(open_key))
            current_odd = valid_odd(fixture.get(current_key))
            if open_odd is None:
                continue
            created = persist_market(
                connection,
                fixture,
                "totals",
                selection,
                2.5,
                open_odd,
                first_seen,
                "upcoming_matches / legacy totals 2.5",
                None,
            )
            inserted += int(created)
            if current_odd is not None and abs(current_odd - open_odd) > 1e-12:
                persist_market(
                    connection,
                    fixture,
                    "totals",
                    selection,
                    2.5,
                    current_odd,
                    current_seen,
                    "upcoming_matches / legacy totals 2.5",
                    None,
                )
    connection.commit()
    return inserted


def find_upcoming_fixture(
    upcoming: list[dict],
    date_text: str,
    league: str,
    home: str,
    away: str,
) -> dict | None:
    candidates: list[tuple[float, dict]] = []
    for row in upcoming:
        if str(row.get("date")) != date_text:
            continue
        if str(row.get("league")) != str(league):
            continue
        home_score = team_similarity(league, home, row.get("home_team"))
        away_score = team_similarity(league, away, row.get("away_team"))
        if home_score < 0.72 or away_score < 0.72:
            continue
        candidates.append((home_score + away_score, row))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.10:
        return None
    return candidates[0][1]


def add_price(
    store: dict[tuple[str, str, float], dict[str, float]],
    key: tuple[str, str, float],
    bookmaker: str,
    odd: object,
) -> None:
    price = valid_odd(odd)
    if price is None:
        return
    store.setdefault(key, {})[bookmaker] = price


def aggregate_prices(
    store: dict[tuple[str, str, float], dict[str, float]],
) -> list[tuple[str, str, float, float, int]]:
    records: list[tuple[str, str, float, float, int]] = []
    for (market, selection, line), by_bookmaker in store.items():
        values = list(by_bookmaker.values())
        if not values:
            continue
        records.append(
            (market, selection, line, float(median(values)), len(values))
        )
    return records


def is_full_match_market(bet_name: str) -> bool:
    excluded = (
        "first half", "1st half", "second half", "2nd half",
        "half time", "halftime", "corner", "card", "booking",
        "shot", "period", "10 minutes", "15 minutes",
    )
    return not any(token in bet_name for token in excluded)


def api_football_prices(entry: dict, fixture: dict) -> dict[tuple[str, str, float], dict[str, float]]:
    prices: dict[tuple[str, str, float], dict[str, float]] = {}

    for bookmaker in entry.get("bookmakers", []) or []:
        if not isinstance(bookmaker, dict):
            continue
        bookmaker_name = str(bookmaker.get("name") or bookmaker.get("id") or "unknown")
        for bet in bookmaker.get("bets", []) or []:
            if not isinstance(bet, dict):
                continue
            bet_name = clean_name(bet.get("name"))
            if not is_full_match_market(bet_name):
                continue

            values = bet.get("values", []) or []
            if not isinstance(values, list):
                continue

            is_btts = "both teams" in bet_name and "score" in bet_name
            is_match_winner = (
                "match winner" in bet_name
                or bet_name in {"1x2", "winner"}
            )
            is_home_total = any(
                token in bet_name
                for token in (
                    "home team total", "home total", "home goals over under",
                    "total goals home",
                )
            )
            is_away_total = any(
                token in bet_name
                for token in (
                    "away team total", "away total", "away goals over under",
                    "total goals away",
                )
            )
            is_total = (
                not is_home_total
                and not is_away_total
                and any(
                    token in bet_name
                    for token in ("goals over under", "total goals", "asian total")
                )
            )

            for value in values:
                if not isinstance(value, dict):
                    continue
                raw_value = value.get("value")
                value_text = clean_name(raw_value)
                odd = value.get("odd")

                if is_btts:
                    if value_text in {"yes", "no"}:
                        add_price(prices, ("btts", value_text, NO_LINE), bookmaker_name, odd)
                    continue

                if is_match_winner:
                    selection = None
                    if value_text in {"home", "1"}:
                        selection = "home"
                    elif value_text in {"draw", "x"}:
                        selection = "draw"
                    elif value_text in {"away", "2"}:
                        selection = "away"
                    elif team_similarity(str(fixture["league"]), str(raw_value), str(fixture["home_team"])) >= 0.86:
                        selection = "home"
                    elif team_similarity(str(fixture["league"]), str(raw_value), str(fixture["away_team"])) >= 0.86:
                        selection = "away"
                    if selection:
                        add_price(prices, ("h2h", selection, NO_LINE), bookmaker_name, odd)
                    continue

                parsed = parse_side_line(raw_value)
                if parsed is None:
                    continue
                selection, line = parsed

                if is_home_total and line in TEAM_TOTAL_LINES:
                    add_price(prices, ("team_total_home", selection, line), bookmaker_name, odd)
                elif is_away_total and line in TEAM_TOTAL_LINES:
                    add_price(prices, ("team_total_away", selection, line), bookmaker_name, odd)
                elif is_total and line in TOTAL_LINES:
                    add_price(prices, ("totals", selection, line), bookmaker_name, odd)

    return prices


def api_football_fixture_details(item: dict) -> tuple[int, str, str, str, str] | None:
    fixture = item.get("fixture")
    teams = item.get("teams")
    league_data = item.get("league")
    if not isinstance(fixture, dict) or not isinstance(teams, dict):
        return None

    try:
        fixture_id = int(fixture.get("id"))
    except (TypeError, ValueError):
        return None

    kickoff = parse_api_datetime(fixture.get("date"))
    if kickoff is None:
        return None

    league_id = None
    league_name = None
    if isinstance(league_data, dict):
        try:
            league_id = int(league_data.get("id"))
        except (TypeError, ValueError):
            league_id = None
        league_name = league_data.get("name")

    league = API_FOOTBALL_LEAGUES.get(league_id) if league_id is not None else None
    if league is None:
        normalized = normalize_league(league_name)
        league = None if normalized is None else str(normalized)
    if not league:
        return None

    home_data = teams.get("home")
    away_data = teams.get("away")
    home = home_data.get("name") if isinstance(home_data, dict) else None
    away = away_data.get("name") if isinstance(away_data, dict) else None
    if not home or not away:
        return None

    return fixture_id, kickoff.strftime("%Y-%m-%d"), str(league), str(home), str(away)


def collect_api_football(connection, upcoming: list[dict], observed_at: str) -> dict:
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    max_requests = env_int("MULTI_MARKET_API_FOOTBALL_MAX_REQUESTS", 24, 2)
    if not api_key:
        return {"source": "API-Football", "requests": 0, "inserted": 0, "updated": 0, "error": "sem chave"}

    dates = sorted({str(row["date"]) for row in upcoming})
    requests_used = 0
    inserted = 0
    updated = 0
    matched_fixtures = 0
    last_remaining = None

    for date_text in dates:
        if requests_used >= max_requests:
            break

        try:
            fixture_payload, headers = api_football_get_json(
                "/fixtures",
                {"date": date_text, "timezone": "Europe/Lisbon"},
                api_key,
            )
        except Exception as error:
            return {
                "source": "API-Football", "requests": requests_used,
                "inserted": inserted, "updated": updated, "error": str(error),
                "remaining": last_remaining,
            }
        requests_used += 1
        last_remaining = headers.get("x-ratelimit-requests-remaining", headers.get("x-ratelimit-remaining", last_remaining))

        fixture_map: dict[int, dict] = {}
        response = fixture_payload.get("response", [])
        if isinstance(response, list):
            for item in response:
                if not isinstance(item, dict):
                    continue
                details = api_football_fixture_details(item)
                if details is None:
                    continue
                fixture_id, fixture_date, league, home, away = details
                target = find_upcoming_fixture(upcoming, fixture_date, league, home, away)
                if target is not None:
                    fixture_map[fixture_id] = target

        if not fixture_map:
            continue

        page = 1
        total_pages = 1
        seen_fixture_ids: set[int] = set()
        while page <= total_pages and requests_used < max_requests:
            try:
                odds_payload, headers = api_football_get_json(
                    "/odds",
                    {"date": date_text, "timezone": "Europe/Lisbon", "page": page},
                    api_key,
                )
            except Exception as error:
                return {
                    "source": "API-Football", "requests": requests_used,
                    "inserted": inserted, "updated": updated, "error": str(error),
                    "remaining": last_remaining,
                }
            requests_used += 1
            last_remaining = headers.get("x-ratelimit-requests-remaining", headers.get("x-ratelimit-remaining", last_remaining))

            paging = odds_payload.get("paging", {})
            if isinstance(paging, dict):
                try:
                    total_pages = max(1, int(paging.get("total", 1)))
                except (TypeError, ValueError):
                    total_pages = 1

            odds_rows = odds_payload.get("response", [])
            if isinstance(odds_rows, list):
                for entry in odds_rows:
                    if not isinstance(entry, dict):
                        continue
                    fixture_data = entry.get("fixture")
                    if not isinstance(fixture_data, dict):
                        continue
                    try:
                        fixture_id = int(fixture_data.get("id"))
                    except (TypeError, ValueError):
                        continue
                    target = fixture_map.get(fixture_id)
                    if target is None:
                        continue
                    seen_fixture_ids.add(fixture_id)
                    store = api_football_prices(entry, target)
                    for market, selection, line, odd, count in aggregate_prices(store):
                        created = persist_market(
                            connection, target, market, selection, line, odd,
                            observed_at,
                            "API-Football / multi-market median",
                            count,
                            fixture_id,
                        )
                        inserted += int(created)
                        updated += int(not created)
            page += 1

        matched_fixtures += len(seen_fixture_ids)
        connection.commit()

    return {
        "source": "API-Football",
        "requests": requests_used,
        "inserted": inserted,
        "updated": updated,
        "matched_fixtures": matched_fixtures,
        "remaining": last_remaining,
        "error": None,
    }


def existing_keys(connection, fixture_key: str) -> set[tuple[str, str, float]]:
    rows = connection.execute(
        f"SELECT market, selection, line FROM {TABLE_NAME} WHERE fixture_key = ?",
        (fixture_key,),
    ).fetchall()
    return {(str(row[0]), str(row[1]), float(row[2])) for row in rows}


def missing_odds_api_markets(keys: set[tuple[str, str, float]]) -> list[str]:
    markets: list[str] = []
    if any(("h2h", selection, NO_LINE) not in keys for selection in ("home", "draw", "away")):
        markets.append("h2h")
    if any(("btts", selection, NO_LINE) not in keys for selection in ("yes", "no")):
        markets.append("btts")
    if any(("totals", side, line) not in keys for line in TOTAL_LINES for side in ("over", "under")):
        markets.append("alternate_totals")
    if any(
        (market, side, line) not in keys
        for market in ("team_total_home", "team_total_away")
        for line in TEAM_TOTAL_LINES
        for side in ("over", "under")
    ):
        markets.append("alternate_team_totals")
    return markets


def odds_api_prices(event: dict, fixture: dict) -> dict[tuple[str, str, float], dict[str, float]]:
    prices: dict[tuple[str, str, float], dict[str, float]] = {}
    league = str(fixture["league"])

    for bookmaker in event.get("bookmakers", []) or []:
        if not isinstance(bookmaker, dict):
            continue
        bookmaker_name = str(bookmaker.get("title") or bookmaker.get("key") or "unknown")
        for market in bookmaker.get("markets", []) or []:
            if not isinstance(market, dict):
                continue
            market_key = str(market.get("key") or "")
            for outcome in market.get("outcomes", []) or []:
                if not isinstance(outcome, dict):
                    continue
                raw_name = outcome.get("name")
                name = clean_name(raw_name)
                odd = outcome.get("price")
                point = valid_line(outcome.get("point"))

                if market_key == "h2h":
                    selection = None
                    if name == "draw":
                        selection = "draw"
                    elif team_similarity(league, raw_name, fixture["home_team"]) >= 0.86:
                        selection = "home"
                    elif team_similarity(league, raw_name, fixture["away_team"]) >= 0.86:
                        selection = "away"
                    if selection:
                        add_price(prices, ("h2h", selection, NO_LINE), bookmaker_name, odd)

                elif market_key == "btts" and name in {"yes", "no"}:
                    add_price(prices, ("btts", name, NO_LINE), bookmaker_name, odd)

                elif market_key == "alternate_totals" and point in TOTAL_LINES and name in {"over", "under"}:
                    add_price(prices, ("totals", name, float(point)), bookmaker_name, odd)

                elif market_key == "alternate_team_totals" and point in TEAM_TOTAL_LINES and name in {"over", "under"}:
                    description = outcome.get("description")
                    if description is None:
                        continue
                    if team_similarity(league, description, fixture["home_team"]) >= 0.86:
                        target_market = "team_total_home"
                    elif team_similarity(league, description, fixture["away_team"]) >= 0.86:
                        target_market = "team_total_away"
                    else:
                        continue
                    add_price(prices, (target_market, name, float(point)), bookmaker_name, odd)

    return prices


def odds_api_quota(api_key: str) -> tuple[set[str], int | None]:
    payload, headers = http_get_json(f"{ODDS_API_BASE_URL}/sports/?apiKey={api_key}")
    active = {
        str(item.get("key"))
        for item in payload
        if isinstance(payload, list)
        for item in payload
        if isinstance(item, dict) and item.get("active") is True and item.get("key")
    } if isinstance(payload, list) else set()
    try:
        remaining = int(headers.get("x-requests-remaining"))
    except (TypeError, ValueError):
        remaining = None
    return active, remaining


def collect_odds_api_fallback(connection, upcoming: list[dict], observed_at: str) -> dict:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        return {"source": "The Odds API", "events": 0, "inserted": 0, "updated": 0, "error": "sem chave"}

    reserve = env_int("MULTI_MARKET_ODDS_API_RESERVE", 100, 0)
    max_events = env_int("MULTI_MARKET_ODDS_API_MAX_EVENTS", 6, 0)
    if max_events == 0:
        return {"source": "The Odds API", "events": 0, "inserted": 0, "updated": 0, "error": "desativado por limite"}

    try:
        active_sports, remaining = odds_api_quota(api_key)
    except Exception as error:
        return {"source": "The Odds API", "events": 0, "inserted": 0, "updated": 0, "error": str(error)}

    inserted = 0
    updated = 0
    event_calls = 0
    credits_spent = 0

    leagues = sorted({str(row["league"]) for row in upcoming})
    for league in leagues:
        if event_calls >= max_events:
            break
        sport_key = ODDS_API_SPORT_KEYS.get(league)
        if not sport_key or sport_key not in active_sports:
            continue

        query = urlencode({"apiKey": api_key, "dateFormat": "iso"})
        try:
            events, _ = http_get_json(f"{ODDS_API_BASE_URL}/sports/{sport_key}/events?{query}")
        except Exception as error:
            print(f"⚠ The Odds API events {league}: {error}")
            continue
        if not isinstance(events, list):
            continue

        for event in events:
            if event_calls >= max_events:
                break
            if not isinstance(event, dict):
                continue
            kickoff = parse_api_datetime(event.get("commence_time"))
            if kickoff is None:
                continue
            target = find_upcoming_fixture(
                upcoming,
                kickoff.strftime("%Y-%m-%d"),
                league,
                str(event.get("home_team") or ""),
                str(event.get("away_team") or ""),
            )
            if target is None:
                continue

            needed = missing_odds_api_markets(existing_keys(connection, str(target["fixture_key"])))
            if not needed:
                continue
            estimated_cost = len(needed)
            if remaining is not None and remaining - estimated_cost < reserve:
                return {
                    "source": "The Odds API", "events": event_calls,
                    "inserted": inserted, "updated": updated,
                    "credits_spent": credits_spent, "remaining": remaining,
                    "reserve": reserve, "error": None,
                }

            event_id = event.get("id")
            if not event_id:
                continue
            event_query = urlencode(
                {
                    "apiKey": api_key,
                    "regions": "eu",
                    "markets": ",".join(needed),
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                }
            )
            try:
                payload, headers = http_get_json(
                    f"{ODDS_API_BASE_URL}/sports/{sport_key}/events/{event_id}/odds?{event_query}"
                )
            except Exception as error:
                print(f"⚠ The Odds API event {event_id}: {error}")
                continue

            event_calls += 1
            try:
                last_cost = int(headers.get("x-requests-last", 0))
            except (TypeError, ValueError):
                last_cost = 0
            credits_spent += max(0, last_cost)
            try:
                remaining = int(headers.get("x-requests-remaining"))
            except (TypeError, ValueError):
                pass

            if not isinstance(payload, dict):
                continue
            store = odds_api_prices(payload, target)
            for market, selection, line, odd, count in aggregate_prices(store):
                created = persist_market(
                    connection, target, market, selection, line, odd,
                    observed_at,
                    "The Odds API / EU multi-market median",
                    count,
                    event_id,
                )
                inserted += int(created)
                updated += int(not created)
            connection.commit()

    return {
        "source": "The Odds API",
        "events": event_calls,
        "inserted": inserted,
        "updated": updated,
        "credits_spent": credits_spent,
        "remaining": remaining,
        "reserve": reserve,
        "error": None,
    }


def coverage_summary(connection, upcoming: list[dict]) -> dict[str, int]:
    fixture_keys = [str(row["fixture_key"]) for row in upcoming]
    if not fixture_keys:
        return {}
    counts = {
        "fixtures": len(fixture_keys),
        "h2h": 0,
        "btts": 0,
        "totals": 0,
        "team_totals": 0,
    }
    for key in fixture_keys:
        existing = existing_keys(connection, key)
        if any(item[0] == "h2h" for item in existing):
            counts["h2h"] += 1
        if any(item[0] == "btts" for item in existing):
            counts["btts"] += 1
        if any(item[0] == "totals" and item[2] != 2.5 for item in existing):
            counts["totals"] += 1
        if any(item[0].startswith("team_total_") for item in existing):
            counts["team_totals"] += 1
    return counts


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — MULTI-MARKET ODDS CLOUD")
    print("==========================================")
    print()

    horizon_days = env_int("MULTI_MARKET_HORIZON_DAYS", 3, 1)
    observed_at = datetime.now(APP_TIMEZONE).isoformat(timespec="seconds")

    connection = core.connect_turso()
    try:
        if not core.table_exists(connection, UPCOMING_TABLE):
            raise RuntimeError("A tabela upcoming_matches ainda não existe.")

        ensure_table(connection)
        upcoming = load_upcoming(connection, horizon_days)
        print(f"Jogos upcoming no horizonte : {len(upcoming)}")
        print(f"Horizonte                    : {horizon_days} dia(s)")

        if not upcoming:
            print("Nenhum jogo para enriquecer.")
            return

        seeded = seed_legacy_over25(connection, upcoming)
        print(f"Seeds legacy O/U 2.5         : {seeded}")

        api_info = collect_api_football(connection, upcoming, observed_at)
        print(
            "API-Football                 : "
            f"requests={api_info.get('requests', 0)} | "
            f"novos={api_info.get('inserted', 0)} | "
            f"updates={api_info.get('updated', 0)} | "
            f"erro={api_info.get('error')}"
        )

        odds_info = collect_odds_api_fallback(connection, upcoming, observed_at)
        print(
            "The Odds API fallback        : "
            f"eventos={odds_info.get('events', 0)} | "
            f"créditos={odds_info.get('credits_spent', 0)} | "
            f"novos={odds_info.get('inserted', 0)} | "
            f"erro={odds_info.get('error')}"
        )

        summary = coverage_summary(connection, upcoming)
        print()
        print("Cobertura no horizonte:")
        print(f"  1X2         : {summary.get('h2h', 0)}/{summary.get('fixtures', 0)}")
        print(f"  BTTS        : {summary.get('btts', 0)}/{summary.get('fixtures', 0)}")
        print(f"  Totals extra: {summary.get('totals', 0)}/{summary.get('fixtures', 0)}")
        print(f"  Team totals : {summary.get('team_totals', 0)}/{summary.get('fixtures', 0)}")

        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
