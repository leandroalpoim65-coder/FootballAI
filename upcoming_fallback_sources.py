from __future__ import annotations

import os
from datetime import datetime, timedelta

from ml.paper_trade import normalize_league, normalize_team

from closing_odds_v3_cloud import (
    APP_TIMEZONE,
    LEAGUE_SPORT_KEY,
    ODDS_API_BASE_URL,
    api_football_get_json,
    extract_api_football_prematch_totals,
    extract_odds_api_totals,
    http_get_json,
    parse_api_datetime,
)


# Stable API-Football competition ids for the leagues used by FootballAI.
API_FOOTBALL_LEAGUE_ID_MAP = {
    39: "England",
    40: "England Championship",
    41: "England League One",
    42: "England League Two",
    61: "France",
    62: "France Ligue 2",
    78: "Germany",
    79: "Germany 2 Bundesliga",
    88: "Netherlands",
    94: "Portugal",
    135: "Italy",
    136: "Italy Serie B",
    140: "Spain",
    141: "Spain Segunda Division",
    144: "Belgium First Division A",
    179: "Scotland",
    197: "Greece Super League",
    203: "Turkey Super Lig",
}

ODDS_API_PRIORITY = [
    "England",
    "England Championship",
    "Spain",
    "Italy",
    "Germany",
    "France",
    "Portugal",
    "Netherlands",
    "Belgium First Division A",
    "Turkey Super Lig",
    "Greece Super League",
    "Scotland",
    "England League One",
    "England League Two",
    "France Ligue 2",
    "Germany 2 Bundesliga",
    "Italy Serie B",
    "Spain Segunda Division",
]


def safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 1.0:
        return None
    return number


def canonical_team(league: str, value: object) -> str | None:
    normalized = normalize_team(league, value)
    if normalized is None:
        return None
    text = str(normalized).strip()
    return text or None


def fixture_key(date_value: str, league: str, home: str, away: str) -> str:
    return "|".join(
        [
            date_value.casefold(),
            league.casefold(),
            home.casefold(),
            away.casefold(),
        ]
    )


def build_observation(
    *,
    league: str,
    kickoff: datetime,
    home_team: object,
    away_team: object,
    over: float,
    under: float,
    source: str,
) -> dict | None:
    canonical_league = normalize_league(league)
    if canonical_league is None:
        return None
    league_text = str(canonical_league)

    home = canonical_team(league_text, home_team)
    away = canonical_team(league_text, away_team)
    if home is None or away is None:
        return None

    over_value = safe_float(over)
    under_value = safe_float(under)
    if over_value is None or under_value is None:
        return None

    local = kickoff.astimezone(APP_TIMEZONE)
    date_text = local.strftime("%Y-%m-%d")
    time_text = local.strftime("%H:%M")

    return {
        "fixture_key": fixture_key(date_text, league_text, home, away),
        "date": date_text,
        "time": time_text,
        "league": league_text,
        "home_team": home,
        "away_team": away,
        "odd_over25_current": over_value,
        "odd_under25_current": under_value,
        "source": source,
    }


def api_football_fixture_details(item: dict):
    fixture = item.get("fixture")
    teams = item.get("teams")
    league = item.get("league")
    if not isinstance(fixture, dict) or not isinstance(teams, dict):
        return None

    fixture_id = fixture.get("id")
    kickoff = parse_api_datetime(fixture.get("date"))
    home = teams.get("home")
    away = teams.get("away")
    league_id = league.get("id") if isinstance(league, dict) else None

    try:
        fixture_id = int(fixture_id)
        league_id = int(league_id)
    except (TypeError, ValueError):
        return None

    if kickoff is None:
        return None

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

    canonical_league = API_FOOTBALL_LEAGUE_ID_MAP.get(league_id)
    if canonical_league is None or not home_name or not away_name:
        return None

    return fixture_id, kickoff, canonical_league, home_name, away_name


def fetch_api_football_fallback(
    today,
    api_key: str,
    horizon_days: int = 3,
    max_requests: int = 16,
) -> tuple[list[dict], dict]:
    """
    Emergency source with fixtures + pre-match Over/Under 2.5.

    Uses one /fixtures request per date and paginated /odds requests, joined
    by fixture id. The request budget prevents the free daily quota being
    consumed by a single outage.
    """
    if not api_key:
        return [], {"source": "API-Football", "error": "sem chave", "requests": 0}

    observations: dict[str, dict] = {}
    requests_used = 0
    remaining = None

    for offset in range(max(1, horizon_days)):
        if requests_used >= max_requests:
            break

        date_text = (today + timedelta(days=offset)).strftime("%Y-%m-%d")

        try:
            fixture_payload, headers = api_football_get_json(
                "/fixtures",
                {"date": date_text, "timezone": "Europe/Lisbon"},
                api_key,
            )
        except Exception as error:
            return [], {
                "source": "API-Football",
                "error": str(error),
                "requests": requests_used,
                "remaining": remaining,
            }

        requests_used += 1
        remaining = headers.get(
            "x-ratelimit-requests-remaining",
            headers.get("x-ratelimit-remaining", remaining),
        )

        fixture_rows = fixture_payload.get("response", [])
        fixture_by_id: dict[int, tuple] = {}
        if isinstance(fixture_rows, list):
            for item in fixture_rows:
                if not isinstance(item, dict):
                    continue
                details = api_football_fixture_details(item)
                if details is None:
                    continue
                fixture_by_id[details[0]] = details

        if not fixture_by_id or requests_used >= max_requests:
            continue

        page = 1
        total_pages = 1
        while page <= total_pages and requests_used < max_requests:
            try:
                odds_payload, headers = api_football_get_json(
                    "/odds",
                    {
                        "date": date_text,
                        "timezone": "Europe/Lisbon",
                        "page": page,
                    },
                    api_key,
                )
            except Exception as error:
                return list(observations.values()), {
                    "source": "API-Football",
                    "error": str(error),
                    "requests": requests_used,
                    "remaining": remaining,
                }

            requests_used += 1
            remaining = headers.get(
                "x-ratelimit-requests-remaining",
                headers.get("x-ratelimit-remaining", remaining),
            )

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
                    fixture = entry.get("fixture")
                    if not isinstance(fixture, dict):
                        continue
                    try:
                        fixture_id = int(fixture.get("id"))
                    except (TypeError, ValueError):
                        continue

                    details = fixture_by_id.get(fixture_id)
                    if details is None:
                        continue

                    _, kickoff, league, home, away = details
                    over, under, _ = extract_api_football_prematch_totals(
                        {"response": [entry]}
                    )
                    if over is None or under is None:
                        continue

                    observation = build_observation(
                        league=league,
                        kickoff=kickoff,
                        home_team=home,
                        away_team=away,
                        over=over,
                        under=under,
                        source="API-Football / emergency upcoming fallback",
                    )
                    if observation is not None:
                        observations[observation["fixture_key"]] = observation

            page += 1

    return list(observations.values()), {
        "source": "API-Football",
        "requests": requests_used,
        "remaining": remaining,
        "error": None,
    }


def odds_api_quota(api_key: str):
    url = f"{ODDS_API_BASE_URL}/sports/?apiKey={api_key}"
    payload, headers = http_get_json(url)
    active_keys = {
        str(item.get("key"))
        for item in payload
        if isinstance(item, dict) and item.get("active") is True and item.get("key")
    } if isinstance(payload, list) else set()

    try:
        remaining = int(headers.get("x-requests-remaining"))
    except (TypeError, ValueError):
        remaining = None

    return active_keys, remaining


def fetch_odds_api_fallback(
    today,
    api_key: str,
    reserve_credits: int = 100,
    max_calls: int = 18,
) -> tuple[list[dict], dict]:
    """
    Last-resort emergency source.

    /sports is quota-free and is used first to inspect available leagues and
    remaining credits. Totals requests stop before the configured reserve so
    closing-odds capture keeps priority.
    """
    if not api_key:
        return [], {"source": "The Odds API", "error": "sem chave", "requests": 0}

    try:
        active_keys, remaining = odds_api_quota(api_key)
    except Exception as error:
        return [], {
            "source": "The Odds API",
            "error": str(error),
            "requests": 0,
        }

    reverse = {sport_key: league for league, sport_key in LEAGUE_SPORT_KEY.items()}
    observations: dict[str, dict] = {}
    calls = 0
    last_remaining = remaining
    now = datetime.now(APP_TIMEZONE)

    for league in ODDS_API_PRIORITY:
        if calls >= max_calls:
            break

        sport_key = LEAGUE_SPORT_KEY.get(league)
        if not sport_key or sport_key not in active_keys:
            continue

        if last_remaining is not None and last_remaining <= reserve_credits:
            break

        url = (
            f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds/"
            f"?apiKey={api_key}&regions=eu&markets=totals"
            f"&oddsFormat=decimal&dateFormat=iso"
        )

        try:
            payload, headers = http_get_json(url)
        except Exception as error:
            print(f"⚠ The Odds API fallback {league}: {error}")
            continue

        calls += 1
        try:
            last_remaining = int(headers.get("x-requests-remaining"))
        except (TypeError, ValueError):
            pass

        if not isinstance(payload, list):
            continue

        canonical_league = reverse.get(sport_key, league)

        for event in payload:
            if not isinstance(event, dict):
                continue
            kickoff = parse_api_datetime(event.get("commence_time"))
            if kickoff is None or kickoff < now:
                continue
            if kickoff.date() < today:
                continue

            over, under = extract_odds_api_totals(event)
            if over is None or under is None:
                continue

            observation = build_observation(
                league=canonical_league,
                kickoff=kickoff,
                home_team=event.get("home_team"),
                away_team=event.get("away_team"),
                over=over,
                under=under,
                source="The Odds API / emergency upcoming fallback",
            )
            if observation is not None:
                observations[observation["fixture_key"]] = observation

    return list(observations.values()), {
        "source": "The Odds API",
        "requests": calls,
        "remaining_before": remaining,
        "remaining_after": last_remaining,
        "reserve": reserve_credits,
        "error": None,
    }


def fetch_emergency_upcoming(today):
    """
    Fallback chain used only after Football-Data.co.uk is unavailable.
    """
    api_football_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    odds_api_key = os.getenv("ODDS_API_KEY", "").strip()

    try:
        api_horizon = max(1, int(os.getenv("API_FOOTBALL_FALLBACK_DAYS", "3")))
    except ValueError:
        api_horizon = 3

    try:
        api_budget = max(2, int(os.getenv("API_FOOTBALL_FALLBACK_MAX_REQUESTS", "16")))
    except ValueError:
        api_budget = 16

    try:
        odds_reserve = max(0, int(os.getenv("ODDS_API_UPCOMING_RESERVE", "100")))
    except ValueError:
        odds_reserve = 100

    try:
        odds_calls = max(1, int(os.getenv("ODDS_API_UPCOMING_MAX_CALLS", "18")))
    except ValueError:
        odds_calls = 18

    observations, info = fetch_api_football_fallback(
        today,
        api_football_key,
        horizon_days=api_horizon,
        max_requests=api_budget,
    )
    if observations:
        return observations, info

    print(
        "⚠ API-Football fallback sem dados: "
        + str(info.get("error") or "nenhum jogo/odd disponível")
    )

    observations, odds_info = fetch_odds_api_fallback(
        today,
        odds_api_key,
        reserve_credits=odds_reserve,
        max_calls=odds_calls,
    )
    return observations, {
        "api_football": info,
        "odds_api": odds_info,
        "source": odds_info.get("source"),
        "error": odds_info.get("error"),
    }
