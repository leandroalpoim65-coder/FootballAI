from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import libsql

from agent.uefa_competitions import UEFA_COMPETITIONS
from closing_odds_v3_cloud import api_football_get_json, parse_api_datetime


TABLE = "uefa_matches"
RUNS_TABLE = "uefa_sync_runs"
DEFAULT_SEASONS_BACK = 5
APP_TIMEZONE = ZoneInfo("Europe/Lisbon")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect():
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("Turso não configurada.")
    return libsql.connect(database=url, auth_token=token)


def ensure_tables(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            fixture_id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT,
            season TEXT NOT NULL,
            league TEXT NOT NULL,
            league_id INTEGER NOT NULL,
            round TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL,
            status TEXT,
            source TEXT NOT NULL,
            synced_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{TABLE}_league_date
        ON {TABLE}(league, date)
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            synced_at TEXT NOT NULL,
            seasons_json TEXT NOT NULL,
            competitions INTEGER NOT NULL,
            requests_used INTEGER NOT NULL,
            rows_seen INTEGER NOT NULL,
            rows_saved INTEGER NOT NULL,
            errors INTEGER NOT NULL
        )
        """
    )
    connection.commit()


def current_season_start() -> int:
    today = datetime.now(APP_TIMEZONE).date()
    return today.year if today.month >= 7 else today.year - 1


def seasons_to_sync() -> list[int]:
    explicit = os.getenv("UEFA_HISTORY_SEASONS", "").strip()
    if explicit:
        values = []
        for item in explicit.split(","):
            item = item.strip()
            if item.isdigit():
                values.append(int(item))
        if values:
            return sorted(set(values))
    back = max(1, int(os.getenv("UEFA_HISTORY_SEASONS_BACK", DEFAULT_SEASONS_BACK)))
    current = current_season_start()
    return list(range(current - back, current + 1))


def completed_match(item: dict, expected_league_id: int, expected_league: str) -> dict | None:
    fixture = item.get("fixture") or {}
    league = item.get("league") or {}
    teams = item.get("teams") or {}
    goals = item.get("goals") or {}

    try:
        fixture_id = int(fixture.get("id"))
        league_id = int(league.get("id"))
    except (TypeError, ValueError):
        return None
    if league_id != expected_league_id:
        return None

    status = str((fixture.get("status") or {}).get("short") or "").upper()
    if status not in {"FT", "AET", "PEN"}:
        return None

    kickoff = parse_api_datetime(fixture.get("date"))
    if kickoff is None:
        return None

    home = (teams.get("home") or {}).get("name")
    away = (teams.get("away") or {}).get("name")
    try:
        home_goals = int(goals.get("home"))
        away_goals = int(goals.get("away"))
    except (TypeError, ValueError):
        return None
    if not home or not away:
        return None

    season_value = league.get("season")
    try:
        season_start = int(season_value)
        season = f"{season_start}/{season_start + 1}"
    except (TypeError, ValueError):
        season = str(season_value or "")

    local = kickoff.astimezone(APP_TIMEZONE)
    return {
        "fixture_id": fixture_id,
        "date": local.strftime("%Y-%m-%d"),
        "time": local.strftime("%H:%M"),
        "season": season,
        "league": expected_league,
        "league_id": expected_league_id,
        "round": str(league.get("round") or ""),
        "home_team": str(home).strip(),
        "away_team": str(away).strip(),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "status": status,
        "source": "API-Football historical fixtures",
    }


def upsert(connection, row: dict, synced_at: str) -> None:
    connection.execute(
        f"""
        INSERT INTO {TABLE} (
            fixture_id,date,time,season,league,league_id,round,
            home_team,away_team,home_goals,away_goals,status,source,synced_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(fixture_id) DO UPDATE SET
            date=excluded.date,
            time=excluded.time,
            season=excluded.season,
            league=excluded.league,
            league_id=excluded.league_id,
            round=excluded.round,
            home_team=excluded.home_team,
            away_team=excluded.away_team,
            home_goals=excluded.home_goals,
            away_goals=excluded.away_goals,
            status=excluded.status,
            source=excluded.source,
            synced_at=excluded.synced_at
        """,
        (
            row["fixture_id"], row["date"], row["time"], row["season"],
            row["league"], row["league_id"], row["round"], row["home_team"],
            row["away_team"], row["home_goals"], row["away_goals"],
            row["status"], row["source"], synced_at,
        ),
    )


def main() -> None:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY não configurada.")

    seasons = seasons_to_sync()
    connection = connect()
    synced_at = now_iso()
    requests_used = rows_seen = rows_saved = errors = 0
    try:
        ensure_tables(connection)
        print("FOOTBALLAI — UEFA HISTORY SYNC")
        print("Épocas:", ", ".join(str(x) for x in seasons))

        for league_id, league_name in UEFA_COMPETITIONS.items():
            for season in seasons:
                try:
                    payload, _ = api_football_get_json(
                        "/fixtures",
                        {"league": league_id, "season": season, "timezone": "Europe/Lisbon"},
                        key,
                    )
                    requests_used += 1
                except Exception as exc:
                    errors += 1
                    print(f"ERRO {league_name} {season}: {exc}")
                    continue

                response = payload.get("response", [])
                if not isinstance(response, list):
                    continue
                rows_seen += len(response)
                saved_here = 0
                for item in response:
                    if not isinstance(item, dict):
                        continue
                    row = completed_match(item, league_id, league_name)
                    if row is None:
                        continue
                    upsert(connection, row, synced_at)
                    rows_saved += 1
                    saved_here += 1
                connection.commit()
                print(f"{league_name:24s} {season}: {saved_here} jogos concluídos")

        connection.execute(
            f"INSERT INTO {RUNS_TABLE} (synced_at,seasons_json,competitions,requests_used,rows_seen,rows_saved,errors) VALUES (?,?,?,?,?,?,?)",
            (synced_at, json.dumps(seasons), len(UEFA_COMPETITIONS), requests_used, rows_seen, rows_saved, errors),
        )
        connection.commit()
        print(f"Guardados/atualizados: {rows_saved} | requests: {requests_used} | erros: {errors}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
