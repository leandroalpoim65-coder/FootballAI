from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import json

import pandas as pd

from update_results_cloud import (
    canonical_team,
    connect_turso,
    fetch_dicts,
    find_paper_table,
    parse_db_date,
    safe_int,
    table_columns,
    team_names_match,
    update_trade,
)


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
LOOKBACK_DAYS = 30

SOFASCORE_BASE_URLS = (
    "https://www.sofascore.com/api/v1",
    "https://api.sofascore.com/api/v1",
)


# ============================================================
# HTTP
# ============================================================

def fetch_json(url: str) -> object:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
            "Referer": "https://www.sofascore.com/",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
    except HTTPError as error:
        try:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
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

    try:
        return json.loads(
            content.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            "Resposta JSON inválida."
        ) from error


# ============================================================
# SOFASCORE
# ============================================================

def fetch_sofascore_day(
    fixture_date: date,
) -> tuple[list[dict], str | None]:
    date_text = fixture_date.isoformat()
    errors: list[str] = []

    for base_url in SOFASCORE_BASE_URLS:
        url = (
            f"{base_url}/sport/football/"
            f"scheduled-events/{date_text}"
        )

        try:
            payload = fetch_json(url)
        except Exception as error:
            errors.append(
                f"{base_url}: {error}"
            )
            continue

        if not isinstance(payload, dict):
            errors.append(
                f"{base_url}: resposta inesperada"
            )
            continue

        events = payload.get("events", [])
        if not isinstance(events, list):
            errors.append(
                f"{base_url}: campo events inválido"
            )
            continue

        return (
            [
                event
                for event in events
                if isinstance(event, dict)
            ],
            None,
        )

    return [], " | ".join(errors)


def sofascore_event_finished(event: dict) -> bool:
    status = event.get("status")
    if not isinstance(status, dict):
        return False

    status_type = str(
        status.get("type") or ""
    ).strip().casefold()

    if status_type in {
        "finished",
        "afterpenalties",
        "afterextratime",
    }:
        return True

    try:
        return int(status.get("code")) == 100
    except (TypeError, ValueError):
        return False


def sofascore_event_local_date(
    event: dict,
) -> date | None:
    timestamp = event.get("startTimestamp")

    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return None

    try:
        return datetime.fromtimestamp(
            value,
            tz=timezone.utc,
        ).astimezone(
            APP_TIMEZONE
        ).date()
    except (
        OverflowError,
        OSError,
        ValueError,
    ):
        return None


def sofascore_team_name(
    event: dict,
    side: str,
) -> str | None:
    key = (
        "homeTeam"
        if side.casefold() == "home"
        else "awayTeam"
    )

    team = event.get(key)
    if not isinstance(team, dict):
        return None

    for field in (
        "name",
        "shortName",
        "nameCode",
    ):
        value = team.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()

    return None


def canonical_sofascore_team(
    league: str | None,
    value: object,
) -> str:
    canonical = canonical_team(
        league,
        value,
    )

    aliases = {
        "1 nurnberg": "nurnberg",
        "arminia bielefeld": "bielefeld",
        "real sociedad": "sociedad",
        "real oviedo": "oviedo",
        "rcd espanyol": "espanol",
        "espanyol": "espanol",
        "standard liege": "standard",
        "standard de liege": "standard",
        "oh leuven": "oud heverlee leuven",
        "oud heverlee leuven": "oud heverlee leuven",
        "caykur rizespor": "rizespor",
        "goztepe sk": "goztepe",
        "stade brestois 29": "brest",
        "stade brestois": "brest",
        "aj auxerre": "auxerre",
        "angers sco": "angers",
        "estac troyes": "troyes",
        "sporting gijon": "sporting gijon",
        "sporting de gijon": "sporting gijon",
        "sp gijon": "sporting gijon",
        "sporting cp": "sporting lisbon",
        "sporting clube de portugal": "sporting lisbon",
        "sp lisbon": "sporting lisbon",
        "sporting lisbon": "sporting lisbon",
        "paris saint germain": "psg",
        "paris sg": "psg",
        "psg": "psg",
    }

    return aliases.get(
        canonical,
        canonical,
    )


def sofascore_team_match(
    league: str | None,
    left: object,
    right: object,
) -> bool:
    if team_names_match(
        league,
        left,
        right,
    ):
        return True

    a = canonical_sofascore_team(
        league,
        left,
    )
    b = canonical_sofascore_team(
        league,
        right,
    )

    return bool(a and b and a == b)


def find_sofascore_event(
    events: list[dict],
    league: str,
    fixture_date: date,
    home_team: str,
    away_team: str,
) -> dict | None:
    matches: list[dict] = []

    for event in events:
        if not sofascore_event_finished(event):
            continue

        event_date = sofascore_event_local_date(
            event
        )

        if (
            event_date is not None
            and event_date != fixture_date
        ):
            continue

        api_home = sofascore_team_name(
            event,
            "home",
        )
        api_away = sofascore_team_name(
            event,
            "away",
        )

        if not sofascore_team_match(
            league,
            api_home,
            home_team,
        ):
            continue

        if not sofascore_team_match(
            league,
            api_away,
            away_team,
        ):
            continue

        matches.append(event)

    return matches[0] if len(matches) == 1 else None


def sofascore_score(
    event: dict,
    side: str,
) -> int | None:
    key = (
        "homeScore"
        if side.casefold() == "home"
        else "awayScore"
    )

    score = event.get(key)
    if not isinstance(score, dict):
        return None

    for field in (
        "current",
        "display",
        "normaltime",
    ):
        value = safe_int(
            score.get(field)
        )
        if value is not None:
            return value

    return None


def sofascore_tournament_text(
    event: dict,
) -> str:
    tournament = event.get("tournament")
    if not isinstance(tournament, dict):
        return ""

    names: list[str] = []

    for value in (
        tournament.get("name"),
        tournament.get("slug"),
    ):
        if value:
            names.append(str(value))

    unique = tournament.get("uniqueTournament")
    if isinstance(unique, dict):
        for value in (
            unique.get("name"),
            unique.get("slug"),
        ):
            if value:
                names.append(str(value))

        category = unique.get("category")
        if isinstance(category, dict):
            value = category.get("name")
            if value:
                names.append(str(value))

    return " / ".join(names)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — SOFASCORE RESULTS FALLBACK")
    print("==========================================")
    print()

    today = datetime.now(
        APP_TIMEZONE
    ).date()
    cutoff = today - timedelta(
        days=LOOKBACK_DAYS
    )

    try:
        connection = connect_turso()
    except Exception as error:
        print(
            f"❌ Não foi possível ligar à Turso: {error}"
        )
        raise SystemExit(1) from error

    try:
        table = find_paper_table(
            connection
        )

        if table is None:
            print(
                "❌ Tabela de Paper Trade não encontrada."
            )
            return

        columns = table_columns(
            connection,
            table,
        )

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
            return

        rows = fetch_dicts(
            connection,
            f"""
            SELECT *
            FROM {table}
            WHERE COALESCE(status, 'OPEN') != 'RESOLVED'
              AND date({fixture_date_column}) <= date(?)
              AND date({fixture_date_column}) >= date(?)
            ORDER BY {fixture_date_column}, id
            """,
            (
                today.isoformat(),
                cutoff.isoformat(),
            ),
        )

        print(
            f"OPEN a verificar          : {len(rows)}"
        )
        print(
            f"Janela                    : {cutoff} → {today}"
        )

        if not rows:
            print(
                "✅ Não existem resultados por recuperar."
            )
            return

        dates_needed = sorted(
            {
                fixture_date
                for row in rows
                if (
                    fixture_date := parse_db_date(
                        row.get(fixture_date_column)
                    )
                )
                is not None
            }
        )

        events_by_date: dict[
            date,
            list[dict],
        ] = {}
        errors: list[str] = []

        print()
        print("Sofascore:")

        for fixture_date in dates_needed:
            events, error = fetch_sofascore_day(
                fixture_date
            )

            if error is not None:
                errors.append(
                    f"{fixture_date}: {error}"
                )
                print(
                    f"  ⚠ {fixture_date}: falhou"
                )
                continue

            events_by_date[
                fixture_date
            ] = events

            finished_count = sum(
                1
                for event in events
                if sofascore_event_finished(event)
            )

            print(
                f"  ✅ {fixture_date}: "
                f"{len(events)} eventos | "
                f"{finished_count} terminados"
            )

        resolved = 0
        unmatched: list[dict] = []

        print()
        print("Resultados encontrados:")

        for row in rows:
            fixture_date = parse_db_date(
                row.get(fixture_date_column)
            )

            if fixture_date is None:
                unmatched.append(row)
                continue

            league = str(
                row.get("league") or ""
            ).strip()
            home_team = str(
                row.get("home_team") or ""
            ).strip()
            away_team = str(
                row.get("away_team") or ""
            ).strip()

            event = find_sofascore_event(
                events_by_date.get(
                    fixture_date,
                    [],
                ),
                league,
                fixture_date,
                home_team,
                away_team,
            )

            if event is None:
                unmatched.append(row)
                continue

            home_goals = sofascore_score(
                event,
                "home",
            )
            away_goals = sofascore_score(
                event,
                "away",
            )

            if (
                home_goals is None
                or away_goals is None
            ):
                unmatched.append(row)
                continue

            update_trade(
                connection,
                table,
                row,
                columns,
                home_goals,
                away_goals,
            )

            resolved += 1

            tournament = sofascore_tournament_text(
                event
            )

            print(
                f"✅ [Sofascore] {fixture_date} | "
                f"{home_team} {home_goals}-{away_goals} "
                f"{away_team}"
                + (
                    f" | {tournament}"
                    if tournament
                    else ""
                )
            )

        connection.commit()

        print()
        print("==========================================")
        print(" RESUMO SOFASCORE")
        print("==========================================")
        print(
            f"Resolvidos Sofascore      : {resolved}"
        )
        print(
            f"Ainda sem correspondência : {len(unmatched)}"
        )

        if errors:
            print()
            print("Avisos Sofascore:")
            for error in errors[:10]:
                print(f"  - {error}")

        if unmatched:
            print()
            print("Ainda por resolver no Sofascore:")
            for row in unmatched[:30]:
                print(
                    f"  - {row.get(fixture_date_column)} | "
                    f"{row.get('league')} | "
                    f"{row.get('home_team')} vs "
                    f"{row.get('away_team')}"
                )

        print()
        print(
            "Este fallback usa apenas jogos marcados como terminados "
            "e exige correspondência única de equipa da casa e visitante."
        )
        print(
            "O endpoint usado pelo site Sofascore é não-oficial e pode "
            "mudar; se falhar, o workflow continua para as outras fontes."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
