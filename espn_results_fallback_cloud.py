from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from update_results_cloud import (
    connect_turso,
    fetch_dicts,
    find_paper_table,
    parse_db_date,
    safe_int,
    table_columns,
    update_trade,
)


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
LOOKBACK_DAYS = 7
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

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

ALIASES = {
    "1 fc union berlin": "union berlin",
    "fc union berlin": "union berlin",
    "union berlin": "union berlin",
    "ein frankfurt": "eintracht frankfurt",
    "eintracht frankfurt": "eintracht frankfurt",
    "1 fsv mainz 05": "mainz",
    "fsv mainz 05": "mainz",
    "mainz 05": "mainz",
    "mainz": "mainz",
    "sc paderborn 07": "paderborn",
    "paderborn 07": "paderborn",
    "paderborn": "paderborn",
    "1 fc heidenheim 1846": "heidenheim",
    "fc heidenheim": "heidenheim",
    "heidenheim": "heidenheim",
    "dynamo dresden": "dresden",
    "sg dynamo dresden": "dresden",
    "dresden": "dresden",
    "1 fc nurnberg": "nurnberg",
    "fc nurnberg": "nurnberg",
    "nurnberg": "nurnberg",
    "arminia bielefeld": "bielefeld",
    "bielefeld": "bielefeld",
    "eintracht braunschweig": "braunschweig",
    "braunschweig": "braunschweig",
    "hertha bsc": "hertha",
    "hertha berlin": "hertha",
    "hertha": "hertha",
    "vfl bochum": "bochum",
    "bochum": "bochum",
    "vfl osnabruck": "osnabruck",
    "osnabruck": "osnabruck",
    "charlton athletic": "charlton",
    "charlton": "charlton",
    "preston north end": "preston",
    "preston": "preston",
    "real sociedad": "sociedad",
    "sociedad": "sociedad",
    "rcd espanyol": "espanol",
    "espanyol": "espanol",
    "espanol": "espanol",
    "real oviedo": "oviedo",
    "oviedo": "oviedo",
    "atletico madrid": "ath madrid",
    "atletico de madrid": "ath madrid",
    "ath madrid": "ath madrid",
    "paris saint germain": "paris sg",
    "paris saint germain fc": "paris sg",
    "paris sg": "paris sg",
    "psg": "paris sg",
    "oh leuven": "oud heverlee leuven",
    "oud heverlee leuven": "oud heverlee leuven",
    "standard liege": "standard",
    "standard de liege": "standard",
    "standard": "standard",
    "caykur rizespor": "rizespor",
    "rizespor": "rizespor",
    "goztepe": "goztep",
    "goztepe sk": "goztep",
    "goztep": "goztep",
    "aj auxerre": "auxerre",
    "auxerre": "auxerre",
    "pec zwolle": "zwolle",
    "zwolle": "zwolle",
    "nec nijmegen": "nijmegen",
    "nijmegen": "nijmegen",
    "sc braga": "sp braga",
    "sporting braga": "sp braga",
    "sp braga": "sp braga",
}

STOPWORDS = {
    "fc",
    "cf",
    "afc",
    "sc",
    "club",
    "fk",
    "ac",
    "sv",
    "rcd",
    "cd",
}

MIN_TEAM_SIMILARITY = 0.72
MIN_COMBINED_SIMILARITY = 1.60
MIN_MARGIN = 0.10


def fetch_json(url: str) -> object:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 FootballAI/Cloud",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
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
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Resposta JSON inválida.") from error


def normalize_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [
        token
        for token in text.split()
        if token not in STOPWORDS
    ]
    normalized = " ".join(tokens).strip()
    return ALIASES.get(normalized, normalized)


def similarity(left: object, right: object) -> float:
    a = normalize_name(left)
    b = normalize_name(right)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        if longer > 0 and shorter / longer >= 0.55:
            return max(
                0.86,
                SequenceMatcher(None, a, b).ratio(),
            )

    return SequenceMatcher(None, a, b).ratio()


def fetch_scoreboard(
    league_slug: str,
    fixture_date: date,
) -> tuple[list[dict], str | None]:
    url = (
        f"{ESPN_BASE_URL}/{league_slug}/scoreboard"
        f"?dates={fixture_date.strftime('%Y%m%d')}"
    )

    try:
        payload = fetch_json(url)
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


def event_completed(event: dict) -> bool:
    status = event.get("status")
    if not isinstance(status, dict):
        return False

    status_type = status.get("type")
    if not isinstance(status_type, dict):
        return False

    if bool(status_type.get("completed")):
        return True

    state = str(status_type.get("state") or "").casefold()
    name = str(status_type.get("name") or "").casefold()
    return state == "post" or "final" in name


def competitors(event: dict) -> list[dict]:
    competitions = event.get("competitions", [])
    if not isinstance(competitions, list) or not competitions:
        return []

    competition = competitions[0]
    if not isinstance(competition, dict):
        return []

    values = competition.get("competitors", [])
    if not isinstance(values, list):
        return []

    return [
        item
        for item in values
        if isinstance(item, dict)
    ]


def competitor(event: dict, side: str) -> dict | None:
    found = [
        item
        for item in competitors(event)
        if str(item.get("homeAway") or "").casefold()
        == side.casefold()
    ]
    return found[0] if len(found) == 1 else None


def team_names(comp: dict | None) -> list[str]:
    if comp is None:
        return []

    team = comp.get("team")
    if not isinstance(team, dict):
        return []

    values: list[str] = []
    for field in (
        "displayName",
        "shortDisplayName",
        "name",
        "location",
        "abbreviation",
    ):
        value = team.get(field)
        if value is not None and str(value).strip():
            values.append(str(value).strip())

    return values


def best_team_similarity(
    requested: str,
    candidates: list[str],
) -> float:
    if not candidates:
        return 0.0
    return max(
        similarity(requested, candidate)
        for candidate in candidates
    )


def event_score(
    event: dict,
    home_team: str,
    away_team: str,
) -> tuple[float, float, float]:
    home = competitor(event, "home")
    away = competitor(event, "away")

    home_sim = best_team_similarity(
        home_team,
        team_names(home),
    )
    away_sim = best_team_similarity(
        away_team,
        team_names(away),
    )

    return (
        home_sim,
        away_sim,
        home_sim + away_sim,
    )


def find_event(
    events: list[dict],
    home_team: str,
    away_team: str,
) -> tuple[dict | None, str]:
    completed = [
        event
        for event in events
        if event_completed(event)
    ]

    if not completed:
        return None, "sem eventos terminados"

    ranked: list[
        tuple[float, float, float, dict]
    ] = []

    for event in completed:
        home_sim, away_sim, combined = event_score(
            event,
            home_team,
            away_team,
        )
        ranked.append(
            (combined, home_sim, away_sim, event)
        )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_combined, best_home, best_away, best_event = ranked[0]
    second_combined = (
        ranked[1][0]
        if len(ranked) > 1
        else 0.0
    )

    best_home_name = (
        team_names(competitor(best_event, "home"))[0]
        if team_names(competitor(best_event, "home"))
        else "?"
    )
    best_away_name = (
        team_names(competitor(best_event, "away"))[0]
        if team_names(competitor(best_event, "away"))
        else "?"
    )

    debug = (
        f"melhor: {best_home_name} vs {best_away_name} "
        f"| sim={best_home:.2f}/{best_away:.2f} "
        f"| total={best_combined:.2f} "
        f"| margem={best_combined-second_combined:.2f}"
    )

    if (
        best_home < MIN_TEAM_SIMILARITY
        or best_away < MIN_TEAM_SIMILARITY
        or best_combined < MIN_COMBINED_SIMILARITY
    ):
        return None, debug

    if (
        len(ranked) > 1
        and best_combined - second_combined < MIN_MARGIN
        and best_combined < 1.90
    ):
        return None, debug + " | ambíguo"

    return best_event, debug


def extract_score(event: dict, side: str) -> int | None:
    comp = competitor(event, side)
    if comp is None:
        return None

    score = safe_int(comp.get("score"))
    if score is not None:
        return score

    score_obj = comp.get("score")
    if isinstance(score_obj, dict):
        for key in (
            "value",
            "displayValue",
            "current",
        ):
            value = safe_int(score_obj.get(key))
            if value is not None:
                return value

    return None


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — ESPN ROBUST RESULTS FALLBACK")
    print("==========================================")
    print()

    today = datetime.now(APP_TIMEZONE).date()
    cutoff = today - timedelta(days=LOOKBACK_DAYS)

    try:
        connection = connect_turso()
    except Exception as error:
        print(f"❌ Não foi possível ligar à Turso: {error}")
        raise SystemExit(1) from error

    try:
        table = find_paper_table(connection)
        if table is None:
            print("❌ Tabela de Paper Trade não encontrada.")
            return

        columns = table_columns(connection, table)
        fixture_date_column = (
            "fixture_date"
            if "fixture_date" in columns
            else "date"
            if "date" in columns
            else None
        )

        if fixture_date_column is None:
            print("❌ A tabela não tem fixture_date/date.")
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
            (today.isoformat(), cutoff.isoformat()),
        )

        print(f"OPEN a verificar          : {len(rows)}")
        print(f"Janela                    : {cutoff} → {today}")

        needed: set[tuple[str, date]] = set()
        row_meta: dict[int, tuple[str, date, str]] = {}

        for row in rows:
            league = str(row.get("league") or "").strip()
            fixture_date = parse_db_date(
                row.get(fixture_date_column)
            )
            slug = ESPN_LEAGUE_MAP.get(league)

            if fixture_date is None or not slug:
                continue

            row_meta[int(row["id"])] = (
                league,
                fixture_date,
                slug,
            )
            needed.add((slug, fixture_date))

        events_by_key: dict[
            tuple[str, date],
            list[dict],
        ] = {}

        print()
        print("ESPN:")

        for slug, fixture_date in sorted(
            needed,
            key=lambda item: (item[1], item[0]),
        ):
            events, error = fetch_scoreboard(
                slug,
                fixture_date,
            )

            if error is not None:
                print(
                    f"  ⚠ {slug} {fixture_date}: {error}"
                )
                continue

            events_by_key[(slug, fixture_date)] = events
            completed_count = sum(
                1
                for event in events
                if event_completed(event)
            )

            print(
                f"  ✅ {slug} {fixture_date}: "
                f"{len(events)} eventos | "
                f"{completed_count} terminados"
            )

        resolved = 0
        unresolved: list[str] = []

        print()
        print("Resultados encontrados:")

        for row in rows:
            row_id = int(row["id"])
            meta = row_meta.get(row_id)

            if meta is None:
                unresolved.append(
                    f"{row.get(fixture_date_column)} | "
                    f"{row.get('league')} | "
                    f"{row.get('home_team')} vs {row.get('away_team')} "
                    f"| liga ESPN não mapeada"
                )
                continue

            league, fixture_date, slug = meta
            event, debug = find_event(
                events_by_key.get((slug, fixture_date), []),
                str(row.get("home_team") or ""),
                str(row.get("away_team") or ""),
            )

            if event is None:
                unresolved.append(
                    f"{fixture_date} | {league} | "
                    f"{row.get('home_team')} vs {row.get('away_team')} "
                    f"| {debug}"
                )
                continue

            home_goals = extract_score(event, "home")
            away_goals = extract_score(event, "away")

            if home_goals is None or away_goals is None:
                unresolved.append(
                    f"{fixture_date} | {league} | "
                    f"{row.get('home_team')} vs {row.get('away_team')} "
                    f"| score inválido | {debug}"
                )
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
            print(
                f"✅ [ESPN robusto] {fixture_date} | "
                f"{row.get('home_team')} {home_goals}-{away_goals} "
                f"{row.get('away_team')} | {debug}"
            )

        connection.commit()

        print()
        print("==========================================")
        print(" RESUMO ESPN ROBUSTO")
        print("==========================================")
        print(f"Resolvidos ESPN robusto   : {resolved}")
        print(f"Ainda sem correspondência : {len(unresolved)}")

        if unresolved:
            print()
            print("Ainda por resolver no ESPN robusto:")
            for item in unresolved[:40]:
                print(f"  - {item}")

        print()
        print(
            "Matching: aliases + fuzzy matching conservador, "
            "sempre limitado à mesma liga e data."
        )
        print(
            "Só eventos ESPN terminados são aceites."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
