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

# Só aceitamos fuzzy matching quando ambos os nomes das equipas têm
# uma semelhança elevada. Além disso, o melhor candidato tem de ser
# claramente melhor do que o segundo melhor. Isto evita resolver um
# Paper Trade com o jogo errado.
MIN_TEAM_SIMILARITY = 0.78
MIN_COMBINED_SIMILARITY = 1.66
MIN_WINNER_MARGIN = 0.08


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
            "Origin": "https://www.sofascore.com",
            "Cache-Control": "no-cache",
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
# SOFASCORE — DOWNLOAD
# ============================================================

def fetch_sofascore_day(
    fixture_date: date,
) -> tuple[list[dict], str | None, str | None]:
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

        clean_events = [
            event
            for event in events
            if isinstance(event, dict)
        ]

        return clean_events, None, base_url

    return [], " | ".join(errors), None


# ============================================================
# SOFASCORE — EVENTOS / SCORE
# ============================================================

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
# NORMALIZAÇÃO EXTRA PARA SOFASCORE
# ============================================================

def simple_name(value: object) -> str:
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
    text = text.casefold()
    text = text.replace("&", " and ")

    text = re.sub(
        r"\b(fc|cf|afc|sc|ac|sv|fk|sk|kv|rc|cd|ud|sd|ssc)\b",
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

    # Alguns clubes alemães aparecem com um "1" inicial.
    text = re.sub(r"^1\s+", "", text)

    return text


def canonical_sofascore_team(
    league: str | None,
    value: object,
) -> str:
    canonical = canonical_team(
        league,
        value,
    )
    canonical = simple_name(canonical)

    aliases = {
        "nurnberg": "nurnberg",
        "1 nurnberg": "nurnberg",
        "arminia bielefeld": "bielefeld",
        "bielefeld": "bielefeld",
        "real sociedad": "sociedad",
        "sociedad": "sociedad",
        "real oviedo": "oviedo",
        "oviedo": "oviedo",
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
        "goztep": "goztepe",
        "stade brestois 29": "brest",
        "stade brestois": "brest",
        "brest": "brest",
        "aj auxerre": "auxerre",
        "auxerre": "auxerre",
        "angers sco": "angers",
        "angers": "angers",
        "estac troyes": "troyes",
        "troyes": "troyes",
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
        "eintracht frankfurt": "ein frankfurt",
        "ein frankfurt": "ein frankfurt",
        "union berlin": "union berlin",
        "mainz 05": "mainz",
        "fsv mainz 05": "mainz",
        "mainz": "mainz",
        "paderborn 07": "paderborn",
        "paderborn": "paderborn",
        "heidenheim 1846": "heidenheim",
        "heidenheim": "heidenheim",
        "dynamo dresden": "dresden",
        "dresden": "dresden",
        "charlton athletic": "charlton",
        "charlton": "charlton",
        "preston north end": "preston",
        "preston": "preston",
        "real betis": "betis",
        "betis": "betis",
        "kv kortrijk": "kortrijk",
        "kortrijk": "kortrijk",
        "royal charleroi": "charleroi",
        "sporting charleroi": "charleroi",
        "charleroi": "charleroi",
        "albacete balompie": "albacete",
        "albacete": "albacete",
        "pec zwolle": "zwolle",
        "zwolle": "zwolle",
        "nec nijmegen": "nijmegen",
        "nijmegen": "nijmegen",
        "atletico madrid": "ath madrid",
        "athletic madrid": "ath madrid",
        "ath madrid": "ath madrid",
        "ado den haag": "den haag",
        "den haag": "den haag",
        "psv eindhoven": "psv eindhoven",
        "psv": "psv eindhoven",
    }

    return aliases.get(
        canonical,
        canonical,
    )


def token_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if len(a) >= 4 and len(b) >= 4:
        if a in b or b in a:
            return 0.94

    seq = SequenceMatcher(
        None,
        a,
        b,
    ).ratio()

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if tokens_a and tokens_b:
        intersection = len(
            tokens_a & tokens_b
        )
        union = len(
            tokens_a | tokens_b
        )
        jaccard = (
            intersection / union
            if union
            else 0.0
        )
    else:
        jaccard = 0.0

    return max(
        seq,
        jaccard,
    )


def team_similarity(
    league: str | None,
    left: object,
    right: object,
) -> float:
    if team_names_match(
        league,
        left,
        right,
    ):
        return 1.0

    a = canonical_sofascore_team(
        league,
        left,
    )
    b = canonical_sofascore_team(
        league,
        right,
    )

    return token_similarity(a, b)


# ============================================================
# MATCHING SEGURO
# ============================================================

def ranked_sofascore_candidates(
    events: list[dict],
    league: str,
    fixture_date: date,
    home_team: str,
    away_team: str,
) -> list[tuple[float, float, float, dict]]:
    ranked: list[
        tuple[float, float, float, dict]
    ] = []

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

        home_similarity = team_similarity(
            league,
            api_home,
            home_team,
        )
        away_similarity = team_similarity(
            league,
            api_away,
            away_team,
        )

        combined = (
            home_similarity
            + away_similarity
        )

        ranked.append(
            (
                combined,
                home_similarity,
                away_similarity,
                event,
            )
        )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return ranked


def find_sofascore_event(
    events: list[dict],
    league: str,
    fixture_date: date,
    home_team: str,
    away_team: str,
) -> tuple[dict | None, str]:
    ranked = ranked_sofascore_candidates(
        events,
        league,
        fixture_date,
        home_team,
        away_team,
    )

    if not ranked:
        return None, "sem jogos terminados nesse dia"

    best_combined, best_home, best_away, best_event = (
        ranked[0]
    )

    api_home = sofascore_team_name(
        best_event,
        "home",
    )
    api_away = sofascore_team_name(
        best_event,
        "away",
    )

    if (
        best_home < MIN_TEAM_SIMILARITY
        or best_away < MIN_TEAM_SIMILARITY
        or best_combined < MIN_COMBINED_SIMILARITY
    ):
        return (
            None,
            (
                f"melhor candidato: {api_home} vs {api_away} "
                f"| sim={best_home:.2f}/{best_away:.2f}"
            ),
        )

    if len(ranked) > 1:
        second_combined = ranked[1][0]
        if (
            best_combined - second_combined
            < MIN_WINNER_MARGIN
        ):
            return (
                None,
                (
                    f"matching ambíguo: {api_home} vs {api_away} "
                    f"| margem={best_combined - second_combined:.2f}"
                ),
            )

    home_goals = sofascore_score(
        best_event,
        "home",
    )
    away_goals = sofascore_score(
        best_event,
        "away",
    )

    if (
        home_goals is None
        or away_goals is None
    ):
        return (
            None,
            f"jogo encontrado sem score final: {api_home} vs {api_away}",
        )

    return (
        best_event,
        (
            f"match: {api_home} vs {api_away} "
            f"| sim={best_home:.2f}/{best_away:.2f}"
        ),
    )


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
            f"OPEN a verificar           : {len(rows)}"
        )
        print(
            f"Janela                     : {cutoff} → {today}"
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
            events, error, source_base = (
                fetch_sofascore_day(
                    fixture_date
                )
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
                f"{finished_count} terminados | "
                f"fonte={source_base}"
            )

        resolved = 0
        unmatched: list[
            tuple[dict, str]
        ] = []

        print()
        print("Resultados encontrados:")

        for row in rows:
            fixture_date = parse_db_date(
                row.get(fixture_date_column)
            )

            if fixture_date is None:
                unmatched.append(
                    (row, "data inválida")
                )
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

            event, reason = find_sofascore_event(
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
                unmatched.append(
                    (row, reason)
                )
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
                unmatched.append(
                    (row, "score final inválido")
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
                + f" | {reason}"
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
            for row, reason in unmatched[:30]:
                print(
                    f"  - {row.get(fixture_date_column)} | "
                    f"{row.get('league')} | "
                    f"{row.get('home_team')} vs "
                    f"{row.get('away_team')} | "
                    f"{reason}"
                )

        print()
        print(
            "Matching Sofascore: primeiro tenta os nomes canónicos; "
            "depois usa aliases e fuzzy matching conservador."
        )
        print(
            "Só jogos terminados são aceites e o melhor candidato "
            "tem de ser claramente superior ao segundo."
        )
        print(
            "O endpoint do Sofascore é não-oficial e pode mudar. "
            "Se estiver indisponível, o workflow continua para "
            "Football-Data, ESPN e The Odds API."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
