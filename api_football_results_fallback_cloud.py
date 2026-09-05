from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from update_results_cloud import (
    add_column_if_missing,
    connect_turso,
    fetch_dicts,
    find_paper_table,
    parse_db_date,
    safe_int,
    table_columns,
    update_trade,
)


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

STALE_AFTER_MINUTES = 180
LOOKBACK_DAYS = 30
RECENT_RECHECK_HOURS = 6
OLD_RECHECK_HOURS = 24
RECENT_DAYS = 2

MIN_TEAM_SIMILARITY = 0.72
MIN_COMBINED_SIMILARITY = 1.58
MIN_MARGIN = 0.08
MAX_KICKOFF_DIFFERENCE_HOURS = 8

FINISHED_STATUSES = {"FT", "AET", "PEN"}
POSTPONED_STATUSES = {"PST", "CANC", "ABD", "AWD", "WO"}

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

ALIASES = {
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
    "nec nijmegen": "nijmegen",
    "nijmegen": "nijmegen",
    "sc braga": "sp braga",
    "sporting braga": "sp braga",
    "sp braga": "sp braga",
}


def http_get_json(
    path: str,
    params: dict[str, object],
    api_key: str,
) -> tuple[dict, dict[str, str]]:
    url = (
        f"{API_FOOTBALL_BASE_URL}{path}?"
        + urlencode(params)
    )
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 FootballAI/Cloud",
            "Accept": "application/json",
            "x-apisports-key": api_key,
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
            headers = {
                key.lower(): value
                for key, value in response.headers.items()
            }
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

    try:
        payload = json.loads(
            raw.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            "Resposta JSON inválida."
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            "API-Football devolveu resposta inesperada."
        )

    errors = payload.get("errors")
    if errors:
        raise RuntimeError(
            f"API-Football: {errors}"
        )

    return payload, headers


def normalize_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\butd\b", " united ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    tokens = [
        token
        for token in text.split()
        if token not in STOPWORDS
    ]
    normalized = re.sub(
        r"\s+",
        " ",
        " ".join(tokens),
    ).strip()
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
        if longer and shorter / longer >= 0.55:
            return max(
                0.86,
                SequenceMatcher(None, a, b).ratio(),
            )

    return SequenceMatcher(None, a, b).ratio()


def parse_api_datetime(value: object) -> datetime | None:
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
    )


def parse_db_datetime(
    date_value: object,
    time_value: object,
) -> datetime | None:
    fixture_date = parse_db_date(date_value)
    if fixture_date is None:
        return None

    time_text = str(
        time_value if time_value is not None else "00:00"
    ).strip()
    if not time_text:
        time_text = "00:00"

    try:
        parsed = datetime.fromisoformat(
            f"{fixture_date.isoformat()} {time_text[:8]}"
        )
    except ValueError:
        try:
            parsed = datetime.fromisoformat(
                f"{fixture_date.isoformat()} {time_text[:5]}"
            )
        except ValueError:
            return None

    return parsed.replace(
        tzinfo=APP_TIMEZONE
    )


def parse_checked_at(value: object) -> datetime | None:
    return parse_api_datetime(value)


def should_recheck(
    row: dict,
    now: datetime,
    fixture_datetime: datetime,
) -> bool:
    checked = parse_checked_at(
        row.get("results_api_football_checked_at")
    )
    if checked is None:
        return True

    age_days = (
        now.date() - fixture_datetime.date()
    ).days
    interval_hours = (
        RECENT_RECHECK_HOURS
        if age_days <= RECENT_DAYS
        else OLD_RECHECK_HOURS
    )

    return (
        now - checked
    ).total_seconds() >= interval_hours * 3600


def fixture_details(
    item: dict,
) -> tuple[
    int | None,
    datetime | None,
    str | None,
    str | None,
    str,
    int | None,
    int | None,
]:
    fixture = item.get("fixture")
    teams = item.get("teams")
    goals = item.get("goals")

    if not isinstance(fixture, dict):
        fixture = {}
    if not isinstance(teams, dict):
        teams = {}
    if not isinstance(goals, dict):
        goals = {}

    fixture_id = safe_int(
        fixture.get("id")
    )
    kickoff = parse_api_datetime(
        fixture.get("date")
    )

    status_obj = fixture.get("status")
    if not isinstance(status_obj, dict):
        status_obj = {}
    status = str(
        status_obj.get("short") or ""
    ).strip().upper()

    home_obj = teams.get("home")
    away_obj = teams.get("away")

    home_name = (
        str(home_obj.get("name")).strip()
        if isinstance(home_obj, dict)
        and home_obj.get("name")
        else None
    )
    away_name = (
        str(away_obj.get("name")).strip()
        if isinstance(away_obj, dict)
        and away_obj.get("name")
        else None
    )

    home_goals = safe_int(
        goals.get("home")
    )
    away_goals = safe_int(
        goals.get("away")
    )

    return (
        fixture_id,
        kickoff,
        home_name,
        away_name,
        status,
        home_goals,
        away_goals,
    )


def fetch_fixtures(
    fixture_date: date,
    api_key: str,
) -> tuple[list[dict], str | None]:
    try:
        payload, headers = http_get_json(
            "/fixtures",
            {
                "date": fixture_date.isoformat(),
                "timezone": "Europe/Lisbon",
            },
            api_key,
        )
    except Exception as error:
        return [], str(error)

    response = payload.get("response", [])
    if not isinstance(response, list):
        return [], "campo response inválido"

    remaining = headers.get(
        "x-ratelimit-requests-remaining",
        headers.get("x-ratelimit-remaining"),
    )

    if remaining is not None:
        print(
            f"    API-Football restante: {remaining}"
        )

    return [
        item
        for item in response
        if isinstance(item, dict)
    ], None


def find_fixture(
    fixtures: list[dict],
    fixture_datetime: datetime,
    home_team: str,
    away_team: str,
) -> tuple[dict | None, str]:
    ranked: list[
        tuple[float, float, float, float, dict]
    ] = []

    for item in fixtures:
        (
            _,
            kickoff,
            api_home,
            api_away,
            _,
            _,
            _,
        ) = fixture_details(item)

        if kickoff is None:
            continue

        delta_hours = abs(
            (kickoff - fixture_datetime).total_seconds()
        ) / 3600.0

        if delta_hours > MAX_KICKOFF_DIFFERENCE_HOURS:
            continue

        home_sim = similarity(
            home_team,
            api_home,
        )
        away_sim = similarity(
            away_team,
            api_away,
        )
        combined = home_sim + away_sim

        time_bonus = max(
            0.0,
            0.12
            * (
                1.0
                - delta_hours
                / MAX_KICKOFF_DIFFERENCE_HOURS
            ),
        )

        ranked.append(
            (
                combined + time_bonus,
                combined,
                home_sim,
                away_sim,
                item,
            )
        )

    if not ranked:
        return None, "nenhum candidato no mesmo dia/janela horária"

    ranked.sort(
        key=lambda value: value[0],
        reverse=True,
    )

    (
        best_score,
        best_combined,
        best_home,
        best_away,
        best_item,
    ) = ranked[0]

    second_score = (
        ranked[1][0]
        if len(ranked) > 1
        else 0.0
    )

    (
        fixture_id,
        kickoff,
        api_home,
        api_away,
        status,
        _,
        _,
    ) = fixture_details(best_item)

    debug = (
        f"melhor={api_home} vs {api_away} "
        f"| fixture={fixture_id} "
        f"| kickoff={kickoff.strftime('%H:%M') if kickoff else '?'} "
        f"| sim={best_home:.2f}/{best_away:.2f} "
        f"| total={best_combined:.2f} "
        f"| margem={best_score-second_score:.2f} "
        f"| status={status or '?'}"
    )

    if (
        best_home < MIN_TEAM_SIMILARITY
        or best_away < MIN_TEAM_SIMILARITY
        or best_combined < MIN_COMBINED_SIMILARITY
    ):
        return None, debug + " | similaridade insuficiente"

    if (
        len(ranked) > 1
        and best_score - second_score < MIN_MARGIN
        and best_combined < 1.90
    ):
        return None, debug + " | ambíguo"

    return best_item, debug


def update_check_metadata(
    connection,
    table: str,
    row_id: int,
    checked_at: datetime,
    fixture_id: int | None,
    status: str,
) -> None:
    connection.execute(
        f"""
        UPDATE {table}
        SET
            results_api_football_checked_at = ?,
            results_api_football_fixture_id = ?,
            results_api_football_status = ?
        WHERE id = ?
        """,
        (
            checked_at.astimezone(
                timezone.utc
            ).replace(
                microsecond=0
            ).isoformat(),
            fixture_id,
            status or None,
            row_id,
        ),
    )


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — API-FOOTBALL RESULTS FALLBACK")
    print("==========================================")
    print()

    api_key = os.getenv(
        "API_FOOTBALL_KEY",
        "",
    ).strip()

    if not api_key:
        print(
            "⚪ API_FOOTBALL_KEY não configurada."
        )
        return

    now = datetime.now(
        APP_TIMEZONE
    )
    cutoff = now.date() - timedelta(
        days=LOOKBACK_DAYS
    )

    connection = connect_turso()

    try:
        table = find_paper_table(
            connection
        )
        if table is None:
            raise RuntimeError(
                "Tabela de Paper Trade não encontrada."
            )

        columns = table_columns(
            connection,
            table,
        )

        for name, sql_type in (
            (
                "results_api_football_checked_at",
                "TEXT",
            ),
            (
                "results_api_football_fixture_id",
                "INTEGER",
            ),
            (
                "results_api_football_status",
                "TEXT",
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
            raise RuntimeError(
                "Tabela sem data/hora da fixture."
            )

        rows = fetch_dicts(
            connection,
            f"""
            SELECT *
            FROM {table}
            WHERE COALESCE(status, 'OPEN') != 'RESOLVED'
              AND date({date_col}) <= date(?)
              AND date({date_col}) >= date(?)
            ORDER BY {date_col}, {time_col}, id
            """,
            (
                now.date().isoformat(),
                cutoff.isoformat(),
            ),
        )

        stale: list[
            tuple[dict, datetime]
        ] = []
        too_recent = 0
        throttled = 0
        invalid = 0

        for row in rows:
            fixture_datetime = parse_db_datetime(
                row.get(date_col),
                row.get(time_col),
            )
            if fixture_datetime is None:
                invalid += 1
                continue

            minutes_since_kickoff = (
                now - fixture_datetime
            ).total_seconds() / 60.0

            if minutes_since_kickoff < STALE_AFTER_MINUTES:
                too_recent += 1
                continue

            if not should_recheck(
                row,
                now,
                fixture_datetime,
            ):
                throttled += 1
                continue

            stale.append(
                (
                    row,
                    fixture_datetime,
                )
            )

        print(
            f"OPEN na janela            : {len(rows)}"
        )
        print(
            f"OPEN > {STALE_AFTER_MINUTES} min       : "
            f"{len(stale) + throttled}"
        )
        print(
            f"A verificar agora         : {len(stale)}"
        )
        print(
            f"Recheck throttled         : {throttled}"
        )
        print(
            f"Ainda recentes/em jogo    : {too_recent}"
        )
        if invalid:
            print(
                f"Data/hora inválida        : {invalid}"
            )

        if not stale:
            print(
                "✅ Nenhum resultado atrasado precisa de consulta agora."
            )
            return

        dates_needed = sorted(
            {
                fixture_datetime.date()
                for _, fixture_datetime
                in stale
            }
        )

        fixtures_by_date: dict[
            date,
            list[dict],
        ] = {}

        print()
        print("API-Football fixtures:")

        for fixture_date in dates_needed:
            fixtures, error = fetch_fixtures(
                fixture_date,
                api_key,
            )
            if error is not None:
                print(
                    f"  ⚠ {fixture_date}: {error}"
                )
                continue

            fixtures_by_date[
                fixture_date
            ] = fixtures

            finished = sum(
                1
                for item in fixtures
                if fixture_details(item)[4]
                in FINISHED_STATUSES
            )

            print(
                f"  ✅ {fixture_date}: "
                f"{len(fixtures)} fixtures | "
                f"{finished} terminadas"
            )

        resolved = 0
        waiting = 0
        unmatched = 0
        postponed = 0

        print()
        print("Auditoria dos OPEN atrasados:")

        for row, fixture_datetime in stale:
            row_id = int(
                row["id"]
            )
            fixture_date = (
                fixture_datetime.date()
            )

            fixture, debug = find_fixture(
                fixtures_by_date.get(
                    fixture_date,
                    [],
                ),
                fixture_datetime,
                str(
                    row.get("home_team")
                    or ""
                ),
                str(
                    row.get("away_team")
                    or ""
                ),
            )

            if fixture is None:
                update_check_metadata(
                    connection,
                    table,
                    row_id,
                    now,
                    None,
                    "UNMATCHED",
                )
                unmatched += 1
                print(
                    f"  ⚠ {fixture_date} | "
                    f"{row.get('home_team')} vs "
                    f"{row.get('away_team')} "
                    f"| {debug}"
                )
                continue

            (
                fixture_id,
                _,
                api_home,
                api_away,
                status,
                home_goals,
                away_goals,
            ) = fixture_details(
                fixture
            )

            update_check_metadata(
                connection,
                table,
                row_id,
                now,
                fixture_id,
                status,
            )

            if status in FINISHED_STATUSES:
                if (
                    home_goals is None
                    or away_goals is None
                ):
                    waiting += 1
                    print(
                        f"  ⚠ {fixture_date} | "
                        f"{row.get('home_team')} vs "
                        f"{row.get('away_team')} "
                        f"| terminado mas score inválido "
                        f"| {debug}"
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
                    f"  ✅ [API-Football] "
                    f"{fixture_date} | "
                    f"{row.get('home_team')} "
                    f"{home_goals}-{away_goals} "
                    f"{row.get('away_team')} "
                    f"| API: {api_home} vs {api_away}"
                )
                continue

            if status in POSTPONED_STATUSES:
                postponed += 1
                print(
                    f"  🟠 {fixture_date} | "
                    f"{row.get('home_team')} vs "
                    f"{row.get('away_team')} "
                    f"| status API-Football={status} "
                    f"| não resolvido"
                )
                continue

            waiting += 1
            print(
                f"  🟡 {fixture_date} | "
                f"{row.get('home_team')} vs "
                f"{row.get('away_team')} "
                f"| status={status or '?'} "
                f"| aguarda final"
            )

        connection.commit()

        print()
        print("==========================================")
        print(" RESUMO API-FOOTBALL RESULTS")
        print("==========================================")
        print(
            f"Resolvidos               : {resolved}"
        )
        print(
            f"Adiados/cancelados       : {postponed}"
        )
        print(
            f"Ainda a aguardar          : {waiting}"
        )
        print(
            f"Sem matching              : {unmatched}"
        )
        print()
        print(
            "Apenas fixtures com estado final são marcadas RESOLVED."
        )
        print(
            "Jogos adiados/cancelados ficam OPEN e identificados "
            "em results_api_football_status."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
