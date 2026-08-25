from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ml.paper_trade import normalize_league, normalize_team


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATABASE_PATH = (
    ROOT_DIR
    / "database"
    / "football.db"
)

ENV_PATH = (
    ROOT_DIR
    / ".env"
)


# ============================================================
# THE ODDS API
# ============================================================

ODDS_API_BASE_URL = (
    "https://api.the-odds-api.com/v4"
)

# Só consultamos uma liga se existir um Paper Trade cujo kickoff
# esteja dentro desta janela.
#
# A cada nova execução, a odd CLOSE é substituída apenas por uma
# observação MAIS PRÓXIMA do kickoff.
REFRESH_WINDOW_MINUTES = 120

# Nunca capturar depois do início.
MIN_MINUTES_TO_KICKOFF = 0.0

# Região e mercado iguais ao downloader principal.
ODDS_API_REGION = "eu"
ODDS_API_MARKET = "totals"


# ============================================================
# MAPA CANÓNICO -> SPORT KEY
# ============================================================

LEAGUE_SPORT_KEY = {
    "Belgium First Division A":
        "soccer_belgium_first_div",

    "England":
        "soccer_epl",

    "England Championship":
        "soccer_efl_champ",

    "England League One":
        "soccer_england_league1",

    "England League Two":
        "soccer_england_league2",

    "France":
        "soccer_france_ligue_one",

    "France Ligue 2":
        "soccer_france_ligue_two",

    "Germany":
        "soccer_germany_bundesliga",

    "Germany 2 Bundesliga":
        "soccer_germany_bundesliga2",

    "Greece Super League":
        "soccer_greece_super_league",

    "Italy":
        "soccer_italy_serie_a",

    "Italy Serie B":
        "soccer_italy_serie_b",

    "Netherlands":
        "soccer_netherlands_eredivisie",

    "Portugal":
        "soccer_portugal_primeira_liga",

    "Scotland":
        "soccer_spl",

    "Spain":
        "soccer_spain_la_liga",

    "Spain Segunda Division":
        "soccer_spain_segunda_division",

    "Turkey Super Lig":
        "soccer_turkey_super_league",
}


QUOTA_ERROR_MARKERS = (
    "OUT_OF_USAGE_CREDITS",
    "Usage quota has been reached",
)


# ============================================================
# .ENV
# ============================================================

def load_simple_env(
    path: Path,
) -> None:

    if not path.exists():
        return

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        key = key.strip()

        value = (
            value
            .strip()
            .strip('"')
            .strip("'")
        )

        if key:
            os.environ.setdefault(
                key,
                value,
            )


# ============================================================
# HTTP
# ============================================================

def http_get_json(
    url: str,
) -> tuple[
    object,
    dict[str, str],
]:

    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 FootballAI/2.0"
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
                for key, value
                in response.headers.items()
            }

    except HTTPError as error:

        try:

            detail = (
                error
                .read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

        except Exception:

            detail = ""

        raise RuntimeError(
            f"HTTP {error.code}: "
            f"{detail[:400]}"
        ) from error

    except URLError as error:

        raise RuntimeError(
            "Erro de ligação: "
            f"{error.reason}"
        ) from error

    try:

        payload = json.loads(
            content.decode(
                "utf-8"
            )
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:

        raise RuntimeError(
            "Resposta JSON inválida."
        ) from error

    return (
        payload,
        headers,
    )


def is_quota_error(
    error: Exception,
) -> bool:

    text = str(
        error
    )

    return any(
        marker in text
        for marker in QUOTA_ERROR_MARKERS
    )


# ============================================================
# BASE DE DADOS
# ============================================================

def find_paper_table(
    connection: sqlite3.Connection,
) -> str | None:

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
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:

    return {
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def add_column_if_missing(
    connection: sqlite3.Connection,
    table: str,
    columns: set[str],
    name: str,
    sql_type: str,
) -> None:

    if name in columns:
        return

    connection.execute(
        f"""
        ALTER TABLE {table}
        ADD COLUMN {name} {sql_type}
        """
    )

    columns.add(
        name
    )


# ============================================================
# DATAS / HORAS
# ============================================================

def parse_fixture_datetime(
    date_value: object,
    time_value: object,
) -> datetime | None:

    if date_value is None:
        return None

    date_text = str(
        date_value
    ).strip()

    if not date_text:
        return None

    if time_value is None:

        time_text = "00:00"

    else:

        time_text = str(
            time_value
        ).strip()

        if not time_text:
            time_text = "00:00"

    candidates = (
        f"{date_text[:10]} {time_text}",
        f"{date_text[:10]} {time_text[:5]}",
    )

    for candidate in candidates:

        try:

            return datetime.fromisoformat(
                candidate
            )

        except ValueError:

            continue

    return None


def parse_api_datetime(
    value: object,
) -> datetime | None:

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    try:

        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:

        return None

    if parsed.tzinfo is None:

        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed


# ============================================================
# NORMALIZAÇÃO DE EQUIPAS
# ============================================================

def canonical_team(
    league: str,
    value: object,
) -> str | None:

    normalized = normalize_team(
        league,
        value,
    )

    if normalized is None:
        return None

    text = str(
        normalized
    ).strip()

    return (
        text.casefold()
        if text
        else None
    )


def teams_match(
    league: str,
    left: object,
    right: object,
) -> bool:

    a = canonical_team(
        league,
        left,
    )

    b = canonical_team(
        league,
        right,
    )

    if (
        a is None
        or b is None
    ):
        return False

    return a == b


# ============================================================
# EXTRAIR OVER / UNDER 2.5
# ============================================================

def extract_average_totals_25(
    event: dict,
) -> tuple[
    float | None,
    float | None,
]:

    over_prices: list[
        float
    ] = []

    under_prices: list[
        float
    ] = []

    bookmakers = event.get(
        "bookmakers",
        [],
    )

    if not isinstance(
        bookmakers,
        list,
    ):

        return (
            None,
            None,
        )

    for bookmaker in bookmakers:

        if not isinstance(
            bookmaker,
            dict,
        ):
            continue

        markets = bookmaker.get(
            "markets",
            [],
        )

        if not isinstance(
            markets,
            list,
        ):
            continue

        for market in markets:

            if not isinstance(
                market,
                dict,
            ):
                continue

            if market.get(
                "key"
            ) != "totals":
                continue

            outcomes = market.get(
                "outcomes",
                [],
            )

            if not isinstance(
                outcomes,
                list,
            ):
                continue

            for outcome in outcomes:

                if not isinstance(
                    outcome,
                    dict,
                ):
                    continue

                name = str(
                    outcome.get(
                        "name",
                        "",
                    )
                ).strip().casefold()

                try:

                    point = float(
                        outcome.get(
                            "point"
                        )
                    )

                    price = float(
                        outcome.get(
                            "price"
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

                if (
                    abs(
                        point - 2.5
                    )
                    > 1e-9
                    or price <= 1.0
                ):

                    continue

                if name == "over":

                    over_prices.append(
                        price
                    )

                elif name == "under":

                    under_prices.append(
                        price
                    )

    if (
        not over_prices
        or not under_prices
    ):

        return (
            None,
            None,
        )

    over_average = (
        sum(
            over_prices
        )
        / len(
            over_prices
        )
    )

    under_average = (
        sum(
            under_prices
        )
        / len(
            under_prices
        )
    )

    return (
        float(
            over_average
        ),
        float(
            under_average
        ),
    )


# ============================================================
# MATCH DE EVENTO
# ============================================================

def find_unique_api_event(
    events: list[dict],
    canonical_league: str,
    fixture_datetime: datetime,
    home_team: object,
    away_team: object,
) -> dict | None:

    matches: list[
        dict
    ] = []

    for event in events:

        if not isinstance(
            event,
            dict,
        ):
            continue

        kickoff_utc = parse_api_datetime(
            event.get(
                "commence_time"
            )
        )

        if kickoff_utc is None:
            continue

        api_local = (
            kickoff_utc
            .astimezone()
            .replace(
                tzinfo=None
            )
        )

        # O nome dos clubes e a data são os identificadores
        # principais. A hora serve também para evitar colisões.
        if (
            api_local.date()
            != fixture_datetime.date()
        ):
            continue

        if abs(
            (
                api_local
                - fixture_datetime
            ).total_seconds()
        ) > 3 * 60 * 60:

            continue

        if not teams_match(
            canonical_league,
            event.get(
                "home_team"
            ),
            home_team,
        ):
            continue

        if not teams_match(
            canonical_league,
            event.get(
                "away_team"
            ),
            away_team,
        ):
            continue

        matches.append(
            event
        )

    if len(
        matches
    ) == 1:

        return matches[
            0
        ]

    return None


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=========================================="
    )

    print(
        " FOOTBALLAI — REFRESH CLOSING ODDS"
    )

    print(
        "=========================================="
    )

    print()

    load_simple_env(
        ENV_PATH
    )

    api_key = os.getenv(
        "ODDS_API_KEY",
        "",
    ).strip()

    if not DATABASE_PATH.exists():

        print(
            "❌ Base de dados não encontrada:"
        )

        print(
            DATABASE_PATH
        )

        return

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    table = find_paper_table(
        connection
    )

    if table is None:

        connection.close()

        print(
            "❌ Tabela de Paper Trade não encontrada."
        )

        return

    columns = table_columns(
        connection,
        table,
    )

    # ========================================================
    # GARANTIR COLUNAS DE AUDITORIA
    # ========================================================

    for name, sql_type in (
        (
            "odd_over25_close",
            "REAL",
        ),
        (
            "odd_under25_close",
            "REAL",
        ),
        (
            "closing_observed_at",
            "TEXT",
        ),
        (
            "closing_minutes_before_kickoff",
            "REAL",
        ),
        (
            "closing_source",
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

    fixture_date_column = (
        "fixture_date"
        if "fixture_date" in columns
        else
        "date"
        if "date" in columns
        else
        None
    )

    fixture_time_column = (
        "fixture_time"
        if "fixture_time" in columns
        else
        "time"
        if "time" in columns
        else
        None
    )

    if fixture_date_column is None:

        connection.close()

        print(
            "❌ A tabela não tem fixture_date/date."
        )

        return

    if fixture_time_column is None:

        connection.close()

        print(
            "❌ A tabela não tem fixture_time/time."
        )

        return

    required = {
        "id",
        "league",
        "home_team",
        "away_team",
        fixture_date_column,
        fixture_time_column,
    }

    missing = (
        required
        - columns
    )

    if missing:

        connection.close()

        print(
            "❌ Colunas em falta:"
        )

        for column in sorted(
            missing
        ):

            print(
                f"   - {column}"
            )

        return

    now = datetime.now()

    # ========================================================
    # PROCURAR SÓ JOGOS NA JANELA
    # ========================================================

    rows = connection.execute(
        f"""
        SELECT *
        FROM {table}
        WHERE COALESCE(
            status,
            'OPEN'
        ) = 'OPEN'
        ORDER BY
            {fixture_date_column},
            {fixture_time_column},
            id
        """
    ).fetchall()

    candidates: list[
        dict[str, object]
    ] = []

    skipped_past = 0
    skipped_far = 0
    skipped_invalid_datetime = 0
    skipped_unsupported_league = 0

    for row in rows:

        fixture_datetime = parse_fixture_datetime(
            row[
                fixture_date_column
            ],
            row[
                fixture_time_column
            ],
        )

        if fixture_datetime is None:

            skipped_invalid_datetime += 1
            continue

        minutes_to_start = (
            fixture_datetime
            - now
        ).total_seconds() / 60.0

        if (
            minutes_to_start
            < MIN_MINUTES_TO_KICKOFF
        ):

            skipped_past += 1
            continue

        if (
            minutes_to_start
            > REFRESH_WINDOW_MINUTES
        ):

            skipped_far += 1
            continue

        canonical_league = normalize_league(
            row[
                "league"
            ]
        )

        if canonical_league is None:

            skipped_unsupported_league += 1
            continue

        sport_key = LEAGUE_SPORT_KEY.get(
            canonical_league
        )

        if not sport_key:

            skipped_unsupported_league += 1
            continue

        candidates.append(
            {
                "row":
                    row,

                "fixture_datetime":
                    fixture_datetime,

                "minutes_to_start":
                    float(
                        minutes_to_start
                    ),

                "canonical_league":
                    canonical_league,

                "sport_key":
                    sport_key,
            }
        )

    print(
        f"Paper Trades OPEN         : "
        f"{len(rows)}"
    )

    print(
        f"Na janela 0-{REFRESH_WINDOW_MINUTES} min   : "
        f"{len(candidates)}"
    )

    print(
        f"Kickoff já passou        : "
        f"{skipped_past}"
    )

    print(
        f"Ainda longe do kickoff   : "
        f"{skipped_far}"
    )

    if skipped_invalid_datetime:

        print(
            f"Data/hora inválida        : "
            f"{skipped_invalid_datetime}"
        )

    if skipped_unsupported_league:

        print(
            f"Liga não suportada        : "
            f"{skipped_unsupported_league}"
        )

    # Zero candidatos = zero pedidos à API.
    if not candidates:

        connection.close()

        print()
        print(
            "✅ Nenhum jogo precisa de refresh agora."
        )

        print(
            "Pedidos à The Odds API: 0"
        )

        return

    if not api_key:

        connection.close()

        print()
        print(
            "❌ ODDS_API_KEY não configurada."
        )

        print(
            "Nenhuma closing odd foi alterada."
        )

        return

    # ========================================================
    # MOSTRAR JOGOS CANDIDATOS
    # ========================================================

    print()
    print(
        "Jogos candidatos:"
    )

    for item in candidates:

        row = item[
            "row"
        ]

        print(
            f"  - "
            f"{row['home_team']} vs "
            f"{row['away_team']} "
            f"| {item['canonical_league']} "
            f"| {item['minutes_to_start']:.0f} min"
        )

    # ========================================================
    # UMA CHAMADA POR LIGA NECESSÁRIA
    # ========================================================

    required_sport_keys = sorted(
        {
            str(
                item[
                    "sport_key"
                ]
            )
            for item in candidates
        }
    )

    print()
    print(
        f"Ligas a consultar         : "
        f"{len(required_sport_keys)}"
    )

    events_by_sport: dict[
        str,
        list[dict],
    ] = {}

    successful_sports: set[
        str
    ] = set()

    failed_sports: list[
        str
    ] = []

    requests_used = 0

    requests_remaining: (
        str | None
    ) = None

    quota_exhausted = False

    for sport_key in required_sport_keys:

        query = urlencode(
            {
                "apiKey":
                    api_key,

                "regions":
                    ODDS_API_REGION,

                "markets":
                    ODDS_API_MARKET,

                "oddsFormat":
                    "decimal",

                "dateFormat":
                    "iso",
            }
        )

        url = (
            f"{ODDS_API_BASE_URL}"
            f"/sports/"
            f"{sport_key}"
            f"/odds/?{query}"
        )

        try:

            payload, headers = http_get_json(
                url
            )

        except Exception as error:

            failed_sports.append(
                sport_key
            )

            print(
                f"⚠ {sport_key}: {error}"
            )

            if is_quota_error(
                error
            ):

                quota_exhausted = True

                print(
                    "⚠ Quota esgotada; "
                    "não serão feitos mais pedidos."
                )

                break

            continue

        requests_used += 1

        if not isinstance(
            payload,
            list,
        ):

            failed_sports.append(
                sport_key
            )

            print(
                f"⚠ {sport_key}: resposta inesperada."
            )

            continue

        events_by_sport[
            sport_key
        ] = [
            event
            for event in payload
            if isinstance(
                event,
                dict,
            )
        ]

        successful_sports.add(
            sport_key
        )

        requests_remaining = headers.get(
            "x-requests-remaining",
            requests_remaining,
        )

        print(
            f"✅ {sport_key}: "
            f"{len(events_by_sport[sport_key])} eventos"
        )

    # ========================================================
    # GRAVAR APENAS OBSERVAÇÕES FRESCAS E INEQUÍVOCAS
    # ========================================================

    updated = 0
    skipped_no_fresh_source = 0
    skipped_not_found = 0
    skipped_invalid_odds = 0
    skipped_not_closer = 0

    observed_at = datetime.now().isoformat(
        timespec="seconds"
    )

    for item in candidates:

        row = item[
            "row"
        ]

        sport_key = str(
            item[
                "sport_key"
            ]
        )

        if sport_key not in successful_sports:

            skipped_no_fresh_source += 1
            continue

        canonical_league = str(
            item[
                "canonical_league"
            ]
        )

        fixture_datetime = item[
            "fixture_datetime"
        ]

        if not isinstance(
            fixture_datetime,
            datetime,
        ):

            skipped_not_found += 1
            continue

        event = find_unique_api_event(
            events_by_sport.get(
                sport_key,
                [],
            ),
            canonical_league,
            fixture_datetime,
            row[
                "home_team"
            ],
            row[
                "away_team"
            ],
        )

        if event is None:

            skipped_not_found += 1
            continue

        (
            over_close,
            under_close,
        ) = extract_average_totals_25(
            event
        )

        if (
            over_close is None
            or under_close is None
        ):

            skipped_invalid_odds += 1
            continue

        new_minutes = float(
            item[
                "minutes_to_start"
            ]
        )

        old_minutes = row[
            "closing_minutes_before_kickoff"
        ]

        if old_minutes is not None:

            try:

                old_minutes_float = float(
                    old_minutes
                )

            except (
                TypeError,
                ValueError,
            ):

                old_minutes_float = None

            # Quanto menor, mais perto do kickoff.
            if (
                old_minutes_float is not None
                and new_minutes >= old_minutes_float
            ):

                skipped_not_closer += 1
                continue

        connection.execute(
            f"""
            UPDATE {table}
            SET
                odd_over25_close = ?,
                odd_under25_close = ?,
                closing_observed_at = ?,
                closing_minutes_before_kickoff = ?,
                closing_source = ?
            WHERE id = ?
            """,
            (
                over_close,
                under_close,
                observed_at,
                new_minutes,
                "The Odds API / EU average totals 2.5",
                int(
                    row[
                        "id"
                    ]
                ),
            ),
        )

        updated += 1

        print(
            f"✅ CLOSE "
            f"{row['home_team']} vs "
            f"{row['away_team']} "
            f"| {new_minutes:.0f} min "
            f"| Over {over_close:.3f} "
            f"| Under {under_close:.3f}"
        )

    connection.commit()
    connection.close()

    # ========================================================
    # RESUMO
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        " RESUMO"
    )

    print(
        "=========================================="
    )

    print(
        f"Pedidos API usados        : "
        f"{requests_used}"
    )

    print(
        f"Ligas com resposta fresca : "
        f"{len(successful_sports)}"
    )

    print(
        f"Closing odds atualizadas  : "
        f"{updated}"
    )

    print(
        f"Sem fonte fresca          : "
        f"{skipped_no_fresh_source}"
    )

    print(
        f"Evento não encontrado     : "
        f"{skipped_not_found}"
    )

    print(
        f"Odd 2.5 inválida          : "
        f"{skipped_invalid_odds}"
    )

    print(
        f"Observação não mais próxima: "
        f"{skipped_not_closer}"
    )

    if requests_remaining is not None:

        print(
            f"Créditos API restantes    : "
            f"{requests_remaining}"
        )

    if quota_exhausted:

        print(
            "Quota Odds API            : ESGOTADA"
        )

    if failed_sports:

        print(
            f"Ligas com falha           : "
            f"{len(failed_sports)}"
        )

    print()
    print(
        "Regra de segurança:"
    )

    print(
        "Só uma resposta fresca da API pode "
        "alterar uma CLOSE."
    )

    print(
        "Cada nova execução só substitui a CLOSE "
        "se estiver mais perto do kickoff."
    )


if __name__ == "__main__":
    main()