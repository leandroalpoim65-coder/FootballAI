from __future__ import annotations

import io
import json
import os
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import libsql
import numpy as np
import pandas as pd

from ml.paper_trade import normalize_league, normalize_team


# ============================================================
# CONFIGURAÇÃO
# ============================================================

APP_TIMEZONE = ZoneInfo("Europe/Lisbon")

FOOTBALL_DATA_URL = (
    "https://www.football-data.co.uk/fixtures.csv"
)

ODDS_API_BASE_URL = (
    "https://api.the-odds-api.com/v4"
)

TABLE_NAME = "upcoming_matches"


# ============================================================
# LIGAS
# ============================================================

DEFAULT_ODDS_API_SPORT_KEYS = [
    "soccer_belgium_first_div",
    "soccer_epl",
    "soccer_efl_champ",
    "soccer_france_ligue_one",
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_italy_serie_a",
    "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga",
    "soccer_spain_la_liga",
    "soccer_spain_segunda_division",
    "soccer_turkey_super_league",
]

ODDS_API_LEAGUE_MAP = {
    "soccer_belgium_first_div":
        "Belgium First Division A",
    "soccer_epl":
        "England",
    "soccer_efl_champ":
        "England Championship",
    "soccer_france_ligue_one":
        "France",
    "soccer_germany_bundesliga":
        "Germany",
    "soccer_germany_bundesliga2":
        "Germany 2 Bundesliga",
    "soccer_italy_serie_a":
        "Italy",
    "soccer_netherlands_eredivisie":
        "Netherlands",
    "soccer_portugal_primeira_liga":
        "Portugal",
    "soccer_spain_la_liga":
        "Spain",
    "soccer_spain_segunda_division":
        "Spain Segunda Division",
    "soccer_turkey_super_league":
        "Turkey Super Lig",
}

DIVISION_LEAGUE_MAP = {
    "E0": "England",
    "SP1": "Spain",
    "I1": "Italy",
    "D1": "Germany",
    "F1": "France",
    "N1": "Netherlands",
    "P1": "Portugal",
}

QUOTA_ERROR_MARKERS = (
    "OUT_OF_USAGE_CREDITS",
    "Usage quota has been reached",
)


# ============================================================
# BASE DE DADOS TURSO
# ============================================================

def connect_turso():
    url = os.getenv(
        "TURSO_DATABASE_URL",
        "",
    ).strip()

    token = os.getenv(
        "TURSO_AUTH_TOKEN",
        "",
    ).strip()

    if not url:
        raise RuntimeError(
            "TURSO_DATABASE_URL não configurada."
        )

    if not token:
        raise RuntimeError(
            "TURSO_AUTH_TOKEN não configurado."
        )

    return libsql.connect(
        database=url,
        auth_token=token,
    )


def ensure_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            fixture_key TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            odd_over25_open REAL NOT NULL,
            odd_under25_open REAL NOT NULL,
            odd_over25_current REAL NOT NULL,
            odd_under25_current REAL NOT NULL,
            first_observed_at TEXT NOT NULL,
            current_observed_at TEXT NOT NULL,
            source TEXT,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{TABLE_NAME}_date_active
        ON {TABLE_NAME}(date, active)
        """
    )

    connection.commit()


# ============================================================
# HTTP
# ============================================================

def redact_url_secrets(url: str) -> str:
    token = "apiKey="

    if token not in url:
        return url

    prefix, remainder = url.split(
        token,
        1,
    )

    if "&" in remainder:
        _, suffix = remainder.split(
            "&",
            1,
        )

        return (
            f"{prefix}{token}***&{suffix}"
        )

    return f"{prefix}{token}***"


def http_get_bytes(
    url: str,
    timeout: int = 30,
) -> tuple[bytes, dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 FootballAI/Cloud"
        },
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
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
            f"HTTP {error.code} em "
            f"{redact_url_secrets(url)}: "
            f"{detail[:400]}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Erro de ligação a "
            f"{redact_url_secrets(url)}: "
            f"{error.reason}"
        ) from error

    if not content:
        raise RuntimeError(
            "Resposta vazia de "
            f"{redact_url_secrets(url)}"
        )

    return content, headers


def http_get_json(
    url: str,
) -> tuple[object, dict[str, str]]:
    content, headers = http_get_bytes(
        url
    )

    try:
        payload = json.loads(
            content.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            "Resposta JSON inválida."
        ) from error

    return payload, headers


def is_quota_error(error: Exception) -> bool:
    text = str(error)

    return any(
        marker in text
        for marker in QUOTA_ERROR_MARKERS
    )


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def clean_text(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if (
        not text
        or text.casefold()
        in {
            "nan",
            "none",
            "<na>",
        }
    ):
        return None

    return text


def canonical_league(value: object) -> str | None:
    normalized = normalize_league(
        value
    )

    if normalized is None:
        return None

    text = str(normalized).strip()

    return text if text else None


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

    text = str(normalized).strip()

    return text if text else None


def fixture_key(
    date_value: str,
    league: str,
    home_team: str,
    away_team: str,
) -> str:
    return "|".join(
        [
            date_value.strip().casefold(),
            league.strip().casefold(),
            home_team.strip().casefold(),
            away_team.strip().casefold(),
        ]
    )


def normalize_observation(
    *,
    date_value: object,
    time_value: object,
    league_value: object,
    home_value: object,
    away_value: object,
    over_value: object,
    under_value: object,
    source: str,
) -> dict[str, object] | None:
    date_text = clean_text(
        date_value
    )

    league = canonical_league(
        league_value
    )

    if (
        date_text is None
        or league is None
    ):
        return None

    home = canonical_team(
        league,
        home_value,
    )

    away = canonical_team(
        league,
        away_value,
    )

    if (
        home is None
        or away is None
    ):
        return None

    try:
        over = float(over_value)
        under = float(under_value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        not np.isfinite(over)
        or not np.isfinite(under)
        or over <= 1.0
        or under <= 1.0
    ):
        return None

    time_text = clean_text(
        time_value
    )

    key = fixture_key(
        date_text,
        league,
        home,
        away,
    )

    return {
        "fixture_key": key,
        "date": date_text,
        "time": time_text,
        "league": league,
        "home_team": home,
        "away_team": away,
        "odd_over25_current": over,
        "odd_under25_current": under,
        "source": source,
    }


# ============================================================
# ODDS OVER / UNDER 2.5
# ============================================================

def extract_average_totals_25(
    event: dict,
) -> tuple[float | None, float | None]:
    over_prices: list[float] = []
    under_prices: list[float] = []

    bookmakers = event.get(
        "bookmakers",
        [],
    )

    if not isinstance(
        bookmakers,
        list,
    ):
        return None, None

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

            if market.get("key") != "totals":
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
                        outcome.get("point")
                    )
                    price = float(
                        outcome.get("price")
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    abs(point - 2.5) > 1e-9
                    or price <= 1.0
                ):
                    continue

                if name == "over":
                    over_prices.append(price)
                elif name == "under":
                    under_prices.append(price)

    if (
        not over_prices
        or not under_prices
    ):
        return None, None

    return (
        float(np.mean(over_prices)),
        float(np.mean(under_prices)),
    )


# ============================================================
# THE ODDS API
# ============================================================

def requested_sport_keys() -> list[str]:
    forced = clean_text(
        os.getenv(
            "ODDS_API_SPORT_KEYS"
        )
    )

    if forced:
        return list(
            dict.fromkeys(
                item.strip()
                for item in forced.split(",")
                if item.strip()
            )
        )

    return DEFAULT_ODDS_API_SPORT_KEYS.copy()


def fetch_odds_api(
    today: pd.Timestamp,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    api_key = os.getenv(
        "ODDS_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY não configurada."
        )

    observations: list[
        dict[str, object]
    ] = []

    errors: list[str] = []
    successful_sports = 0
    requests_used = 0
    requests_remaining: str | None = None
    quota_exhausted = False

    for sport_key in requested_sport_keys():
        query = urlencode(
            {
                "apiKey": api_key,
                "regions": "eu",
                "markets": "totals",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            }
        )

        url = (
            f"{ODDS_API_BASE_URL}/sports/"
            f"{sport_key}/odds/?{query}"
        )

        try:
            payload, headers = http_get_json(
                url
            )
        except Exception as error:
            errors.append(
                f"{sport_key}: {error}"
            )

            if is_quota_error(error):
                quota_exhausted = True
                break

            continue

        requests_used += 1

        if not isinstance(
            payload,
            list,
        ):
            errors.append(
                f"{sport_key}: resposta inesperada"
            )
            continue

        successful_sports += 1

        requests_remaining = headers.get(
            "x-requests-remaining",
            requests_remaining,
        )

        league = ODDS_API_LEAGUE_MAP.get(
            sport_key,
            sport_key,
        )

        for event in payload:
            if not isinstance(
                event,
                dict,
            ):
                continue

            raw_kickoff = clean_text(
                event.get("commence_time")
            )

            if raw_kickoff is None:
                continue

            try:
                kickoff = datetime.fromisoformat(
                    raw_kickoff.replace(
                        "Z",
                        "+00:00",
                    )
                )
            except ValueError:
                continue

            local_kickoff = kickoff.astimezone(
                APP_TIMEZONE
            )

            event_date = pd.Timestamp(
                local_kickoff.date()
            )

            if event_date < today:
                continue

            over, under = (
                extract_average_totals_25(
                    event
                )
            )

            if (
                over is None
                or under is None
            ):
                continue

            normalized = normalize_observation(
                date_value=local_kickoff.strftime(
                    "%Y-%m-%d"
                ),
                time_value=local_kickoff.strftime(
                    "%H:%M"
                ),
                league_value=league,
                home_value=event.get(
                    "home_team"
                ),
                away_value=event.get(
                    "away_team"
                ),
                over_value=over,
                under_value=under,
                source="The Odds API",
            )

            if normalized is not None:
                observations.append(
                    normalized
                )

    return (
        observations,
        {
            "successful_sports":
                successful_sports,
            "requests_used":
                requests_used,
            "requests_remaining":
                requests_remaining,
            "quota_exhausted":
                quota_exhausted,
            "errors":
                errors,
        },
    )


# ============================================================
# FOOTBALL-DATA
# ============================================================

def parse_dates(series: pd.Series) -> pd.Series:
    text = (
        series
        .astype("string")
        .str.strip()
    )

    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    for date_format in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
    ):
        mask = (
            parsed.isna()
            & text.notna()
            & (text != "")
        )

        if not mask.any():
            break

        parsed.loc[mask] = pd.to_datetime(
            text.loc[mask],
            format=date_format,
            errors="coerce",
        )

    return parsed


def first_numeric_column(
    dataframe: pd.DataFrame,
    names: list[str],
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=dataframe.index,
        dtype=float,
    )

    for name in names:
        if name not in dataframe.columns:
            continue

        candidate = pd.to_numeric(
            dataframe[name],
            errors="coerce",
        )

        candidate = candidate.where(
            candidate > 1.0
        )

        result = result.fillna(
            candidate
        )

    return result


def read_csv_content(
    content: bytes,
) -> pd.DataFrame:
    last_error: Exception | None = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):
        try:
            return pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Não foi possível interpretar "
        "o CSV Football-Data."
    )


def fetch_football_data(
    today: pd.Timestamp,
) -> list[dict[str, object]]:
    content, _ = http_get_bytes(
        FOOTBALL_DATA_URL
    )

    raw = read_csv_content(
        content
    )

    required = {
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
    }

    missing = required - set(
        raw.columns
    )

    if missing:
        raise RuntimeError(
            "Football-Data sem colunas: "
            + ", ".join(sorted(missing))
        )

    raw = raw.copy()

    raw["_date"] = parse_dates(
        raw["Date"]
    )

    filtered = raw[
        raw["Div"].notna()
        & raw["_date"].notna()
        & (raw["_date"] >= today)
    ].copy()

    divisions = (
        filtered["Div"]
        .astype("string")
        .str.strip()
    )

    filtered["league"] = (
        divisions
        .map(DIVISION_LEAGUE_MAP)
        .fillna(divisions)
    )

    filtered["odd_over"] = (
        first_numeric_column(
            filtered,
            [
                "Avg>2.5",
                "B365>2.5",
                "P>2.5",
                "Max>2.5",
            ],
        )
    )

    filtered["odd_under"] = (
        first_numeric_column(
            filtered,
            [
                "Avg<2.5",
                "B365<2.5",
                "P<2.5",
                "Max<2.5",
            ],
        )
    )

    observations: list[
        dict[str, object]
    ] = []

    for _, row in filtered.iterrows():
        parsed_date = row.get("_date")

        if pd.isna(parsed_date):
            continue

        time_value = (
            row.get("Time")
            if "Time" in filtered.columns
            else None
        )

        normalized = normalize_observation(
            date_value=parsed_date.strftime(
                "%Y-%m-%d"
            ),
            time_value=time_value,
            league_value=row.get("league"),
            home_value=row.get("HomeTeam"),
            away_value=row.get("AwayTeam"),
            over_value=row.get("odd_over"),
            under_value=row.get("odd_under"),
            source="Football-Data",
        )

        if normalized is not None:
            observations.append(
                normalized
            )

    return observations


# ============================================================
# JUNTAR FONTES
# ============================================================

def combine_observations(
    odds_api: list[dict[str, object]],
    football_data: list[dict[str, object]],
) -> list[dict[str, object]]:
    combined: dict[
        str,
        dict[str, object],
    ] = {}

    # Odds API tem prioridade.
    for item in odds_api:
        key = str(item["fixture_key"])
        combined.setdefault(
            key,
            item,
        )

    for item in football_data:
        key = str(item["fixture_key"])
        combined.setdefault(
            key,
            item,
        )

    return list(
        combined.values()
    )


# ============================================================
# PERSISTÊNCIA
# ============================================================

def persist_observations(
    connection,
    observations: list[dict[str, object]],
    observed_at: str,
) -> tuple[int, int, int]:
    created = 0
    updated = 0
    preserved_open = 0

    for item in observations:
        key = str(
            item["fixture_key"]
        )

        existing = connection.execute(
            f"""
            SELECT
                odd_over25_open,
                odd_under25_open
            FROM {TABLE_NAME}
            WHERE fixture_key = ?
            """,
            (key,),
        ).fetchone()

        if existing is None:
            connection.execute(
                f"""
                INSERT INTO {TABLE_NAME} (
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
                    current_observed_at,
                    source,
                    active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    key,
                    item["date"],
                    item["time"],
                    item["league"],
                    item["home_team"],
                    item["away_team"],
                    item["odd_over25_current"],
                    item["odd_under25_current"],
                    item["odd_over25_current"],
                    item["odd_under25_current"],
                    observed_at,
                    observed_at,
                    item["source"],
                ),
            )

            created += 1
            continue

        connection.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET
                time = ?,
                odd_over25_current = ?,
                odd_under25_current = ?,
                current_observed_at = ?,
                source = ?,
                active = 1
            WHERE fixture_key = ?
            """,
            (
                item["time"],
                item["odd_over25_current"],
                item["odd_under25_current"],
                observed_at,
                item["source"],
                key,
            ),
        )

        updated += 1
        preserved_open += 1

    connection.commit()

    return (
        created,
        updated,
        preserved_open,
    )


def deactivate_past_rows(
    connection,
    today_iso: str,
) -> int:
    cursor = connection.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET active = 0
        WHERE date < ?
          AND active != 0
        """,
        (today_iso,),
    )

    connection.commit()

    rowcount = getattr(
        cursor,
        "rowcount",
        0,
    )

    return (
        int(rowcount)
        if isinstance(rowcount, int)
        and rowcount >= 0
        else 0
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — UPCOMING CLOUD")
    print("==========================================")
    print()

    today = pd.Timestamp(
        datetime.now(APP_TIMEZONE).date()
    )

    observed_at = (
        datetime.now(APP_TIMEZONE)
        .isoformat(timespec="seconds")
    )

    print(
        f"Data local                : "
        f"{today.date()}"
    )
    print(
        "Base de dados             : Turso Cloud"
    )

    try:
        connection = connect_turso()
    except Exception as error:
        print(
            f"❌ Não foi possível ligar à Turso: "
            f"{error}"
        )
        raise SystemExit(1) from error

    try:
        ensure_table(
            connection
        )

        football_data: list[
            dict[str, object]
        ] = []

        try:
            football_data = fetch_football_data(
                today
            )

            print(
                f"Football-Data prontos     : "
                f"{len(football_data)}"
            )
        except Exception as error:
            print(
                f"⚠ Football-Data falhou    : "
                f"{error}"
            )

        try:
            odds_api, info = fetch_odds_api(
                today
            )
        except Exception as error:
            print(
                f"❌ The Odds API falhou     : "
                f"{error}"
            )
            raise SystemExit(1) from error

        print(
            f"Odds API ligas OK         : "
            f"{info['successful_sports']}"
        )
        print(
            f"Pedidos Odds API usados   : "
            f"{info['requests_used']}"
        )
        print(
            f"Jogos Odds API prontos    : "
            f"{len(odds_api)}"
        )

        if info.get("requests_remaining") is not None:
            print(
                f"Créditos API restantes    : "
                f"{info['requests_remaining']}"
            )

        if info.get("quota_exhausted"):
            print(
                "⚠ Quota Odds API           : ESGOTADA"
            )

        errors = info.get(
            "errors",
            [],
        )

        if isinstance(errors, list) and errors:
            print(
                f"Ligas Odds API com falha  : "
                f"{len(errors)}"
            )

            for error in errors[:5]:
                print(
                    f"  - {error}"
                )

        # Ao contrário do antigo CSV, a tabela é incremental.
        # Uma falha parcial não apaga dados antigos de outras ligas.
        observations = combine_observations(
            odds_api,
            football_data,
        )

        if not observations:
            print()
            print(
                "⚠ Nenhuma observação válida recebida."
            )
            print(
                "A tabela cloud ficou intacta."
            )
            return

        (
            created,
            updated,
            preserved_open,
        ) = persist_observations(
            connection,
            observations,
            observed_at,
        )

        deactivated = deactivate_past_rows(
            connection,
            today.strftime("%Y-%m-%d"),
        )

        total_active = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TABLE_NAME}
            WHERE active = 1
              AND date >= ?
            """,
            (
                today.strftime(
                    "%Y-%m-%d"
                ),
            ),
        ).fetchone()[0]

        print()
        print("==========================================")
        print(" RESULTADO FINAL")
        print("==========================================")
        print()
        print(
            f"Observações combinadas    : "
            f"{len(observations)}"
        )
        print(
            f"Novos jogos / OPEN criada: "
            f"{created}"
        )
        print(
            f"Jogos CURRENT atualizados : "
            f"{updated}"
        )
        print(
            f"OPEN preservadas          : "
            f"{preserved_open}"
        )
        print(
            f"Jogos antigos desativados : "
            f"{deactivated}"
        )
        print(
            f"Jogos futuros ativos      : "
            f"{total_active}"
        )

        print()
        print("Regra das odds:")
        print(
            "OPEN fica congelada desde a primeira "
            "observação do fixture canónico."
        )
        print(
            "CURRENT é atualizada apenas quando existe "
            "uma observação fresca."
        )
        print(
            "Uma falha parcial nunca apaga OPEN/CURRENT "
            "já guardadas na Turso."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
