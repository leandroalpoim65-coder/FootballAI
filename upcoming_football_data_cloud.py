from __future__ import annotations

import io
import os
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import libsql
import numpy as np
import pandas as pd

from ml.paper_trade import normalize_league, normalize_team


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/fixtures.csv"
TABLE_NAME = "upcoming_matches"
TEMPORARY_SOURCE_EXIT_CODE = 75


class FootballDataUnavailable(RuntimeError):
    """Falha temporária da fonte externa, sem alterar dados existentes."""


DIVISION_LEAGUE_MAP = {
    "B1": "Belgium First Division A",
    "E0": "England",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "F1": "France",
    "F2": "France Ligue 2",
    "D1": "Germany",
    "D2": "Germany 2 Bundesliga",
    "G1": "Greece Super League",
    "I1": "Italy",
    "I2": "Italy Serie B",
    "N1": "Netherlands",
    "P1": "Portugal",
    "SC0": "Scotland",
    "SP1": "Spain",
    "SP2": "Spain Segunda Division",
    "T1": "Turkey Super Lig",
}


def connect_turso():
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    if not url:
        raise RuntimeError("TURSO_DATABASE_URL não configurada.")
    if not token:
        raise RuntimeError("TURSO_AUTH_TOKEN não configurado.")
    return libsql.connect(database=url, auth_token=token)


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
        CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_date_active
        ON {TABLE_NAME}(date, active)
        """
    )
    connection.commit()


def http_get_bytes(
    url: str,
    attempts: int = 5,
    base_delay_seconds: int = 8,
) -> bytes:
    """
    Faz download com retry para falhas temporárias da Football-Data.

    O site devolve ocasionalmente 5xx/429 durante manutenção ou picos.
    Nessas situações não devemos falhar logo à primeira tentativa.
    """
    retryable_http = {429, 500, 502, 503, 504}
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 FootballAI/Cloud",
                "Accept": "text/csv,text/plain,*/*",
                "Cache-Control": "no-cache",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                content = response.read()

            if not content:
                raise RuntimeError("Football-Data devolveu resposta vazia.")

            if attempt > 1:
                print(
                    f"✅ Football-Data recuperou na tentativa "
                    f"{attempt}/{attempts}."
                )
            return content

        except HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""

            last_error = (
                FootballDataUnavailable(
                    f"HTTP {error.code}: {detail[:300]}"
                )
                if error.code in retryable_http
                else RuntimeError(
                    f"HTTP {error.code}: {detail[:300]}"
                )
            )

            if error.code not in retryable_http or attempt >= attempts:
                raise last_error from error

            delay = base_delay_seconds * attempt
            print(
                f"⚠ Football-Data HTTP {error.code} "
                f"(tentativa {attempt}/{attempts}). "
                f"Nova tentativa em {delay}s..."
            )
            time.sleep(delay)

        except (URLError, TimeoutError) as error:
            last_error = FootballDataUnavailable(
                f"Erro temporário de ligação: "
                f"{getattr(error, 'reason', error)}"
            )

            if attempt >= attempts:
                raise last_error from error

            delay = base_delay_seconds * attempt
            print(
                f"⚠ Football-Data indisponível "
                f"(tentativa {attempt}/{attempts}). "
                f"Nova tentativa em {delay}s..."
            )
            time.sleep(delay)

    raise last_error or RuntimeError(
        "Não foi possível obter Football-Data."
    )


def read_csv(content: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
                low_memory=False,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError("Não foi possível interpretar fixtures.csv.")


def parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        mask = parsed.isna() & text.notna() & (text != "")
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(
            text.loc[mask],
            format=fmt,
            errors="coerce",
        )
    return parsed


def first_numeric_column(dataframe: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=dataframe.index, dtype=float)
    for name in names:
        if name not in dataframe.columns:
            continue
        candidate = pd.to_numeric(dataframe[name], errors="coerce")
        candidate = candidate.where(candidate > 1.0)
        result = result.fillna(candidate)
    return result


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        return None
    return text


def canonical_team(league: str, value: object) -> str | None:
    normalized = normalize_team(league, value)
    if normalized is None:
        return None
    text = str(normalized).strip()
    return text if text else None


def fixture_key(date_value: str, league: str, home_team: str, away_team: str) -> str:
    return "|".join(
        [
            date_value.casefold(),
            league.casefold(),
            home_team.casefold(),
            away_team.casefold(),
        ]
    )


def normalize_row(row: pd.Series) -> dict[str, object] | None:
    div = clean_text(row.get("Div"))
    if div is None:
        return None

    league = DIVISION_LEAGUE_MAP.get(div)
    if not league:
        return None

    normalized_league = normalize_league(league)
    if normalized_league is None:
        return None
    league = str(normalized_league)

    parsed_date = row.get("_date")
    if pd.isna(parsed_date):
        return None

    date_text = pd.Timestamp(parsed_date).strftime("%Y-%m-%d")
    time_text = clean_text(row.get("Time"))

    home = canonical_team(league, row.get("HomeTeam"))
    away = canonical_team(league, row.get("AwayTeam"))
    if home is None or away is None:
        return None

    try:
        over = float(row.get("_odd_over"))
        under = float(row.get("_odd_under"))
    except (TypeError, ValueError):
        return None

    if (
        not np.isfinite(over)
        or not np.isfinite(under)
        or over <= 1.0
        or under <= 1.0
    ):
        return None

    return {
        "fixture_key": fixture_key(date_text, league, home, away),
        "date": date_text,
        "time": time_text,
        "league": league,
        "home_team": home,
        "away_team": away,
        "odd_over25_current": over,
        "odd_under25_current": under,
        "source": "Football-Data fixtures",
    }


def fetch_observations(today: pd.Timestamp) -> tuple[list[dict], dict]:
    raw = read_csv(http_get_bytes(FOOTBALL_DATA_URL))

    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(
            "Football-Data sem colunas: " + ", ".join(sorted(missing))
        )

    raw = raw.copy()
    raw["_date"] = parse_dates(raw["Date"])
    raw["_odd_over"] = first_numeric_column(
        raw,
        ["Avg>2.5", "B365>2.5", "P>2.5", "Max>2.5"],
    )
    raw["_odd_under"] = first_numeric_column(
        raw,
        ["Avg<2.5", "B365<2.5", "P<2.5", "Max<2.5"],
    )

    future = raw[
        raw["_date"].notna()
        & (raw["_date"] >= today)
    ].copy()

    observations: list[dict] = []
    skipped_unknown_division: set[str] = set()
    skipped_no_odds = 0

    for _, row in future.iterrows():
        div = clean_text(row.get("Div"))
        if div and div not in DIVISION_LEAGUE_MAP:
            skipped_unknown_division.add(div)
            continue

        normalized = normalize_row(row)
        if normalized is None:
            try:
                over = float(row.get("_odd_over"))
                under = float(row.get("_odd_under"))
                if not np.isfinite(over) or not np.isfinite(under):
                    skipped_no_odds += 1
            except (TypeError, ValueError):
                skipped_no_odds += 1
            continue

        observations.append(normalized)

    return observations, {
        "rows_future": len(future),
        "unknown_divisions": sorted(skipped_unknown_division),
        "skipped_no_odds": skipped_no_odds,
    }


def persist_observations(
    connection,
    observations: list[dict],
    observed_at: str,
) -> tuple[int, int, int]:
    created = 0
    updated = 0
    preserved_open = 0

    for item in observations:
        key = str(item["fixture_key"])
        existing = connection.execute(
            f"""
            SELECT odd_over25_open, odd_under25_open
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
    return created, updated, preserved_open


def deactivate_past(connection, today_iso: str) -> None:
    connection.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET active = 0
        WHERE date < ?
          AND active != 0
        """,
        (today_iso,),
    )
    connection.commit()


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — UPCOMING CLOUD")
    print("==========================================")
    print()

    today = pd.Timestamp(datetime.now(APP_TIMEZONE).date())
    observed_at = datetime.now(APP_TIMEZONE).isoformat(timespec="seconds")

    connection = connect_turso()
    try:
        ensure_table(connection)
        try:
            observations, info = fetch_observations(today)
        except FootballDataUnavailable as error:
            print()
            print("==========================================")
            print(" FONTE TEMPORARIAMENTE INDISPONÍVEL")
            print("==========================================")
            print(f"⚠ {error}")
            print("✅ upcoming_matches ficou intacta na Turso.")
            print("⏭ Paper Trade/Telegram devem ser ignorados nesta execução.")
            raise SystemExit(TEMPORARY_SOURCE_EXIT_CODE)

        print("Fonte                     : Football-Data fixtures")
        print("The Odds API              : NÃO usada neste workflow")
        print(f"Linhas futuras no CSV     : {info['rows_future']}")
        print(f"Jogos válidos com odds    : {len(observations)}")
        print(f"Sem odds 2.5 válidas      : {info['skipped_no_odds']}")

        unknown = info["unknown_divisions"]
        if unknown:
            print(
                "Divisões não mapeadas     : " + ", ".join(unknown[:20])
            )

        if not observations:
            print()
            print(
                "⚠ Nenhuma observação válida recebida. "
                "A tabela cloud ficou intacta."
            )
            return

        created, updated, preserved_open = persist_observations(
            connection,
            observations,
            observed_at,
        )

        deactivate_past(connection, today.strftime("%Y-%m-%d"))

        total_active = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TABLE_NAME}
            WHERE active = 1
              AND date >= ?
            """,
            (today.strftime("%Y-%m-%d"),),
        ).fetchone()[0]

        print()
        print("==========================================")
        print(" RESULTADO FINAL")
        print("==========================================")
        print(f"Novos jogos / OPEN criada: {created}")
        print(f"CURRENT atualizadas       : {updated}")
        print(f"OPEN preservadas          : {preserved_open}")
        print(f"Jogos futuros ativos      : {total_active}")
        print()
        print("Regras:")
        print("- OPEN fica congelada na primeira observação.")
        print("- CURRENT acompanha o Football-Data fixtures.")
        print("- The Odds API fica reservada exclusivamente para CLOSE.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
