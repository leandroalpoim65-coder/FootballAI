from __future__ import annotations

import io
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
SOURCE_NAME = "Football-Data fixtures"

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
    import os

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


def http_get_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 FootballAI/Cloud"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {error.code}: {detail[:400]}") from error
    except URLError as error:
        raise RuntimeError(f"Erro de ligação: {error.reason}") from error

    if not content:
        raise RuntimeError("Football-Data devolveu resposta vazia.")

    return content


def read_csv_content(content: bytes) -> pd.DataFrame:
    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
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

    raise RuntimeError("Não foi possível interpretar fixtures.csv.")


def clean_text(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        return None
    return text


def canonical_league(value: object) -> str | None:
    normalized = normalize_league(value)
    if normalized is None:
        return None
    text = str(normalized).strip()
    return text if text else None


def canonical_team(league: str, value: object) -> str | None:
    normalized = normalize_team(league, value)
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


def parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        mask = parsed.isna() & text.notna() & (text != "")
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
    result = pd.Series(np.nan, index=dataframe.index, dtype=float)

    for name in names:
        if name not in dataframe.columns:
            continue

        candidate = pd.to_numeric(dataframe[name], errors="coerce")
        candidate = candidate.where(candidate > 1.0)
        result = result.fillna(candidate)

    return result


def fetch_football_data(today: pd.Timestamp) -> list[dict[str, object]]:
    raw = read_csv_content(http_get_bytes(FOOTBALL_DATA_URL))

    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(
            "Football-Data sem colunas: " + ", ".join(sorted(missing))
        )

    raw = raw.copy()
    raw["_date"] = parse_dates(raw["Date"])

    filtered = raw[
        raw["Div"].notna()
        & raw["_date"].notna()
        & (raw["_date"] >= today)
    ].copy()

    divisions = filtered["Div"].astype("string").str.strip()
    filtered["league"] = divisions.map(DIVISION_LEAGUE_MAP).fillna(divisions)

    filtered["odd_over"] = first_numeric_column(
        filtered,
        ["Avg>2.5", "B365>2.5", "P>2.5", "Max>2.5"],
    )
    filtered["odd_under"] = first_numeric_column(
        filtered,
        ["Avg<2.5", "B365<2.5", "P<2.5", "Max<2.5"],
    )

    observations: list[dict[str, object]] = []

    for _, row in filtered.iterrows():
        parsed_date = row.get("_date")
        if pd.isna(parsed_date):
            continue

        league = canonical_league(row.get("league"))
        if league is None:
            continue

        home = canonical_team(league, row.get("HomeTeam"))
        away = canonical_team(league, row.get("AwayTeam"))
        if home is None or away is None:
            continue

        try:
            over = float(row.get("odd_over"))
            under = float(row.get("odd_under"))
        except (TypeError, ValueError):
            continue

        if (
            not np.isfinite(over)
            or not np.isfinite(under)
            or over <= 1.0
            or under <= 1.0
        ):
            continue

        date_text = parsed_date.strftime("%Y-%m-%d")
        time_text = clean_text(row.get("Time")) if "Time" in filtered.columns else None

        observations.append(
            {
                "fixture_key": fixture_key(date_text, league, home, away),
                "date": date_text,
                "time": time_text,
                "league": league,
                "home_team": home,
                "away_team": away,
                "odd_over25_current": over,
                "odd_under25_current": under,
                "source": SOURCE_NAME,
            }
        )

    return observations


def persist_observations(
    connection,
    observations: list[dict[str, object]],
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


def deactivate_past_rows(connection, today_iso: str) -> int:
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

    rowcount = getattr(cursor, "rowcount", 0)
    return int(rowcount) if isinstance(rowcount, int) and rowcount >= 0 else 0


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — UPCOMING FOOTBALL-DATA")
    print("==========================================")
    print()

    today = pd.Timestamp(datetime.now(APP_TIMEZONE).date())
    observed_at = datetime.now(APP_TIMEZONE).isoformat(timespec="seconds")

    print(f"Data local                : {today.date()}")
    print("Base de dados             : Turso Cloud")
    print("Fonte das odds upcoming   : Football-Data")
    print("Pedidos The Odds API      : 0")

    try:
        connection = connect_turso()
    except Exception as error:
        print(f"❌ Não foi possível ligar à Turso: {error}")
        raise SystemExit(1) from error

    try:
        ensure_table(connection)

        try:
            observations = fetch_football_data(today)
        except Exception as error:
            print(f"❌ Football-Data falhou: {error}")
            print("A tabela upcoming existente ficou intacta.")
            raise SystemExit(1) from error

        print(f"Jogos com odds válidas    : {len(observations)}")

        if not observations:
            print("⚠ Nenhuma observação válida recebida.")
            print("A tabela upcoming existente ficou intacta.")
            return

        created, updated, preserved_open = persist_observations(
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
            (today.strftime("%Y-%m-%d"),),
        ).fetchone()[0]

        print()
        print("==========================================")
        print(" RESULTADO FINAL")
        print("==========================================")
        print(f"Novos jogos / OPEN criada: {created}")
        print(f"Jogos CURRENT atualizados : {updated}")
        print(f"OPEN preservadas          : {preserved_open}")
        print(f"Jogos antigos desativados : {deactivated}")
        print(f"Jogos futuros ativos      : {total_active}")
        print()
        print("A The Odds API fica agora reservada para CLOSE pré-jogo.")
        print("Football-Data fornece as odds de upcoming sem gastar créditos.")
        print("OPEN continua congelada desde a primeira observação canónica.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
