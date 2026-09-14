from __future__ import annotations

import traceback
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

import goal_model_multileague_backtest as backtest

CACHE = Path("backtest_data_cache")
REPORTS = Path("reports")
SOURCES = [
    "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
    "https://raw.githubusercontent.com/aftermathematic/football_brain/main/data/scraped/{season}/{league}.csv",
]
REQUIRED = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def _parse(raw: bytes, source: str) -> pd.DataFrame:
    if len(raw) < 200:
        raise ValueError(f"Resposta demasiado pequena de {source}: {len(raw)} bytes")

    df = pd.read_csv(BytesIO(raw))
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV invalido em {source}; faltam colunas: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for col in ["FTHG", "FTAG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    ).copy()
    if df.empty:
        raise ValueError(f"CSV sem jogos validos em {source}")

    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


def load_csv_robust(season: str, league: str) -> pd.DataFrame:
    cache_file = CACHE / season / f"{league}.csv"

    if cache_file.exists():
        try:
            raw = cache_file.read_bytes()
            df = _parse(raw, str(cache_file))
            print(f"CACHE {season}/{league}: {len(df)} jogos")
            return df
        except Exception as exc:
            print(f"CACHE INVALIDO {season}/{league}: {exc}")
            try:
                cache_file.unlink()
            except OSError:
                pass

    errors: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0 FootballAI-local-backtest/1.0"}

    for template in SOURCES:
        url = template.format(season=season, league=league)
        try:
            print(f"DOWNLOAD {season}/{league}: {url}")
            response = requests.get(url, headers=headers, timeout=45)
            response.raise_for_status()
            df = _parse(response.content, url)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(response.content)
            print(f"OK {season}/{league}: {len(df)} jogos")
            return df
        except Exception as exc:
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
            print(f"FALHOU {season}/{league}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Nao foi possivel obter {season}/{league}. Tentativas:\n- "
        + "\n- ".join(errors)
    )


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    error_file = REPORTS / "goal_model_backtest_error.log"
    if error_file.exists():
        error_file.unlink()

    # Mantem toda a metodologia do backtest original; troca apenas a camada de download.
    backtest.load_csv = load_csv_robust

    try:
        backtest.main()
    except Exception:
        details = traceback.format_exc()
        error_file.write_text(details, encoding="utf-8")
        print("\nERRO FATAL NO BACKTEST")
        print(details)
        print(f"Diagnostico guardado em: {error_file.resolve()}")
        raise


if __name__ == "__main__":
    main()
