from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from ml.paper_trade import normalize_league, normalize_team


class AssistantProviderError(RuntimeError):
    """Safe-to-display error from the configured AI provider."""


def match_open_trades(upcoming: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    required_upcoming = {"date", "league", "home_team", "away_team"}
    required_paper = {
        "fixture_date",
        "league",
        "home_team",
        "away_team",
        "status",
        "signal",
        "clv_probability",
        "odd_over25_open",
    }
    if upcoming.empty or paper.empty:
        return pd.DataFrame()
    if not required_upcoming.issubset(upcoming.columns):
        return pd.DataFrame()
    if not required_paper.issubset(paper.columns):
        return pd.DataFrame()

    future = upcoming.copy()
    if "minutes_to_kickoff" in future.columns:
        minutes = pd.to_numeric(future["minutes_to_kickoff"], errors="coerce")
        future = future.loc[minutes >= 0].copy()
    if future.empty:
        return pd.DataFrame()

    open_paper = paper.copy()
    open_paper["status"] = open_paper["status"].fillna("").astype(str).str.upper()
    open_paper["signal"] = pd.to_numeric(
        open_paper["signal"], errors="coerce"
    ).fillna(0).astype(int)
    open_paper = open_paper.loc[open_paper["status"] == "OPEN"].copy()
    if open_paper.empty:
        return pd.DataFrame()

    if "prediction_timestamp" in open_paper.columns:
        open_paper["prediction_timestamp"] = pd.to_datetime(
            open_paper["prediction_timestamp"], errors="coerce"
        )
        open_paper = open_paper.sort_values("prediction_timestamp")

    key_pairs = (
        ("date", "fixture_date"),
        ("league", "league"),
        ("home_team", "home_team"),
        ("away_team", "away_team"),
    )
    join_keys = []
    for upcoming_column, paper_column in key_pairs:
        key = f"__assistant_key_{upcoming_column}"
        if upcoming_column == "league":
            future["__assistant_canonical_league"] = future[
                upcoming_column
            ].map(_canonical_league)
            open_paper["__assistant_canonical_league"] = open_paper[
                paper_column
            ].map(_canonical_league)
            future[key] = future["__assistant_canonical_league"].map(_key_text)
            open_paper[key] = open_paper["__assistant_canonical_league"].map(
                _key_text
            )
        elif upcoming_column in {"home_team", "away_team"}:
            league_key = "__assistant_canonical_league"
            future[key] = [
                _key_text(normalize_team(league, team))
                for league, team in zip(future[league_key], future[upcoming_column])
            ]
            open_paper[key] = [
                _key_text(normalize_team(league, team))
                for league, team in zip(
                    open_paper[league_key],
                    open_paper[paper_column],
                )
            ]
        else:
            future[key] = future[upcoming_column].map(_key_text)
            open_paper[key] = open_paper[paper_column].map(_key_text)
        join_keys.append(key)

    open_paper["paper_odd_over25_open"] = pd.to_numeric(
        open_paper["odd_over25_open"], errors="coerce"
    )
    open_paper["clv_probability"] = pd.to_numeric(
        open_paper["clv_probability"], errors="coerce"
    )
    open_paper = open_paper.drop_duplicates(join_keys, keep="last")
    right = open_paper[
        join_keys + ["paper_odd_over25_open", "clv_probability", "signal", "status"]
    ]
    matches = future.merge(right, on=join_keys, how="inner")

    output_columns = [
        "date",
        "time",
        "league",
        "home_team",
        "away_team",
        "odd_over25_open",
        "odd_over25_current",
        "source",
        "current_observed_at",
        "minutes_to_kickoff",
        "paper_odd_over25_open",
        "clv_probability",
        "signal",
        "status",
    ]
    return matches[[column for column in output_columns if column in matches.columns]]


def _canonical_league(value: object) -> str | None:
    return normalize_league(value)


def _key_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def signal_candidates(upcoming: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    matches = match_open_trades(upcoming, paper)
    if matches.empty:
        return matches
    return matches.loc[matches["signal"] == 1].reset_index(drop=True)


def to_json_records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    available = [column for column in columns if column in frame.columns]
    records = []
    for record in frame[available].to_dict(orient="records"):
        safe_record = {}
        for key, value in record.items():
            if value is None or pd.isna(value):
                safe_record[key] = None
            elif hasattr(value, "isoformat"):
                safe_record[key] = value.isoformat()
            elif hasattr(value, "item"):
                safe_record[key] = value.item()
            else:
                safe_record[key] = value
        records.append(safe_record)
    return records


def request_chat_completion(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    base_url: str,
) -> str:
    if not api_key:
        raise AssistantProviderError("Configure an AI API key in Streamlit secrets.")
    if not base_url.startswith("https://"):
        raise AssistantProviderError("The AI provider URL must use HTTPS.")

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {"model": model, "messages": messages, "max_tokens": 800, "temperature": 0.2}
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise AssistantProviderError(
            f"AI provider returned HTTP {error.code}. Check the key, model, and API quota."
        ) from None
    except (URLError, TimeoutError):
        raise AssistantProviderError(
            "Could not reach the AI provider. Check the configured HTTPS endpoint and try again."
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise AssistantProviderError("AI provider returned an unreadable response.") from None

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise AssistantProviderError("AI provider response did not contain a chat answer.") from None
    if not isinstance(content, str) or not content.strip():
        raise AssistantProviderError("AI provider returned an empty answer.")
    return content.strip()