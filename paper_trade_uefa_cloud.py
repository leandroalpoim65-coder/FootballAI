from __future__ import annotations

import pandas as pd

import paper_trade_cloud as core
from ml.paper_trade import parse_date


_original_load_historical_matches = core.load_historical_matches


def load_historical_matches_with_uefa(connection) -> pd.DataFrame:
    domestic = _original_load_historical_matches(connection)

    if not core.table_exists(connection, "uefa_matches"):
        return domestic

    rows = core.fetch_dicts(
        connection,
        """
        SELECT
            fixture_id,
            date,
            league,
            home_team,
            away_team,
            home_goals,
            away_goals
        FROM uefa_matches
        ORDER BY date, fixture_id
        """,
    )
    if not rows:
        return domestic

    uefa = pd.DataFrame(rows)
    uefa["id"] = pd.to_numeric(uefa["fixture_id"], errors="coerce") + 1_000_000_000
    uefa["date"] = uefa["date"].apply(parse_date)
    uefa["home_goals"] = pd.to_numeric(uefa["home_goals"], errors="coerce")
    uefa["away_goals"] = pd.to_numeric(uefa["away_goals"], errors="coerce")
    uefa = uefa.dropna(
        subset=["id", "date", "league", "home_team", "away_team", "home_goals", "away_goals"]
    )
    uefa = uefa[["id", "date", "league", "home_team", "away_team", "home_goals", "away_goals"]]

    combined = pd.concat([domestic, uefa], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "league", "home_team", "away_team"],
        keep="last",
    )
    return combined.sort_values(["date", "id"]).reset_index(drop=True)


core.load_historical_matches = load_historical_matches_with_uefa


if __name__ == "__main__":
    core.main()
