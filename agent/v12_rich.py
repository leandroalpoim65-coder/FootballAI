from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from agent.v12_history import league_family


RICH_FEATURES = (
    "home_shots_for_5", "home_shots_against_5",
    "home_sot_for_5", "home_sot_against_5",
    "home_corners_for_5", "home_corners_against_5",
    "away_shots_for_5", "away_shots_against_5",
    "away_sot_for_5", "away_sot_against_5",
    "away_corners_for_5", "away_corners_against_5",
    "home_shots_for_10", "home_shots_against_10",
    "home_sot_for_10", "home_sot_against_10",
    "away_shots_for_10", "away_shots_against_10",
    "away_sot_for_10", "away_sot_against_10",
)


def _mean(values: deque[float], minimum: int = 3) -> float:
    usable = [float(value) for value in values if value is not None and np.isfinite(value)]
    if len(usable) < minimum:
        return np.nan
    return float(np.mean(usable))


def build_rich_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build strictly pre-match rolling creation/pressure features.

    Histories are keyed by exact team identity inside the domestic league family,
    so promotion/relegation keeps history but unrelated clubs are never fuzzily
    linked. All fixtures on the same calendar date are frozen before that date's
    match statistics are appended to history.
    """
    histories: dict[tuple[str, str], dict[str, deque[float]]] = defaultdict(
        lambda: {
            "shots_for": deque(maxlen=10),
            "shots_against": deque(maxlen=10),
            "sot_for": deque(maxlen=10),
            "sot_against": deque(maxlen=10),
            "corners_for": deque(maxlen=10),
            "corners_against": deque(maxlen=10),
        }
    )
    rows: list[dict[str, float | int]] = []
    ordered = matches.sort_values(["date", "id"])

    for _, games in ordered.groupby("date", sort=True):
        pending: list[tuple[tuple[str, str], dict[str, float]]] = []
        for match in games.itertuples():
            family = league_family(match.league)
            home_key = (family, str(match.home_team))
            away_key = (family, str(match.away_team))
            home = histories[home_key]
            away = histories[away_key]

            rows.append({
                "id": int(match.id),
                "home_shots_for_5": _mean(deque(list(home["shots_for"])[-5:])),
                "home_shots_against_5": _mean(deque(list(home["shots_against"])[-5:])),
                "home_sot_for_5": _mean(deque(list(home["sot_for"])[-5:])),
                "home_sot_against_5": _mean(deque(list(home["sot_against"])[-5:])),
                "home_corners_for_5": _mean(deque(list(home["corners_for"])[-5:])),
                "home_corners_against_5": _mean(deque(list(home["corners_against"])[-5:])),
                "away_shots_for_5": _mean(deque(list(away["shots_for"])[-5:])),
                "away_shots_against_5": _mean(deque(list(away["shots_against"])[-5:])),
                "away_sot_for_5": _mean(deque(list(away["sot_for"])[-5:])),
                "away_sot_against_5": _mean(deque(list(away["sot_against"])[-5:])),
                "away_corners_for_5": _mean(deque(list(away["corners_for"])[-5:])),
                "away_corners_against_5": _mean(deque(list(away["corners_against"])[-5:])),
                "home_shots_for_10": _mean(home["shots_for"]),
                "home_shots_against_10": _mean(home["shots_against"]),
                "home_sot_for_10": _mean(home["sot_for"]),
                "home_sot_against_10": _mean(home["sot_against"]),
                "away_shots_for_10": _mean(away["shots_for"]),
                "away_shots_against_10": _mean(away["shots_against"]),
                "away_sot_for_10": _mean(away["sot_for"]),
                "away_sot_against_10": _mean(away["sot_against"]),
            })

            home_values = {
                "shots_for": match.home_shots,
                "shots_against": match.away_shots,
                "sot_for": match.home_shots_target,
                "sot_against": match.away_shots_target,
                "corners_for": match.home_corners,
                "corners_against": match.away_corners,
            }
            away_values = {
                "shots_for": match.away_shots,
                "shots_against": match.home_shots,
                "sot_for": match.away_shots_target,
                "sot_against": match.home_shots_target,
                "corners_for": match.away_corners,
                "corners_against": match.home_corners,
            }
            pending.append((home_key, home_values))
            pending.append((away_key, away_values))

        for key, values in pending:
            history = histories[key]
            for name, value in values.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    history[name].append(number)

    return pd.DataFrame(rows).set_index("id")
