from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from agent.v12_calibrated import V12_SPORT_FEATURES, devig_market_probability
from agent.v12_dataset import build_dataset
from agent.v12_history import league_family
from agent.v12_modeling import apply_calibrator, candidates, ece, fit_calibrator, probabilities, probability_metrics
from agent.v12_rich import RICH_FEATURES, build_rich_features
from paper_trade_cloud import simplify_team_name


RICH_ANALYST_VERSION = "v12-rich-analyst-73f-v1"
BASE_TRAIN_END = pd.Timestamp("2025-05-31")
CALIBRATION_START = pd.Timestamp("2025-06-01")
CALIBRATION_END = pd.Timestamp("2026-05-31")
RICH_ANALYST_FEATURES = tuple(V12_SPORT_FEATURES) + tuple(RICH_FEATURES)


@dataclass(frozen=True)
class FrozenRichBundle:
    version: str
    feature_names: tuple[str, ...]
    base_model: object
    calibrator: object
    training_fingerprint: str
    train_n: int
    calibration_n: int
    calibration_metrics: dict[str, float]
    calibration_ece: float


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _source_fingerprint(matches: pd.DataFrame) -> str:
    cols = [
        "id", "date", "league", "home_team", "away_team",
        "home_goals", "away_goals",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners",
    ]
    frame = matches.loc[:, cols].copy().sort_values(["date", "id"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    payload = frame.fillna("<NA>").astype(str).to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_rich_training_frame(matches: pd.DataFrame) -> pd.DataFrame:
    frozen = matches[pd.to_datetime(matches.date) <= CALIBRATION_END].copy()
    base = build_dataset(frozen)
    rich = build_rich_features(frozen)
    data = base.set_index("id").join(rich, how="left").reset_index()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=list(RICH_ANALYST_FEATURES)).copy()
    return data.sort_values(["date", "id"]).reset_index(drop=True)


def fit_frozen_rich_analyst(matches: pd.DataFrame) -> FrozenRichBundle:
    source = matches[pd.to_datetime(matches.date) <= CALIBRATION_END].copy()
    data = build_rich_training_frame(source)
    train = data[pd.to_datetime(data.date) <= BASE_TRAIN_END].copy()
    calibration = data[
        (pd.to_datetime(data.date) >= CALIBRATION_START)
        & (pd.to_datetime(data.date) <= CALIBRATION_END)
    ].copy()
    if len(train) < 1000 or len(calibration) < 500:
        raise RuntimeError(
            f"Amostra insuficiente para V12 Rich congelado: train={len(train)} calibration={len(calibration)}"
        )

    feature_names = list(RICH_ANALYST_FEATURES)
    base = clone(candidates()["logistic"])
    base.fit(train[feature_names], train.actual_over25.astype(int))
    raw_cal = probabilities(base, calibration[feature_names])
    calibrator = fit_calibrator("sigmoid", raw_cal, calibration)
    calibrated = apply_calibrator("sigmoid", calibrator, raw_cal, calibration)

    return FrozenRichBundle(
        version=RICH_ANALYST_VERSION,
        feature_names=RICH_ANALYST_FEATURES,
        base_model=base,
        calibrator=calibrator,
        training_fingerprint=_source_fingerprint(source),
        train_n=int(len(train)),
        calibration_n=int(len(calibration)),
        calibration_metrics=probability_metrics(calibration.actual_over25, calibrated),
        calibration_ece=float(ece(calibration.actual_over25, calibrated)),
    )


def predict_rich_probability(bundle: FrozenRichBundle, feature_row: dict[str, float]) -> float:
    missing = [name for name in bundle.feature_names if name not in feature_row]
    if missing:
        raise ValueError("Features V12 Rich em falta: " + ", ".join(missing))
    X = pd.DataFrame([{name: float(feature_row[name]) for name in bundle.feature_names}])
    raw = probabilities(bundle.base_model, X)
    calibrated = apply_calibrator("sigmoid", bundle.calibrator, raw, X)
    return float(np.clip(calibrated[0], 1e-6, 1.0 - 1e-6))


def _available_teams(matches: pd.DataFrame, league: str, fixture_date: pd.Timestamp) -> list[str]:
    family = league_family(league)
    dates = pd.to_datetime(matches.date, errors="coerce")
    leagues = matches.league.map(league_family)
    before = matches[(dates < fixture_date) & (leagues == family)]
    values = pd.concat([before.home_team, before.away_team], ignore_index=True).dropna().astype(str)
    return sorted(set(values.tolist()))


def safe_resolve_team(matches: pd.DataFrame, league: str, team: str, fixture_date: pd.Timestamp) -> str:
    available = _available_teams(matches, league, fixture_date)
    if team in available:
        return team
    target = simplify_team_name(team)
    if not target:
        raise ValueError(f"Nome de equipa vazio: {team}")

    exact = [candidate for candidate in available if simplify_team_name(candidate) == target]
    if len(exact) == 1:
        return exact[0]

    target_tokens = set(target.split())
    contained: list[str] = []
    for candidate in available:
        key = simplify_team_name(candidate)
        tokens = set(key.split())
        overlap = target_tokens & tokens
        if not any(len(token) >= 4 for token in overlap):
            continue
        if tokens.issubset(target_tokens) or target_tokens.issubset(tokens):
            contained.append(candidate)
    if len(contained) == 1:
        return contained[0]

    raise ValueError(f"Sem correspondência segura V12 Rich para {team} em {league}")


def _synthetic_fixture_row(
    fixture_id: int,
    fixture_date: pd.Timestamp,
    league: str,
    home_team: str,
    away_team: str,
    odd_over25: float,
    odd_under25: float,
) -> dict:
    return {
        "id": fixture_id,
        "date": fixture_date,
        "league": league,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": 0.0,
        "away_goals": 0.0,
        "actual_over25": 0,
        "odd_over25_open": float(odd_over25),
        "odd_under25_open": float(odd_under25),
        "home_shots": np.nan,
        "away_shots": np.nan,
        "home_shots_target": np.nan,
        "away_shots_target": np.nan,
        "home_corners": np.nan,
        "away_corners": np.nan,
    }


def build_fixture_feature_rows(
    matches: pd.DataFrame,
    fixtures: list[dict],
) -> dict[str, dict]:
    """Build future feature rows with the exact historical training pipeline.

    Fixtures are evaluated by calendar date. Only real matches strictly before that
    date are available, so synthetic fixtures can never contaminate later fixtures.
    All fixtures on the same date are frozen together before any synthetic result.
    """
    if not fixtures:
        return {}

    resolved: list[dict] = []
    for fixture in fixtures:
        fixture_date = pd.Timestamp(str(fixture["fixture_date"]))
        league = str(fixture["league"])
        home = safe_resolve_team(matches, league, str(fixture["home_team"]), fixture_date)
        away = safe_resolve_team(matches, league, str(fixture["away_team"]), fixture_date)
        over = _finite_number(fixture.get("odd_over25_open"))
        under = _finite_number(fixture.get("odd_under25_open"))
        if over is None or under is None or over <= 1.0 or under <= 1.0:
            raise ValueError("Odds inválidas para fixture V12 Rich")
        resolved.append({**fixture, "fixture_date_ts": fixture_date, "resolved_home": home, "resolved_away": away, "odd_over25_open": over, "odd_under25_open": under})

    results: dict[str, dict] = {}
    max_id = int(pd.to_numeric(matches.id, errors="coerce").max()) if len(matches) else 0
    for fixture_date, group in pd.DataFrame(resolved).groupby("fixture_date_ts", sort=True):
        history = matches[pd.to_datetime(matches.date, errors="coerce") < pd.Timestamp(fixture_date)].copy()
        synthetic_rows: list[dict] = []
        id_to_fixture: dict[int, dict] = {}
        for offset, fixture in enumerate(group.to_dict("records"), start=1):
            synthetic_id = max_id + offset
            synthetic_rows.append(_synthetic_fixture_row(
                synthetic_id,
                pd.Timestamp(fixture_date),
                str(fixture["league"]),
                str(fixture["resolved_home"]),
                str(fixture["resolved_away"]),
                float(fixture["odd_over25_open"]),
                float(fixture["odd_under25_open"]),
            ))
            id_to_fixture[synthetic_id] = fixture

        extended = pd.concat([history, pd.DataFrame(synthetic_rows)], ignore_index=True, sort=False)
        base = build_dataset(extended)
        rich = build_rich_features(extended)
        combined = base.set_index("id").join(rich, how="left")

        for synthetic_id, fixture in id_to_fixture.items():
            if synthetic_id not in combined.index:
                raise ValueError(
                    f"Features base insuficientes para {fixture['home_team']} vs {fixture['away_team']}"
                )
            row = combined.loc[synthetic_id]
            values: dict[str, float] = {}
            for name in RICH_ANALYST_FEATURES:
                value = _finite_number(row.get(name))
                if value is None:
                    raise ValueError(
                        f"Feature {name} insuficiente para {fixture['home_team']} vs {fixture['away_team']}"
                    )
                values[name] = value
            market_probability, overround = devig_market_probability(
                fixture["odd_over25_open"], fixture["odd_under25_open"]
            )
            results[str(fixture["fixture_key"])] = {
                "features": values,
                "resolved_home": str(fixture["resolved_home"]),
                "resolved_away": str(fixture["resolved_away"]),
                "market_probability": float(market_probability),
                "market_overround": float(overround),
            }

    return results
