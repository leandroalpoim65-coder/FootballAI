from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BASE_SPORT_FEATURES = (
    "home_avg_scored_5", "home_avg_conceded_5", "home_avg_goal_diff_5",
    "home_points_avg_5", "home_win_rate_5", "home_btts_rate_5",
    "home_over25_rate_5", "home_scored_rate_5", "home_clean_sheet_rate_5",
    "away_avg_scored_5", "away_avg_conceded_5", "away_avg_goal_diff_5",
    "away_points_avg_5", "away_win_rate_5", "away_btts_rate_5",
    "away_over25_rate_5", "away_scored_rate_5", "away_clean_sheet_rate_5",
    "home_home_avg_scored_5", "home_home_avg_conceded_5", "home_home_points_avg_5",
    "away_away_avg_scored_5", "away_away_avg_conceded_5", "away_away_points_avg_5",
    "home_rest_days", "away_rest_days", "home_elo", "away_elo", "elo_diff",
    "expected_home_goals", "expected_away_goals", "expected_total_goals",
    "expected_goal_difference", "total_expected_scored", "total_expected_conceded",
    "home_power_rating", "away_power_rating", "power_rating_diff", "form_points_diff",
    "btts_signal", "over25_signal", "scoring_signal", "clean_sheet_signal",
)
V12_CONTEXT_FEATURES = (
    "home_avg_scored_10", "home_avg_conceded_10", "home_over25_rate_10", "home_btts_rate_10",
    "away_avg_scored_10", "away_avg_conceded_10", "away_over25_rate_10", "away_btts_rate_10",
    "league_over25_rate_200", "league_avg_goals_200",
)
V12_SPORT_FEATURES = BASE_SPORT_FEATURES + V12_CONTEXT_FEATURES
V12_MARKET_FEATURES = ("market_probability", "market_overround", "log_odd_over25")
V12_FEATURES = V12_SPORT_FEATURES + V12_MARKET_FEATURES


@dataclass(frozen=True)
class V12Decision:
    probability: float
    market_probability: float
    edge: float
    expected_value: float
    min_edge: float
    min_ev: float
    decision: str
    confidence: str


def valid_odd(value: Any) -> float | None:
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(odd) or odd <= 1.0:
        return None
    return odd


def devig_market_probability(odd_over: Any, odd_under: Any) -> tuple[float, float]:
    over = valid_odd(odd_over); under = valid_odd(odd_under)
    if over is None or under is None:
        raise ValueError("Odds Over/Under inválidas para V12.")
    implied_over = 1.0 / over; implied_under = 1.0 / under
    total = implied_over + implied_under
    return implied_over / total, total - 1.0


def _team_long_snapshot(history: pd.DataFrame, league: str, team: str, fixture_date: pd.Timestamp) -> dict[str, float]:
    dates = pd.to_datetime(history["date"], errors="coerce")
    rows = history[
        (history["league"].astype(str) == str(league))
        & (history["team"].astype(str) == str(team))
        & (dates < fixture_date)
    ].sort_values(["date", "id"])
    if len(rows) < 3:
        raise ValueError(f"Histórico V12 insuficiente para {team}: {len(rows)} jogos.")
    recent = rows.tail(10)
    return {
        "avg_scored_10": float(recent["goals_for"].mean()),
        "avg_conceded_10": float(recent["goals_against"].mean()),
        "over25_rate_10": float(recent["over25"].mean()),
        "btts_rate_10": float(recent["btts"].mean()),
    }


def build_v12_context_features(history: pd.DataFrame, league: str, home_team: str, away_team: str, fixture_date: pd.Timestamp) -> dict[str, float]:
    fixture_date = pd.Timestamp(fixture_date)
    dates = pd.to_datetime(history["date"], errors="coerce")
    home = _team_long_snapshot(history, league, home_team, fixture_date)
    away = _team_long_snapshot(history, league, away_team, fixture_date)
    league_rows = history[(history["league"].astype(str) == str(league)) & (dates < fixture_date)].sort_values(["date", "id"])
    recent_league = league_rows.tail(400)
    if recent_league.empty:
        league_over, league_goals = 0.50, 2.50
    else:
        league_over = float(recent_league["over25"].mean())
        league_goals = float((recent_league["goals_for"] + recent_league["goals_against"]).mean())
    return {
        "home_avg_scored_10": home["avg_scored_10"], "home_avg_conceded_10": home["avg_conceded_10"],
        "home_over25_rate_10": home["over25_rate_10"], "home_btts_rate_10": home["btts_rate_10"],
        "away_avg_scored_10": away["avg_scored_10"], "away_avg_conceded_10": away["avg_conceded_10"],
        "away_over25_rate_10": away["over25_rate_10"], "away_btts_rate_10": away["btts_rate_10"],
        "league_over25_rate_200": league_over, "league_avg_goals_200": league_goals,
    }


def build_v12_feature_row(sports_features: dict[str, float], history: pd.DataFrame, league: str, home_team: str, away_team: str, fixture_date: pd.Timestamp, odd_over25: Any, odd_under25: Any) -> dict[str, float]:
    fair, overround = devig_market_probability(odd_over25, odd_under25)
    odd_over = valid_odd(odd_over25)
    if odd_over is None:
        raise ValueError("Odd Over inválida para V12.")
    row = {name: float(sports_features[name]) for name in BASE_SPORT_FEATURES}
    row.update(build_v12_context_features(history, league, home_team, away_team, fixture_date))
    row.update({"market_probability": float(fair), "market_overround": float(overround), "log_odd_over25": float(math.log(odd_over))})
    return row


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X); classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("O modelo V12 não contém a classe positiva 1.")
    return np.asarray(p[:, classes.index(1)], dtype=float)


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values)).reshape(-1, 1)


def market_blend_matrix(raw: np.ndarray, market_probability: np.ndarray, market_overround: np.ndarray) -> np.ndarray:
    raw_logit = _logit(raw).ravel()
    market_logit = _logit(market_probability).ravel()
    return np.column_stack([raw_logit, market_logit, np.asarray(market_overround, dtype=float)])


def calibrate_probabilities(bundle: dict[str, Any], raw: np.ndarray, market_probability: np.ndarray | None = None, market_overround: np.ndarray | None = None) -> np.ndarray:
    kind = str(bundle.get("calibration_kind") or "raw"); calibrator = bundle.get("calibrator")
    raw = np.asarray(raw, dtype=float)
    if kind == "sigmoid":
        p = calibrator.predict_proba(_logit(raw)); return np.asarray(p[:, list(calibrator.classes_).index(1)], dtype=float)
    if kind == "isotonic":
        return np.asarray(calibrator.predict(raw), dtype=float)
    if kind == "market_blend":
        if market_probability is None or market_overround is None:
            raise ValueError("Mercado em falta para calibrador market_blend.")
        X = market_blend_matrix(raw, market_probability, market_overround)
        p = calibrator.predict_proba(X); return np.asarray(p[:, list(calibrator.classes_).index(1)], dtype=float)
    return np.clip(raw, 1e-6, 1.0 - 1e-6)


def predict_v12_probability(bundle: dict[str, Any], feature_row: dict[str, float]) -> float:
    names = list(bundle.get("sport_feature_names") or V12_SPORT_FEATURES)
    missing = [name for name in names if name not in feature_row]
    if missing:
        raise ValueError("Features V12 em falta: " + ", ".join(missing))
    X = pd.DataFrame([{name: float(feature_row[name]) for name in names}])
    raw = _positive_probability(bundle["base_model"], X)
    calibrated = calibrate_probabilities(
        bundle, raw,
        np.asarray([float(feature_row["market_probability"])]),
        np.asarray([float(feature_row["market_overround"])]),
    )
    return float(np.clip(calibrated[0], 1e-6, 1.0 - 1e-6))


def evaluate_v12(probability: float, odd_over25: Any, odd_under25: Any, policy: dict[str, Any]) -> V12Decision:
    market_probability, _ = devig_market_probability(odd_over25, odd_under25)
    odd_over = valid_odd(odd_over25)
    if odd_over is None:
        raise ValueError("Odd Over inválida para decisão V12.")
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    edge = probability - market_probability; ev = probability * odd_over - 1.0
    min_edge = float(policy.get("min_edge", 0.04)); min_ev = float(policy.get("min_ev", 0.03))
    min_odd = float(policy.get("min_odd", 1.20)); max_odd = float(policy.get("max_odd", 4.00))
    qualifies = min_odd <= odd_over <= max_odd and edge >= min_edge and ev >= min_ev
    decision = "BET_SHADOW" if qualifies else "PASS"
    confidence = "HIGH" if qualifies and edge >= max(.08, 2*min_edge) and ev >= max(.08, 2*min_ev) else "NORMAL" if qualifies else "PASS"
    return V12Decision(probability, float(market_probability), float(edge), float(ev), min_edge, min_ev, decision, confidence)
