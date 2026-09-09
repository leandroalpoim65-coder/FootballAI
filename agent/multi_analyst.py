from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 50.0
    return _clamp((value - low) / (high - low) * 100.0)


@dataclass(frozen=True)
class AnalystScores:
    goal_profile: float
    attack_pressure: float
    ml_score: float
    value_score: float
    stats_score: float


def score_pre_match(
    features_json: str | dict[str, Any],
    sports_probability: float,
    expected_value: float,
    sports_edge_market: float,
) -> AnalystScores:
    f = json.loads(features_json) if isinstance(features_json, str) else dict(features_json)

    expected_total = _safe_float(f.get("expected_total_goals"))
    over_avg = (
        _safe_float(f.get("home_over25_rate_5"))
        + _safe_float(f.get("away_over25_rate_5"))
    ) / 2.0
    btts_avg = (
        _safe_float(f.get("home_btts_rate_5"))
        + _safe_float(f.get("away_btts_rate_5"))
    ) / 2.0
    over_signal = _safe_float(f.get("over25_signal"))

    goal_profile = (
        0.35 * _scale(expected_total, 2.0, 3.6)
        + 0.30 * _scale(over_avg, 0.25, 0.85)
        + 0.20 * _scale(btts_avg, 0.20, 0.80)
        + 0.15 * _scale(over_signal, 0.0, 1.0)
    )

    scoring_avg = (
        _safe_float(f.get("home_scored_rate_5"))
        + _safe_float(f.get("away_scored_rate_5"))
    ) / 2.0
    scored_sum = (
        _safe_float(f.get("home_avg_scored_5"))
        + _safe_float(f.get("away_avg_scored_5"))
    )
    conceded_sum = (
        _safe_float(f.get("home_avg_conceded_5"))
        + _safe_float(f.get("away_avg_conceded_5"))
    )
    expected_scored = _safe_float(f.get("total_expected_scored"))
    clean_sheet_avg = (
        _safe_float(f.get("home_clean_sheet_rate_5"))
        + _safe_float(f.get("away_clean_sheet_rate_5"))
    ) / 2.0

    attack_pressure = (
        0.25 * _scale(scoring_avg, 0.35, 0.95)
        + 0.25 * _scale(scored_sum, 1.0, 4.0)
        + 0.20 * _scale(conceded_sum, 1.0, 4.0)
        + 0.15 * _scale(expected_scored, 1.2, 4.2)
        + 0.15 * (100.0 - _scale(clean_sheet_avg, 0.10, 0.65))
    )

    ml_score = _scale(float(sports_probability), 0.45, 0.70)
    value_score = (
        0.60 * _scale(float(expected_value), -0.05, 0.20)
        + 0.40 * _scale(float(sports_edge_market), -0.03, 0.10)
    )

    stats_score = 0.55 * goal_profile + 0.45 * attack_pressure

    return AnalystScores(
        goal_profile=round(_clamp(goal_profile), 3),
        attack_pressure=round(_clamp(attack_pressure), 3),
        ml_score=round(_clamp(ml_score), 3),
        value_score=round(_clamp(value_score), 3),
        stats_score=round(_clamp(stats_score), 3),
    )


def weighted_score(scores: AnalystScores, weights: tuple[float, float, float, float]) -> float:
    g, a, m, v = weights
    total = g + a + m + v
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return (
        scores.goal_profile * g
        + scores.attack_pressure * a
        + scores.ml_score * m
        + scores.value_score * v
    ) / total
