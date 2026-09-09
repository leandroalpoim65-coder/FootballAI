from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _f(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key)
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class GameAnalysis:
    game_score: float
    reasons: tuple[str, ...]


def analyze_features(features_json: str | dict[str, Any]) -> GameAnalysis:
    f = json.loads(features_json) if isinstance(features_json, str) else dict(features_json)
    score = 50.0
    reasons: list[str] = []

    exp_total = _f(f, "expected_total_goals")
    if exp_total >= 3.2:
        score += 18; reasons.append(f"xG total muito alto {exp_total:.2f}")
    elif exp_total >= 2.9:
        score += 13; reasons.append(f"xG total alto {exp_total:.2f}")
    elif exp_total >= 2.6:
        score += 8; reasons.append(f"xG total favorável {exp_total:.2f}")
    elif exp_total and exp_total < 2.2:
        score -= 12; reasons.append(f"xG total baixo {exp_total:.2f}")

    over_home = _f(f, "home_over25_rate_5")
    over_away = _f(f, "away_over25_rate_5")
    over_avg = (over_home + over_away) / 2 if (over_home or over_away) else 0.0
    if over_avg >= .70:
        score += 14; reasons.append(f"tendência Over muito forte {over_avg:.0%}")
    elif over_avg >= .55:
        score += 9; reasons.append(f"tendência Over forte {over_avg:.0%}")
    elif over_avg and over_avg < .35:
        score -= 9; reasons.append(f"tendência Over fraca {over_avg:.0%}")

    btts_avg = (_f(f, "home_btts_rate_5") + _f(f, "away_btts_rate_5")) / 2
    if btts_avg >= .65:
        score += 7; reasons.append(f"BTTS alto {btts_avg:.0%}")
    elif btts_avg and btts_avg < .30:
        score -= 5; reasons.append(f"BTTS baixo {btts_avg:.0%}")

    scoring_avg = (_f(f, "home_scored_rate_5") + _f(f, "away_scored_rate_5")) / 2
    if scoring_avg >= .80:
        score += 7; reasons.append(f"ambas marcam com frequência {scoring_avg:.0%}")
    elif scoring_avg and scoring_avg < .50:
        score -= 5; reasons.append(f"capacidade de marcar limitada {scoring_avg:.0%}")

    home_for = _f(f, "home_avg_scored_5")
    away_for = _f(f, "away_avg_scored_5")
    home_against = _f(f, "home_avg_conceded_5")
    away_against = _f(f, "away_avg_conceded_5")
    attack_def_mix = (home_for + away_for + home_against + away_against) / 2.0
    if attack_def_mix >= 3.2:
        score += 10; reasons.append("produção/concessão recente muito alta")
    elif attack_def_mix >= 2.7:
        score += 6; reasons.append("produção/concessão recente favorável")
    elif attack_def_mix and attack_def_mix < 2.0:
        score -= 7; reasons.append("produção/concessão recente baixa")

    home_cs = _f(f, "home_clean_sheet_rate_5")
    away_cs = _f(f, "away_clean_sheet_rate_5")
    if (home_cs + away_cs) / 2 >= .50:
        score -= 6; reasons.append("taxa de clean sheet elevada")

    over_signal = _f(f, "over25_signal")
    if over_signal >= .5:
        score += 6; reasons.append("sinal Over interno positivo")

    scoring_signal = _f(f, "scoring_signal")
    if scoring_signal >= .5:
        score += 4; reasons.append("sinal de scoring positivo")

    return GameAnalysis(game_score=round(_clamp(score), 2), reasons=tuple(reasons))
