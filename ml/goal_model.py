from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Mapping


@dataclass(frozen=True)
class GoalModelConfig:
    """Configuração do motor de golos."""
    general_weight: float = 0.45
    venue_weight: float = 0.55
    league_shrinkage: float = 0.10
    elo_adjustment_strength: float = 0.0
    min_lambda: float = 0.15
    max_lambda: float = 4.50
    max_goals: int = 12


DEFAULT_CONFIG = GoalModelConfig()


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_probability(value: float) -> float:
    return _clip(float(value), 0.0, 1.0)


def fair_decimal_odd(probability: float) -> float | None:
    """Odd decimal justa para um mercado binário sem push."""
    probability = _safe_probability(probability)
    return None if probability <= 0.0 else 1.0 / probability


def expected_value(probability: float, market_odd: float) -> float:
    """EV por unidade apostada num mercado binário."""
    probability = _safe_probability(probability)
    market_odd = float(market_odd)
    if market_odd <= 1.0:
        raise ValueError("A odd de mercado deve ser superior a 1.0.")
    return probability * market_odd - 1.0


# ============================================================
# POISSON
# ============================================================

def poisson_probabilities(expected_goals: float, max_goals: int = 12) -> list[float]:
    """P(X=k), k=0..max_goals, por recorrência numérica estável."""
    expected_goals = float(expected_goals)
    if expected_goals < 0.0:
        raise ValueError("expected_goals não pode ser negativo.")
    if max_goals < 1:
        raise ValueError("max_goals deve ser pelo menos 1.")

    probabilities = [exp(-expected_goals)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * expected_goals / goals)
    return probabilities


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 12,
) -> list[list[float]]:
    """Matriz P(golos_casa=i, golos_fora=j), normalizada."""
    home_probs = poisson_probabilities(lambda_home, max_goals)
    away_probs = poisson_probabilities(lambda_away, max_goals)

    matrix = [
        [home_p * away_p for away_p in away_probs]
        for home_p in home_probs
    ]
    total = sum(sum(row) for row in matrix)
    if total <= 0.0:
        raise ValueError("A matriz de resultados tem probabilidade total inválida.")

    return [[probability / total for probability in row] for row in matrix]


def total_goals_distribution(matrix: list[list[float]]) -> dict[int, float]:
    distribution: dict[int, float] = {}
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            total = home_goals + away_goals
            distribution[total] = distribution.get(total, 0.0) + probability
    return distribution


def team_goal_distribution(
    matrix: list[list[float]],
    team: str,
) -> dict[int, float]:
    team = str(team).strip().lower()
    if team not in {"home", "away"}:
        raise ValueError("team deve ser 'home' ou 'away'.")

    if team == "home":
        return {goals: sum(row) for goals, row in enumerate(matrix)}

    return {
        goals: sum(row[goals] for row in matrix)
        for goals in range(len(matrix[0]))
    }


# ============================================================
# LAMBDAS A PARTIR DAS 43 FEATURES EXISTENTES
# ============================================================

def estimate_lambdas_from_features(
    sports_features: Mapping[str, Any],
    *,
    league_home_avg: float | None = None,
    league_away_avg: float | None = None,
    config: GoalModelConfig = DEFAULT_CONFIG,
) -> tuple[float, float]:
    """
    Converte as features de build_sports_features() em lambda_home/lambda_away.

    Versão inicial deliberadamente simples para ser calibrada em backtest:
    - forma geral ataque/defesa;
    - forma específica casa/fora;
    - regressão opcional à média da liga;
    - Elo opcional, desligado por defeito.
    """
    required = [
        "home_avg_scored_5",
        "home_avg_conceded_5",
        "away_avg_scored_5",
        "away_avg_conceded_5",
        "home_home_avg_scored_5",
        "home_home_avg_conceded_5",
        "away_away_avg_scored_5",
        "away_away_avg_conceded_5",
    ]
    missing = [
        name for name in required
        if _to_float(sports_features.get(name)) is None
    ]
    if missing:
        raise ValueError(
            "Faltam features válidas para o goal model: " + ", ".join(missing)
        )

    if config.general_weight < 0.0 or config.venue_weight < 0.0:
        raise ValueError("Os pesos do goal model não podem ser negativos.")

    weight_total = config.general_weight + config.venue_weight
    if weight_total <= 0.0:
        raise ValueError("A soma dos pesos do goal model deve ser superior a zero.")

    general_weight = config.general_weight / weight_total
    venue_weight = config.venue_weight / weight_total

    # Já corresponde à lógica atual de expected_home_goals / expected_away_goals.
    home_general = (
        float(sports_features["home_avg_scored_5"])
        + float(sports_features["away_avg_conceded_5"])
    ) / 2.0
    away_general = (
        float(sports_features["away_avg_scored_5"])
        + float(sports_features["home_avg_conceded_5"])
    ) / 2.0

    # Acrescenta a componente específica casa/fora já disponível nas 43 features.
    home_venue = (
        float(sports_features["home_home_avg_scored_5"])
        + float(sports_features["away_away_avg_conceded_5"])
    ) / 2.0
    away_venue = (
        float(sports_features["away_away_avg_scored_5"])
        + float(sports_features["home_home_avg_conceded_5"])
    ) / 2.0

    lambda_home = general_weight * home_general + venue_weight * home_venue
    lambda_away = general_weight * away_general + venue_weight * away_venue

    shrinkage = _clip(float(config.league_shrinkage), 0.0, 1.0)
    league_home = _to_float(league_home_avg)
    league_away = _to_float(league_away_avg)

    if league_home is not None and league_home > 0.0:
        lambda_home = (1.0 - shrinkage) * lambda_home + shrinkage * league_home
    if league_away is not None and league_away > 0.0:
        lambda_away = (1.0 - shrinkage) * lambda_away + shrinkage * league_away

    elo_diff = _to_float(sports_features.get("elo_diff"), 0.0) or 0.0
    if config.elo_adjustment_strength != 0.0:
        multiplier = exp(config.elo_adjustment_strength * elo_diff / 400.0)
        lambda_home *= multiplier
        lambda_away /= multiplier

    return (
        _clip(float(lambda_home), config.min_lambda, config.max_lambda),
        _clip(float(lambda_away), config.min_lambda, config.max_lambda),
    )


# ============================================================
# PROBABILIDADES DE MERCADO
# ============================================================

def _match_result_probabilities(matrix: list[list[float]]) -> dict[str, float]:
    home = draw = away = 0.0
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            if home_goals > away_goals:
                home += probability
            elif home_goals < away_goals:
                away += probability
            else:
                draw += probability
    return {
        "home": _safe_probability(home),
        "draw": _safe_probability(draw),
        "away": _safe_probability(away),
    }


def _btts_probabilities(matrix: list[list[float]]) -> dict[str, float]:
    yes = sum(
        probability
        for home_goals, row in enumerate(matrix)
        for away_goals, probability in enumerate(row)
        if home_goals > 0 and away_goals > 0
    )
    yes = _safe_probability(yes)
    return {"yes": yes, "no": 1.0 - yes}


def _over_probability(distribution: Mapping[int, float], line: float) -> float:
    return _safe_probability(
        sum(probability for goals, probability in distribution.items() if goals > line)
    )


def _under_probability(distribution: Mapping[int, float], line: float) -> float:
    return _safe_probability(
        sum(probability for goals, probability in distribution.items() if goals < line)
    )


# ============================================================
# ASIAN TOTALS
# ============================================================

def _single_line_settlement(goals: int, line: float, side: str) -> float:
    side = side.lower()
    if side == "over":
        return 1.0 if goals > line else (0.0 if goals == line else -1.0)
    if side == "under":
        return 1.0 if goals < line else (0.0 if goals == line else -1.0)
    raise ValueError("side deve ser 'over' ou 'under'.")


def _split_asian_line(line: float) -> tuple[float, ...]:
    scaled = round(float(line) * 4)
    normalized = scaled / 4.0
    if abs(normalized - float(line)) > 1e-8:
        raise ValueError("A linha asiática deve usar incrementos de 0.25.")

    quarter = scaled % 4
    if quarter in {0, 2}:
        return (normalized,)
    return (normalized - 0.25, normalized + 0.25)


def asian_total_probabilities(
    total_distribution: Mapping[int, float],
    line: float,
    side: str,
) -> dict[str, float | None]:
    """
    Probabilidades de win / half_win / push / half_loss / loss
    e odd justa para uma linha Asian Total.
    """
    split_lines = _split_asian_line(line)
    outcomes = {
        "win": 0.0,
        "half_win": 0.0,
        "push": 0.0,
        "half_loss": 0.0,
        "loss": 0.0,
    }

    for goals, probability in total_distribution.items():
        settlement = sum(
            _single_line_settlement(goals, split_line, side)
            for split_line in split_lines
        ) / len(split_lines)

        if settlement >= 0.999999:
            key = "win"
        elif settlement >= 0.499999:
            key = "half_win"
        elif settlement > -0.000001:
            key = "push"
        elif settlement > -0.999999:
            key = "half_loss"
        else:
            key = "loss"
        outcomes[key] += float(probability)

    win_equivalent = outcomes["win"] + 0.5 * outcomes["half_win"]
    loss_equivalent = outcomes["loss"] + 0.5 * outcomes["half_loss"]
    fair_odd = (
        1.0 + loss_equivalent / win_equivalent
        if win_equivalent > 0.0
        else None
    )

    return {
        **{key: _safe_probability(value) for key, value in outcomes.items()},
        "fair_odd": fair_odd,
    }


def asian_expected_value(
    settlement_probabilities: Mapping[str, float | None],
    market_odd: float,
) -> float:
    """EV por unidade para uma linha Asian Total."""
    market_odd = float(market_odd)
    if market_odd <= 1.0:
        raise ValueError("A odd de mercado deve ser superior a 1.0.")

    profit = market_odd - 1.0
    return (
        float(settlement_probabilities["win"] or 0.0) * profit
        + float(settlement_probabilities["half_win"] or 0.0) * 0.5 * profit
        - float(settlement_probabilities["half_loss"] or 0.0) * 0.5
        - float(settlement_probabilities["loss"] or 0.0)
    )


# ============================================================
# PROJEÇÃO COMPLETA
# ============================================================

def project_match_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    *,
    config: GoalModelConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    lambda_home = _clip(float(lambda_home), config.min_lambda, config.max_lambda)
    lambda_away = _clip(float(lambda_away), config.min_lambda, config.max_lambda)

    matrix = score_matrix(lambda_home, lambda_away, config.max_goals)
    total_distribution = total_goals_distribution(matrix)
    home_distribution = team_goal_distribution(matrix, "home")
    away_distribution = team_goal_distribution(matrix, "away")

    match_result = _match_result_probabilities(matrix)
    btts = _btts_probabilities(matrix)

    totals: dict[str, dict[str, float | None]] = {}
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over = _over_probability(total_distribution, line)
        under = _under_probability(total_distribution, line)
        totals[f"{line:.1f}"] = {
            "over": over,
            "under": under,
            "fair_odd_over": fair_decimal_odd(over),
            "fair_odd_under": fair_decimal_odd(under),
        }

    team_totals: dict[str, dict[str, dict[str, float | None]]] = {
        "home": {},
        "away": {},
    }
    for team_name, distribution in [
        ("home", home_distribution),
        ("away", away_distribution),
    ]:
        for line in [0.5, 1.5, 2.5, 3.5]:
            over = _over_probability(distribution, line)
            under = _under_probability(distribution, line)
            team_totals[team_name][f"{line:.1f}"] = {
                "over": over,
                "under": under,
                "fair_odd_over": fair_decimal_odd(over),
                "fair_odd_under": fair_decimal_odd(under),
            }

    asian_totals: dict[str, dict[str, dict[str, float | None]]] = {}
    for line in [2.0, 2.25, 2.5, 2.75, 3.0]:
        asian_totals[f"{line:.2f}"] = {
            "over": asian_total_probabilities(total_distribution, line, "over"),
            "under": asian_total_probabilities(total_distribution, line, "under"),
        }

    score_candidates = [
        {
            "home_goals": home_goals,
            "away_goals": away_goals,
            "probability": probability,
        }
        for home_goals, row in enumerate(matrix)
        for away_goals, probability in enumerate(row)
    ]
    top_scores = sorted(
        score_candidates,
        key=lambda item: float(item["probability"]),
        reverse=True,
    )[:5]

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "expected_total_goals": lambda_home + lambda_away,
        "match_result": {
            **match_result,
            "fair_odd_home": fair_decimal_odd(match_result["home"]),
            "fair_odd_draw": fair_decimal_odd(match_result["draw"]),
            "fair_odd_away": fair_decimal_odd(match_result["away"]),
        },
        "btts": {
            **btts,
            "fair_odd_yes": fair_decimal_odd(btts["yes"]),
            "fair_odd_no": fair_decimal_odd(btts["no"]),
        },
        "totals": totals,
        "asian_totals": asian_totals,
        "team_totals": team_totals,
        "top_scores": top_scores,
    }


def project_match(
    sports_features: Mapping[str, Any],
    *,
    league_home_avg: float | None = None,
    league_away_avg: float | None = None,
    config: GoalModelConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Entrada principal para o get_todays_matches.py."""
    lambda_home, lambda_away = estimate_lambdas_from_features(
        sports_features,
        league_home_avg=league_home_avg,
        league_away_avg=league_away_avg,
        config=config,
    )
    return project_match_from_lambdas(
        lambda_home,
        lambda_away,
        config=config,
    )


# ============================================================
# FORMATO PLANO PARA CSV / SQLITE / SHADOW TEST
# ============================================================

def flatten_projection(projection: Mapping[str, Any]) -> dict[str, float | None]:
    """Campos principais prontos para persistência e análise."""
    result: dict[str, float | None] = {
        "goal_lambda_home": float(projection["lambda_home"]),
        "goal_lambda_away": float(projection["lambda_away"]),
        "goal_expected_total": float(projection["expected_total_goals"]),
        "goal_prob_home_win": float(projection["match_result"]["home"]),
        "goal_prob_draw": float(projection["match_result"]["draw"]),
        "goal_prob_away_win": float(projection["match_result"]["away"]),
        "goal_prob_btts_yes": float(projection["btts"]["yes"]),
        "goal_prob_btts_no": float(projection["btts"]["no"]),
    }

    for line in ["1.5", "2.5", "3.5"]:
        suffix = line.replace(".", "")
        result[f"goal_prob_over_{suffix}"] = float(
            projection["totals"][line]["over"]
        )
        result[f"goal_prob_under_{suffix}"] = float(
            projection["totals"][line]["under"]
        )

    for team_name in ["home", "away"]:
        for line in ["0.5", "1.5", "2.5"]:
            suffix = line.replace(".", "")
            result[f"goal_prob_{team_name}_over_{suffix}"] = float(
                projection["team_totals"][team_name][line]["over"]
            )

    return result


if __name__ == "__main__":
    # Smoke test independente da base de dados.
    demo_features = {
        "home_avg_scored_5": 1.8,
        "home_avg_conceded_5": 0.9,
        "away_avg_scored_5": 1.1,
        "away_avg_conceded_5": 1.6,
        "home_home_avg_scored_5": 2.0,
        "home_home_avg_conceded_5": 0.8,
        "away_away_avg_scored_5": 0.9,
        "away_away_avg_conceded_5": 1.8,
        "elo_diff": 85.0,
    }

    demo = project_match(demo_features)
    print("lambda_home:", round(demo["lambda_home"], 3))
    print("lambda_away:", round(demo["lambda_away"], 3))
    print("Over 2.5:", round(demo["totals"]["2.5"]["over"] * 100, 2), "%")
    print("BTTS:", round(demo["btts"]["yes"] * 100, 2), "%")
    print("Campos flat:", len(flatten_projection(demo)))
