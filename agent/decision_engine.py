from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


DEFAULT_POLICY_VERSION = "agent-v1.0"
POLICY_VERSION = DEFAULT_POLICY_VERSION

# Defaults remain conservative. The autonomous learner may promote
# stricter/looser values only after temporal holdout validation.
MIN_CLV_PROBABILITY = 0.55
MIN_EXPECTED_VALUE = 0.03
MIN_EDGE_MARKET = 0.02
MIN_ODD = 1.30
MAX_ODD = 4.00

STRONG_CLV_PROBABILITY = 0.60
STRONG_EXPECTED_VALUE = 0.05
STRONG_EDGE_MARKET = 0.03

# Live shadow data can veto weak leagues only after enough observations.
LEAGUE_MIN_SAMPLE_FOR_VETO = 40
LEAGUE_BAD_SHRUNK_ROI = -0.05
LEAGUE_MIN_HQ_CLV_FOR_VETO = 15
LEAGUE_BAD_HQ_CLV = 0.0

# Empirical-Bayes shrinkage prevents small league samples from dominating.
ROI_PRIOR_STRENGTH = 50.0
HIT_PRIOR_STRENGTH = 40.0


@dataclass(frozen=True)
class PolicyConfig:
    version: str = DEFAULT_POLICY_VERSION
    min_clv_probability: float = MIN_CLV_PROBABILITY
    min_expected_value: float = MIN_EXPECTED_VALUE
    min_edge_market: float = MIN_EDGE_MARKET
    min_odd: float = MIN_ODD
    max_odd: float = MAX_ODD
    strong_clv_probability: float = STRONG_CLV_PROBABILITY
    strong_expected_value: float = STRONG_EXPECTED_VALUE
    strong_edge_market: float = STRONG_EDGE_MARKET


DEFAULT_POLICY = PolicyConfig()


@dataclass(frozen=True)
class PerformanceStats:
    sample: int = 0
    wins: int = 0
    stake: float = 0.0
    profit: float = 0.0
    hit_rate: float | None = None
    roi: float | None = None
    hq_clv_sample: int = 0
    hq_clv_mean: float | None = None


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    tier: str
    expected_value: float
    sports_edge_market: float
    league_roi_shrunk: float | None
    league_hit_rate_shrunk: float | None
    reasons: tuple[str, ...]


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def shrink_mean(
    local_value: float | None,
    local_n: int,
    global_value: float | None,
    prior_strength: float,
) -> float | None:
    if local_value is None and global_value is None:
        return None
    if global_value is None:
        return local_value
    if local_value is None or local_n <= 0:
        return global_value

    return (
        local_value * float(local_n)
        + global_value * float(prior_strength)
    ) / (float(local_n) + float(prior_strength))


def high_quality_close(source: Any) -> bool:
    text = str(source or "").strip().casefold()
    if not text:
        return False

    if "the odds api" in text:
        return True

    if "football-data" in text and not (
        "fixtures" in text and "pre-match snapshot" in text
    ):
        return True

    return False


def build_stats(rows: list[dict[str, Any]]) -> PerformanceStats:
    if not rows:
        return PerformanceStats()

    sample = 0
    wins = 0
    total_stake = 0.0
    total_profit = 0.0
    hq_clv_values: list[float] = []

    for row in rows:
        result = safe_float(row.get("result_over25"))
        if result not in (0.0, 1.0):
            continue

        sample += 1
        wins += int(result)

        stake = safe_float(row.get("planned_stake"))
        if stake is None or stake <= 0:
            stake = 1.0
        total_stake += stake

        profit = safe_float(row.get("profit"))
        if profit is not None:
            total_profit += profit

        clv = safe_float(row.get("clv"))
        if clv is not None and high_quality_close(row.get("closing_source")):
            hq_clv_values.append(clv)

    hit_rate = (wins / sample) if sample > 0 else None
    roi = (total_profit / total_stake) if total_stake > 0 else None
    hq_clv_mean = (
        sum(hq_clv_values) / len(hq_clv_values)
        if hq_clv_values
        else None
    )

    return PerformanceStats(
        sample=sample,
        wins=wins,
        stake=total_stake,
        profit=total_profit,
        hit_rate=hit_rate,
        roi=roi,
        hq_clv_sample=len(hq_clv_values),
        hq_clv_mean=hq_clv_mean,
    )


def evaluate_candidate(
    row: dict[str, Any],
    sports_probability: float,
    global_stats: PerformanceStats,
    league_stats: PerformanceStats,
    policy: PolicyConfig | None = None,
) -> DecisionResult:
    policy = policy or DEFAULT_POLICY

    odd = safe_float(row.get("odd_over25_open"))
    market_probability = safe_float(row.get("fair_probability_open"))
    clv_probability = safe_float(row.get("clv_probability"))
    signal_threshold = safe_float(row.get("signal_threshold"))

    if odd is None or odd <= 1.0:
        raise ValueError("odd_over25_open inválida")
    if market_probability is None or not (0.0 < market_probability < 1.0):
        raise ValueError("fair_probability_open inválida")
    if clv_probability is None or not (0.0 <= clv_probability <= 1.0):
        raise ValueError("clv_probability inválida")
    if not (0.0 <= sports_probability <= 1.0):
        raise ValueError("sports_probability inválida")

    if signal_threshold is None:
        signal_threshold = policy.min_clv_probability

    frozen_clv_threshold = max(
        policy.min_clv_probability,
        signal_threshold,
    )

    expected_value = sports_probability * odd - 1.0
    sports_edge_market = sports_probability - market_probability

    league_roi_shrunk = shrink_mean(
        league_stats.roi,
        league_stats.sample,
        global_stats.roi,
        ROI_PRIOR_STRENGTH,
    )
    league_hit_rate_shrunk = shrink_mean(
        league_stats.hit_rate,
        league_stats.sample,
        global_stats.hit_rate,
        HIT_PRIOR_STRENGTH,
    )

    reasons: list[str] = []

    clv_signal = safe_int(row.get("signal"), 0) == 1
    clv_ok = (
        clv_signal
        and clv_probability >= frozen_clv_threshold
    )
    odds_ok = policy.min_odd <= odd <= policy.max_odd
    ev_ok = expected_value >= policy.min_expected_value
    edge_ok = sports_edge_market >= policy.min_edge_market

    if not clv_signal:
        reasons.append("sem sinal CLV original")
    elif clv_probability < frozen_clv_threshold:
        reasons.append(
            f"CLV {clv_probability:.1%} abaixo do limiar "
            f"{frozen_clv_threshold:.1%}"
        )

    if not odds_ok:
        reasons.append(
            f"odd {odd:.3f} fora do intervalo "
            f"{policy.min_odd:.2f}-{policy.max_odd:.2f}"
        )

    if not ev_ok:
        reasons.append(
            f"EV desportivo {expected_value:.1%} abaixo de "
            f"{policy.min_expected_value:.1%}"
        )

    if not edge_ok:
        reasons.append(
            f"edge vs mercado {sports_edge_market:.1%} abaixo de "
            f"{policy.min_edge_market:.1%}"
        )

    if not clv_ok:
        return DecisionResult(
            decision="REJECT",
            tier="CLV_REJECT",
            expected_value=expected_value,
            sports_edge_market=sports_edge_market,
            league_roi_shrunk=league_roi_shrunk,
            league_hit_rate_shrunk=league_hit_rate_shrunk,
            reasons=tuple(reasons),
        )

    if not odds_ok:
        return DecisionResult(
            decision="REJECT",
            tier="ODDS_REJECT",
            expected_value=expected_value,
            sports_edge_market=sports_edge_market,
            league_roi_shrunk=league_roi_shrunk,
            league_hit_rate_shrunk=league_hit_rate_shrunk,
            reasons=tuple(reasons),
        )

    if not (ev_ok and edge_ok):
        reasons.append(
            "modelos CLV e desportivo não estão suficientemente alinhados"
        )
        return DecisionResult(
            decision="WATCH",
            tier="MODEL_DISAGREEMENT",
            expected_value=expected_value,
            sports_edge_market=sports_edge_market,
            league_roi_shrunk=league_roi_shrunk,
            league_hit_rate_shrunk=league_hit_rate_shrunk,
            reasons=tuple(reasons),
        )

    league_veto = (
        league_stats.sample >= LEAGUE_MIN_SAMPLE_FOR_VETO
        and league_roi_shrunk is not None
        and league_roi_shrunk < LEAGUE_BAD_SHRUNK_ROI
        and league_stats.hq_clv_sample >= LEAGUE_MIN_HQ_CLV_FOR_VETO
        and league_stats.hq_clv_mean is not None
        and league_stats.hq_clv_mean <= LEAGUE_BAD_HQ_CLV
    )

    if league_veto:
        reasons.append(
            "liga em degradação no shadow: ROI ajustado negativo "
            "e CLV de alta qualidade não positivo"
        )
        return DecisionResult(
            decision="WATCH",
            tier="LEAGUE_DEGRADED",
            expected_value=expected_value,
            sports_edge_market=sports_edge_market,
            league_roi_shrunk=league_roi_shrunk,
            league_hit_rate_shrunk=league_hit_rate_shrunk,
            reasons=tuple(reasons),
        )

    strong = (
        clv_probability >= max(
            policy.strong_clv_probability,
            frozen_clv_threshold,
        )
        and expected_value >= max(
            policy.strong_expected_value,
            policy.min_expected_value,
        )
        and sports_edge_market >= max(
            policy.strong_edge_market,
            policy.min_edge_market,
        )
    )

    if league_stats.sample < LEAGUE_MIN_SAMPLE_FOR_VETO:
        reasons.append(
            f"liga ainda com amostra limitada ({league_stats.sample})"
        )

    reasons.append(
        "CLV + modelo desportivo + mercado alinhados "
        f"({policy.version})"
    )

    return DecisionResult(
        decision="QUALIFIED_SHADOW",
        tier="A" if strong else "B",
        expected_value=expected_value,
        sports_edge_market=sports_edge_market,
        league_roi_shrunk=league_roi_shrunk,
        league_hit_rate_shrunk=league_hit_rate_shrunk,
        reasons=tuple(reasons),
    )
