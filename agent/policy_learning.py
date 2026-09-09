from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import sqrt
from statistics import stdev
from typing import Any

from agent.decision_engine import (
    DEFAULT_POLICY,
    PolicyConfig,
    high_quality_close,
    safe_float,
    safe_int,
)


MIN_RESOLVED_ROWS = 120
MIN_TRAIN_ROWS = 75
MIN_VALIDATION_ROWS = 40
MIN_TRAIN_BETS = 20
MIN_VALIDATION_BETS = 12

# Approximate one-sided 90% lower confidence bound.
LCB_Z = 1.282

MIN_TRAIN_ROI_LIFT = 0.02
MIN_VALIDATION_ROI_LIFT = 0.03
MIN_VALIDATION_LCB = 0.00
MIN_WINNER_RETENTION = 0.35
MIN_LOSER_REJECTION = 0.20

CLV_GRID = (0.55, 0.575, 0.60, 0.625, 0.65)
EV_GRID = (0.03, 0.04, 0.05, 0.075, 0.10)
EDGE_GRID = (0.02, 0.025, 0.03, 0.04, 0.05)


@dataclass(frozen=True)
class PolicyMetrics:
    sample: int = 0
    wins: int = 0
    losses: int = 0
    profit_units: float = 0.0
    roi: float | None = None
    hit_rate: float | None = None
    roi_lcb: float | None = None
    hq_clv_sample: int = 0
    hq_clv_mean: float | None = None


@dataclass(frozen=True)
class LearningResult:
    decision: str
    candidate: PolicyConfig | None
    active_train: PolicyMetrics
    candidate_train: PolicyMetrics
    active_validation: PolicyMetrics
    candidate_validation: PolicyMetrics
    winner_retention: float | None
    loser_rejection: float | None
    reasons: tuple[str, ...]
    train_rows: int
    validation_rows: int
    split_date: str | None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def unit_return(row: dict[str, Any]) -> float | None:
    result = safe_float(row.get("result_over25"))
    odd = safe_float(row.get("odd_over25_open"))

    if result not in (0.0, 1.0):
        return None
    if odd is None or odd <= 1.0:
        return None

    if result == 1.0:
        return odd - 1.0

    return -1.0


def qualifies(
    row: dict[str, Any],
    policy: PolicyConfig,
) -> bool:
    if safe_int(row.get("original_clv_signal"), 0) != 1:
        return False

    odd = safe_float(row.get("odd_over25_open"))
    clv_probability = safe_float(row.get("clv_probability"))
    expected_value = safe_float(row.get("expected_value"))
    sports_edge = safe_float(row.get("sports_edge_market"))

    if (
        odd is None
        or clv_probability is None
        or expected_value is None
        or sports_edge is None
    ):
        return False

    return (
        policy.min_odd <= odd <= policy.max_odd
        and clv_probability >= policy.min_clv_probability
        and expected_value >= policy.min_expected_value
        and sports_edge >= policy.min_edge_market
    )


def policy_metrics(
    rows: list[dict[str, Any]],
    policy: PolicyConfig,
) -> PolicyMetrics:
    returns: list[float] = []
    wins = 0
    hq_clv: list[float] = []

    for row in rows:
        if not qualifies(row, policy):
            continue

        value = unit_return(row)
        result = safe_float(row.get("result_over25"))

        if value is None or result not in (0.0, 1.0):
            continue

        returns.append(value)
        wins += int(result)

        clv = safe_float(row.get("clv"))
        if (
            clv is not None
            and high_quality_close(row.get("closing_source"))
        ):
            hq_clv.append(clv)

    sample = len(returns)
    losses = sample - wins
    roi = _mean(returns)
    hit_rate = (wins / sample) if sample else None

    roi_lcb = None
    if roi is not None:
        if sample >= 2:
            roi_lcb = roi - (
                LCB_Z
                * stdev(returns)
                / sqrt(sample)
            )
        else:
            roi_lcb = roi

    return PolicyMetrics(
        sample=sample,
        wins=wins,
        losses=losses,
        profit_units=sum(returns),
        roi=roi,
        hit_rate=hit_rate,
        roi_lcb=roi_lcb,
        hq_clv_sample=len(hq_clv),
        hq_clv_mean=_mean(hq_clv),
    )


def temporal_split(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    str | None,
]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("fixture_date") or ""),
            safe_int(row.get("paper_trade_id"), 0),
        ),
    )

    total = len(ordered)
    if total < MIN_RESOLVED_ROWS:
        return ordered, [], None

    validation_size = max(
        MIN_VALIDATION_ROWS,
        int(round(total * 0.30)),
    )

    validation_size = min(
        validation_size,
        total - MIN_TRAIN_ROWS,
    )

    if validation_size < MIN_VALIDATION_ROWS:
        return ordered, [], None

    split_index = total - validation_size
    train = ordered[:split_index]
    validation = ordered[split_index:]

    split_date = (
        str(validation[0].get("fixture_date") or "")
        if validation
        else None
    )

    return train, validation, split_date


def candidate_grid(
    active: PolicyConfig,
) -> list[PolicyConfig]:
    clv_values = sorted(
        {
            *CLV_GRID,
            round(active.min_clv_probability, 3),
            round(
                max(
                    DEFAULT_POLICY.min_clv_probability,
                    active.min_clv_probability - 0.025,
                ),
                3,
            ),
            round(
                min(
                    0.70,
                    active.min_clv_probability + 0.025,
                ),
                3,
            ),
        }
    )

    ev_values = sorted(
        {
            *EV_GRID,
            round(active.min_expected_value, 3),
            round(
                max(
                    DEFAULT_POLICY.min_expected_value,
                    active.min_expected_value - 0.01,
                ),
                3,
            ),
            round(
                min(
                    0.15,
                    active.min_expected_value + 0.01,
                ),
                3,
            ),
        }
    )

    edge_values = sorted(
        {
            *EDGE_GRID,
            round(active.min_edge_market, 3),
            round(
                max(
                    DEFAULT_POLICY.min_edge_market,
                    active.min_edge_market - 0.005,
                ),
                3,
            ),
            round(
                min(
                    0.08,
                    active.min_edge_market + 0.005,
                ),
                3,
            ),
        }
    )

    candidates: list[PolicyConfig] = []

    for clv, ev, edge in product(
        clv_values,
        ev_values,
        edge_values,
    ):
        candidates.append(
            PolicyConfig(
                version="candidate",
                min_clv_probability=float(clv),
                min_expected_value=float(ev),
                min_edge_market=float(edge),
                min_odd=active.min_odd,
                max_odd=active.max_odd,
                strong_clv_probability=max(
                    active.strong_clv_probability,
                    float(clv),
                ),
                strong_expected_value=max(
                    active.strong_expected_value,
                    float(ev),
                ),
                strong_edge_market=max(
                    active.strong_edge_market,
                    float(edge),
                ),
            )
        )

    return candidates


def same_policy(
    left: PolicyConfig,
    right: PolicyConfig,
) -> bool:
    fields = (
        "min_clv_probability",
        "min_expected_value",
        "min_edge_market",
        "min_odd",
        "max_odd",
    )

    return all(
        abs(
            float(getattr(left, field))
            - float(getattr(right, field))
        ) < 1e-12
        for field in fields
    )


def filter_diagnostics(
    rows: list[dict[str, Any]],
    candidate: PolicyConfig,
) -> tuple[float | None, float | None]:
    original = [
        row
        for row in rows
        if (
            safe_int(
                row.get("original_clv_signal"),
                0,
            )
            == 1
            and unit_return(row) is not None
        )
    ]

    qualified = [
        row
        for row in original
        if qualifies(row, candidate)
    ]

    original_wins = sum(
        int(safe_float(row.get("result_over25")) == 1.0)
        for row in original
    )
    original_losses = len(original) - original_wins

    qualified_wins = sum(
        int(safe_float(row.get("result_over25")) == 1.0)
        for row in qualified
    )
    qualified_losses = len(qualified) - qualified_wins

    winner_retention = (
        qualified_wins / original_wins
        if original_wins > 0
        else None
    )

    loser_rejection = (
        (original_losses - qualified_losses)
        / original_losses
        if original_losses > 0
        else None
    )

    return winner_retention, loser_rejection


def choose_candidate(
    train: list[dict[str, Any]],
    active: PolicyConfig,
) -> tuple[PolicyConfig | None, PolicyMetrics, PolicyMetrics]:
    active_metrics = policy_metrics(
        train,
        active,
    )

    best_policy: PolicyConfig | None = None
    best_metrics = PolicyMetrics()
    best_score = float("-inf")

    for candidate in candidate_grid(active):
        metrics = policy_metrics(
            train,
            candidate,
        )

        if metrics.sample < MIN_TRAIN_BETS:
            continue

        if metrics.roi is None or metrics.roi_lcb is None:
            continue

        if (
            active_metrics.roi is not None
            and metrics.roi
            < active_metrics.roi + MIN_TRAIN_ROI_LIFT
        ):
            continue

        score = metrics.roi_lcb

        # Tie-break toward more evidence and smaller changes.
        if score > best_score:
            best_score = score
            best_policy = candidate
            best_metrics = metrics
        elif (
            best_policy is not None
            and abs(score - best_score) < 1e-12
            and metrics.sample > best_metrics.sample
        ):
            best_policy = candidate
            best_metrics = metrics

    return best_policy, active_metrics, best_metrics


def learn_policy(
    rows: list[dict[str, Any]],
    active: PolicyConfig,
) -> LearningResult:
    valid_rows = [
        row
        for row in rows
        if unit_return(row) is not None
    ]

    train, validation, split_date = temporal_split(
        valid_rows
    )

    empty = PolicyMetrics()

    if len(valid_rows) < MIN_RESOLVED_ROWS:
        return LearningResult(
            decision="WAIT",
            candidate=None,
            active_train=policy_metrics(train, active),
            candidate_train=empty,
            active_validation=empty,
            candidate_validation=empty,
            winner_retention=None,
            loser_rejection=None,
            reasons=(
                "amostra resolvida ainda insuficiente "
                f"({len(valid_rows)}/{MIN_RESOLVED_ROWS})",
            ),
            train_rows=len(train),
            validation_rows=len(validation),
            split_date=split_date,
        )

    if len(validation) < MIN_VALIDATION_ROWS:
        return LearningResult(
            decision="WAIT",
            candidate=None,
            active_train=policy_metrics(train, active),
            candidate_train=empty,
            active_validation=empty,
            candidate_validation=empty,
            winner_retention=None,
            loser_rejection=None,
            reasons=(
                "janela temporal de validação ainda insuficiente",
            ),
            train_rows=len(train),
            validation_rows=len(validation),
            split_date=split_date,
        )

    candidate, active_train, candidate_train = (
        choose_candidate(
            train,
            active,
        )
    )

    if candidate is None:
        return LearningResult(
            decision="KEEP",
            candidate=None,
            active_train=active_train,
            candidate_train=empty,
            active_validation=policy_metrics(
                validation,
                active,
            ),
            candidate_validation=empty,
            winner_retention=None,
            loser_rejection=None,
            reasons=(
                "nenhuma política candidata melhorou o treino "
                "com amostra mínima",
            ),
            train_rows=len(train),
            validation_rows=len(validation),
            split_date=split_date,
        )

    active_validation = policy_metrics(
        validation,
        active,
    )
    candidate_validation = policy_metrics(
        validation,
        candidate,
    )

    winner_retention, loser_rejection = (
        filter_diagnostics(
            validation,
            candidate,
        )
    )

    reasons: list[str] = []

    if same_policy(candidate, active):
        reasons.append(
            "melhor candidato é equivalente à política ativa"
        )

    if candidate_validation.sample < MIN_VALIDATION_BETS:
        reasons.append(
            "candidato tem poucas apostas na validação "
            f"({candidate_validation.sample}/{MIN_VALIDATION_BETS})"
        )

    if (
        candidate_validation.roi is None
        or active_validation.roi is None
        or candidate_validation.roi
        < active_validation.roi + MIN_VALIDATION_ROI_LIFT
    ):
        reasons.append(
            "candidato não confirmou lift mínimo de ROI "
            "na validação temporal"
        )

    if (
        candidate_validation.roi_lcb is None
        or candidate_validation.roi_lcb
        < MIN_VALIDATION_LCB
    ):
        reasons.append(
            "limite inferior de confiança do ROI "
            "ainda não é positivo"
        )

    if (
        candidate_validation.hq_clv_sample >= 8
        and (
            candidate_validation.hq_clv_mean is None
            or candidate_validation.hq_clv_mean < 0.0
        )
    ):
        reasons.append(
            "CLV de alta qualidade do candidato não é positivo"
        )

    if (
        winner_retention is not None
        and winner_retention < MIN_WINNER_RETENTION
    ):
        reasons.append(
            "retenção de vencedoras demasiado baixa"
        )

    if (
        loser_rejection is not None
        and loser_rejection < MIN_LOSER_REJECTION
    ):
        reasons.append(
            "rejeição de perdedoras demasiado baixa"
        )

    if reasons:
        return LearningResult(
            decision="KEEP",
            candidate=candidate,
            active_train=active_train,
            candidate_train=candidate_train,
            active_validation=active_validation,
            candidate_validation=candidate_validation,
            winner_retention=winner_retention,
            loser_rejection=loser_rejection,
            reasons=tuple(reasons),
            train_rows=len(train),
            validation_rows=len(validation),
            split_date=split_date,
        )

    return LearningResult(
        decision="PROMOTE",
        candidate=candidate,
        active_train=active_train,
        candidate_train=candidate_train,
        active_validation=active_validation,
        candidate_validation=candidate_validation,
        winner_retention=winner_retention,
        loser_rejection=loser_rejection,
        reasons=(
            "candidato selecionado apenas no passado e "
            "confirmado numa janela temporal posterior",
            "ROI, confiança e eficiência do filtro "
            "cumpriram os gates de promoção",
        ),
        train_rows=len(train),
        validation_rows=len(validation),
        split_date=split_date,
    )


def metrics_dict(
    metrics: PolicyMetrics,
) -> dict[str, Any]:
    return asdict(metrics)
