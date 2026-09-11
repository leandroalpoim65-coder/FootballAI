from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Iterable


@dataclass(frozen=True)
class OutcomePolicy:
    goal_threshold: float
    attack_threshold: float
    ml_threshold: float
    value_threshold: float
    min_votes: int
    min_expected_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyMetrics:
    n: int
    wins: int
    hit_rate: float | None
    profit: float
    roi: float | None
    avg_odd: float | None
    max_drawdown: float
    longest_losing_streak: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CHAMPION_POLICY = OutcomePolicy(
    goal_threshold=55.0,
    attack_threshold=55.0,
    ml_threshold=65.0,
    value_threshold=60.0,
    min_votes=3,
    min_expected_value=0.0,
)


def qualifies(row: dict[str, Any], policy: OutcomePolicy) -> bool:
    if float(row.get("expected_value") or 0.0) < policy.min_expected_value:
        return False

    votes = sum(
        (
            float(row.get("goal_score") or 0.0) >= policy.goal_threshold,
            float(row.get("attack_score") or 0.0) >= policy.attack_threshold,
            float(row.get("ml_score") or 0.0) >= policy.ml_threshold,
            float(row.get("value_score") or 0.0) >= policy.value_threshold,
        )
    )
    return votes >= policy.min_votes


def counterfactual_profit(row: dict[str, Any]) -> float:
    odd = float(row["odd_over25_open"])
    return odd - 1.0 if int(row["result_over25"]) == 1 else -1.0


def metrics(rows: Iterable[dict[str, Any]], policy: OutcomePolicy) -> PolicyMetrics:
    selected = [row for row in rows if qualifies(row, policy)]
    if not selected:
        return PolicyMetrics(0, 0, None, 0.0, None, None, 0.0, 0)

    ordered = sorted(
        selected,
        key=lambda r: (
            str(r.get("fixture_date") or ""),
            str(r.get("fixture_time") or ""),
            int(r.get("id") or 0),
        ),
    )

    wins = sum(int(row["result_over25"]) for row in ordered)
    profits = [counterfactual_profit(row) for row in ordered]
    total_profit = sum(profits)

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    losing = 0
    longest_losing = 0
    for row_profit in profits:
        cumulative += row_profit
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
        if row_profit < 0:
            losing += 1
            longest_losing = max(longest_losing, losing)
        else:
            losing = 0

    n = len(ordered)
    return PolicyMetrics(
        n=n,
        wins=wins,
        hit_rate=wins / n,
        profit=total_profit,
        roi=total_profit / n,
        avg_odd=sum(float(row["odd_over25_open"]) for row in ordered) / n,
        max_drawdown=max_drawdown,
        longest_losing_streak=longest_losing,
    )


def unique_date_split(
    rows: list[dict[str, Any]],
    train_fraction: float = 0.70,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    dates = sorted({str(row["fixture_date"]) for row in rows})
    if len(dates) < 2:
        return list(rows), [], dates[0] if dates else None

    split_index = int(len(dates) * train_fraction)
    split_index = max(1, min(split_index, len(dates) - 1))
    train_dates = set(dates[:split_index])
    split_date = dates[split_index]

    train = [row for row in rows if str(row["fixture_date"]) in train_dates]
    validation = [row for row in rows if str(row["fixture_date"]) not in train_dates]
    return train, validation, split_date


def policy_grid() -> list[OutcomePolicy]:
    policies: list[OutcomePolicy] = []
    for goal in (50.0, 55.0, 60.0, 65.0):
        for attack in (50.0, 55.0, 60.0, 65.0):
            for ml in (55.0, 60.0, 65.0, 70.0):
                for value in (50.0, 55.0, 60.0, 65.0):
                    for min_votes in (2, 3, 4):
                        for min_ev in (0.0, 0.02, 0.05):
                            policies.append(
                                OutcomePolicy(
                                    goal_threshold=goal,
                                    attack_threshold=attack,
                                    ml_threshold=ml,
                                    value_threshold=value,
                                    min_votes=min_votes,
                                    min_expected_value=min_ev,
                                )
                            )
    return policies


def _selection_score(metric: PolicyMetrics) -> float:
    if metric.n == 0 or metric.roi is None or metric.hit_rate is None:
        return -999.0

    # Shrink small-sample ROI and penalize drawdown. This is deliberately
    # conservative: the learner is looking for stability, not the biggest
    # historical ROI.
    shrink = metric.n / (metric.n + 40.0)
    adjusted_roi = metric.roi * shrink
    drawdown_penalty = metric.max_drawdown / max(metric.n, 1) * 0.20
    sample_bonus = min(metric.n, 100) / 5000.0
    return adjusted_roi - drawdown_penalty + sample_bonus


def choose_candidate(
    train_rows: list[dict[str, Any]],
    min_train_bets: int = 25,
) -> tuple[OutcomePolicy | None, PolicyMetrics | None, int]:
    best_policy: OutcomePolicy | None = None
    best_metric: PolicyMetrics | None = None
    best_score = -999.0
    evaluated = 0

    for policy in policy_grid():
        metric = metrics(train_rows, policy)
        evaluated += 1
        if metric.n < min_train_bets or metric.roi is None or metric.roi <= 0:
            continue

        score = _selection_score(metric)
        if score > best_score:
            best_score = score
            best_policy = policy
            best_metric = metric

    return best_policy, best_metric, evaluated


def challenger_gate(
    champion: PolicyMetrics,
    candidate: PolicyMetrics,
    *,
    min_validation_bets: int = 15,
    min_roi_improvement: float = 0.03,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if candidate.n < min_validation_bets:
        reasons.append(
            f"candidate_validation_bets={candidate.n} < {min_validation_bets}"
        )
    if candidate.roi is None or candidate.roi <= 0:
        reasons.append("candidate_validation_roi_not_positive")

    champion_roi = champion.roi if champion.roi is not None else 0.0
    candidate_roi = candidate.roi if candidate.roi is not None else -999.0
    if candidate_roi < champion_roi + min_roi_improvement:
        reasons.append(
            "candidate_roi_improvement_below_gate"
        )

    # A challenger with materially worse risk is not promoted just because its
    # ROI is higher on a small sample.
    risk_limit = max(champion.max_drawdown * 1.25, 3.0)
    if candidate.max_drawdown > risk_limit:
        reasons.append("candidate_drawdown_above_gate")

    return len(reasons) == 0, reasons


def score_bin(value: Any) -> str:
    x = float(value or 0.0)
    if x < 50:
        return "<50"
    if x < 55:
        return "50-55"
    if x < 60:
        return "55-60"
    if x < 65:
        return "60-65"
    if x < 70:
        return "65-70"
    if x < 75:
        return "70-75"
    return "75+"


def vote_signature(row: dict[str, Any]) -> str:
    parts = []
    for label, field in (
        ("G", "goal_vote"),
        ("A", "attack_vote"),
        ("M", "ml_vote"),
        ("V", "value_vote"),
    ):
        if int(row.get(field) or 0):
            parts.append(label)
    return "+".join(parts) if parts else "NONE"


def build_patterns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions: list[tuple[str, callable]] = [
        ("decision", lambda r: str(r.get("decision") or "UNKNOWN")),
        ("vote_count", lambda r: str(int(r.get("vote_count") or 0))),
        ("vote_signature", vote_signature),
        ("goal_score", lambda r: score_bin(r.get("goal_score"))),
        ("attack_score", lambda r: score_bin(r.get("attack_score"))),
        ("ml_score", lambda r: score_bin(r.get("ml_score"))),
        ("value_score", lambda r: score_bin(r.get("value_score"))),
    ]

    out: list[dict[str, Any]] = []
    for dimension, segment_fn in dimensions:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(segment_fn(row), []).append(row)

        for segment, group in sorted(grouped.items()):
            n = len(group)
            wins = sum(int(row["result_over25"]) for row in group)
            profit = sum(counterfactual_profit(row) for row in group)
            out.append(
                {
                    "dimension": dimension,
                    "segment": segment,
                    "n": n,
                    "wins": wins,
                    "over_rate": wins / n if n else None,
                    "avg_odd": (
                        sum(float(row["odd_over25_open"]) for row in group) / n
                        if n else None
                    ),
                    "counterfactual_profit": profit,
                    "counterfactual_roi": profit / n if n else None,
                }
            )
    return out
