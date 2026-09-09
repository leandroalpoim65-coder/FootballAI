from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.decision_engine import PerformanceStats, build_stats, evaluate_candidate


@dataclass(frozen=True)
class ReplaySummary:
    eligible: int
    qualified: int
    watch: int
    rejected: int
    qualified_wins: int
    qualified_profit: float
    qualified_roi: float | None


def flat_profit(result_over25: int, odd: float) -> float:
    return (odd - 1.0) if int(result_over25) == 1 else -1.0


def replay_rows(
    rows: list[dict[str, Any]],
    sports_probability_fn: Callable[[dict[str, Any]], float],
    policy,
) -> tuple[list[dict[str, Any]], ReplaySummary]:
    """Chronological blind replay.

    The decision step receives only pre-match fields. Outcome/profit/CLV/closing
    fields are revealed only after the decision has been frozen. Performance
    context for each fixture contains strictly earlier revealed rows.
    """
    ordered = sorted(rows, key=lambda r: (str(r.get("fixture_date") or ""), int(r.get("id") or 0)))
    revealed_signal_rows: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []

    for raw in ordered:
        # Deliberately whitelist pre-match data. Never pass result/profit/closing
        # fields into evaluate_candidate or the sports model callback.
        pre = {
            key: raw.get(key)
            for key in (
                "id", "prediction_timestamp", "fixture_date", "fixture_time",
                "league", "home_team", "away_team", "odd_over25_open",
                "fair_probability_open", "clv_probability", "signal_threshold",
                "signal", "features_json",
            )
        }

        global_stats = build_stats(revealed_signal_rows)
        league = str(pre.get("league") or "").strip()
        league_rows = [r for r in revealed_signal_rows if str(r.get("league") or "").strip() == league]
        league_stats = build_stats(league_rows) if league_rows else PerformanceStats()

        sports_probability = float(sports_probability_fn(pre))
        decision = evaluate_candidate(
            row=pre,
            sports_probability=sports_probability,
            global_stats=global_stats,
            league_stats=league_stats,
            policy=policy,
        )

        # Outcome is opened only after the decision object exists.
        result = int(raw["result_over25"])
        odd = float(raw["odd_over25_open"])
        cf_profit = flat_profit(result, odd)
        output.append({
            "paper_trade_id": int(raw["id"]),
            "fixture_date": raw.get("fixture_date"),
            "league": league,
            "home_team": raw.get("home_team"),
            "away_team": raw.get("away_team"),
            "decision": decision.decision,
            "tier": decision.tier,
            "sports_probability": sports_probability,
            "clv_probability": float(raw["clv_probability"]),
            "odd_over25_open": odd,
            "expected_value": decision.expected_value,
            "sports_edge_market": decision.sports_edge_market,
            "result_over25": result,
            "counterfactual_profit": cf_profit,
            "reasons": list(decision.reasons),
        })

        # Only after reveal may this row affect later context, matching the live
        # engine's use of resolved original CLV signals.
        if int(raw.get("signal") or 0) == 1:
            revealed_signal_rows.append({
                "fixture_date": raw.get("fixture_date"),
                "league": league,
                "signal": 1,
                "status": "RESOLVED",
                "planned_stake": 1.0,
                "profit": cf_profit,
                "result_over25": result,
                "clv": raw.get("clv"),
                "closing_source": raw.get("closing_source"),
            })

    qualified_rows = [r for r in output if r["decision"] == "QUALIFIED_SHADOW"]
    q_profit = sum(float(r["counterfactual_profit"]) for r in qualified_rows)
    summary = ReplaySummary(
        eligible=len(output),
        qualified=len(qualified_rows),
        watch=sum(r["decision"] == "WATCH" for r in output),
        rejected=sum(r["decision"] == "REJECT" for r in output),
        qualified_wins=sum(int(r["result_over25"]) for r in qualified_rows),
        qualified_profit=q_profit,
        qualified_roi=(q_profit / len(qualified_rows)) if qualified_rows else None,
    )
    return output, summary
