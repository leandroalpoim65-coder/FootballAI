from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from agent.decision_engine import high_quality_close


MONITOR_VERSION = "agent-monitor-v2.0"


@dataclass(frozen=True)
class PerformanceMetrics:
    sample: int = 0
    wins: int = 0
    losses: int = 0
    profit_units: float = 0.0
    roi: float | None = None
    hit_rate: float | None = None
    avg_open_odd: float | None = None
    avg_clv: float | None = None
    hq_clv_sample: int = 0
    hq_clv_mean: float | None = None
    avg_expected_value: float | None = None
    avg_sports_edge: float | None = None
    sports_brier: float | None = None


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


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def unit_profit(row: dict[str, Any]) -> float | None:
    result = safe_float(row.get("result_over25"))
    odd = safe_float(row.get("odd_over25_open"))

    if result not in (0.0, 1.0):
        return None
    if odd is None or odd <= 1.0:
        return None

    return (odd - 1.0) if result == 1.0 else -1.0


def build_metrics(rows: list[dict[str, Any]]) -> PerformanceMetrics:
    wins = 0
    profit_units = 0.0
    odds: list[float] = []
    clv_values: list[float] = []
    hq_clv_values: list[float] = []
    ev_values: list[float] = []
    edge_values: list[float] = []
    brier_values: list[float] = []

    for row in rows:
        profit = unit_profit(row)
        result = safe_float(row.get("result_over25"))
        odd = safe_float(row.get("odd_over25_open"))

        if profit is None or result not in (0.0, 1.0) or odd is None:
            continue

        wins += int(result)
        profit_units += profit
        odds.append(odd)

        clv = safe_float(row.get("clv"))
        if clv is not None:
            clv_values.append(clv)
            if high_quality_close(row.get("closing_source")):
                hq_clv_values.append(clv)

        expected_value = safe_float(row.get("expected_value"))
        if expected_value is not None:
            ev_values.append(expected_value)

        edge = safe_float(row.get("sports_edge_market"))
        if edge is not None:
            edge_values.append(edge)

        sports_probability = safe_float(row.get("sports_probability"))
        if sports_probability is not None and 0.0 <= sports_probability <= 1.0:
            brier_values.append((sports_probability - result) ** 2)

    sample = len(odds)
    losses = sample - wins
    roi = (profit_units / sample) if sample else None
    hit_rate = (wins / sample) if sample else None

    return PerformanceMetrics(
        sample=sample,
        wins=wins,
        losses=losses,
        profit_units=profit_units,
        roi=roi,
        hit_rate=hit_rate,
        avg_open_odd=_mean(odds),
        avg_clv=_mean(clv_values),
        hq_clv_sample=len(hq_clv_values),
        hq_clv_mean=_mean(hq_clv_values),
        avg_expected_value=_mean(ev_values),
        avg_sports_edge=_mean(edge_values),
        sports_brier=_mean(brier_values),
    )


def _bin_label(
    value: Any,
    cuts: tuple[float, ...],
    labels: tuple[str, ...],
) -> str:
    number = safe_float(value)
    if number is None:
        return "MISSING"

    for index, cut in enumerate(cuts):
        if number < cut:
            return labels[index]

    return labels[-1]


def clv_probability_bin(value: Any) -> str:
    return _bin_label(
        value,
        (0.45, 0.50, 0.55, 0.60, 0.65),
        (
            "<0.45",
            "0.45-0.50",
            "0.50-0.55",
            "0.55-0.60",
            "0.60-0.65",
            "0.65+",
        ),
    )


def sports_probability_bin(value: Any) -> str:
    return _bin_label(
        value,
        (0.45, 0.50, 0.55, 0.60, 0.65),
        (
            "<0.45",
            "0.45-0.50",
            "0.50-0.55",
            "0.55-0.60",
            "0.60-0.65",
            "0.65+",
        ),
    )


def expected_value_bin(value: Any) -> str:
    return _bin_label(
        value,
        (0.0, 0.03, 0.05, 0.10),
        (
            "<0",
            "0-0.03",
            "0.03-0.05",
            "0.05-0.10",
            "0.10+",
        ),
    )


def odds_bin(value: Any) -> str:
    return _bin_label(
        value,
        (1.60, 1.90, 2.20, 2.60),
        (
            "<1.60",
            "1.60-1.90",
            "1.90-2.20",
            "2.20-2.60",
            "2.60+",
        ),
    )


def build_segments(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    segments: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    def add(
        segment_type: str,
        key: str,
        row: dict[str, Any],
    ) -> None:
        segments.setdefault(
            (segment_type, key),
            [],
        ).append(row)

    for row in rows:
        add("baseline", "ALL_AGENT_DECISIONS", row)

        original_signal = (
            safe_int(row.get("original_clv_signal"), 0) == 1
        )
        decision = str(
            row.get("decision") or ""
        ).strip().upper()
        tier = str(
            row.get("tier") or ""
        ).strip().upper()
        league = (
            str(row.get("league") or "UNKNOWN").strip()
            or "UNKNOWN"
        )

        if original_signal:
            add("strategy", "ORIGINAL_SIGNAL", row)
            add("league_original", league, row)
            add(
                "clv_bin_original",
                clv_probability_bin(
                    row.get("clv_probability")
                ),
                row,
            )
            add(
                "sports_bin_original",
                sports_probability_bin(
                    row.get("sports_probability")
                ),
                row,
            )
            add(
                "ev_bin_original",
                expected_value_bin(
                    row.get("expected_value")
                ),
                row,
            )
            add(
                "odds_bin_original",
                odds_bin(
                    row.get("odd_over25_open")
                ),
                row,
            )

        if decision == "QUALIFIED_SHADOW":
            add("strategy", "QUALIFIED_SHADOW", row)

            if tier in {"A", "B"}:
                add(
                    "strategy",
                    f"QUALIFIED_{tier}",
                    row,
                )

            add("league_qualified", league, row)
            add(
                "clv_bin_qualified",
                clv_probability_bin(
                    row.get("clv_probability")
                ),
                row,
            )
            add(
                "sports_bin_qualified",
                sports_probability_bin(
                    row.get("sports_probability")
                ),
                row,
            )
            add(
                "ev_bin_qualified",
                expected_value_bin(
                    row.get("expected_value")
                ),
                row,
            )
            add(
                "odds_bin_qualified",
                odds_bin(
                    row.get("odd_over25_open")
                ),
                row,
            )

        elif decision == "WATCH":
            add("strategy", "WATCH", row)

        elif decision == "REJECT":
            add("strategy", "REJECT", row)

        if original_signal:
            if decision == "QUALIFIED_SHADOW":
                add(
                    "signal_filter",
                    "ORIGINAL_SIGNAL_QUALIFIED",
                    row,
                )
            else:
                add(
                    "signal_filter",
                    "ORIGINAL_SIGNAL_NOT_QUALIFIED",
                    row,
                )

            if decision == "WATCH":
                add(
                    "signal_filter",
                    "ORIGINAL_SIGNAL_WATCH",
                    row,
                )

            elif decision == "REJECT":
                add(
                    "signal_filter",
                    "ORIGINAL_SIGNAL_REJECT",
                    row,
                )

    return segments


def filter_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    original_rows = [
        row
        for row in rows
        if safe_int(
            row.get("original_clv_signal"),
            0,
        ) == 1
    ]

    qualified_rows = [
        row
        for row in original_rows
        if str(
            row.get("decision") or ""
        ).strip().upper() == "QUALIFIED_SHADOW"
    ]

    not_qualified_rows = [
        row
        for row in original_rows
        if str(
            row.get("decision") or ""
        ).strip().upper() != "QUALIFIED_SHADOW"
    ]

    original = build_metrics(original_rows)
    qualified = build_metrics(qualified_rows)
    rejected = build_metrics(not_qualified_rows)

    winner_retention = (
        qualified.wins / original.wins
        if original.wins > 0
        else None
    )

    loser_rejection = (
        (original.losses - qualified.losses)
        / original.losses
        if original.losses > 0
        else None
    )

    selection_rate = (
        qualified.sample / original.sample
        if original.sample > 0
        else None
    )

    roi_lift = (
        qualified.roi - original.roi
        if (
            qualified.roi is not None
            and original.roi is not None
        )
        else None
    )

    return {
        "original_sample": original.sample,
        "original_wins": original.wins,
        "original_losses": original.losses,
        "original_roi": original.roi,
        "qualified_sample": qualified.sample,
        "qualified_wins": qualified.wins,
        "qualified_losses": qualified.losses,
        "qualified_roi": qualified.roi,
        "not_qualified_sample": rejected.sample,
        "not_qualified_roi": rejected.roi,
        "winner_retention": winner_retention,
        "loser_rejection": loser_rejection,
        "selection_rate": selection_rate,
        "roi_lift": roi_lift,
    }


def monitor_state(
    diagnostics: dict[str, float | int | None],
) -> str:
    qualified_sample = int(
        diagnostics.get("qualified_sample") or 0
    )

    original_roi = safe_float(
        diagnostics.get("original_roi")
    )
    qualified_roi = safe_float(
        diagnostics.get("qualified_roi")
    )

    if qualified_sample < 20:
        return "COLD_START"

    if qualified_sample < 50:
        return "EVIDENCE_BUILDING"

    if (
        original_roi is not None
        and qualified_roi is not None
        and qualified_roi < original_roi - 0.05
    ):
        return "REVIEW"

    return "TRACKING"
