from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

import pandas as pd

MEMORY_VERSION = "agent-learning-v5.0"


@dataclass(frozen=True)
class MemorySummary:
    historical_rows: int
    resolved_rows: int
    open_rows: int
    resolved_roi: float | None
    resolved_hit_rate: float | None
    open_clv_mean: float | None
    open_sports_mean: float | None
    open_ev_mean: float | None
    drift_score: float | None
    state: str


def _float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if isfinite(x) else None


def _mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def resolved_roi(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    if frame is None or frame.empty:
        return None, None
    result = pd.to_numeric(frame.get("result_over25"), errors="coerce")
    odd = pd.to_numeric(frame.get("odd_over25_open"), errors="coerce")
    valid = result.isin([0, 1]) & odd.gt(1)
    if not valid.any():
        return None, None
    r = result[valid]
    o = odd[valid]
    profit = pd.Series([-1.0] * len(r), index=r.index)
    profit.loc[r.eq(1)] = o.loc[r.eq(1)] - 1.0
    return float(profit.mean()), float(r.mean())


def standardized_mean_shift(
    resolved: pd.DataFrame,
    open_rows: pd.DataFrame,
    columns: tuple[str, ...] = (
        "clv_probability",
        "sports_probability",
        "expected_value",
        "sports_edge_market",
        "odd_over25_open",
    ),
) -> float | None:
    if resolved is None or open_rows is None or resolved.empty or open_rows.empty:
        return None
    shifts: list[float] = []
    for column in columns:
        if column not in resolved.columns or column not in open_rows.columns:
            continue
        a = pd.to_numeric(resolved[column], errors="coerce").dropna()
        b = pd.to_numeric(open_rows[column], errors="coerce").dropna()
        if len(a) < 10 or len(b) < 5:
            continue
        scale = float(a.std(ddof=0))
        if not isfinite(scale) or scale <= 1e-9:
            continue
        shifts.append(abs(float(b.mean()) - float(a.mean())) / scale)
    if not shifts:
        return None
    return float(sqrt(sum(x * x for x in shifts) / len(shifts)))


def memory_state(summary: MemorySummary) -> str:
    if summary.resolved_rows < 30:
        return "COLD_START"
    if summary.drift_score is not None and summary.drift_score >= 1.0:
        return "DRIFT_REVIEW"
    if summary.resolved_rows < 120:
        return "EVIDENCE_BUILDING"
    return "TRACKING"


def build_memory_summary(
    historical: pd.DataFrame,
    resolved: pd.DataFrame,
    open_rows: pd.DataFrame,
) -> MemorySummary:
    roi, hit = resolved_roi(resolved)
    draft = MemorySummary(
        historical_rows=0 if historical is None else len(historical),
        resolved_rows=0 if resolved is None else len(resolved),
        open_rows=0 if open_rows is None else len(open_rows),
        resolved_roi=roi,
        resolved_hit_rate=hit,
        open_clv_mean=_mean(None if open_rows is None else open_rows.get("clv_probability")),
        open_sports_mean=_mean(None if open_rows is None else open_rows.get("sports_probability")),
        open_ev_mean=_mean(None if open_rows is None else open_rows.get("expected_value")),
        drift_score=standardized_mean_shift(resolved, open_rows),
        state="PENDING",
    )
    return MemorySummary(**{**draft.__dict__, "state": memory_state(draft)})
