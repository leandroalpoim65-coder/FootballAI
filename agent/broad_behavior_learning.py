from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable


LEARNING_VERSION = "broad-behavior-v7.0"


@dataclass(frozen=True)
class SegmentMetrics:
    sample: int
    wins: int
    profit: float
    roi: float | None
    hit_rate: float | None
    avg_odd: float | None
    avg_clv_probability: float | None
    avg_sports_probability: float | None
    avg_expected_value: float | None
    avg_sports_edge_market: float | None


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flat_profit(result_over25: int, odd: float) -> float:
    return (float(odd) - 1.0) if int(result_over25) == 1 else -1.0


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def metrics(rows: list[dict[str, Any]]) -> SegmentMetrics:
    valid: list[dict[str, Any]] = []
    for row in rows:
        result = _f(row.get("result_over25"))
        odd = _f(row.get("odd_over25_open"))
        if result not in (0.0, 1.0) or odd is None or odd <= 1.0:
            continue
        valid.append(row)

    profits = [flat_profit(int(r["result_over25"]), float(r["odd_over25_open"])) for r in valid]
    wins = sum(int(r["result_over25"]) for r in valid)

    def vals(key: str) -> list[float]:
        out: list[float] = []
        for row in valid:
            value = _f(row.get(key))
            if value is not None:
                out.append(value)
        return out

    sample = len(valid)
    total_profit = sum(profits)
    return SegmentMetrics(
        sample=sample,
        wins=wins,
        profit=total_profit,
        roi=(total_profit / sample) if sample else None,
        hit_rate=(wins / sample) if sample else None,
        avg_odd=mean(vals("odd_over25_open")),
        avg_clv_probability=mean(vals("clv_probability")),
        avg_sports_probability=mean(vals("sports_probability")),
        avg_expected_value=mean(vals("expected_value")),
        avg_sports_edge_market=mean(vals("sports_edge_market")),
    )


def _bin(value: float, cuts: tuple[tuple[float, str], ...], final_label: str) -> str:
    for upper, label in cuts:
        if value < upper:
            return label
    return final_label


def clv_bin(v: float) -> str:
    return _bin(v, ((0.40, "<0.40"), (0.45, "0.40-0.45"), (0.50, "0.45-0.50"), (0.55, "0.50-0.55"), (0.60, "0.55-0.60"), (0.65, "0.60-0.65")), ">=0.65")


def sports_bin(v: float) -> str:
    return _bin(v, ((0.50, "<0.50"), (0.55, "0.50-0.55"), (0.60, "0.55-0.60"), (0.65, "0.60-0.65")), ">=0.65")


def ev_bin(v: float) -> str:
    return _bin(v, ((0.0, "<0"), (0.03, "0-3%"), (0.05, "3-5%"), (0.10, "5-10%")), ">=10%")


def edge_bin(v: float) -> str:
    return _bin(v, ((0.0, "<0"), (0.02, "0-2%"), (0.03, "2-3%"), (0.05, "3-5%")), ">=5%")


def odds_bin(v: float) -> str:
    return _bin(v, ((1.50, "<1.50"), (1.75, "1.50-1.75"), (2.00, "1.75-2.00"), (2.50, "2.00-2.50")), ">=2.50")


def agreement_bin(row: dict[str, Any]) -> str:
    clv = float(row["clv_probability"])
    sports = float(row["sports_probability"])
    clv_high = clv >= 0.55
    sports_high = sports >= 0.55
    if clv_high and sports_high:
        return "CLV_HIGH+SPORTS_HIGH"
    if clv_high and not sports_high:
        return "CLV_HIGH+SPORTS_LOW"
    if not clv_high and sports_high:
        return "CLV_LOW+SPORTS_HIGH"
    return "CLV_LOW+SPORTS_LOW"


def build_learning_rows(
    pre_match_rows: list[dict[str, Any]],
    sports_probability_fn: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    """Analyze every eligible game independently of betting-policy gates.

    Only pre-match fields are passed to the sports model. The outcome is read
    afterwards solely for supervision/measurement. CLV <55% never stops the
    learning pipeline.
    """
    output: list[dict[str, Any]] = []
    for raw in sorted(pre_match_rows, key=lambda r: (str(r.get("fixture_date") or ""), int(r.get("id") or 0))):
        pre = {
            key: raw.get(key)
            for key in (
                "id", "prediction_timestamp", "fixture_date", "fixture_time",
                "league", "home_team", "away_team", "odd_over25_open",
                "fair_probability_open", "clv_probability", "signal_threshold",
                "signal", "features_json",
            )
        }
        sports_probability = float(sports_probability_fn(pre))
        odd = float(pre["odd_over25_open"])
        market_probability = float(pre["fair_probability_open"])
        expected_value = sports_probability * odd - 1.0
        sports_edge_market = sports_probability - market_probability

        # Outcome reveal happens only after all pre-match calculations above.
        result = int(raw["result_over25"])
        row = {
            "paper_trade_id": int(raw["id"]),
            "fixture_date": raw.get("fixture_date"),
            "league": str(raw.get("league") or "").strip(),
            "home_team": raw.get("home_team"),
            "away_team": raw.get("away_team"),
            "odd_over25_open": odd,
            "market_probability": market_probability,
            "clv_probability": float(pre["clv_probability"]),
            "sports_probability": sports_probability,
            "expected_value": expected_value,
            "sports_edge_market": sports_edge_market,
            "original_signal": int(pre.get("signal") or 0),
            "result_over25": result,
            "counterfactual_profit": flat_profit(result, odd),
        }
        row["clv_bin"] = clv_bin(row["clv_probability"])
        row["sports_bin"] = sports_bin(row["sports_probability"])
        row["ev_bin"] = ev_bin(row["expected_value"])
        row["edge_bin"] = edge_bin(row["sports_edge_market"])
        row["odds_bin"] = odds_bin(row["odd_over25_open"])
        row["agreement_bin"] = agreement_bin(row)
        output.append(row)
    return output


def segment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []

    def add(kind: str, label: str, subset: list[dict[str, Any]]):
        m = metrics(subset)
        segments.append({
            "segment_type": kind,
            "segment_value": label,
            "sample": m.sample,
            "wins": m.wins,
            "profit": m.profit,
            "roi": m.roi,
            "hit_rate": m.hit_rate,
            "avg_odd": m.avg_odd,
            "avg_clv_probability": m.avg_clv_probability,
            "avg_sports_probability": m.avg_sports_probability,
            "avg_expected_value": m.avg_expected_value,
            "avg_sports_edge_market": m.avg_sports_edge_market,
        })

    add("ALL", "ALL", rows)
    for key, kind in (
        ("clv_bin", "CLV_BIN"),
        ("sports_bin", "SPORTS_BIN"),
        ("ev_bin", "EV_BIN"),
        ("edge_bin", "EDGE_BIN"),
        ("odds_bin", "ODDS_BIN"),
        ("agreement_bin", "AGREEMENT"),
        ("league", "LEAGUE"),
    ):
        for value in sorted({str(r.get(key) or "UNKNOWN") for r in rows}):
            add(kind, value, [r for r in rows if str(r.get(key) or "UNKNOWN") == value])

    add("ORIGINAL_SIGNAL", "SIGNAL_1", [r for r in rows if int(r.get("original_signal") or 0) == 1])
    add("ORIGINAL_SIGNAL", "SIGNAL_0", [r for r in rows if int(r.get("original_signal") or 0) == 0])
    return segments
