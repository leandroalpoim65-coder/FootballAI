from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import goal_model_v2_multileague_research as v2
import goal_model_v4_structural_research as v4


OUT = Path("reports")

RESEARCH_SEASONS = ["2122", "2223", "2324"]
VALIDATION_SEASON = "2425"
CONFIRMATION_SEASON = "2526"  # already inspected: confirmation only
STRESS_SEASON = "2021"        # independent backward stress test
MARKET = "HomeO2.5"

# Frozen V5 HOME configuration selected before this V6 research.
V5_HOME_CONFIG = {
    "window": 20,
    "decay": 1.0,
    "prior": 8.0,
    "venue_weight": 0.5,
    "blend": 0.85,
    "extra_elo": 0.1,
}

# Small, conservative specialist calibration grid. Selection uses ONLY
# 2021/22-2023/24. Later seasons are evaluation only.
V5_WEIGHTS = [0.00, 0.25, 0.50, 0.75, 1.00]
SLOPES = [0.90, 0.95, 1.00, 1.05, 1.10]
INTERCEPTS = [-0.10, -0.05, 0.00, 0.05, 0.10]
EPS = 1e-9
BOOTSTRAPS = 3000
RNG_SEED = 20260914


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def specialist_probability(
    p2: np.ndarray,
    p5: np.ndarray,
    *,
    v5_weight: float,
    slope: float,
    intercept: float,
) -> np.ndarray:
    # Blend in log-odds space, then apply a tiny calibration layer.
    z = (1.0 - float(v5_weight)) * logit(p2) + float(v5_weight) * logit(p5)
    return clip_prob(sigmoid(float(intercept) + float(slope) * z))


def load_seasons(seasons: list[str]) -> tuple[dict[str, list[dict]], list[str]]:
    by_season, failures = v4.build_data(seasons)
    missing = [s for s in seasons if not by_season.get(s)]
    if missing:
        raise RuntimeError(f"Missing seasons {missing}. Failures: {failures}")
    return by_season, failures


def base_and_v5(data: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lh2, la2 = v2.lambdas_for_config(data, v4.V2_CONFIG)
    p2 = v2.probabilities(lh2, la2)[MARKET]

    lh5, _ = v4.v4_lambdas(data, V5_HOME_CONFIG)
    # Home team total depends only on lambda_home. Keep V2 away lambda merely
    # to construct the common probability dictionary consistently.
    p5 = v2.probabilities(lh5, la2)[MARKET]
    y = v2.y_array(data, MARKET)
    return y, p2, p5


def logloss_vector(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = clip_prob(p)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def bootstrap_delta(
    y: np.ndarray,
    p_new: np.ndarray,
    p_old: np.ndarray,
    *,
    seed: int,
) -> dict:
    diff = logloss_vector(y, p_new) - logloss_vector(y, p_old)
    n = len(diff)
    if n == 0:
        return {
            "mean_logloss_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "probability_new_better": None,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    samples = diff[indices].mean(axis=1)
    return {
        "mean_logloss_delta": float(diff.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_new_better": float(np.mean(samples < 0.0)),
    }


def metric_row(scope: str, y: np.ndarray, p2: np.ndarray, p5: np.ndarray, p6: np.ndarray) -> dict:
    ll2 = v2.logloss(y, p2)
    ll5 = v2.logloss(y, p5)
    ll6 = v2.logloss(y, p6)
    br2 = v2.brier(y, p2)
    br5 = v2.brier(y, p5)
    br6 = v2.brier(y, p6)
    return {
        "scope": scope,
        "n": len(y),
        "actual": float(np.mean(y)),
        "v2_pred": float(np.mean(p2)),
        "v5_pred": float(np.mean(p5)),
        "v6_pred": float(np.mean(p6)),
        "v2_logloss": ll2,
        "v5_logloss": ll5,
        "v6_logloss": ll6,
        "v6_vs_v2_logloss_pct": (ll6 / ll2 - 1.0) * 100.0,
        "v6_vs_v5_logloss_pct": (ll6 / ll5 - 1.0) * 100.0,
        "v2_brier": br2,
        "v5_brier": br5,
        "v6_brier": br6,
        "v6_vs_v2_brier_pct": (br6 / br2 - 1.0) * 100.0,
        "v6_vs_v5_brier_pct": (br6 / br5 - 1.0) * 100.0,
        "v2_ece": v2.ece(y, p2),
        "v5_ece": v2.ece(y, p5),
        "v6_ece": v2.ece(y, p6),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Parameter selection is restricted to the original three research seasons.
    research, failures = load_seasons(RESEARCH_SEASONS)
    cache = {season: base_and_v5(research[season]) for season in RESEARCH_SEASONS}

    grid_rows = []
    total = len(V5_WEIGHTS) * len(SLOPES) * len(INTERCEPTS)
    checked = 0

    for v5_weight in V5_WEIGHTS:
        for slope in SLOPES:
            for intercept in INTERCEPTS:
                rel_best = []
                rel_v2 = []
                rel_v5 = []
                season_changes = {}

                for season in RESEARCH_SEASONS:
                    y, p2, p5 = cache[season]
                    p6 = specialist_probability(
                        p2,
                        p5,
                        v5_weight=v5_weight,
                        slope=slope,
                        intercept=intercept,
                    )
                    ll2 = v2.logloss(y, p2)
                    ll5 = v2.logloss(y, p5)
                    ll6 = v2.logloss(y, p6)
                    rel_best.append(ll6 / min(ll2, ll5))
                    rel_v2.append(ll6 / ll2)
                    rel_v5.append(ll6 / ll5)
                    season_changes[season] = (ll6 / min(ll2, ll5) - 1.0) * 100.0

                mean_best = float(np.mean(rel_best))
                std_best = float(np.std(rel_best))
                worst_best = float(max(rel_best))
                # Penalise instability and any season that loses to the better baseline.
                stability = mean_best + 0.50 * std_best + 0.50 * max(0.0, worst_best - 1.0)

                grid_rows.append({
                    "v5_weight": v5_weight,
                    "slope": slope,
                    "intercept": intercept,
                    "research_relative_best": mean_best,
                    "research_relative_v2": float(np.mean(rel_v2)),
                    "research_relative_v5": float(np.mean(rel_v5)),
                    "research_std_best": std_best,
                    "research_worst_relative_best": worst_best,
                    "research_stability_score": stability,
                    **{
                        f"{season}_vs_best_logloss_pct": float(season_changes[season])
                        for season in RESEARCH_SEASONS
                    },
                })
                checked += 1
                if checked % 25 == 0 or checked == total:
                    print(f"GRID {checked}/{total}")

    grid_rows.sort(key=lambda r: (r["research_stability_score"], r["research_relative_best"]))
    best = grid_rows[0]
    calibration = {
        "v5_weight": float(best["v5_weight"]),
        "slope": float(best["slope"]),
        "intercept": float(best["intercept"]),
    }

    print("\nV6 HOME O2.5 CALIBRATION FROZEN USING RESEARCH ONLY:")
    print(calibration)

    # Only now load evaluation periods. None is used to alter the frozen parameters.
    eval_seasons = [VALIDATION_SEASON, CONFIRMATION_SEASON, STRESS_SEASON]
    evaluation, eval_failures = load_seasons(eval_seasons)
    failures.extend(eval_failures)

    scopes: list[tuple[str, list[dict]]] = []
    for season in RESEARCH_SEASONS:
        scopes.append((f"RESEARCH_{season}", research[season]))
    scopes.extend([
        ("VALIDATION_2425", evaluation[VALIDATION_SEASON]),
        ("CONFIRMATION_2526", evaluation[CONFIRMATION_SEASON]),
        ("STRESS_2021", evaluation[STRESS_SEASON]),
    ])

    metrics = []
    bootstrap = {}
    for idx, (scope, data) in enumerate(scopes):
        y, p2, p5 = base_and_v5(data)
        p6 = specialist_probability(p2, p5, **calibration)
        metrics.append(metric_row(scope, y, p2, p5, p6))

        if not scope.startswith("RESEARCH_"):
            bootstrap[scope] = {
                "v6_vs_v2": bootstrap_delta(y, p6, p2, seed=RNG_SEED + idx * 17 + 1),
                "v6_vs_v5": bootstrap_delta(y, p6, p5, seed=RNG_SEED + idx * 17 + 2),
            }

    research_metrics = [r for r in metrics if r["scope"].startswith("RESEARCH_")]
    eval_metrics = [r for r in metrics if not r["scope"].startswith("RESEARCH_")]

    research_beats_best = sum(
        r["v6_logloss"] < min(r["v2_logloss"], r["v5_logloss"])
        for r in research_metrics
    )
    eval_beats_v2 = sum(r["v6_logloss"] < r["v2_logloss"] for r in eval_metrics)
    eval_beats_v5 = sum(r["v6_logloss"] < r["v5_logloss"] for r in eval_metrics)
    eval_beats_both = sum(
        r["v6_logloss"] < min(r["v2_logloss"], r["v5_logloss"])
        for r in eval_metrics
    )

    historical_gate = bool(
        float(best["research_relative_best"]) < 1.0
        and research_beats_best >= 2
        and eval_beats_v2 >= 2
        and eval_beats_v5 >= 2
    )

    pd.DataFrame(grid_rows).to_csv(
        OUT / "goal_model_v6_home_o25_grid.csv", index=False
    )
    pd.DataFrame(metrics).to_csv(
        OUT / "goal_model_v6_home_o25_metrics.csv", index=False
    )

    summary = {
        "market": MARKET,
        "selection_rule": (
            "Choose calibration only on 2021/22-2023/24 by mean relative logloss "
            "against the better of frozen V2/V5, with stability and worst-season penalties. "
            "2024/25, 2025/26 and 2020/21 are evaluation only and never change parameters."
        ),
        "v5_home_config": V5_HOME_CONFIG,
        "v6_calibration": calibration,
        "grid_size": total,
        "research_relative_best": float(best["research_relative_best"]),
        "research_relative_v2": float(best["research_relative_v2"]),
        "research_relative_v5": float(best["research_relative_v5"]),
        "research_beats_best_seasons": int(research_beats_best),
        "research_seasons_total": len(RESEARCH_SEASONS),
        "evaluation_beats_v2": int(eval_beats_v2),
        "evaluation_beats_v5": int(eval_beats_v5),
        "evaluation_beats_both": int(eval_beats_both),
        "evaluation_scopes_total": len(eval_metrics),
        "historical_gate_passed": historical_gate,
        "production_promotion_allowed": False,
        "production_note": (
            "Historical specialization alone is insufficient for promotion. Freeze V6 if it "
            "shows stable support, then require prospective shadow evidence using real Home Team "
            "Over/Under 2.5 odds, EV, ROI and ideally CLV."
        ),
        "metrics": metrics,
        "bootstrap_logloss_delta": bootstrap,
        "failures": failures,
    }

    (OUT / "goal_model_v6_home_o25_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== V6 HOME O2.5 SPECIALIST SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nReports written to", OUT.resolve())


if __name__ == "__main__":
    main()
