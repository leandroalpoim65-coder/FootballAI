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
CONFIRMATION_SEASON = "2526"  # already inspected in prior V4 research: confirmation, not pristine holdout
STRESS_SEASON = "2021"        # independent backward stress test, not used for tuning
LIVE_SEASON = "2627"          # optional prospective checkpoint; incomplete season is diagnostic only

HOME_MARKETS = ["HomeO0.5", "HomeO1.5", "HomeO2.5"]
AWAY_MARKETS = ["AwayO0.5", "AwayO1.5", "AwayO2.5"]
TEAM_MARKETS = HOME_MARKETS + AWAY_MARKETS
BROAD_MARKETS = ["BTTS", "O1.5", "O2.5", "O3.5"]
MARKETS = BROAD_MARKETS + TEAM_MARKETS

SHORTLIST_SIDE = 15


def config_grid() -> list[dict]:
    rows = []
    for window in v4.WINDOWS:
        for decay in v4.DECAYS:
            for prior in v4.PRIORS:
                for venue_weight in v4.VENUE_WEIGHTS:
                    for blend in v4.BLENDS:
                        for extra_elo in v4.EXTRA_ELO:
                            rows.append({
                                "window": window,
                                "decay": decay,
                                "prior": prior,
                                "venue_weight": venue_weight,
                                "blend": blend,
                                "extra_elo": extra_elo,
                            })
    return rows


def cfg_key(cfg: dict) -> tuple:
    return (
        int(cfg["window"]),
        float(cfg["decay"]),
        float(cfg["prior"]),
        float(cfg["venue_weight"]),
        float(cfg["blend"]),
        float(cfg["extra_elo"]),
    )


def mean_relative_logloss(data: list[dict], probs: dict, base_probs: dict, markets: list[str]) -> float:
    vals = []
    for market in markets:
        y = v2.y_array(data, market)
        vals.append(v2.logloss(y, probs[market]) / v2.logloss(y, base_probs[market]))
    return float(np.mean(vals))


def side_score(
    data: list[dict],
    side_lh: np.ndarray,
    side_la: np.ndarray,
    base_lh: np.ndarray,
    base_la: np.ndarray,
    side: str,
) -> tuple[float, float, float]:
    if side == "home":
        lh, la = side_lh, base_la
        side_markets = HOME_MARKETS
    elif side == "away":
        lh, la = base_lh, side_la
        side_markets = AWAY_MARKETS
    else:
        raise ValueError(side)

    probs = v2.probabilities(lh, la)
    base_probs = v2.probabilities(base_lh, base_la)
    team = mean_relative_logloss(data, probs, base_probs, side_markets)
    broad = mean_relative_logloss(data, probs, base_probs, BROAD_MARKETS)
    # Focus on the side-specific scoring markets, but keep a broad-goal guard.
    score = 0.85 * team + 0.15 * broad
    return score, team, broad


def combined_score(
    data: list[dict],
    lh: np.ndarray,
    la: np.ndarray,
    base_lh: np.ndarray,
    base_la: np.ndarray,
) -> tuple[float, float, float, float, float]:
    probs = v2.probabilities(lh, la)
    base_probs = v2.probabilities(base_lh, base_la)
    home = mean_relative_logloss(data, probs, base_probs, HOME_MARKETS)
    away = mean_relative_logloss(data, probs, base_probs, AWAY_MARKETS)
    team = 0.5 * (home + away)
    broad = mean_relative_logloss(data, probs, base_probs, BROAD_MARKETS)
    combined = 0.75 * team + 0.25 * broad
    return combined, team, broad, home, away


def compare_rows(scope: str, data: list[dict], base_probs: dict, p5: dict) -> list[dict]:
    rows = []
    for market in MARKETS:
        y = v2.y_array(data, market)
        ll2 = v2.logloss(y, base_probs[market])
        ll5 = v2.logloss(y, p5[market])
        br2 = v2.brier(y, base_probs[market])
        br5 = v2.brier(y, p5[market])
        rows.append({
            "scope": scope,
            "market": market,
            "n": len(y),
            "v2_logloss": ll2,
            "v5_logloss": ll5,
            "logloss_change_pct": (ll5 / ll2 - 1.0) * 100.0,
            "v2_brier": br2,
            "v5_brier": br5,
            "brier_change_pct": (br5 / br2 - 1.0) * 100.0,
            "v2_ece": v2.ece(y, base_probs[market]),
            "v5_ece": v2.ece(y, p5[market]),
            "v2_pred": float(np.mean(base_probs[market])),
            "v5_pred": float(np.mean(p5[market])),
            "actual": float(np.mean(y)),
        })
    return rows


def load_scope(seasons: list[str], strict: bool = True) -> tuple[dict[str, list[dict]], list[str]]:
    by_season, failures = v4.build_data(seasons)
    if strict:
        missing = [s for s in seasons if not by_season.get(s)]
        if missing:
            raise RuntimeError(f"Missing season groups {missing}. Failures: {failures}")
    return by_season, failures


def baseline_lambdas(data: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    lh2, la2 = v2.lambdas_for_config(data, v4.V2_CONFIG)
    return lh2, la2, v2.probabilities(lh2, la2)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Selection uses only 2021/22-2024/25. 2025/26 is not used for parameter choice,
    # but it has already been inspected in V4, so it is labelled confirmation only.
    pre = RESEARCH_SEASONS + [VALIDATION_SEASON]
    by_season, failures = load_scope(pre, strict=True)
    research = [r for s in RESEARCH_SEASONS for r in by_season[s]]
    validation = by_season[VALIDATION_SEASON]

    base_research = {}
    for season in RESEARCH_SEASONS:
        base_research[season] = baseline_lambdas(by_season[season])
    base_val = baseline_lambdas(validation)

    grid = config_grid()
    grid_rows = []
    total = len(grid)

    # One structural calculation per config/season feeds BOTH the home and away searches.
    for idx, cfg in enumerate(grid, start=1):
        home_season = []
        away_season = []
        home_team = []
        away_team = []
        home_broad = []
        away_broad = []

        for season in RESEARCH_SEASONS:
            data = by_season[season]
            base_lh, base_la, _ = base_research[season]
            lh4, la4 = v4.v4_lambdas(data, cfg)

            hs, ht, hb = side_score(data, lh4, la4, base_lh, base_la, "home")
            as_, at, ab = side_score(data, lh4, la4, base_lh, base_la, "away")
            home_season.append(hs)
            away_season.append(as_)
            home_team.append(ht)
            away_team.append(at)
            home_broad.append(hb)
            away_broad.append(ab)

        home_mean = float(np.mean(home_season))
        away_mean = float(np.mean(away_season))
        home_std = float(np.std(home_season))
        away_std = float(np.std(away_season))

        grid_rows.append({
            **cfg,
            "home_research_score": home_mean,
            "home_research_team": float(np.mean(home_team)),
            "home_research_broad": float(np.mean(home_broad)),
            "home_research_std": home_std,
            "home_stability_score": home_mean + 0.25 * home_std,
            "away_research_score": away_mean,
            "away_research_team": float(np.mean(away_team)),
            "away_research_broad": float(np.mean(away_broad)),
            "away_research_std": away_std,
            "away_stability_score": away_mean + 0.25 * away_std,
        })

        if idx % 50 == 0 or idx == total:
            print(f"GRID {idx}/{total}")

    home_short = sorted(
        grid_rows,
        key=lambda r: (r["home_stability_score"], r["home_research_score"]),
    )[:SHORTLIST_SIDE]
    away_short = sorted(
        grid_rows,
        key=lambda r: (r["away_stability_score"], r["away_research_score"]),
    )[:SHORTLIST_SIDE]

    # Cache validation lambdas once for every candidate appearing in either shortlist.
    unique_cfgs: dict[tuple, dict] = {}
    for row in home_short + away_short:
        cfg = {k: row[k] for k in ["window", "decay", "prior", "venue_weight", "blend", "extra_elo"]}
        unique_cfgs[cfg_key(cfg)] = cfg

    val_lambdas = {}
    for key, cfg in unique_cfgs.items():
        val_lambdas[key] = v4.v4_lambdas(validation, cfg)

    base_val_lh, base_val_la, base_val_probs = base_val
    pair_rows = []
    for hrow in home_short:
        hcfg = {k: hrow[k] for k in ["window", "decay", "prior", "venue_weight", "blend", "extra_elo"]}
        hlh, _ = val_lambdas[cfg_key(hcfg)]
        for arow in away_short:
            acfg = {k: arow[k] for k in ["window", "decay", "prior", "venue_weight", "blend", "extra_elo"]}
            _, ala = val_lambdas[cfg_key(acfg)]
            combined, team, broad, home, away = combined_score(
                validation, hlh, ala, base_val_lh, base_val_la
            )
            pair_rows.append({
                "validation_relative_logloss": combined,
                "validation_team_relative_logloss": team,
                "validation_broad_relative_logloss": broad,
                "validation_home_relative_logloss": home,
                "validation_away_relative_logloss": away,
                **{f"home_{k}": v for k, v in hcfg.items()},
                **{f"away_{k}": v for k, v in acfg.items()},
            })

    pair_rows.sort(key=lambda r: (
        r["validation_relative_logloss"],
        r["validation_team_relative_logloss"],
    ))
    best = pair_rows[0]
    home_cfg = {k: best[f"home_{k}"] for k in ["window", "decay", "prior", "venue_weight", "blend", "extra_elo"]}
    away_cfg = {k: best[f"away_{k}"] for k in ["window", "decay", "prior", "venue_weight", "blend", "extra_elo"]}

    print("\nV5 CONFIG FROZEN BEFORE CONFIRMATION/STRESS:")
    print("HOME:", home_cfg)
    print("AWAY:", away_cfg)

    # Only after freezing, load confirmation and independent stress season.
    post, post_failures = load_scope([CONFIRMATION_SEASON, STRESS_SEASON], strict=True)
    failures.extend(post_failures)
    confirmation = post[CONFIRMATION_SEASON]
    stress = post[STRESS_SEASON]

    # Optional current-season prospective checkpoint. It may be unavailable or still tiny.
    live_rows = []
    live_failures = []
    try:
        live_data, live_failures = load_scope([LIVE_SEASON], strict=False)
        live_rows = live_data.get(LIVE_SEASON, [])
    except Exception as exc:
        live_failures = [f"{LIVE_SEASON}: {type(exc).__name__}: {exc}"]
    failures.extend(live_failures)

    scopes = {
        "VALIDATION_2425": validation,
        "CONFIRMATION_2526": confirmation,
        "STRESS_2021": stress,
    }
    if live_rows:
        scopes["LIVE_2627"] = live_rows

    comparisons = []
    metric_summary = {}

    for scope, data in scopes.items():
        base_lh, base_la, base_probs = baseline_lambdas(data)
        hlh, _ = v4.v4_lambdas(data, home_cfg)
        _, ala = v4.v4_lambdas(data, away_cfg)
        p5 = v2.probabilities(hlh, ala)
        combined, team, broad, home, away = combined_score(data, hlh, ala, base_lh, base_la)
        metric_summary[scope] = {
            "n": len(data),
            "relative_logloss": combined,
            "team_relative_logloss": team,
            "broad_relative_logloss": broad,
            "home_relative_logloss": home,
            "away_relative_logloss": away,
        }
        comparisons.extend(compare_rows(scope, data, base_probs, p5))

    confirm_cmp = [r for r in comparisons if r["scope"] == "CONFIRMATION_2526"]
    confirm_team = [r for r in confirm_cmp if r["market"] in TEAM_MARKETS]
    confirm_home = [r for r in confirm_cmp if r["market"] in HOME_MARKETS]
    confirm_away = [r for r in confirm_cmp if r["market"] in AWAY_MARKETS]
    confirm_broad = [r for r in confirm_cmp if r["market"] in BROAD_MARKETS]

    improved_team = sum(r["logloss_change_pct"] < 0 for r in confirm_team)
    improved_home = sum(r["logloss_change_pct"] < 0 for r in confirm_home)
    improved_away = sum(r["logloss_change_pct"] < 0 for r in confirm_away)
    worst_broad = max((r["logloss_change_pct"] for r in confirm_broad), default=0.0)

    validation_pass = (
        metric_summary["VALIDATION_2425"]["relative_logloss"] < 1.0
        and metric_summary["VALIDATION_2425"]["team_relative_logloss"] < 1.0
    )
    confirmation_support = (
        metric_summary["CONFIRMATION_2526"]["team_relative_logloss"] < 1.0
        and improved_team >= 4
        and worst_broad < 0.50
    )
    stress_support = metric_summary["STRESS_2021"]["team_relative_logloss"] < 1.0

    # This is deliberately NOT called a production-promotion gate because 2025/26
    # has already informed the research direction through V4 inspection.
    research_gate_passed = bool(validation_pass and confirmation_support and stress_support)

    pd.DataFrame(grid_rows).to_csv(OUT / "goal_model_v5_side_grid.csv", index=False)
    pd.DataFrame(home_short).to_csv(OUT / "goal_model_v5_home_shortlist.csv", index=False)
    pd.DataFrame(away_short).to_csv(OUT / "goal_model_v5_away_shortlist.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "goal_model_v5_validation_pairs.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "goal_model_v5_vs_v2.csv", index=False)

    summary = {
        "research_seasons": RESEARCH_SEASONS,
        "validation_season": VALIDATION_SEASON,
        "confirmation_season": CONFIRMATION_SEASON,
        "stress_season": STRESS_SEASON,
        "live_season": LIVE_SEASON if live_rows else None,
        "selection_rule": (
            "Rank the same 324 structural configurations independently for home and away on "
            "2021/22-2023/24 (85% side team-total logloss + 15% broad-goal guard), keep top 15 "
            "per side, select the best 15x15 home/away pair on 2024/25, then freeze. "
            "2025/26 is confirmation only because it was already inspected in V4; 2020/21 is an "
            "independent backward stress test; 2026/27 is optional prospective diagnostics only."
        ),
        "home_config": home_cfg,
        "away_config": away_cfg,
        "validation_pair": {
            k: float(best[k])
            for k in [
                "validation_relative_logloss",
                "validation_team_relative_logloss",
                "validation_broad_relative_logloss",
                "validation_home_relative_logloss",
                "validation_away_relative_logloss",
            ]
        },
        "scopes": metric_summary,
        "confirmation_team_markets_improved_logloss": int(improved_team),
        "confirmation_home_markets_improved_logloss": int(improved_home),
        "confirmation_away_markets_improved_logloss": int(improved_away),
        "confirmation_worst_broad_regression_pct": float(worst_broad),
        "research_gate_passed": research_gate_passed,
        "production_promotion_allowed": False,
        "production_note": (
            "Do not promote from this retrospective test alone. Freeze the winning V5 configuration "
            "and require prospective 2026/27 shadow evidence, ideally with real team-total odds/CLV."
        ),
        "failures": failures,
    }

    (OUT / "goal_model_v5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== GOAL MODEL V5 SPLIT HOME/AWAY SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nReports written to", OUT.resolve())


if __name__ == "__main__":
    main()
