from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import goal_model_v2_multileague_research as v2

OUT = Path("reports")
TEAM_MARKETS = [
    "HomeO0.5", "HomeO1.5", "HomeO2.5",
    "AwayO0.5", "AwayO1.5", "AwayO2.5",
]
TOTAL_MARKETS = ["BTTS", "O1.5", "O2.5", "O3.5"]
ALL_MARKETS = TOTAL_MARKETS + TEAM_MARKETS

BASE_V2_CONFIG = {
    "general_window": 10,
    "venue_window": 10,
    "general_weight": 0.55,
    "league_shrinkage": 0.20,
    "elo_strength": 0.30,
}

HOME_SCALES = [round(x, 2) for x in np.arange(0.94, 1.101, 0.01)]
AWAY_SCALES = [round(x, 2) for x in np.arange(0.90, 1.071, 0.01)]
SHORTLIST_N = 20


def load_snapshots() -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    failures: list[str] = []
    for season in v2.SEASONS:
        for league, league_name in v2.LEAGUES.items():
            try:
                df = v2.load_csv_robust(season, league)
                part = v2.prepare_snapshots(df, season, league, league_name)
                rows.extend(part)
                print(f"SNAPSHOT {season} {league}: raw={len(df)} usable={len(part)}")
            except Exception as exc:
                msg = f"{season}/{league}: {type(exc).__name__}: {exc}"
                failures.append(msg)
                print("WARNING", msg)
    if not rows:
        raise RuntimeError("No usable snapshots")
    return rows, failures


def scaled_probabilities(data: list[dict], home_scale: float, away_scale: float):
    lh, la = v2.lambdas_for_config(data, BASE_V2_CONFIG)
    lh = np.clip(lh * float(home_scale), 0.15, 4.50)
    la = np.clip(la * float(away_scale), 0.15, 4.50)
    return lh, la, v2.probabilities(lh, la)


def team_objective(data: list[dict], candidate_probs: dict, baseline_probs: dict) -> float:
    ratios = []
    for market in TEAM_MARKETS:
        y = v2.y_array(data, market)
        ll_base = v2.logloss(y, baseline_probs[market])
        ll_new = v2.logloss(y, candidate_probs[market])
        ratios.append(ll_new / ll_base)
    return float(np.mean(ratios))


def guard_objective(data: list[dict], candidate_probs: dict, baseline_probs: dict) -> float:
    team = team_objective(data, candidate_probs, baseline_probs)
    broad_ratios = []
    for market in TOTAL_MARKETS:
        y = v2.y_array(data, market)
        broad_ratios.append(
            v2.logloss(y, candidate_probs[market]) /
            v2.logloss(y, baseline_probs[market])
        )
    broad = float(np.mean(broad_ratios))
    return 0.80 * team + 0.20 * broad


def compare_rows(scope: str, data: list[dict], p2: dict, p3: dict) -> list[dict]:
    rows = []
    for market in ALL_MARKETS:
        y = v2.y_array(data, market)
        ll2, ll3 = v2.logloss(y, p2[market]), v2.logloss(y, p3[market])
        br2, br3 = v2.brier(y, p2[market]), v2.brier(y, p3[market])
        rows.append({
            "scope": scope,
            "market": market,
            "v2_logloss": ll2,
            "v3_logloss": ll3,
            "logloss_change_pct": (ll3 / ll2 - 1.0) * 100.0,
            "v2_brier": br2,
            "v3_brier": br3,
            "brier_change_pct": (br3 / br2 - 1.0) * 100.0,
            "v2_ece": v2.ece(y, p2[market]),
            "v3_ece": v2.ece(y, p3[market]),
            "v2_pred": float(np.mean(p2[market])),
            "v3_pred": float(np.mean(p3[market])),
            "actual": float(np.mean(y)),
        })
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    snapshots, failures = load_snapshots()
    dev = [r for r in snapshots if r["season"] == "2122"]
    val = [r for r in snapshots if r["season"] == "2223"]
    hold = [r for r in snapshots if r["season"] == "2324"]

    scope_data = {"DEV": dev, "VALIDATION": val, "HOLDOUT": hold}

    base_probs = {}
    for scope, data in scope_data.items():
        _, _, pp = scaled_probabilities(data, 1.0, 1.0)
        base_probs[scope] = pp

    grid = []
    for hs in HOME_SCALES:
        for aws in AWAY_SCALES:
            _, _, pp = scaled_probabilities(dev, hs, aws)
            grid.append({
                "home_scale": hs,
                "away_scale": aws,
                "dev_team_relative_logloss": team_objective(dev, pp, base_probs["DEV"]),
                "dev_guard_objective": guard_objective(dev, pp, base_probs["DEV"]),
            })

    grid.sort(key=lambda r: (r["dev_guard_objective"], r["dev_team_relative_logloss"]))
    shortlist = grid[:SHORTLIST_N]

    for row in shortlist:
        _, _, pp = scaled_probabilities(val, row["home_scale"], row["away_scale"])
        row["validation_team_relative_logloss"] = team_objective(
            val, pp, base_probs["VALIDATION"]
        )
        row["validation_guard_objective"] = guard_objective(
            val, pp, base_probs["VALIDATION"]
        )

    shortlist.sort(
        key=lambda r: (
            r["validation_guard_objective"],
            r["validation_team_relative_logloss"],
            r["dev_guard_objective"],
        )
    )
    best = shortlist[0]

    compare = []
    final_probs = {}
    for scope, data in scope_data.items():
        _, _, p3 = scaled_probabilities(
            data, best["home_scale"], best["away_scale"]
        )
        final_probs[scope] = p3
        compare.extend(compare_rows(scope, data, base_probs[scope], p3))

    hold_team_ratios = []
    hold_improved = 0
    for market in TEAM_MARKETS:
        y = v2.y_array(hold, market)
        ll2 = v2.logloss(y, base_probs["HOLDOUT"][market])
        ll3 = v2.logloss(y, final_probs["HOLDOUT"][market])
        hold_team_ratios.append(ll3 / ll2)
        hold_improved += int(ll3 < ll2)

    holdout_team_relative = float(np.mean(hold_team_ratios))
    pass_rule = bool(
        holdout_team_relative < 1.0
        and hold_improved >= 4
        and best["validation_team_relative_logloss"] < 1.0
    )

    pd.DataFrame(grid).to_csv(OUT / "goal_model_v3_grid_all.csv", index=False)
    pd.DataFrame(shortlist).to_csv(OUT / "goal_model_v3_grid_shortlist.csv", index=False)
    pd.DataFrame(compare).to_csv(OUT / "goal_model_v3_vs_v2.csv", index=False)

    summary = {
        "n": len(snapshots),
        "dev": len(dev),
        "validation": len(val),
        "holdout": len(hold),
        "selection_rule": (
            "Tune home/away lambda scales on 2021/22 with 80% team-total "
            "logloss objective + 20% broad-goal guard; shortlist top 20; "
            "select on 2022/23; evaluate 2023/24 only after freezing."
        ),
        "base_v2_config": BASE_V2_CONFIG,
        "v3_calibration": {
            "home_scale": float(best["home_scale"]),
            "away_scale": float(best["away_scale"]),
        },
        "dev_team_relative_logloss": float(best["dev_team_relative_logloss"]),
        "validation_team_relative_logloss": float(best["validation_team_relative_logloss"]),
        "holdout_team_relative_logloss": holdout_team_relative,
        "holdout_team_markets_improved_logloss": int(hold_improved),
        "holdout_team_markets_total": len(TEAM_MARKETS),
        "promotion_gate_passed": pass_rule,
        "promotion_gate": (
            "Validation team relative logloss < 1.0, holdout team relative "
            "logloss < 1.0, and at least 4/6 team-total markets improve logloss."
        ),
        "note": (
            "This test evaluates probability quality only. Historical team-total "
            "odds are not present in the current public dataset, so profitability "
            "requires live shadow odds or a separate historical odds source."
        ),
        "failures": failures,
    }

    (OUT / "goal_model_v3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== GOAL MODEL V3 TEAM TOTAL CALIBRATION ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nReports written to", OUT.resolve())


if __name__ == "__main__":
    main()
