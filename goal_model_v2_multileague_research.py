from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from goal_model_multileague_backtest_local import load_csv_robust
from ml.goal_model import project_match_from_lambdas

SEASONS = ["2122", "2223", "2324"]
LEAGUES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Liga Portugal",
}
OUT = Path("reports")
EPS = 1e-12
MIN_HISTORY = 5
MAX_HISTORY = 10
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 65.0
THRESHOLDS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
ALPHAS = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
MARKETS = [
    "BTTS", "O1.5", "O2.5", "O3.5",
    "HomeO0.5", "HomeO1.5", "HomeO2.5",
    "AwayO0.5", "AwayO1.5", "AwayO2.5",
]


def sf(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if math.isfinite(x) else np.nan


def no_vig2(a: float, b: float) -> tuple[float, float]:
    ia, ib = 1.0 / a, 1.0 / b
    total = ia + ib
    return ia / total, ib / total


def elo_expected(home_rating: float, away_rating: float) -> float:
    diff = (home_rating + ELO_HOME_ADVANTAGE) - away_rating
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def append_result(state, team: str, scored: int, conceded: int, venue: str) -> None:
    state[team]["all"].append((scored, conceded))
    state[team][venue].append((scored, conceded))
    state[team]["all"] = state[team]["all"][-MAX_HISTORY:]
    state[team][venue] = state[team][venue][-MAX_HISTORY:]


def prepare_snapshots(df: pd.DataFrame, season: str, league: str, league_name: str) -> list[dict]:
    state = defaultdict(lambda: {"all": [], "home": [], "away": []})
    ratings = defaultdict(lambda: 1500.0)
    prior: list[tuple[int, int]] = []
    records: list[dict] = []

    for date, day in df.groupby("Date", sort=True):
        league_home_avg = float(np.mean([x[0] for x in prior])) if prior else np.nan
        league_away_avg = float(np.mean([x[1] for x in prior])) if prior else np.nan

        for _, row in day.iterrows():
            home, away = str(row.HomeTeam), str(row.AwayTeam)
            if min(
                len(state[home]["all"]), len(state[away]["all"]),
                len(state[home]["home"]), len(state[away]["away"]),
            ) < MIN_HISTORY:
                continue

            hg, ag = int(row.FTHG), int(row.FTAG)
            total = hg + ag
            rec = {
                "season": season,
                "league": league,
                "league_name": league_name,
                "date": date,
                "home": home,
                "away": away,
                "hg": hg,
                "ag": ag,
                "league_home_avg": league_home_avg,
                "league_away_avg": league_away_avg,
                "elo_diff": float(ratings[home] - ratings[away] + ELO_HOME_ADVANTAGE),
                "h_all_scored": [x[0] for x in state[home]["all"]],
                "h_all_conceded": [x[1] for x in state[home]["all"]],
                "a_all_scored": [x[0] for x in state[away]["all"]],
                "a_all_conceded": [x[1] for x in state[away]["all"]],
                "h_home_scored": [x[0] for x in state[home]["home"]],
                "h_home_conceded": [x[1] for x in state[home]["home"]],
                "a_away_scored": [x[0] for x in state[away]["away"]],
                "a_away_conceded": [x[1] for x in state[away]["away"]],
                "y_BTTS": int(hg > 0 and ag > 0),
                "y_O1.5": int(total > 1.5),
                "y_O2.5": int(total > 2.5),
                "y_O3.5": int(total > 3.5),
                "y_HomeO0.5": int(hg > 0.5),
                "y_HomeO1.5": int(hg > 1.5),
                "y_HomeO2.5": int(hg > 2.5),
                "y_AwayO0.5": int(ag > 0.5),
                "y_AwayO1.5": int(ag > 1.5),
                "y_AwayO2.5": int(ag > 2.5),
            }
            oo, ou = sf(row.get("Avg>2.5")), sf(row.get("Avg<2.5"))
            if math.isfinite(oo) and math.isfinite(ou) and min(oo, ou) > 1.0:
                mo, mu = no_vig2(oo, ou)
                rec.update(odd_o25=oo, odd_u25=ou, market_o25=mo, market_u25=mu)
            else:
                rec.update(odd_o25=np.nan, odd_u25=np.nan, market_o25=np.nan, market_u25=np.nan)
            records.append(rec)

        # Same-date results enter only after every match on the date has been snapshotted.
        for _, row in day.iterrows():
            home, away = str(row.HomeTeam), str(row.AwayTeam)
            hg, ag = int(row.FTHG), int(row.FTAG)
            append_result(state, home, hg, ag, "home")
            append_result(state, away, ag, hg, "away")
            prior.append((hg, ag))

            expected = elo_expected(ratings[home], ratings[away])
            actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = ELO_K * (actual - expected)
            ratings[home] += delta
            ratings[away] -= delta

    return records


def mean_tail(values: list[float], window: int) -> float:
    vals = values[-window:]
    return float(sum(vals) / len(vals))


def lambdas_for_config(snapshots: list[dict], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    lh = np.empty(len(snapshots), dtype=float)
    la = np.empty(len(snapshots), dtype=float)
    gw = float(cfg["general_weight"])
    vw = 1.0 - gw
    shrink = float(cfg["league_shrinkage"])
    elo_strength = float(cfg["elo_strength"])
    general_window = int(cfg["general_window"])
    venue_window = int(cfg["venue_window"])

    for i, r in enumerate(snapshots):
        home_general = 0.5 * (
            mean_tail(r["h_all_scored"], general_window)
            + mean_tail(r["a_all_conceded"], general_window)
        )
        away_general = 0.5 * (
            mean_tail(r["a_all_scored"], general_window)
            + mean_tail(r["h_all_conceded"], general_window)
        )
        home_venue = 0.5 * (
            mean_tail(r["h_home_scored"], venue_window)
            + mean_tail(r["a_away_conceded"], venue_window)
        )
        away_venue = 0.5 * (
            mean_tail(r["a_away_scored"], venue_window)
            + mean_tail(r["h_home_conceded"], venue_window)
        )

        xh = gw * home_general + vw * home_venue
        xa = gw * away_general + vw * away_venue
        if math.isfinite(r["league_home_avg"]) and r["league_home_avg"] > 0:
            xh = (1.0 - shrink) * xh + shrink * float(r["league_home_avg"])
        if math.isfinite(r["league_away_avg"]) and r["league_away_avg"] > 0:
            xa = (1.0 - shrink) * xa + shrink * float(r["league_away_avg"])

        if elo_strength != 0.0:
            mult = math.exp(elo_strength * float(r["elo_diff"]) / 400.0)
            xh *= mult
            xa /= mult

        lh[i] = min(4.50, max(0.15, xh))
        la[i] = min(4.50, max(0.15, xa))
    return lh, la


def poisson_over(lam: np.ndarray, line: float) -> np.ndarray:
    p0 = np.exp(-lam)
    if line == 0.5:
        return 1.0 - p0
    p1 = p0 * lam
    if line == 1.5:
        return 1.0 - p0 - p1
    p2 = p1 * lam / 2.0
    if line == 2.5:
        return 1.0 - p0 - p1 - p2
    p3 = p2 * lam / 3.0
    if line == 3.5:
        return 1.0 - p0 - p1 - p2 - p3
    raise ValueError(line)


def probabilities(lh: np.ndarray, la: np.ndarray) -> dict[str, np.ndarray]:
    total = lh + la
    return {
        "BTTS": (1.0 - np.exp(-lh)) * (1.0 - np.exp(-la)),
        "O1.5": poisson_over(total, 1.5),
        "O2.5": poisson_over(total, 2.5),
        "O3.5": poisson_over(total, 3.5),
        "HomeO0.5": poisson_over(lh, 0.5),
        "HomeO1.5": poisson_over(lh, 1.5),
        "HomeO2.5": poisson_over(lh, 2.5),
        "AwayO0.5": poisson_over(la, 0.5),
        "AwayO1.5": poisson_over(la, 1.5),
        "AwayO2.5": poisson_over(la, 2.5),
    }


def y_array(snapshots: list[dict], market: str) -> np.ndarray:
    return np.asarray([r[f"y_{market}"] for r in snapshots], dtype=float)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ans = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            ans += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ans)


def metric_rows(scope: str, snapshots: list[dict], probs: dict[str, np.ndarray], version: str) -> list[dict]:
    rows = []
    for market in MARKETS:
        y = y_array(snapshots, market)
        p = probs[market]
        rows.append({
            "scope": scope,
            "version": version,
            "market": market,
            "n": len(y),
            "actual": float(y.mean()),
            "pred": float(p.mean()),
            "bias_pp": float((p.mean() - y.mean()) * 100.0),
            "acc": float(((p >= 0.5) == y).mean()),
            "brier": brier(y, p),
            "logloss": logloss(y, p),
            "ece": ece(y, p),
        })
    return rows


def relative_objective(snapshots: list[dict], probs: dict[str, np.ndarray], baseline_ll: dict[str, float]) -> float:
    ratios = []
    for market in MARKETS:
        ratios.append(logloss(y_array(snapshots, market), probs[market]) / baseline_ll[market])
    return float(np.mean(ratios))


def tune_alpha(snapshots: list[dict], model_p: np.ndarray) -> tuple[float, list[dict]]:
    y = y_array(snapshots, "O2.5")
    market = np.asarray([r["market_o25"] for r in snapshots], dtype=float)
    mask = np.isfinite(market)
    rows = []
    for alpha in ALPHAS:
        p = market[mask] + alpha * (model_p[mask] - market[mask])
        rows.append({"alpha": alpha, "logloss": logloss(y[mask], p), "brier": brier(y[mask], p)})
    best = min(rows, key=lambda r: (r["logloss"], r["brier"]))
    return float(best["alpha"]), rows


def o25_market_metrics(snapshots: list[dict], model_p: np.ndarray) -> dict:
    y = y_array(snapshots, "O2.5")
    market = np.asarray([r["market_o25"] for r in snapshots], dtype=float)
    mask = np.isfinite(market)
    return {
        "n": int(mask.sum()),
        "model_brier": brier(y[mask], model_p[mask]),
        "market_brier": brier(y[mask], market[mask]),
        "model_logloss": logloss(y[mask], model_p[mask]),
        "market_logloss": logloss(y[mask], market[mask]),
        "model_ece": ece(y[mask], model_p[mask]),
        "market_ece": ece(y[mask], market[mask]),
    }


def replay_o25(snapshots: list[dict], model_p: np.ndarray, alpha: float, threshold: float) -> dict:
    profits = []
    evs = []
    overs = unders = wins = 0
    for r, pm in zip(snapshots, model_p):
        if not all(math.isfinite(float(r[k])) for k in ["odd_o25", "odd_u25", "market_o25"]):
            continue
        po = float(r["market_o25"]) + alpha * (float(pm) - float(r["market_o25"]))
        pu = 1.0 - po
        evo = po * float(r["odd_o25"]) - 1.0
        evu = pu * float(r["odd_u25"]) - 1.0
        if evo >= evu:
            ev, odd, won, side = evo, float(r["odd_o25"]), int(r["y_O2.5"]) == 1, "O"
        else:
            ev, odd, won, side = evu, float(r["odd_u25"]), int(r["y_O2.5"]) == 0, "U"
        if ev < threshold:
            continue
        profits.append(odd - 1.0 if won else -1.0)
        evs.append(ev)
        wins += int(won)
        overs += int(side == "O")
        unders += int(side == "U")

    n = len(profits)
    equity = peak = max_dd = 0.0
    for x in profits:
        equity += x
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "threshold": threshold,
        "bets": n,
        "wins": wins,
        "hit": wins / n if n else None,
        "profit": float(sum(profits)),
        "roi": float(sum(profits) / n) if n else None,
        "avg_ev": float(np.mean(evs)) if evs else None,
        "max_dd": float(max_dd) if n else None,
        "overs": overs,
        "unders": unders,
    }


def h2h_metrics(snapshots: list[dict], lh: np.ndarray, la: np.ndarray) -> dict:
    correct = 0
    briers = []
    losses = []
    for r, xh, xa in zip(snapshots, lh, la):
        p = project_match_from_lambdas(float(xh), float(xa))["match_result"]
        probs = np.asarray([p["home"], p["draw"], p["away"]], dtype=float)
        y_idx = 0 if r["hg"] > r["ag"] else (1 if r["hg"] == r["ag"] else 2)
        one = np.eye(3)[y_idx]
        correct += int(int(np.argmax(probs)) == y_idx)
        briers.append(float(np.sum((probs - one) ** 2)))
        losses.append(float(-math.log(max(EPS, probs[y_idx]))))
    return {
        "n": len(snapshots),
        "acc": correct / len(snapshots),
        "brier": float(np.mean(briers)),
        "logloss": float(np.mean(losses)),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    snapshots: list[dict] = []
    failures: list[str] = []
    for season in SEASONS:
        for league, league_name in LEAGUES.items():
            try:
                df = load_csv_robust(season, league)
                rows = prepare_snapshots(df, season, league, league_name)
                snapshots.extend(rows)
                print(f"SNAPSHOT {season} {league}: raw={len(df)} usable={len(rows)}")
            except Exception as exc:
                failures.append(f"{season}/{league}: {type(exc).__name__}: {exc}")
                print("WARNING", failures[-1])

    if not snapshots:
        raise RuntimeError("No usable snapshots")

    dev = [r for r in snapshots if r["season"] == "2122"]
    val = [r for r in snapshots if r["season"] == "2223"]
    hold = [r for r in snapshots if r["season"] == "2324"]

    v1_cfg = {
        "general_window": 5,
        "venue_window": 5,
        "general_weight": 0.45,
        "league_shrinkage": 0.10,
        "elo_strength": 0.0,
    }

    v1_probs = {}
    v1_ll = {}
    v1_lambdas = {}
    for scope_name, data in [("DEV", dev), ("VALIDATION", val), ("HOLDOUT", hold)]:
        lh, la = lambdas_for_config(data, v1_cfg)
        pp = probabilities(lh, la)
        v1_lambdas[scope_name] = (lh, la)
        v1_probs[scope_name] = pp
        v1_ll[scope_name] = {m: logloss(y_array(data, m), pp[m]) for m in MARKETS}

    grid = []
    for general_window in [5, 8, 10]:
        for venue_window in [5, 8, 10]:
            for general_weight in [0.35, 0.45, 0.55]:
                for league_shrinkage in [0.10, 0.20, 0.30]:
                    for elo_strength in [0.00, 0.10, 0.20, 0.30]:
                        cfg = {
                            "general_window": general_window,
                            "venue_window": venue_window,
                            "general_weight": general_weight,
                            "league_shrinkage": league_shrinkage,
                            "elo_strength": elo_strength,
                        }
                        lh, la = lambdas_for_config(dev, cfg)
                        pp = probabilities(lh, la)
                        objective = relative_objective(dev, pp, v1_ll["DEV"])
                        grid.append({**cfg, "dev_relative_logloss": objective})

    grid.sort(key=lambda r: r["dev_relative_logloss"])
    shortlist = grid[:20]
    for row in shortlist:
        cfg = {k: row[k] for k in ["general_window", "venue_window", "general_weight", "league_shrinkage", "elo_strength"]}
        lh, la = lambdas_for_config(val, cfg)
        pp = probabilities(lh, la)
        row["validation_relative_logloss"] = relative_objective(val, pp, v1_ll["VALIDATION"])

    shortlist.sort(key=lambda r: (r["validation_relative_logloss"], r["dev_relative_logloss"]))
    best = shortlist[0]
    best_cfg = {k: best[k] for k in ["general_window", "venue_window", "general_weight", "league_shrinkage", "elo_strength"]}

    metric_output = []
    compare_output = []
    scope_data = {"DEV": dev, "VALIDATION": val, "HOLDOUT": hold}
    v2_probs = {}
    v2_lambdas = {}

    for scope_name, data in scope_data.items():
        lh2, la2 = lambdas_for_config(data, best_cfg)
        pp2 = probabilities(lh2, la2)
        v2_lambdas[scope_name] = (lh2, la2)
        v2_probs[scope_name] = pp2
        metric_output.extend(metric_rows(scope_name, data, v1_probs[scope_name], "V1"))
        metric_output.extend(metric_rows(scope_name, data, pp2, "V2"))

        for market in MARKETS:
            y = y_array(data, market)
            p1 = v1_probs[scope_name][market]
            p2 = pp2[market]
            ll1, ll2 = logloss(y, p1), logloss(y, p2)
            br1, br2 = brier(y, p1), brier(y, p2)
            compare_output.append({
                "scope": scope_name,
                "market": market,
                "v1_logloss": ll1,
                "v2_logloss": ll2,
                "logloss_change_pct": (ll2 / ll1 - 1.0) * 100.0,
                "v1_brier": br1,
                "v2_brier": br2,
                "brier_change_pct": (br2 / br1 - 1.0) * 100.0,
                "v1_ece": ece(y, p1),
                "v2_ece": ece(y, p2),
            })

    alpha, alpha_grid = tune_alpha(dev, v2_probs["DEV"]["O2.5"])
    hold_replay = [replay_o25(hold, v2_probs["HOLDOUT"]["O2.5"], alpha, t) for t in THRESHOLDS]

    # League breakdown for the frozen holdout config.
    league_rows = []
    offset = 0
    for league, league_name in LEAGUES.items():
        league_hold = [r for r in hold if r["league"] == league]
        if not league_hold:
            continue
        lh, la = lambdas_for_config(league_hold, best_cfg)
        pp = probabilities(lh, la)
        for market in MARKETS:
            y = y_array(league_hold, market)
            league_rows.append({
                "league": league,
                "league_name": league_name,
                "market": market,
                "n": len(y),
                "actual": float(y.mean()),
                "pred": float(pp[market].mean()),
                "bias_pp": float((pp[market].mean() - y.mean()) * 100.0),
                "brier": brier(y, pp[market]),
                "logloss": logloss(y, pp[market]),
                "ece": ece(y, pp[market]),
            })

    h2h_v1 = h2h_metrics(hold, *v1_lambdas["HOLDOUT"])
    h2h_v2 = h2h_metrics(hold, *v2_lambdas["HOLDOUT"])

    pd.DataFrame(grid).to_csv(OUT / "goal_model_v2_grid_all.csv", index=False)
    pd.DataFrame(shortlist).to_csv(OUT / "goal_model_v2_grid_shortlist.csv", index=False)
    pd.DataFrame(metric_output).to_csv(OUT / "goal_model_v2_probability_metrics.csv", index=False)
    pd.DataFrame(compare_output).to_csv(OUT / "goal_model_v2_vs_v1.csv", index=False)
    pd.DataFrame(league_rows).to_csv(OUT / "goal_model_v2_holdout_by_league.csv", index=False)
    pd.DataFrame(alpha_grid).to_csv(OUT / "goal_model_v2_alpha_grid.csv", index=False)
    pd.DataFrame(hold_replay).to_csv(OUT / "goal_model_v2_holdout_o25_replay.csv", index=False)

    summary = {
        "n": len(snapshots),
        "dev": len(dev),
        "validation": len(val),
        "holdout": len(hold),
        "selection_rule": "Top 20 by 2021/22 relative logloss, then best 2022/23 relative logloss; 2023/24 untouched until final evaluation.",
        "v1_config": v1_cfg,
        "v2_config": best_cfg,
        "v2_dev_relative_logloss": float(best["dev_relative_logloss"]),
        "v2_validation_relative_logloss": float(best["validation_relative_logloss"]),
        "o25_alpha_from_dev": alpha,
        "o25_dev_v2_vs_market": o25_market_metrics(dev, v2_probs["DEV"]["O2.5"]),
        "o25_validation_v2_vs_market": o25_market_metrics(val, v2_probs["VALIDATION"]["O2.5"]),
        "o25_holdout_v2_vs_market": o25_market_metrics(hold, v2_probs["HOLDOUT"]["O2.5"]),
        "h2h_holdout_v1": h2h_v1,
        "h2h_holdout_v2": h2h_v2,
        "holdout_o25_replay": hold_replay,
        "failures": failures,
    }
    (OUT / "goal_model_v2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== GOAL MODEL V2 RESEARCH SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nReports written to", OUT.resolve())


if __name__ == "__main__":
    main()
