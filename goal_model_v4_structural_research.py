from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import goal_model_v2_multileague_research as v2
from goal_model_multileague_backtest_local import load_csv_robust

OUT = Path("reports")

RESEARCH_SEASONS = ["2122", "2223", "2324"]
VALIDATION_SEASON = "2425"
HOLDOUT_SEASON = "2526"
ALL_SEASONS = RESEARCH_SEASONS + [VALIDATION_SEASON, HOLDOUT_SEASON]

LEAGUES = v2.LEAGUES
MARKETS = v2.MARKETS
TEAM_MARKETS = [
    "HomeO0.5", "HomeO1.5", "HomeO2.5",
    "AwayO0.5", "AwayO1.5", "AwayO2.5",
]
BROAD_MARKETS = ["BTTS", "O1.5", "O2.5", "O3.5"]

MAX_HISTORY = 20
MIN_HISTORY = 5
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 65.0
EPS = 1e-12

V2_CONFIG = {
    "general_window": 10,
    "venue_window": 10,
    "general_weight": 0.55,
    "league_shrinkage": 0.20,
    "elo_strength": 0.30,
}

WINDOWS = [8, 12, 20]
DECAYS = [0.85, 0.93, 1.00]
PRIORS = [4.0, 8.0]
VENUE_WEIGHTS = [0.50, 0.70, 0.90]
BLENDS = [0.35, 0.60, 0.85]
EXTRA_ELO = [0.00, 0.10]
SHORTLIST_N = 25

THRESHOLDS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
ALPHAS = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]


def sf(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if math.isfinite(x) else np.nan


def no_vig2(a: float, b: float) -> tuple[float, float]:
    ia, ib = 1.0 / a, 1.0 / b
    s = ia + ib
    return ia / s, ib / s


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
                len(state[home]["all"]),
                len(state[away]["all"]),
                len(state[home]["home"]),
                len(state[away]["away"]),
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
                rec.update(
                    odd_o25=oo,
                    odd_u25=ou,
                    market_o25=mo,
                    market_u25=mu,
                )
            else:
                rec.update(
                    odd_o25=np.nan,
                    odd_u25=np.nan,
                    market_o25=np.nan,
                    market_u25=np.nan,
                )
            records.append(rec)

        # Same-date results are added only after every match on that date was snapshotted.
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


def weighted_mean(values: list[float], window: int, decay: float) -> tuple[float, int]:
    vals = np.asarray(values[-window:], dtype=float)
    n = len(vals)
    if n == 0:
        return np.nan, 0
    if decay >= 0.999999:
        return float(vals.mean()), n
    # Most recent observation gets weight 1.0.
    powers = np.arange(n - 1, -1, -1, dtype=float)
    weights = decay ** powers
    return float(np.average(vals, weights=weights)), n


def shrunk_ratio(values: list[float], baseline: float, window: int, decay: float, prior: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0:
        return 1.0
    mean, n = weighted_mean(values, window, decay)
    if n == 0 or not math.isfinite(mean):
        return 1.0
    raw = max(0.05, mean / baseline)
    ratio = (n * raw + prior) / (n + prior)
    return float(min(2.75, max(0.35, ratio)))


def geom_blend(a: float, b: float, weight_b: float) -> float:
    a = max(EPS, a)
    b = max(EPS, b)
    return float(math.exp((1.0 - weight_b) * math.log(a) + weight_b * math.log(b)))


def v4_lambdas(data: list[dict], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    v2_home, v2_away = v2.lambdas_for_config(data, V2_CONFIG)

    window = int(cfg["window"])
    decay = float(cfg["decay"])
    prior = float(cfg["prior"])
    venue_weight = float(cfg["venue_weight"])
    blend = float(cfg["blend"])
    extra_elo = float(cfg["extra_elo"])

    out_h = np.empty(len(data), dtype=float)
    out_a = np.empty(len(data), dtype=float)

    for i, r in enumerate(data):
        lh_avg = float(r["league_home_avg"])
        la_avg = float(r["league_away_avg"])
        if not math.isfinite(lh_avg) or lh_avg <= 0:
            lh_avg = 1.45
        if not math.isfinite(la_avg) or la_avg <= 0:
            la_avg = 1.15
        lg_avg = 0.5 * (lh_avg + la_avg)

        h_att_g = shrunk_ratio(r["h_all_scored"], lg_avg, window, decay, prior)
        h_def_g = shrunk_ratio(r["h_all_conceded"], lg_avg, window, decay, prior)
        a_att_g = shrunk_ratio(r["a_all_scored"], lg_avg, window, decay, prior)
        a_def_g = shrunk_ratio(r["a_all_conceded"], lg_avg, window, decay, prior)

        h_att_v = shrunk_ratio(r["h_home_scored"], lh_avg, window, decay, prior)
        h_def_v = shrunk_ratio(r["h_home_conceded"], la_avg, window, decay, prior)
        a_att_v = shrunk_ratio(r["a_away_scored"], la_avg, window, decay, prior)
        a_def_v = shrunk_ratio(r["a_away_conceded"], lh_avg, window, decay, prior)

        h_att = geom_blend(h_att_g, h_att_v, venue_weight)
        h_def = geom_blend(h_def_g, h_def_v, venue_weight)
        a_att = geom_blend(a_att_g, a_att_v, venue_weight)
        a_def = geom_blend(a_def_g, a_def_v, venue_weight)

        # Classic multiplicative attack/defence structure around league scoring baselines.
        ad_home = lh_avg * h_att * a_def
        ad_away = la_avg * a_att * h_def

        if extra_elo != 0.0:
            mult = math.exp(extra_elo * float(r["elo_diff"]) / 400.0)
            ad_home *= mult
            ad_away /= mult

        xh = geom_blend(float(v2_home[i]), ad_home, blend)
        xa = geom_blend(float(v2_away[i]), ad_away, blend)

        out_h[i] = min(4.50, max(0.15, xh))
        out_a[i] = min(4.50, max(0.15, xa))

    return out_h, out_a


def objective(data: list[dict], candidate_probs: dict, baseline_probs: dict) -> tuple[float, float, float]:
    team_ratios = []
    broad_ratios = []
    for market in TEAM_MARKETS:
        y = v2.y_array(data, market)
        base = v2.logloss(y, baseline_probs[market])
        new = v2.logloss(y, candidate_probs[market])
        team_ratios.append(new / base)
    for market in BROAD_MARKETS:
        y = v2.y_array(data, market)
        base = v2.logloss(y, baseline_probs[market])
        new = v2.logloss(y, candidate_probs[market])
        broad_ratios.append(new / base)

    team = float(np.mean(team_ratios))
    broad = float(np.mean(broad_ratios))
    combined = 0.70 * team + 0.30 * broad
    return combined, team, broad


def metric_rows(scope: str, data: list[dict], probs: dict[str, np.ndarray], version: str) -> list[dict]:
    return v2.metric_rows(scope, data, probs, version)


def compare_rows(scope: str, data: list[dict], p2: dict, p4: dict) -> list[dict]:
    rows = []
    for market in MARKETS:
        y = v2.y_array(data, market)
        ll2, ll4 = v2.logloss(y, p2[market]), v2.logloss(y, p4[market])
        br2, br4 = v2.brier(y, p2[market]), v2.brier(y, p4[market])
        rows.append({
            "scope": scope,
            "market": market,
            "v2_logloss": ll2,
            "v4_logloss": ll4,
            "logloss_change_pct": (ll4 / ll2 - 1.0) * 100.0,
            "v2_brier": br2,
            "v4_brier": br4,
            "brier_change_pct": (br4 / br2 - 1.0) * 100.0,
            "v2_ece": v2.ece(y, p2[market]),
            "v4_ece": v2.ece(y, p4[market]),
            "v2_pred": float(np.mean(p2[market])),
            "v4_pred": float(np.mean(p4[market])),
            "actual": float(np.mean(y)),
        })
    return rows


def tune_alpha(data: list[dict], model_p: np.ndarray) -> tuple[float, list[dict]]:
    y = v2.y_array(data, "O2.5")
    market = np.asarray([r["market_o25"] for r in data], dtype=float)
    mask = np.isfinite(market)
    rows = []
    for alpha in ALPHAS:
        p = market[mask] + alpha * (model_p[mask] - market[mask])
        rows.append({
            "alpha": alpha,
            "n": int(mask.sum()),
            "logloss": v2.logloss(y[mask], p),
            "brier": v2.brier(y[mask], p),
        })
    best = min(rows, key=lambda r: (r["logloss"], r["brier"]))
    return float(best["alpha"]), rows


def replay_o25(data: list[dict], model_p: np.ndarray, alpha: float, threshold: float) -> dict:
    return v2.replay_o25(data, model_p, alpha, threshold)


def market_o25_metrics(data: list[dict], model_p: np.ndarray) -> dict:
    return v2.o25_market_metrics(data, model_p)


def build_data(seasons: list[str]) -> tuple[dict[str, list[dict]], list[str]]:
    by_season: dict[str, list[dict]] = {s: [] for s in seasons}
    failures: list[str] = []

    for season in seasons:
        for league, league_name in LEAGUES.items():
            try:
                df = load_csv_robust(season, league)
                part = prepare_snapshots(df, season, league, league_name)
                by_season[season].extend(part)
                print(f"SNAPSHOT {season} {league}: raw={len(df)} usable={len(part)}")
            except Exception as exc:
                msg = f"{season}/{league}: {type(exc).__name__}: {exc}"
                failures.append(msg)
                print("WARNING", msg)

    return by_season, failures


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Phase 1: research + validation only. The 2025/26 holdout is not even loaded
    # until the winning configuration is fully frozen.
    pre_holdout_seasons = RESEARCH_SEASONS + [VALIDATION_SEASON]
    by_season, failures = build_data(pre_holdout_seasons)

    missing = [s for s in pre_holdout_seasons if not by_season[s]]
    if missing:
        raise RuntimeError(f"Missing research/validation season groups: {missing}. Failures: {failures}")

    research = [r for s in RESEARCH_SEASONS for r in by_season[s]]
    validation = by_season[VALIDATION_SEASON]

    # Frozen V2 benchmark on exactly the same V4 snapshots.
    baseline_by_scope = {}
    for name, data in [
        ("RESEARCH", research),
        ("VALIDATION", validation),
    ]:
        lh2, la2 = v2.lambdas_for_config(data, V2_CONFIG)
        baseline_by_scope[name] = v2.probabilities(lh2, la2)

    baseline_by_research_season = {}
    for season in RESEARCH_SEASONS:
        d = by_season[season]
        lh2, la2 = v2.lambdas_for_config(d, V2_CONFIG)
        baseline_by_research_season[season] = v2.probabilities(lh2, la2)

    grid = []
    total_configs = (
        len(WINDOWS) * len(DECAYS) * len(PRIORS) *
        len(VENUE_WEIGHTS) * len(BLENDS) * len(EXTRA_ELO)
    )
    checked = 0

    for window in WINDOWS:
        for decay in DECAYS:
            for prior in PRIORS:
                for venue_weight in VENUE_WEIGHTS:
                    for blend in BLENDS:
                        for extra_elo in EXTRA_ELO:
                            cfg = {
                                "window": window,
                                "decay": decay,
                                "prior": prior,
                                "venue_weight": venue_weight,
                                "blend": blend,
                                "extra_elo": extra_elo,
                            }

                            season_scores = []
                            team_scores = []
                            broad_scores = []
                            for season in RESEARCH_SEASONS:
                                data = by_season[season]
                                lh4, la4 = v4_lambdas(data, cfg)
                                p4 = v2.probabilities(lh4, la4)
                                combined, team, broad = objective(
                                    data,
                                    p4,
                                    baseline_by_research_season[season],
                                )
                                season_scores.append(combined)
                                team_scores.append(team)
                                broad_scores.append(broad)

                            mean_score = float(np.mean(season_scores))
                            std_score = float(np.std(season_scores))
                            stability_score = mean_score + 0.25 * std_score
                            grid.append({
                                **cfg,
                                "research_relative_logloss": mean_score,
                                "research_team_relative_logloss": float(np.mean(team_scores)),
                                "research_broad_relative_logloss": float(np.mean(broad_scores)),
                                "research_std": std_score,
                                "research_stability_score": stability_score,
                                **{
                                    f"research_{s}_relative_logloss": float(score)
                                    for s, score in zip(RESEARCH_SEASONS, season_scores)
                                },
                            })
                            checked += 1
                            if checked % 50 == 0 or checked == total_configs:
                                print(f"GRID {checked}/{total_configs}")

    grid.sort(key=lambda r: (r["research_stability_score"], r["research_relative_logloss"]))
    shortlist = grid[:SHORTLIST_N]

    for row in shortlist:
        cfg = {
            k: row[k]
            for k in ["window", "decay", "prior", "venue_weight", "blend", "extra_elo"]
        }
        lh4, la4 = v4_lambdas(validation, cfg)
        p4 = v2.probabilities(lh4, la4)
        combined, team, broad = objective(
            validation,
            p4,
            baseline_by_scope["VALIDATION"],
        )
        row["validation_relative_logloss"] = combined
        row["validation_team_relative_logloss"] = team
        row["validation_broad_relative_logloss"] = broad

    # Freeze winner using validation only. Holdout is still untouched here.
    shortlist.sort(
        key=lambda r: (
            r["validation_relative_logloss"],
            r["research_stability_score"],
        )
    )
    best = shortlist[0]
    best_cfg = {
        k: best[k]
        for k in ["window", "decay", "prior", "venue_weight", "blend", "extra_elo"]
    }

    print("\nCONFIG FROZEN BEFORE HOLDOUT:", best_cfg)

    # Phase 2: only now load the untouched 2025/26 holdout.
    holdout_data, holdout_failures = build_data([HOLDOUT_SEASON])
    failures.extend(holdout_failures)
    holdout = holdout_data.get(HOLDOUT_SEASON, [])
    if not holdout:
        raise RuntimeError(
            f"Missing untouched holdout season {HOLDOUT_SEASON}. Failures: {holdout_failures}"
        )

    lh2_hold, la2_hold = v2.lambdas_for_config(holdout, V2_CONFIG)
    baseline_by_scope["HOLDOUT"] = v2.probabilities(lh2_hold, la2_hold)

    scope_data = {
        "RESEARCH": research,
        "VALIDATION": validation,
        "HOLDOUT": holdout,
    }
    p4_by_scope = {}
    metrics = []
    comparisons = []

    for scope, data in scope_data.items():
        lh4, la4 = v4_lambdas(data, best_cfg)
        p4 = v2.probabilities(lh4, la4)
        p4_by_scope[scope] = p4
        metrics.extend(metric_rows(scope, data, baseline_by_scope[scope], "V2"))
        metrics.extend(metric_rows(scope, data, p4, "V4"))
        comparisons.extend(compare_rows(scope, data, baseline_by_scope[scope], p4))

    validation_obj = objective(
        validation,
        p4_by_scope["VALIDATION"],
        baseline_by_scope["VALIDATION"],
    )
    holdout_obj = objective(
        holdout,
        p4_by_scope["HOLDOUT"],
        baseline_by_scope["HOLDOUT"],
    )

    holdout_cmp = [r for r in comparisons if r["scope"] == "HOLDOUT"]
    holdout_improved_all = sum(r["logloss_change_pct"] < 0 for r in holdout_cmp)
    holdout_team_cmp = [r for r in holdout_cmp if r["market"] in TEAM_MARKETS]
    holdout_improved_team = sum(r["logloss_change_pct"] < 0 for r in holdout_team_cmp)
    worst_broad_regression = max(
        (r["logloss_change_pct"] for r in holdout_cmp if r["market"] in BROAD_MARKETS),
        default=0.0,
    )

    # Market blend is tuned only on research seasons, then frozen.
    alpha, alpha_grid = tune_alpha(research, p4_by_scope["RESEARCH"]["O2.5"])
    validation_replay = [
        replay_o25(validation, p4_by_scope["VALIDATION"]["O2.5"], alpha, t)
        for t in THRESHOLDS
    ]
    holdout_replay = [
        replay_o25(holdout, p4_by_scope["HOLDOUT"]["O2.5"], alpha, t)
        for t in THRESHOLDS
    ]

    promotion_gate = (
        validation_obj[0] < 1.0
        and holdout_obj[0] < 1.0
        and holdout_improved_team >= 4
        and holdout_improved_all >= 6
        and worst_broad_regression <= 0.50
    )

    pd.DataFrame(grid).to_csv(OUT / "goal_model_v4_grid_all.csv", index=False)
    pd.DataFrame(shortlist).to_csv(OUT / "goal_model_v4_grid_shortlist.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "goal_model_v4_probability_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "goal_model_v4_vs_v2.csv", index=False)
    pd.DataFrame(alpha_grid).to_csv(OUT / "goal_model_v4_alpha_grid.csv", index=False)
    pd.DataFrame(validation_replay).to_csv(
        OUT / "goal_model_v4_validation_o25_replay.csv", index=False
    )
    pd.DataFrame(holdout_replay).to_csv(
        OUT / "goal_model_v4_holdout_o25_replay.csv", index=False
    )

    summary = {
        "research_seasons": RESEARCH_SEASONS,
        "validation_season": VALIDATION_SEASON,
        "holdout_season": HOLDOUT_SEASON,
        "n_research": len(research),
        "n_validation": len(validation),
        "n_holdout": len(holdout),
        "selection_rule": (
            "Tune 324 structural attack/defence configurations on 2021/22-2023/24 "
            "using mean relative logloss + stability penalty; shortlist top 25; "
            "select winner on 2024/25; load/evaluate 2025/26 only after freezing."
        ),
        "v2_config": V2_CONFIG,
        "v4_config": best_cfg,
        "research_relative_logloss": float(best["research_relative_logloss"]),
        "research_stability_score": float(best["research_stability_score"]),
        "validation_relative_logloss": float(validation_obj[0]),
        "validation_team_relative_logloss": float(validation_obj[1]),
        "validation_broad_relative_logloss": float(validation_obj[2]),
        "holdout_relative_logloss": float(holdout_obj[0]),
        "holdout_team_relative_logloss": float(holdout_obj[1]),
        "holdout_broad_relative_logloss": float(holdout_obj[2]),
        "holdout_markets_improved_logloss": int(holdout_improved_all),
        "holdout_markets_total": len(MARKETS),
        "holdout_team_markets_improved_logloss": int(holdout_improved_team),
        "holdout_team_markets_total": len(TEAM_MARKETS),
        "worst_holdout_broad_logloss_regression_pct": float(worst_broad_regression),
        "o25_alpha_from_research": alpha,
        "o25_validation_v4_vs_market": market_o25_metrics(
            validation, p4_by_scope["VALIDATION"]["O2.5"]
        ),
        "o25_holdout_v4_vs_market": market_o25_metrics(
            holdout, p4_by_scope["HOLDOUT"]["O2.5"]
        ),
        "validation_o25_replay": validation_replay,
        "holdout_o25_replay": holdout_replay,
        "promotion_gate_passed": bool(promotion_gate),
        "promotion_gate": (
            "Validation and holdout combined relative logloss < 1.0; "
            "at least 4/6 holdout team-total markets improve; at least 6/10 total "
            "markets improve; no broad market worsens by more than 0.50% logloss."
        ),
        "failures": failures,
    }

    (OUT / "goal_model_v4_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== GOAL MODEL V4 STRUCTURAL RESEARCH SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nReports written to", OUT.resolve())


if __name__ == "__main__":
    main()
