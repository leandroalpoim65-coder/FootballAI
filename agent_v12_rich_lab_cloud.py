from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from agent.v12_calibrated import V12_SPORT_FEATURES
from agent.v12_dataset import build_dataset
from agent.v12_modeling import (
    apply_calibrator,
    candidates,
    choose_base,
    choose_calibrator,
    ece,
    fit_calibrator,
    probabilities,
    probability_metrics,
)
from agent.v12_rich import RICH_FEATURES, build_rich_features
from footballai_agent_cloud import connect_turso, fetch_dicts


def load_matches() -> pd.DataFrame:
    connection = connect_turso()
    try:
        rows = fetch_dicts(
            connection,
            """
            SELECT
                id, date, league, home_team, away_team,
                home_goals, away_goals,
                COALESCE(odd_over25_open, odd_over25) AS odd_over25_open,
                COALESCE(odd_under25_open, odd_under25) AS odd_under25_open,
                home_shots, away_shots,
                home_shots_target, away_shots_target,
                home_corners, away_corners
            FROM matches
            ORDER BY date, id
            """,
        )
    finally:
        connection.close()
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"], errors="coerce", format="%Y-%m-%d")
    numeric = [
        "home_goals", "away_goals", "odd_over25_open", "odd_under25_open",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ("league", "home_team", "away_team"):
        data[column] = data[column].astype("string").str.strip()
    data = data.dropna(subset=["id", "date", "league", "home_team", "away_team", "home_goals", "away_goals"])
    data = data.sort_values(["date", "id"]).drop_duplicates("id", keep="last")
    data = data.drop_duplicates(["date", "league", "home_team", "away_team"], keep="last").reset_index(drop=True)
    data["actual_over25"] = ((data.home_goals + data.away_goals) >= 3).astype(int)
    return data


def date_cuts(data: pd.DataFrame):
    dates = np.array(sorted(pd.Timestamp(value) for value in data.date.dropna().unique()))
    def at(frac: float):
        return pd.Timestamp(dates[min(int(len(dates) * frac), len(dates) - 1)])
    return at(.55), at(.65), at(.75), at(.80), at(.90)


def market_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame.market_eligible == 1)
        & frame.market_probability.notna()
        & frame.market_overround.notna()
    ].copy()


def delta(model: dict, market: dict) -> dict[str, float]:
    return {
        "brier": float(model["brier"] - market["brier"]),
        "log_loss": float(model["log_loss"] - market["log_loss"]),
        "auc": float(model["auc"] - market["auc"]),
    }


def evaluate_feature_set(data: pd.DataFrame, feature_names: list[str], label: str, cuts) -> dict:
    d55, d65, d75, d80, d90 = cuts
    train = data[data.date < d55].copy()
    select = data[(data.date >= d55) & (data.date < d65)].copy()
    sport_cal_fit = data[(data.date >= d65) & (data.date < d75)].copy()
    sport_cal_select = data[(data.date >= d75) & (data.date < d80)].copy()
    market_cal = market_only(data[(data.date >= d65) & (data.date < d80)])
    policy = market_only(data[(data.date >= d80) & (data.date < d90)])
    test = market_only(data[data.date >= d90])

    winner, model_comparison = choose_base(train, select, feature_names)
    base = clone(candidates()[winner])
    base.fit(data[data.date < d65][feature_names], data[data.date < d65].actual_over25)

    independent_kind, _, independent_comparison = choose_calibrator(
        base,
        sport_cal_fit,
        sport_cal_select,
        feature_names,
        kinds=("sigmoid", "isotonic"),
    )
    sport_cal = data[(data.date >= d65) & (data.date < d80)].copy()
    independent_cal = fit_calibrator(
        independent_kind,
        probabilities(base, sport_cal[feature_names]),
        sport_cal,
    )
    consensus_cal = fit_calibrator(
        "market_blend",
        probabilities(base, market_cal[feature_names]),
        market_cal,
    )

    raw_policy = probabilities(base, policy[feature_names])
    policy_ind = apply_calibrator(independent_kind, independent_cal, raw_policy, policy)
    policy_cons = apply_calibrator("market_blend", consensus_cal, raw_policy, policy)
    policy_market = probability_metrics(policy.actual_over25, policy.market_probability.to_numpy(float))
    policy_cons_metrics = probability_metrics(policy.actual_over25, policy_cons)

    raw_test = probabilities(base, test[feature_names])
    test_ind = apply_calibrator(independent_kind, independent_cal, raw_test, test)
    test_cons = apply_calibrator("market_blend", consensus_cal, raw_test, test)
    test_market = probability_metrics(test.actual_over25, test.market_probability.to_numpy(float))
    test_ind_metrics = probability_metrics(test.actual_over25, test_ind)
    test_cons_metrics = probability_metrics(test.actual_over25, test_cons)

    result = {
        "label": label,
        "features": len(feature_names),
        "rows": len(data),
        "winner_model": winner,
        "model_selection": model_comparison,
        "independent_calibrator": independent_kind,
        "independent_selection": independent_comparison,
        "consensus_coefficients": {
            "sports_logit": float(consensus_cal.coef_[0][0]),
            "market_logit": float(consensus_cal.coef_[0][1]),
            "overround": float(consensus_cal.coef_[0][2]),
            "intercept": float(consensus_cal.intercept_[0]),
        },
        "policy_n": len(policy),
        "policy_market": policy_market,
        "policy_consensus": policy_cons_metrics,
        "policy_delta": delta(policy_cons_metrics, policy_market),
        "test_n": len(test),
        "test_market": test_market,
        "test_independent": test_ind_metrics,
        "test_consensus": test_cons_metrics,
        "test_delta": delta(test_cons_metrics, test_market),
        "test_market_ece": ece(test.actual_over25, test.market_probability.to_numpy(float)),
        "test_independent_ece": ece(test.actual_over25, test_ind),
        "test_consensus_ece": ece(test.actual_over25, test_cons),
    }
    print(f"RICH_LAB {label}: {json.dumps(result, ensure_ascii=False)}")
    return result


def main() -> None:
    print("=" * 76)
    print("V12 RICH FEATURE LAB | base 53 vs remates + SOT + cantos")
    print("=" * 76)
    matches = load_matches()
    base = build_dataset(matches)
    rich = build_rich_features(matches)
    data = base.set_index("id").join(rich, how="left").reset_index()

    rich_complete = data.dropna(subset=list(RICH_FEATURES)).copy()
    print(f"Base elegível          : {len(data)}")
    print(f"Rich completo          : {len(rich_complete)} ({len(rich_complete)/len(data):.2%})")
    print(f"Rich período           : {rich_complete.date.min().date()} -> {rich_complete.date.max().date()}")

    cuts = date_cuts(rich_complete)
    base_features = list(V12_SPORT_FEATURES)
    rich_features = base_features + list(RICH_FEATURES)

    base_result = evaluate_feature_set(rich_complete, base_features, "BASE_SAME_ROWS", cuts)
    rich_result = evaluate_feature_set(rich_complete, rich_features, "RICH", cuts)

    comparison = {
        "policy_brier_gain_rich_vs_base": float(base_result["policy_consensus"]["brier"] - rich_result["policy_consensus"]["brier"]),
        "policy_logloss_gain_rich_vs_base": float(base_result["policy_consensus"]["log_loss"] - rich_result["policy_consensus"]["log_loss"]),
        "policy_auc_gain_rich_vs_base": float(rich_result["policy_consensus"]["auc"] - base_result["policy_consensus"]["auc"]),
        "test_brier_gain_rich_vs_base": float(base_result["test_consensus"]["brier"] - rich_result["test_consensus"]["brier"]),
        "test_logloss_gain_rich_vs_base": float(base_result["test_consensus"]["log_loss"] - rich_result["test_consensus"]["log_loss"]),
        "test_auc_gain_rich_vs_base": float(rich_result["test_consensus"]["auc"] - base_result["test_consensus"]["auc"]),
    }
    print("RICH_LAB COMPARISON:", json.dumps(comparison, ensure_ascii=False))


if __name__ == "__main__":
    main()
