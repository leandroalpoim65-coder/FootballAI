from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from agent.v12_calibrated import V12_SPORT_FEATURES
from agent.v12_dataset import build_dataset
from agent.v12_modeling import (
    apply_calibrator, betting_metrics, calibration_table, candidates,
    choose_base, choose_calibrator, edge_distribution, ece, fit_calibrator,
    probabilities, probability_metrics,
)
from footballai_agent_cloud import connect_turso, fetch_dicts

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ml" / "modelo_over25_v12.pkl"
META_PATH = ROOT / "ml" / "metadata_modelo_over25_v12.json"
REPORT_PATH = ROOT / "reports" / "v12_test_calibration.csv"
VERSION = "v12-calibrated-65k-v4"


def load_matches() -> pd.DataFrame:
    conn = connect_turso()
    try:
        rows = fetch_dicts(conn, """
            SELECT id, date, league, home_team, away_team,
                   home_goals, away_goals,
                   COALESCE(odd_over25_open, odd_over25) AS odd_over25_open,
                   COALESCE(odd_under25_open, odd_under25) AS odd_under25_open
            FROM matches
            ORDER BY date, id
        """)
    finally:
        conn.close()
    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("Tabela matches vazia.")
    data["date"] = pd.to_datetime(data.date, errors="coerce", format="%Y-%m-%d")
    for col in ("home_goals", "away_goals", "odd_over25_open", "odd_under25_open"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ("league", "home_team", "away_team"):
        data[col] = data[col].astype("string").str.strip()
    data = data.dropna(subset=["id","date","league","home_team","away_team","home_goals","away_goals"])
    data = data.sort_values(["date","id"]).drop_duplicates("id", keep="last")
    data = data.drop_duplicates(["date","league","home_team","away_team"], keep="last").reset_index(drop=True)
    data["actual_over25"] = ((data.home_goals + data.away_goals) >= 3).astype(int)
    return data


def date_cuts(data: pd.DataFrame):
    dates = np.array(sorted(pd.Timestamp(x) for x in data.date.dropna().unique()))
    if len(dates) < 100:
        raise RuntimeError("Poucas datas únicas para validação temporal V12.")
    def at(frac: float):
        return pd.Timestamp(dates[min(int(len(dates) * frac), len(dates) - 1)])
    return at(.55), at(.65), at(.75), at(.80), at(.90)


def market_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.market_eligible == 1].dropna(
        subset=["odd_over25_open","odd_under25_open","market_probability","market_overround"]
    ).copy()


def metric_delta(model_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "brier": float(model_metrics["brier"] - baseline_metrics["brier"]),
        "log_loss": float(model_metrics["log_loss"] - baseline_metrics["log_loss"]),
        "auc": float(model_metrics["auc"] - baseline_metrics["auc"]),
        "calibration_bias_abs": float(abs(model_metrics["calibration_bias"]) - abs(baseline_metrics["calibration_bias"])),
    }


def main() -> None:
    print("=" * 76)
    print("V12 CALIBRATED AGENT | futebol independente + consenso de mercado")
    print("=" * 76)
    matches = load_matches()
    print(f"Histórico bruto deduplicado : {len(matches)}")
    data = build_dataset(matches)
    print(f"Elegíveis modelo futebol    : {len(data)}")
    print(f"Elegíveis também com odds   : {int(data.market_eligible.sum())}")
    print(f"Período elegível            : {data.date.min().date()} -> {data.date.max().date()}")

    d55, d65, d75, d80, d90 = date_cuts(data)
    train = data[data.date < d55].copy()
    select = data[(data.date >= d55) & (data.date < d65)].copy()
    sport_cal_fit = data[(data.date >= d65) & (data.date < d75)].copy()
    sport_cal_select = data[(data.date >= d75) & (data.date < d80)].copy()
    market_cal_fit = market_only(sport_cal_fit)
    market_cal_select = market_only(sport_cal_select)
    policy_set = market_only(data[(data.date >= d80) & (data.date < d90)])
    test_sport = data[data.date >= d90].copy()
    test_market = market_only(test_sport)
    print(
        "Splits: "
        f"train={len(train)} select={len(select)} "
        f"sport_cal_fit={len(sport_cal_fit)} sport_cal_select={len(sport_cal_select)} "
        f"market_cal_fit={len(market_cal_fit)} market_cal_select={len(market_cal_select)} "
        f"policy={len(policy_set)} test_sport={len(test_sport)} test_market={len(test_market)}"
    )
    required = [train, select, sport_cal_fit, sport_cal_select, market_cal_fit, market_cal_select, policy_set, test_market]
    if min(len(x) for x in required) < 100:
        raise RuntimeError("Um dos blocos temporais V12 tem menos de 100 jogos.")

    features = list(V12_SPORT_FEATURES)
    winner, base_comparison = choose_base(train, select, features)
    print("Modelos base:", json.dumps(base_comparison, ensure_ascii=False))
    print("Vencedor base:", winner)

    base_fit = data[data.date < d65].copy()
    base = clone(candidates()[winner])
    base.fit(base_fit[features], base_fit.actual_over25)

    independent_kind, _, independent_comparison = choose_calibrator(
        base, sport_cal_fit, sport_cal_select, features,
        kinds=("sigmoid", "isotonic"),
    )
    consensus_kind, _, consensus_comparison = choose_calibrator(
        base, market_cal_fit, market_cal_select, features,
        kinds=("market_blend",),
    )
    print("Calibradores independentes:", json.dumps(independent_comparison, ensure_ascii=False))
    print("Vencedor independente:", independent_kind)
    print("Calibrador consenso:", json.dumps(consensus_comparison, ensure_ascii=False))

    selected_row = next(row for row in independent_comparison if row["calibrator"] == independent_kind)
    calibration_ece = float(selected_row["ece"])
    calibration_bias = abs(float(selected_row["calibration_bias"]))
    safety = float(np.clip(max(0.03, 1.5 * calibration_ece + calibration_bias), 0.03, 0.08))
    policy = {"min_edge":round(safety,4), "min_ev":round(max(.03,safety),4), "min_odd":1.20, "max_odd":4.00}
    print(f"Policy independente sem otimizar ROI: {policy}")

    sport_cal_all = data[(data.date >= d65) & (data.date < d80)].copy()
    raw_sport_cal = probabilities(base, sport_cal_all[features])
    independent_calibrator = fit_calibrator(independent_kind, raw_sport_cal, sport_cal_all)

    market_cal_all = market_only(sport_cal_all)
    raw_market_cal = probabilities(base, market_cal_all[features])
    consensus_calibrator = fit_calibrator(consensus_kind, raw_market_cal, market_cal_all)
    consensus_coefficients = {
        "sports_logit": float(consensus_calibrator.coef_[0][0]),
        "market_logit": float(consensus_calibrator.coef_[0][1]),
        "overround": float(consensus_calibrator.coef_[0][2]),
        "intercept": float(consensus_calibrator.intercept_[0]),
    }
    print("Pesos consenso market_blend:", json.dumps(consensus_coefficients, ensure_ascii=False))

    raw_policy = probabilities(base, policy_set[features])
    p_policy_ind = apply_calibrator(independent_kind, independent_calibrator, raw_policy, policy_set)
    p_policy_cons = apply_calibrator(consensus_kind, consensus_calibrator, raw_policy, policy_set)
    policy_market_baseline = probability_metrics(
        policy_set.actual_over25,
        policy_set.market_probability.to_numpy(dtype=float),
    )
    policy_consensus_metrics = probability_metrics(policy_set.actual_over25, p_policy_cons)
    print("POLICY market baseline:", json.dumps(policy_market_baseline, ensure_ascii=False))
    print("POLICY independent probability:", json.dumps(probability_metrics(policy_set.actual_over25, p_policy_ind), ensure_ascii=False))
    print("POLICY independent betting:", json.dumps(betting_metrics(policy_set, p_policy_ind, policy), ensure_ascii=False))
    print("POLICY independent edges:", json.dumps(edge_distribution(policy_set, p_policy_ind), ensure_ascii=False))
    print("POLICY consensus probability:", json.dumps(policy_consensus_metrics, ensure_ascii=False))
    print("POLICY consensus vs market delta:", json.dumps(metric_delta(policy_consensus_metrics, policy_market_baseline), ensure_ascii=False))

    raw_test_sport = probabilities(base, test_sport[features])
    p_test_sport_ind = apply_calibrator(independent_kind, independent_calibrator, raw_test_sport, test_sport)
    raw_test_market = probabilities(base, test_market[features])
    p_test_market_ind = apply_calibrator(independent_kind, independent_calibrator, raw_test_market, test_market)
    p_test_cons = apply_calibrator(consensus_kind, consensus_calibrator, raw_test_market, test_market)

    test_market_baseline = probability_metrics(
        test_market.actual_over25,
        test_market.market_probability.to_numpy(dtype=float),
    )
    test_market_ece = ece(
        test_market.actual_over25,
        test_market.market_probability.to_numpy(dtype=float),
    )
    test_ind_sport = probability_metrics(test_sport.actual_over25, p_test_sport_ind)
    test_ind_market = probability_metrics(test_market.actual_over25, p_test_market_ind)
    test_consensus = probability_metrics(test_market.actual_over25, p_test_cons)
    test_bets = betting_metrics(test_market, p_test_market_ind, policy)
    test_ind_ece = ece(test_market.actual_over25, p_test_market_ind)
    test_cons_ece = ece(test_market.actual_over25, p_test_cons)
    test_consensus_delta = metric_delta(test_consensus, test_market_baseline)

    print("TEST market baseline:", json.dumps(test_market_baseline, ensure_ascii=False))
    print(f"TEST market baseline ECE={test_market_ece:.4f}")
    print("TEST independent sports:", json.dumps(test_ind_sport, ensure_ascii=False))
    print("TEST independent market-subset:", json.dumps(test_ind_market, ensure_ascii=False))
    print("TEST independent betting:", json.dumps(test_bets, ensure_ascii=False))
    print("TEST independent edges:", json.dumps(edge_distribution(test_market, p_test_market_ind), ensure_ascii=False))
    print(f"TEST independent ECE={test_ind_ece:.4f}")
    print("TEST consensus calibrated:", json.dumps(test_consensus, ensure_ascii=False))
    print("TEST consensus vs market delta:", json.dumps(test_consensus_delta, ensure_ascii=False))
    print(f"TEST consensus ECE={test_cons_ece:.4f}")

    prod_base_train = data[data.date < d90].copy()
    prod_base = clone(candidates()[winner])
    prod_base.fit(prod_base_train[features], prod_base_train.actual_over25)

    prod_sport_cal = data[data.date >= d90].copy()
    prod_raw_sport = probabilities(prod_base, prod_sport_cal[features])
    prod_independent_calibrator = fit_calibrator(independent_kind, prod_raw_sport, prod_sport_cal)

    prod_market_cal = market_only(prod_sport_cal)
    prod_raw_market = probabilities(prod_base, prod_market_cal[features])
    prod_consensus_calibrator = fit_calibrator(consensus_kind, prod_raw_market, prod_market_cal)

    generated_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "version":VERSION,
        "sport_feature_names":features,
        "base_model":prod_base,
        "independent_calibration_kind":independent_kind,
        "independent_calibrator":prod_independent_calibrator,
        "consensus_calibration_kind":consensus_kind,
        "consensus_calibrator":prod_consensus_calibrator,
        "calibration_kind":consensus_kind,
        "calibrator":prod_consensus_calibrator,
        "policy":policy,
        "winner_model":winner,
        "generated_at":generated_at,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    calibration_table(test_market.actual_over25, p_test_cons).to_csv(REPORT_PATH, index=False)

    metadata = {
        "version":VERSION,
        "generated_at":generated_at,
        "raw_matches":int(len(matches)),
        "sport_eligible_matches":int(len(data)),
        "market_eligible_matches":int(data.market_eligible.sum()),
        "sport_feature_count":len(features),
        "sport_features":features,
        "date_min":data.date.min().date().isoformat(),
        "date_max":data.date.max().date().isoformat(),
        "split_dates":{
            "train_end":d55.date().isoformat(),
            "base_selection_end":d65.date().isoformat(),
            "calibration_fit_end":d75.date().isoformat(),
            "calibration_selection_end":d80.date().isoformat(),
            "policy_end":d90.date().isoformat(),
        },
        "split_sizes":{
            "train":len(train), "selection":len(select),
            "sport_calibration_fit":len(sport_cal_fit), "sport_calibration_selection":len(sport_cal_select),
            "market_calibration_fit":len(market_cal_fit), "market_calibration_selection":len(market_cal_select),
            "policy":len(policy_set), "test_sport":len(test_sport), "test_market":len(test_market),
        },
        "base_model_comparison":base_comparison,
        "winner_model":winner,
        "independent_calibrator_comparison":independent_comparison,
        "independent_calibration_kind":independent_kind,
        "consensus_calibrator_comparison":consensus_comparison,
        "consensus_calibration_kind":consensus_kind,
        "consensus_coefficients":consensus_coefficients,
        "policy":policy,
        "policy_market_baseline_metrics":policy_market_baseline,
        "policy_consensus_metrics":policy_consensus_metrics,
        "policy_consensus_vs_market_delta":metric_delta(policy_consensus_metrics, policy_market_baseline),
        "test_market_baseline_metrics":test_market_baseline,
        "test_market_baseline_ece":test_market_ece,
        "test_independent_sport_metrics":test_ind_sport,
        "test_independent_market_metrics":test_ind_market,
        "test_independent_ece":test_ind_ece,
        "test_independent_betting_metrics":test_bets,
        "test_independent_edge_distribution":edge_distribution(test_market, p_test_market_ind),
        "test_consensus_metrics":test_consensus,
        "test_consensus_ece":test_cons_ece,
        "test_consensus_vs_market_delta":test_consensus_delta,
        "test_probability_metrics":test_consensus,
        "test_ece":test_cons_ece,
        "test_betting_metrics":test_bets,
        "methodology":"football model independent of odds; chronological unique-date blocks; independent sports probability is audited but not promoted as betting edge after negative untouched-test ROI; market blend is evaluated only as calibrated consensus and explicitly compared against market-only baseline; thresholds are never selected on final-test ROI; same-day-safe Elo/league context; promoted teams keep exact-identity history within domestic league family",
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Modelo V12 gerado: {MODEL_PATH.stat().st_size/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
