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
    choose_base, choose_calibrator, ece, fit_calibrator,
    probabilities, probability_metrics,
)
from footballai_agent_cloud import connect_turso, fetch_dicts

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ml" / "modelo_over25_v12.pkl"
META_PATH = ROOT / "ml" / "metadata_modelo_over25_v12.json"
REPORT_PATH = ROOT / "reports" / "v12_test_calibration.csv"
VERSION = "v12-calibrated-65k-v2"


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
    data["date"] = pd.to_datetime(data.date, errors="coerce", dayfirst=True)
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


def main() -> None:
    print("=" * 76)
    print("V12 CALIBRATED AGENT | futebol 65k + calibração de mercado")
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
    cal_fit = market_only(data[(data.date >= d65) & (data.date < d75)])
    cal_select = market_only(data[(data.date >= d75) & (data.date < d80)])
    policy_set = market_only(data[(data.date >= d80) & (data.date < d90)])
    test_sport = data[data.date >= d90].copy()
    test = market_only(test_sport)
    print(f"Splits: train={len(train)} select={len(select)} cal_fit={len(cal_fit)} cal_select={len(cal_select)} policy={len(policy_set)} test_market={len(test)}")
    if min(len(train), len(select), len(cal_fit), len(cal_select), len(policy_set), len(test)) < 100:
        raise RuntimeError("Um dos blocos temporais V12 tem menos de 100 jogos.")

    features = list(V12_SPORT_FEATURES)
    winner, base_comparison = choose_base(train, select, features)
    print("Modelos base:", json.dumps(base_comparison, ensure_ascii=False))
    print("Vencedor base:", winner)

    base_fit = data[data.date < d65].copy()
    base = clone(candidates()[winner])
    base.fit(base_fit[features], base_fit.actual_over25)
    cal_kind, _, cal_comparison = choose_calibrator(base, cal_fit, cal_select, features)
    print("Calibradores:", json.dumps(cal_comparison, ensure_ascii=False))
    print("Vencedor calibração:", cal_kind)

    selected_row = next(row for row in cal_comparison if row["calibrator"] == cal_kind)
    calibration_ece = float(selected_row["ece"])
    calibration_bias = abs(float(selected_row["calibration_bias"]))
    safety = float(np.clip(max(0.03, 1.5 * calibration_ece + calibration_bias), 0.03, 0.08))
    policy = {"min_edge":round(safety,4), "min_ev":round(max(.03,safety),4), "min_odd":1.20, "max_odd":4.00}
    print(f"Policy sem otimizar ROI: {policy}")

    cal_all = market_only(data[(data.date >= d65) & (data.date < d80)])
    raw_cal_all = probabilities(base, cal_all[features])
    calibrator = fit_calibrator(cal_kind, raw_cal_all, cal_all)

    raw_policy = probabilities(base, policy_set[features])
    p_policy = apply_calibrator(cal_kind, calibrator, raw_policy, policy_set)
    policy_prob = probability_metrics(policy_set.actual_over25, p_policy)
    policy_bets = betting_metrics(policy_set, p_policy, policy)

    raw_test_sport = probabilities(base, test_sport[features])
    raw_test = probabilities(base, test[features])
    p_test = apply_calibrator(cal_kind, calibrator, raw_test, test)
    sport_test_prob = probability_metrics(test_sport.actual_over25, raw_test_sport)
    test_prob = probability_metrics(test.actual_over25, p_test)
    test_bets = betting_metrics(test, p_test, policy)
    test_ece = ece(test.actual_over25, p_test)
    print("TEST sports raw:", json.dumps(sport_test_prob, ensure_ascii=False))
    print("TEST calibrated:", json.dumps(test_prob, ensure_ascii=False))
    print("TEST betting:", json.dumps(test_bets, ensure_ascii=False))
    print(f"TEST ECE={test_ece:.4f}")

    # Production refit only after final-test results above are fixed.
    prod_base_train = data[data.date < d90].copy()
    prod_cal = market_only(data[data.date >= d90])
    prod_base = clone(candidates()[winner])
    prod_base.fit(prod_base_train[features], prod_base_train.actual_over25)
    prod_raw = probabilities(prod_base, prod_cal[features])
    prod_calibrator = fit_calibrator(cal_kind, prod_raw, prod_cal)

    generated_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "version":VERSION, "sport_feature_names":features, "base_model":prod_base,
        "calibration_kind":cal_kind, "calibrator":prod_calibrator,
        "policy":policy, "winner_model":winner, "generated_at":generated_at,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True); META_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    calibration_table(test.actual_over25, p_test).to_csv(REPORT_PATH, index=False)

    metadata = {
        "version":VERSION, "generated_at":generated_at,
        "raw_matches":int(len(matches)), "sport_eligible_matches":int(len(data)), "market_eligible_matches":int(data.market_eligible.sum()),
        "sport_feature_count":len(features), "sport_features":features,
        "date_min":data.date.min().date().isoformat(), "date_max":data.date.max().date().isoformat(),
        "split_dates":{"train_end":d55.date().isoformat(),"base_selection_end":d65.date().isoformat(),"calibration_fit_end":d75.date().isoformat(),"calibration_selection_end":d80.date().isoformat(),"policy_end":d90.date().isoformat()},
        "split_sizes":{"train":len(train),"selection":len(select),"calibration_fit":len(cal_fit),"calibration_selection":len(cal_select),"policy":len(policy_set),"test_sport":len(test_sport),"test_market":len(test)},
        "base_model_comparison":base_comparison, "winner_model":winner,
        "calibrator_comparison":cal_comparison, "calibration_kind":cal_kind,
        "policy":policy, "policy_set_probability_metrics":policy_prob, "policy_set_betting_metrics":policy_bets,
        "test_sport_raw_metrics":sport_test_prob, "test_probability_metrics":test_prob, "test_ece":test_ece,
        "test_betting_metrics":test_bets,
        "methodology":"football base uses all eligible historical matches regardless of odds; 55/10/10/5/10/10 chronological unique-date blocks; base model selected before calibration; calibrator fitted then selected on later data; policy buffer from out-of-fit calibration error, never optimized on ROI; final 10% untouched until evaluation; same-day-safe Elo and league context",
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Modelo V12 gerado: {MODEL_PATH.stat().st_size/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
