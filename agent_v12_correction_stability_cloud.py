from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent.v12_calibrated import V12_SPORT_FEATURES
from agent.v12_dataset import build_dataset
from agent.v12_modeling import ece, probability_metrics
from agent_v12_market_correction_lab_cloud import logit, make_model, predict, metric_delta
from agent_v12_rich_lab_cloud import load_matches, market_only


C_FIXED = 0.003
FEATURES = ["market_logit", "market_overround", *V12_SPORT_FEATURES]


def main() -> None:
    print("=" * 78)
    print("V12 CORRECTION STABILITY | walk-forward anual, regra congelada")
    print("=" * 78)

    data = market_only(build_dataset(load_matches()))
    data["market_logit"] = logit(data.market_probability)
    data = data.sort_values(["date", "id"]).reset_index(drop=True)

    years = sorted(int(year) for year in data.date.dt.year.unique())
    rows = []
    for year in years:
        train = data[data.date.dt.year < year].copy()
        test = data[data.date.dt.year == year].copy()
        if len(train) < 5000 or len(test) < 300:
            continue

        model = make_model(C_FIXED)
        model.fit(train[FEATURES], train.actual_over25)
        p = predict(model, test, FEATURES)
        market_p = test.market_probability.to_numpy(float)

        market_metrics = probability_metrics(test.actual_over25, market_p)
        model_metrics = probability_metrics(test.actual_over25, p)
        delta = metric_delta(model_metrics, market_metrics)
        row = {
            "year": year,
            "train_n": len(train),
            "test_n": len(test),
            "market_brier": market_metrics["brier"],
            "model_brier": model_metrics["brier"],
            "brier_gain": delta["brier_gain"],
            "market_logloss": market_metrics["log_loss"],
            "model_logloss": model_metrics["log_loss"],
            "logloss_gain": delta["logloss_gain"],
            "auc_gain": delta["auc_gain"],
            "market_ece": ece(test.actual_over25, market_p),
            "model_ece": ece(test.actual_over25, p),
            "both_gain": bool(delta["brier_gain"] > 0 and delta["logloss_gain"] > 0),
        }
        rows.append(row)
        print("STABILITY YEAR:", json.dumps(row, ensure_ascii=False))

    if not rows:
        raise RuntimeError("Sem períodos walk-forward elegíveis.")

    frame = pd.DataFrame(rows)
    weighted_brier_gain = float(np.average(frame.brier_gain, weights=frame.test_n))
    weighted_logloss_gain = float(np.average(frame.logloss_gain, weights=frame.test_n))
    summary = {
        "periods": int(len(frame)),
        "periods_both_gain": int(frame.both_gain.sum()),
        "share_both_gain": float(frame.both_gain.mean()),
        "weighted_brier_gain": weighted_brier_gain,
        "weighted_logloss_gain": weighted_logloss_gain,
        "median_brier_gain": float(frame.brier_gain.median()),
        "median_logloss_gain": float(frame.logloss_gain.median()),
        "accepted_stability": bool(
            frame.both_gain.mean() >= 0.70
            and weighted_brier_gain > 0
            and weighted_logloss_gain > 0
            and frame.brier_gain.median() > 0
            and frame.logloss_gain.median() > 0
        ),
    }
    print("STABILITY SUMMARY:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
