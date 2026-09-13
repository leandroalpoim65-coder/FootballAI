from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agent.v12_calibrated import V12_SPORT_FEATURES
from agent.v12_modeling import ece, probability_metrics
from agent.v12_rich import RICH_FEATURES, build_rich_features
from agent_v12_rich_lab_cloud import date_cuts, load_matches, market_only
from agent.v12_dataset import build_dataset


C_GRID = (0.001, 0.003, 0.01, 0.03, 0.10, 0.30)


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def make_model(c: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=float(c), max_iter=4000, random_state=42)),
    ])


def predict(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    p = model.predict_proba(frame[features])
    classes = list(model.named_steps["model"].classes_)
    return np.asarray(p[:, classes.index(1)], dtype=float)


def metric_delta(model_metrics: dict, market_metrics: dict) -> dict[str, float]:
    return {
        "brier_gain": float(market_metrics["brier"] - model_metrics["brier"]),
        "logloss_gain": float(market_metrics["log_loss"] - model_metrics["log_loss"]),
        "auc_gain": float(model_metrics["auc"] - market_metrics["auc"]),
    }


def tune_family(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    label: str,
) -> dict:
    market_metrics = probability_metrics(
        validation.actual_over25,
        validation.market_probability.to_numpy(float),
    )
    rows = []
    for c in C_GRID:
        model = make_model(c)
        model.fit(train[features], train.actual_over25)
        p = predict(model, validation, features)
        metrics = probability_metrics(validation.actual_over25, p)
        rows.append({
            "label": label,
            "C": c,
            **metrics,
            "ece": ece(validation.actual_over25, p),
            **metric_delta(metrics, market_metrics),
        })
    rows.sort(key=lambda row: (row["brier"], row["log_loss"], -row["auc"]))
    return {"winner": rows[0], "grid": rows}


def evaluate_fixed(
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    features: list[str],
    c: float,
) -> dict:
    model = make_model(c)
    model.fit(fit_frame[features], fit_frame.actual_over25)
    p = predict(model, eval_frame, features)
    model_metrics = probability_metrics(eval_frame.actual_over25, p)
    market_metrics = probability_metrics(
        eval_frame.actual_over25,
        eval_frame.market_probability.to_numpy(float),
    )
    return {
        "n": len(eval_frame),
        "market": market_metrics,
        "model": model_metrics,
        "market_ece": ece(eval_frame.actual_over25, eval_frame.market_probability.to_numpy(float)),
        "model_ece": ece(eval_frame.actual_over25, p),
        "delta": metric_delta(model_metrics, market_metrics),
    }


def main() -> None:
    print("=" * 78)
    print("V12 MARKET CORRECTION LAB | mercado puro vs correção com futebol")
    print("=" * 78)

    matches = load_matches()
    base = build_dataset(matches)
    rich = build_rich_features(matches)
    data = base.set_index("id").join(rich, how="left").reset_index()
    data = data.dropna(subset=list(RICH_FEATURES)).copy()
    data = market_only(data)
    data["market_logit"] = logit(data.market_probability)

    cuts = date_cuts(data)
    _, _, d75, d80, d90 = cuts

    train = data[data.date < d75].copy()
    validation = data[(data.date >= d75) & (data.date < d80)].copy()
    policy = data[(data.date >= d80) & (data.date < d90)].copy()
    test = data[data.date >= d90].copy()

    families = {
        "MARKET_RECAL": ["market_logit", "market_overround"],
        "MARKET_PLUS_BASE": ["market_logit", "market_overround", *V12_SPORT_FEATURES],
        "MARKET_PLUS_RICH": ["market_logit", "market_overround", *V12_SPORT_FEATURES, *RICH_FEATURES],
    }

    print(
        f"Rows={len(data)} train={len(train)} validation={len(validation)} "
        f"policy={len(policy)} test={len(test)}"
    )

    tuned = {}
    for label, features in families.items():
        result = tune_family(train, validation, list(features), label)
        tuned[label] = result
        print(f"CORRECTION VALIDATION {label}: {json.dumps(result, ensure_ascii=False)}")

    # Family selection is fixed on validation only. Test is not used to choose.
    family_order = sorted(
        families,
        key=lambda label: (
            tuned[label]["winner"]["brier"],
            tuned[label]["winner"]["log_loss"],
            -tuned[label]["winner"]["auc"],
        ),
    )
    winner_label = family_order[0]
    winner_c = float(tuned[winner_label]["winner"]["C"])
    winner_features = list(families[winner_label])
    print(f"CORRECTION SELECTED ON VALIDATION: {winner_label} C={winner_c}")

    fit_policy = data[data.date < d80].copy()
    policy_result = evaluate_fixed(fit_policy, policy, winner_features, winner_c)
    print("CORRECTION POLICY CONFIRMATION:", json.dumps(policy_result, ensure_ascii=False))

    # Final historical block is reported only after family+C were frozen above.
    fit_test = data[data.date < d90].copy()
    test_result = evaluate_fixed(fit_test, test, winner_features, winner_c)
    print("CORRECTION FINAL TEST:", json.dumps(test_result, ensure_ascii=False))

    accepted = (
        policy_result["delta"]["brier_gain"] > 0
        and policy_result["delta"]["logloss_gain"] > 0
        and test_result["delta"]["brier_gain"] > 0
        and test_result["delta"]["logloss_gain"] > 0
    )
    print("CORRECTION HISTORICAL_GATE:", json.dumps({
        "accepted": bool(accepted),
        "winner": winner_label,
        "C": winner_c,
        "note": "Historical gate only; prospective shadow is still required before any promotion.",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
