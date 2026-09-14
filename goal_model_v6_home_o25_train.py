from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import goal_model_v2_multileague_research as v2
import goal_model_v4_structural_research as v4
import goal_model_v6_home_o25_research as v6


OUT = Path("reports")
MODEL_PATH = Path("ml") / "goal_model_v6_home_o25_trained.pkl"

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
MARKET = "HomeO2.5"
MODEL_KINDS = ["linear", "quadratic"]
C_VALUES = [0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0, 30.0, 100.0]
EPS = 1e-9


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def feature_matrix(p2: np.ndarray, p5: np.ndarray, kind: str) -> np.ndarray:
    z2 = logit(p2)
    z5 = logit(p5)
    if kind == "linear":
        return np.column_stack([z2, z5])
    if kind == "quadratic":
        return np.column_stack([
            z2,
            z5,
            z2 * z5,
            z2 * z2,
            z5 * z5,
        ])
    raise ValueError(f"Unknown model kind: {kind}")


def feature_names(kind: str) -> list[str]:
    if kind == "linear":
        return ["logit_v2", "logit_v5"]
    if kind == "quadratic":
        return [
            "logit_v2",
            "logit_v5",
            "logit_v2_x_logit_v5",
            "logit_v2_sq",
            "logit_v5_sq",
        ]
    raise ValueError(kind)


def make_model(c_value: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "logit",
            LogisticRegression(
                C=float(c_value),
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                class_weight=None,
            ),
        ),
    ])


def season_arrays(data: list[dict]) -> dict[str, np.ndarray]:
    y, p2, p5 = v6.base_and_v5(data)
    p6_frozen = v6.specialist_probability(
        p2,
        p5,
        v5_weight=0.75,
        slope=1.05,
        intercept=0.0,
    )
    return {
        "y": np.asarray(y, dtype=float),
        "p2": np.asarray(p2, dtype=float),
        "p5": np.asarray(p5, dtype=float),
        "p6_frozen": np.asarray(p6_frozen, dtype=float),
    }


def concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=0)


def score_prediction(y: np.ndarray, pred: np.ndarray, p2: np.ndarray, p5: np.ndarray, p6f: np.ndarray) -> dict:
    ll = v2.logloss(y, pred)
    ll2 = v2.logloss(y, p2)
    ll5 = v2.logloss(y, p5)
    ll6f = v2.logloss(y, p6f)
    best = min(ll2, ll5, ll6f)
    br = v2.brier(y, pred)
    return {
        "logloss": ll,
        "brier": br,
        "ece": v2.ece(y, pred),
        "relative_v2": ll / ll2,
        "relative_v5": ll / ll5,
        "relative_v6_frozen": ll / ll6f,
        "relative_best": ll / best,
        "v2_logloss": ll2,
        "v5_logloss": ll5,
        "v6_frozen_logloss": ll6f,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    by_season, failures = v4.build_data(SEASONS)
    missing = [s for s in SEASONS if not by_season.get(s)]
    if missing:
        raise RuntimeError(f"Missing seasons {missing}. Failures: {failures}")

    arrays = {season: season_arrays(by_season[season]) for season in SEASONS}

    # Expanding-window CV. Each validation season is later than every season used to fit it.
    folds = [
        (["2122"], "2223"),
        (["2122", "2223"], "2324"),
        (["2122", "2223", "2324"], "2425"),
        (["2122", "2223", "2324", "2425"], "2526"),
    ]

    grid_rows: list[dict] = []
    fold_rows: list[dict] = []

    for kind in MODEL_KINDS:
        for c_value in C_VALUES:
            rel_best_values = []
            candidate_fold_rows = []

            for train_seasons, val_season in folds:
                y_train = concat([arrays[s]["y"] for s in train_seasons])
                p2_train = concat([arrays[s]["p2"] for s in train_seasons])
                p5_train = concat([arrays[s]["p5"] for s in train_seasons])

                model = make_model(c_value)
                model.fit(feature_matrix(p2_train, p5_train, kind), y_train)

                val = arrays[val_season]
                pred = model.predict_proba(
                    feature_matrix(val["p2"], val["p5"], kind)
                )[:, 1]
                score = score_prediction(
                    val["y"], pred, val["p2"], val["p5"], val["p6_frozen"]
                )
                rel_best_values.append(score["relative_best"])
                candidate_fold_rows.append({
                    "kind": kind,
                    "C": float(c_value),
                    "train_seasons": "+".join(train_seasons),
                    "validation_season": val_season,
                    "n_train": int(len(y_train)),
                    "n_validation": int(len(val["y"])),
                    **score,
                })

            mean_rel = float(np.mean(rel_best_values))
            std_rel = float(np.std(rel_best_values))
            worst_rel = float(max(rel_best_values))
            stability = mean_rel + 0.25 * std_rel + 0.50 * max(0.0, worst_rel - 1.0)

            grid_rows.append({
                "kind": kind,
                "C": float(c_value),
                "mean_relative_best": mean_rel,
                "std_relative_best": std_rel,
                "worst_relative_best": worst_rel,
                "stability_score": stability,
                "folds_beating_best": int(sum(x < 1.0 for x in rel_best_values)),
                "folds_total": len(rel_best_values),
            })
            fold_rows.extend(candidate_fold_rows)

    grid_rows.sort(key=lambda r: (
        r["stability_score"],
        r["mean_relative_best"],
        r["worst_relative_best"],
    ))
    best = grid_rows[0]
    best_kind = str(best["kind"])
    best_c = float(best["C"])

    # Rebuild the honest expanding-window predictions for the selected architecture.
    selected_oof = []
    y_oof_parts = []
    p_oof_parts = []
    p2_oof_parts = []
    p5_oof_parts = []
    p6f_oof_parts = []

    for train_seasons, val_season in folds:
        y_train = concat([arrays[s]["y"] for s in train_seasons])
        p2_train = concat([arrays[s]["p2"] for s in train_seasons])
        p5_train = concat([arrays[s]["p5"] for s in train_seasons])
        model = make_model(best_c)
        model.fit(feature_matrix(p2_train, p5_train, best_kind), y_train)

        val = arrays[val_season]
        pred = model.predict_proba(
            feature_matrix(val["p2"], val["p5"], best_kind)
        )[:, 1]
        score = score_prediction(
            val["y"], pred, val["p2"], val["p5"], val["p6_frozen"]
        )
        selected_oof.append({
            "validation_season": val_season,
            "train_seasons": "+".join(train_seasons),
            "n": int(len(val["y"])),
            **score,
        })
        y_oof_parts.append(val["y"])
        p_oof_parts.append(pred)
        p2_oof_parts.append(val["p2"])
        p5_oof_parts.append(val["p5"])
        p6f_oof_parts.append(val["p6_frozen"])

    y_oof = concat(y_oof_parts)
    pred_oof = concat(p_oof_parts)
    p2_oof = concat(p2_oof_parts)
    p5_oof = concat(p5_oof_parts)
    p6f_oof = concat(p6f_oof_parts)
    oof_score = score_prediction(y_oof, pred_oof, p2_oof, p5_oof, p6f_oof)

    # Final fit uses every completed season through 2025/26. From this point onward,
    # only genuinely future matches can judge this trained model out of sample.
    y_all = concat([arrays[s]["y"] for s in SEASONS])
    p2_all = concat([arrays[s]["p2"] for s in SEASONS])
    p5_all = concat([arrays[s]["p5"] for s in SEASONS])
    final_model = make_model(best_c)
    final_model.fit(feature_matrix(p2_all, p5_all, best_kind), y_all)

    lr = final_model.named_steps["logit"]
    scaler = final_model.named_steps["scale"]
    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    artifact = {
        "model_version": "goal_model_v6_trained_home_o25_v1",
        "market": MARKET,
        "trained_at": trained_at,
        "trained_seasons": SEASONS,
        "feature_kind": best_kind,
        "feature_names": feature_names(best_kind),
        "C": best_c,
        "model": final_model,
        "v2_config": dict(v4.V2_CONFIG),
        "v5_home_config": dict(v6.V5_HOME_CONFIG),
        "v6_frozen_calibration": {
            "v5_weight": 0.75,
            "slope": 1.05,
            "intercept": 0.0,
        },
        "selection_method": "expanding_window_cv_2122_to_2526",
    }
    joblib.dump(artifact, MODEL_PATH)

    pd.DataFrame(grid_rows).to_csv(
        OUT / "goal_model_v6_trained_grid.csv", index=False
    )
    pd.DataFrame(selected_oof).to_csv(
        OUT / "goal_model_v6_trained_oof.csv", index=False
    )

    summary = {
        "model_version": artifact["model_version"],
        "market": MARKET,
        "trained_at": trained_at,
        "training_seasons": SEASONS,
        "selection_rule": (
            "Select linear/quadratic logistic stack and L2 regularization C using expanding-window "
            "validation only; then refit once on all completed seasons 2021/22-2025/26."
        ),
        "chosen_kind": best_kind,
        "chosen_C": best_c,
        "feature_names": feature_names(best_kind),
        "grid_size": len(grid_rows),
        "cv_summary": best,
        "selected_fold_metrics": selected_oof,
        "combined_oof": oof_score,
        "final_fit_n": int(len(y_all)),
        "positive_rate": float(np.mean(y_all)),
        "standard_scaler_mean": [float(x) for x in scaler.mean_],
        "standard_scaler_scale": [float(x) for x in scaler.scale_],
        "logistic_intercept": [float(x) for x in lr.intercept_],
        "logistic_coefficients": [[float(x) for x in row] for row in lr.coef_],
        "artifact_path": str(MODEL_PATH),
        "prospective_only_from_now": True,
        "production_promotion_allowed": False,
        "production_note": (
            "The final trained model has consumed every completed season through 2025/26. "
            "Use only future frozen shadow predictions to evaluate promotion, with real team-total odds, ROI and CLV."
        ),
        "failures": failures,
    }
    (OUT / "goal_model_v6_trained_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== V6 TRAINED HOME O2.5 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nArtifact written to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
