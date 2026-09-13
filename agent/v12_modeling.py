from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agent.v12_calibrated import market_blend_matrix

RANDOM_STATE = 42


def candidates() -> dict[str, object]:
    return {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350, max_depth=14, min_samples_leaf=10,
            min_samples_split=20, max_features=0.70, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def probabilities(model, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    classes = list(model.classes_)
    return np.asarray(p[:, classes.index(1)], dtype=float)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def fit_calibrator(kind: str, raw: np.ndarray, frame: pd.DataFrame):
    y = frame.actual_over25.astype(int)
    if kind == "sigmoid":
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
        model.fit(logit(raw), y)
        return model
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        model.fit(raw, y)
        return model
    if kind == "market_blend":
        X = market_blend_matrix(raw, frame.market_probability.to_numpy(float), frame.market_overround.to_numpy(float))
        model = LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_STATE)
        model.fit(X, y)
        return model
    raise ValueError(f"Calibrador V12 desconhecido: {kind}")


def apply_calibrator(kind: str, calibrator, raw: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    if kind == "sigmoid":
        p = calibrator.predict_proba(logit(raw))
        return np.asarray(p[:, list(calibrator.classes_).index(1)], dtype=float)
    if kind == "isotonic":
        return np.asarray(calibrator.predict(raw), dtype=float)
    if kind == "market_blend":
        X = market_blend_matrix(raw, frame.market_probability.to_numpy(float), frame.market_overround.to_numpy(float))
        p = calibrator.predict_proba(X)
        return np.asarray(p[:, list(calibrator.classes_).index(1)], dtype=float)
    return np.asarray(raw, dtype=float)


def probability_metrics(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    yv = y.astype(int).to_numpy()
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "n": int(len(yv)), "over_rate": float(yv.mean()), "mean_probability": float(p.mean()),
        "calibration_bias": float(p.mean() - yv.mean()), "brier": float(brier_score_loss(yv, p)),
        "log_loss": float(log_loss(yv, p, labels=[0, 1])),
        "auc": float(roc_auc_score(yv, p)) if len(np.unique(yv)) > 1 else 0.5,
    }


def calibration_table(y: pd.Series, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y.astype(int).to_numpy(), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.cut(frame.p, bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    table = frame.groupby("bin", observed=False).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    table["abs_gap"] = (table.predicted - table.actual).abs()
    return table


def ece(y: pd.Series, p: np.ndarray) -> float:
    table = calibration_table(y, p)
    total = max(int(table.n.sum()), 1)
    return float(((table.n / total) * table.abs_gap.fillna(0.0)).sum())


def betting_metrics(frame: pd.DataFrame, p: np.ndarray, policy: dict[str, float]) -> dict[str, float]:
    odds = frame.odd_over25_open.to_numpy(dtype=float); market = frame.market_probability.to_numpy(dtype=float)
    y = frame.actual_over25.to_numpy(dtype=int); edge = np.asarray(p, dtype=float) - market
    ev = np.asarray(p, dtype=float) * odds - 1.0
    mask = ((odds >= policy["min_odd"]) & (odds <= policy["max_odd"]) & (edge >= policy["min_edge"]) & (ev >= policy["min_ev"]))
    n = int(mask.sum())
    if n == 0:
        return {"bets":0,"wins":0,"hit_rate":0.0,"profit":0.0,"roi":0.0,"avg_odd":0.0,"max_drawdown":0.0,"mean_edge":0.0}
    profits = np.where(y[mask] == 1, odds[mask] - 1.0, -1.0)
    curve = np.cumsum(profits); peaks = np.maximum.accumulate(np.r_[0.0, curve])[:len(curve)]
    max_dd = float(np.max(peaks - curve)) if len(curve) else 0.0
    return {"bets":n,"wins":int(y[mask].sum()),"hit_rate":float(y[mask].mean()),"profit":float(profits.sum()),"roi":float(profits.mean()),"avg_odd":float(odds[mask].mean()),"max_drawdown":max_dd,"mean_edge":float(edge[mask].mean())}


def edge_distribution(frame: pd.DataFrame, p: np.ndarray) -> dict[str, float]:
    edge = np.asarray(p, dtype=float) - frame.market_probability.to_numpy(dtype=float)
    return {
        "mean": float(np.mean(edge)),
        "p50": float(np.quantile(edge, 0.50)),
        "p75": float(np.quantile(edge, 0.75)),
        "p90": float(np.quantile(edge, 0.90)),
        "p95": float(np.quantile(edge, 0.95)),
        "max": float(np.max(edge)),
    }


def choose_base(train: pd.DataFrame, select: pd.DataFrame, features: list[str]):
    rows = []
    for name, candidate in candidates().items():
        model = clone(candidate); model.fit(train[features], train.actual_over25)
        rows.append({"model":name, **probability_metrics(select.actual_over25, probabilities(model, select[features]))})
    rows.sort(key=lambda r: (r["brier"], r["log_loss"], -r["auc"]))
    return str(rows[0]["model"]), rows


def choose_calibrator(
    base,
    fit_frame: pd.DataFrame,
    select_frame: pd.DataFrame,
    features: list[str],
    kinds: tuple[str, ...] = ("sigmoid", "isotonic", "market_blend"),
):
    raw_fit = probabilities(base, fit_frame[features]); raw_select = probabilities(base, select_frame[features])
    rows = []
    fitted = {}
    for kind in kinds:
        calibrator = fit_calibrator(kind, raw_fit, fit_frame)
        p = apply_calibrator(kind, calibrator, raw_select, select_frame)
        row = {"calibrator":kind, **probability_metrics(select_frame.actual_over25, p), "ece":ece(select_frame.actual_over25, p)}
        rows.append(row); fitted[kind] = calibrator
    rows.sort(key=lambda r: (r["brier"], r["log_loss"], r["ece"]))
    winner = str(rows[0]["calibrator"])
    return winner, fitted[winner], rows
