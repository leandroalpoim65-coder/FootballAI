from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score


MIN_ROWS = 300
MIN_TRAIN_ROWS = 180
MIN_VALIDATION_ROWS = 80
MIN_AUC_LIFT = 0.003
MIN_BRIER_IMPROVEMENT = 0.001
RANDOM_STATE = 42


@dataclass(frozen=True)
class CandidateFeature:
    name: str
    requires: tuple[str, ...]
    builder: Callable[[pd.DataFrame], pd.Series]
    description: str


@dataclass(frozen=True)
class FeatureResult:
    feature: str
    status: str
    train_rows: int
    validation_rows: int
    baseline_auc: float | None
    candidate_auc: float | None
    auc_lift: float | None
    baseline_brier: float | None
    candidate_brier: float | None
    brier_improvement: float | None
    reason: str


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(a, errors="coerce") / denominator


def candidate_library() -> tuple[CandidateFeature, ...]:
    return (
        CandidateFeature(
            "market_model_gap",
            ("sports_probability", "market_probability"),
            lambda d: d["sports_probability"] - d["market_probability"],
            "Diferença entre probabilidade do modelo desportivo e mercado.",
        ),
        CandidateFeature(
            "clv_market_gap",
            ("clv_probability", "market_probability"),
            lambda d: d["clv_probability"] - d["market_probability"],
            "Diferença entre probabilidade CLV e probabilidade justa de mercado.",
        ),
        CandidateFeature(
            "model_clv_agreement",
            ("sports_probability", "clv_probability"),
            lambda d: 1.0 - (d["sports_probability"] - d["clv_probability"]).abs(),
            "Grau de concordância entre modelo desportivo e modelo CLV.",
        ),
        CandidateFeature(
            "value_pressure",
            ("sports_probability", "odd_over25_open"),
            lambda d: d["sports_probability"] * d["odd_over25_open"] - 1.0,
            "EV contínuo do modelo desportivo à odd de abertura.",
        ),
        CandidateFeature(
            "price_vs_fair",
            ("odd_over25_open", "market_probability"),
            lambda d: safe_div(1.0 / d["odd_over25_open"], d["market_probability"]),
            "Relação entre probabilidade implícita da odd e probabilidade justa do mercado.",
        ),
    )


def temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = data.sort_values(["fixture_date", "paper_trade_id"]).reset_index(drop=True)
    if len(ordered) < MIN_ROWS:
        return ordered, ordered.iloc[0:0].copy()
    validation_n = max(MIN_VALIDATION_ROWS, int(round(len(ordered) * 0.30)))
    validation_n = min(validation_n, len(ordered) - MIN_TRAIN_ROWS)
    if validation_n < MIN_VALIDATION_ROWS:
        return ordered, ordered.iloc[0:0].copy()
    return ordered.iloc[:-validation_n].copy(), ordered.iloc[-validation_n:].copy()


def _metrics(y: pd.Series, p: np.ndarray) -> tuple[float | None, float | None]:
    auc = float(roc_auc_score(y, p)) if y.nunique() >= 2 else None
    brier = float(brier_score_loss(y, p))
    return auc, brier


def evaluate_feature(
    data: pd.DataFrame,
    base_features: list[str],
    candidate: CandidateFeature,
    target: str = "result_over25",
) -> FeatureResult:
    required = set(base_features) | set(candidate.requires) | {target, "fixture_date", "paper_trade_id"}
    missing = sorted(required - set(data.columns))
    if missing:
        return FeatureResult(candidate.name, "UNAVAILABLE", 0, 0, None, None, None, None, None, None, "Faltam colunas: " + ", ".join(missing))

    frame = data[list(required)].copy()
    for column in set(base_features) | set(candidate.requires) | {target}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[candidate.name] = candidate.builder(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame[target].isin([0, 1])].copy()

    train, validation = temporal_split(frame)
    if len(validation) < MIN_VALIDATION_ROWS:
        return FeatureResult(candidate.name, "WAIT", len(train), len(validation), None, None, None, None, None, None, f"Amostra insuficiente ({len(frame)}/{MIN_ROWS}).")

    y_train = train[target].astype(int)
    y_val = validation[target].astype(int)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return FeatureResult(candidate.name, "WAIT", len(train), len(validation), None, None, None, None, None, None, "Treino/validação sem as duas classes.")

    def model():
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=8,
            min_samples_split=15,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    baseline = model()
    baseline.fit(train[base_features], y_train)
    baseline_p = baseline.predict_proba(validation[base_features])[:, list(baseline.classes_).index(1)]
    baseline_auc, baseline_brier = _metrics(y_val, baseline_p)

    extended = [*base_features, candidate.name]
    challenger = model()
    challenger.fit(train[extended], y_train)
    challenger_p = challenger.predict_proba(validation[extended])[:, list(challenger.classes_).index(1)]
    candidate_auc, candidate_brier = _metrics(y_val, challenger_p)

    auc_lift = None if baseline_auc is None or candidate_auc is None else candidate_auc - baseline_auc
    brier_improvement = baseline_brier - candidate_brier

    promoted = (
        auc_lift is not None
        and auc_lift >= MIN_AUC_LIFT
        and brier_improvement >= MIN_BRIER_IMPROVEMENT
    )
    status = "CHALLENGER_PASS" if promoted else "REJECT"
    reason = (
        "Melhorou AUC e Brier na janela temporal posterior."
        if promoted
        else "Não confirmou melhoria simultânea de AUC e Brier fora da amostra."
    )

    return FeatureResult(
        candidate.name, status, len(train), len(validation),
        baseline_auc, candidate_auc, auc_lift,
        baseline_brier, candidate_brier, brier_improvement, reason,
    )
