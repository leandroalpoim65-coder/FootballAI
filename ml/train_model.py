from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "dados_preparados.csv"
)

REPORTS_DIR = ROOT_DIR / "reports"
ML_DIR = ROOT_DIR / "ml"

FEATURES_BTTS_PATH = ML_DIR / "features_btts.pkl"
FEATURES_OVER25_PATH = ML_DIR / "features_over25.pkl"

MODEL_BTTS_PATH = ML_DIR / "modelo_btts.pkl"
MODEL_OVER25_PATH = ML_DIR / "modelo_over25.pkl"

VALIDATION_BTTS_PATH = (
    ML_DIR / "modelo_btts_validacao.pkl"
)

VALIDATION_OVER25_PATH = (
    ML_DIR / "modelo_over25_validacao.pkl"
)

METADATA_BTTS_PATH = (
    ML_DIR / "metadata_modelo_btts.json"
)

METADATA_OVER25_PATH = (
    ML_DIR / "metadata_modelo_over25.json"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RANDOM_STATE = 42
CALIBRATION_SPLITS = 3

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ML_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def parse_dates(series: pd.Series) -> pd.Series:
    """
    Converte as datas processadas no formato YYYY-MM-DD.
    """
    return pd.to_datetime(
        series,
        format="%Y-%m-%d",
        errors="coerce",
    )


def get_positive_probability(
    model,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Obtém a probabilidade da classe positiva, isto é, classe 1.
    """
    probabilities = model.predict_proba(features)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "O modelo não contém a classe positiva 1."
        )

    positive_index = classes.index(1)

    return probabilities[:, positive_index]


def calculate_metrics(
    target: pd.Series,
    probabilities: np.ndarray,
) -> dict:
    """
    Calcula métricas de classificação e calibração.
    """
    target = target.astype(int)

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    if target.nunique() >= 2:
        auc = roc_auc_score(
            target,
            probabilities,
        )

        loss = log_loss(
            target,
            probabilities,
            labels=[0, 1],
        )

    else:
        auc = np.nan
        loss = np.nan

    actual_rate = float(target.mean())
    predicted_rate = float(
        probabilities.mean()
    )

    return {
        "accuracy": float(
            accuracy_score(
                target,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(auc),
        "brier_score": float(
            brier_score_loss(
                target,
                probabilities,
            )
        ),
        "log_loss": float(loss),
        "taxa_real": actual_rate,
        "probabilidade_media": predicted_rate,
        "erro_calibracao": (
            predicted_rate - actual_rate
        ),
    }


def create_candidates() -> dict:
    """
    Modelos testados sem dependências externas.
    """
    logistic_regression = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    C=0.50,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "Logistic Regression":
            logistic_regression,
        "Random Forest":
            random_forest,
        "Extra Trees":
            extra_trees,
    }


def create_calibrated_model(base_model):
    """
    Cria um modelo calibrado utilizando divisões temporais.

    O método sigmoid reduz probabilidades excessivamente
    confiantes sem utilizar dados da época de teste.
    """
    temporal_cv = TimeSeriesSplit(
        n_splits=CALIBRATION_SPLITS,
    )

    try:
        return CalibratedClassifierCV(
            estimator=clone(base_model),
            method="sigmoid",
            cv=temporal_cv,
            n_jobs=-1,
        )

    except TypeError:
        # Compatibilidade com versões antigas do sklearn.
        return CalibratedClassifierCV(
            base_estimator=clone(base_model),
            method="sigmoid",
            cv=temporal_cv,
            n_jobs=-1,
        )


def print_metrics(metrics: dict) -> None:
    """
    Mostra as métricas de forma legível.
    """
    print(
        f"Accuracy             : "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Precision            : "
        f"{metrics['precision']:.2%}"
    )

    print(
        f"Recall               : "
        f"{metrics['recall']:.2%}"
    )

    print(
        f"F1-score             : "
        f"{metrics['f1']:.2%}"
    )

    if np.isnan(metrics["roc_auc"]):
        print("ROC-AUC              : indisponível")
    else:
        print(
            f"ROC-AUC              : "
            f"{metrics['roc_auc']:.4f}"
        )

    print(
        f"Brier Score          : "
        f"{metrics['brier_score']:.4f}"
    )

    if np.isnan(metrics["log_loss"]):
        print("Log Loss             : indisponível")
    else:
        print(
            f"Log Loss             : "
            f"{metrics['log_loss']:.4f}"
        )

    print(
        f"Taxa real            : "
        f"{metrics['taxa_real']:.2%}"
    )

    print(
        f"Probabilidade média  : "
        f"{metrics['probabilidade_media']:.2%}"
    )

    print(
        f"Erro de calibração   : "
        f"{metrics['erro_calibracao']:+.2%}"
    )


# ============================================================
# TREINAR UM MERCADO
# ============================================================

def train_target(
    dataframe: pd.DataFrame,
    target_column: str,
    features_path: Path,
    production_model_path: Path,
    validation_model_path: Path,
    metadata_path: Path,
) -> None:
    """
    Compara modelos, testa na época mais recente,
    calibra o vencedor e guarda o modelo final.
    """

    print("\n")
    print("=" * 60)
    print(
        f" TREINO DO MERCADO: "
        f"{target_column.upper()}"
    )
    print("=" * 60)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Lista de features não encontrada: "
            f"{features_path}"
        )

    features = joblib.load(
        features_path
    )

    required_columns = [
        "date",
        "season",
        target_column,
        *features,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltam colunas no dataset: "
            + ", ".join(missing_columns)
        )

    data = dataframe[
        required_columns
    ].copy()

    data["date"] = parse_dates(
        data["date"]
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce",
    )

    for feature in features:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before_cleaning = len(data)

    data = data.dropna(
        subset=[
            "date",
            "season",
            target_column,
            *features,
        ]
    ).copy()

    removed = before_cleaning - len(data)

    data[target_column] = (
        data[target_column].astype(int)
    )

    data = data[
        data[target_column].isin([0, 1])
    ].copy()

    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            f"Não existem dados válidos para "
            f"{target_column}."
        )

    season_order = (
        data.groupby("season")["date"]
        .max()
        .sort_values()
    )

    if len(season_order) < 2:
        raise ValueError(
            "São necessárias pelo menos duas épocas."
        )

    test_season = season_order.index[-1]

    train = data[
        data["season"] != test_season
    ].copy()

    test = data[
        data["season"] == test_season
    ].copy()

    if train.empty:
        raise ValueError(
            "O conjunto de treino ficou vazio."
        )

    if test.empty:
        raise ValueError(
            "O conjunto de teste ficou vazio."
        )

    if train[target_column].nunique() < 2:
        raise ValueError(
            "O treino não contém as duas classes."
        )

    if test[target_column].nunique() < 2:
        raise ValueError(
            "A época de teste não contém "
            "as duas classes."
        )

    X_train = train[features]
    y_train = train[target_column]

    X_test = test[features]
    y_test = test[target_column]

    print("\nDados utilizados:")
    print(f"Linhas removidas     : {removed}")
    print(f"Features             : {len(features)}")

    print(
        f"Treino               : "
        f"{len(train)} jogos"
    )

    print(
        f"Período de treino    : "
        f"{train['date'].min().date()} "
        f"a {train['date'].max().date()}"
    )

    print(
        f"Época de teste       : "
        f"{test_season}"
    )

    print(
        f"Teste                : "
        f"{len(test)} jogos"
    )

    print(
        f"Período de teste     : "
        f"{test['date'].min().date()} "
        f"a {test['date'].max().date()}"
    )

    print(
        f"Taxa positiva treino : "
        f"{y_train.mean():.2%}"
    )

    print(
        f"Taxa positiva teste  : "
        f"{y_test.mean():.2%}"
    )

    # ========================================================
    # COMPARAÇÃO DOS MODELOS
    # ========================================================

    candidates = create_candidates()

    comparison_rows = []
    trained_candidates = {}

    print("\nComparação de modelos:")

    for model_name, candidate in (
        candidates.items()
    ):
        print(
            f"\nA treinar: {model_name}..."
        )

        model = clone(candidate)

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            get_positive_probability(
                model,
                X_test,
            )
        )

        metrics = calculate_metrics(
            y_test,
            probabilities,
        )

        comparison_rows.append(
            {
                "modelo": model_name,
                "calibrado": False,
                **metrics,
            }
        )

        trained_candidates[
            model_name
        ] = model

        print_metrics(metrics)

    comparison = pd.DataFrame(
        comparison_rows
    )

    selection_table = comparison.copy()

    selection_table[
        "_auc_selection"
    ] = selection_table[
        "roc_auc"
    ].fillna(-1)

    selection_table = (
        selection_table.sort_values(
            by=[
                "_auc_selection",
                "brier_score",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    best_model_name = (
        selection_table.iloc[0]["modelo"]
    )

    best_base_model = candidates[
        best_model_name
    ]

    print("\n" + "-" * 60)
    print(
        f"Melhor modelo base: "
        f"{best_model_name}"
    )
    print("-" * 60)

    # ========================================================
    # CALIBRAÇÃO TEMPORAL DO VENCEDOR
    # ========================================================

    print(
        "\nA calibrar probabilidades "
        "com validação temporal..."
    )

    validation_model = (
        create_calibrated_model(
            best_base_model
        )
    )

    validation_model.fit(
        X_train,
        y_train,
    )

    calibrated_probabilities = (
        get_positive_probability(
            validation_model,
            X_test,
        )
    )

    calibrated_metrics = calculate_metrics(
        y_test,
        calibrated_probabilities,
    )

    print("\nResultado após calibração:")
    print_metrics(calibrated_metrics)

    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {
                        "modelo":
                            f"{best_model_name} calibrado",
                        "calibrado": True,
                        **calibrated_metrics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # ========================================================
    # GUARDAR MODELO DE VALIDAÇÃO
    # ========================================================

    joblib.dump(
        validation_model,
        validation_model_path,
    )

    # ========================================================
    # TREINAR MODELO FINAL COM TODOS OS DADOS
    # ========================================================

    print(
        "\nA treinar modelo final "
        "com todas as épocas..."
    )

    final_model = create_calibrated_model(
        best_base_model
    )

    final_model.fit(
        data[features],
        data[target_column],
    )

    joblib.dump(
        final_model,
        production_model_path,
    )

    # ========================================================
    # GUARDAR RELATÓRIOS
    # ========================================================

    comparison_path = (
        REPORTS_DIR
        / f"comparacao_modelos_{target_column}.csv"
    )

    predictions_path = (
        REPORTS_DIR
        / f"previsoes_validacao_{target_column}.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    validation_predictions = test[
        [
            "date",
            "season",
            target_column,
        ]
    ].copy()

    validation_predictions[
        "probabilidade"
    ] = calibrated_probabilities

    validation_predictions[
        "previsao"
    ] = (
        calibrated_probabilities >= 0.50
    ).astype(int)

    validation_predictions.to_csv(
        predictions_path,
        index=False,
    )

    metadata = {
        "generated_at":
            datetime.now().isoformat(),
        "target":
            target_column,
        "selected_model":
            best_model_name,
        "calibration":
            "sigmoid",
        "calibration_splits":
            CALIBRATION_SPLITS,
        "feature_count":
            len(features),
        "features":
            list(features),
        "training_rows":
            int(len(train)),
        "test_rows":
            int(len(test)),
        "total_rows":
            int(len(data)),
        "test_season":
            str(test_season),
        "training_date_min":
            train["date"].min().date().isoformat(),
        "training_date_max":
            train["date"].max().date().isoformat(),
        "test_date_min":
            test["date"].min().date().isoformat(),
        "test_date_max":
            test["date"].max().date().isoformat(),
        "validation_metrics":
            {
                key: (
                    None
                    if np.isnan(value)
                    else float(value)
                )
                for key, value
                in calibrated_metrics.items()
            },
        "production_model_trained_with_all_data":
            True,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFicheiros guardados:")

    print(
        f"✅ "
        f"{production_model_path.relative_to(ROOT_DIR)}"
    )

    print(
        f"✅ "
        f"{validation_model_path.relative_to(ROOT_DIR)}"
    )

    print(
        f"✅ "
        f"{comparison_path.relative_to(ROOT_DIR)}"
    )

    print(
        f"✅ "
        f"{predictions_path.relative_to(ROOT_DIR)}"
    )

    print(
        f"✅ "
        f"{metadata_path.relative_to(ROOT_DIR)}"
    )


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATA_PATH}"
        )

    dataframe = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print("\n================================")
    print(" FOOTBALLAI — TREINO DE MODELOS")
    print("================================\n")

    print(
        f"Jogos carregados: "
        f"{len(dataframe)}"
    )

    train_target(
        dataframe=dataframe,
        target_column="btts",
        features_path=FEATURES_BTTS_PATH,
        production_model_path=MODEL_BTTS_PATH,
        validation_model_path=VALIDATION_BTTS_PATH,
        metadata_path=METADATA_BTTS_PATH,
    )

    train_target(
        dataframe=dataframe,
        target_column="over25",
        features_path=FEATURES_OVER25_PATH,
        production_model_path=MODEL_OVER25_PATH,
        validation_model_path=VALIDATION_OVER25_PATH,
        metadata_path=METADATA_OVER25_PATH,
    )

    print("\n")
    print("=" * 60)
    print(" TREINO CONCLUÍDO")
    print("=" * 60)

    print(
        "\n✅ Modelos BTTS e Over 2.5 "
        "treinados e calibrados."
    )


if __name__ == "__main__":
    main()