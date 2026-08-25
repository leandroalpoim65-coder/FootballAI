from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


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

BASE_FEATURES_PATH = (
    ROOT_DIR
    / "ml"
    / "features_over25.pkl"
)

MODEL_PATH = (
    ROOT_DIR
    / "ml"
    / "modelo_clv_over25.pkl"
)

FEATURES_CLV_PATH = (
    ROOT_DIR
    / "ml"
    / "features_clv_over25.pkl"
)

METADATA_PATH = (
    ROOT_DIR
    / "ml"
    / "metadata_modelo_clv_over25.json"
)

TRAINING_REPORT_PATH = (
    ROOT_DIR
    / "reports"
    / "treino_modelo_clv_over25.csv"
)


# ============================================================
# CONFIGURAÇÃO CONGELADA
# ============================================================

TARGET_RESULT = "over25"
TARGET_CLV = "clv_positive"

ODD_OVER_OPEN = "odd_over25_open"
ODD_UNDER_OPEN = "odd_under25_open"

ODD_OVER_CLOSE = "odd_over25_close"
ODD_UNDER_CLOSE = "odd_under25_close"

# Estes valores vêm do backtest CLV.
# Não devem ser alterados durante o paper trading.
MIN_SIGNAL_PROBABILITY = 0.55
MIN_EDGE_OVER_BASE_RATE = 0.04

MIN_ODD = 1.20
MAX_ODD = 4.00

LOGISTIC_C = 0.10
RANDOM_STATE = 42


# ============================================================
# CRIAR PASTAS
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

TRAINING_REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FUNÇÕES
# ============================================================

def create_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Cria o mesmo modelo utilizado no backtest CLV.

    As features numéricas são normalizadas.
    A liga é transformada através de One-Hot Encoding.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        C=LOGISTIC_C,
        max_iter=5000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def get_positive_probability(
    model: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Obtém a probabilidade da classe positiva.
    """

    probabilities = model.predict_proba(
        features
    )

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
    Calcula métricas apenas como diagnóstico do treino.

    Estas métricas são in-sample e não substituem
    o backtest walk-forward.
    """

    y_true = target.astype(int).to_numpy()

    y_probability = np.asarray(
        probabilities,
        dtype=float,
    )

    y_probability = np.clip(
        y_probability,
        0.000001,
        0.999999,
    )

    y_prediction = (
        y_probability >= 0.50
    ).astype(int)

    if len(np.unique(y_true)) >= 2:
        roc_auc = roc_auc_score(
            y_true,
            y_probability,
        )
    else:
        roc_auc = np.nan

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_probability,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                y_probability,
                labels=[0, 1],
            )
        ),
        "taxa_real_clv_positivo": float(
            y_true.mean()
        ),
        "probabilidade_media": float(
            y_probability.mean()
        ),
    }


def clean_text_column(
    series: pd.Series,
) -> pd.Series:
    """
    Limpa colunas de texto.
    """

    cleaned = (
        series
        .astype("string")
        .str.strip()
    )

    invalid_values = [
        "",
        "nan",
        "None",
        "<NA>",
    ]

    return cleaned.mask(
        cleaned.isin(invalid_values)
    )


# ============================================================
# CARREGAR FICHEIROS
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset não encontrado: {DATA_PATH}"
    )

if not BASE_FEATURES_PATH.exists():
    raise FileNotFoundError(
        "Lista de features desportivas não encontrada: "
        f"{BASE_FEATURES_PATH}"
    )

data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

base_features = joblib.load(
    BASE_FEATURES_PATH
)

if not isinstance(base_features, list):
    base_features = list(base_features)


# ============================================================
# VALIDAR COLUNAS
# ============================================================

required_columns = [
    "id",
    "date",
    "season",
    "league",
    "home_team",
    "away_team",
    TARGET_RESULT,
    ODD_OVER_OPEN,
    ODD_UNDER_OPEN,
    ODD_OVER_CLOSE,
    ODD_UNDER_CLOSE,
    *base_features,
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "Faltam colunas no dataset: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CONVERTER TIPOS
# ============================================================

data["date"] = pd.to_datetime(
    data["date"],
    format="%Y-%m-%d",
    errors="coerce",
)

text_columns = [
    "season",
    "league",
    "home_team",
    "away_team",
]

for column in text_columns:
    data[column] = clean_text_column(
        data[column]
    )

numeric_columns = [
    TARGET_RESULT,
    ODD_OVER_OPEN,
    ODD_UNDER_OPEN,
    ODD_OVER_CLOSE,
    ODD_UNDER_CLOSE,
    *base_features,
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

data = data.replace(
    [np.inf, -np.inf],
    np.nan,
)


# ============================================================
# VALIDAR ODDS
# ============================================================

valid_open_odds = (
    (data[ODD_OVER_OPEN] > 1.0)
    & (data[ODD_UNDER_OPEN] > 1.0)
)

valid_close_odds = (
    (data[ODD_OVER_CLOSE] > 1.0)
    & (data[ODD_UNDER_CLOSE] > 1.0)
)

data.loc[
    ~valid_open_odds,
    [
        ODD_OVER_OPEN,
        ODD_UNDER_OPEN,
    ],
] = np.nan

data.loc[
    ~valid_close_odds,
    [
        ODD_OVER_CLOSE,
        ODD_UNDER_CLOSE,
    ],
] = np.nan


# ============================================================
# FEATURES DE MERCADO — ABERTURA
# ============================================================

data["implied_over_open"] = (
    1 / data[ODD_OVER_OPEN]
)

data["implied_under_open"] = (
    1 / data[ODD_UNDER_OPEN]
)

data["market_margin_open"] = (
    data["implied_over_open"]
    + data["implied_under_open"]
    - 1
)

data["fair_probability_open"] = (
    data["implied_over_open"]
    / (
        data["implied_over_open"]
        + data["implied_under_open"]
    )
)

clipped_open_probability = (
    data["fair_probability_open"]
    .clip(
        lower=0.0001,
        upper=0.9999,
    )
)

data["market_logit_open"] = np.log(
    clipped_open_probability
    / (
        1
        - clipped_open_probability
    )
)


# ============================================================
# PROBABILIDADE JUSTA DE FECHO
# ============================================================

data["implied_over_close"] = (
    1 / data[ODD_OVER_CLOSE]
)

data["implied_under_close"] = (
    1 / data[ODD_UNDER_CLOSE]
)

data["fair_probability_close"] = (
    data["implied_over_close"]
    / (
        data["implied_over_close"]
        + data["implied_under_close"]
    )
)


# ============================================================
# TARGET DE CLV
# ============================================================

data["clv"] = (
    data[ODD_OVER_OPEN]
    / data[ODD_OVER_CLOSE]
    - 1
)

data["fair_probability_movement"] = (
    data["fair_probability_close"]
    - data["fair_probability_open"]
)

data[TARGET_CLV] = (
    data["clv"] > 0
).astype(int)


# ============================================================
# FEATURES DO MODELO
# ============================================================

market_features = [
    ODD_OVER_OPEN,
    ODD_UNDER_OPEN,
    "fair_probability_open",
    "market_logit_open",
    "market_margin_open",
]

numeric_features = [
    *base_features,
    *market_features,
]

categorical_features = [
    "league",
]

model_features = [
    *numeric_features,
    *categorical_features,
]


# ============================================================
# LIMPEZA FINAL
# ============================================================

initial_rows = len(data)

data = data.dropna(
    subset=[
        "id",
        "date",
        "season",
        "league",
        TARGET_RESULT,
        TARGET_CLV,
        ODD_OVER_OPEN,
        ODD_UNDER_OPEN,
        ODD_OVER_CLOSE,
        ODD_UNDER_CLOSE,
        "clv",
        *numeric_features,
    ]
).copy()

data[TARGET_RESULT] = (
    data[TARGET_RESULT].astype(int)
)

data[TARGET_CLV] = (
    data[TARGET_CLV].astype(int)
)

data = data[
    data[TARGET_RESULT].isin([0, 1])
    & data[TARGET_CLV].isin([0, 1])
].copy()

data = data.sort_values(
    [
        "date",
        "id",
    ]
).reset_index(drop=True)

removed_rows = (
    initial_rows
    - len(data)
)

if data.empty:
    raise ValueError(
        "Não existem dados válidos para treinar "
        "o modelo CLV."
    )

if data[TARGET_CLV].nunique() < 2:
    raise ValueError(
        "O target CLV não contém as duas classes."
    )


# ============================================================
# TAXA-BASE E LIMIAR CONGELADO
# ============================================================

training_base_rate = float(
    data[TARGET_CLV].mean()
)

signal_threshold = max(
    MIN_SIGNAL_PROBABILITY,
    training_base_rate
    + MIN_EDGE_OVER_BASE_RATE,
)


# ============================================================
# TREINAR MODELO FINAL
# ============================================================

print("\n==========================================")
print(" FOOTBALLAI — TREINO FINAL DO MODELO CLV")
print("==========================================\n")

print(f"Linhas no dataset inicial : {initial_rows}")
print(f"Linhas removidas          : {removed_rows}")
print(f"Jogos utilizados          : {len(data)}")
print(f"Features desportivas      : {len(base_features)}")
print(f"Features de mercado       : {len(market_features)}")
print(f"Features numéricas        : {len(numeric_features)}")
print(f"Features categóricas      : {len(categorical_features)}")
print(f"Épocas utilizadas         : {data['season'].nunique()}")

print(
    f"Período                    : "
    f"{data['date'].min().date()} "
    f"a {data['date'].max().date()}"
)

print(
    f"Taxa-base CLV positivo    : "
    f"{training_base_rate:.2%}"
)

print(
    f"Limiar congelado          : "
    f"{signal_threshold:.2%}"
)

model = create_model(
    numeric_features=numeric_features,
    categorical_features=categorical_features,
)

model.fit(
    data[model_features],
    data[TARGET_CLV],
)


# ============================================================
# DIAGNÓSTICO IN-SAMPLE
# ============================================================

training_probabilities = (
    get_positive_probability(
        model,
        data[model_features],
    )
)

metrics = calculate_metrics(
    data[TARGET_CLV],
    training_probabilities,
)

data["clv_probability_training"] = (
    training_probabilities
)

data["training_signal"] = (
    (
        data["clv_probability_training"]
        >= signal_threshold
    )
    & (
        data[ODD_OVER_OPEN]
        >= MIN_ODD
    )
    & (
        data[ODD_OVER_OPEN]
        <= MAX_ODD
    )
)

training_signals = data[
    data["training_signal"]
].copy()

training_signal_count = len(
    training_signals
)

training_signal_clv = (
    float(training_signals["clv"].mean())
    if not training_signals.empty
    else np.nan
)

training_signal_positive_rate = (
    float(
        training_signals[TARGET_CLV].mean()
    )
    if not training_signals.empty
    else np.nan
)


# ============================================================
# GUARDAR BUNDLE DO MODELO
# ============================================================

generated_at = datetime.now().isoformat()

model_bundle = {
    "model": model,

    "target": TARGET_CLV,
    "result_target": TARGET_RESULT,

    "base_features": base_features,
    "market_features": market_features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "model_features": model_features,

    "odd_over_open_column": ODD_OVER_OPEN,
    "odd_under_open_column": ODD_UNDER_OPEN,

    "training_base_rate": training_base_rate,
    "signal_threshold": signal_threshold,

    "min_signal_probability":
        MIN_SIGNAL_PROBABILITY,

    "min_edge_over_base_rate":
        MIN_EDGE_OVER_BASE_RATE,

    "min_odd": MIN_ODD,
    "max_odd": MAX_ODD,

    "logistic_c": LOGISTIC_C,
    "random_state": RANDOM_STATE,

    "training_rows": int(len(data)),
    "training_date_min":
        data["date"].min().date().isoformat(),
    "training_date_max":
        data["date"].max().date().isoformat(),

    "generated_at": generated_at,
}

joblib.dump(
    model_bundle,
    MODEL_PATH,
)

joblib.dump(
    model_features,
    FEATURES_CLV_PATH,
)


# ============================================================
# GUARDAR METADADOS
# ============================================================

metadata = {
    "generated_at": generated_at,

    "model_type":
        "LogisticRegression",

    "model_purpose":
        "Prever probabilidade de CLV positivo no mercado Over 2.5",

    "training_rows":
        int(len(data)),

    "removed_rows":
        int(removed_rows),

    "training_date_min":
        data["date"].min().date().isoformat(),

    "training_date_max":
        data["date"].max().date().isoformat(),

    "seasons":
        sorted(
            data["season"]
            .astype(str)
            .unique()
            .tolist()
        ),

    "leagues":
        sorted(
            data["league"]
            .astype(str)
            .unique()
            .tolist()
        ),

    "training_base_rate":
        training_base_rate,

    "signal_threshold":
        signal_threshold,

    "min_signal_probability":
        MIN_SIGNAL_PROBABILITY,

    "min_edge_over_base_rate":
        MIN_EDGE_OVER_BASE_RATE,

    "min_odd":
        MIN_ODD,

    "max_odd":
        MAX_ODD,

    "logistic_c":
        LOGISTIC_C,

    "base_feature_count":
        len(base_features),

    "market_feature_count":
        len(market_features),

    "numeric_feature_count":
        len(numeric_features),

    "categorical_feature_count":
        len(categorical_features),

    "base_features":
        base_features,

    "market_features":
        market_features,

    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    "diagnostic_metrics_in_sample": {
        key: (
            None
            if isinstance(value, float)
            and np.isnan(value)
            else float(value)
        )
        for key, value in metrics.items()
    },

    "training_signal_count":
        int(training_signal_count),

    "training_signal_mean_clv": (
        None
        if np.isnan(training_signal_clv)
        else training_signal_clv
    ),

    "training_signal_positive_clv_rate": (
        None
        if np.isnan(
            training_signal_positive_rate
        )
        else training_signal_positive_rate
    ),

    "important_warning": (
        "As métricas de diagnóstico deste ficheiro são "
        "in-sample. A validação válida é o backtest "
        "walk-forward e o paper trading futuro."
    ),
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as metadata_file:
    json.dump(
        metadata,
        metadata_file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# GUARDAR RELATÓRIO DE TREINO
# ============================================================

training_report_columns = [
    "id",
    "date",
    "season",
    "league",
    "home_team",
    "away_team",
    ODD_OVER_OPEN,
    ODD_UNDER_OPEN,
    ODD_OVER_CLOSE,
    ODD_UNDER_CLOSE,
    "fair_probability_open",
    "fair_probability_close",
    "fair_probability_movement",
    "clv",
    TARGET_CLV,
    "clv_probability_training",
    "training_signal",
]

training_report = data[
    training_report_columns
].copy()

training_report.to_csv(
    TRAINING_REPORT_PATH,
    index=False,
)


# ============================================================
# MOSTRAR RESULTADO
# ============================================================

print("\n==========================================")
print(" DIAGNÓSTICO DO TREINO")
print("==========================================\n")

print(
    f"Accuracy                   : "
    f"{metrics['accuracy']:.2%}"
)

print(
    f"Precision                  : "
    f"{metrics['precision']:.2%}"
)

print(
    f"Recall                     : "
    f"{metrics['recall']:.2%}"
)

if np.isnan(metrics["roc_auc"]):
    print(
        "ROC-AUC                    : "
        "indisponível"
    )
else:
    print(
        f"ROC-AUC                    : "
        f"{metrics['roc_auc']:.4f}"
    )

print(
    f"Brier Score                : "
    f"{metrics['brier_score']:.4f}"
)

print(
    f"Log Loss                   : "
    f"{metrics['log_loss']:.4f}"
)

print(
    f"Taxa real CLV positivo     : "
    f"{metrics['taxa_real_clv_positivo']:.2%}"
)

print(
    f"Probabilidade média        : "
    f"{metrics['probabilidade_media']:.2%}"
)

print(
    f"Sinais no treino           : "
    f"{training_signal_count}"
)

if np.isnan(training_signal_clv):
    print(
        "CLV médio dos sinais       : "
        "indisponível"
    )
else:
    print(
        f"CLV médio dos sinais       : "
        f"{training_signal_clv:.2%}"
    )

if np.isnan(training_signal_positive_rate):
    print(
        "CLV positivo nos sinais    : "
        "indisponível"
    )
else:
    print(
        f"CLV positivo nos sinais    : "
        f"{training_signal_positive_rate:.2%}"
    )


# ============================================================
# FICHEIROS CRIADOS
# ============================================================

print("\n==========================================")
print(" MODELO FINAL GUARDADO")
print("==========================================\n")

print(
    f"✅ {MODEL_PATH.relative_to(ROOT_DIR)}"
)

print(
    f"✅ {FEATURES_CLV_PATH.relative_to(ROOT_DIR)}"
)

print(
    f"✅ {METADATA_PATH.relative_to(ROOT_DIR)}"
)

print(
    f"✅ {TRAINING_REPORT_PATH.relative_to(ROOT_DIR)}"
)

print(
    "\n⚠ O modelo está congelado para paper trading."
)

print(
    "⚠ Não alteres os limiares durante a validação futura."
)

print(
    "⚠ Não utilizar dinheiro real nesta fase."
)