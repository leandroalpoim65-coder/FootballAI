from __future__ import annotations

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

FEATURES_PATH = (
    ROOT_DIR
    / "ml"
    / "features_over25.pkl"
)

REPORTS_DIR = ROOT_DIR / "reports"


# ============================================================
# CONFIGURAÇÃO
# ============================================================

TARGET_RESULT = "over25"

ODD_OVER_OPEN = "odd_over25_open"
ODD_UNDER_OPEN = "odd_under25_open"

ODD_OVER_CLOSE = "odd_over25_close"
ODD_UNDER_CLOSE = "odd_under25_close"

# Com sete épocas disponíveis, três são usadas inicialmente
# para treino e as restantes para teste walk-forward.
MIN_TRAIN_SEASONS = 3

# Uma seleção é criada quando o modelo estima uma
# probabilidade suficientemente elevada de CLV positivo.
MIN_SIGNAL_PROBABILITY = 0.55

# Exige também que a probabilidade prevista seja superior
# à frequência histórica de CLV positivo no treino.
MIN_EDGE_OVER_BASE_RATE = 0.04

# Limites aplicados apenas ao cálculo financeiro secundário.
MIN_ODD = 1.20
MAX_ODD = 4.00

STAKE = 1.0
RANDOM_STATE = 42

REPORTS_DIR.mkdir(
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
    Modelo conservador para prever CLV positivo.

    As features numéricas são normalizadas e a liga é
    transformada em variáveis binárias.
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
        C=0.10,
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


def positive_probability(
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


def calculate_classification_metrics(
    target: pd.Series,
    probabilities: pd.Series | np.ndarray,
) -> dict:
    """
    Calcula métricas de classificação e calibração.
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
        auc = roc_auc_score(
            y_true,
            y_probability,
        )
    else:
        auc = np.nan

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
        "roc_auc": float(auc),
        "brier": float(
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
        "real_positive_rate": float(
            y_true.mean()
        ),
        "predicted_positive_rate": float(
            y_probability.mean()
        ),
    }


def calculate_max_drawdown(
    profits: pd.Series,
) -> float:
    """
    Calcula a maior queda da banca desde um máximo anterior.
    """
    if profits.empty:
        return 0.0

    cumulative_profit = (
        profits.cumsum().to_numpy()
    )

    bankroll_curve = np.concatenate(
        [
            np.array([0.0]),
            cumulative_profit,
        ]
    )

    running_peak = np.maximum.accumulate(
        bankroll_curve
    )

    drawdown = (
        running_peak
        - bankroll_curve
    )

    return float(drawdown.max())


def calculate_profit_factor(
    profits: pd.Series,
) -> float:
    """
    Ganhos brutos divididos pelas perdas brutas.
    """
    gross_profit = float(
        profits[
            profits > 0
        ].sum()
    )

    gross_loss = abs(
        float(
            profits[
                profits < 0
            ].sum()
        )
    )

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")

        return 0.0

    return gross_profit / gross_loss


def safe_mean(
    series: pd.Series,
) -> float:
    """
    Calcula a média ignorando valores vazios.
    """
    valid_values = series.dropna()

    if valid_values.empty:
        return np.nan

    return float(valid_values.mean())


def create_group_report(
    signals: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Cria relatório por época ou liga.
    """
    if signals.empty:
        return pd.DataFrame()

    report = (
        signals.groupby(
            group_column,
            dropna=False,
        )
        .agg(
            selecoes=("profit", "size"),
            clv_positivos=(
                "clv_positive",
                "sum",
            ),
            clv_medio=(
                "clv",
                "mean",
            ),
            clv_mediano=(
                "clv",
                "median",
            ),
            probabilidade_media=(
                "clv_probability",
                "mean",
            ),
            movimento_probabilidade=(
                "fair_probability_movement",
                "mean",
            ),
            odd_abertura_media=(
                ODD_OVER_OPEN,
                "mean",
            ),
            odd_fecho_media=(
                ODD_OVER_CLOSE,
                "mean",
            ),
            vitorias=(
                "won",
                "sum",
            ),
            lucro=(
                "profit",
                "sum",
            ),
        )
        .reset_index()
    )

    report["clv_positivo_percent"] = (
        report["clv_positivos"]
        / report["selecoes"]
        * 100
    )

    report["taxa_acerto"] = (
        report["vitorias"]
        / report["selecoes"]
        * 100
    )

    report["roi"] = (
        report["lucro"]
        / (
            report["selecoes"]
            * STAKE
        )
        * 100
    )

    report["yield"] = report["roi"]

    return report.sort_values(
        "clv_medio",
        ascending=False,
    )


# ============================================================
# CARREGAR DADOS
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset não encontrado: {DATA_PATH}"
    )

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Lista de features não encontrada: "
        f"{FEATURES_PATH}"
    )

data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

base_features = joblib.load(
    FEATURES_PATH
)


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
# PROBABILIDADE JUSTA DE ABERTURA
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

open_probability_clipped = (
    data["fair_probability_open"]
    .clip(
        lower=0.0001,
        upper=0.9999,
    )
)

data["market_logit_open"] = np.log(
    open_probability_clipped
    / (
        1
        - open_probability_clipped
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

data["market_margin_close"] = (
    data["implied_over_close"]
    + data["implied_under_close"]
    - 1
)

data["fair_probability_close"] = (
    data["implied_over_close"]
    / (
        data["implied_over_close"]
        + data["implied_under_close"]
    )
)


# ============================================================
# TARGET CLV
# ============================================================

# CLV positivo quando a odd disponível na abertura é
# superior à odd final de fecho.
data["clv"] = (
    data[ODD_OVER_OPEN]
    / data[ODD_OVER_CLOSE]
    - 1
)

data["fair_probability_movement"] = (
    data["fair_probability_close"]
    - data["fair_probability_open"]
)

data["clv_positive"] = (
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
# LIMPEZA
# ============================================================

before_cleaning = len(data)

data = data.dropna(
    subset=[
        "date",
        "season",
        "league",
        TARGET_RESULT,
        ODD_OVER_OPEN,
        ODD_UNDER_OPEN,
        ODD_OVER_CLOSE,
        ODD_UNDER_CLOSE,
        "clv",
        "clv_positive",
        *numeric_features,
    ]
).copy()

data[TARGET_RESULT] = (
    data[TARGET_RESULT].astype(int)
)

data["clv_positive"] = (
    data["clv_positive"].astype(int)
)

data = data[
    data[TARGET_RESULT].isin([0, 1])
    & data["clv_positive"].isin([0, 1])
].copy()

data = data.sort_values(
    [
        "date",
        "id",
    ]
).reset_index(drop=True)

removed_rows = (
    before_cleaning
    - len(data)
)

if data.empty:
    raise ValueError(
        "Não existem jogos com odds completas "
        "de abertura e fecho."
    )


# ============================================================
# ORDENAR ÉPOCAS
# ============================================================

season_order = (
    data.groupby("season")["date"]
    .min()
    .sort_values()
)

seasons = season_order.index.tolist()

if len(seasons) <= MIN_TRAIN_SEASONS:
    raise ValueError(
        "Não existem épocas suficientes para "
        "o backtest walk-forward."
    )

test_seasons = seasons[
    MIN_TRAIN_SEASONS:
]


# ============================================================
# BACKTEST WALK-FORWARD
# ============================================================

print("\n==========================================")
print(" FOOTBALLAI — BACKTEST CLV")
print("==========================================\n")

print(f"Jogos com odds completas : {len(data)}")
print(f"Linhas removidas         : {removed_rows}")
print(f"Features desportivas     : {len(base_features)}")
print(f"Features de mercado      : {len(market_features)}")
print(f"Épocas disponíveis       : {len(seasons)}")
print(f"Épocas de teste          : {len(test_seasons)}")
print(
    f"Probabilidade mínima     : "
    f"{MIN_SIGNAL_PROBABILITY:.1%}"
)
print(
    f"Edge sobre taxa-base     : "
    f"{MIN_EDGE_OVER_BASE_RATE:.1%}"
)

all_predictions = []
fold_results = []

for test_season in test_seasons:
    test_position = seasons.index(
        test_season
    )

    training_seasons = seasons[
        :test_position
    ]

    train = data[
        data["season"].isin(
            training_seasons
        )
    ].copy()

    test = data[
        data["season"] == test_season
    ].copy()

    if train.empty or test.empty:
        continue

    if train["clv_positive"].nunique() < 2:
        print(
            f"\nÉpoca {test_season} ignorada: "
            "o treino não contém duas classes."
        )
        continue

    training_base_rate = float(
        train["clv_positive"].mean()
    )

    signal_threshold = max(
        MIN_SIGNAL_PROBABILITY,
        training_base_rate
        + MIN_EDGE_OVER_BASE_RATE,
    )

    model = create_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model.fit(
        train[model_features],
        train["clv_positive"],
    )

    test["clv_probability"] = (
        positive_probability(
            model,
            test[model_features],
        )
    )

    test["baseline_probability"] = (
        training_base_rate
    )

    test["signal_threshold"] = (
        signal_threshold
    )

    valid_signal_odds = (
        (test[ODD_OVER_OPEN] >= MIN_ODD)
        & (test[ODD_OVER_OPEN] <= MAX_ODD)
    )

    test["signal"] = (
        valid_signal_odds
        & (
            test["clv_probability"]
            >= signal_threshold
        )
    )

    test["won"] = (
        test[TARGET_RESULT] == 1
    ).astype(int)

    # O lucro é uma métrica secundária.
    # A seleção foi feita apenas pela previsão de CLV.
    test["profit"] = np.where(
        test["signal"],
        np.where(
            test["won"] == 1,
            (
                test[ODD_OVER_OPEN]
                - 1
            )
            * STAKE,
            -STAKE,
        ),
        0.0,
    )

    test["test_season"] = test_season

    all_predictions.append(test)

    model_metrics = (
        calculate_classification_metrics(
            test["clv_positive"],
            test["clv_probability"],
        )
    )

    baseline_metrics = (
        calculate_classification_metrics(
            test["clv_positive"],
            test["baseline_probability"],
        )
    )

    fold_signals = test[
        test["signal"]
    ].copy()

    fold_clv_mean = safe_mean(
        fold_signals["clv"]
    )

    fold_clv_positive_rate = (
        float(
            fold_signals[
                "clv_positive"
            ].mean()
            * 100
        )
        if not fold_signals.empty
        else np.nan
    )

    fold_profit = float(
        fold_signals["profit"].sum()
    )

    fold_invested = (
        len(fold_signals)
        * STAKE
    )

    fold_roi = (
        fold_profit
        / fold_invested
        * 100
        if fold_invested > 0
        else 0.0
    )

    fold_results.append(
        {
            "test_season": test_season,
            "training_seasons":
                len(training_seasons),
            "train_matches": len(train),
            "test_matches": len(test),
            "training_base_rate":
                training_base_rate,
            "signal_threshold":
                signal_threshold,
            "model_auc":
                model_metrics["roc_auc"],
            "model_brier":
                model_metrics["brier"],
            "baseline_brier":
                baseline_metrics["brier"],
            "signals":
                len(fold_signals),
            "average_clv":
                fold_clv_mean,
            "positive_clv_rate":
                fold_clv_positive_rate,
            "profit":
                fold_profit,
            "roi":
                fold_roi,
        }
    )

    print("\n------------------------------------------")
    print(f"Época de teste       : {test_season}")
    print(f"Treino               : {len(train)} jogos")
    print(f"Teste                : {len(test)} jogos")
    print(
        f"Taxa-base treino     : "
        f"{training_base_rate:.2%}"
    )
    print(
        f"Limiar do sinal      : "
        f"{signal_threshold:.2%}"
    )
    print(
        f"ROC-AUC modelo       : "
        f"{model_metrics['roc_auc']:.4f}"
    )
    print(
        f"Brier modelo         : "
        f"{model_metrics['brier']:.4f}"
    )
    print(
        f"Brier previsão-base  : "
        f"{baseline_metrics['brier']:.4f}"
    )
    print(f"Seleções             : {len(fold_signals)}")

    if np.isnan(fold_clv_mean):
        print("CLV médio            : indisponível")
        print("CLV positivo         : indisponível")
    else:
        print(
            f"CLV médio            : "
            f"{fold_clv_mean:.2%}"
        )
        print(
            f"CLV positivo         : "
            f"{fold_clv_positive_rate:.2f}%"
        )

    print(f"Lucro secundário     : {fold_profit:.2f}")
    print(f"ROI secundário       : {fold_roi:.2f}%")


# ============================================================
# JUNTAR RESULTADOS
# ============================================================

if not all_predictions:
    raise RuntimeError(
        "Nenhuma janela walk-forward foi concluída."
    )

predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

predictions = predictions.sort_values(
    [
        "date",
        "id",
    ]
).reset_index(drop=True)

signals = predictions[
    predictions["signal"]
].copy()

fold_report = pd.DataFrame(
    fold_results
)


# ============================================================
# MÉTRICAS GLOBAIS
# ============================================================

global_model_metrics = (
    calculate_classification_metrics(
        predictions["clv_positive"],
        predictions["clv_probability"],
    )
)

global_baseline_metrics = (
    calculate_classification_metrics(
        predictions["clv_positive"],
        predictions["baseline_probability"],
    )
)

all_games_clv_mean = safe_mean(
    predictions["clv"]
)

all_games_clv_positive_rate = float(
    predictions["clv_positive"].mean()
    * 100
)

number_of_signals = len(signals)

signal_clv_mean = safe_mean(
    signals["clv"]
)

signal_clv_median = (
    float(signals["clv"].median())
    if not signals.empty
    else np.nan
)

signal_clv_positive_rate = (
    float(
        signals["clv_positive"].mean()
        * 100
    )
    if not signals.empty
    else np.nan
)

average_probability_movement = (
    safe_mean(
        signals[
            "fair_probability_movement"
        ]
    )
)

number_of_wins = int(
    signals["won"].sum()
)

invested = (
    number_of_signals
    * STAKE
)

profit = float(
    signals["profit"].sum()
)

roi = (
    profit
    / invested
    * 100
    if invested > 0
    else 0.0
)

hit_rate = (
    number_of_wins
    / number_of_signals
    * 100
    if number_of_signals > 0
    else 0.0
)

average_opening_odd = (
    safe_mean(
        signals[ODD_OVER_OPEN]
    )
)

max_drawdown = calculate_max_drawdown(
    signals["profit"]
)

profit_factor = calculate_profit_factor(
    signals["profit"]
)


# ============================================================
# RELATÓRIOS AGRUPADOS
# ============================================================

season_report = create_group_report(
    signals,
    "season",
)

league_report = create_group_report(
    signals,
    "league",
)


# ============================================================
# MOSTRAR RESULTADO GLOBAL
# ============================================================

print("\n")
print("==========================================")
print(" RESULTADO GLOBAL CLV")
print("==========================================\n")

print("Previsão de CLV positivo:")
print(
    f"Jogos previstos          : "
    f"{len(predictions)}"
)
print(
    f"ROC-AUC modelo           : "
    f"{global_model_metrics['roc_auc']:.4f}"
)
print(
    f"Brier modelo             : "
    f"{global_model_metrics['brier']:.4f}"
)
print(
    f"Brier previsão-base      : "
    f"{global_baseline_metrics['brier']:.4f}"
)
print(
    f"Log Loss modelo          : "
    f"{global_model_metrics['log_loss']:.4f}"
)
print(
    f"Log Loss previsão-base   : "
    f"{global_baseline_metrics['log_loss']:.4f}"
)
print(
    f"Taxa real CLV positivo   : "
    f"{global_model_metrics['real_positive_rate']:.2%}"
)
print(
    f"Probabilidade média      : "
    f"{global_model_metrics['predicted_positive_rate']:.2%}"
)

print("\nComparação das seleções:")
print(
    f"CLV médio — todos jogos  : "
    f"{all_games_clv_mean:.2%}"
)
print(
    f"CLV positivo — todos     : "
    f"{all_games_clv_positive_rate:.2f}%"
)
print(f"Seleções                 : {number_of_signals}")

if np.isnan(signal_clv_mean):
    print("CLV médio — seleções     : indisponível")
    print("CLV mediano — seleções   : indisponível")
    print("CLV positivo — seleções  : indisponível")
else:
    print(
        f"CLV médio — seleções     : "
        f"{signal_clv_mean:.2%}"
    )
    print(
        f"CLV mediano — seleções   : "
        f"{signal_clv_median:.2%}"
    )
    print(
        f"CLV positivo — seleções  : "
        f"{signal_clv_positive_rate:.2f}%"
    )
    print(
        f"Movimento prob. justo    : "
        f"{average_probability_movement:.2%}"
    )

print("\nResultado financeiro secundário:")
print(f"Apostas                  : {number_of_signals}")
print(f"Vitórias                 : {number_of_wins}")
print(f"Taxa de acerto           : {hit_rate:.2f}%")

if np.isnan(average_opening_odd):
    print("Odd média abertura       : indisponível")
else:
    print(
        f"Odd média abertura       : "
        f"{average_opening_odd:.2f}"
    )

print(f"Investido                : {invested:.2f}")
print(f"Lucro                    : {profit:.2f}")
print(f"ROI / Yield              : {roi:.2f}%")
print(f"Drawdown máximo          : {max_drawdown:.2f}")

if np.isinf(profit_factor):
    print("Profit Factor            : infinito")
else:
    print(
        f"Profit Factor            : "
        f"{profit_factor:.2f}"
    )


if not season_report.empty:
    print("\n==========================================")
    print(" CLV POR ÉPOCA")
    print("==========================================\n")

    print(
        season_report.to_string(
            index=False,
            formatters={
                "clv_medio":
                    "{:.2%}".format,
                "clv_mediano":
                    "{:.2%}".format,
                "probabilidade_media":
                    "{:.2%}".format,
                "movimento_probabilidade":
                    "{:.2%}".format,
                "odd_abertura_media":
                    "{:.2f}".format,
                "odd_fecho_media":
                    "{:.2f}".format,
                "lucro":
                    "{:.2f}".format,
                "clv_positivo_percent":
                    "{:.2f}%".format,
                "taxa_acerto":
                    "{:.2f}%".format,
                "roi":
                    "{:.2f}%".format,
                "yield":
                    "{:.2f}%".format,
            },
        )
    )


if not league_report.empty:
    print("\n==========================================")
    print(" CLV POR LIGA")
    print("==========================================\n")

    print(
        league_report.to_string(
            index=False,
            formatters={
                "clv_medio":
                    "{:.2%}".format,
                "clv_mediano":
                    "{:.2%}".format,
                "probabilidade_media":
                    "{:.2%}".format,
                "movimento_probabilidade":
                    "{:.2%}".format,
                "odd_abertura_media":
                    "{:.2f}".format,
                "odd_fecho_media":
                    "{:.2f}".format,
                "lucro":
                    "{:.2f}".format,
                "clv_positivo_percent":
                    "{:.2f}%".format,
                "taxa_acerto":
                    "{:.2f}%".format,
                "roi":
                    "{:.2f}%".format,
                "yield":
                    "{:.2f}%".format,
            },
        )
    )


# ============================================================
# GUARDAR RELATÓRIOS
# ============================================================

predictions.to_csv(
    REPORTS_DIR
    / "clv_predictions_over25.csv",
    index=False,
)

signals.to_csv(
    REPORTS_DIR
    / "clv_signals_over25.csv",
    index=False,
)

fold_report.to_csv(
    REPORTS_DIR
    / "clv_folds_over25.csv",
    index=False,
)

season_report.to_csv(
    REPORTS_DIR
    / "clv_by_season_over25.csv",
    index=False,
)

league_report.to_csv(
    REPORTS_DIR
    / "clv_by_league_over25.csv",
    index=False,
)

print("\nRelatórios guardados:")

print(
    "✅ reports/clv_predictions_over25.csv"
)

print(
    "✅ reports/clv_signals_over25.csv"
)

print(
    "✅ reports/clv_folds_over25.csv"
)

print(
    "✅ reports/clv_by_season_over25.csv"
)

print(
    "✅ reports/clv_by_league_over25.csv"
)

print(
    "\n✅ Backtest de CLV concluído."
)