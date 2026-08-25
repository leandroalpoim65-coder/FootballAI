from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


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

TARGET = "over25"

ODD_OVER_COLUMN = "odd_over25"
ODD_UNDER_COLUMN = "odd_under25"

# São usadas pelo menos cinco épocas anteriores para treinar.
MIN_TRAIN_SEASONS = 5

# Vantagem mínima do modelo sobre a probabilidade justa da casa.
MIN_EDGE = 0.05

MIN_ODD = 1.20
MAX_ODD = 4.00

STAKE = 1.0

CALIBRATION_SPLITS = 3
RANDOM_STATE = 42

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FUNÇÕES
# ============================================================

def create_base_model() -> ExtraTreesClassifier:
    """
    Modelo usado em todas as janelas do backtest.
    """
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_calibrated_model():
    """
    Cria o modelo calibrado usando validação temporal.
    """
    temporal_cv = TimeSeriesSplit(
        n_splits=CALIBRATION_SPLITS,
    )

    base_model = create_base_model()

    try:
        return CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=temporal_cv,
            n_jobs=-1,
        )

    except TypeError:
        # Compatibilidade com versões anteriores do sklearn.
        return CalibratedClassifierCV(
            base_estimator=base_model,
            method="sigmoid",
            cv=temporal_cv,
            n_jobs=-1,
        )


def positive_probability(
    model,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Obtém a probabilidade da classe 1.
    """
    probabilities = model.predict_proba(features)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "O modelo não possui a classe positiva 1."
        )

    positive_index = classes.index(1)

    return probabilities[:, positive_index]


def calculate_max_drawdown(
    profits: pd.Series,
) -> float:
    """
    Calcula a maior queda da banca desde um máximo anterior.
    """
    if profits.empty:
        return 0.0

    cumulative_profit = profits.cumsum().to_numpy()

    bankroll_curve = np.concatenate(
        [
            np.array([0.0]),
            cumulative_profit,
        ]
    )

    running_peak = np.maximum.accumulate(
        bankroll_curve
    )

    drawdowns = (
        running_peak
        - bankroll_curve
    )

    return float(drawdowns.max())


def calculate_profit_factor(
    profits: pd.Series,
) -> float:
    """
    Ganhos brutos divididos pelas perdas brutas.
    """
    gross_profit = float(
        profits[profits > 0].sum()
    )

    gross_loss = abs(
        float(
            profits[profits < 0].sum()
        )
    )

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")

        return 0.0

    return gross_profit / gross_loss


def calculate_model_metrics(
    target: pd.Series,
    probabilities: pd.Series,
) -> dict:
    """
    Métricas das previsões fora da amostra.
    """
    target = target.astype(int)

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            target,
            predictions,
        ),
        "precision": precision_score(
            target,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            target,
            predictions,
            zero_division=0,
        ),
        "brier": brier_score_loss(
            target,
            probabilities,
        ),
        "log_loss": log_loss(
            target,
            probabilities,
            labels=[0, 1],
        ),
        "taxa_real": target.mean(),
        "probabilidade_media":
            probabilities.mean(),
    }

    if target.nunique() >= 2:
        metrics["roc_auc"] = roc_auc_score(
            target,
            probabilities,
        )
    else:
        metrics["roc_auc"] = np.nan

    metrics["erro_calibracao"] = (
        metrics["probabilidade_media"]
        - metrics["taxa_real"]
    )

    return metrics


def create_betting_report(
    bets: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Cria relatório financeiro por época ou liga.
    """
    if bets.empty:
        return pd.DataFrame()

    report = (
        bets.groupby(
            group_column,
            dropna=False,
        )
        .agg(
            apostas=("lucro", "size"),
            vitorias=("ganhou", "sum"),
            lucro=("lucro", "sum"),
            odd_media=(
                ODD_OVER_COLUMN,
                "mean",
            ),
            probabilidade_media=(
                "prob_modelo",
                "mean",
            ),
            probabilidade_mercado=(
                "prob_mercado_justa",
                "mean",
            ),
            edge_medio=("edge", "mean"),
        )
        .reset_index()
    )

    report["taxa_acerto"] = (
        report["vitorias"]
        / report["apostas"]
        * 100
    )

    report["roi"] = (
        report["lucro"]
        / (
            report["apostas"]
            * STAKE
        )
        * 100
    )

    report["yield"] = report["roi"]

    return report.sort_values(
        "roi",
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

features = joblib.load(
    FEATURES_PATH
)

required_columns = [
    "id",
    "date",
    "season",
    "league",
    "home_team",
    "away_team",
    TARGET,
    ODD_OVER_COLUMN,
    ODD_UNDER_COLUMN,
    *features,
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
# LIMPEZA
# ============================================================

data["date"] = pd.to_datetime(
    data["date"],
    format="%Y-%m-%d",
    errors="coerce",
)

numeric_columns = [
    TARGET,
    ODD_OVER_COLUMN,
    ODD_UNDER_COLUMN,
    *features,
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

before_cleaning = len(data)

# As odds não são removidas aqui porque ainda podemos avaliar
# o modelo em jogos que não possuem odds.
data = data.dropna(
    subset=[
        "date",
        "season",
        TARGET,
        *features,
    ]
).copy()

data[TARGET] = (
    data[TARGET].astype(int)
)

data = data[
    data[TARGET].isin([0, 1])
].copy()

data = data.sort_values(
    ["date", "id"],
).reset_index(drop=True)

removed_rows = (
    before_cleaning
    - len(data)
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
# WALK-FORWARD
# ============================================================

print("\n==========================================")
print(" FOOTBALLAI — BACKTEST WALK-FORWARD")
print("==========================================\n")

print(f"Jogos válidos        : {len(data)}")
print(f"Linhas removidas     : {removed_rows}")
print(f"Features             : {len(features)}")
print(f"Épocas disponíveis   : {len(seasons)}")
print(f"Épocas de teste      : {len(test_seasons)}")
print(f"Edge mínimo          : {MIN_EDGE:.1%}")

all_predictions = []
fold_rows = []

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

    if train[TARGET].nunique() < 2:
        continue

    model = create_calibrated_model()

    model.fit(
        train[features],
        train[TARGET],
    )

    test["prob_modelo"] = (
        positive_probability(
            model,
            test[features],
        )
    )

    # Probabilidades implícitas brutas.
    test["implicita_over"] = (
        1 / test[ODD_OVER_COLUMN]
    )

    test["implicita_under"] = (
        1 / test[ODD_UNDER_COLUMN]
    )

    test["margem_mercado"] = (
        test["implicita_over"]
        + test["implicita_under"]
        - 1
    )

    # Remoção proporcional da margem da casa.
    test["prob_mercado_justa"] = (
        test["implicita_over"]
        / (
            test["implicita_over"]
            + test["implicita_under"]
        )
    )

    test["edge"] = (
        test["prob_modelo"]
        - test["prob_mercado_justa"]
    )

    valid_odds = (
        test[ODD_OVER_COLUMN].notna()
        & test[ODD_UNDER_COLUMN].notna()
        & (test[ODD_OVER_COLUMN] >= MIN_ODD)
        & (test[ODD_OVER_COLUMN] <= MAX_ODD)
        & (test[ODD_UNDER_COLUMN] > 1.00)
    )

    test["aposta"] = (
        valid_odds
        & (test["edge"] >= MIN_EDGE)
    )

    test["ganhou"] = (
        test[TARGET] == 1
    ).astype(int)

    test["lucro"] = np.where(
        test["aposta"],
        np.where(
            test["ganhou"] == 1,
            (
                test[ODD_OVER_COLUMN]
                - 1
            )
            * STAKE,
            -STAKE,
        ),
        0.0,
    )

    test["epoca_teste"] = (
        test_season
    )

    all_predictions.append(test)

    fold_bets = test[
        test["aposta"]
    ].copy()

    fold_profit = float(
        fold_bets["lucro"].sum()
    )

    fold_staked = (
        len(fold_bets)
        * STAKE
    )

    fold_roi = (
        fold_profit
        / fold_staked
        * 100
        if fold_staked > 0
        else 0.0
    )

    fold_auc = (
        roc_auc_score(
            test[TARGET],
            test["prob_modelo"],
        )
        if test[TARGET].nunique() >= 2
        else np.nan
    )

    fold_rows.append(
        {
            "epoca_teste": test_season,
            "epocas_treino":
                len(training_seasons),
            "jogos_treino": len(train),
            "jogos_teste": len(test),
            "roc_auc": fold_auc,
            "brier_modelo":
                brier_score_loss(
                    test[TARGET],
                    test["prob_modelo"],
                ),
            "apostas": len(fold_bets),
            "lucro": fold_profit,
            "roi": fold_roi,
        }
    )

    print("\n------------------------------------------")
    print(f"Época de teste : {test_season}")
    print(
        f"Treino          : {len(train)} jogos "
        f"até {train['date'].max().date()}"
    )
    print(f"Teste           : {len(test)} jogos")
    print(f"ROC-AUC         : {fold_auc:.4f}")
    print(f"Apostas         : {len(fold_bets)}")
    print(f"Lucro           : {fold_profit:.2f}")
    print(f"ROI             : {fold_roi:.2f}%")


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
    ["date", "id"],
).reset_index(drop=True)

bets = predictions[
    predictions["aposta"]
].copy()

fold_report = pd.DataFrame(
    fold_rows
)


# ============================================================
# MÉTRICAS GLOBAIS FORA DA AMOSTRA
# ============================================================

model_metrics = calculate_model_metrics(
    predictions[TARGET],
    predictions["prob_modelo"],
)

market_rows = predictions.dropna(
    subset=["prob_mercado_justa"]
).copy()

if not market_rows.empty:
    market_brier = brier_score_loss(
        market_rows[TARGET],
        market_rows["prob_mercado_justa"],
    )

    market_log_loss = log_loss(
        market_rows[TARGET],
        market_rows["prob_mercado_justa"],
        labels=[0, 1],
    )
else:
    market_brier = np.nan
    market_log_loss = np.nan


# ============================================================
# MÉTRICAS FINANCEIRAS
# ============================================================

number_of_bets = len(bets)
number_of_wins = int(
    bets["ganhou"].sum()
)

invested = (
    number_of_bets
    * STAKE
)

profit = float(
    bets["lucro"].sum()
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
    / number_of_bets
    * 100
    if number_of_bets > 0
    else 0.0
)

average_odd = (
    float(
        bets[ODD_OVER_COLUMN].mean()
    )
    if number_of_bets > 0
    else 0.0
)

average_edge = (
    float(
        bets["edge"].mean()
    )
    if number_of_bets > 0
    else 0.0
)

max_drawdown = calculate_max_drawdown(
    bets["lucro"]
)

profit_factor = calculate_profit_factor(
    bets["lucro"]
)


# ============================================================
# RELATÓRIOS AGRUPADOS
# ============================================================

season_report = create_betting_report(
    bets,
    "season",
)

league_report = create_betting_report(
    bets,
    "league",
)


# ============================================================
# MOSTRAR RESULTADO
# ============================================================

print("\n")
print("==========================================")
print(" RESULTADO GLOBAL FORA DA AMOSTRA")
print("==========================================\n")

print("Métricas do modelo:")
print(
    f"Jogos previstos       : "
    f"{len(predictions)}"
)
print(
    f"Accuracy              : "
    f"{model_metrics['accuracy']:.2%}"
)
print(
    f"Precision             : "
    f"{model_metrics['precision']:.2%}"
)
print(
    f"Recall                : "
    f"{model_metrics['recall']:.2%}"
)
print(
    f"ROC-AUC               : "
    f"{model_metrics['roc_auc']:.4f}"
)
print(
    f"Brier modelo          : "
    f"{model_metrics['brier']:.4f}"
)
print(
    f"Log Loss modelo       : "
    f"{model_metrics['log_loss']:.4f}"
)
print(
    f"Taxa real             : "
    f"{model_metrics['taxa_real']:.2%}"
)
print(
    f"Probabilidade média   : "
    f"{model_metrics['probabilidade_media']:.2%}"
)
print(
    f"Erro de calibração    : "
    f"{model_metrics['erro_calibracao']:+.2%}"
)

print("\nComparação com o mercado:")

if np.isnan(market_brier):
    print("Brier mercado         : indisponível")
else:
    print(
        f"Brier mercado         : "
        f"{market_brier:.4f}"
    )
    print(
        f"Log Loss mercado      : "
        f"{market_log_loss:.4f}"
    )

print("\nMétricas financeiras:")
print(f"Edge mínimo           : {MIN_EDGE:.1%}")
print(f"Apostas               : {number_of_bets}")
print(f"Vitórias              : {number_of_wins}")
print(f"Taxa de acerto        : {hit_rate:.2f}%")
print(f"Odd média             : {average_odd:.2f}")
print(f"Edge médio            : {average_edge:.2%}")
print(f"Investido             : {invested:.2f}")
print(f"Lucro                 : {profit:.2f}")
print(f"ROI / Yield           : {roi:.2f}%")
print(
    f"Drawdown máximo       : "
    f"{max_drawdown:.2f}"
)

if np.isinf(profit_factor):
    print("Profit Factor         : infinito")
else:
    print(
        f"Profit Factor         : "
        f"{profit_factor:.2f}"
    )


if not season_report.empty:
    print("\n==========================================")
    print(" RESULTADOS POR ÉPOCA")
    print("==========================================\n")

    print(
        season_report.to_string(
            index=False,
            formatters={
                "lucro": "{:.2f}".format,
                "odd_media": "{:.2f}".format,
                "probabilidade_media":
                    "{:.2%}".format,
                "probabilidade_mercado":
                    "{:.2%}".format,
                "edge_medio": "{:.2%}".format,
                "taxa_acerto":
                    "{:.2f}%".format,
                "roi": "{:.2f}%".format,
                "yield": "{:.2f}%".format,
            },
        )
    )


if not league_report.empty:
    print("\n==========================================")
    print(" RESULTADOS POR LIGA")
    print("==========================================\n")

    print(
        league_report.to_string(
            index=False,
            formatters={
                "lucro": "{:.2f}".format,
                "odd_media": "{:.2f}".format,
                "probabilidade_media":
                    "{:.2%}".format,
                "probabilidade_mercado":
                    "{:.2%}".format,
                "edge_medio": "{:.2%}".format,
                "taxa_acerto":
                    "{:.2f}%".format,
                "roi": "{:.2f}%".format,
                "yield": "{:.2f}%".format,
            },
        )
    )


# ============================================================
# GUARDAR RELATÓRIOS
# ============================================================

predictions.to_csv(
    REPORTS_DIR
    / "walk_forward_previsoes_over25.csv",
    index=False,
)

bets.to_csv(
    REPORTS_DIR
    / "walk_forward_apostas_over25.csv",
    index=False,
)

fold_report.to_csv(
    REPORTS_DIR
    / "walk_forward_janelas_over25.csv",
    index=False,
)

season_report.to_csv(
    REPORTS_DIR
    / "walk_forward_por_epoca_over25.csv",
    index=False,
)

league_report.to_csv(
    REPORTS_DIR
    / "walk_forward_por_liga_over25.csv",
    index=False,
)

print("\nRelatórios guardados:")

print(
    "✅ reports/"
    "walk_forward_previsoes_over25.csv"
)

print(
    "✅ reports/"
    "walk_forward_apostas_over25.csv"
)

print(
    "✅ reports/"
    "walk_forward_janelas_over25.csv"
)

print(
    "✅ reports/"
    "walk_forward_por_epoca_over25.csv"
)

print(
    "✅ reports/"
    "walk_forward_por_liga_over25.csv"
)

print(
    "\n✅ Backtest walk-forward concluído."
)