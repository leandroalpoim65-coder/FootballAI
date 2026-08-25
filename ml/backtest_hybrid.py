from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

ODD_OVER_OPEN = "odd_over25_open"
ODD_UNDER_OPEN = "odd_under25_open"

ODD_OVER_CLOSE = "odd_over25_close"
ODD_UNDER_CLOSE = "odd_under25_close"

# Número mínimo de épocas anteriores para iniciar os testes.
MIN_TRAIN_SEASONS = 5

# Para existir aposta, o modelo precisa de apresentar:
# - pelo menos 3% de valor esperado;
# - pelo menos 2 pontos percentuais acima do mercado justo.
MIN_EXPECTED_VALUE = 0.03
MIN_EDGE_MARKET = 0.02

MIN_ODD = 1.30
MAX_ODD = 4.00

STAKE = 1.0
RANDOM_STATE = 42

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def create_model() -> Pipeline:
    """
    Modelo conservador.

    A regressão logística é apropriada para fazer pequenas
    correções à previsão do mercado, reduzindo o risco de
    overfitting.
    """
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.10,
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def positive_probability(
    model: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Obtém a probabilidade correspondente à classe 1.
    """
    probabilities = model.predict_proba(features)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "O modelo não possui a classe positiva 1."
        )

    positive_index = classes.index(1)

    return probabilities[:, positive_index]


def calculate_metrics(
    target: pd.Series,
    probabilities: pd.Series | np.ndarray,
) -> dict:
    """
    Calcula métricas de classificação e probabilidade.
    """
    target_array = target.astype(int).to_numpy()

    probabilities_array = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities_array = np.clip(
        probabilities_array,
        0.000001,
        0.999999,
    )

    predictions = (
        probabilities_array >= 0.50
    ).astype(int)

    if len(np.unique(target_array)) >= 2:
        auc = roc_auc_score(
            target_array,
            probabilities_array,
        )
    else:
        auc = np.nan

    return {
        "accuracy": float(
            accuracy_score(
                target_array,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                target_array,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target_array,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(auc),
        "brier": float(
            brier_score_loss(
                target_array,
                probabilities_array,
            )
        ),
        "log_loss": float(
            log_loss(
                target_array,
                probabilities_array,
                labels=[0, 1],
            )
        ),
        "taxa_real": float(
            target_array.mean()
        ),
        "probabilidade_media": float(
            probabilities_array.mean()
        ),
    }


def calculate_max_drawdown(
    profits: pd.Series,
) -> float:
    """
    Calcula a maior queda da banca desde o máximo anterior.
    """
    if profits.empty:
        return 0.0

    cumulative = profits.cumsum().to_numpy()

    curve = np.concatenate(
        [
            np.array([0.0]),
            cumulative,
        ]
    )

    peaks = np.maximum.accumulate(curve)

    drawdowns = peaks - curve

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
        return (
            float("inf")
            if gross_profit > 0
            else 0.0
        )

    return gross_profit / gross_loss


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
            apostas=("profit", "size"),
            vitorias=("won", "sum"),
            lucro=("profit", "sum"),
            odd_media=(ODD_OVER_OPEN, "mean"),
            prob_modelo_media=(
                "model_probability",
                "mean",
            ),
            prob_mercado_media=(
                "market_probability_open",
                "mean",
            ),
            edge_medio=(
                "edge_market",
                "mean",
            ),
            ev_medio=(
                "expected_value",
                "mean",
            ),
            clv_medio=(
                "clv",
                "mean",
            ),
            clv_positivo=(
                "clv",
                lambda values: (
                    (
                        values.dropna() > 0
                    ).mean()
                    * 100
                    if values.notna().any()
                    else np.nan
                ),
            ),
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

base_features = joblib.load(
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
# CONVERSÃO E LIMPEZA
# ============================================================

data["date"] = pd.to_datetime(
    data["date"],
    format="%Y-%m-%d",
    errors="coerce",
)

numeric_columns = [
    TARGET,
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

data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="coerce",
)

# ============================================================
# PROBABILIDADE JUSTA DO MERCADO DE ABERTURA
# ============================================================

valid_open_odds = (
    (data[ODD_OVER_OPEN] > 1.0)
    & (data[ODD_UNDER_OPEN] > 1.0)
)

data["implied_over_open"] = np.where(
    valid_open_odds,
    1 / data[ODD_OVER_OPEN],
    np.nan,
)

data["implied_under_open"] = np.where(
    valid_open_odds,
    1 / data[ODD_UNDER_OPEN],
    np.nan,
)

data["market_margin_open"] = (
    data["implied_over_open"]
    + data["implied_under_open"]
    - 1
)

data["market_probability_open"] = (
    data["implied_over_open"]
    / (
        data["implied_over_open"]
        + data["implied_under_open"]
    )
)

# Logit da probabilidade do mercado.
clipped_market_probability = (
    data["market_probability_open"]
    .clip(
        lower=0.0001,
        upper=0.9999,
    )
)

data["market_logit_open"] = np.log(
    clipped_market_probability
    / (
        1 - clipped_market_probability
    )
)


# ============================================================
# PROBABILIDADE JUSTA DO MERCADO DE FECHO
# ============================================================

valid_close_odds = (
    (data[ODD_OVER_CLOSE] > 1.0)
    & (data[ODD_UNDER_CLOSE] > 1.0)
)

data["implied_over_close"] = np.where(
    valid_close_odds,
    1 / data[ODD_OVER_CLOSE],
    np.nan,
)

data["implied_under_close"] = np.where(
    valid_close_odds,
    1 / data[ODD_UNDER_CLOSE],
    np.nan,
)

data["market_probability_close"] = (
    data["implied_over_close"]
    / (
        data["implied_over_close"]
        + data["implied_under_close"]
    )
)


# ============================================================
# FEATURES HÍBRIDAS
# ============================================================

market_features = [
    "market_logit_open",
    "market_margin_open",
]

features = [
    *base_features,
    *market_features,
]

before_cleaning = len(data)

data = data.dropna(
    subset=[
        "date",
        "season",
        TARGET,
        ODD_OVER_OPEN,
        ODD_UNDER_OPEN,
        *features,
    ]
).copy()

data[TARGET] = data[TARGET].astype(int)

data = data[
    data[TARGET].isin([0, 1])
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
        "Não existem jogos com odds de abertura completas."
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
print(" FOOTBALLAI — BACKTEST HÍBRIDO")
print("==========================================\n")

print(f"Jogos no dataset       : {len(data)}")
print(f"Linhas sem abertura    : {removed_rows}")
print(f"Features desportivas   : {len(base_features)}")
print(f"Features do mercado    : {len(market_features)}")
print(f"Épocas disponíveis     : {len(seasons)}")
print(f"Épocas de teste        : {len(test_seasons)}")
print(f"EV mínimo              : {MIN_EXPECTED_VALUE:.1%}")
print(f"Edge mínimo mercado    : {MIN_EDGE_MARKET:.1%}")

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

    if train[TARGET].nunique() < 2:
        continue

    model = create_model()

    model.fit(
        train[features],
        train[TARGET],
    )

    test["model_probability"] = (
        positive_probability(
            model,
            test[features],
        )
    )

    test["market_probability"] = (
        test["market_probability_open"]
    )

    # Edge do modelo relativamente à probabilidade justa
    # de abertura.
    test["edge_market"] = (
        test["model_probability"]
        - test["market_probability_open"]
    )

    # Valor esperado usando a odd que estaria disponível
    # no momento da abertura.
    test["expected_value"] = (
        test["model_probability"]
        * test[ODD_OVER_OPEN]
        - 1
    )

    valid_betting_odds = (
        (test[ODD_OVER_OPEN] >= MIN_ODD)
        & (test[ODD_OVER_OPEN] <= MAX_ODD)
    )

    test["bet"] = (
        valid_betting_odds
        & (
            test["expected_value"]
            >= MIN_EXPECTED_VALUE
        )
        & (
            test["edge_market"]
            >= MIN_EDGE_MARKET
        )
    )

    test["won"] = (
        test[TARGET] == 1
    ).astype(int)

    test["profit"] = np.where(
        test["bet"],
        np.where(
            test["won"] == 1,
            (
                test[ODD_OVER_OPEN] - 1
            )
            * STAKE,
            -STAKE,
        ),
        0.0,
    )

    # Closing Line Value:
    # positivo quando a odd de abertura obtida foi superior
    # à odd de fecho.
    test["clv"] = np.where(
        (
            test[ODD_OVER_CLOSE].notna()
            & (test[ODD_OVER_CLOSE] > 1.0)
        ),
        (
            test[ODD_OVER_OPEN]
            / test[ODD_OVER_CLOSE]
        )
        - 1,
        np.nan,
    )

    # Movimento da probabilidade do mercado para o Over.
    test["market_probability_movement"] = (
        test["market_probability_close"]
        - test["market_probability_open"]
    )

    test["test_season"] = test_season

    all_predictions.append(test)

    model_metrics = calculate_metrics(
        test[TARGET],
        test["model_probability"],
    )

    market_metrics = calculate_metrics(
        test[TARGET],
        test["market_probability_open"],
    )

    fold_bets = test[
        test["bet"]
    ].copy()

    fold_profit = float(
        fold_bets["profit"].sum()
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

    fold_clv = (
        float(
            fold_bets["clv"].mean()
        )
        if fold_bets["clv"].notna().any()
        else np.nan
    )

    fold_results.append(
        {
            "test_season": test_season,
            "training_seasons":
                len(training_seasons),
            "train_matches": len(train),
            "test_matches": len(test),
            "model_auc":
                model_metrics["roc_auc"],
            "market_auc":
                market_metrics["roc_auc"],
            "model_brier":
                model_metrics["brier"],
            "market_brier":
                market_metrics["brier"],
            "model_log_loss":
                model_metrics["log_loss"],
            "market_log_loss":
                market_metrics["log_loss"],
            "bets": len(fold_bets),
            "profit": fold_profit,
            "roi": fold_roi,
            "average_clv": fold_clv,
        }
    )

    print("\n------------------------------------------")
    print(f"Época de teste     : {test_season}")
    print(f"Treino             : {len(train)} jogos")
    print(f"Teste              : {len(test)} jogos")
    print(
        f"ROC-AUC modelo     : "
        f"{model_metrics['roc_auc']:.4f}"
    )
    print(
        f"ROC-AUC mercado    : "
        f"{market_metrics['roc_auc']:.4f}"
    )
    print(
        f"Brier modelo       : "
        f"{model_metrics['brier']:.4f}"
    )
    print(
        f"Brier mercado      : "
        f"{market_metrics['brier']:.4f}"
    )
    print(f"Apostas            : {len(fold_bets)}")
    print(f"Lucro              : {fold_profit:.2f}")
    print(f"ROI                : {fold_roi:.2f}%")

    if np.isnan(fold_clv):
        print("CLV médio          : indisponível")
    else:
        print(f"CLV médio          : {fold_clv:.2%}")


# ============================================================
# RESULTADO GLOBAL
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

bets = predictions[
    predictions["bet"]
].copy()

fold_report = pd.DataFrame(
    fold_results
)

global_model_metrics = calculate_metrics(
    predictions[TARGET],
    predictions["model_probability"],
)

global_market_metrics = calculate_metrics(
    predictions[TARGET],
    predictions["market_probability_open"],
)


# ============================================================
# MÉTRICAS FINANCEIRAS
# ============================================================

number_of_bets = len(bets)

number_of_wins = int(
    bets["won"].sum()
)

invested = (
    number_of_bets
    * STAKE
)

profit = float(
    bets["profit"].sum()
)

roi = (
    profit / invested * 100
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
        bets[ODD_OVER_OPEN].mean()
    )
    if number_of_bets > 0
    else 0.0
)

average_edge = (
    float(
        bets["edge_market"].mean()
    )
    if number_of_bets > 0
    else 0.0
)

average_ev = (
    float(
        bets["expected_value"].mean()
    )
    if number_of_bets > 0
    else 0.0
)

average_clv = (
    float(
        bets["clv"].mean()
    )
    if bets["clv"].notna().any()
    else np.nan
)

positive_clv_rate = (
    float(
        (
            bets.loc[
                bets["clv"].notna(),
                "clv",
            ]
            > 0
        ).mean()
        * 100
    )
    if bets["clv"].notna().any()
    else np.nan
)

max_drawdown = calculate_max_drawdown(
    bets["profit"]
)

profit_factor = calculate_profit_factor(
    bets["profit"]
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
print(" RESULTADO GLOBAL HÍBRIDO")
print("==========================================\n")

print("Comparação probabilística:")
print(
    f"Jogos previstos       : "
    f"{len(predictions)}"
)
print(
    f"ROC-AUC modelo        : "
    f"{global_model_metrics['roc_auc']:.4f}"
)
print(
    f"ROC-AUC mercado       : "
    f"{global_market_metrics['roc_auc']:.4f}"
)
print(
    f"Brier modelo          : "
    f"{global_model_metrics['brier']:.4f}"
)
print(
    f"Brier mercado         : "
    f"{global_market_metrics['brier']:.4f}"
)
print(
    f"Log Loss modelo       : "
    f"{global_model_metrics['log_loss']:.4f}"
)
print(
    f"Log Loss mercado      : "
    f"{global_market_metrics['log_loss']:.4f}"
)
print(
    f"Taxa real             : "
    f"{global_model_metrics['taxa_real']:.2%}"
)
print(
    f"Probabilidade modelo  : "
    f"{global_model_metrics['probabilidade_media']:.2%}"
)
print(
    f"Probabilidade mercado : "
    f"{global_market_metrics['probabilidade_media']:.2%}"
)

print("\nMétricas financeiras:")
print(
    f"EV mínimo             : "
    f"{MIN_EXPECTED_VALUE:.1%}"
)
print(
    f"Edge mínimo mercado   : "
    f"{MIN_EDGE_MARKET:.1%}"
)
print(f"Apostas               : {number_of_bets}")
print(f"Vitórias              : {number_of_wins}")
print(f"Taxa de acerto        : {hit_rate:.2f}%")
print(f"Odd média abertura    : {average_odd:.2f}")
print(f"Edge médio            : {average_edge:.2%}")
print(f"EV médio estimado     : {average_ev:.2%}")
print(f"Investido             : {invested:.2f}")
print(f"Lucro                 : {profit:.2f}")
print(f"ROI / Yield           : {roi:.2f}%")
print(f"Drawdown máximo       : {max_drawdown:.2f}")

if np.isinf(profit_factor):
    print("Profit Factor         : infinito")
else:
    print(
        f"Profit Factor         : "
        f"{profit_factor:.2f}"
    )

if np.isnan(average_clv):
    print("CLV médio             : indisponível")
    print("CLV positivo          : indisponível")
else:
    print(f"CLV médio             : {average_clv:.2%}")
    print(
        f"CLV positivo          : "
        f"{positive_clv_rate:.2f}%"
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
                "prob_modelo_media":
                    "{:.2%}".format,
                "prob_mercado_media":
                    "{:.2%}".format,
                "edge_medio":
                    "{:.2%}".format,
                "ev_medio":
                    "{:.2%}".format,
                "clv_medio":
                    "{:.2%}".format,
                "clv_positivo":
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
    print(" RESULTADOS POR LIGA")
    print("==========================================\n")

    print(
        league_report.to_string(
            index=False,
            formatters={
                "lucro": "{:.2f}".format,
                "odd_media": "{:.2f}".format,
                "prob_modelo_media":
                    "{:.2%}".format,
                "prob_mercado_media":
                    "{:.2%}".format,
                "edge_medio":
                    "{:.2%}".format,
                "ev_medio":
                    "{:.2%}".format,
                "clv_medio":
                    "{:.2%}".format,
                "clv_positivo":
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
    / "hybrid_predictions_over25.csv",
    index=False,
)

bets.to_csv(
    REPORTS_DIR
    / "hybrid_bets_over25.csv",
    index=False,
)

fold_report.to_csv(
    REPORTS_DIR
    / "hybrid_folds_over25.csv",
    index=False,
)

season_report.to_csv(
    REPORTS_DIR
    / "hybrid_by_season_over25.csv",
    index=False,
)

league_report.to_csv(
    REPORTS_DIR
    / "hybrid_by_league_over25.csv",
    index=False,
)

print("\nRelatórios guardados:")

print(
    "✅ reports/hybrid_predictions_over25.csv"
)

print(
    "✅ reports/hybrid_bets_over25.csv"
)

print(
    "✅ reports/hybrid_folds_over25.csv"
)

print(
    "✅ reports/hybrid_by_season_over25.csv"
)

print(
    "✅ reports/hybrid_by_league_over25.csv"
)

print(
    "\n✅ Backtest híbrido concluído."
)