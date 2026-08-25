from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

SIGNALS_PATH = (
    ROOT_DIR
    / "reports"
    / "clv_signals_over25.csv"
)

PREDICTIONS_PATH = (
    ROOT_DIR
    / "reports"
    / "clv_predictions_over25.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "reports"
    / "clv_bootstrap_summary.csv"
)

SEASON_OUTPUT_PATH = (
    ROOT_DIR
    / "reports"
    / "clv_validation_by_season.csv"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

N_BOOTSTRAPS = 5000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

STAKE = 1.0


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def confidence_interval(
    values: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """
    Calcula um intervalo de confiança utilizando
    os percentis das reamostragens bootstrap.
    """

    alpha = 1 - confidence_level

    lower = np.quantile(
        values,
        alpha / 2,
    )

    upper = np.quantile(
        values,
        1 - alpha / 2,
    )

    return float(lower), float(upper)


def format_percentage(
    value: float,
) -> str:
    """
    Formata um número decimal como percentagem.
    """

    return f"{value:.2%}"


# ============================================================
# VALIDAR FICHEIROS
# ============================================================

if not SIGNALS_PATH.exists():
    raise FileNotFoundError(
        f"Ficheiro não encontrado: {SIGNALS_PATH}"
    )

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Ficheiro não encontrado: {PREDICTIONS_PATH}"
    )


# ============================================================
# CARREGAR DADOS
# ============================================================

signals = pd.read_csv(
    SIGNALS_PATH,
    low_memory=False,
)

predictions = pd.read_csv(
    PREDICTIONS_PATH,
    low_memory=False,
)


# ============================================================
# VALIDAR COLUNAS
# ============================================================

required_signal_columns = [
    "date",
    "season",
    "clv",
    "profit",
]

missing_signal_columns = [
    column
    for column in required_signal_columns
    if column not in signals.columns
]

if missing_signal_columns:
    raise ValueError(
        "Faltam colunas no ficheiro de sinais: "
        + ", ".join(missing_signal_columns)
    )


required_prediction_columns = [
    "clv",
]

missing_prediction_columns = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]

if missing_prediction_columns:
    raise ValueError(
        "Faltam colunas no ficheiro de previsões: "
        + ", ".join(missing_prediction_columns)
    )


# ============================================================
# CONVERTER TIPOS
# ============================================================

signals["date"] = pd.to_datetime(
    signals["date"],
    format="%Y-%m-%d",
    errors="coerce",
)

signals["season"] = (
    signals["season"]
    .astype("string")
    .str.strip()
)

signals["clv"] = pd.to_numeric(
    signals["clv"],
    errors="coerce",
)

signals["profit"] = pd.to_numeric(
    signals["profit"],
    errors="coerce",
)

predictions["clv"] = pd.to_numeric(
    predictions["clv"],
    errors="coerce",
)

signals = signals.replace(
    [np.inf, -np.inf],
    np.nan,
)

predictions = predictions.replace(
    [np.inf, -np.inf],
    np.nan,
)


# ============================================================
# LIMPEZA
# ============================================================

signals = signals.dropna(
    subset=[
        "date",
        "season",
        "clv",
        "profit",
    ]
).copy()

predictions = predictions.dropna(
    subset=[
        "clv",
    ]
).copy()

signals = signals[
    signals["season"] != ""
].copy()

signals = signals.sort_values(
    "date"
).reset_index(drop=True)

if signals.empty:
    raise ValueError(
        "Não existem seleções válidas para analisar."
    )

if predictions.empty:
    raise ValueError(
        "Não existem previsões válidas para comparar."
    )


# ============================================================
# VARIÁVEIS DE ANÁLISE
# ============================================================

signals["clv_positive"] = (
    signals["clv"] > 0
).astype(int)

# Os jogos da mesma semana podem estar relacionados.
# O bootstrap será realizado por blocos semanais,
# em vez de assumir que todos os jogos são independentes.
signals["week"] = (
    signals["date"]
    .dt.to_period("W")
    .astype(str)
)


# ============================================================
# RESULTADOS OBSERVADOS
# ============================================================

number_of_selections = len(
    signals
)

observed_clv = float(
    signals["clv"].mean()
)

observed_median_clv = float(
    signals["clv"].median()
)

observed_positive_rate = float(
    signals["clv_positive"].mean()
)

observed_profit = float(
    signals["profit"].sum()
)

observed_roi = (
    observed_profit
    / (
        number_of_selections
        * STAKE
    )
)

all_games_clv = float(
    predictions["clv"].mean()
)

clv_lift = (
    observed_clv
    - all_games_clv
)


# ============================================================
# AGREGAR POR BLOCOS SEMANAIS
# ============================================================

blocks = (
    signals.groupby(
        [
            "season",
            "week",
        ],
        sort=False,
    )
    .agg(
        games=(
            "clv",
            "size",
        ),
        clv_sum=(
            "clv",
            "sum",
        ),
        positive_sum=(
            "clv_positive",
            "sum",
        ),
        profit_sum=(
            "profit",
            "sum",
        ),
    )
    .reset_index()
)

if blocks.empty:
    raise ValueError(
        "Não foi possível criar blocos semanais."
    )

season_blocks = {
    season: season_data.reset_index(
        drop=True
    )
    for season, season_data in blocks.groupby(
        "season",
        sort=False,
    )
}


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_clv = np.empty(
    N_BOOTSTRAPS,
    dtype=float,
)

bootstrap_positive_rate = np.empty(
    N_BOOTSTRAPS,
    dtype=float,
)

bootstrap_roi = np.empty(
    N_BOOTSTRAPS,
    dtype=float,
)

for iteration in range(
    N_BOOTSTRAPS
):
    total_games = 0
    total_clv = 0.0
    total_positive = 0
    total_profit = 0.0

    # Cada época é reamostrada separadamente.
    # Isso ajuda a manter a estrutura temporal dos dados.
    for season_data in season_blocks.values():
        number_of_blocks = len(
            season_data
        )

        if number_of_blocks == 0:
            continue

        sampled_indices = rng.integers(
            low=0,
            high=number_of_blocks,
            size=number_of_blocks,
        )

        sampled = season_data.iloc[
            sampled_indices
        ]

        total_games += int(
            sampled["games"].sum()
        )

        total_clv += float(
            sampled["clv_sum"].sum()
        )

        total_positive += int(
            sampled["positive_sum"].sum()
        )

        total_profit += float(
            sampled["profit_sum"].sum()
        )

    if total_games == 0:
        bootstrap_clv[iteration] = np.nan
        bootstrap_positive_rate[iteration] = np.nan
        bootstrap_roi[iteration] = np.nan

        continue

    bootstrap_clv[iteration] = (
        total_clv
        / total_games
    )

    bootstrap_positive_rate[iteration] = (
        total_positive
        / total_games
    )

    bootstrap_roi[iteration] = (
        total_profit
        / (
            total_games
            * STAKE
        )
    )


# ============================================================
# REMOVER RESULTADOS INVÁLIDOS
# ============================================================

bootstrap_clv = bootstrap_clv[
    np.isfinite(
        bootstrap_clv
    )
]

bootstrap_positive_rate = (
    bootstrap_positive_rate[
        np.isfinite(
            bootstrap_positive_rate
        )
    ]
)

bootstrap_roi = bootstrap_roi[
    np.isfinite(
        bootstrap_roi
    )
]

if (
    len(bootstrap_clv) == 0
    or len(bootstrap_positive_rate) == 0
    or len(bootstrap_roi) == 0
):
    raise RuntimeError(
        "O bootstrap não produziu resultados válidos."
    )


# ============================================================
# INTERVALOS DE CONFIANÇA
# ============================================================

clv_lower, clv_upper = confidence_interval(
    bootstrap_clv
)

positive_lower, positive_upper = (
    confidence_interval(
        bootstrap_positive_rate
    )
)

roi_lower, roi_upper = confidence_interval(
    bootstrap_roi
)


# ============================================================
# PROBABILIDADES BOOTSTRAP
# ============================================================

probability_clv_above_zero = float(
    np.mean(
        bootstrap_clv > 0
    )
)

probability_positive_rate_above_50 = float(
    np.mean(
        bootstrap_positive_rate > 0.50
    )
)

probability_roi_above_zero = float(
    np.mean(
        bootstrap_roi > 0
    )
)


# ============================================================
# RESULTADO POR ÉPOCA
# ============================================================

season_report = (
    signals.groupby(
        "season",
        sort=False,
    )
    .agg(
        selections=(
            "clv",
            "size",
        ),
        mean_clv=(
            "clv",
            "mean",
        ),
        median_clv=(
            "clv",
            "median",
        ),
        positive_clv_rate=(
            "clv_positive",
            "mean",
        ),
        profit=(
            "profit",
            "sum",
        ),
    )
    .reset_index()
)

season_report["roi"] = (
    season_report["profit"]
    / (
        season_report["selections"]
        * STAKE
    )
)

season_report.to_csv(
    SEASON_OUTPUT_PATH,
    index=False,
)


# ============================================================
# GUARDAR RESUMO
# ============================================================

summary = pd.DataFrame(
    [
        {
            "metric":
                "mean_clv",

            "observed":
                observed_clv,

            "confidence_lower":
                clv_lower,

            "confidence_upper":
                clv_upper,

            "reference":
                0.0,

            "probability_above_reference":
                probability_clv_above_zero,
        },
        {
            "metric":
                "positive_clv_rate",

            "observed":
                observed_positive_rate,

            "confidence_lower":
                positive_lower,

            "confidence_upper":
                positive_upper,

            "reference":
                0.50,

            "probability_above_reference":
                probability_positive_rate_above_50,
        },
        {
            "metric":
                "roi",

            "observed":
                observed_roi,

            "confidence_lower":
                roi_lower,

            "confidence_upper":
                roi_upper,

            "reference":
                0.0,

            "probability_above_reference":
                probability_roi_above_zero,
        },
    ]
)

summary.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n==========================================")
print(" VALIDAÇÃO ESTATÍSTICA DO CLV")
print("==========================================\n")

print(
    f"Seleções analisadas       : "
    f"{number_of_selections}"
)

print(
    f"Épocas                    : "
    f"{signals['season'].nunique()}"
)

print(
    f"Blocos semanais           : "
    f"{len(blocks)}"
)

print(
    f"Reamostragens bootstrap   : "
    f"{N_BOOTSTRAPS}"
)


# ============================================================
# CLV MÉDIO
# ============================================================

print("\nCLV médio:")

print(
    f"Observado                 : "
    f"{format_percentage(observed_clv)}"
)

print(
    f"Mediano                   : "
    f"{format_percentage(observed_median_clv)}"
)

print(
    f"Intervalo de confiança    : "
    f"[{format_percentage(clv_lower)}, "
    f"{format_percentage(clv_upper)}]"
)

print(
    f"Probabilidade CLV > 0     : "
    f"{probability_clv_above_zero:.2%}"
)


# ============================================================
# FREQUÊNCIA DE CLV POSITIVO
# ============================================================

print("\nPercentagem de CLV positivo:")

print(
    f"Observado                 : "
    f"{format_percentage(observed_positive_rate)}"
)

print(
    f"Intervalo de confiança    : "
    f"[{format_percentage(positive_lower)}, "
    f"{format_percentage(positive_upper)}]"
)

print(
    f"Probabilidade acima 50%   : "
    f"{probability_positive_rate_above_50:.2%}"
)


# ============================================================
# ROI
# ============================================================

print("\nROI secundário:")

print(
    f"Observado                 : "
    f"{format_percentage(observed_roi)}"
)

print(
    f"Intervalo de confiança    : "
    f"[{format_percentage(roi_lower)}, "
    f"{format_percentage(roi_upper)}]"
)

print(
    f"Probabilidade ROI > 0     : "
    f"{probability_roi_above_zero:.2%}"
)


# ============================================================
# COMPARAÇÃO COM TODOS OS JOGOS
# ============================================================

print("\nComparação com todos os jogos:")

print(
    f"CLV médio todos os jogos  : "
    f"{format_percentage(all_games_clv)}"
)

print(
    f"CLV médio das seleções    : "
    f"{format_percentage(observed_clv)}"
)

print(
    f"Melhoria das seleções     : "
    f"{format_percentage(clv_lift)}"
)


# ============================================================
# DECISÃO CORRIGIDA
# ============================================================

print("\n==========================================")
print(" DECISÃO")
print("==========================================\n")

if (
    clv_lower > 0
    and positive_lower > 0.50
    and roi_lower > 0
):
    print(
        "✅ O CLV, a frequência de CLV positivo "
        "e o ROI apresentam evidência estatística favorável."
    )

    print(
        "Ainda é necessária validação prospetiva "
        "com novos jogos antes de utilizar dinheiro real."
    )

elif (
    clv_lower > 0
    and positive_lower > 0.50
):
    print(
        "⚠ O CLV médio e a frequência de CLV positivo "
        "apresentam evidência favorável."
    )

    print(
        "No entanto, o ROI ainda não apresenta "
        "evidência estatística de rentabilidade."
    )

    print(
        "O sistema deve passar para paper trading "
        "sem apostas reais."
    )

elif clv_lower > 0:
    print(
        "⚠ O CLV médio é estatisticamente positivo, "
        "mas a percentagem de seleções com CLV positivo "
        "ainda não é conclusiva."
    )

    print(
        "O ROI também não apresenta evidência "
        "de rentabilidade."
    )

    print(
        "O sistema deve passar para acompanhamento "
        "prospetivo sem apostas reais."
    )

elif probability_clv_above_zero >= 0.90:
    print(
        "⚠ Existe um possível sinal de CLV, mas o "
        "intervalo de confiança do CLV médio inclui zero."
    )

    print(
        "O sistema deve permanecer em acompanhamento "
        "sem apostas reais."
    )

else:
    print(
        "❌ O sinal de CLV não é estatisticamente "
        "convincente."
    )

    print(
        "Não deve ser usado para selecionar apostas."
    )


# ============================================================
# FICHEIROS CRIADOS
# ============================================================

print("\nFicheiros guardados:")

print(
    "✅ reports/clv_bootstrap_summary.csv"
)

print(
    "✅ reports/clv_validation_by_season.csv"
)