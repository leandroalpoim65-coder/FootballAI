import sys
import sqlite3
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from ml.paper_trade import (
    MODEL_PATH,
    load_historical_matches,
    create_team_history,
    calculate_current_elo,
    build_sports_features,
    build_market_features,
)


DATABASE_PATH = (
    ROOT_DIR
    / "database"
    / "football.db"
)


# ============================================================
# JOGO DE TESTE
# ============================================================

LEAGUE = "Netherlands"

HOME_TEAM = "Twente"

AWAY_TEAM = "Zwolle"

FIXTURE_DATE = pd.Timestamp(
    "2026-08-16 13:30:00"
)


# ============================================================
# ODDS DA TABELA DO PAPER TRADE
# ============================================================

ODD_OVER = 1.34

ODD_UNDER = 3.04


# ============================================================
# CABEÇALHO
# ============================================================

print()
print(
    "======================================================="
)

print(
    "       DEBUG PAPER TRADE - TWENTE vs ZWOLLE"
)

print(
    "======================================================="
)


# ============================================================
# CARREGAR MODEL BUNDLE
# ============================================================

print()
print(
    "📦 A carregar model bundle..."
)


model_bundle = joblib.load(
    MODEL_PATH
)


required_keys = [
    "model",
    "base_features",
    "market_features",
    "model_features",
    "training_base_rate",
    "signal_threshold",
    "min_odd",
    "max_odd",
    "generated_at",
]


missing_keys = [
    key
    for key in required_keys
    if key not in model_bundle
]


if missing_keys:

    print()
    print(
        "❌ Faltam elementos no model bundle:"
    )

    for key in missing_keys:
        print(
            f"   - {key}"
        )

    raise SystemExit(1)


# ============================================================
# EXTRAIR CONFIGURAÇÃO DO BUNDLE
# ============================================================

model = model_bundle[
    "model"
]


base_features = list(
    model_bundle[
        "base_features"
    ]
)


market_features = list(
    model_bundle[
        "market_features"
    ]
)


model_features = list(
    model_bundle[
        "model_features"
    ]
)


training_base_rate = float(
    model_bundle[
        "training_base_rate"
    ]
)


signal_threshold = float(
    model_bundle[
        "signal_threshold"
    ]
)


min_odd = float(
    model_bundle[
        "min_odd"
    ]
)


max_odd = float(
    model_bundle[
        "max_odd"
    ]
)


generated_at = str(
    model_bundle[
        "generated_at"
    ]
)


# ============================================================
# MOSTRAR CONFIGURAÇÃO
# ============================================================

print()
print(
    f"Modelo bundle:"
)

print(
    MODEL_PATH
)


print()
print(
    f"Base features   : "
    f"{len(base_features)}"
)

print(
    f"Market features : "
    f"{len(market_features)}"
)

print(
    f"Model features  : "
    f"{len(model_features)}"
)


print()
print(
    f"Training base rate : "
    f"{training_base_rate:.6f} "
    f"({training_base_rate:.2%})"
)

print(
    f"Signal threshold   : "
    f"{signal_threshold:.6f} "
    f"({signal_threshold:.2%})"
)

print(
    f"Odd mínima         : "
    f"{min_odd:.2f}"
)

print(
    f"Odd máxima         : "
    f"{max_odd:.2f}"
)

print(
    f"Modelo gerado em   : "
    f"{generated_at}"
)


# ============================================================
# CARREGAR HISTÓRICO
# ============================================================

print()
print(
    "📚 A carregar histórico..."
)


connection = sqlite3.connect(
    DATABASE_PATH
)


try:

    matches = load_historical_matches(
        connection
    )

finally:

    connection.close()


matches[
    "date"
] = pd.to_datetime(
    matches[
        "date"
    ],
    errors="coerce",
)


# ============================================================
# APENAS JOGOS ANTERIORES AO FIXTURE
# ============================================================

matches_before = matches[
    matches[
        "date"
    ]
    < FIXTURE_DATE
].copy()


print(
    f"Jogos históricos usados: "
    f"{len(matches_before)}"
)


# ============================================================
# PREPARAR HISTÓRICO
# ============================================================

team_history = create_team_history(
    matches_before
)


elo_ratings = calculate_current_elo(
    matches_before
)


# ============================================================
# FEATURES DESPORTIVAS
# ============================================================

sports_features = (
    build_sports_features(
        history=team_history,
        elo_ratings=elo_ratings,
        league=LEAGUE,
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        fixture_date=FIXTURE_DATE,
    )
)


# ============================================================
# FEATURES DE MERCADO
# ============================================================

market_values = (
    build_market_features(
        ODD_OVER,
        ODD_UNDER,
    )
)


# ============================================================
# MOSTRAR FEATURES DE MERCADO
# ============================================================

print()
print(
    "======================================================="
)

print(
    "              FEATURES DE MERCADO"
)

print(
    "======================================================="
)


for key, value in (
    market_values.items()
):

    print(
        f"{key:<30} "
        f"= {value:.9f}"
    )


# ============================================================
# JUNTAR TUDO COMO O PAPER TRADE
# ============================================================

prediction_values = {
    **sports_features,
    **market_values,
    "league": LEAGUE,
}


prediction_frame = pd.DataFrame(
    [
        prediction_values
    ]
)


# ============================================================
# VERIFICAR FEATURES DO MODELO
# ============================================================

missing_model_features = [
    feature
    for feature in model_features
    if feature
    not in prediction_frame.columns
]


if missing_model_features:

    print()
    print(
        "❌ Features em falta:"
    )

    for feature in (
        missing_model_features
    ):

        print(
            f"   - {feature}"
        )

    raise SystemExit(1)


# ============================================================
# MOSTRAR FEATURES QUE O MODELO RECEBE
# ============================================================

print()
print(
    "======================================================="
)

print(
    "          FEATURES RECEBIDAS PELO MODELO"
)

print(
    "======================================================="
)


for number, feature in enumerate(
    model_features,
    start=1,
):

    value = prediction_frame.iloc[
        0
    ][feature]


    if isinstance(
        value,
        str,
    ):

        print(
            f"{number:02d}. "
            f"{feature:<32} "
            f"= {value}"
        )

    else:

        print(
            f"{number:02d}. "
            f"{feature:<32} "
            f"= {float(value):.6f}"
        )


# ============================================================
# PREVISÃO
# ============================================================

probabilities = model.predict_proba(
    prediction_frame[
        model_features
    ]
)


classes = list(
    model.classes_
)


if 1 not in classes:

    raise ValueError(
        "Modelo não contém "
        "a classe positiva 1."
    )


positive_index = (
    classes.index(1)
)


clv_probability = float(
    probabilities[
        0,
        positive_index,
    ]
)


# ============================================================
# EDGE
# ============================================================

edge = (
    clv_probability
    - training_base_rate
)


# ============================================================
# SINAL
# ============================================================

valid_odd = (
    min_odd
    <= ODD_OVER
    <= max_odd
)


signal = (
    valid_odd
    and (
        clv_probability
        >= signal_threshold
    )
)


# ============================================================
# RESULTADO
# ============================================================

print()
print(
    "======================================================="
)

print(
    "                    RESULTADO"
)

print(
    "======================================================="
)


print(
    f"Odd Over 2.5: "
    f"{ODD_OVER:.2f}"
)


print(
    f"Odd Under 2.5: "
    f"{ODD_UNDER:.2f}"
)


print()
print(
    f"Fair probability mercado: "
    f"{market_values['fair_probability_open']:.6f} "
    f"({market_values['fair_probability_open']:.2%})"
)


print(
    f"Paper Trade modelo      : "
    f"{clv_probability:.6f} "
    f"({clv_probability:.2%})"
)


print(
    f"Training base rate      : "
    f"{training_base_rate:.6f} "
    f"({training_base_rate:.2%})"
)


print(
    f"Edge sobre base rate    : "
    f"{edge:+.6f} "
    f"({edge:+.2%})"
)


print(
    f"Signal threshold        : "
    f"{signal_threshold:.6f} "
    f"({signal_threshold:.2%})"
)


print(
    f"Odd dentro do intervalo : "
    f"{valid_odd}"
)


print(
    f"Sinal                   : "
    f"{signal}"
)


print()
print(
    "======================================================="
)

print(
    "VALORES ESPERADOS DA TABELA:"
)

print(
    "Fair probability : "
    "0.694064 (69.41%)"
)

print(
    "Paper Trade      : "
    "0.447852 (44.79%)"
)

print(
    "Edge             : "
    "+0.009666 (+0.97%)"
)

print(
    "======================================================="
)