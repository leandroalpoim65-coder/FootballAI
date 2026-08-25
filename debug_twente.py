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
    sys.path.insert(0, str(ROOT_DIR))


from ml.paper_trade import (
    load_historical_matches,
    create_team_history,
    calculate_current_elo,
    build_sports_features,
)


DATABASE_PATH = (
    ROOT_DIR
    / "database"
    / "football.db"
)

MODEL_PRODUCAO_PATH = (
    ROOT_DIR
    / "ml"
    / "modelo_over25.pkl"
)

MODEL_VALIDACAO_PATH = (
    ROOT_DIR
    / "ml"
    / "modelo_over25_validacao.pkl"
)

FEATURES_PATH = (
    ROOT_DIR
    / "ml"
    / "features_over25.pkl"
)


# ============================================================
# JOGO QUE VAMOS TESTAR
# ============================================================

LEAGUE = "Netherlands"

HOME_TEAM = "Twente"

AWAY_TEAM = "Zwolle"

FIXTURE_DATE = pd.Timestamp(
    "2026-08-16 13:30:00"
)


# ============================================================
# FUNÇÃO PARA OBTER PROBABILIDADE DA CLASSE 1
# ============================================================

def get_probability(
    model,
    dataframe,
):

    probabilities = model.predict_proba(
        dataframe
    )

    classes = list(
        model.classes_
    )

    if 1 not in classes:
        raise ValueError(
            "O modelo não contém a classe 1."
        )

    positive_index = classes.index(
        1
    )

    return float(
        probabilities[
            0,
            positive_index,
        ]
    )


# ============================================================
# CARREGAR MODELOS
# ============================================================

print()
print(
    "======================================================="
)

print(
    "             DEBUG TWENTE vs ZWOLLE"
)

print(
    "======================================================="
)


modelo_producao = joblib.load(
    MODEL_PRODUCAO_PATH
)

modelo_validacao = joblib.load(
    MODEL_VALIDACAO_PATH
)

features_modelo = joblib.load(
    FEATURES_PATH
)


print()
print(
    f"Features esperadas pelo modelo: "
    f"{len(features_modelo)}"
)

print(
    "Classes produção:",
    modelo_producao.classes_,
)

print(
    "Classes validação:",
    modelo_validacao.classes_,
)


# ============================================================
# CARREGAR BASE DE DADOS
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


print(
    f"Jogos totais na base: "
    f"{len(matches)}"
)


# ============================================================
# GARANTIR DATAS
# ============================================================

matches["date"] = pd.to_datetime(
    matches["date"],
    errors="coerce",
)


# ============================================================
# IMPORTANTE:
# USAR APENAS JOGOS ANTERIORES AO TWENTE vs ZWOLLE
# ============================================================

matches_before = matches[
    matches["date"] < FIXTURE_DATE
].copy()


print(
    f"Jogos anteriores a "
    f"{FIXTURE_DATE}: "
    f"{len(matches_before)}"
)


# ============================================================
# VER ÚLTIMOS JOGOS DO TWENTE
# ============================================================

twente_matches = matches_before[
    (
        matches_before["league"]
        == LEAGUE
    )
    &
    (
        (
            matches_before["home_team"]
            == HOME_TEAM
        )
        |
        (
            matches_before["away_team"]
            == HOME_TEAM
        )
    )
].sort_values(
    "date"
)


print()
print(
    "======================================================="
)

print(
    "ÚLTIMOS 5 JOGOS DO TWENTE"
)

print(
    "======================================================="
)


print(
    twente_matches[
        [
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ]
    .tail(5)
    .to_string(
        index=False
    )
)


# ============================================================
# VER ÚLTIMOS JOGOS DO ZWOLLE
# ============================================================

zwolle_matches = matches_before[
    (
        matches_before["league"]
        == LEAGUE
    )
    &
    (
        (
            matches_before["home_team"]
            == AWAY_TEAM
        )
        |
        (
            matches_before["away_team"]
            == AWAY_TEAM
        )
    )
].sort_values(
    "date"
)


print()
print(
    "======================================================="
)

print(
    "ÚLTIMOS 5 JOGOS DO ZWOLLE"
)

print(
    "======================================================="
)


print(
    zwolle_matches[
        [
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ]
    .tail(5)
    .to_string(
        index=False
    )
)


# ============================================================
# CRIAR HISTÓRICO DAS EQUIPAS
# ============================================================

print()
print(
    "📊 A criar histórico das equipas..."
)


team_history = create_team_history(
    matches_before
)


# ============================================================
# CALCULAR ELO PRÉ-JOGO
# ============================================================

print(
    "📈 A calcular Elo pré-jogo..."
)


elo_ratings = calculate_current_elo(
    matches_before
)


# ============================================================
# CRIAR AS MESMAS FEATURES DO PAPER TRADE
# ============================================================

sports_features = build_sports_features(
    history=team_history,
    elo_ratings=elo_ratings,
    league=LEAGUE,
    home_team=HOME_TEAM,
    away_team=AWAY_TEAM,
    fixture_date=FIXTURE_DATE,
)


df_base = pd.DataFrame(
    [
        sports_features
    ]
)


# ============================================================
# VERIFICAR FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features_modelo
    if feature not in df_base.columns
]


if missing_features:

    print()
    print(
        "❌ Faltam features:"
    )

    for feature in missing_features:
        print(
            f"   - {feature}"
        )

    raise SystemExit(1)


# ============================================================
# COLOCAR NA ORDEM DO MODELO
# ============================================================

df_modelo = df_base[
    features_modelo
].copy()


df_modelo = df_modelo.apply(
    pd.to_numeric,
    errors="coerce",
)


if df_modelo.isna().any().any():

    print()
    print(
        "❌ Existem valores NaN."
    )

    print(
        df_modelo.columns[
            df_modelo.isna().any()
        ].tolist()
    )

    raise SystemExit(1)


# ============================================================
# MOSTRAR AS 43 FEATURES
# ============================================================

print()
print(
    "======================================================="
)

print(
    "                 43 FEATURES"
)

print(
    "======================================================="
)


for numero, feature in enumerate(
    features_modelo,
    start=1,
):

    valor = df_modelo.iloc[
        0
    ][feature]

    print(
        f"{numero:02d}. "
        f"{feature:<32} "
        f"= {valor:.6f}"
    )


# ============================================================
# PROBABILIDADE MODELO DE PRODUÇÃO
# ============================================================

prob_producao = get_probability(
    modelo_producao,
    df_modelo,
)


# ============================================================
# PROBABILIDADE MODELO DE VALIDAÇÃO
# ============================================================

prob_validacao = get_probability(
    modelo_validacao,
    df_modelo,
)


# ============================================================
# RESULTADO
# ============================================================

print()
print(
    "======================================================="
)

print(
    "                 PROBABILIDADES"
)

print(
    "======================================================="
)


print(
    f"Modelo PRODUÇÃO : "
    f"{prob_producao:.6f} "
    f"({prob_producao * 100:.2f}%)"
)


print(
    f"Modelo VALIDAÇÃO: "
    f"{prob_validacao:.6f} "
    f"({prob_validacao * 100:.2f}%)"
)


print()
print(
    "Valores que já vimos anteriormente:"
)

print(
    "Paper Trade          : "
    "0.447852  (44.79%)"
)

print(
    "Get Today's Matches  : "
    "aprox. 59.8%"
)


print()
print(
    "======================================================="
)

print(
    "INTERPRETAÇÃO:"
)

print(
    "Se PRODUÇÃO ficar perto de 59.8% "
    "e VALIDAÇÃO perto de 44.8%,"
)

print(
    "então as features estão corretas e "
    "a diferença vem dos modelos."
)

print()
print(
    "Se nenhum deles ficar perto de 44.8%, "
    "então vamos comparar as features "
    "do Paper Trade."
)

print(
    "======================================================="
)