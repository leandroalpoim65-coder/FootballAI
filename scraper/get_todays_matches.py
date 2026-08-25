import sys
import sqlite3
from datetime import date
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from ml.paper_trade import (
    load_historical_matches,
    create_team_history,
    calculate_current_elo,
    build_sports_features,
    normalize_league,
)


DATABASE_PATH = ROOT_DIR / "database" / "football.db"

MODEL_BTTS_PATH = ROOT_DIR / "ml" / "modelo_btts.pkl"
MODEL_OVER25_PATH = ROOT_DIR / "ml" / "modelo_over25.pkl"

FEATURES_BTTS_PATH = ROOT_DIR / "ml" / "features_btts.pkl"
FEATURES_OVER25_PATH = ROOT_DIR / "ml" / "features_over25.pkl"

UPCOMING_PATH = (
    ROOT_DIR
    / "data"
    / "upcoming"
    / "upcoming_matches.csv"
)


# ============================================================
# LIMITES DOS SINAIS
# ============================================================

BTTS_ALTO = 65.0
BTTS_BAIXO = 35.0

OVER_ALTO = 65.0
OVER_BAIXO = 35.0


# ============================================================
# ALIASES DE LIGAS CONFIRMADOS
# ============================================================

LEAGUE_ALIASES_EXTRA = {

    # --------------------------------------------------------
    # INGLATERRA
    # --------------------------------------------------------

    "e1": "England Championship",

    "championship":
        "England Championship",

    # --------------------------------------------------------
    # BÉLGICA
    # --------------------------------------------------------

    "b1":
        "Belgium First Division A",

    "belgium first div":
        "Belgium First Division A",

    "belgium first division":
        "Belgium First Division A",

    # --------------------------------------------------------
    # ALEMANHA
    # --------------------------------------------------------

    "d2":
        "Germany 2 Bundesliga",

    # --------------------------------------------------------
    # ESPANHA
    # --------------------------------------------------------

    "sp2":
        "Spain Segunda Division",

    # --------------------------------------------------------
    # TURQUIA
    # --------------------------------------------------------

    "t1":
        "Turkey Super Lig",
}


# ============================================================
# ALIASES DE EQUIPAS CONFIRMADOS
# ============================================================

TEAM_ALIASES_EXTRA = {

    (
        "Belgium First Division A",
        "Beveren",
    ):
        "Waasland-Beveren",
}


# ============================================================
# NORMALIZAR LIGA
# ============================================================

def normalizar_liga_jogo(
    league_value,
    division_value=None,
):

    league = None

    if (
        league_value is not None
        and not pd.isna(league_value)
    ):

        league = str(
            league_value
        ).strip()


    division = None

    if (
        division_value is not None
        and not pd.isna(division_value)
    ):

        division = str(
            division_value
        ).strip()


    # --------------------------------------------------------
    # ALIASES EXTRA
    # --------------------------------------------------------

    if league:

        league_lower = league.lower()

        if league_lower in LEAGUE_ALIASES_EXTRA:

            return LEAGUE_ALIASES_EXTRA[
                league_lower
            ]


    if division:

        division_lower = division.lower()

        if division_lower in LEAGUE_ALIASES_EXTRA:

            return LEAGUE_ALIASES_EXTRA[
                division_lower
            ]


    # --------------------------------------------------------
    # NORMALIZAÇÃO DO PAPER TRADE
    # --------------------------------------------------------

    return normalize_league(
        league_value=league,
        division_value=division,
    )


# ============================================================
# NORMALIZAR EQUIPA
# ============================================================

def normalizar_equipa(
    league,
    team,
):

    team = str(
        team
    ).strip()


    key = (
        league,
        team,
    )


    if key in TEAM_ALIASES_EXTRA:

        return TEAM_ALIASES_EXTRA[
            key
        ]


    return team


# ============================================================
# PROBABILIDADE DA CLASSE POSITIVA
# ============================================================

def probabilidade_classe_positiva(
    modelo,
    dataframe,
):

    probabilities = modelo.predict_proba(
        dataframe
    )


    classes = list(
        modelo.classes_
    )


    if 1 not in classes:

        raise ValueError(
            "O modelo não contém "
            "a classe positiva 1."
        )


    positive_index = classes.index(
        1
    )


    probability = probabilities[
        0,
        positive_index,
    ]


    return float(
        probability * 100
    )


# ============================================================
# CARREGAR MODELOS
# ============================================================

try:

    modelo_btts = joblib.load(
        MODEL_BTTS_PATH
    )

    modelo_over25 = joblib.load(
        MODEL_OVER25_PATH
    )

    features_btts = joblib.load(
        FEATURES_BTTS_PATH
    )

    features_over25 = joblib.load(
        FEATURES_OVER25_PATH
    )


except FileNotFoundError as erro:

    print()
    print(
        "❌ Não foi possível carregar "
        "os modelos ou as features."
    )

    print(
        f"Motivo: {erro}"
    )

    raise SystemExit(1)


print()
print(
    "🧠 Modelos carregados com sucesso!"
)

print(
    "Classes BTTS:",
    modelo_btts.classes_,
)

print(
    "Classes OVER:",
    modelo_over25.classes_,
)

print(
    f"Features BTTS: "
    f"{len(features_btts)}"
)

print(
    f"Features OVER: "
    f"{len(features_over25)}"
)


# ============================================================
# VERIFICAR FICHEIROS
# ============================================================

if not DATABASE_PATH.exists():

    print()
    print(
        "❌ Base de dados não encontrada:"
    )

    print(
        DATABASE_PATH
    )

    raise SystemExit(1)


if not UPCOMING_PATH.exists():

    print()
    print(
        "❌ Ficheiro de jogos futuros "
        "não encontrado:"
    )

    print(
        UPCOMING_PATH
    )

    raise SystemExit(1)


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


if matches.empty:

    print()
    print(
        "❌ Não existem jogos históricos "
        "na base de dados."
    )

    raise SystemExit(1)


print(
    f"Jogos históricos: "
    f"{len(matches)}"
)


# ============================================================
# LIGAS DISPONÍVEIS
# ============================================================

ligas_disponiveis = set(
    matches[
        "league"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)


print(
    f"Ligas disponíveis na base: "
    f"{len(ligas_disponiveis)}"
)


# ============================================================
# HISTÓRICO DAS EQUIPAS
# ============================================================

print(
    "📊 A preparar histórico "
    "das equipas..."
)


team_history = create_team_history(
    matches
)


print(
    f"Registos de equipas: "
    f"{len(team_history)}"
)


# ============================================================
# ELO
# ============================================================

print(
    "📈 A calcular ratings Elo..."
)


elo_ratings = calculate_current_elo(
    matches
)


print(
    f"Equipas com Elo: "
    f"{len(elo_ratings)}"
)


# ============================================================
# CARREGAR JOGOS FUTUROS
# ============================================================

upcoming = pd.read_csv(
    UPCOMING_PATH
)


if upcoming.empty:

    print()
    print(
        "⚠ O ficheiro de jogos futuros "
        "está vazio."
    )

    raise SystemExit(0)


# ============================================================
# VERIFICAR COLUNAS
# ============================================================

required_columns = [
    "date",
    "league",
    "home_team",
    "away_team",
]


missing_columns = [
    column
    for column in required_columns
    if column not in upcoming.columns
]


if missing_columns:

    print()
    print(
        "❌ Faltam colunas no ficheiro:"
    )

    for column in missing_columns:

        print(
            f"   - {column}"
        )

    raise SystemExit(1)


# ============================================================
# DATAS
# ============================================================

upcoming[
    "date"
] = pd.to_datetime(
    upcoming["date"],
    errors="coerce",
)


upcoming = upcoming.dropna(
    subset=[
        "date",
        "home_team",
        "away_team",
    ]
).copy()


hoje = pd.Timestamp(
    date.today()
)


jogos_hoje = upcoming[
    upcoming[
        "date"
    ].dt.normalize()
    == hoje.normalize()
].copy()


# ============================================================
# CABEÇALHO
# ============================================================

print()
print(
    "======================================================="
)

print(
    f" JOGOS DO DIA: "
    f"{hoje.date()}"
)

print(
    "======================================================="
)

print(
    f"Jogos encontrados: "
    f"{len(jogos_hoje)}"
)


if jogos_hoje.empty:

    print()
    print(
        "Nenhum jogo encontrado "
        "para hoje."
    )

    raise SystemExit(0)


# ============================================================
# ALERTAS
# ============================================================

alertas_btts = []

alertas_over = []


# ============================================================
# CONTADORES
# ============================================================

analisados = 0

ignorados = 0

ligas_nao_suportadas = 0

historico_insuficiente = 0


# ============================================================
# ANALISAR JOGOS
# ============================================================

for _, jogo in jogos_hoje.iterrows():

    # --------------------------------------------------------
    # LIGA ORIGINAL
    # --------------------------------------------------------

    league_raw = str(
        jogo.get(
            "league",
            "",
        )
    ).strip()


    # --------------------------------------------------------
    # DIVISION
    # --------------------------------------------------------

    division_raw = None


    if "division" in jogo.index:

        division_raw = jogo.get(
            "division"
        )


    elif "Div" in jogo.index:

        division_raw = jogo.get(
            "Div"
        )


    # --------------------------------------------------------
    # NORMALIZAR LIGA
    # --------------------------------------------------------

    league = normalizar_liga_jogo(
        league_value=league_raw,
        division_value=division_raw,
    )


    if league is None:

        print()
        print(
            f"⚠ Liga inválida: "
            f"{league_raw}"
        )

        ignorados += 1

        continue


    # --------------------------------------------------------
    # EQUIPAS ORIGINAIS
    # --------------------------------------------------------

    home_team_raw = str(
        jogo[
            "home_team"
        ]
    ).strip()


    away_team_raw = str(
        jogo[
            "away_team"
        ]
    ).strip()


    # --------------------------------------------------------
    # NORMALIZAR EQUIPAS
    # --------------------------------------------------------

    home_team = normalizar_equipa(
        league,
        home_team_raw,
    )


    away_team = normalizar_equipa(
        league,
        away_team_raw,
    )


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    fixture_date = pd.Timestamp(
        jogo["date"]
    )


    # --------------------------------------------------------
    # MOSTRAR JOGO
    # --------------------------------------------------------

    print()
    print(
        f"🔎 [{league_raw} → {league}] "
        f"{home_team_raw} vs "
        f"{away_team_raw}"
    )


    # --------------------------------------------------------
    # MOSTRAR ALTERAÇÃO DO NOME
    # --------------------------------------------------------

    if home_team != home_team_raw:

        print(
            f"   🔄 Equipa casa: "
            f"{home_team_raw} "
            f"→ {home_team}"
        )


    if away_team != away_team_raw:

        print(
            f"   🔄 Equipa fora: "
            f"{away_team_raw} "
            f"→ {away_team}"
        )


    # ========================================================
    # VERIFICAR LIGA
    # ========================================================

    if league not in ligas_disponiveis:

        print(
            "   ⛔ Liga não suportada "
            "pela base histórica."
        )

        ignorados += 1

        ligas_nao_suportadas += 1

        continue


    # ========================================================
    # CRIAR FEATURES
    # ========================================================

    try:

        sports_features = build_sports_features(
            history=team_history,
            elo_ratings=elo_ratings,
            league=league,
            home_team=home_team,
            away_team=away_team,
            fixture_date=fixture_date,
        )


    except Exception as erro:

        print(
            "   ⚠ Não foi possível criar "
            "as features deste jogo."
        )

        print(
            f"   Motivo: {erro}"
        )

        ignorados += 1

        historico_insuficiente += 1

        continue


    # ========================================================
    # DATAFRAME
    # ========================================================

    df_base = pd.DataFrame(
        [
            sports_features
        ]
    )


    # ========================================================
    # VERIFICAR FEATURES BTTS
    # ========================================================

    missing_btts = [
        feature
        for feature in features_btts
        if feature not in df_base.columns
    ]


    if missing_btts:

        print(
            "   ⚠ Faltam features BTTS:"
        )

        for feature in missing_btts:

            print(
                f"      - {feature}"
            )


    # ========================================================
    # VERIFICAR FEATURES OVER
    # ========================================================

    missing_over = [
        feature
        for feature in features_over25
        if feature not in df_base.columns
    ]


    if missing_over:

        print(
            "   ⚠ Faltam features OVER:"
        )

        for feature in missing_over:

            print(
                f"      - {feature}"
            )


    if missing_btts or missing_over:

        print(
            "   ⚠ Jogo ignorado porque "
            "as features não coincidem "
            "com o modelo."
        )

        ignorados += 1

        continue


    # ========================================================
    # ORDEM DAS FEATURES
    # ========================================================

    df_btts = df_base[
        features_btts
    ].copy()


    df_over = df_base[
        features_over25
    ].copy()


    # ========================================================
    # CONVERTER PARA NÚMEROS
    # ========================================================

    df_btts = df_btts.apply(
        pd.to_numeric,
        errors="coerce",
    )


    df_over = df_over.apply(
        pd.to_numeric,
        errors="coerce",
    )


    # ========================================================
    # VALIDAR BTTS
    # ========================================================

    if df_btts.isna().any().any():

        print(
            "   ⚠ Existem valores inválidos "
            "nas features BTTS."
        )

        ignorados += 1

        continue


    # ========================================================
    # VALIDAR OVER
    # ========================================================

    if df_over.isna().any().any():

        print(
            "   ⚠ Existem valores inválidos "
            "nas features OVER."
        )

        ignorados += 1

        continue


    # ========================================================
    # PREVISÃO BTTS
    # ========================================================

    try:

        prob_btts = (
            probabilidade_classe_positiva(
                modelo_btts,
                df_btts,
            )
        )


    except Exception as erro:

        print(
            f"   ❌ Erro no modelo BTTS: "
            f"{erro}"
        )

        ignorados += 1

        continue


    # ========================================================
    # PREVISÃO OVER
    # ========================================================

    try:

        prob_over = (
            probabilidade_classe_positiva(
                modelo_over25,
                df_over,
            )
        )


    except Exception as erro:

        print(
            f"   ❌ Erro no modelo OVER: "
            f"{erro}"
        )

        ignorados += 1

        continue


    # ========================================================
    # JOGO ANALISADO
    # ========================================================

    analisados += 1


    # ========================================================
    # MOSTRAR PROBABILIDADES
    # ========================================================

    print(
        f"   🤝 BTTS: "
        f"{prob_btts:.1f}%"
    )

    print(
        f"   ⚽ Over 2.5: "
        f"{prob_over:.1f}%"
    )


    # ========================================================
    # SINAL BTTS
    # ========================================================

    if prob_btts >= BTTS_ALTO:

        alertas_btts.append(
            (
                league,
                home_team_raw,
                away_team_raw,
                prob_btts,
                "SIM",
            )
        )


    elif prob_btts <= BTTS_BAIXO:

        alertas_btts.append(
            (
                league,
                home_team_raw,
                away_team_raw,
                prob_btts,
                "NÃO",
            )
        )


    # ========================================================
    # SINAL OVER / UNDER
    # ========================================================

    if prob_over >= OVER_ALTO:

        alertas_over.append(
            (
                league,
                home_team_raw,
                away_team_raw,
                prob_over,
                "OVER 2.5",
            )
        )


    elif prob_over <= OVER_BAIXO:

        alertas_over.append(
            (
                league,
                home_team_raw,
                away_team_raw,
                prob_over,
                "UNDER 2.5",
            )
        )


# ============================================================
# RESUMO
# ============================================================

print()
print(
    "======================================================="
)

print(
    "                    RESUMO"
)

print(
    "======================================================="
)


print(
    f"Jogos encontrados       : "
    f"{len(jogos_hoje)}"
)

print(
    f"Jogos analisados        : "
    f"{analisados}"
)

print(
    f"Jogos ignorados         : "
    f"{ignorados}"
)

print(
    f"Ligas não suportadas    : "
    f"{ligas_nao_suportadas}"
)

print(
    f"Histórico insuficiente  : "
    f"{historico_insuficiente}"
)


# ============================================================
# BTTS
# ============================================================

print()
print(
    "🔥 MERCADO: AMBAS MARCAM (BTTS) 🔥"
)

print(
    "-------------------------------------------------------"
)


if not alertas_btts:

    print(
        "Nenhum sinal BTTS "
        "com os limites atuais."
    )


else:

    for (
        league,
        home_team,
        away_team,
        probability,
        signal,
    ) in alertas_btts:

        emoji = (
            "🟢"
            if signal == "SIM"
            else "🔴"
        )

        print(
            f"{emoji} [{league}] "
            f"{home_team} vs "
            f"{away_team} "
            f"-> BTTS {signal} "
            f"({probability:.1f}%)"
        )


# ============================================================
# OVER / UNDER
# ============================================================

print()
print(
    "⚽ MERCADO: GOLOS "
    "(OVER / UNDER 2.5) ⚽"
)

print(
    "-------------------------------------------------------"
)


if not alertas_over:

    print(
        "Nenhum sinal Over/Under 2.5 "
        "com os limites atuais."
    )


else:

    for (
        league,
        home_team,
        away_team,
        probability,
        signal,
    ) in alertas_over:

        emoji = (
            "🟢"
            if signal == "OVER 2.5"
            else "🔴"
        )

        print(
            f"{emoji} [{league}] "
            f"{home_team} vs "
            f"{away_team} "
            f"-> {signal} "
            f"({probability:.1f}%)"
        )


# ============================================================
# FIM
# ============================================================

print()
print(
    "======================================================="
)

print(
    "                 ANÁLISE TERMINADA"
)

print(
    "======================================================="
)