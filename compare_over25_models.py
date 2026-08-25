import sys
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


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
    load_historical_matches,
    create_team_history,
    calculate_current_elo,
    build_sports_features,
    normalize_league,
)


DATABASE_PATH = (
    ROOT_DIR
    / "database"
    / "football.db"
)

SPORT_MODEL_PATH = (
    ROOT_DIR
    / "ml"
    / "modelo_over25.pkl"
)

SPORT_FEATURES_PATH = (
    ROOT_DIR
    / "ml"
    / "features_over25.pkl"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "analysis"
    / "comparacao_modelos_over25.csv"
)


# ============================================================
# FUNÇÕES
# ============================================================

def positive_probability(
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
            "Modelo não contém a classe positiva 1."
        )

    index = classes.index(
        1
    )

    return float(
        probabilities[
            0,
            index,
        ]
    )


def encontrar_tabela_paper_trade(
    connection,
):
    """
    Procura automaticamente a tabela do Paper Trade
    através das colunas que esperamos encontrar.
    """

    required_columns = {
        "fixture_date",
        "league",
        "home_team",
        "away_team",
        "odd_over25_open",
        "fair_probability_open",
        "clv_probability",
        "status",
        "home_goals",
        "away_goals",
    }


    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """,
        connection,
    )


    for table_name in tables["name"]:

        safe_name = str(
            table_name
        ).replace(
            '"',
            '""',
        )

        info = pd.read_sql_query(
            f'PRAGMA table_info("{safe_name}")',
            connection,
        )

        columns = set(
            info["name"].astype(str)
        )

        if required_columns.issubset(
            columns
        ):

            return str(
                table_name
            )


    raise ValueError(
        "Não encontrei a tabela do Paper Trade "
        "na base de dados."
    )


def calcular_metricas(
    nome,
    y_true,
    probabilities,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )


    probabilities = np.clip(
        probabilities,
        0.000001,
        0.999999,
    )


    predicted = (
        probabilities >= 0.50
    ).astype(int)


    actual_rate = float(
        np.mean(
            y_true
        )
    )

    predicted_rate = float(
        np.mean(
            probabilities
        )
    )


    brier = brier_score_loss(
        y_true,
        probabilities,
    )


    loss = log_loss(
        y_true,
        probabilities,
        labels=[
            0,
            1,
        ],
    )


    accuracy = accuracy_score(
        y_true,
        predicted,
    )


    if len(
        np.unique(
            y_true
        )
    ) >= 2:

        auc = roc_auc_score(
            y_true,
            probabilities,
        )

    else:

        auc = np.nan


    calibration_error = (
        predicted_rate
        - actual_rate
    )


    return {
        "modelo": nome,
        "jogos": len(
            y_true
        ),
        "taxa_real_over": actual_rate,
        "prob_media": predicted_rate,
        "erro_calibracao": calibration_error,
        "brier": brier,
        "log_loss": loss,
        "auc": auc,
        "accuracy_50": accuracy,
    }


def analisar_thresholds(
    nome,
    probabilities,
    y_true,
    odds,
):

    resultados = []

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    odds = np.asarray(
        odds,
        dtype=float,
    )


    thresholds = [
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
    ]


    for threshold in thresholds:

        mask = (
            probabilities
            >= threshold
        )


        numero_apostas = int(
            mask.sum()
        )


        if numero_apostas == 0:

            resultados.append(
                {
                    "modelo":
                        nome,

                    "threshold":
                        threshold,

                    "apostas":
                        0,

                    "wins":
                        0,

                    "hit_rate":
                        np.nan,

                    "odd_media":
                        np.nan,

                    "lucro_units":
                        0.0,

                    "roi":
                        np.nan,
                }
            )

            continue


        selected_y = (
            y_true[
                mask
            ]
        )

        selected_odds = (
            odds[
                mask
            ]
        )


        wins = int(
            selected_y.sum()
        )


        hit_rate = float(
            selected_y.mean()
        )


        profits = np.where(
            selected_y == 1,
            selected_odds - 1.0,
            -1.0,
        )


        total_profit = float(
            profits.sum()
        )


        roi = (
            total_profit
            / numero_apostas
        )


        resultados.append(
            {
                "modelo":
                    nome,

                "threshold":
                    threshold,

                "apostas":
                    numero_apostas,

                "wins":
                    wins,

                "hit_rate":
                    hit_rate,

                "odd_media":
                    float(
                        selected_odds.mean()
                    ),

                "lucro_units":
                    total_profit,

                "roi":
                    roi,
            }
        )


    return resultados


# ============================================================
# CABEÇALHO
# ============================================================

print()
print(
    "======================================================="
)

print(
    "        COMPARAÇÃO MODELOS OVER 2.5"
)

print(
    "======================================================="
)


# ============================================================
# CARREGAR MODELO DESPORTIVO
# ============================================================

print()
print(
    "🧠 A carregar modelo desportivo..."
)


sport_model = joblib.load(
    SPORT_MODEL_PATH
)


sport_features = joblib.load(
    SPORT_FEATURES_PATH
)


print(
    f"Features do modelo desportivo: "
    f"{len(sport_features)}"
)


# ============================================================
# BASE DE DADOS
# ============================================================

connection = sqlite3.connect(
    DATABASE_PATH
)


# ============================================================
# DESCOBRIR TABELA PAPER TRADE
# ============================================================

paper_table = (
    encontrar_tabela_paper_trade(
        connection
    )
)


print()
print(
    f"📋 Tabela Paper Trade encontrada: "
    f"{paper_table}"
)


safe_table = paper_table.replace(
    '"',
    '""',
)


paper = pd.read_sql_query(
    f'''
    SELECT *
    FROM "{safe_table}"
    WHERE UPPER(status) = 'RESOLVED'
    ORDER BY
        fixture_date,
        fixture_time
    ''',
    connection,
)


if paper.empty:

    connection.close()

    print()
    print(
        "❌ Não existem jogos RESOLVED "
        "no Paper Trade."
    )

    raise SystemExit(1)


# ============================================================
# FILTRAR MERCADO OVER 2.5
# ============================================================

if "market" in paper.columns:

    market_text = (
        paper["market"]
        .astype(str)
        .str.lower()
    )


    over_mask = (
        market_text.str.contains(
            "over",
            na=False,
        )
    )


    if over_mask.any():

        paper = paper[
            over_mask
        ].copy()


# ============================================================
# CARREGAR HISTÓRICO
# ============================================================

print(
    "📚 A carregar histórico..."
)


matches = load_historical_matches(
    connection
)


connection.close()


matches[
    "date"
] = pd.to_datetime(
    matches["date"],
    errors="coerce",
)


print(
    f"Jogos históricos: "
    f"{len(matches)}"
)


# ============================================================
# DATAS DOS JOGOS PAPER TRADE
# ============================================================

paper[
    "fixture_date"
] = pd.to_datetime(
    paper[
        "fixture_date"
    ],
    errors="coerce",
)


def criar_fixture_datetime(
    row,
):

    date_value = row[
        "fixture_date"
    ]


    if pd.isna(
        date_value
    ):

        return pd.NaT


    if (
        "fixture_time"
        not in row.index
        or pd.isna(
            row[
                "fixture_time"
            ]
        )
    ):

        return pd.Timestamp(
            date_value
        )


    time_value = str(
        row[
            "fixture_time"
        ]
    ).strip()


    try:

        return pd.Timestamp(
            f"{date_value.date()} "
            f"{time_value}"
        )

    except Exception:

        return pd.Timestamp(
            date_value
        )


paper[
    "fixture_datetime"
] = paper.apply(
    criar_fixture_datetime,
    axis=1,
)


paper = paper.dropna(
    subset=[
        "fixture_datetime",
        "home_goals",
        "away_goals",
        "clv_probability",
        "fair_probability_open",
        "odd_over25_open",
    ]
).copy()


print()
print(
    f"Jogos RESOLVED disponíveis: "
    f"{len(paper)}"
)


# ============================================================
# RESULTADO REAL
# ============================================================

paper[
    "home_goals"
] = pd.to_numeric(
    paper[
        "home_goals"
    ],
    errors="coerce",
)


paper[
    "away_goals"
] = pd.to_numeric(
    paper[
        "away_goals"
    ],
    errors="coerce",
)


paper[
    "total_goals"
] = (
    paper[
        "home_goals"
    ]
    + paper[
        "away_goals"
    ]
)


paper[
    "actual_over25"
] = (
    paper[
        "total_goals"
    ]
    >= 3
).astype(int)


# ============================================================
# VERIFICAR SE TODO O HISTÓRICO É ANTERIOR
# ============================================================

historical_max_date = (
    matches[
        "date"
    ].max()
)


first_paper_date = (
    paper[
        "fixture_datetime"
    ].min()
)


print()
print(
    f"Último jogo histórico: "
    f"{historical_max_date}"
)

print(
    f"Primeiro Paper Trade: "
    f"{first_paper_date}"
)


if (
    pd.notna(
        historical_max_date
    )
    and pd.notna(
        first_paper_date
    )
    and historical_max_date
    < first_paper_date
):

    print(
        "✅ Todo o histórico é anterior "
        "aos jogos avaliados."
    )

else:

    print(
        "⚠ Existem datas sobrepostas. "
        "O script irá usar apenas histórico "
        "anterior a cada jogo."
    )


# ============================================================
# PREPARAÇÃO GLOBAL
# ============================================================

global_history_safe = (
    pd.notna(
        historical_max_date
    )
    and pd.notna(
        first_paper_date
    )
    and historical_max_date
    < first_paper_date
)


if global_history_safe:

    team_history_global = (
        create_team_history(
            matches
        )
    )


    elo_global = (
        calculate_current_elo(
            matches
        )
    )


else:

    team_history_global = None

    elo_global = None


# ============================================================
# CALCULAR PROBABILIDADE DO MODELO DESPORTIVO
# ============================================================

resultados = []

ignorados = []


print()
print(
    "⚽ A reconstruir previsões "
    "do modelo desportivo..."
)


for _, row in paper.iterrows():

    fixture_datetime = pd.Timestamp(
        row[
            "fixture_datetime"
        ]
    )


    league_raw = str(
        row[
            "league"
        ]
    ).strip()


    league = normalize_league(
        league_raw
    )


    if league is None:

        league = league_raw


    home_team = str(
        row[
            "home_team"
        ]
    ).strip()


    away_team = str(
        row[
            "away_team"
        ]
    ).strip()


    # --------------------------------------------------------
    # HISTÓRICO
    # --------------------------------------------------------

    if global_history_safe:

        team_history = (
            team_history_global
        )

        elo_ratings = (
            elo_global
        )


    else:

        matches_before = matches[
            matches[
                "date"
            ]
            < fixture_datetime
        ].copy()


        team_history = (
            create_team_history(
                matches_before
            )
        )


        elo_ratings = (
            calculate_current_elo(
                matches_before
            )
        )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    try:

        sports_values = (
            build_sports_features(
                history=team_history,
                elo_ratings=elo_ratings,
                league=league,
                home_team=home_team,
                away_team=away_team,
                fixture_date=fixture_datetime,
            )
        )


        frame = pd.DataFrame(
            [
                sports_values
            ]
        )


        missing = [
            feature
            for feature in sport_features
            if feature
            not in frame.columns
        ]


        if missing:

            raise ValueError(
                "Features em falta: "
                + ", ".join(
                    missing
                )
            )


        frame = frame[
            sport_features
        ].copy()


        frame = frame.apply(
            pd.to_numeric,
            errors="coerce",
        )


        if frame.isna().any().any():

            raise ValueError(
                "Existem valores NaN "
                "nas features."
            )


        sport_probability = (
            positive_probability(
                sport_model,
                frame,
            )
        )


    except Exception as erro:

        ignorados.append(
            {
                "date":
                    fixture_datetime,

                "league":
                    league,

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "erro":
                    str(
                        erro
                    ),
            }
        )

        continue


    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    resultados.append(
        {
            "fixture_datetime":
                fixture_datetime,

            "league":
                league,

            "home_team":
                home_team,

            "away_team":
                away_team,

            "home_goals":
                int(
                    row[
                        "home_goals"
                    ]
                ),

            "away_goals":
                int(
                    row[
                        "away_goals"
                    ]
                ),

            "actual_over25":
                int(
                    row[
                        "actual_over25"
                    ]
                ),

            "odd_over25_open":
                float(
                    row[
                        "odd_over25_open"
                    ]
                ),

            "market_probability":
                float(
                    row[
                        "fair_probability_open"
                    ]
                ),

            "paper_probability":
                float(
                    row[
                        "clv_probability"
                    ]
                ),

            "sport_probability":
                sport_probability,
        }
    )


comparison = pd.DataFrame(
    resultados
)


if comparison.empty:

    print()
    print(
        "❌ Não foi possível comparar "
        "nenhum jogo."
    )

    raise SystemExit(1)


# ============================================================
# GUARDAR CSV
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


comparison.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# MOSTRAR JOGOS
# ============================================================

print()
print(
    "======================================================="
)

print(
    "                 JOGOS COMPARADOS"
)

print(
    "======================================================="
)


display = comparison[
    [
        "fixture_datetime",
        "home_team",
        "away_team",
        "actual_over25",
        "market_probability",
        "paper_probability",
        "sport_probability",
    ]
].copy()


display[
    "market_probability"
] = (
    display[
        "market_probability"
    ]
    * 100
)


display[
    "paper_probability"
] = (
    display[
        "paper_probability"
    ]
    * 100
)


display[
    "sport_probability"
] = (
    display[
        "sport_probability"
    ]
    * 100
)


print(
    display.to_string(
        index=False,
        formatters={
            "market_probability":
                "{:.1f}%".format,

            "paper_probability":
                "{:.1f}%".format,

            "sport_probability":
                "{:.1f}%".format,
        },
    )
)


# ============================================================
# MÉTRICAS
# ============================================================

y_true = comparison[
    "actual_over25"
].to_numpy()


metricas = []


metricas.append(
    calcular_metricas(
        "Mercado",
        y_true,
        comparison[
            "market_probability"
        ],
    )
)


metricas.append(
    calcular_metricas(
        "Paper Trade",
        y_true,
        comparison[
            "paper_probability"
        ],
    )
)


metricas.append(
    calcular_metricas(
        "Desportivo",
        y_true,
        comparison[
            "sport_probability"
        ],
    )
)


metricas_df = pd.DataFrame(
    metricas
)


# ============================================================
# MOSTRAR MÉTRICAS
# ============================================================

print()
print(
    "======================================================="
)

print(
    "              MÉTRICAS PRINCIPAIS"
)

print(
    "======================================================="
)


for _, row in metricas_df.iterrows():

    print()
    print(
        f"--- {row['modelo']} ---"
    )

    print(
        f"Jogos             : "
        f"{int(row['jogos'])}"
    )

    print(
        f"Taxa real Over     : "
        f"{row['taxa_real_over']:.2%}"
    )

    print(
        f"Probabilidade média: "
        f"{row['prob_media']:.2%}"
    )

    print(
        f"Erro calibração    : "
        f"{row['erro_calibracao']:+.2%}"
    )

    print(
        f"Brier Score        : "
        f"{row['brier']:.4f}"
    )

    print(
        f"Log Loss           : "
        f"{row['log_loss']:.4f}"
    )

    print(
        f"ROC AUC            : "
        f"{row['auc']:.4f}"
    )

    print(
        f"Accuracy @ 50%     : "
        f"{row['accuracy_50']:.2%}"
    )


# ============================================================
# THRESHOLDS
# ============================================================

threshold_results = []


threshold_results.extend(
    analisar_thresholds(
        "Paper Trade",
        comparison[
            "paper_probability"
        ],
        y_true,
        comparison[
            "odd_over25_open"
        ],
    )
)


threshold_results.extend(
    analisar_thresholds(
        "Desportivo",
        comparison[
            "sport_probability"
        ],
        y_true,
        comparison[
            "odd_over25_open"
        ],
    )
)


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# MOSTRAR THRESHOLDS
# ============================================================

print()
print(
    "======================================================="
)

print(
    "             TESTE DE THRESHOLDS"
)

print(
    "======================================================="
)


for model_name in [
    "Paper Trade",
    "Desportivo",
]:

    print()
    print(
        f"--- {model_name} ---"
    )


    subset = threshold_df[
        threshold_df[
            "modelo"
        ]
        == model_name
    ]


    for _, row in subset.iterrows():

        threshold = (
            row[
                "threshold"
            ]
        )


        apostas = int(
            row[
                "apostas"
            ]
        )


        if apostas == 0:

            print(
                f"{threshold:.0%}: "
                f"0 apostas"
            )

            continue


        print(
            f"{threshold:.0%}: "
            f"{apostas} apostas | "
            f"{int(row['wins'])} wins | "
            f"Hit {row['hit_rate']:.1%} | "
            f"Odd média {row['odd_media']:.2f} | "
            f"P/L {row['lucro_units']:+.2f}u | "
            f"ROI {row['roi']:+.1%}"
        )


# ============================================================
# IGNORADOS
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
    f"Paper Trade RESOLVED : "
    f"{len(paper)}"
)


print(
    f"Jogos comparados     : "
    f"{len(comparison)}"
)


print(
    f"Jogos ignorados      : "
    f"{len(ignorados)}"
)


if ignorados:

    print()
    print(
        "Jogos ignorados:"
    )

    for item in ignorados:

        print(
            f"- {item['home_team']} "
            f"vs {item['away_team']} "
            f"-> {item['erro']}"
        )


print()
print(
    f"CSV guardado em:"
)

print(
    OUTPUT_PATH
)


print()
print(
    "======================================================="
)

print(
    "                 TESTE TERMINADO"
)

print(
    "======================================================="
)