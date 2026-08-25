from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATABASE_PATH = ROOT_DIR / "database" / "football.db"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "dados_preparados.csv"

ML_DIR = ROOT_DIR / "ml"
FEATURES_BTTS_PATH = ML_DIR / "features_btts.pkl"
FEATURES_OVER25_PATH = ML_DIR / "features_over25.pkl"
METADATA_PATH = PROCESSED_DIR / "feature_metadata.json"


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROLLING_WINDOW = 5
MIN_HISTORY = 3
HOME_ADVANTAGE_ELO = 65
ELO_K_FACTOR = 20

MATCH_KEY = [
    "date",
    "league",
    "home_team",
    "away_team",
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def parse_dates(series: pd.Series) -> pd.Series:
    """
    Converte datas armazenadas em diferentes formatos:

    YYYY-MM-DD
    DD/MM/YYYY
    DD/MM/YY
    """
    text = series.astype("string").str.strip()

    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    supported_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
    ]

    for date_format in supported_formats:
        missing_mask = (
            parsed.isna()
            & text.notna()
            & (text != "")
        )

        if not missing_mask.any():
            break

        parsed.loc[missing_mask] = pd.to_datetime(
            text.loc[missing_mask],
            format=date_format,
            errors="coerce",
        )

    return parsed


def calculate_season(date: pd.Timestamp) -> str | None:
    """
    Calcula a época para campeonatos europeus.

    Julho de 2024 até junho de 2025 -> 2024/2025
    """
    if pd.isna(date):
        return None

    if date.month >= 7:
        start_year = date.year
    else:
        start_year = date.year - 1

    return f"{start_year}/{start_year + 1}"


def shifted_rolling_mean(
    series: pd.Series,
    window: int = ROLLING_WINDOW,
    min_periods: int = MIN_HISTORY,
) -> pd.Series:
    """
    Média móvel que exclui o jogo atual.

    O shift(1) é essencial para impedir data leakage.
    """
    return (
        series
        .shift(1)
        .rolling(
            window=window,
            min_periods=min_periods,
        )
        .mean()
    )


def add_elo_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o Elo pré-jogo.

    As ratings guardadas em cada linha são as ratings existentes
    antes desse jogo acontecer.
    """
    ratings: dict[tuple[str, str], float] = {}

    home_elos: list[float] = []
    away_elos: list[float] = []

    ordered = matches.sort_values(
        ["date", "id"],
    ).copy()

    for row in ordered.itertuples(index=False):
        league = str(row.league)
        home_team = str(row.home_team)
        away_team = str(row.away_team)

        home_key = (league, home_team)
        away_key = (league, away_team)

        home_rating = ratings.get(home_key, 1500.0)
        away_rating = ratings.get(away_key, 1500.0)

        home_elos.append(home_rating)
        away_elos.append(away_rating)

        expected_home = 1 / (
            1
            + 10
            ** (
                (
                    away_rating
                    - (home_rating + HOME_ADVANTAGE_ELO)
                )
                / 400
            )
        )

        expected_away = 1 - expected_home

        if row.home_goals > row.away_goals:
            actual_home = 1.0
            actual_away = 0.0

        elif row.home_goals < row.away_goals:
            actual_home = 0.0
            actual_away = 1.0

        else:
            actual_home = 0.5
            actual_away = 0.5

        new_home_rating = (
            home_rating
            + ELO_K_FACTOR
            * (actual_home - expected_home)
        )

        new_away_rating = (
            away_rating
            + ELO_K_FACTOR
            * (actual_away - expected_away)
        )

        ratings[home_key] = new_home_rating
        ratings[away_key] = new_away_rating

    ordered["home_elo"] = home_elos
    ordered["away_elo"] = away_elos
    ordered["elo_diff"] = (
        ordered["home_elo"]
        - ordered["away_elo"]
    )

    return ordered


# ============================================================
# CARREGAR BASE DE DADOS
# ============================================================

if not DATABASE_PATH.exists():
    raise FileNotFoundError(
        f"Base de dados não encontrada: {DATABASE_PATH}"
    )

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ML_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with sqlite3.connect(DATABASE_PATH) as connection:
    matches = pd.read_sql_query(
        """
        SELECT *
        FROM matches
        """,
        connection,
    )

print("\n================================")
print(" FEATURE ENGINEERING")
print("================================\n")

print(f"Jogos carregados: {len(matches)}")


# ============================================================
# VALIDAR COLUNAS
# ============================================================

required_columns = [
    "id",
    "date",
    "league",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]

missing_columns = [
    column
    for column in required_columns
    if column not in matches.columns
]

if missing_columns:
    raise ValueError(
        "Faltam colunas na tabela matches: "
        + ", ".join(missing_columns)
    )


# ============================================================
# LIMPEZA INICIAL
# ============================================================

matches["date"] = parse_dates(
    matches["date"]
)

numeric_columns = [
    "home_goals",
    "away_goals",
]

for column in numeric_columns:
    matches[column] = pd.to_numeric(
        matches[column],
        errors="coerce",
    )
print("\nDiagnóstico antes da limpeza:")

print("Datas inválidas       :", int(matches["date"].isna().sum()))
print("Golos casa vazios     :", int(matches["home_goals"].isna().sum()))
print("Golos visitante vazios:", int(matches["away_goals"].isna().sum()))
print("Ligas vazias          :", int(matches["league"].isna().sum()))
print("Equipas casa vazias   :", int(matches["home_team"].isna().sum()))
print("Equipas fora vazias   :", int(matches["away_team"].isna().sum()))
matches["home_team"] = (
    matches["home_team"]
    .astype("string")
    .str.strip()
)

matches["away_team"] = (
    matches["away_team"]
    .astype("string")
    .str.strip()
)

matches["league"] = (
    matches["league"]
    .astype("string")
    .str.strip()
)

initial_count = len(matches)

matches = matches.dropna(
    subset=[
        "id",
        "date",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
).copy()

matches = matches[
    (matches["home_team"] != "")
    & (matches["away_team"] != "")
    & (matches["league"] != "")
].copy()

invalid_removed = initial_count - len(matches)

print(f"Jogos inválidos removidos: {invalid_removed}")


# ============================================================
# GARANTIR QUE NÃO EXISTEM DUPLICADOS
# ============================================================

duplicate_ids = int(
    matches.duplicated(
        subset=["id"],
        keep=False,
    ).sum()
)

duplicate_matches = int(
    matches.duplicated(
        subset=MATCH_KEY,
        keep="first",
    ).sum()
)

print(f"IDs duplicados: {duplicate_ids}")
print(f"Jogos duplicados: {duplicate_matches}")

if duplicate_ids > 0:
    raise ValueError(
        "Existem IDs duplicados na tabela matches."
    )

if duplicate_matches > 0:
    raise ValueError(
        "Ainda existem jogos duplicados na base de dados. "
        "Executa novamente database/remover_duplicados.py."
    )


# ============================================================
# TARGETS
# ============================================================

calculated_over25 = (
    (
        matches["home_goals"]
        + matches["away_goals"]
    )
    > 2
).astype(int)

calculated_btts = (
    (matches["home_goals"] > 0)
    & (matches["away_goals"] > 0)
).astype(int)

if "over25" not in matches.columns:
    matches["over25"] = calculated_over25

else:
    matches["over25"] = pd.to_numeric(
        matches["over25"],
        errors="coerce",
    )

    matches["over25"] = (
        matches["over25"]
        .fillna(calculated_over25)
        .astype(int)
    )

if "btts" not in matches.columns:
    matches["btts"] = calculated_btts

else:
    matches["btts"] = pd.to_numeric(
        matches["btts"],
        errors="coerce",
    )

    matches["btts"] = (
        matches["btts"]
        .fillna(calculated_btts)
        .astype(int)
    )


# ============================================================
# ÉPOCA
# ============================================================

calculated_season = matches["date"].apply(
    calculate_season
)

if "season" not in matches.columns:
    matches["season"] = calculated_season

else:
    season_text = (
        matches["season"]
        .astype("string")
        .str.strip()
    )

    invalid_season = (
        matches["season"].isna()
        | season_text.isin(
            ["", "nan", "None", "<NA>"]
        )
    )

    matches.loc[
        invalid_season,
        "season",
    ] = calculated_season.loc[
        invalid_season
    ]


# ============================================================
# ORDENAR CRONOLOGICAMENTE
# ============================================================

matches = matches.sort_values(
    ["date", "id"],
).reset_index(drop=True)

database_row_count = len(matches)


# ============================================================
# CRIAR HISTÓRICO LONGO POR EQUIPA
# ============================================================

base_columns = [
    "id",
    "date",
    "season",
    "league",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "over25",
    "btts",
]

home_history = matches[
    base_columns
].copy()

home_history["team"] = (
    home_history["home_team"]
)

home_history["opponent"] = (
    home_history["away_team"]
)

home_history["venue"] = "home"

home_history["goals_for"] = (
    home_history["home_goals"]
)

home_history["goals_against"] = (
    home_history["away_goals"]
)


away_history = matches[
    base_columns
].copy()

away_history["team"] = (
    away_history["away_team"]
)

away_history["opponent"] = (
    away_history["home_team"]
)

away_history["venue"] = "away"

away_history["goals_for"] = (
    away_history["away_goals"]
)

away_history["goals_against"] = (
    away_history["home_goals"]
)


team_history = pd.concat(
    [
        home_history,
        away_history,
    ],
    ignore_index=True,
)

team_history["goal_diff"] = (
    team_history["goals_for"]
    - team_history["goals_against"]
)

team_history["win"] = (
    team_history["goals_for"]
    > team_history["goals_against"]
).astype(int)

team_history["draw"] = (
    team_history["goals_for"]
    == team_history["goals_against"]
).astype(int)

team_history["loss"] = (
    team_history["goals_for"]
    < team_history["goals_against"]
).astype(int)

team_history["points"] = np.select(
    [
        team_history["win"] == 1,
        team_history["draw"] == 1,
    ],
    [
        3,
        1,
    ],
    default=0,
)

team_history["scored"] = (
    team_history["goals_for"] > 0
).astype(int)

team_history["clean_sheet"] = (
    team_history["goals_against"] == 0
).astype(int)

team_history = team_history.sort_values(
    [
        "league",
        "team",
        "date",
        "id",
    ],
).reset_index(drop=True)

group_columns = [
    "league",
    "team",
]

team_history["matches_before"] = (
    team_history
    .groupby(
        group_columns,
        sort=False,
    )
    .cumcount()
)

team_history["rest_days"] = (
    team_history
    .groupby(
        group_columns,
        sort=False,
    )["date"]
    .diff()
    .dt.days
)

team_history["rest_days"] = (
    team_history["rest_days"]
    .clip(
        lower=0,
        upper=60,
    )
)


# ============================================================
# FORMA GERAL — ÚLTIMOS 5 JOGOS
# ============================================================

rolling_source_columns = {
    "goals_for": "avg_scored_5",
    "goals_against": "avg_conceded_5",
    "goal_diff": "avg_goal_diff_5",
    "points": "points_avg_5",
    "win": "win_rate_5",
    "btts": "btts_rate_5",
    "over25": "over25_rate_5",
    "scored": "scored_rate_5",
    "clean_sheet": "clean_sheet_rate_5",
}

for source_column, new_column in (
    rolling_source_columns.items()
):
    team_history[new_column] = (
        team_history
        .groupby(
            group_columns,
            sort=False,
        )[source_column]
        .transform(shifted_rolling_mean)
    )


# ============================================================
# FORMA ESPECÍFICA CASA/FORA
# ============================================================

team_history["venue_avg_scored_5"] = np.nan
team_history["venue_avg_conceded_5"] = np.nan
team_history["venue_points_avg_5"] = np.nan

for venue in ["home", "away"]:
    venue_mask = (
        team_history["venue"] == venue
    )

    venue_data = team_history.loc[
        venue_mask
    ]

    venue_groups = (
        venue_data
        .groupby(
            group_columns,
            sort=False,
        )
    )

    team_history.loc[
        venue_mask,
        "venue_avg_scored_5",
    ] = venue_groups[
        "goals_for"
    ].transform(
        lambda series: shifted_rolling_mean(
            series,
            min_periods=2,
        )
    )

    team_history.loc[
        venue_mask,
        "venue_avg_conceded_5",
    ] = venue_groups[
        "goals_against"
    ].transform(
        lambda series: shifted_rolling_mean(
            series,
            min_periods=2,
        )
    )

    team_history.loc[
        venue_mask,
        "venue_points_avg_5",
    ] = venue_groups[
        "points"
    ].transform(
        lambda series: shifted_rolling_mean(
            series,
            min_periods=2,
        )
    )


# ============================================================
# SEPARAR FEATURES DE CASA E FORA
# ============================================================

general_history_columns = [
    "avg_scored_5",
    "avg_conceded_5",
    "avg_goal_diff_5",
    "points_avg_5",
    "win_rate_5",
    "btts_rate_5",
    "over25_rate_5",
    "scored_rate_5",
    "clean_sheet_rate_5",
    "rest_days",
    "matches_before",
]

home_feature_columns = [
    "id",
    *general_history_columns,
    "venue_avg_scored_5",
    "venue_avg_conceded_5",
    "venue_points_avg_5",
]

away_feature_columns = [
    "id",
    *general_history_columns,
    "venue_avg_scored_5",
    "venue_avg_conceded_5",
    "venue_points_avg_5",
]

home_features = team_history.loc[
    team_history["venue"] == "home",
    home_feature_columns,
].copy()

away_features = team_history.loc[
    team_history["venue"] == "away",
    away_feature_columns,
].copy()


home_rename = {
    column: f"home_{column}"
    for column in general_history_columns
}

home_rename.update(
    {
        "venue_avg_scored_5":
            "home_home_avg_scored_5",
        "venue_avg_conceded_5":
            "home_home_avg_conceded_5",
        "venue_points_avg_5":
            "home_home_points_avg_5",
    }
)

away_rename = {
    column: f"away_{column}"
    for column in general_history_columns
}

away_rename.update(
    {
        "venue_avg_scored_5":
            "away_away_avg_scored_5",
        "venue_avg_conceded_5":
            "away_away_avg_conceded_5",
        "venue_points_avg_5":
            "away_away_points_avg_5",
    }
)

home_features = home_features.rename(
    columns=home_rename,
)

away_features = away_features.rename(
    columns=away_rename,
)


# ============================================================
# MERGE SEGURO PELO ID DO JOGO
# ============================================================

prepared = matches.merge(
    home_features,
    on="id",
    how="left",
    validate="one_to_one",
)

prepared = prepared.merge(
    away_features,
    on="id",
    how="left",
    validate="one_to_one",
)

if len(prepared) != database_row_count:
    raise RuntimeError(
        "O número de linhas aumentou durante os merges. "
        f"Antes: {database_row_count}; "
        f"depois: {len(prepared)}."
    )


# ============================================================
# ELO PRÉ-JOGO
# ============================================================

elo_features = add_elo_features(
    matches[
        [
            "id",
            "date",
            "league",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ].copy()
)

elo_features = elo_features[
    [
        "id",
        "home_elo",
        "away_elo",
        "elo_diff",
    ]
]

prepared = prepared.merge(
    elo_features,
    on="id",
    how="left",
    validate="one_to_one",
)

if len(prepared) != database_row_count:
    raise RuntimeError(
        "O número de linhas mudou após adicionar o Elo."
    )


# ============================================================
# PREENCHER FEATURES CASA/FORA
# ============================================================

prepared["home_home_avg_scored_5"] = (
    prepared["home_home_avg_scored_5"]
    .fillna(
        prepared["home_avg_scored_5"]
    )
)

prepared["home_home_avg_conceded_5"] = (
    prepared["home_home_avg_conceded_5"]
    .fillna(
        prepared["home_avg_conceded_5"]
    )
)

prepared["home_home_points_avg_5"] = (
    prepared["home_home_points_avg_5"]
    .fillna(
        prepared["home_points_avg_5"]
    )
)

prepared["away_away_avg_scored_5"] = (
    prepared["away_away_avg_scored_5"]
    .fillna(
        prepared["away_avg_scored_5"]
    )
)

prepared["away_away_avg_conceded_5"] = (
    prepared["away_away_avg_conceded_5"]
    .fillna(
        prepared["away_avg_conceded_5"]
    )
)

prepared["away_away_points_avg_5"] = (
    prepared["away_away_points_avg_5"]
    .fillna(
        prepared["away_points_avg_5"]
    )
)

prepared["home_rest_days"] = (
    prepared["home_rest_days"]
    .fillna(7)
)

prepared["away_rest_days"] = (
    prepared["away_rest_days"]
    .fillna(7)
)


# ============================================================
# FEATURES COMBINADAS
# ============================================================

prepared["expected_home_goals"] = (
    (
        prepared["home_avg_scored_5"]
        + prepared["away_avg_conceded_5"]
    )
    / 2
)

prepared["expected_away_goals"] = (
    (
        prepared["away_avg_scored_5"]
        + prepared["home_avg_conceded_5"]
    )
    / 2
)

prepared["expected_total_goals"] = (
    prepared["expected_home_goals"]
    + prepared["expected_away_goals"]
)

prepared["expected_goal_difference"] = (
    prepared["expected_home_goals"]
    - prepared["expected_away_goals"]
)

prepared["total_expected_scored"] = (
    prepared["home_avg_scored_5"]
    + prepared["away_avg_scored_5"]
)

prepared["total_expected_conceded"] = (
    prepared["home_avg_conceded_5"]
    + prepared["away_avg_conceded_5"]
)

prepared["home_power_rating"] = (
    prepared["home_avg_goal_diff_5"]
    + prepared["home_points_avg_5"]
)

prepared["away_power_rating"] = (
    prepared["away_avg_goal_diff_5"]
    + prepared["away_points_avg_5"]
)

prepared["power_rating_diff"] = (
    prepared["home_power_rating"]
    - prepared["away_power_rating"]
)

prepared["form_points_diff"] = (
    prepared["home_points_avg_5"]
    - prepared["away_points_avg_5"]
)

prepared["btts_signal"] = (
    (
        prepared["home_btts_rate_5"]
        + prepared["away_btts_rate_5"]
    )
    / 2
)

prepared["over25_signal"] = (
    (
        prepared["home_over25_rate_5"]
        + prepared["away_over25_rate_5"]
    )
    / 2
)

prepared["scoring_signal"] = (
    (
        prepared["home_scored_rate_5"]
        + prepared["away_scored_rate_5"]
    )
    / 2
)

prepared["clean_sheet_signal"] = (
    (
        prepared["home_clean_sheet_rate_5"]
        + prepared["away_clean_sheet_rate_5"]
    )
    / 2
)


# ============================================================
# LISTA FINAL DE FEATURES
# ============================================================

features = [
    "home_avg_scored_5",
    "home_avg_conceded_5",
    "home_avg_goal_diff_5",
    "home_points_avg_5",
    "home_win_rate_5",
    "home_btts_rate_5",
    "home_over25_rate_5",
    "home_scored_rate_5",
    "home_clean_sheet_rate_5",

    "away_avg_scored_5",
    "away_avg_conceded_5",
    "away_avg_goal_diff_5",
    "away_points_avg_5",
    "away_win_rate_5",
    "away_btts_rate_5",
    "away_over25_rate_5",
    "away_scored_rate_5",
    "away_clean_sheet_rate_5",

    "home_home_avg_scored_5",
    "home_home_avg_conceded_5",
    "home_home_points_avg_5",

    "away_away_avg_scored_5",
    "away_away_avg_conceded_5",
    "away_away_points_avg_5",

    "home_rest_days",
    "away_rest_days",

    "home_elo",
    "away_elo",
    "elo_diff",

    "expected_home_goals",
    "expected_away_goals",
    "expected_total_goals",
    "expected_goal_difference",

    "total_expected_scored",
    "total_expected_conceded",

    "home_power_rating",
    "away_power_rating",
    "power_rating_diff",

    "form_points_diff",
    "btts_signal",
    "over25_signal",
    "scoring_signal",
    "clean_sheet_signal",
]


# ============================================================
# REMOVER APENAS JOGOS SEM HISTÓRICO SUFICIENTE
# ============================================================

before_history_filter = len(prepared)

prepared = prepared.dropna(
    subset=features,
).copy()

removed_without_history = (
    before_history_filter
    - len(prepared)
)

prepared = prepared.sort_values(
    ["date", "id"],
).reset_index(drop=True)


# ============================================================
# VALIDAÇÃO FINAL
# ============================================================

final_duplicates = int(
    prepared.duplicated(
        subset=MATCH_KEY,
        keep="first",
    ).sum()
)

if final_duplicates > 0:
    raise RuntimeError(
        f"O dataset final contém "
        f"{final_duplicates} jogos duplicados."
    )

if len(prepared) > database_row_count:
    raise RuntimeError(
        "O dataset processado tem mais linhas "
        "do que a base de dados."
    )

non_numeric_features = [
    feature
    for feature in features
    if not pd.api.types.is_numeric_dtype(
        prepared[feature]
    )
]

if non_numeric_features:
    raise TypeError(
        "Estas features não são numéricas: "
        + ", ".join(non_numeric_features)
    )


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

prepared.to_csv(
    OUTPUT_PATH,
    index=False,
)

joblib.dump(
    features,
    FEATURES_BTTS_PATH,
)

joblib.dump(
    features,
    FEATURES_OVER25_PATH,
)

metadata = {
    "generated_at": datetime.now().isoformat(),
    "database_rows": int(database_row_count),
    "processed_rows": int(len(prepared)),
    "removed_without_history": int(
        removed_without_history
    ),
    "feature_count": len(features),
    "features": features,
    "minimum_history": MIN_HISTORY,
    "rolling_window": ROLLING_WINDOW,
    "date_min": (
        prepared["date"].min().date().isoformat()
        if not prepared.empty
        else None
    ),
    "date_max": (
        prepared["date"].max().date().isoformat()
        if not prepared.empty
        else None
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
# RESUMO
# ============================================================

print("\n================================")
print(" RESULTADO")
print("================================\n")

print(f"Jogos na base             : {database_row_count}")
print(
    "Sem histórico suficiente : "
    f"{removed_without_history}"
)
print(f"Jogos finais              : {len(prepared)}")
print(f"Duplicados finais         : {final_duplicates}")
print(f"Features criadas          : {len(features)}")

if not prepared.empty:
    print(
        "Período                    : "
        f"{prepared['date'].min().date()} "
        f"a {prepared['date'].max().date()}"
    )

    print(
        "Épocas                     : "
        f"{prepared['season'].nunique()}"
    )

print("\nFicheiros guardados:")

print(
    f"✅ {OUTPUT_PATH.relative_to(ROOT_DIR)}"
)

print(
    f"✅ {FEATURES_BTTS_PATH.relative_to(ROOT_DIR)}"
)

print(
    f"✅ {FEATURES_OVER25_PATH.relative_to(ROOT_DIR)}"
)

print(
    f"✅ {METADATA_PATH.relative_to(ROOT_DIR)}"
)

print(
    "\n✅ Feature engineering concluído "
    "sem duplicar jogos."
)