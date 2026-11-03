from __future__ import annotations

import difflib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    ROOT_DIR
    / "database"
    / "football.db"
)

MODEL_PATH = (
    ROOT_DIR
    / "ml"
    / "modelo_clv_over25.pkl"
)

UPCOMING_DIR = (
    ROOT_DIR
    / "data"
    / "upcoming"
)

UPCOMING_PATH = (
    UPCOMING_DIR
    / "upcoming_matches.csv"
)

REPORTS_DIR = (
    ROOT_DIR
    / "reports"
)

PREDICTIONS_REPORT_PATH = (
    REPORTS_DIR
    / "paper_trade_predictions_latest.csv"
)

SIGNALS_REPORT_PATH = (
    REPORTS_DIR
    / "paper_trade_signals_latest.csv"
)

SKIPPED_REPORT_PATH = (
    REPORTS_DIR
    / "paper_trade_skipped_latest.csv"
)

REGISTRY_REPORT_PATH = (
    REPORTS_DIR
    / "paper_trade_registry.csv"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MARKET = "over25"

ROLLING_WINDOW = 5
MIN_GENERAL_HISTORY = 3
MIN_VENUE_HISTORY = 2

HOME_ADVANTAGE_ELO = 65
ELO_K_FACTOR = 20

DIVISION_LEAGUE_MAP = {
    "E0": "England",
    "SP1": "Spain",
    "I1": "Italy",
    "D1": "Germany",
    "F1": "France",
    "N1": "Netherlands",
    "P1": "Portugal",
}

LEAGUE_ALIASES = {
    # Inglaterra - Premier League
    "england": "England",
    "premier league": "England",
    "english premier league": "England",
    "epl": "England",

    # Inglaterra - Championship
    "england championship": "England Championship",
    "championship": "England Championship",
    "e1": "England Championship",

    # Espanha - La Liga
    "spain": "Spain",
    "la liga": "Spain",
    "laliga": "Spain",
    "la liga - spain": "Spain",

    # Espanha - Segunda Division
    "spain segunda division": "Spain Segunda Division",
    "segunda division": "Spain Segunda Division",
    "segunda división": "Spain Segunda Division",
    "la liga 2 - spain": "Spain Segunda Division",
    "sp2": "Spain Segunda Division",

    # Italia - Serie A
    "italy": "Italy",
    "serie a": "Italy",
    "serie a - italy": "Italy",

    # Alemanha - Bundesliga
    "germany": "Germany",
    "bundesliga": "Germany",
    "bundesliga - germany": "Germany",

    # Alemanha - 2. Bundesliga
    "germany 2 bundesliga": "Germany 2 Bundesliga",
    "2. bundesliga": "Germany 2 Bundesliga",
    "bundesliga 2": "Germany 2 Bundesliga",
    "bundesliga 2 - germany": "Germany 2 Bundesliga",
    "d2": "Germany 2 Bundesliga",

    # Franca - Ligue 1
    "france": "France",
    "ligue 1": "France",
    "ligue 1 - france": "France",

    # França - Ligue 2
    "france ligue 2": "France Ligue 2",
    "f2": "France Ligue 2",

    # Paises Baixos - Eredivisie
    "netherlands": "Netherlands",
    "eredivisie": "Netherlands",
    "dutch eredivisie": "Netherlands",
    "holland": "Netherlands",

    # Portugal - Primeira Liga
    "portugal": "Portugal",
    "primeira liga": "Portugal",
    "liga portugal": "Portugal",
    "primeira liga - portugal": "Portugal",

    # Belgica - First Division A
    "belgium first division a": "Belgium First Division A",
    "belgium first division": "Belgium First Division A",
    "belgium first div": "Belgium First Division A",
    "belgium first div a": "Belgium First Division A",
    "b1": "Belgium First Division A",

    # Turquia - Super Lig
    "turkey super lig": "Turkey Super Lig",
    "turkey super league": "Turkey Super Lig",
    "super lig": "Turkey Super Lig",
    "süper lig": "Turkey Super Lig",
    "t1": "Turkey Super Lig",
}


TEAM_ALIASES = {
    # Bélgica
    ("Belgium First Division A", "cercle brugge ksv"): "Cercle Brugge",
    ("Belgium First Division A", "kv mechelen"): "Mechelen",
    ("Belgium First Division A", "royal antwerp"): "Antwerp",
    ("Belgium First Division A", "standard liege"): "Standard",
    ("Belgium First Division A", "raal la louvière"): "RAAL La Louviere",
    ("Belgium First Division A", "raal la louviere"): "RAAL La Louviere",
    ("Belgium First Division A", "union saint-gilloise"): "St. Gilloise",
    ("Belgium First Division A", "union st gilloise"): "St. Gilloise",
    ("Belgium First Division A", "royale union saint-gilloise"): "St. Gilloise",

    # Inglaterra - Premier League
    ("England", "brighton and hove albion"): "Brighton",
    ("England", "ipswich town"): "Ipswich",
    ("England", "manchester city"): "Man City",
    ("England", "newcastle united"): "Newcastle",
    ("England", "nottingham forest"): "Nott'm Forest",
    ("England", "tottenham hotspur"): "Tottenham",
    ("England", "leeds united"): "Leeds",

    # Inglaterra - Championship
    ("England Championship", "birmingham city"): "Birmingham",
    ("England Championship", "bolton wanderers"): "Bolton",
    ("England Championship", "blackburn rovers"): "Blackburn",
    ("England Championship", "cardiff city"): "Cardiff",
    ("England Championship", "derby county"): "Derby",
    ("England Championship", "norwich city"): "Norwich",
    ("England Championship", "preston north end"): "Preston",
    ("England Championship", "stoke city"): "Stoke",
    ("England Championship", "swansea city"): "Swansea",
    ("England Championship", "west bromwich albion"): "West Brom",
    ("England Championship", "wrexham afc"): "Wrexham",
    ("England Championship", "queens park rangers"): "QPR",
    ("England Championship", "wolverhampton wanderers"): "Wolves",

    # Alemanha - Bundesliga
    ("Germany", "1. fc köln"): "FC Koln",
    ("Germany", "1. fc koln"): "FC Koln",
    ("Germany", "borussia dortmund"): "Dortmund",
    ("Germany", "eintracht frankfurt"): "Ein Frankfurt",
    ("Germany", "fc schalke 04"): "Schalke 04",
    ("Germany", "fsv mainz 05"): "Mainz",
    ("Germany", "sc freiburg"): "Freiburg",
    ("Germany", "vfb stuttgart"): "Stuttgart",
    ("Germany", "borussia monchengladbach"): "M'gladbach",
    ("Germany", "borussia mönchengladbach"): "M'gladbach",
    ("Germany", "hamburger sv"): "Hamburg",
    ("Germany", "sc paderborn"): "Paderborn",
    ("Germany", "tsg hoffenheim"): "Hoffenheim",

    # Alemanha - 2. Bundesliga
    ("Germany 2 Bundesliga", "eintracht braunschweig"): "Braunschweig",
    ("Germany 2 Bundesliga", "hertha berlin"): "Hertha",
    ("Germany 2 Bundesliga", "dynamo dresden"): "Dresden",
    ("Germany 2 Bundesliga", "greuther fürth"): "Greuther Furth",
    ("Germany 2 Bundesliga", "1. fc nürnberg"): "Nurnberg",
    ("Germany 2 Bundesliga", "sv darmstadt 98"): "Darmstadt",
    ("Germany 2 Bundesliga", "arminia bielefeld"): "Bielefeld",
    ("Germany 2 Bundesliga", "vfl osnabrück"): "Osnabruck",
    ("Germany 2 Bundesliga", "1. fc kaiserslautern"): "Kaiserslautern",
    ("Germany 2 Bundesliga", "1. fc heidenheim"): "Heidenheim",
    ("Germany 2 Bundesliga", "1. fc magdeburg"): "Magdeburg",
    ("Germany 2 Bundesliga", "fc st. pauli"): "St Pauli",
    ("Germany 2 Bundesliga", "vfl bochum"): "Bochum",
    ("Germany 2 Bundesliga", "karlsruher sc"): "Karlsruhe",

    # Países Baixos
    ("Netherlands", "ado den haag"): "Den Haag",
    ("Netherlands", "fc utrecht"): "Utrecht",
    ("Netherlands", "fc zwolle"): "Zwolle",
    ("Netherlands", "fortuna sittard"): "For Sittard",
    ("Netherlands", "nec nijmegen"): "Nijmegen",
    ("Netherlands", "sc cambuur"): "Cambuur",

    # Espanha
    ("Spain", "dep. a coruna"): "La Coruna",
    ("Spain", "deportivo la coruña"): "La Coruna",
    ("Spain", "ca osasuna"): "Osasuna",
    ("Spain", "alavés"): "Alaves",
    ("Spain", "atlético madrid"): "Ath Madrid",
    ("Spain", "celta vigo"): "Celta",
    ("Spain", "elche cf"): "Elche",
    ("Spain", "espanyol"): "Espanol",
    ("Spain", "málaga"): "Malaga",
    ("Spain", "real betis"): "Betis",
    ("Spain", "real sociedad"): "Sociedad",
    ("Spain", "rayo vallecano"): "Vallecano",
    ("Spain", "athletic bilbao"): "Ath Bilbao",
    ("Spain", "athletic club"): "Ath Bilbao",
    ("Spain", "real racing club de santander"): "Santander",
    ("Spain", "racing santander"): "Santander",

    # Inglaterra - League One
    ("England League One", "peterborough united"): "Peterboro",
    ("England League One", "sheffield wednesday"): "Sheffield Weds",
    ("England League One", "milton keynes dons"): "Milton Keynes Dons",

    # França - Ligue 2
    ("France Ligue 2", "stade lavallois"): "Laval",
    ("France Ligue 2", "stade laval"): "Laval",

    # Espanha - Segunda Division
    ("Spain Segunda Division", "cádiz cf"): "Cadiz",
    ("Spain Segunda Division", "cd castellón"): "Castellon",
    ("Spain Segunda Division", "sd eibar"): "Eibar",
    ("Spain Segunda Division", "leganés"): "Leganes",
    ("Spain Segunda Division", "ad ceuta fc"): "Ceuta",
    ("Spain Segunda Division", "almería"): "Almeria",
    ("Spain Segunda Division", "burgos cf"): "Burgos",
    ("Spain Segunda Division", "andorra cf"): "Andorra",
    ("Spain Segunda Division", "girona fc"): "Girona",
    ("Spain Segunda Division", "granada cf"): "Granada",
    ("Spain Segunda Division", "real valladolid cf"): "Valladolid",
    ("Spain Segunda Division", "sabadell fc"): "Sabadell",
    ("Spain Segunda Division", "real sociedad b"): "Sociedad B",
    ("Spain Segunda Division", "sporting gijón"): "Sp Gijon",
    ("Spain Segunda Division", "sporting gijon"): "Sp Gijon",
    ("Spain Segunda Division", "real sporting de gijon"): "Sp Gijon",

    # Portugal
    ("Portugal", "famalicão"): "Famalicao",
    ("Portugal", "fc porto"): "Porto",
    ("Portugal", "sporting lisbon"): "Sp Lisbon",
    ("Portugal", "moreirense fc"): "Moreirense",
    ("Portugal", "rio ave fc"): "Rio Ave",
    ("Portugal", "braga"): "Sp Braga",
    ("Portugal", "cs maritimo"): "Maritimo",
    ("Portugal", "vitória sc"): "Guimaraes",
    ("Portugal", "vitoria sc"): "Guimaraes",
    ("Portugal", "vitória guimarães"): "Guimaraes",
    ("Portugal", "vitoria guimaraes"): "Guimaraes",

    # Itália
    ("Italy", "atalanta bc"): "Atalanta",
    ("Italy", "ac milan"): "Milan",
    ("Italy", "as roma"): "Roma",
    ("Italy", "inter milan"): "Inter",

    # Inglaterra - Championship
    ("England Championship", "charlton athletic"): "Charlton",

    # Bélgica
    ("Belgium First Division A", "kv kortrijk"): "Kortrijk",
    ("Belgium First Division A", "sk beveren"): "Waasland-Beveren",
    ("Belgium First Division A", "sv zulte-waregem"): "Waregem",
    ("Belgium First Division A", "sint truiden"): "St Truiden",

    # Turquia
    ("Turkey Super Lig", "gazişehir gaziantep"): "Gaziantep",
    ("Turkey Super Lig", "kasimpasa sk"): "Kasimpasa",
    ("Turkey Super Lig", "eyüpspor"): "Eyupspor",
    ("Turkey Super Lig", "torku konyaspor"): "Konyaspor",

    # Inglaterra - Premier League
    ("England", "manchester united"): "Man United",

    # França
    ("France", "paris saint germain"): "Paris SG",
    ("France", "as monaco"): "Monaco",
    ("France", "rc lens"): "Lens",

    # Países Baixos
    ("Netherlands", "fc twente enschede"): "Twente",
    ("Netherlands", "sc telstar"): "Telstar",
}


# ============================================================
# FUNÇÕES DE CONVERSÃO
# ============================================================

def clean_text(value) -> str | None:
    """
    Converte um valor para texto limpo.
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return None

    return text


def to_float(value) -> float | None:
    """
    Converte texto ou número para float.
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip().replace(",", ".")

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def to_odd(value) -> float | None:
    """
    Converte e valida uma odd decimal.
    """

    number = to_float(value)

    if number is None or number <= 1.0:
        return None

    return number


def first_value(
    row: pd.Series,
    column_names: list[str],
):
    """
    Obtém o primeiro valor preenchido de várias colunas.
    """

    for column_name in column_names:
        if column_name not in row.index:
            continue

        value = row[column_name]

        if clean_text(value) is not None:
            return value

    return None


def first_valid_odd(
    row: pd.Series,
    column_names: list[str],
) -> float | None:
    """
    Obtém a primeira odd válida de várias colunas.
    """

    for column_name in column_names:
        if column_name not in row.index:
            continue

        odd = to_odd(
            row[column_name]
        )

        if odd is not None:
            return odd

    return None


def parse_date(value) -> pd.Timestamp | None:
    """
    Aceita:

    YYYY-MM-DD
    DD/MM/YYYY
    DD/MM/YY
    """

    text = clean_text(value)

    if text is None:
        return None

    supported_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
    ]

    for date_format in supported_formats:
        try:
            return pd.Timestamp(
                datetime.strptime(
                    text,
                    date_format,
                )
            )
        except ValueError:
            continue

    parsed = pd.to_datetime(
        text,
        dayfirst=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return pd.Timestamp(parsed)


def calculate_season(
    match_date: pd.Timestamp,
) -> str:
    """
    Calcula a época europeia.
    """

    if match_date.month >= 7:
        start_year = match_date.year
    else:
        start_year = match_date.year - 1

    return f"{start_year}/{start_year + 1}"


def normalize_league(
    league_value,
    division_value=None,
) -> str | None:
    """
    Normaliza o nome da liga.
    """

    league = clean_text(
        league_value
    )

    if league is not None:
        return LEAGUE_ALIASES.get(
            league.lower(),
            league,
        )

    division = clean_text(
        division_value
    )

    if division is not None:
        return DIVISION_LEAGUE_MAP.get(
            division,
            division,
        )

    return None


def normalize_team(
    league: str | None,
    team_value,
) -> str | None:
    """
    Normaliza o nome da equipa para o nome usado no histórico.

    Os aliases são específicos por liga para evitar cruzar
    equipas ou históricos de divisões diferentes.
    """

    team = clean_text(
        team_value
    )

    if team is None:
        return None

    if league is None:
        return team

    return TEAM_ALIASES.get(
        (
            league,
            team.casefold(),
        ),
        team,
    )


# ============================================================
# FICHEIRO DE JOGOS FUTUROS
# ============================================================

def create_upcoming_template() -> None:
    """
    Cria o CSV que será preenchido com jogos futuros.
    """

    UPCOMING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = pd.DataFrame(
        columns=[
            "date",
            "time",
            "league",
            "home_team",
            "away_team",
            "odd_over25_open",
            "odd_under25_open",
        ]
    )

    template.to_csv(
        UPCOMING_PATH,
        index=False,
    )


def read_csv_robust(
    csv_path: Path,
) -> pd.DataFrame:
    """
    Lê o CSV utilizando diferentes codificações.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                csv_path,
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        f"Não foi possível ler: {csv_path}"
    )


# ============================================================
# CARREGAR O HISTÓRICO
# ============================================================

def load_historical_matches(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Carrega os jogos concluídos.
    """

    matches = pd.read_sql_query(
        """
        SELECT
            id,
            date,
            league,
            home_team,
            away_team,
            home_goals,
            away_goals
        FROM matches
        """,
        connection,
    )

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

    matches["date"] = (
        matches["date"].apply(
            parse_date
        )
    )

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="coerce",
    )

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="coerce",
    )

    for column in [
        "league",
        "home_team",
        "away_team",
    ]:
        matches[column] = (
            matches[column]
            .astype("string")
            .str.strip()
        )

    matches = matches.dropna(
        subset=required_columns
    ).copy()

    matches = matches.sort_values(
        [
            "date",
            "id",
        ]
    ).reset_index(drop=True)

    if matches.empty:
        raise ValueError(
            "A base não contém jogos históricos válidos."
        )

    return matches


def create_team_history(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria uma linha por equipa e por jogo.
    """

    home_history = pd.DataFrame(
        {
            "id": matches["id"],
            "date": matches["date"],
            "league": matches["league"],
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "venue": "home",
            "goals_for": matches["home_goals"],
            "goals_against": matches["away_goals"],
        }
    )

    away_history = pd.DataFrame(
        {
            "id": matches["id"],
            "date": matches["date"],
            "league": matches["league"],
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "venue": "away",
            "goals_for": matches["away_goals"],
            "goals_against": matches["home_goals"],
        }
    )

    history = pd.concat(
        [
            home_history,
            away_history,
        ],
        ignore_index=True,
    )

    history["goal_diff"] = (
        history["goals_for"]
        - history["goals_against"]
    )

    history["win"] = (
        history["goals_for"]
        > history["goals_against"]
    ).astype(int)

    history["draw"] = (
        history["goals_for"]
        == history["goals_against"]
    ).astype(int)

    history["points"] = np.select(
        [
            history["win"] == 1,
            history["draw"] == 1,
        ],
        [
            3,
            1,
        ],
        default=0,
    )

    history["btts"] = (
        (history["goals_for"] > 0)
        & (history["goals_against"] > 0)
    ).astype(int)

    history["over25"] = (
        (
            history["goals_for"]
            + history["goals_against"]
        )
        > 2
    ).astype(int)

    history["scored"] = (
        history["goals_for"] > 0
    ).astype(int)

    history["clean_sheet"] = (
        history["goals_against"] == 0
    ).astype(int)

    return history.sort_values(
        [
            "league",
            "team",
            "date",
            "id",
        ]
    ).reset_index(drop=True)


# ============================================================
# ELO
# ============================================================

def calculate_current_elo(
    matches: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    """
    Calcula o Elo atual de cada equipa.
    """

    ratings: dict[
        tuple[str, str],
        float,
    ] = {}

    ordered = matches.sort_values(
        [
            "date",
            "id",
        ]
    )

    for row in ordered.itertuples(
        index=False
    ):
        league = str(row.league)
        home_team = str(row.home_team)
        away_team = str(row.away_team)

        home_key = (
            league,
            home_team,
        )

        away_key = (
            league,
            away_team,
        )

        home_rating = ratings.get(
            home_key,
            1500.0,
        )

        away_rating = ratings.get(
            away_key,
            1500.0,
        )

        expected_home = 1 / (
            1
            + 10
            ** (
                (
                    away_rating
                    - (
                        home_rating
                        + HOME_ADVANTAGE_ELO
                    )
                )
                / 400
            )
        )

        if row.home_goals > row.away_goals:
            actual_home = 1.0

        elif row.home_goals < row.away_goals:
            actual_home = 0.0

        else:
            actual_home = 0.5

        expected_away = 1 - expected_home
        actual_away = 1 - actual_home

        ratings[home_key] = (
            home_rating
            + ELO_K_FACTOR
            * (
                actual_home
                - expected_home
            )
        )

        ratings[away_key] = (
            away_rating
            + ELO_K_FACTOR
            * (
                actual_away
                - expected_away
            )
        )

    return ratings


# ============================================================
# FEATURES DAS EQUIPAS
# ============================================================

def team_suggestions(
    history: pd.DataFrame,
    league: str,
    team: str,
) -> list[str]:
    """
    Sugere nomes semelhantes existentes na base.
    """

    available_teams = (
        history.loc[
            history["league"] == league,
            "team",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return difflib.get_close_matches(
        team,
        available_teams,
        n=3,
        cutoff=0.55,
    )


def get_team_snapshot(
    history: pd.DataFrame,
    league: str,
    team: str,
    fixture_date: pd.Timestamp,
    venue: str,
) -> dict[str, float]:
    """
    Calcula a forma pré-jogo de uma equipa.
    """

    team_history = history[
        (history["league"] == league)
        & (history["team"] == team)
        & (history["date"] < fixture_date)
    ].sort_values(
        [
            "date",
            "id",
        ]
    )

    if len(team_history) < MIN_GENERAL_HISTORY:
        suggestions = team_suggestions(
            history,
            league,
            team,
        )

        suggestion_text = ""

        if suggestions:
            suggestion_text = (
                " Sugestões: "
                + ", ".join(suggestions)
            )

        raise ValueError(
            f"Histórico insuficiente para {team} "
            f"em {league}: {len(team_history)} jogos."
            + suggestion_text
        )

    recent = team_history.tail(
        ROLLING_WINDOW
    )

    general = {
        "avg_scored_5":
            float(
                recent["goals_for"].mean()
            ),

        "avg_conceded_5":
            float(
                recent["goals_against"].mean()
            ),

        "avg_goal_diff_5":
            float(
                recent["goal_diff"].mean()
            ),

        "points_avg_5":
            float(
                recent["points"].mean()
            ),

        "win_rate_5":
            float(
                recent["win"].mean()
            ),

        "btts_rate_5":
            float(
                recent["btts"].mean()
            ),

        "over25_rate_5":
            float(
                recent["over25"].mean()
            ),

        "scored_rate_5":
            float(
                recent["scored"].mean()
            ),

        "clean_sheet_rate_5":
            float(
                recent["clean_sheet"].mean()
            ),
    }

    venue_history = team_history[
        team_history["venue"] == venue
    ]

    if len(venue_history) >= MIN_VENUE_HISTORY:
        venue_recent = venue_history.tail(
            ROLLING_WINDOW
        )

        venue_avg_scored = float(
            venue_recent["goals_for"].mean()
        )

        venue_avg_conceded = float(
            venue_recent["goals_against"].mean()
        )

        venue_points_avg = float(
            venue_recent["points"].mean()
        )

    else:
        venue_avg_scored = general[
            "avg_scored_5"
        ]

        venue_avg_conceded = general[
            "avg_conceded_5"
        ]

        venue_points_avg = general[
            "points_avg_5"
        ]

    last_match_date = pd.Timestamp(
        team_history["date"].max()
    )

    rest_days = int(
        (
            fixture_date.normalize()
            - last_match_date.normalize()
        ).days
    )

    rest_days = int(
        np.clip(
            rest_days,
            0,
            60,
        )
    )

    return {
        **general,

        "venue_avg_scored_5":
            venue_avg_scored,

        "venue_avg_conceded_5":
            venue_avg_conceded,

        "venue_points_avg_5":
            venue_points_avg,

        "rest_days":
            float(rest_days),
    }


def build_sports_features(
    history: pd.DataFrame,
    elo_ratings: dict[tuple[str, str], float],
    league: str,
    home_team: str,
    away_team: str,
    fixture_date: pd.Timestamp,
) -> dict[str, float]:
    """
    Cria as 43 features desportivas.
    """

    home = get_team_snapshot(
        history=history,
        league=league,
        team=home_team,
        fixture_date=fixture_date,
        venue="home",
    )

    away = get_team_snapshot(
        history=history,
        league=league,
        team=away_team,
        fixture_date=fixture_date,
        venue="away",
    )

    home_elo = float(
        elo_ratings.get(
            (
                league,
                home_team,
            ),
            1500.0,
        )
    )

    away_elo = float(
        elo_ratings.get(
            (
                league,
                away_team,
            ),
            1500.0,
        )
    )

    features = {
        "home_avg_scored_5":
            home["avg_scored_5"],

        "home_avg_conceded_5":
            home["avg_conceded_5"],

        "home_avg_goal_diff_5":
            home["avg_goal_diff_5"],

        "home_points_avg_5":
            home["points_avg_5"],

        "home_win_rate_5":
            home["win_rate_5"],

        "home_btts_rate_5":
            home["btts_rate_5"],

        "home_over25_rate_5":
            home["over25_rate_5"],

        "home_scored_rate_5":
            home["scored_rate_5"],

        "home_clean_sheet_rate_5":
            home["clean_sheet_rate_5"],

        "away_avg_scored_5":
            away["avg_scored_5"],

        "away_avg_conceded_5":
            away["avg_conceded_5"],

        "away_avg_goal_diff_5":
            away["avg_goal_diff_5"],

        "away_points_avg_5":
            away["points_avg_5"],

        "away_win_rate_5":
            away["win_rate_5"],

        "away_btts_rate_5":
            away["btts_rate_5"],

        "away_over25_rate_5":
            away["over25_rate_5"],

        "away_scored_rate_5":
            away["scored_rate_5"],

        "away_clean_sheet_rate_5":
            away["clean_sheet_rate_5"],

        "home_home_avg_scored_5":
            home["venue_avg_scored_5"],

        "home_home_avg_conceded_5":
            home["venue_avg_conceded_5"],

        "home_home_points_avg_5":
            home["venue_points_avg_5"],

        "away_away_avg_scored_5":
            away["venue_avg_scored_5"],

        "away_away_avg_conceded_5":
            away["venue_avg_conceded_5"],

        "away_away_points_avg_5":
            away["venue_points_avg_5"],

        "home_rest_days":
            home["rest_days"],

        "away_rest_days":
            away["rest_days"],

        "home_elo":
            home_elo,

        "away_elo":
            away_elo,

        "elo_diff":
            home_elo - away_elo,
    }

    features["expected_home_goals"] = (
        features["home_avg_scored_5"]
        + features["away_avg_conceded_5"]
    ) / 2

    features["expected_away_goals"] = (
        features["away_avg_scored_5"]
        + features["home_avg_conceded_5"]
    ) / 2

    features["expected_total_goals"] = (
        features["expected_home_goals"]
        + features["expected_away_goals"]
    )

    features["expected_goal_difference"] = (
        features["expected_home_goals"]
        - features["expected_away_goals"]
    )

    features["total_expected_scored"] = (
        features["home_avg_scored_5"]
        + features["away_avg_scored_5"]
    )

    features["total_expected_conceded"] = (
        features["home_avg_conceded_5"]
        + features["away_avg_conceded_5"]
    )

    features["home_power_rating"] = (
        features["home_avg_goal_diff_5"]
        + features["home_points_avg_5"]
    )

    features["away_power_rating"] = (
        features["away_avg_goal_diff_5"]
        + features["away_points_avg_5"]
    )

    features["power_rating_diff"] = (
        features["home_power_rating"]
        - features["away_power_rating"]
    )

    features["form_points_diff"] = (
        features["home_points_avg_5"]
        - features["away_points_avg_5"]
    )

    features["btts_signal"] = (
        features["home_btts_rate_5"]
        + features["away_btts_rate_5"]
    ) / 2

    features["over25_signal"] = (
        features["home_over25_rate_5"]
        + features["away_over25_rate_5"]
    ) / 2

    features["scoring_signal"] = (
        features["home_scored_rate_5"]
        + features["away_scored_rate_5"]
    ) / 2

    features["clean_sheet_signal"] = (
        features["home_clean_sheet_rate_5"]
        + features["away_clean_sheet_rate_5"]
    ) / 2

    return features


def build_market_features(
    odd_over_open: float,
    odd_under_open: float,
) -> dict[str, float]:
    """
    Cria as cinco features das odds de abertura.
    """

    implied_over = (
        1 / odd_over_open
    )

    implied_under = (
        1 / odd_under_open
    )

    implied_total = (
        implied_over
        + implied_under
    )

    fair_probability = (
        implied_over
        / implied_total
    )

    clipped_probability = float(
        np.clip(
            fair_probability,
            0.0001,
            0.9999,
        )
    )

    market_logit = float(
        np.log(
            clipped_probability
            / (
                1
                - clipped_probability
            )
        )
    )

    return {
        "odd_over25_open":
            odd_over_open,

        "odd_under25_open":
            odd_under_open,

        "fair_probability_open":
            fair_probability,

        "market_logit_open":
            market_logit,

        "market_margin_open":
            implied_total - 1,
    }


# ============================================================
# TABELA DE PAPER TRADING
# ============================================================

def prepare_paper_trade_table(
    connection: sqlite3.Connection,
) -> None:
    """
    Cria a tabela persistente de paper trading.
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            prediction_timestamp TEXT NOT NULL,

            fixture_date TEXT NOT NULL,
            fixture_time TEXT,

            season TEXT NOT NULL,
            league TEXT NOT NULL,

            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,

            market TEXT NOT NULL,

            odd_over25_open REAL NOT NULL,
            odd_under25_open REAL NOT NULL,

            fair_probability_open REAL NOT NULL,
            clv_probability REAL NOT NULL,

            training_base_rate REAL NOT NULL,
            signal_threshold REAL NOT NULL,
            edge_over_base_rate REAL NOT NULL,

            signal INTEGER NOT NULL,
            planned_stake REAL NOT NULL,

            status TEXT NOT NULL,

            odd_over25_close REAL,
            odd_under25_close REAL,

            home_goals INTEGER,
            away_goals INTEGER,

            result_over25 INTEGER,
            clv REAL,
            profit REAL,

            model_generated_at TEXT,
            features_json TEXT NOT NULL,

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        ux_paper_trades_fixture_market
        ON paper_trades (
            fixture_date,
            league,
            home_team,
            away_team,
            market
        )
        """
    )

    connection.commit()


def insert_paper_trade(
    connection: sqlite3.Connection,
    record: dict,
) -> bool:
    """
    Guarda apenas a primeira previsão de cada jogo.

    Uma previsão já existente nunca é alterada.
    """

    changes_before = (
        connection.total_changes
    )

    connection.execute(
        """
        INSERT INTO paper_trades (
            prediction_timestamp,
            fixture_date,
            fixture_time,
            season,
            league,
            home_team,
            away_team,
            market,
            odd_over25_open,
            odd_under25_open,
            fair_probability_open,
            clv_probability,
            training_base_rate,
            signal_threshold,
            edge_over_base_rate,
            signal,
            planned_stake,
            status,
            model_generated_at,
            features_json,
            created_at,
            updated_at
        )
        VALUES (
            :prediction_timestamp,
            :fixture_date,
            :fixture_time,
            :season,
            :league,
            :home_team,
            :away_team,
            :market,
            :odd_over25_open,
            :odd_under25_open,
            :fair_probability_open,
            :clv_probability,
            :training_base_rate,
            :signal_threshold,
            :edge_over_base_rate,
            :signal,
            :planned_stake,
            :status,
            :model_generated_at,
            :features_json,
            :created_at,
            :updated_at
        )
        ON CONFLICT (
            fixture_date,
            league,
            home_team,
            away_team,
            market
        )
        DO NOTHING
        """,
        record,
    )

    return (
        connection.total_changes
        > changes_before
    )


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    UPCOMING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Base de dados não encontrada: "
            f"{DATABASE_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo CLV não encontrado: "
            f"{MODEL_PATH}"
        )

    # Na primeira execução, cria apenas o modelo do CSV.
    if not UPCOMING_PATH.exists():
        create_upcoming_template()

        print("\nFoi criado o ficheiro:")

        print(
            UPCOMING_PATH.relative_to(
                ROOT_DIR
            )
        )

        print(
            "\nPreenche esse ficheiro com jogos futuros "
            "e executa novamente."
        )

        print("\nColunas:")

        print(
            "date,time,league,home_team,away_team,"
            "odd_over25_open,odd_under25_open"
        )

        return

    upcoming = read_csv_robust(
        UPCOMING_PATH
    )

    if upcoming.empty:
        print(
            "\nO ficheiro data/upcoming/"
            "upcoming_matches.csv está vazio."
        )

        print(
            "Adiciona os jogos futuros e executa novamente."
        )

        return

    model_bundle = joblib.load(
        MODEL_PATH
    )

    required_bundle_keys = [
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

    missing_bundle_keys = [
        key
        for key in required_bundle_keys
        if key not in model_bundle
    ]

    if missing_bundle_keys:
        raise ValueError(
            "O modelo guardado está incompleto: "
            + ", ".join(missing_bundle_keys)
        )

    model = model_bundle["model"]

    base_features = list(
        model_bundle["base_features"]
    )

    market_features = list(
        model_bundle["market_features"]
    )

    model_features = list(
        model_bundle["model_features"]
    )

    training_base_rate = float(
        model_bundle["training_base_rate"]
    )

    signal_threshold = float(
        model_bundle["signal_threshold"]
    )

    min_odd = float(
        model_bundle["min_odd"]
    )

    max_odd = float(
        model_bundle["max_odd"]
    )

    model_generated_at = str(
        model_bundle["generated_at"]
    )

    prediction_timestamp = (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )

    today = datetime.now().date()

    predictions = []
    skipped = []

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        prepare_paper_trade_table(
            connection
        )

        matches = load_historical_matches(
            connection
        )

        team_history = create_team_history(
            matches
        )

        elo_ratings = calculate_current_elo(
            matches
        )

        latest_history_date = pd.Timestamp(
            matches["date"].max()
        )

        for source_index, row in upcoming.iterrows():

            csv_row_number = (
                int(source_index) + 2
            )

            try:
                fixture_date = parse_date(
                    first_value(
                        row,
                        [
                            "date",
                            "Date",
                        ],
                    )
                )

                fixture_time = clean_text(
                    first_value(
                        row,
                        [
                            "time",
                            "Time",
                        ],
                    )
                )

                league = normalize_league(
                    first_value(
                        row,
                        [
                            "league",
                            "League",
                        ],
                    ),
                    first_value(
                        row,
                        [
                            "division",
                            "Div",
                        ],
                    ),
                )

                home_team = normalize_team(
                    league,
                    first_value(
                        row,
                        [
                            "home_team",
                            "HomeTeam",
                        ],
                    ),
                )

                away_team = normalize_team(
                    league,
                    first_value(
                        row,
                        [
                            "away_team",
                            "AwayTeam",
                        ],
                    ),
                )

                odd_over_open = first_valid_odd(
                    row,
                    [
                        "odd_over25_open",
                        "Avg>2.5",
                        "B365>2.5",
                        "P>2.5",
                    ],
                )

                odd_under_open = first_valid_odd(
                    row,
                    [
                        "odd_under25_open",
                        "Avg<2.5",
                        "B365<2.5",
                        "P<2.5",
                    ],
                )

                if fixture_date is None:
                    raise ValueError(
                        "data inválida"
                    )

                if fixture_date.date() < today:
                    raise ValueError(
                        "o jogo já ocorreu ou está no passado"
                    )

                if (
                    fixture_date.normalize()
                    <= latest_history_date.normalize()
                ):
                    raise ValueError(
                        "a data não é posterior ao último "
                        f"jogo histórico: "
                        f"{latest_history_date.date()}"
                    )

                if league is None:
                    raise ValueError(
                        "liga em falta"
                    )

                if home_team is None:
                    raise ValueError(
                        "equipa da casa em falta"
                    )

                if away_team is None:
                    raise ValueError(
                        "equipa visitante em falta"
                    )

                if odd_over_open is None:
                    raise ValueError(
                        "odd Over 2.5 inválida"
                    )

                if odd_under_open is None:
                    raise ValueError(
                        "odd Under 2.5 inválida"
                    )

                sports_features = (
                    build_sports_features(
                        history=team_history,
                        elo_ratings=elo_ratings,
                        league=league,
                        home_team=home_team,
                        away_team=away_team,
                        fixture_date=fixture_date,
                    )
                )

                missing_sports_features = [
                    feature
                    for feature in base_features
                    if feature not in sports_features
                ]

                if missing_sports_features:
                    raise ValueError(
                        "features desportivas em falta: "
                        + ", ".join(
                            missing_sports_features
                        )
                    )

                market_values = (
                    build_market_features(
                        odd_over_open,
                        odd_under_open,
                    )
                )

                missing_market_features = [
                    feature
                    for feature in market_features
                    if feature not in market_values
                ]

                if missing_market_features:
                    raise ValueError(
                        "features de mercado em falta: "
                        + ", ".join(
                            missing_market_features
                        )
                    )

                prediction_values = {
                    **sports_features,
                    **market_values,
                    "league": league,
                }

                prediction_frame = pd.DataFrame(
                    [
                        prediction_values
                    ]
                )

                missing_model_features = [
                    feature
                    for feature in model_features
                    if feature
                    not in prediction_frame.columns
                ]

                if missing_model_features:
                    raise ValueError(
                        "features do modelo em falta: "
                        + ", ".join(
                            missing_model_features
                        )
                    )

                probabilities = (
                    model.predict_proba(
                        prediction_frame[
                            model_features
                        ]
                    )
                )

                classes = list(
                    model.classes_
                )

                if 1 not in classes:
                    raise ValueError(
                        "o modelo não contém a classe 1"
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

                edge_over_base_rate = (
                    clv_probability
                    - training_base_rate
                )

                valid_signal_odd = (
                    min_odd
                    <= odd_over_open
                    <= max_odd
                )

                signal = int(
                    valid_signal_odd
                    and (
                        clv_probability
                        >= signal_threshold
                    )
                )

                stored_features = {}

                for feature in model_features:
                    value = prediction_values[
                        feature
                    ]

                    if feature == "league":
                        stored_features[
                            feature
                        ] = str(value)
                    else:
                        stored_features[
                            feature
                        ] = float(value)

                record = {
                    "prediction_timestamp":
                        prediction_timestamp,

                    "fixture_date":
                        fixture_date.strftime(
                            "%Y-%m-%d"
                        ),

                    "fixture_time":
                        fixture_time,

                    "season":
                        calculate_season(
                            fixture_date
                        ),

                    "league":
                        league,

                    "home_team":
                        home_team,

                    "away_team":
                        away_team,

                    "market":
                        MARKET,

                    "odd_over25_open":
                        odd_over_open,

                    "odd_under25_open":
                        odd_under_open,

                    "fair_probability_open":
                        market_values[
                            "fair_probability_open"
                        ],

                    "clv_probability":
                        clv_probability,

                    "training_base_rate":
                        training_base_rate,

                    "signal_threshold":
                        signal_threshold,

                    "edge_over_base_rate":
                        edge_over_base_rate,

                    "signal":
                        signal,

                    "planned_stake":
                        1.0 if signal else 0.0,

                    "status":
                        "OPEN",

                    "model_generated_at":
                        model_generated_at,

                    "features_json":
                        json.dumps(
                            stored_features,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),

                    "created_at":
                        prediction_timestamp,

                    "updated_at":
                        prediction_timestamp,
                }

                inserted = insert_paper_trade(
                    connection,
                    record,
                )

                predictions.append(
                    {
                        "source_row":
                            csv_row_number,

                        "fixture_date":
                            record["fixture_date"],

                        "fixture_time":
                            fixture_time,

                        "season":
                            record["season"],

                        "league":
                            league,

                        "home_team":
                            home_team,

                        "away_team":
                            away_team,

                        "odd_over25_open":
                            odd_over_open,

                        "odd_under25_open":
                            odd_under_open,

                        "fair_probability_open":
                            record[
                                "fair_probability_open"
                            ],

                        "clv_probability":
                            clv_probability,

                        "training_base_rate":
                            training_base_rate,

                        "signal_threshold":
                            signal_threshold,

                        "edge_over_base_rate":
                            edge_over_base_rate,

                        "signal":
                            signal,

                        "registry_status":
                            (
                                "inserted"
                                if inserted
                                else "already_exists"
                            ),
                    }
                )

            except Exception as error:
                skipped.append(
                    {
                        "source_row":
                            csv_row_number,

                        "reason":
                            str(error),
                    }
                )

        connection.commit()

        registry = pd.read_sql_query(
            """
            SELECT *
            FROM paper_trades
            ORDER BY
                fixture_date,
                fixture_time,
                league,
                home_team
            """,
            connection,
        )

    predictions_df = pd.DataFrame(
        predictions
    )

    skipped_df = pd.DataFrame(
        skipped
    )

    if predictions_df.empty:
        signals_df = pd.DataFrame()

    else:
        predictions_df = (
            predictions_df.sort_values(
                [
                    "fixture_date",
                    "fixture_time",
                    "league",
                    "home_team",
                ],
                na_position="last",
            )
        )

        signals_df = predictions_df[
            predictions_df["signal"] == 1
        ].copy()

    predictions_df.to_csv(
        PREDICTIONS_REPORT_PATH,
        index=False,
    )

    signals_df.to_csv(
        SIGNALS_REPORT_PATH,
        index=False,
    )

    skipped_df.to_csv(
        SKIPPED_REPORT_PATH,
        index=False,
    )

    registry.to_csv(
        REGISTRY_REPORT_PATH,
        index=False,
    )

    if predictions_df.empty:
        inserted_count = 0
        existing_count = 0
        signal_count = 0

    else:
        inserted_count = int(
            (
                predictions_df[
                    "registry_status"
                ]
                == "inserted"
            ).sum()
        )

        existing_count = int(
            (
                predictions_df[
                    "registry_status"
                ]
                == "already_exists"
            ).sum()
        )

        signal_count = int(
            predictions_df[
                "signal"
            ].sum()
        )

    print("\n==========================================")
    print(" FOOTBALLAI — PAPER TRADING")
    print("==========================================\n")

    print(
        f"Último jogo histórico    : "
        f"{latest_history_date.date()}"
    )

    print(
        f"Jogos lidos              : "
        f"{len(upcoming)}"
    )

    print(
        f"Previsões válidas        : "
        f"{len(predictions_df)}"
    )

    print(
        f"Novos registos           : "
        f"{inserted_count}"
    )

    print(
        f"Já existentes            : "
        f"{existing_count}"
    )

    print(
        f"Linhas ignoradas         : "
        f"{len(skipped_df)}"
    )

    print(
        f"Sinais CLV               : "
        f"{signal_count}"
    )

    print(
        f"Taxa-base do treino      : "
        f"{training_base_rate:.2%}"
    )

    print(
        f"Limiar congelado         : "
        f"{signal_threshold:.2%}"
    )

    if not signals_df.empty:
        print("\nSinais registados:\n")

        display_columns = [
            "fixture_date",
            "fixture_time",
            "league",
            "home_team",
            "away_team",
            "odd_over25_open",
            "clv_probability",
            "edge_over_base_rate",
        ]

        print(
            signals_df[
                display_columns
            ].to_string(
                index=False,
                formatters={
                    "odd_over25_open":
                        "{:.2f}".format,

                    "clv_probability":
                        "{:.2%}".format,

                    "edge_over_base_rate":
                        "{:+.2%}".format,
                },
            )
        )

    else:
        print(
            "\nNenhum jogo ultrapassou "
            "o limiar congelado."
        )

        print(
            "O resultado correto neste caso é: "
            "sem sinal."
        )

    if not skipped_df.empty:
        print(
            "\nExistem linhas ignoradas. Consulta:"
        )

        print(
            "reports/"
            "paper_trade_skipped_latest.csv"
        )

    print("\nRelatórios guardados:")

    print(
        "✅ reports/"
        "paper_trade_predictions_latest.csv"
    )

    print(
        "✅ reports/"
        "paper_trade_signals_latest.csv"
    )

    print(
        "✅ reports/"
        "paper_trade_skipped_latest.csv"
    )

    print(
        "✅ reports/"
        "paper_trade_registry.csv"
    )

    print(
        "\n✅ A primeira previsão de cada jogo "
        "fica congelada."
    )

    print(
        "✅ Executar novamente não altera "
        "previsões já registadas."
    )

    print(
        "⚠ Paper trading apenas; sem dinheiro real."
    )


if __name__ == "__main__":
    main()