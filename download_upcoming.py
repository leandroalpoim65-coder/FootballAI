from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ml.paper_trade import normalize_league, normalize_team


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

UPCOMING_DIR = (
    ROOT_DIR
    / "data"
    / "upcoming"
)

DOWNLOADS_DIR = (
    UPCOMING_DIR
    / "downloads"
)

OUTPUT_PATH = (
    UPCOMING_DIR
    / "upcoming_matches.csv"
)

ENV_PATH = (
    ROOT_DIR
    / ".env"
)


# ============================================================
# FONTES
# ============================================================

FOOTBALL_DATA_URL = (
    "https://www.football-data.co.uk/fixtures.csv"
)

ODDS_API_BASE_URL = (
    "https://api.the-odds-api.com/v4"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DIVISION_LEAGUE_MAP = {
    "E0": "England",
    "SP1": "Spain",
    "I1": "Italy",
    "D1": "Germany",
    "F1": "France",
    "N1": "Netherlands",
    "P1": "Portugal",
}


# ============================================================
# COLUNAS
# ============================================================

BASE_COLUMNS = [
    "date",
    "time",
    "league",
    "home_team",
    "away_team",
]


CURRENT_COLUMNS = [
    "odd_over25_current",
    "odd_under25_current",
]


OPEN_COLUMNS = [
    "odd_over25_open",
    "odd_under25_open",
]


OUTPUT_COLUMNS = [
    "date",
    "time",
    "league",
    "home_team",
    "away_team",
    "odd_over25_open",
    "odd_under25_open",
    "odd_over25_current",
    "odd_under25_current",
]


# Ligas consultadas por defeito na The Odds API.
#
# A lista é fixa de propósito. Antes, o programa ordenava todas as
# competições de futebol ativas e consultava apenas as primeiras 20.
# Isso fazia ligas importantes desaparecerem entre atualizações.
DEFAULT_ODDS_API_SPORT_KEYS = [
    "soccer_belgium_first_div",
    "soccer_epl",
    "soccer_efl_champ",
    "soccer_france_ligue_one",
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_italy_serie_a",
    "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga",
    "soccer_spain_la_liga",
    "soccer_spain_segunda_division",
    "soccer_turkey_super_league",
]


QUOTA_ERROR_MARKERS = (
    "OUT_OF_USAGE_CREDITS",
    "Usage quota has been reached",
)


# ============================================================
# .ENV
# ============================================================

def load_simple_env(
    path: Path,
) -> None:
    """
    Carrega variáveis simples de um ficheiro .env
    sem dependências externas.

    Não substitui variáveis já existentes
    no ambiente.
    """

    if not path.exists():
        return

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        key = key.strip()

        value = (
            value
            .strip()
            .strip('"')
            .strip("'")
        )

        if key:
            os.environ.setdefault(
                key,
                value,
            )


# ============================================================
# HTTP
# ============================================================

def redact_url_secrets(
    url: str,
) -> str:
    """
    Remove a API key das mensagens de erro.

    Assim, se um erro for copiado para o terminal ou para um relatório,
    a chave da The Odds API não fica exposta.
    """

    token = "apiKey="

    if token not in url:
        return url

    prefix, remainder = url.split(
        token,
        1,
    )

    if "&" in remainder:
        _, suffix = remainder.split(
            "&",
            1,
        )

        return (
            f"{prefix}{token}***&{suffix}"
        )

    return f"{prefix}{token}***"


def http_get_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[
    bytes,
    dict[str, str],
]:

    request_headers = {
        "User-Agent":
            "Mozilla/5.0 FootballAI/2.0",
    }

    if headers:
        request_headers.update(
            headers
        )

    request = Request(
        url,
        headers=request_headers,
    )

    try:

        with urlopen(
            request,
            timeout=timeout,
        ) as response:

            status = getattr(
                response,
                "status",
                200,
            )

            if status != 200:
                raise RuntimeError(
                    "O servidor respondeu "
                    f"com HTTP {status}."
                )

            content = response.read()

            response_headers = {
                key.lower(): value
                for key, value
                in response.headers.items()
            }

    except HTTPError as error:

        try:

            detail = (
                error
                .read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

        except Exception:

            detail = ""

        safe_url = redact_url_secrets(
            url
        )

        message = (
            f"Erro HTTP {error.code} "
            f"ao aceder a {safe_url}: "
            f"{error.reason}"
        )

        if detail:

            message += (
                "\nResposta: "
                f"{detail[:500]}"
            )

        raise RuntimeError(
            message
        ) from error

    except URLError as error:

        safe_url = redact_url_secrets(
            url
        )

        raise RuntimeError(
            f"Não foi possível ligar a "
            f"{safe_url}: {error.reason}"
        ) from error

    except TimeoutError as error:

        safe_url = redact_url_secrets(
            url
        )

        raise RuntimeError(
            f"O pedido a {safe_url} "
            "excedeu o tempo limite."
        ) from error

    if not content:

        safe_url = redact_url_secrets(
            url
        )

        raise RuntimeError(
            f"A resposta de {safe_url} "
            "está vazia."
        )

    return (
        content,
        response_headers,
    )


def http_get_json(
    url: str,
    *,
    timeout: int = 30,
) -> tuple[
    object,
    dict[str, str],
]:

    content, headers = (
        http_get_bytes(
            url,
            timeout=timeout,
        )
    )

    try:

        payload = json.loads(
            content.decode(
                "utf-8"
            )
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:

        raise RuntimeError(
            f"A resposta de {url} "
            "não contém JSON válido."
        ) from error

    return (
        payload,
        headers,
    )


# ============================================================
# FUNÇÕES GERAIS
# ============================================================

def empty_observations() -> pd.DataFrame:

    return pd.DataFrame(
        columns=(
            BASE_COLUMNS
            + CURRENT_COLUMNS
        )
    )


def empty_output() -> pd.DataFrame:

    return pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )


def clean_text_value(
    value: object,
) -> str | None:

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if (
        not text
        or text.lower()
        in {
            "nan",
            "none",
            "<na>",
        }
    ):
        return None

    return text


def normalize_text(
    value: object,
) -> str:

    text = clean_text_value(
        value
    )

    if text is None:
        return ""

    return (
        text
        .strip()
        .casefold()
    )


def parse_dates(
    series: pd.Series,
) -> pd.Series:

    text = (
        series
        .astype("string")
        .str.strip()
    )

    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    for date_format in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
    ):

        mask = (
            parsed.isna()
            & text.notna()
            & (text != "")
        )

        if not mask.any():
            break

        parsed.loc[
            mask
        ] = pd.to_datetime(
            text.loc[
                mask
            ],
            format=date_format,
            errors="coerce",
        )

    return parsed


def first_numeric_column(
    dataframe: pd.DataFrame,
    column_names: list[str],
) -> pd.Series:

    result = pd.Series(
        np.nan,
        index=dataframe.index,
        dtype=float,
    )

    for column_name in column_names:

        if (
            column_name
            not in dataframe.columns
        ):
            continue

        candidate = pd.to_numeric(
            dataframe[
                column_name
            ],
            errors="coerce",
        )

        candidate = candidate.where(
            candidate > 1.0
        )

        result = result.fillna(
            candidate
        )

    return result


def fixture_key(
    date_value: object,
    league: object,
    home_team: object,
    away_team: object,
) -> str:
    """
    Chave EXATA da fonte.

    Continua a ser útil para preservar a opening odd da mesma
    representação do jogo quando ela já existia no CSV anterior.
    """

    return "|".join(
        [
            normalize_text(
                date_value
            ),
            normalize_text(
                league
            ),
            normalize_text(
                home_team
            ),
            normalize_text(
                away_team
            ),
        ]
    )


def canonical_fixture_key(
    date_value: object,
    league: object,
    home_team: object,
    away_team: object,
) -> str:
    """
    Chave canónica partilhada com o Paper Trade.

    Exemplo:
      Championship | Cardiff City | Wrexham AFC
      E1           | Cardiff      | Wrexham

    passam ambos para:
      England Championship | Cardiff | Wrexham

    Assim, duas fontes diferentes não criam duas linhas para o
    mesmo jogo.
    """

    canonical_league = normalize_league(
        league
    )

    canonical_home = normalize_team(
        canonical_league,
        home_team,
    )

    canonical_away = normalize_team(
        canonical_league,
        away_team,
    )

    return "|".join(
        [
            normalize_text(
                date_value
            ),
            normalize_text(
                canonical_league
            ),
            normalize_text(
                canonical_home
            ),
            normalize_text(
                canonical_away
            ),
        ]
    )


# ============================================================
# NORMALIZAR OBSERVAÇÕES ATUAIS
# ============================================================

def normalize_observations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if dataframe.empty:
        return empty_observations()

    result = dataframe.copy()

    required = (
        BASE_COLUMNS
        + CURRENT_COLUMNS
    )

    for column in required:

        if column not in result.columns:
            result[
                column
            ] = pd.NA

    result[
        "odd_over25_current"
    ] = pd.to_numeric(
        result[
            "odd_over25_current"
        ],
        errors="coerce",
    )

    result[
        "odd_under25_current"
    ] = pd.to_numeric(
        result[
            "odd_under25_current"
        ],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "date",
            "league",
            "home_team",
            "away_team",
            "odd_over25_current",
            "odd_under25_current",
        ]
    ).copy()

    result = result[
        (
            result[
                "odd_over25_current"
            ]
            > 1.0
        )
        &
        (
            result[
                "odd_under25_current"
            ]
            > 1.0
        )
    ].copy()

    result[
        "_fixture_key"
    ] = result.apply(
        lambda row: canonical_fixture_key(
            row[
                "date"
            ],
            row[
                "league"
            ],
            row[
                "home_team"
            ],
            row[
                "away_team"
            ],
        ),
        axis=1,
    )

    result = result.drop_duplicates(
        subset=[
            "_fixture_key"
        ],
        keep="first",
    )

    result = result.drop(
        columns=[
            "_fixture_key"
        ]
    )

    result = result.sort_values(
        [
            "date",
            "time",
            "league",
            "home_team",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    return result[
        required
    ]


# ============================================================
# LER CSV EXISTENTE
# ============================================================

def read_existing_output() -> pd.DataFrame:

    if not OUTPUT_PATH.exists():
        return empty_output()

    try:

        existing = pd.read_csv(
            OUTPUT_PATH
        )

    except Exception as error:

        print()
        print(
            "⚠ Não foi possível ler "
            "o upcoming_matches.csv anterior:"
        )

        print(
            error
        )

        return empty_output()

    if existing.empty:
        return empty_output()

    result = existing.copy()

    for column in OUTPUT_COLUMNS:

        if column not in result.columns:
            result[
                column
            ] = pd.NA

    # --------------------------------------------------------
    # COMPATIBILIDADE COM O FORMATO ANTIGO
    # --------------------------------------------------------
    #
    # O CSV antigo só tinha OPEN.
    #
    # Para preservar essas odds históricas:
    # - OPEN continua OPEN;
    # - CURRENT será substituída pela nova observação
    #   assim que a atualização ocorrer.
    # --------------------------------------------------------

    result[
        "odd_over25_open"
    ] = pd.to_numeric(
        result[
            "odd_over25_open"
        ],
        errors="coerce",
    )

    result[
        "odd_under25_open"
    ] = pd.to_numeric(
        result[
            "odd_under25_open"
        ],
        errors="coerce",
    )

    result[
        "odd_over25_current"
    ] = pd.to_numeric(
        result[
            "odd_over25_current"
        ],
        errors="coerce",
    )

    result[
        "odd_under25_current"
    ] = pd.to_numeric(
        result[
            "odd_under25_current"
        ],
        errors="coerce",
    )

    return result[
        OUTPUT_COLUMNS
    ].copy()


# ============================================================
# PRESERVAR OPENING ODDS
# ============================================================

def attach_opening_odds(
    observations: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:

    if observations.empty:
        return (
            empty_output(),
            0,
            0,
        )

    result = observations.copy()

    # Dois mapas:
    #
    # 1) EXACT:
    #    preserva a opening da MESMA representação/fonte quando
    #    essa linha já existia.
    #
    # 2) CANONICAL:
    #    funciona como fallback quando os nomes da fonte mudaram.
    existing_open_exact: dict[
        str,
        tuple[
            float,
            float,
        ],
    ] = {}

    existing_open_canonical: dict[
        str,
        tuple[
            float,
            float,
        ],
    ] = {}

    # ========================================================
    # CARREGAR OPENING ODDS ANTIGAS
    # ========================================================

    if not existing.empty:

        for _, row in (
            existing.iterrows()
        ):

            try:

                over_open = float(
                    row[
                        "odd_over25_open"
                    ]
                )

                under_open = float(
                    row[
                        "odd_under25_open"
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if (
                pd.isna(
                    over_open
                )
                or pd.isna(
                    under_open
                )
                or over_open <= 1.0
                or under_open <= 1.0
            ):

                continue

            exact_key = fixture_key(
                row[
                    "date"
                ],
                row[
                    "league"
                ],
                row[
                    "home_team"
                ],
                row[
                    "away_team"
                ],
            )

            canonical_key = (
                canonical_fixture_key(
                    row[
                        "date"
                    ],
                    row[
                        "league"
                    ],
                    row[
                        "home_team"
                    ],
                    row[
                        "away_team"
                    ],
                )
            )

            existing_open_exact[
                exact_key
            ] = (
                over_open,
                under_open,
            )

            # No fallback canónico, guarda a primeira opening
            # observada em vez de a substituir por um duplicado
            # posterior do mesmo fixture.
            existing_open_canonical.setdefault(
                canonical_key,
                (
                    over_open,
                    under_open,
                ),
            )

    # ========================================================
    # APLICAR
    # ========================================================

    over_open_values: list[
        float
    ] = []

    under_open_values: list[
        float
    ] = []

    preserved = 0
    created = 0

    for _, row in (
        result.iterrows()
    ):

        exact_key = fixture_key(
            row[
                "date"
            ],
            row[
                "league"
            ],
            row[
                "home_team"
            ],
            row[
                "away_team"
            ],
        )

        canonical_key = (
            canonical_fixture_key(
                row[
                    "date"
                ],
                row[
                    "league"
                ],
                row[
                    "home_team"
                ],
                row[
                    "away_team"
                ],
            )
        )

        # Prioridade:
        # 1) mesma linha/fonte anterior;
        # 2) mesmo fixture canónico com nomes diferentes.
        old_open = existing_open_exact.get(
            exact_key
        )

        if old_open is None:

            old_open = (
                existing_open_canonical.get(
                    canonical_key
                )
            )

        if old_open is not None:

            over_open_values.append(
                old_open[
                    0
                ]
            )

            under_open_values.append(
                old_open[
                    1
                ]
            )

            preserved += 1

        else:

            over_current = float(
                row[
                    "odd_over25_current"
                ]
            )

            under_current = float(
                row[
                    "odd_under25_current"
                ]
            )

            over_open_values.append(
                over_current
            )

            under_open_values.append(
                under_current
            )

            created += 1

    result[
        "odd_over25_open"
    ] = over_open_values

    result[
        "odd_under25_open"
    ] = under_open_values

    result = result[
        OUTPUT_COLUMNS
    ].copy()

    result = result.sort_values(
        [
            "date",
            "time",
            "league",
            "home_team",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    return (
        result,
        preserved,
        created,
    )


# ============================================================
# BACKUP DO CSV ANTIGO
# ============================================================

def backup_existing_output() -> Path | None:

    if not OUTPUT_PATH.exists():
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        DOWNLOADS_DIR
        / (
            "upcoming_before_update_"
            f"{timestamp}.csv"
        )
    )

    try:

        backup_path.write_bytes(
            OUTPUT_PATH.read_bytes()
        )

    except Exception:

        return None

    return backup_path


# ============================================================
# FONTE 1 — FOOTBALL-DATA
# ============================================================

def read_csv_content(
    content: bytes,
) -> pd.DataFrame:

    last_error: Exception | None = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):

        try:

            return pd.read_csv(
                io.BytesIO(
                    content
                ),
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError as error:

            last_error = error

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Não foi possível interpretar "
        "o CSV do Football-Data."
    )


def fetch_football_data(
    today: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:

    content, _ = http_get_bytes(
        FOOTBALL_DATA_URL
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    snapshot_path = (
        DOWNLOADS_DIR
        / (
            "football_data_"
            f"{timestamp}.csv"
        )
    )

    snapshot_path.write_bytes(
        content
    )

    raw = read_csv_content(
        content
    )

    required_columns = [
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in raw.columns
    ]

    if missing_columns:

        raise ValueError(
            "O CSV do Football-Data "
            "não contém: "
            + ", ".join(
                missing_columns
            )
        )

    raw = raw.copy()

    raw[
        "_parsed_date"
    ] = parse_dates(
        raw[
            "Date"
        ]
    )

    valid_dates = raw[
        raw[
            "_parsed_date"
        ].notna()
    ]

    latest_source_date = (
        None
        if valid_dates.empty
        else
        valid_dates[
            "_parsed_date"
        ]
        .max()
        .date()
    )

    filtered = raw[
        raw[
            "Div"
        ].notna()
        &
        raw[
            "_parsed_date"
        ].notna()
        &
        (
            raw[
                "_parsed_date"
            ]
            >= today
        )
    ].copy()

    division_codes = (
        filtered[
            "Div"
        ]
        .astype("string")
        .str.strip()
    )

    filtered[
        "league"
    ] = (
        division_codes
        .map(
            DIVISION_LEAGUE_MAP
        )
        .fillna(
            division_codes
        )
    )

    filtered[
        "home_team"
    ] = (
        filtered[
            "HomeTeam"
        ]
        .astype("string")
        .str.strip()
    )

    filtered[
        "away_team"
    ] = (
        filtered[
            "AwayTeam"
        ]
        .astype("string")
        .str.strip()
    )

    if (
        "Time"
        in filtered.columns
    ):

        filtered[
            "time"
        ] = (
            filtered[
                "Time"
            ]
            .astype("string")
            .str.strip()
        )

    else:

        filtered[
            "time"
        ] = pd.NA

    filtered[
        "date"
    ] = (
        filtered[
            "_parsed_date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # Estas são as odds observadas AGORA.
    # Já não são chamadas OPEN aqui.
    # --------------------------------------------------------

    filtered[
        "odd_over25_current"
    ] = first_numeric_column(
        filtered,
        [
            "Avg>2.5",
            "B365>2.5",
            "P>2.5",
            "Max>2.5",
        ],
    )

    filtered[
        "odd_under25_current"
    ] = first_numeric_column(
        filtered,
        [
            "Avg<2.5",
            "B365<2.5",
            "P<2.5",
            "Max<2.5",
        ],
    )

    normalized = (
        normalize_observations(
            filtered[
                BASE_COLUMNS
                + CURRENT_COLUMNS
            ]
        )
    )

    diagnostics = {
        "source":
            "Football-Data",

        "downloaded_rows":
            len(
                raw
            ),

        "future_rows_before_odds":
            len(
                filtered
            ),

        "ready_rows":
            len(
                normalized
            ),

        "latest_source_date":
            latest_source_date,

        "snapshot_path":
            snapshot_path,
    }

    return (
        normalized,
        diagnostics,
    )


# ============================================================
# FONTE 2 — THE ODDS API
# ============================================================

def get_odds_api_key() -> str | None:

    return clean_text_value(
        os.getenv(
            "ODDS_API_KEY"
        )
    )


def get_forced_sport_keys() -> list[str]:

    raw_value = clean_text_value(
        os.getenv(
            "ODDS_API_SPORT_KEYS"
        )
    )

    if raw_value is None:
        return []

    return [
        item.strip()
        for item
        in raw_value.split(",")
        if item.strip()
    ]


def get_requested_sport_keys() -> list[str]:
    """
    Devolve as ligas que o FootballAI deve consultar.

    ODDS_API_SPORT_KEYS continua a poder substituir a lista por defeito,
    mas sem configuração externa usamos sempre a lista fixa acima.
    """

    forced = get_forced_sport_keys()

    if forced:
        return list(dict.fromkeys(forced))

    return DEFAULT_ODDS_API_SPORT_KEYS.copy()


def fetch_active_soccer_sports(
    api_key: str,
) -> list[
    dict[
        str,
        object,
    ]
]:

    query = urlencode(
        {
            "apiKey":
                api_key,
        }
    )

    payload, _ = http_get_json(
        f"{ODDS_API_BASE_URL}"
        f"/sports/?{query}"
    )

    if not isinstance(
        payload,
        list,
    ):

        raise RuntimeError(
            "A lista de desportos da "
            "The Odds API tem formato inesperado."
        )

    soccer_sports: list[
        dict[
            str,
            object,
        ]
    ] = []

    for item in payload:

        if not isinstance(
            item,
            dict,
        ):
            continue

        group = str(
            item.get(
                "group",
                "",
            )
        ).strip().lower()

        active = bool(
            item.get(
                "active",
                False,
            )
        )

        has_outrights = bool(
            item.get(
                "has_outrights",
                False,
            )
        )

        key = clean_text_value(
            item.get(
                "key"
            )
        )

        if (
            group == "soccer"
            and active
            and not has_outrights
            and key
        ):

            soccer_sports.append(
                item
            )

    return soccer_sports


def select_sport_keys(
    soccer_sports: list[
        dict[
            str,
            object,
        ]
    ],
) -> tuple[
    list[str],
    list[str],
]:
    """
    Seleciona apenas as ligas fixas que estão ativas neste momento.

    Retorna:
      - sport_keys a consultar;
      - ligas pedidas mas atualmente inativas/indisponíveis.
    """

    requested = get_requested_sport_keys()

    active_keys = {
        str(item.get("key"))
        for item in soccer_sports
        if item.get("key")
    }

    selected = [
        key
        for key in requested
        if key in active_keys
    ]

    inactive = [
        key
        for key in requested
        if key not in active_keys
    ]

    return (
        selected,
        inactive,
    )

def extract_average_totals_25(
    event: dict[
        str,
        object,
    ],
) -> tuple[
    float | None,
    float | None,
]:

    over_prices: list[
        float
    ] = []

    under_prices: list[
        float
    ] = []

    bookmakers = event.get(
        "bookmakers",
        [],
    )

    if not isinstance(
        bookmakers,
        list,
    ):

        return (
            None,
            None,
        )

    for bookmaker in bookmakers:

        if not isinstance(
            bookmaker,
            dict,
        ):
            continue

        markets = bookmaker.get(
            "markets",
            [],
        )

        if not isinstance(
            markets,
            list,
        ):
            continue

        for market in markets:

            if not isinstance(
                market,
                dict,
            ):
                continue

            if (
                market.get(
                    "key"
                )
                != "totals"
            ):
                continue

            outcomes = market.get(
                "outcomes",
                [],
            )

            if not isinstance(
                outcomes,
                list,
            ):
                continue

            for outcome in outcomes:

                if not isinstance(
                    outcome,
                    dict,
                ):
                    continue

                name = str(
                    outcome.get(
                        "name",
                        "",
                    )
                ).strip().lower()

                try:

                    point = float(
                        outcome.get(
                            "point"
                        )
                    )

                    price = float(
                        outcome.get(
                            "price"
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

                if (
                    abs(
                        point - 2.5
                    )
                    > 1e-9
                    or price <= 1.0
                ):

                    continue

                if name == "over":

                    over_prices.append(
                        price
                    )

                elif name == "under":

                    under_prices.append(
                        price
                    )

    if (
        not over_prices
        or not under_prices
    ):

        return (
            None,
            None,
        )

    return (
        float(
            np.mean(
                over_prices
            )
        ),
        float(
            np.mean(
                under_prices
            )
        ),
    )


def parse_iso_datetime(
    value: object,
) -> datetime | None:

    text = clean_text_value(
        value
    )

    if text is None:
        return None

    try:

        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:

        return None

    if parsed.tzinfo is None:

        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed


def fetch_odds_api(
    today: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:

    api_key = (
        get_odds_api_key()
    )

    if not api_key:

        return (
            empty_observations(),
            {
                "source":
                    "The Odds API",

                "enabled":
                    False,

                "reason":
                    "ODDS_API_KEY não configurada",

                "requested_sports":
                    len(get_requested_sport_keys()),

                "selected_sports":
                    0,

                "successful_sports":
                    0,

                "failed_sports":
                    0,

                "quota_exhausted":
                    False,

                "complete":
                    False,

                "ready_rows":
                    0,
            },
        )

    soccer_sports = (
        fetch_active_soccer_sports(
            api_key
        )
    )

    requested_sport_keys = (
        get_requested_sport_keys()
    )

    (
        sport_keys,
        inactive_sport_keys,
    ) = select_sport_keys(
        soccer_sports
    )

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    requests_used = 0

    requests_remaining: (
        str | None
    ) = None

    successful_sports = 0

    failed_sport_keys: list[str] = []

    errors: list[str] = []

    quota_exhausted = False

    for sport_key in sport_keys:

        query = urlencode(
            {
                "apiKey":
                    api_key,

                "regions":
                    "eu",

                "markets":
                    "totals",

                "oddsFormat":
                    "decimal",

                "dateFormat":
                    "iso",
            }
        )

        url = (
            f"{ODDS_API_BASE_URL}"
            f"/sports/"
            f"{sport_key}"
            f"/odds/?{query}"
        )

        try:

            payload, headers = (
                http_get_json(
                    url
                )
            )

        except Exception as error:

            error_text = str(error)

            errors.append(
                f"{sport_key}: "
                f"{error_text}"
            )

            failed_sport_keys.append(
                sport_key
            )

            if any(
                marker in error_text
                for marker in QUOTA_ERROR_MARKERS
            ):

                quota_exhausted = True

                # Não vale a pena repetir a mesma falha
                # nas restantes ligas quando a quota acabou.
                break

            continue

        requests_used += 1
        successful_sports += 1

        requests_remaining = (
            headers.get(
                "x-requests-remaining",
                requests_remaining,
            )
        )

        if not isinstance(
            payload,
            list,
        ):

            failed_sport_keys.append(
                sport_key
            )

            successful_sports -= 1

            errors.append(
                f"{sport_key}: "
                "resposta inesperada"
            )

            continue

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        snapshot_path = (
            DOWNLOADS_DIR
            / (
                "odds_api_"
                f"{sport_key}_"
                f"{timestamp}.json"
            )
        )

        snapshot_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        for event in payload:

            if not isinstance(
                event,
                dict,
            ):
                continue

            kickoff = (
                parse_iso_datetime(
                    event.get(
                        "commence_time"
                    )
                )
            )

            if kickoff is None:
                continue

            local_kickoff = (
                kickoff.astimezone()
            )

            event_date = pd.Timestamp(
                local_kickoff.date()
            )

            if event_date < today:
                continue

            (
                odd_over,
                odd_under,
            ) = extract_average_totals_25(
                event
            )

            if (
                odd_over is None
                or odd_under is None
            ):
                continue

            rows.append(
                {
                    "date":
                        local_kickoff.strftime(
                            "%Y-%m-%d"
                        ),

                    "time":
                        local_kickoff.strftime(
                            "%H:%M"
                        ),

                    "league":
                        clean_text_value(
                            event.get(
                                "sport_title"
                            )
                        )
                        or sport_key,

                    "home_team":
                        clean_text_value(
                            event.get(
                                "home_team"
                            )
                        ),

                    "away_team":
                        clean_text_value(
                            event.get(
                                "away_team"
                            )
                        ),

                    "odd_over25_current":
                        odd_over,

                    "odd_under25_current":
                        odd_under,
                }
            )

    normalized = (
        normalize_observations(
            pd.DataFrame(
                rows
            )
        )
    )

    complete = (
        bool(sport_keys)
        and not quota_exhausted
        and not failed_sport_keys
    )

    diagnostics = {
        "source":
            "The Odds API",

        "enabled":
            True,

        "active_soccer_sports":
            len(soccer_sports),

        "requested_sports":
            len(requested_sport_keys),

        "selected_sports":
            len(sport_keys),

        "inactive_requested_sports":
            len(inactive_sport_keys),

        "successful_sports":
            successful_sports,

        "failed_sports":
            len(failed_sport_keys),

        "requests_used":
            requests_used,

        "requests_remaining":
            requests_remaining,

        "quota_exhausted":
            quota_exhausted,

        "complete":
            complete,

        "ready_rows":
            len(normalized),

        "errors":
            errors,
    }

    return (
        normalized,
        diagnostics,
    )


# ============================================================
# DIAGNÓSTICOS
# ============================================================

def print_source_diagnostics(
    diagnostics: dict[
        str,
        object,
    ],
) -> None:

    source = diagnostics.get(
        "source",
        "Fonte desconhecida",
    )

    print(
        f"\n--- {source} ---"
    )

    for key, value in (
        diagnostics.items()
    ):

        if key in {
            "source",
            "errors",
        }:
            continue

        print(
            f"{key.replace('_', ' ').capitalize():28}: "
            f"{value}"
        )

    errors = diagnostics.get(
        "errors",
        [],
    )

    if (
        isinstance(
            errors,
            list,
        )
        and errors
    ):

        print(
            "Erros:"
        )

        for error in errors[
            :10
        ]:

            print(
                f"  - {error}"
            )


# ============================================================
# EXECUÇÃO
# ============================================================

def main() -> None:

    load_simple_env(
        ENV_PATH
    )

    UPCOMING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOWNLOADS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    today = (
        pd.Timestamp.now()
        .normalize()
    )

    print()
    print(
        "=========================================="
    )
    print(
        " FOOTBALLAI — ATUALIZAR JOGOS FUTUROS"
    )
    print(
        "=========================================="
    )

    print(
        f"\nData local: "
        f"{today.date()}"
    )

    # ========================================================
    # LER CSV ANTIGO ANTES DE QUALQUER ALTERAÇÃO
    # ========================================================

    existing = (
        read_existing_output()
    )

    print()
    print(
        f"Jogos no CSV anterior: "
        f"{len(existing)}"
    )

    # ========================================================
    # FOOTBALL-DATA
    # ========================================================

    football_data = (
        empty_observations()
    )

    try:

        (
            football_data,
            football_data_info,
        ) = fetch_football_data(
            today
        )

        print_source_diagnostics(
            football_data_info
        )

    except Exception as error:

        print()
        print(
            "--- Football-Data ---"
        )
        print(
            f"Erro: {error}"
        )

    # ========================================================
    # THE ODDS API
    # ========================================================

    odds_api = (
        empty_observations()
    )

    odds_api_info: dict[
        str,
        object,
    ] = {
        "source": "The Odds API",
        "enabled": False,
        "complete": False,
        "ready_rows": 0,
        "errors": [],
    }

    try:

        (
            odds_api,
            odds_api_info,
        ) = fetch_odds_api(
            today
        )

        print_source_diagnostics(
            odds_api_info
        )

    except Exception as error:

        odds_api_info = {
            "source": "The Odds API",
            "enabled": True,
            "complete": False,
            "ready_rows": 0,
            "errors": [str(error)],
        }

        print()
        print(
            "--- The Odds API ---"
        )
        print(
            f"Erro: {error}"
        )

    # ========================================================
    # FAIL-SAFE DA ODDS API
    # ========================================================
    #
    # O capture_closing_odds.py usa a data de modificação do CSV
    # para decidir se as odds CURRENT são recentes.
    #
    # Por isso, se a Odds API falhar total ou parcialmente, NÃO
    # podemos regravar o CSV apenas com Football-Data ou com uma
    # cobertura incompleta. Isso faria odds antigas parecerem novas.
    # ========================================================

    odds_api_complete = bool(
        odds_api_info.get(
            "complete",
            False,
        )
    )

    if not odds_api_complete:

        print()
        print(
            "=========================================="
        )
        print(
            " ATUALIZAÇÃO CANCELADA COM SEGURANÇA"
        )
        print(
            "=========================================="
        )

        print()
        print(
            "⚠ A The Odds API não atualizou todas "
            "as ligas selecionadas."
        )

        if bool(
            odds_api_info.get(
                "quota_exhausted",
                False,
            )
        ):

            print(
                "⚠ A quota da The Odds API está esgotada."
            )

        print()
        print(
            "Por segurança, o upcoming_matches.csv "
            "NÃO será reescrito."
        )

        print(
            "As OPEN e CURRENT anteriores ficam preservadas "
            "e a data de modificação continua antiga."
        )

        print(
            "Assim capture_closing_odds.py não confundirá "
            "odds antigas com closing odds recentes."
        )

        if OUTPUT_PATH.exists():

            print()
            print(
                f"CSV preservado: "
                f"{OUTPUT_PATH.relative_to(ROOT_DIR)}"
            )

            print(
                f"Jogos preservados: "
                f"{len(existing)}"
            )

        else:

            empty_output().to_csv(
                OUTPUT_PATH,
                index=False,
            )

            print()
            print(
                "Não existia CSV anterior; "
                "foi criado apenas um ficheiro vazio."
            )

        return

    # ========================================================
    # JUNTAR FONTES
    # ========================================================
    #
    # A Odds API vem primeiro porque representa a consulta
    # explícita às odds atuais. Football-Data entra como fallback
    # apenas quando a atualização da API ficou completa.
    # ========================================================

    non_empty_frames: list[
        pd.DataFrame
    ] = []

    if not odds_api.empty:
        non_empty_frames.append(
            odds_api
        )

    if not football_data.empty:
        non_empty_frames.append(
            football_data
        )

    canonical_duplicates_removed = 0

    if non_empty_frames:

        raw_observations = pd.concat(
            non_empty_frames,
            ignore_index=True,
        )

        observations = (
            normalize_observations(
                raw_observations
            )
        )

        canonical_duplicates_removed = (
            len(
                raw_observations
            )
            - len(
                observations
            )
        )

    else:

        observations = (
            empty_observations()
        )

    # ========================================================
    # NÃO REESCREVER COM DADOS VAZIOS
    # ========================================================

    if observations.empty:

        print()
        print(
            "=========================================="
        )
        print(
            " RESULTADO FINAL"
        )
        print(
            "=========================================="
        )

        print()
        print(
            "⚠ As fontes responderam, mas não devolveram "
            "jogos com odds atuais válidas."
        )

        if OUTPUT_PATH.exists():

            print(
                "O upcoming_matches.csv anterior foi PRESERVADO."
            )

            print(
                "A data de modificação continua antiga."
            )

        else:

            empty_output().to_csv(
                OUTPUT_PATH,
                index=False,
            )

        return

    # ========================================================
    # PRESERVAR OPENING ODDS
    # ========================================================

    (
        combined,
        preserved_openings,
        new_openings,
    ) = attach_opening_odds(
        observations,
        existing,
    )

    # ========================================================
    # BACKUP
    # ========================================================

    backup_path = (
        backup_existing_output()
    )

    # ========================================================
    # GUARDAR NOVO CSV
    # ========================================================

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    print()
    print(
        "=========================================="
    )
    print(
        " RESULTADO FINAL"
    )
    print(
        "=========================================="
    )

    print()
    print(
        f"Jogos prontos: "
        f"{len(combined)}"
    )
    print(
        f"Duplicados canónicos removidos: "
        f"{canonical_duplicates_removed}"
    )
    print(
        f"Opening odds preservadas: "
        f"{preserved_openings}"
    )
    print(
        f"Novas opening odds: "
        f"{new_openings}"
    )
    print(
        f"Ficheiro: "
        f"{OUTPUT_PATH.relative_to(ROOT_DIR)}"
    )

    if backup_path is not None:

        print(
            f"Backup anterior: "
            f"{backup_path.relative_to(ROOT_DIR)}"
        )

    print()
    print(
        "Próximos jogos encontrados:"
    )
    print()

    print(
        combined.to_string(
            index=False,
            formatters={
                "odd_over25_open":
                    "{:.2f}".format,

                "odd_under25_open":
                    "{:.2f}".format,

                "odd_over25_current":
                    "{:.2f}".format,

                "odd_under25_current":
                    "{:.2f}".format,
            },
        )
    )

    print()
    print(
        "=========================================="
    )
    print(
        " LÓGICA DAS ODDS"
    )
    print(
        "=========================================="
    )

    print()
    print(
        "OPEN:"
    )
    print(
        "  Fica congelada desde a primeira "
        "observação do jogo."
    )

    print()
    print(
        "CURRENT:"
    )
    print(
        "  É substituída pela odd mais recente "
        "apenas quando a atualização é segura."
    )

    print()
    print(
        "CLOSE:"
    )
    print(
        "  Será capturada por "
        "capture_closing_odds.py "
        "perto do início do jogo."
    )

    print()
    print(
        "✅ Próximo comando:"
    )
    print(
        "python ml/paper_trade.py"
    )


if __name__ == "__main__":
    main()