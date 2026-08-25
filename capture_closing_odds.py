from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.paper_trade import normalize_league, normalize_team


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATABASE_PATH = (
    ROOT_DIR
    / "database"
    / "football.db"
)

UPCOMING_PATH = (
    ROOT_DIR
    / "data"
    / "upcoming"
    / "upcoming_matches.csv"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# O CSV tem de ter sido atualizado recentemente.
#
# Isto é importante porque odd_over25_current /
# odd_under25_current só podem ser consideradas "atuais"
# se o ficheiro também for recente.
MAX_CSV_AGE_MINUTES = 20


# Só consideramos uma odd como possível "closing odd"
# dentro desta janela antes do início do jogo.
CAPTURE_WINDOW_MINUTES = 90


# NUNCA usamos estas colunas como closing odds.
OPEN_OVER_COLUMN = "odd_over25_open"
OPEN_UNDER_COLUMN = "odd_under25_open"


# Colunas aceitáveis para a odd atual.
#
# O script usa a primeira combinação que existir.
CURRENT_ODD_COLUMN_PAIRS = [
    (
        "odd_over25_current",
        "odd_under25_current",
    ),
    (
        "odd_over25_live",
        "odd_under25_live",
    ),
    (
        "odd_over25_latest",
        "odd_under25_latest",
    ),
    (
        "odd_over25_close",
        "odd_under25_close",
    ),
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def norm(
    value: object,
) -> str:
    """
    Normalização simples apenas para comparações de texto cru.
    A normalização real de ligas/equipas é feita pelas funções
    importadas de ml.paper_trade.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .casefold()
    )


def normalize_date_value(
    value: object,
) -> str:
    """
    Converte uma data para YYYY-MM-DD.
    """

    if value is None:
        return ""

    text = str(value).strip()

    if (
        not text
        or text.casefold() in {
            "nan",
            "none",
            "<na>",
        }
    ):
        return ""

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=False,
    )

    if pd.isna(parsed):
        parsed = pd.to_datetime(
            text,
            errors="coerce",
            dayfirst=True,
        )

    if pd.isna(parsed):
        return ""

    return pd.Timestamp(
        parsed
    ).strftime(
        "%Y-%m-%d"
    )


def find_table(
    connection: sqlite3.Connection,
) -> str | None:
    """
    Procura a tabela de paper trading.
    """

    tables = {
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
    }

    for candidate in (
        "paper_trades",
        "paper_trade",
        "paper_trading",
        "historico_paper_trade",
    ):
        if candidate in tables:
            return candidate

    return None


def find_current_odd_columns(
    dataframe: pd.DataFrame,
) -> tuple[str, str] | None:
    """
    Encontra a primeira combinação disponível de odds atuais.
    """

    for (
        over_column,
        under_column,
    ) in CURRENT_ODD_COLUMN_PAIRS:

        if (
            over_column in dataframe.columns
            and under_column in dataframe.columns
        ):
            return (
                over_column,
                under_column,
            )

    return None


def parse_fixture_datetime(
    item: pd.Series,
) -> pd.Timestamp | None:
    """
    Constrói a data/hora local do jogo a partir do CSV.
    """

    if "date" not in item.index:
        return None

    date_value = str(
        item["date"]
    ).strip()

    if (
        not date_value
        or date_value.casefold() == "nan"
    ):
        return None

    if "time" not in item.index:
        return None

    time_value = str(
        item["time"]
    ).strip()

    if (
        not time_value
        or time_value.casefold() == "nan"
    ):
        return None

    try:
        fixture_datetime = pd.Timestamp(
            f"{date_value} {time_value}"
        )
    except Exception:
        return None

    return fixture_datetime


def safe_float(
    value: object,
) -> float | None:
    """
    Valida uma odd decimal.
    """

    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if pd.isna(
        result
    ):
        return None

    if result <= 1.0:
        return None

    return result


def prepare_normalized_upcoming(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria colunas auxiliares normalizadas para matching.

    Usa exatamente normalize_league() e normalize_team()
    do ml.paper_trade.
    """

    result = dataframe.copy()

    result[
        "_match_date"
    ] = result[
        "date"
    ].apply(
        normalize_date_value
    )

    result[
        "_match_league"
    ] = result[
        "league"
    ].apply(
        normalize_league
    )

    result[
        "_match_home"
    ] = result.apply(
        lambda row: normalize_team(
            row[
                "_match_league"
            ],
            row[
                "home_team"
            ],
        ),
        axis=1,
    )

    result[
        "_match_away"
    ] = result.apply(
        lambda row: normalize_team(
            row[
                "_match_league"
            ],
            row[
                "away_team"
            ],
        ),
        axis=1,
    )

    return result


def choose_unique_match(
    matched: pd.DataFrame,
    trade_league: object,
    trade_home: object,
    trade_away: object,
) -> pd.Series | None:
    """
    Escolhe um único jogo.

    Se houver vários jogos que normalizam para o mesmo fixture,
    tenta primeiro um único match cru/exato. Se continuar
    ambíguo, não escolhe arbitrariamente.
    """

    if len(
        matched
    ) == 1:
        return matched.iloc[
            0
        ]

    exact = matched[
        (
            matched[
                "league"
            ]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(
                norm(
                    trade_league
                )
            )
        )
        &
        (
            matched[
                "home_team"
            ]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(
                norm(
                    trade_home
                )
            )
        )
        &
        (
            matched[
                "away_team"
            ]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(
                norm(
                    trade_away
                )
            )
        )
    ]

    if len(
        exact
    ) == 1:
        return exact.iloc[
            0
        ]

    return None


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=========================================="
    )
    print(
        " FOOTBALLAI — CAPTURA DE CLOSING ODDS"
    )
    print(
        "=========================================="
    )
    print()

    # ========================================================
    # VERIFICAR CSV
    # ========================================================

    if not UPCOMING_PATH.exists():

        print(
            "❌ upcoming_matches.csv não existe."
        )

        return

    # ========================================================
    # VERIFICAR IDADE DO CSV
    # ========================================================

    modified_timestamp = (
        UPCOMING_PATH
        .stat()
        .st_mtime
    )

    modified_at = datetime.fromtimestamp(
        modified_timestamp
    )

    now = datetime.now()

    csv_age_minutes = (
        now - modified_at
    ).total_seconds() / 60.0

    print(
        "Última atualização do CSV:"
    )

    print(
        modified_at.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print(
        f"Idade do CSV: "
        f"{csv_age_minutes:.1f} minutos"
    )

    if csv_age_minutes < -5:

        print()
        print(
            "❌ A data de modificação do CSV "
            "está no futuro."
        )

        print(
            "Verifica o relógio do computador."
        )

        return

    if (
        csv_age_minutes
        > MAX_CSV_AGE_MINUTES
    ):

        print()
        print(
            "❌ O upcoming_matches.csv "
            "é demasiado antigo."
        )

        print(
            f"Máximo permitido: "
            f"{MAX_CSV_AGE_MINUTES} minutos."
        )

        print()
        print(
            "Atualiza primeiro as odds "
            "e volta a executar."
        )

        print()
        print(
            "Nenhuma closing odd foi gravada."
        )

        return

    # ========================================================
    # CARREGAR CSV
    # ========================================================

    upcoming = pd.read_csv(
        UPCOMING_PATH
    )

    if upcoming.empty:

        print()
        print(
            "❌ upcoming_matches.csv está vazio."
        )

        return

    # ========================================================
    # VERIFICAR COLUNAS BASE
    # ========================================================

    required_columns = [
        "date",
        "time",
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
            "❌ Faltam colunas no CSV:"
        )

        for column in missing_columns:

            print(
                f"   - {column}"
            )

        return

    # ========================================================
    # PROCURAR COLUNAS DE ODDS ATUAIS
    # ========================================================

    current_columns = (
        find_current_odd_columns(
            upcoming
        )
    )

    if current_columns is None:

        print()
        print(
            "❌ NÃO existem colunas de odds atuais "
            "no upcoming_matches.csv."
        )

        print()
        print(
            "Por segurança, este script NÃO vai usar:"
        )

        print(
            f"   {OPEN_OVER_COLUMN}"
        )

        print(
            f"   {OPEN_UNDER_COLUMN}"
        )

        print()
        print(
            "Essas são as odds de ABERTURA "
            "e não podem ser copiadas para CLOSE."
        )

        print()
        print(
            "O scraper que atualiza as odds deverá criar "
            "por exemplo:"
        )

        print(
            "   odd_over25_current"
        )

        print(
            "   odd_under25_current"
        )

        print()
        print(
            "Nenhuma closing odd foi gravada."
        )

        return

    (
        current_over_column,
        current_under_column,
    ) = current_columns

    print()
    print(
        "Colunas usadas como odd atual:"
    )

    print(
        f"Over : {current_over_column}"
    )

    print(
        f"Under: {current_under_column}"
    )

    print()
    print(
        "Matching:"
    )

    print(
        "normalize_league + normalize_team "
        "de ml.paper_trade"
    )

    # ========================================================
    # NORMALIZAR CSV UMA VEZ
    # ========================================================

    try:

        upcoming = (
            prepare_normalized_upcoming(
                upcoming
            )
        )

    except Exception as error:

        print()
        print(
            "❌ Falha ao normalizar "
            "ligas/equipas do CSV."
        )

        print(
            f"Erro: {error}"
        )

        print()
        print(
            "Nenhuma closing odd foi gravada."
        )

        return

    # ========================================================
    # ABRIR DATABASE
    # ========================================================

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    table = find_table(
        connection
    )

    if table is None:

        connection.close()

        print()
        print(
            "❌ Tabela de paper trading "
            "não encontrada."
        )

        return

    # ========================================================
    # COLUNAS DA TABELA
    # ========================================================

    columns = {
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }

    for column in (
        "odd_over25_close",
        "odd_under25_close",
    ):

        if column not in columns:

            connection.execute(
                f"""
                ALTER TABLE {table}
                ADD COLUMN {column} REAL
                """
            )

            columns.add(
                column
            )

    fixture_date_column = (
        "fixture_date"
        if "fixture_date" in columns
        else
        "date"
        if "date" in columns
        else
        None
    )

    if fixture_date_column is None:

        connection.close()

        raise RuntimeError(
            "Não existe fixture_date/date "
            "na tabela."
        )

    # ========================================================
    # JOGOS AINDA ABERTOS
    # ========================================================

    open_rows = connection.execute(
        f"""
        SELECT
            id,
            {fixture_date_column},
            league,
            home_team,
            away_team
        FROM {table}
        WHERE COALESCE(
            status,
            'OPEN'
        ) = 'OPEN'
        """
    ).fetchall()

    print()
    print(
        f"Jogos OPEN no Paper Trade: "
        f"{len(open_rows)}"
    )

    updated = 0
    skipped_not_found = 0
    skipped_time = 0
    skipped_past_date = 0
    skipped_odds = 0
    skipped_ambiguous = 0
    normalized_matches = 0

    today_string = pd.Timestamp(
        now
    ).strftime(
        "%Y-%m-%d"
    )

    # ========================================================
    # PROCESSAR
    # ========================================================

    for row in open_rows:

        # ----------------------------------------------------
        # NORMALIZAR O JOGO DO PAPER TRADE
        # ----------------------------------------------------

        trade_date = normalize_date_value(
            row[
                fixture_date_column
            ]
        )

        trade_league = normalize_league(
            row[
                "league"
            ]
        )

        trade_home = normalize_team(
            trade_league,
            row[
                "home_team"
            ],
        )

        trade_away = normalize_team(
            trade_league,
            row[
                "away_team"
            ],
        )

        if (
            not trade_date
            or trade_league is None
            or trade_home is None
            or trade_away is None
        ):

            skipped_not_found += 1
            continue

        # ----------------------------------------------------
        # JOGOS DE DIAS ANTERIORES JÁ NÃO SÃO CANDIDATOS
        # ----------------------------------------------------

        if trade_date < today_string:

            skipped_past_date += 1
            continue

        # ----------------------------------------------------
        # MATCH NORMALIZADO:
        # DATA + LIGA + CASA + FORA
        # ----------------------------------------------------

        matched = upcoming[
            (
                upcoming[
                    "_match_date"
                ].eq(
                    trade_date
                )
            )
            &
            (
                upcoming[
                    "_match_league"
                ].eq(
                    trade_league
                )
            )
            &
            (
                upcoming[
                    "_match_home"
                ].eq(
                    trade_home
                )
            )
            &
            (
                upcoming[
                    "_match_away"
                ].eq(
                    trade_away
                )
            )
        ]

        if matched.empty:

            skipped_not_found += 1
            continue

        item = choose_unique_match(
            matched,
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

        if item is None:

            skipped_ambiguous += 1

            print(
                f"⚠ AMBÍGUO: "
                f"{row['home_team']} "
                f"vs "
                f"{row['away_team']} "
                f"| {len(matched)} linhas normalizadas"
            )

            continue

        normalized_matches += 1

        # ----------------------------------------------------
        # DATA / HORA DO JOGO
        # ----------------------------------------------------

        fixture_datetime = (
            parse_fixture_datetime(
                item
            )
        )

        if fixture_datetime is None:

            skipped_time += 1
            continue

        now_timestamp = pd.Timestamp(
            datetime.now()
        )

        minutes_to_start = (
            fixture_datetime
            - now_timestamp
        ).total_seconds() / 60.0

        # ----------------------------------------------------
        # NÃO CAPTURAR DEPOIS DO INÍCIO
        # ----------------------------------------------------

        if minutes_to_start < 0:

            skipped_time += 1
            continue

        # ----------------------------------------------------
        # NÃO CAPTURAR LONGE DEMAIS DO JOGO
        # ----------------------------------------------------

        if (
            minutes_to_start
            > CAPTURE_WINDOW_MINUTES
        ):

            skipped_time += 1
            continue

        # ----------------------------------------------------
        # ODDS ATUAIS
        # ----------------------------------------------------

        over = safe_float(
            item[
                current_over_column
            ]
        )

        under = safe_float(
            item[
                current_under_column
            ]
        )

        if (
            over is None
            or under is None
        ):

            skipped_odds += 1
            continue

        # ----------------------------------------------------
        # GRAVAR ÚLTIMA ODD OBSERVADA
        # ----------------------------------------------------

        connection.execute(
            f"""
            UPDATE {table}
            SET
                odd_over25_close = ?,
                odd_under25_close = ?
            WHERE id = ?
            """,
            (
                over,
                under,
                row[
                    "id"
                ],
            ),
        )

        updated += 1

        print(
            f"✅ "
            f"{row['home_team']} "
            f"vs "
            f"{row['away_team']} "
            f"| "
            f"{minutes_to_start:.0f} min "
            f"| Over {over:.2f} "
            f"| Under {under:.2f}"
        )

    # ========================================================
    # GUARDAR
    # ========================================================

    connection.commit()
    connection.close()

    # ========================================================
    # RESUMO
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        " RESUMO"
    )

    print(
        "=========================================="
    )

    print(
        f"Matches normalizados      : "
        f"{normalized_matches}"
    )

    print(
        f"Closing odds atualizadas  : "
        f"{updated}"
    )

    print(
        f"Jogo não encontrado       : "
        f"{skipped_not_found}"
    )

    print(
        f"Jogo ambíguo              : "
        f"{skipped_ambiguous}"
    )

    print(
        f"Data já passada           : "
        f"{skipped_past_date}"
    )

    print(
        f"Fora da janela de fecho   : "
        f"{skipped_time}"
    )

    print(
        f"Odd inválida               : "
        f"{skipped_odds}"
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "odd_over25_open e odd_under25_open "
        "nunca são usadas como closing odds."
    )

    print()
    print(
        "Para obter uma closing odd válida, "
        "o scraper tem de atualizar odds próprias "
        "pouco antes do início do jogo."
    )


if __name__ == "__main__":
    main()