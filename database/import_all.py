from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

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

HISTORICAL_DIR = (
    ROOT_DIR
    / "data"
    / "historical"
)

BACKUP_DIR = (
    ROOT_DIR
    / "database"
    / "backups"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DIVISION_LEAGUE_MAP = {
    # Divisoes principais ja existentes.
    # Os nomes sao mantidos para nao duplicar jogos antigos
    # quando o importador for executado novamente.
    "E0": "England",
    "SP1": "Spain",
    "I1": "Italy",
    "D1": "Germany",
    "F1": "France",
    "N1": "Netherlands",
    "P1": "Portugal",

    # Novas divisoes.
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "SC0": "Scotland Premiership",
    "D2": "Germany 2 Bundesliga",
    "SP2": "Spain Segunda Division",
    "I2": "Italy Serie B",
    "F2": "France Ligue 2",
    "B1": "Belgium First Division A",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}


# ============================================================
# COLUNAS DA BASE DE DADOS
# ============================================================

DATABASE_COLUMNS = {
    "date": "TEXT",
    "season": "TEXT",
    "league": "TEXT",
    "division": "TEXT",

    "home_team": "TEXT",
    "away_team": "TEXT",

    "home_goals": "INTEGER",
    "away_goals": "INTEGER",

    "result": "TEXT",
    "over25": "INTEGER",
    "btts": "INTEGER",

    "odd_home": "REAL",
    "odd_draw": "REAL",
    "odd_away": "REAL",

    "odd_over25": "REAL",
    "odd_under25": "REAL",

    "referee": "TEXT",

    "home_shots": "INTEGER",
    "away_shots": "INTEGER",

    "home_shots_target": "INTEGER",
    "away_shots_target": "INTEGER",

    "home_corners": "INTEGER",
    "away_corners": "INTEGER",

    "home_yellow": "INTEGER",
    "away_yellow": "INTEGER",

    "home_red": "INTEGER",
    "away_red": "INTEGER",

    "odd_over25_open": "REAL",
    "odd_under25_open": "REAL",

    "odd_over25_close": "REAL",
    "odd_under25_close": "REAL",
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
    Converte odds ou outros valores decimais.
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().replace(",", ".")

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not pd.notna(number):
        return None

    return number


def to_odd(value) -> float | None:
    """
    Converte uma odd e rejeita valores inválidos.
    """
    number = to_float(value)

    if number is None:
        return None

    if number <= 1.0:
        return None

    return number


def to_int(value) -> int | None:
    """
    Converte golos e estatísticas para inteiro.
    """
    number = to_float(value)

    if number is None:
        return None

    return int(number)


def first_valid_odd(
    row: pd.Series,
    column_names: list[str],
) -> float | None:
    """
    Devolve a primeira odd válida encontrada.
    """
    for column_name in column_names:
        if column_name not in row.index:
            continue

        odd = to_odd(row[column_name])

        if odd is not None:
            return odd

    return None


# ============================================================
# DATA E ÉPOCA
# ============================================================

def parse_date(value) -> datetime | None:
    """
    Converte os formatos de data usados nos CSV:

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
            return datetime.strptime(
                text,
                date_format,
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

    return parsed.to_pydatetime()


def calculate_season(
    match_date: datetime,
) -> str:
    """
    Calcula a época europeia.

    Agosto de 2025 -> 2025/2026
    Maio de 2026   -> 2025/2026
    """
    if match_date.month >= 7:
        start_year = match_date.year
    else:
        start_year = match_date.year - 1

    return f"{start_year}/{start_year + 1}"


# ============================================================
# LEITURA DOS CSV
# ============================================================

def read_csv_robust(
    csv_path: Path,
) -> pd.DataFrame:
    """
    Tenta diferentes codificações comuns.
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
                on_bad_lines="skip",
            )

        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        f"Não foi possível ler {csv_path}"
    )


def detect_league(
    csv_path: Path,
    division: str | None,
) -> str:
    """
    Identifica a competicao pelo codigo Div.

    O codigo da divisao tem prioridade sobre o nome da pasta,
    porque uma pasta como England pode conter E0, E1, E2 e E3.
    """

    if division in DIVISION_LEAGUE_MAP:
        return DIVISION_LEAGUE_MAP[division]

    relative_path = csv_path.relative_to(
        HISTORICAL_DIR
    )

    if len(relative_path.parts) >= 2:
        folder_name = clean_text(
            relative_path.parts[0]
        )

        if folder_name is not None:
            return folder_name

    if division is not None:
        return division

    return "Unknown"


# ============================================================
# CRIAR REGISTO DO JOGO
# ============================================================

def create_match_record(
    row: pd.Series,
    csv_path: Path,
) -> dict | None:
    """
    Converte uma linha do CSV num jogo para a base de dados.
    """
    match_date = parse_date(
        row.get("Date")
    )

    home_team = clean_text(
        row.get("HomeTeam")
    )

    away_team = clean_text(
        row.get("AwayTeam")
    )

    division = clean_text(
        row.get("Div")
    )

    home_goals = to_int(
        row.get("FTHG")
    )

    away_goals = to_int(
        row.get("FTAG")
    )

    if (
        match_date is None
        or home_team is None
        or away_team is None
        or home_goals is None
        or away_goals is None
    ):
        return None

    league = detect_league(
        csv_path,
        division,
    )

    result = clean_text(
        row.get("FTR")
    )

    if result not in {"H", "D", "A"}:
        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"

    total_goals = (
        home_goals
        + away_goals
    )

    over25 = int(
        total_goals > 2
    )

    btts = int(
        home_goals > 0
        and away_goals > 0
    )

    # ========================================================
    # ODDS DE ABERTURA
    # ========================================================

    odd_home_open = first_valid_odd(
        row,
        [
            "AvgH",
            "B365H",
            "PSH",
        ],
    )

    odd_draw_open = first_valid_odd(
        row,
        [
            "AvgD",
            "B365D",
            "PSD",
        ],
    )

    odd_away_open = first_valid_odd(
        row,
        [
            "AvgA",
            "B365A",
            "PSA",
        ],
    )

    odd_over25_open = first_valid_odd(
        row,
        [
            "Avg>2.5",
            "B365>2.5",
            "P>2.5",
        ],
    )

    odd_under25_open = first_valid_odd(
        row,
        [
            "Avg<2.5",
            "B365<2.5",
            "P<2.5",
        ],
    )

    # ========================================================
    # ODDS DE FECHO
    # ========================================================

    odd_home_close = first_valid_odd(
        row,
        [
            "AvgCH",
            "B365CH",
            "PSCH",
        ],
    )

    odd_draw_close = first_valid_odd(
        row,
        [
            "AvgCD",
            "B365CD",
            "PSCD",
        ],
    )

    odd_away_close = first_valid_odd(
        row,
        [
            "AvgCA",
            "B365CA",
            "PSCA",
        ],
    )

    odd_over25_close = first_valid_odd(
        row,
        [
            "AvgC>2.5",
            "B365C>2.5",
            "PC>2.5",
        ],
    )

    odd_under25_close = first_valid_odd(
        row,
        [
            "AvgC<2.5",
            "B365C<2.5",
            "PC<2.5",
        ],
    )

    # Colunas antigas mantidas por compatibilidade.
    # Preferem a odd de fecho, quando disponível.
    odd_home = (
        odd_home_close
        if odd_home_close is not None
        else odd_home_open
    )

    odd_draw = (
        odd_draw_close
        if odd_draw_close is not None
        else odd_draw_open
    )

    odd_away = (
        odd_away_close
        if odd_away_close is not None
        else odd_away_open
    )

    odd_over25 = (
        odd_over25_close
        if odd_over25_close is not None
        else odd_over25_open
    )

    odd_under25 = (
        odd_under25_close
        if odd_under25_close is not None
        else odd_under25_open
    )

    return {
        "date": match_date.strftime(
            "%Y-%m-%d"
        ),
        "season": calculate_season(
            match_date
        ),
        "league": league,
        "division": division,

        "home_team": home_team,
        "away_team": away_team,

        "home_goals": home_goals,
        "away_goals": away_goals,

        "result": result,
        "over25": over25,
        "btts": btts,

        "odd_home": odd_home,
        "odd_draw": odd_draw,
        "odd_away": odd_away,

        "odd_over25": odd_over25,
        "odd_under25": odd_under25,

        "referee": clean_text(
            row.get("Referee")
        ),

        "home_shots": to_int(
            row.get("HS")
        ),
        "away_shots": to_int(
            row.get("AS")
        ),

        "home_shots_target": to_int(
            row.get("HST")
        ),
        "away_shots_target": to_int(
            row.get("AST")
        ),

        "home_corners": to_int(
            row.get("HC")
        ),
        "away_corners": to_int(
            row.get("AC")
        ),

        "home_yellow": to_int(
            row.get("HY")
        ),
        "away_yellow": to_int(
            row.get("AY")
        ),

        "home_red": to_int(
            row.get("HR")
        ),
        "away_red": to_int(
            row.get("AR")
        ),

        "odd_over25_open":
            odd_over25_open,

        "odd_under25_open":
            odd_under25_open,

        "odd_over25_close":
            odd_over25_close,

        "odd_under25_close":
            odd_under25_close,
    }


# ============================================================
# PREPARAR BASE DE DADOS
# ============================================================

def prepare_database(
    connection: sqlite3.Connection,
) -> None:
    """
    Cria a tabela ou acrescenta colunas em falta.
    """
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
        """
    )

    existing_columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(matches)"
        ).fetchall()
    }

    for column_name, column_type in (
        DATABASE_COLUMNS.items()
    ):
        if column_name in existing_columns:
            continue

        connection.execute(
            f"""
            ALTER TABLE matches
            ADD COLUMN {column_name} {column_type}
            """
        )

        print(
            f"Coluna criada: {column_name}"
        )

    connection.commit()

    try:
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            ux_matches_unique_game
            ON matches (
                date,
                league,
                home_team,
                away_team
            )
            """
        )

        connection.commit()

    except sqlite3.IntegrityError as error:
        raise RuntimeError(
            "Não foi possível criar o índice único. "
            "Ainda existem jogos duplicados na base."
        ) from error


# ============================================================
# SQL DE INSERÇÃO E ATUALIZAÇÃO
# ============================================================

UPSERT_SQL = """
INSERT INTO matches (
    date,
    season,
    league,
    division,

    home_team,
    away_team,

    home_goals,
    away_goals,

    result,
    over25,
    btts,

    odd_home,
    odd_draw,
    odd_away,

    odd_over25,
    odd_under25,

    referee,

    home_shots,
    away_shots,

    home_shots_target,
    away_shots_target,

    home_corners,
    away_corners,

    home_yellow,
    away_yellow,

    home_red,
    away_red,

    odd_over25_open,
    odd_under25_open,

    odd_over25_close,
    odd_under25_close
)
VALUES (
    :date,
    :season,
    :league,
    :division,

    :home_team,
    :away_team,

    :home_goals,
    :away_goals,

    :result,
    :over25,
    :btts,

    :odd_home,
    :odd_draw,
    :odd_away,

    :odd_over25,
    :odd_under25,

    :referee,

    :home_shots,
    :away_shots,

    :home_shots_target,
    :away_shots_target,

    :home_corners,
    :away_corners,

    :home_yellow,
    :away_yellow,

    :home_red,
    :away_red,

    :odd_over25_open,
    :odd_under25_open,

    :odd_over25_close,
    :odd_under25_close
)
ON CONFLICT (
    date,
    league,
    home_team,
    away_team
)
DO UPDATE SET
    season = COALESCE(
        excluded.season,
        matches.season
    ),

    division = COALESCE(
        excluded.division,
        matches.division
    ),

    home_goals = COALESCE(
        excluded.home_goals,
        matches.home_goals
    ),

    away_goals = COALESCE(
        excluded.away_goals,
        matches.away_goals
    ),

    result = COALESCE(
        excluded.result,
        matches.result
    ),

    over25 = COALESCE(
        excluded.over25,
        matches.over25
    ),

    btts = COALESCE(
        excluded.btts,
        matches.btts
    ),

    odd_home = COALESCE(
        excluded.odd_home,
        matches.odd_home
    ),

    odd_draw = COALESCE(
        excluded.odd_draw,
        matches.odd_draw
    ),

    odd_away = COALESCE(
        excluded.odd_away,
        matches.odd_away
    ),

    odd_over25 = COALESCE(
        excluded.odd_over25,
        matches.odd_over25
    ),

    odd_under25 = COALESCE(
        excluded.odd_under25,
        matches.odd_under25
    ),

    referee = COALESCE(
        excluded.referee,
        matches.referee
    ),

    home_shots = COALESCE(
        excluded.home_shots,
        matches.home_shots
    ),

    away_shots = COALESCE(
        excluded.away_shots,
        matches.away_shots
    ),

    home_shots_target = COALESCE(
        excluded.home_shots_target,
        matches.home_shots_target
    ),

    away_shots_target = COALESCE(
        excluded.away_shots_target,
        matches.away_shots_target
    ),

    home_corners = COALESCE(
        excluded.home_corners,
        matches.home_corners
    ),

    away_corners = COALESCE(
        excluded.away_corners,
        matches.away_corners
    ),

    home_yellow = COALESCE(
        excluded.home_yellow,
        matches.home_yellow
    ),

    away_yellow = COALESCE(
        excluded.away_yellow,
        matches.away_yellow
    ),

    home_red = COALESCE(
        excluded.home_red,
        matches.home_red
    ),

    away_red = COALESCE(
        excluded.away_red,
        matches.away_red
    ),

    odd_over25_open = COALESCE(
        excluded.odd_over25_open,
        matches.odd_over25_open
    ),

    odd_under25_open = COALESCE(
        excluded.odd_under25_open,
        matches.odd_under25_open
    ),

    odd_over25_close = COALESCE(
        excluded.odd_over25_close,
        matches.odd_over25_close
    ),

    odd_under25_close = COALESCE(
        excluded.odd_under25_close,
        matches.odd_under25_close
    )
"""


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Base de dados não encontrada: "
            f"{DATABASE_PATH}"
        )

    if not HISTORICAL_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de históricos não encontrada: "
            f"{HISTORICAL_DIR}"
        )

    csv_files = sorted(
        HISTORICAL_DIR.rglob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "Não foram encontrados CSV na pasta "
            "data/historical."
        )

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        BACKUP_DIR
        / f"football_antes_importacao_{timestamp}.db"
    )

    shutil.copy2(
        DATABASE_PATH,
        backup_path,
    )

    print("\n================================")
    print(" FOOTBALLAI — IMPORTAÇÃO")
    print("================================\n")

    print(f"CSV encontrados : {len(csv_files)}")
    print(f"Backup           : {backup_path}")

    files_processed = 0
    files_failed = 0

    rows_read = 0
    valid_rows = 0
    invalid_rows = 0

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        prepare_database(
            connection
        )

        for file_number, csv_path in enumerate(
            csv_files,
            start=1,
        ):
            relative_path = csv_path.relative_to(
                ROOT_DIR
            )

            print(
                f"\n[{file_number}/{len(csv_files)}] "
                f"{relative_path}"
            )

            try:
                dataframe = read_csv_robust(
                    csv_path
                )

            except Exception as error:
                files_failed += 1

                print(
                    f"❌ Erro ao ler: {error}"
                )

                continue

            rows_read += len(dataframe)

            file_records = []
            file_invalid = 0

            for _, row in dataframe.iterrows():
                record = create_match_record(
                    row,
                    csv_path,
                )

                if record is None:
                    invalid_rows += 1
                    file_invalid += 1
                    continue

                file_records.append(
                    record
                )

            if file_records:
                try:
                    connection.executemany(
                        UPSERT_SQL,
                        file_records,
                    )

                    connection.commit()

                except Exception:
                    connection.rollback()
                    raise

            valid_rows += len(file_records)
            files_processed += 1

            print(
                f"   Linhas lidas    : "
                f"{len(dataframe)}"
            )

            print(
                f"   Jogos válidos   : "
                f"{len(file_records)}"
            )

            print(
                f"   Linhas ignoradas: "
                f"{file_invalid}"
            )

        # ====================================================
        # RESUMO DA BASE
        # ====================================================

        total_matches = connection.execute(
            """
            SELECT COUNT(*)
            FROM matches
            """
        ).fetchone()[0]

        duplicate_groups = connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    date,
                    league,
                    home_team,
                    away_team,
                    COUNT(*) AS quantidade
                FROM matches
                GROUP BY
                    date,
                    league,
                    home_team,
                    away_team
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        open_pair_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM matches
            WHERE odd_over25_open IS NOT NULL
              AND odd_under25_open IS NOT NULL
            """
        ).fetchone()[0]

        close_pair_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM matches
            WHERE odd_over25_close IS NOT NULL
              AND odd_under25_close IS NOT NULL
            """
        ).fetchone()[0]

        season_count = connection.execute(
            """
            SELECT COUNT(DISTINCT season)
            FROM matches
            WHERE season IS NOT NULL
            """
        ).fetchone()[0]

        league_rows = connection.execute(
            """
            SELECT
                league,
                COUNT(*) AS jogos,

                COUNT(
                    CASE
                        WHEN odd_over25_open IS NOT NULL
                         AND odd_under25_open IS NOT NULL
                        THEN 1
                    END
                ) AS odds_abertura,

                COUNT(
                    CASE
                        WHEN odd_over25_close IS NOT NULL
                         AND odd_under25_close IS NOT NULL
                        THEN 1
                    END
                ) AS odds_fecho

            FROM matches

            GROUP BY league
            ORDER BY league
            """
        ).fetchall()

    print("\n")
    print("================================")
    print(" IMPORTAÇÃO CONCLUÍDA")
    print("================================\n")

    print(f"Ficheiros processados : {files_processed}")
    print(f"Ficheiros com erro    : {files_failed}")
    print(f"Linhas lidas          : {rows_read}")
    print(f"Jogos válidos lidos   : {valid_rows}")
    print(f"Linhas ignoradas      : {invalid_rows}")

    print(f"\nJogos na base         : {total_matches}")
    print(f"Grupos duplicados     : {duplicate_groups}")
    print(f"Épocas                : {season_count}")
    print(f"Odds abertura completas: {open_pair_count}")
    print(f"Odds fecho completas   : {close_pair_count}")

    print("\nResumo por liga:\n")

    print(
        f"{'Liga':<18}"
        f"{'Jogos':>8}"
        f"{'Abertura':>12}"
        f"{'Fecho':>10}"
    )

    print("-" * 48)

    for (
        league,
        matches_count,
        open_count,
        close_count,
    ) in league_rows:

        print(
            f"{str(league):<18}"
            f"{matches_count:>8}"
            f"{open_count:>12}"
            f"{close_count:>10}"
        )

    if duplicate_groups == 0:
        print(
            "\n✅ Nenhum jogo duplicado."
        )
    else:
        print(
            "\n⚠ Foram encontrados grupos duplicados."
        )

    print(
        "\n✅ Os jogos existentes foram atualizados."
    )

    print(
        "✅ Nenhum jogo foi apagado."
    )

    print(
        "✅ Odds de abertura e fecho importadas."
    )


if __name__ == "__main__":
    main()