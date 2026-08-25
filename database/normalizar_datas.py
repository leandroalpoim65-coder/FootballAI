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

BACKUP_DIR = (
    ROOT_DIR
    / "database"
    / "backups"
)

MATCH_KEY = [
    "date",
    "league",
    "home_team",
    "away_team",
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_date(value) -> str | None:
    """
    Converte diferentes formatos para YYYY-MM-DD.

    Formatos aceites:
    - YYYY-MM-DD
    - DD/MM/YYYY
    - DD/MM/YY
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    supported_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
    ]

    for date_format in supported_formats:
        try:
            parsed = datetime.strptime(
                text,
                date_format,
            )

            return parsed.strftime("%Y-%m-%d")

        except ValueError:
            continue

    parsed = pd.to_datetime(
        text,
        dayfirst=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y-%m-%d")


def is_missing(value) -> bool:
    """
    Verifica se um valor está vazio.
    """
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        return value.strip() == ""

    return False


def sqlite_value(value):
    """
    Converte valores do pandas para tipos aceites pelo SQLite.
    """
    if is_missing(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass

    return value


# ============================================================
# VALIDAÇÕES
# ============================================================

if not DATABASE_PATH.exists():
    raise FileNotFoundError(
        f"Base de dados não encontrada: {DATABASE_PATH}"
    )

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BACKUP
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

backup_path = (
    BACKUP_DIR
    / f"football_antes_normalizar_datas_{timestamp}.db"
)

shutil.copy2(
    DATABASE_PATH,
    backup_path,
)

print("\nBackup criado:")
print(backup_path)


# ============================================================
# CARREGAR DADOS
# ============================================================

connection = sqlite3.connect(
    DATABASE_PATH
)

try:
    columns = [
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(matches)"
        ).fetchall()
    ]

    required_columns = [
        "id",
        *MATCH_KEY,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltam colunas na tabela matches: "
            + ", ".join(missing_columns)
        )

    matches = pd.read_sql_query(
        "SELECT * FROM matches",
        connection,
    )

    initial_count = len(matches)

    print("\n================================")
    print(" NORMALIZAÇÃO DAS DATAS")
    print("================================\n")

    print(f"Jogos antes da normalização: {initial_count}")

    matches["_original_date"] = (
        matches["date"].astype("string")
    )

    matches["date"] = (
        matches["date"].apply(normalize_date)
    )

    invalid_dates = int(
        matches["date"].isna().sum()
    )

    changed_dates = int(
        (
            matches["_original_date"]
            != matches["date"].astype("string")
        )
        .fillna(False)
        .sum()
    )

    print(f"Datas alteradas             : {changed_dates}")
    print(f"Datas inválidas             : {invalid_dates}")

    if invalid_dates > 0:
        invalid_examples = matches.loc[
            matches["date"].isna(),
            [
                "id",
                "_original_date",
                "league",
                "home_team",
                "away_team",
            ],
        ].head(10)

        print("\nExemplos de datas inválidas:")
        print(invalid_examples.to_string(index=False))

        raise ValueError(
            "Existem datas que não foi possível converter. "
            "Nenhuma alteração foi feita na base."
        )

    # ========================================================
    # DETETAR DUPLICADOS APÓS NORMALIZAÇÃO
    # ========================================================

    data_columns = [
        column
        for column in columns
        if column != "id"
    ]

    quality_data = matches[
        data_columns
    ].copy()

    object_columns = quality_data.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in object_columns:
        quality_data[column] = (
            quality_data[column]
            .replace(
                r"^\s*$",
                pd.NA,
                regex=True,
            )
        )

    matches["_completeness"] = (
        quality_data.notna().sum(axis=1)
    )

    update_columns = [
        column
        for column in columns
        if column != "id"
    ]

    update_rows = []
    ids_to_delete = []

    grouped = matches.groupby(
        MATCH_KEY,
        dropna=False,
        sort=False,
    )

    duplicate_groups = 0

    for _, group in grouped:
        ordered_group = group.sort_values(
            by=[
                "_completeness",
                "id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        keeper = ordered_group.iloc[0].copy()

        if len(ordered_group) > 1:
            duplicate_groups += 1

            duplicate_ids = (
                ordered_group.iloc[1:]["id"]
                .astype(int)
                .tolist()
            )

            ids_to_delete.extend(
                duplicate_ids
            )

        # Se o registo principal tiver uma coluna vazia,
        # tenta recuperá-la de outra cópia do mesmo jogo.
        for column in update_columns:
            if not is_missing(keeper[column]):
                continue

            for candidate in ordered_group[column]:
                if not is_missing(candidate):
                    keeper[column] = candidate
                    break

        update_values = [
            sqlite_value(keeper[column])
            for column in update_columns
        ]

        update_values.append(
            int(keeper["id"])
        )

        update_rows.append(
            tuple(update_values)
        )

    print(f"Grupos duplicados ocultos   : {duplicate_groups}")
    print(f"Linhas duplicadas a remover : {len(ids_to_delete)}")

    # ========================================================
    # ATUALIZAR BASE
    # ========================================================

    set_clause = ", ".join(
        f'"{column}" = ?'
        for column in update_columns
    )

    update_sql = f"""
        UPDATE matches
        SET {set_clause}
        WHERE id = ?
    """

    try:
        connection.execute("BEGIN")

        # O índice antigo usa a data ainda não normalizada.
        connection.execute(
            "DROP INDEX IF EXISTS ux_matches_unique_game"
        )

        connection.execute(
            "DROP INDEX IF EXISTS idx_matches_unique"
        )

        connection.execute(
            "DROP INDEX IF EXISTS idx_matches_unique_game"
        )

        # Apagar eventuais duplicados que se tornaram visíveis
        # depois da normalização das datas.
        batch_size = 500

        for start in range(
            0,
            len(ids_to_delete),
            batch_size,
        ):
            batch = ids_to_delete[
                start:start + batch_size
            ]

            placeholders = ",".join(
                "?" for _ in batch
            )

            connection.execute(
                f"""
                DELETE FROM matches
                WHERE id IN ({placeholders})
                """,
                batch,
            )

        # Atualizar datas e recuperar os campos mais completos.
        connection.executemany(
            update_sql,
            update_rows,
        )

        # Impedir que o importador volte a criar duplicados.
        connection.execute(
            """
            CREATE UNIQUE INDEX
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

    except Exception:
        connection.rollback()
        raise

    # ========================================================
    # VERIFICAÇÃO FINAL
    # ========================================================

    final_count = connection.execute(
        "SELECT COUNT(*) FROM matches"
    ).fetchone()[0]

    remaining_duplicates = connection.execute(
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

    non_iso_dates = connection.execute(
        """
        SELECT COUNT(*)
        FROM matches
        WHERE date IS NULL
           OR LENGTH(date) != 10
           OR SUBSTR(date, 5, 1) != '-'
           OR SUBSTR(date, 8, 1) != '-'
        """
    ).fetchone()[0]

finally:
    connection.close()


# ============================================================
# RESUMO
# ============================================================

print("\n================================")
print(" NORMALIZAÇÃO CONCLUÍDA")
print("================================\n")

print(f"Jogos antes             : {initial_count}")
print(f"Datas alteradas         : {changed_dates}")
print(f"Duplicados removidos    : {len(ids_to_delete)}")
print(f"Jogos depois            : {final_count}")
print(f"Grupos duplicados       : {remaining_duplicates}")
print(f"Datas fora de ISO       : {non_iso_dates}")

if remaining_duplicates == 0 and non_iso_dates == 0:
    print("\n✅ Todas as datas estão em YYYY-MM-DD.")
    print("✅ O índice único foi recriado.")
    print("✅ A base está pronta para a importação.")