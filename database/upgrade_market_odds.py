from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


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


# ============================================================
# CONFIGURAÇÃO
# ============================================================

NEW_COLUMNS = {
    "odd_over25_open": "REAL",
    "odd_under25_open": "REAL",
    "odd_over25_close": "REAL",
    "odd_under25_close": "REAL",
}


# ============================================================
# VALIDAÇÃO
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
    / f"football_antes_market_odds_{timestamp}.db"
)

shutil.copy2(
    DATABASE_PATH,
    backup_path,
)

print("\nBackup criado:")
print(backup_path)


# ============================================================
# ATUALIZAR A TABELA
# ============================================================

with sqlite3.connect(DATABASE_PATH) as connection:
    cursor = connection.cursor()

    existing_tables = {
        row[0]
        for row in cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
    }

    if "matches" not in existing_tables:
        raise RuntimeError(
            "A tabela matches não existe."
        )

    existing_columns = {
        row[1]
        for row in cursor.execute(
            "PRAGMA table_info(matches)"
        ).fetchall()
    }

    print("\nColunas encontradas:")
    print(", ".join(sorted(existing_columns)))

    added_columns = []

    for column_name, column_type in (
        NEW_COLUMNS.items()
    ):
        if column_name in existing_columns:
            print(
                f"Já existe: {column_name}"
            )
            continue

        cursor.execute(
            f"""
            ALTER TABLE matches
            ADD COLUMN {column_name} {column_type}
            """
        )

        added_columns.append(column_name)

        print(
            f"Coluna adicionada: {column_name}"
        )

    # As odds antigas foram importadas anteriormente usando,
    # na maioria dos ficheiros, as odds médias de fecho.
    # Copiamos esses valores para as novas colunas de fecho.
    if (
        "odd_over25" in existing_columns
        or "odd_over25" in {
            row[1]
            for row in cursor.execute(
                "PRAGMA table_info(matches)"
            ).fetchall()
        }
    ):
        cursor.execute(
            """
            UPDATE matches
            SET odd_over25_close = odd_over25
            WHERE odd_over25_close IS NULL
              AND odd_over25 IS NOT NULL
            """
        )

    if (
        "odd_under25" in existing_columns
        or "odd_under25" in {
            row[1]
            for row in cursor.execute(
                "PRAGMA table_info(matches)"
            ).fetchall()
        }
    ):
        cursor.execute(
            """
            UPDATE matches
            SET odd_under25_close = odd_under25
            WHERE odd_under25_close IS NULL
              AND odd_under25 IS NOT NULL
            """
        )

    connection.commit()

    # ========================================================
    # VERIFICAÇÃO
    # ========================================================

    final_columns = {
        row[1]
        for row in cursor.execute(
            "PRAGMA table_info(matches)"
        ).fetchall()
    }

    missing_after_upgrade = [
        column
        for column in NEW_COLUMNS
        if column not in final_columns
    ]

    if missing_after_upgrade:
        raise RuntimeError(
            "Não foi possível criar as colunas: "
            + ", ".join(missing_after_upgrade)
        )

    total_matches = cursor.execute(
        """
        SELECT COUNT(*)
        FROM matches
        """
    ).fetchone()[0]

    old_over_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM matches
        WHERE odd_over25 IS NOT NULL
        """
    ).fetchone()[0]

    open_over_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM matches
        WHERE odd_over25_open IS NOT NULL
        """
    ).fetchone()[0]

    close_over_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM matches
        WHERE odd_over25_close IS NOT NULL
        """
    ).fetchone()[0]

    open_under_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM matches
        WHERE odd_under25_open IS NOT NULL
        """
    ).fetchone()[0]

    close_under_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM matches
        WHERE odd_under25_close IS NOT NULL
        """
    ).fetchone()[0]


# ============================================================
# RESUMO
# ============================================================

print("\n================================")
print(" ATUALIZAÇÃO CONCLUÍDA")
print("================================\n")

print(f"Jogos preservados          : {total_matches}")
print(f"Colunas adicionadas        : {len(added_columns)}")
print(f"Odds antigas Over 2.5      : {old_over_count}")
print(f"Odds abertura Over 2.5     : {open_over_count}")
print(f"Odds fecho Over 2.5        : {close_over_count}")
print(f"Odds abertura Under 2.5    : {open_under_count}")
print(f"Odds fecho Under 2.5       : {close_under_count}")

print("\n✅ Base atualizada sem apagar jogos.")
print("✅ Backup criado antes da alteração.")