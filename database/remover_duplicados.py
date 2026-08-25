from pathlib import Path
from datetime import datetime
import shutil
import sqlite3

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT_DIR / "database" / "football.db"
BACKUP_DIR = ROOT_DIR / "database" / "backups"

MATCH_KEY = [
    "date",
    "league",
    "home_team",
    "away_team",
]


# ============================================================
# VALIDAÇÕES
# ============================================================

if not DATABASE_PATH.exists():
    raise FileNotFoundError(
        f"Base de dados não encontrada: {DATABASE_PATH}"
    )

BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CRIAR BACKUP
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_path = (
    BACKUP_DIR
    / f"football_antes_deduplicacao_{timestamp}.db"
)

shutil.copy2(DATABASE_PATH, backup_path)

print("\nBackup criado:")
print(backup_path)


# ============================================================
# CARREGAR A TABELA
# ============================================================

conn = sqlite3.connect(DATABASE_PATH)

try:
    df = pd.read_sql_query(
        "SELECT * FROM matches",
        conn,
    )

    print("\n===============================")
    print(" ANTES DA LIMPEZA")
    print("===============================")

    print(f"Linhas totais : {len(df)}")

    required_columns = ["id", *MATCH_KEY]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltam colunas na tabela matches: "
            + ", ".join(missing_columns)
        )

    duplicated_count = int(
        df.duplicated(
            subset=MATCH_KEY,
            keep=False,
        ).sum()
    )

    duplicated_to_remove = int(
        df.duplicated(
            subset=MATCH_KEY,
            keep="first",
        ).sum()
    )

    print(
        "Linhas pertencentes a grupos duplicados:",
        duplicated_count,
    )

    print(
        "Linhas duplicadas a remover:",
        duplicated_to_remove,
    )

    if duplicated_to_remove == 0:
        print("\nNão existem duplicados para remover.")

    else:
        # Colunas usadas para avaliar qual versão do jogo
        # contém mais informação.
        value_columns = [
            column
            for column in df.columns
            if column not in ["id", *MATCH_KEY]
        ]

        values = df[value_columns].replace(
            r"^\s*$",
            pd.NA,
            regex=True,
        )

        # Quanto maior este valor, mais completa é a linha.
        df["_completude"] = values.notna().sum(axis=1)

        # Para cada jogo:
        # 1. manter a linha com mais dados preenchidos;
        # 2. em caso de empate, manter o ID mais antigo.
        sort_columns = [
            *MATCH_KEY,
            "_completude",
            "id",
        ]

        ascending = [
            True,
            True,
            True,
            True,
            False,
            True,
        ]

        ordered = df.sort_values(
            sort_columns,
            ascending=ascending,
        )

        duplicate_mask = ordered.duplicated(
            subset=MATCH_KEY,
            keep="first",
        )

        ids_to_delete = (
            ordered.loc[duplicate_mask, "id"]
            .astype(int)
            .tolist()
        )

        cursor = conn.cursor()

        # Apagar em pequenos blocos para evitar o limite
        # de parâmetros do SQLite.
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

            cursor.execute(
                f"""
                DELETE FROM matches
                WHERE id IN ({placeholders})
                """,
                batch,
            )

        conn.commit()

        print(
            f"\nForam removidas "
            f"{len(ids_to_delete)} linhas duplicadas."
        )

    # Impede que o mesmo jogo volte a ser inserido
    # mais de uma vez.
    conn.execute(
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

    conn.commit()

    final_count = conn.execute(
        "SELECT COUNT(*) FROM matches"
    ).fetchone()[0]

    remaining_duplicates = pd.read_sql_query(
        """
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
        """,
        conn,
    )

    print("\n===============================")
    print(" DEPOIS DA LIMPEZA")
    print("===============================")

    print(f"Linhas totais       : {final_count}")
    print(
        "Grupos duplicados   :",
        len(remaining_duplicates),
    )

    if len(remaining_duplicates) == 0:
        print("\n✅ Base de dados limpa com sucesso.")
        print(
            "✅ Índice único criado para impedir "
            "novos duplicados."
        )
    else:
        print(
            "\n⚠ Ainda existem grupos duplicados."
        )

finally:
    conn.close()