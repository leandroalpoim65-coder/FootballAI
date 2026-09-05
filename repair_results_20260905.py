from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timezone

import libsql


TARGETS = (
    ("2026-08-22", "nijmegen", "excelsior"),
    ("2026-08-16", "sp braga", "gil vicente"),
)

ALIASES = {
    "nec": "nijmegen",
    "n e c": "nijmegen",
    "nec nijmegen": "nijmegen",
    "nijmegen": "nijmegen",
    "sbv excelsior": "excelsior",
    "excelsior rotterdam": "excelsior",
    "excelsior": "excelsior",
    "sporting braga": "sp braga",
    "sporting clube de braga": "sp braga",
    "sc braga": "sp braga",
    "braga": "sp braga",
    "sp braga": "sp braga",
    "gil vicente fc": "gil vicente",
    "gil vicente": "gil vicente",
}

STOPWORDS = {"fc", "cf", "afc", "club", "fk", "ac", "sv", "rcd", "cd"}


def normalize_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in ALIASES:
        return ALIASES[text]

    tokens = [t for t in text.split() if t not in STOPWORDS]
    cleaned = " ".join(tokens)
    return ALIASES.get(cleaned, cleaned)


def fetch_dicts(connection, sql: str, params: tuple = ()) -> list[dict]:
    cursor = connection.execute(sql, params)
    rows = cursor.fetchall()
    names = [item[0] for item in (cursor.description or [])]
    return [dict(zip(names, row)) for row in rows]


def find_paper_table(connection) -> str:
    existing = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    for candidate in (
        "paper_trades",
        "paper_trade",
        "paper_trading",
        "historico_paper_trade",
    ):
        if candidate in existing:
            return candidate
    raise RuntimeError("Tabela de Paper Trade não encontrada.")


def main() -> None:
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("Credenciais Turso não configuradas no GitHub Actions.")

    connection = libsql.connect(database=url, auth_token=token)

    try:
        table = find_paper_table(connection)
        columns = {
            row[1]
            for row in connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }

        date_col = (
            "fixture_date"
            if "fixture_date" in columns
            else "date"
            if "date" in columns
            else None
        )
        if date_col is None:
            raise RuntimeError("Tabela sem fixture_date/date.")

        updated_at = datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat()

        print("=== MARCAR JOGOS ADIADOS COMO VOID ===")

        for target_date, expected_home, expected_away in TARGETS:
            rows = fetch_dicts(
                connection,
                f"""
                SELECT *
                FROM {table}
                WHERE date({date_col}) = date(?)
                ORDER BY id
                """,
                (target_date,),
            )

            matches = [
                row
                for row in rows
                if normalize_name(row.get("home_team")) == expected_home
                and normalize_name(row.get("away_team")) == expected_away
            ]

            if len(matches) != 1:
                print(
                    f"ERRO {target_date} {expected_home} vs {expected_away}: "
                    f"{len(matches)} correspondências."
                )
                for row in rows:
                    print(
                        f"  candidato id={row.get('id')} "
                        f"{row.get('home_team')} vs {row.get('away_team')} "
                        f"status={row.get('status')}"
                    )
                raise RuntimeError("Matching manual ambíguo ou inexistente.")

            row = matches[0]
            current_status = str(row.get("status") or "OPEN").strip().upper()

            if current_status == "VOID":
                print(
                    f"JÁ VOID id={row['id']} | {target_date} | "
                    f"{row.get('home_team')} vs {row.get('away_team')}"
                )
                continue

            if current_status == "RESOLVED":
                raise RuntimeError(
                    f"Proteção: id={row['id']} já está RESOLVED; não foi alterado."
                )

            assignments = ["status = 'VOID'"]
            params: list[object] = []

            for col in ("home_goals", "away_goals", "result_over25", "clv"):
                if col in columns:
                    assignments.append(f"{col} = NULL")

            if "profit" in columns:
                assignments.append("profit = ?")
                params.append(0.0)

            if "updated_at" in columns:
                assignments.append("updated_at = ?")
                params.append(updated_at)

            params.append(int(row["id"]))

            connection.execute(
                f"""
                UPDATE {table}
                SET {", ".join(assignments)}
                WHERE id = ?
                """,
                tuple(params),
            )

            print(
                f"VOID id={row['id']} | {target_date} | "
                f"{row.get('home_team')} vs {row.get('away_team')}"
            )

        connection.commit()

        print()
        print("=== PENDENTES ATÉ 2026-09-05 ===")
        remaining = fetch_dicts(
            connection,
            f"""
            SELECT *
            FROM {table}
            WHERE date({date_col}) <= date('2026-09-05')
              AND UPPER(COALESCE(status, 'OPEN'))
                  NOT IN ('RESOLVED', 'VOID')
            ORDER BY {date_col}, id
            """,
        )

        for row in remaining:
            print(
                f"id={row.get('id')} | {row.get(date_col)} | "
                f"{row.get('league')} | "
                f"{row.get('home_team')} vs {row.get('away_team')} | "
                f"status={row.get('status')}"
            )

        print(f"TOTAL_PENDENTES={len(remaining)}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
