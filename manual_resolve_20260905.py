from __future__ import annotations

from update_results_cloud import (
    connect_turso,
    fetch_dicts,
    find_paper_table,
    table_columns,
    update_trade,
)

TARGETS = {
    612: {
        "date": "2026-09-05",
        "home": "Burton",
        "away": "Cambridge",
        "home_goals": 2,
        "away_goals": 2,
    },
    625: {
        "date": "2026-09-05",
        "home": "Ath Bilbao",
        "away": "Ath Madrid",
        "home_goals": 3,
        "away_goals": 0,
    },
    651: {
        "date": "2026-09-05",
        "home": "Villarreal",
        "away": "La Coruna",
        "home_goals": 2,
        "away_goals": 3,
    },
    653: {
        "date": "2026-09-05",
        "home": "Sp Lisbon",
        "away": "Nacional",
        "home_goals": 2,
        "away_goals": 0,
    },
}


def main() -> None:
    connection = connect_turso()

    try:
        table = find_paper_table(connection)
        if table is None:
            raise RuntimeError("Tabela de Paper Trade não encontrada.")

        columns = table_columns(connection, table)
        date_col = (
            "fixture_date"
            if "fixture_date" in columns
            else "date"
            if "date" in columns
            else None
        )
        if date_col is None:
            raise RuntimeError("Tabela sem fixture_date/date.")

        print("=== RESOLUÇÃO MANUAL VALIDADA 2026-09-05 ===")

        for row_id, target in TARGETS.items():
            rows = fetch_dicts(
                connection,
                f"SELECT * FROM {table} WHERE id = ?",
                (row_id,),
            )
            if len(rows) != 1:
                raise RuntimeError(
                    f"id={row_id}: esperado 1 registo, encontrados {len(rows)}."
                )

            row = rows[0]
            actual_date = str(row.get(date_col) or "")[:10]
            actual_home = str(row.get("home_team") or "").strip()
            actual_away = str(row.get("away_team") or "").strip()
            status = str(row.get("status") or "OPEN").strip().upper()

            if (
                actual_date != target["date"]
                or actual_home != target["home"]
                or actual_away != target["away"]
            ):
                raise RuntimeError(
                    "Proteção de identidade falhou: "
                    f"id={row_id} é {actual_date} | "
                    f"{actual_home} vs {actual_away}"
                )

            if status == "RESOLVED":
                print(
                    f"JÁ RESOLVED id={row_id} | "
                    f"{actual_home} vs {actual_away}"
                )
                continue

            if status == "VOID":
                raise RuntimeError(
                    f"Proteção: id={row_id} está VOID e não foi alterado."
                )

            update_trade(
                connection,
                table,
                row,
                columns,
                int(target["home_goals"]),
                int(target["away_goals"]),
            )

            print(
                f"RESOLVED id={row_id} | "
                f"{actual_home} "
                f"{target['home_goals']}-{target['away_goals']} "
                f"{actual_away}"
            )

        connection.commit()
        print("MANUAL_RESOLVE_OK")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
