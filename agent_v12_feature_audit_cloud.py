from __future__ import annotations

import json

from footballai_agent_cloud import connect_turso, fetch_dicts


COLUMNS = [
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_corners",
    "away_corners",
    "home_yellow",
    "away_yellow",
    "home_red",
    "away_red",
]


def main() -> None:
    connection = connect_turso()
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(matches)").fetchall()
        }
        available = [column for column in COLUMNS if column in columns]
        if not available:
            raise RuntimeError("Nenhuma estatística rica disponível em matches.")

        expressions = ["COUNT(*) AS total"]
        for column in available:
            expressions.append(f"SUM(CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END) AS {column}_n")
        row = fetch_dicts(
            connection,
            "SELECT " + ", ".join(expressions) + " FROM matches",
        )[0]
        total = int(row["total"] or 0)
        print("=" * 72)
        print("V12 FEATURE AUDIT | cobertura histórica")
        print("=" * 72)
        print(f"Jogos totais: {total}")
        coverage = {}
        for column in available:
            n = int(row.get(f"{column}_n") or 0)
            coverage[column] = {"n": n, "coverage": n / total if total else 0.0}
            print(f"{column:22s}: {n:6d} ({coverage[column]['coverage']:.2%})")

        paired = [
            ("shots", "home_shots", "away_shots"),
            ("shots_target", "home_shots_target", "away_shots_target"),
            ("corners", "home_corners", "away_corners"),
            ("yellow", "home_yellow", "away_yellow"),
            ("red", "home_red", "away_red"),
        ]
        for label, home, away in paired:
            if home not in available or away not in available:
                continue
            rows = fetch_dicts(
                connection,
                f"""
                SELECT
                    COUNT(*) AS n,
                    MIN(date) AS date_min,
                    MAX(date) AS date_max,
                    AVG(CAST({home} AS REAL)) AS home_avg,
                    AVG(CAST({away} AS REAL)) AS away_avg
                FROM matches
                WHERE {home} IS NOT NULL AND {away} IS NOT NULL
                """,
            )
            print(f"PAIR {label}: {json.dumps(rows[0], ensure_ascii=False)}")

        if all(column in available for column in ["home_shots", "away_shots", "home_shots_target", "away_shots_target", "home_corners", "away_corners"]):
            rows = fetch_dicts(
                connection,
                """
                SELECT
                    COUNT(*) AS n,
                    MIN(date) AS date_min,
                    MAX(date) AS date_max
                FROM matches
                WHERE home_shots IS NOT NULL
                  AND away_shots IS NOT NULL
                  AND home_shots_target IS NOT NULL
                  AND away_shots_target IS NOT NULL
                  AND home_corners IS NOT NULL
                  AND away_corners IS NOT NULL
                """,
            )
            print("CORE_RICH_SET:", json.dumps(rows[0], ensure_ascii=False))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
