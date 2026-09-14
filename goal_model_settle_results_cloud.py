from __future__ import annotations

import paper_trade_uefa_cloud as uefa_cloud
import goal_model_shadow_cloud as shadow


core = uefa_cloud.core


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — GOAL MODEL RESULT SETTLEMENT")
    print("==========================================")
    print()

    connection = core.connect_turso()
    try:
        if not core.table_exists(connection, "matches"):
            raise RuntimeError("A tabela matches não existe na Turso.")

        if not core.table_exists(connection, shadow.GOAL_SHADOW_TABLE):
            print("A tabela goal_model_shadow ainda não existe. Nada para liquidar.")
            return

        # paper_trade_uefa_cloud substitui o loader por uma versão que
        # combina histórico doméstico + UEFA quando uefa_matches existe.
        matches = core.load_historical_matches(connection)
        settled = shadow.settle_existing_shadow(connection, matches)

        open_count = connection.execute(
            f"SELECT COUNT(*) FROM {shadow.GOAL_SHADOW_TABLE} WHERE status = 'OPEN'"
        ).fetchone()[0]
        settled_count = connection.execute(
            f"SELECT COUNT(*) FROM {shadow.GOAL_SHADOW_TABLE} WHERE status = 'SETTLED'"
        ).fetchone()[0]

        print(f"Resultados liquidados agora : {settled}")
        print(f"Shadow OPEN total            : {open_count}")
        print(f"Shadow SETTLED total         : {settled_count}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
