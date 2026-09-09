from __future__ import annotations

import os
from datetime import datetime

from agent.uefa_competitions import UEFA_COMPETITIONS
from upcoming_fallback_sources import (
    API_FOOTBALL_LEAGUE_ID_MAP,
    fetch_api_football_fallback,
)
from upcoming_football_data_cloud import (
    APP_TIMEZONE,
    connect_turso,
    ensure_table,
    persist_observations,
)


DEFAULT_HORIZON_DAYS = 7
DEFAULT_MAX_REQUESTS = 28


def main() -> None:
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not api_key:
        raise RuntimeError("API_FOOTBALL_KEY não configurada.")

    API_FOOTBALL_LEAGUE_ID_MAP.update(UEFA_COMPETITIONS)

    today = datetime.now(APP_TIMEZONE).date()
    horizon = max(1, int(os.getenv("UEFA_UPCOMING_DAYS", DEFAULT_HORIZON_DAYS)))
    budget = max(4, int(os.getenv("UEFA_UPCOMING_MAX_REQUESTS", DEFAULT_MAX_REQUESTS)))

    observations, metadata = fetch_api_football_fallback(
        today=today,
        api_key=api_key,
        horizon_days=horizon,
        max_requests=budget,
    )
    allowed = set(UEFA_COMPETITIONS.values())
    observations = [row for row in observations if row.get("league") in allowed]

    connection = connect_turso()
    try:
        ensure_table(connection)
        observed_at = datetime.now(APP_TIMEZONE).isoformat()
        created, updated, preserved_open = persist_observations(
            connection,
            observations,
            observed_at,
        )
        connection.commit()
    finally:
        connection.close()

    print("FOOTBALLAI — UEFA UPCOMING")
    print(f"Janela                   : {horizon} dias")
    print(f"Observações UEFA         : {len(observations)}")
    print(f"Criadas                  : {created}")
    print(f"Atualizadas              : {updated}")
    print(f"OPEN preservadas         : {preserved_open}")
    print(f"Requests API-Football    : {metadata.get('requests')}")
    print(f"Quota restante reportada : {metadata.get('remaining')}")
    if metadata.get("error"):
        print(f"Aviso                    : {metadata.get('error')}")


if __name__ == "__main__":
    main()
