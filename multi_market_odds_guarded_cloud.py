from __future__ import annotations

from datetime import datetime

import multi_market_odds_cloud as collector


_original_load_upcoming = collector.load_upcoming


def _fixture_is_future(row: dict) -> bool:
    """Evita guardar uma odd observada depois do início do jogo."""
    date_text = str(row.get("date") or "").strip()
    time_text = str(row.get("time") or "").strip()
    if not date_text:
        return False

    # Sem hora confirmada mantemos o jogo; as fontes usadas são pre-match.
    if not time_text:
        return True

    for candidate in (
        f"{date_text[:10]} {time_text[:5]}",
        f"{date_text[:10]} {time_text}",
    ):
        try:
            kickoff = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=collector.APP_TIMEZONE)
        else:
            kickoff = kickoff.astimezone(collector.APP_TIMEZONE)
        return kickoff > datetime.now(collector.APP_TIMEZONE)

    return True


def load_upcoming_future_only(connection, horizon_days: int) -> list[dict]:
    rows = _original_load_upcoming(connection, horizon_days)
    return [row for row in rows if _fixture_is_future(row)]


collector.load_upcoming = load_upcoming_future_only


if __name__ == "__main__":
    collector.main()
