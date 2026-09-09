from __future__ import annotations

UEFA_COMPETITIONS = {
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    848: "UEFA Conference League",
}

UEFA_LEAGUE_ALIASES = {
    "champions league": "UEFA Champions League",
    "uefa champions league": "UEFA Champions League",
    "europa league": "UEFA Europa League",
    "uefa europa league": "UEFA Europa League",
    "conference league": "UEFA Conference League",
    "uefa conference league": "UEFA Conference League",
    "uefa europa conference league": "UEFA Conference League",
}


def canonical_uefa_league(value: object) -> str | None:
    text = str(value or "").strip().casefold()
    return UEFA_LEAGUE_ALIASES.get(text)


def is_uefa_club_competition(value: object) -> bool:
    return canonical_uefa_league(value) is not None
