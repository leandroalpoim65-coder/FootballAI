from __future__ import annotations

import json
from datetime import datetime, timezone

import goal_model_value_shadow_cloud as base


DIAGNOSTIC_ONLY_MARKETS = {"h2h"}


def update_open_rows_rankable(connection) -> int:
    """Keep every market for diagnostics, but rank only calibrated families.

    Controlled EPL holdout showed the current Poisson v1 with zero Elo
    adjustment materially overstates 1X2 value. 1X2 therefore stays inside
    market_values_json for analysis but cannot become best_market until a
    later calibrated version explicitly removes it from DIAGNOSTIC_ONLY_MARKETS.
    """
    rows = base.core.fetch_dicts(
        connection,
        f"""
        SELECT
            id,
            fixture_key,
            projection_json,
            odd_over25_open,
            odd_under25_open
        FROM {base.SHADOW_TABLE}
        WHERE status = 'OPEN'
          AND fixture_key IS NOT NULL
        """,
    )

    updated = 0
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for row in rows:
        try:
            projection = json.loads(str(row["projection_json"]))
        except (TypeError, json.JSONDecodeError):
            continue

        odds_rows = base.load_market_odds(connection, row.get("fixture_key"))
        base.append_legacy_over25(odds_rows, row)
        opportunities = base.build_opportunities(projection, odds_rows)

        rankable = [
            item
            for item in opportunities
            if str(item.get("market") or "") not in DIAGNOSTIC_ONLY_MARKETS
        ]
        rankable.sort(key=lambda item: float(item["ev"]), reverse=True)
        best = rankable[0] if rankable else None

        # Keep all opportunities (including 1X2) for diagnostic reports.
        fields = {
            "market_odds_json": json.dumps(
                odds_rows,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "market_values_json": json.dumps(
                opportunities,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "market_odds_count": len(opportunities),
            "best_market": best.get("market") if best else None,
            "best_selection": best.get("selection") if best else None,
            "best_line": best.get("line") if best else None,
            "best_odd": best.get("odd") if best else None,
            "best_fair_odd": best.get("fair_odd") if best else None,
            "best_probability": best.get("model_probability") if best else None,
            "best_implied_probability": best.get("implied_probability") if best else None,
            "best_edge_pp": best.get("edge_pp") if best else None,
            "best_ev": best.get("ev") if best else None,
            "best_positive_ev": int(
                best is not None and float(best["ev"]) > 0.0
            ),
        }

        assignments = ", ".join(f"{name} = ?" for name in fields)
        connection.execute(
            f"UPDATE {base.SHADOW_TABLE} "
            f"SET {assignments}, updated_at = ? WHERE id = ?",
            tuple(fields.values()) + (now, int(row["id"])),
        )
        updated += 1

    connection.commit()
    return updated


base.update_open_rows = update_open_rows_rankable


if __name__ == "__main__":
    base.main()
