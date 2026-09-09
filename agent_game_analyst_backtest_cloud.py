from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from agent.game_analyst import analyze_features
from footballai_agent_cloud import connect_turso, fetch_dicts, load_sports_model, predict_sports_probability


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def flat_profit(result: int, odd: float) -> float:
    return odd - 1.0 if int(result) == 1 else -1.0


def main():
    print("\n==========================================")
    print(" FOOTBALLAI — GAME ANALYST BACKTEST V8")
    print("==========================================\n")

    c = connect_turso()
    try:
        rows = fetch_dicts(c, """
            SELECT id, fixture_date, fixture_time, league, home_team, away_team,
                   odd_over25_open, fair_probability_open, clv_probability,
                   features_json, result_over25
            FROM paper_trades
            WHERE UPPER(COALESCE(status,'OPEN'))='RESOLVED'
              AND result_over25 IN (0,1)
              AND odd_over25_open IS NOT NULL
              AND fair_probability_open IS NOT NULL
              AND features_json IS NOT NULL
            ORDER BY fixture_date, fixture_time, id
        """)

        model, model_features, _ = load_sports_model()
        by_date: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_date[str(r.get("fixture_date") or "")].append(r)

        prior_all: list[dict] = []
        decisions: list[dict] = []

        for date in sorted(by_date):
            day_rows = by_date[date]
            frozen_day: list[dict] = []

            for raw in day_rows:
                pre = {
                    k: raw.get(k)
                    for k in (
                        "id","fixture_date","fixture_time","league","home_team","away_team",
                        "odd_over25_open","fair_probability_open","clv_probability","features_json"
                    )
                }
                ga = analyze_features(pre["features_json"])
                sports_p = float(predict_sports_probability(model, model_features, pre))
                odd = float(pre["odd_over25_open"])
                market_p = float(pre["fair_probability_open"])
                ev = sports_p * odd - 1.0
                edge = sports_p - market_p

                league = str(pre.get("league") or "")
                hist = [x for x in prior_all if x["league"] == league and x["game_score"] >= 60]
                hist_hit = (sum(x["result"] for x in hist) / len(hist)) if hist else None

                final_score = 0.55 * ga.game_score + 0.30 * (sports_p * 100.0)
                final_score += 8.0 if ev >= 0.05 else (4.0 if ev >= 0.02 else (-6.0 if ev < 0 else 0.0))
                final_score += 5.0 if edge >= 0.03 else (2.0 if edge >= 0.01 else (-4.0 if edge < 0 else 0.0))
                if hist_hit is not None and len(hist) >= 8:
                    final_score += max(-5.0, min(5.0, (hist_hit - 0.55) * 20.0))

                final_score = max(0.0, min(100.0, final_score))
                if final_score >= 68 and ev >= 0.0:
                    decision = "BET"
                elif final_score >= 60:
                    decision = "WATCH"
                else:
                    decision = "PASS"

                frozen_day.append({
                    "paper_trade_id": int(raw["id"]),
                    "fixture_date": date,
                    "league": league,
                    "home_team": raw.get("home_team"),
                    "away_team": raw.get("away_team"),
                    "odd": odd,
                    "clv_probability": raw.get("clv_probability"),
                    "game_score": ga.game_score,
                    "sports_probability": sports_p,
                    "ev": ev,
                    "edge": edge,
                    "final_score": final_score,
                    "decision": decision,
                    "reasons": ga.reasons,
                    "_raw_result": int(raw["result_over25"]),
                })

            # reveal only after all matches of the date have been decided
            for d in frozen_day:
                result = int(d.pop("_raw_result"))
                d["result"] = result
                d["profit"] = flat_profit(result, d["odd"])
                decisions.append(d)
                prior_all.append(d)

        bets = [d for d in decisions if d["decision"] == "BET"]
        watches = [d for d in decisions if d["decision"] == "WATCH"]
        passes = [d for d in decisions if d["decision"] == "PASS"]
        bet_profit = sum(d["profit"] for d in bets)
        bet_roi = bet_profit / len(bets) if bets else None
        bet_hit = sum(d["result"] for d in bets) / len(bets) if bets else None

        low_clv_bets = [d for d in bets if d.get("clv_probability") is not None and float(d["clv_probability"]) < 0.55]
        low_clv_profit = sum(d["profit"] for d in low_clv_bets)
        low_clv_roi = low_clv_profit / len(low_clv_bets) if low_clv_bets else None
        low_clv_hit = sum(d["result"] for d in low_clv_bets) / len(low_clv_bets) if low_clv_bets else None

        print(f"Jogos resolvidos analisados : {len(decisions)}")
        print(f"BET                        : {len(bets)}")
        print(f"WATCH                      : {len(watches)}")
        print(f"PASS                       : {len(passes)}")
        if bet_roi is not None:
            print(f"Hit rate BET               : {bet_hit:.2%}")
            print(f"ROI BET                    : {bet_roi:.2%}")
            print(f"Lucro BET (1u cada)        : {bet_profit:.2f}u")
        print(f"BET com CLV <55%           : {len(low_clv_bets)}")
        if low_clv_roi is not None:
            print(f"Hit rate CLV<55            : {low_clv_hit:.2%}")
            print(f"ROI CLV<55                 : {low_clv_roi:.2%}")
            print(f"Lucro CLV<55               : {low_clv_profit:.2f}u")

        print("\nTop 15 decisões BET por score:")
        for d in sorted(bets, key=lambda x: x["final_score"], reverse=True)[:15]:
            print(
                f"{d['fixture_date']} | {d['home_team']} vs {d['away_team']} | "
                f"score={d['final_score']:.1f} game={d['game_score']:.1f} "
                f"ml={d['sports_probability']:.1%} ev={d['ev']:.1%} "
                f"clv={float(d['clv_probability']):.1%} odd={d['odd']:.2f} "
                f"result={'WIN' if d['result'] else 'LOSS'} profit={d['profit']:.2f}u"
            )
    finally:
        c.close()


if __name__ == "__main__":
    main()
