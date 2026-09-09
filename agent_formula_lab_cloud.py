from __future__ import annotations

from collections import defaultdict

from agent.game_analyst import analyze_features
from footballai_agent_cloud import connect_turso, fetch_dicts, load_sports_model, predict_sports_probability

THRESHOLDS = (55, 60, 65, 68, 70, 72, 75, 80)


def clamp(x): return max(0.0, min(100.0, x))
def profit(result, odd): return odd - 1.0 if int(result) == 1 else -1.0


def formulas(game, ml, ev, edge):
    ml100 = ml * 100
    ev100 = ev * 100
    edge100 = edge * 100
    ev_score = clamp(50 + ev100 * 2.5)
    edge_score = clamp(50 + edge100 * 4.0)
    return {
        "F1_STATS_PURE": game,
        "F2_ML_VALUE": clamp(.70*ml100 + .30*ev_score),
        "F3_BALANCED": clamp(.45*game + .30*ml100 + .15*ev_score + .10*edge_score),
        "F4_STATS_DOMINANT": clamp(.70*game + .20*ml100 + .10*ev_score),
        "F5_ML_DOMINANT": clamp(.25*game + .55*ml100 + .15*ev_score + .05*edge_score),
        "F6_VALUE_DOMINANT": clamp(.25*game + .20*ml100 + .35*ev_score + .20*edge_score),
        "F7_CONSENSUS": clamp((game + ml100 + ev_score + edge_score)/4),
        "F8_STATS_ML": clamp(.60*game + .40*ml100),
        "F9_STATS_VALUE": clamp(.60*game + .25*ev_score + .15*edge_score),
        "F10_ENSEMBLE": clamp(.40*game + .30*ml100 + .20*ev_score + .10*edge_score),
    }


def metrics(rows):
    if not rows: return None
    p = sum(r["profit"] for r in rows)
    return len(rows), sum(r["result"] for r in rows)/len(rows), p, p/len(rows), sum(r["odd"] for r in rows)/len(rows)


def main():
    print("\n==========================================")
    print(" FOOTBALLAI — FORMULA LAB V9")
    print("==========================================\n")
    c = connect_turso()
    try:
        rows = fetch_dicts(c, """
        SELECT id, fixture_date, fixture_time, league, home_team, away_team,
               odd_over25_open, fair_probability_open, clv_probability,
               features_json, result_over25
        FROM paper_trades
        WHERE UPPER(COALESCE(status,'OPEN'))='RESOLVED'
          AND result_over25 IN (0,1) AND odd_over25_open IS NOT NULL
          AND fair_probability_open IS NOT NULL AND features_json IS NOT NULL
        ORDER BY fixture_date, fixture_time, id
        """)
        model, model_features, _ = load_sports_model()
        data=[]
        for r in rows:
            pre={k:r.get(k) for k in ("id","fixture_date","fixture_time","league","home_team","away_team","odd_over25_open","fair_probability_open","clv_probability","features_json")}
            game=analyze_features(pre["features_json"]).game_score
            ml=float(predict_sports_probability(model, model_features, pre))
            odd=float(pre["odd_over25_open"]); market=float(pre["fair_probability_open"])
            ev=ml*odd-1; edge=ml-market
            result=int(r["result_over25"])
            data.append({"date":str(r["fixture_date"]),"odd":odd,"result":result,"profit":profit(result,odd),"clv":r.get("clv_probability"),"scores":formulas(game,ml,ev,edge),"ev":ev})

        dates=sorted(set(x["date"] for x in data))
        cut=dates[max(0, int(len(dates)*.70)-1)] if dates else ""
        train=[x for x in data if x["date"]<=cut]
        valid=[x for x in data if x["date"]>cut]
        print(f"Jogos: {len(data)} | treino temporal: {len(train)} | validação cega: {len(valid)} | corte: {cut}")
        print("CLV NÃO entra em nenhuma fórmula.\n")

        candidates=[]
        for name in formulas(50,.5,0,0):
            for t in THRESHOLDS:
                tr=[x for x in train if x["scores"][name]>=t and x["ev"]>=0]
                va=[x for x in valid if x["scores"][name]>=t and x["ev"]>=0]
                tm=metrics(tr); vm=metrics(va)
                if tm and vm:
                    candidates.append((name,t,tm,vm))

        candidates.sort(key=lambda z: (z[3][3] if z[3][0]>=10 else -99, z[2][3]), reverse=True)
        print("TOP FÓRMULAS — ordenadas por ROI na validação (mínimo preferido 10 bets):")
        for name,t,tr,va in candidates[:25]:
            print(f"{name:18s} >= {t:2d} | TRAIN n={tr[0]:3d} hit={tr[1]:6.1%} roi={tr[3]:7.1%} | VALID n={va[0]:3d} hit={va[1]:6.1%} roi={va[3]:7.1%} lucro={va[2]:6.2f}u odd={va[4]:.2f}")

        robust=[z for z in candidates if z[2][0]>=20 and z[3][0]>=10 and z[2][3]>0 and z[3][3]>0]
        print("\nROBUSTAS POSITIVAS EM TREINO + VALIDAÇÃO:")
        for name,t,tr,va in robust[:15]:
            low=[x for x in valid if x["scores"][name]>=t and x["ev"]>=0 and x.get("clv") is not None and float(x["clv"])<.55]
            lm=metrics(low)
            lowtxt=f" | CLV<55 valid n={lm[0]} roi={lm[3]:.1%}" if lm else ""
            print(f"{name:18s} >= {t:2d} | train {tr[0]} roi={tr[3]:.1%} | valid {va[0]} roi={va[3]:.1%}{lowtxt}")
    finally:
        c.close()

if __name__ == "__main__": main()
