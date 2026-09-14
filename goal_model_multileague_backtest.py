from __future__ import annotations

import json
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ml.goal_model import estimate_lambdas_from_features, project_match_from_lambdas

SEASONS = ["2122", "2223", "2324"]
LEAGUES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Liga Portugal",
}
URL = "https://raw.githubusercontent.com/aftermathematic/football_brain/main/data/scraped/{season}/{league}.csv"
OUT = Path("reports")
EPS = 1e-12
THRESHOLDS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
ALPHAS = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]


def load_csv(season: str, league: str) -> pd.DataFrame:
    req = Request(URL.format(season=season, league=league), headers={"User-Agent": "FootballAI-backtest"})
    with urlopen(req, timeout=30) as response:
        raw = response.read()
    df = pd.read_csv(BytesIO(raw))
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in ["FTHG", "FTAG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


def sf(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def means(rows):
    rows = rows[-5:]
    return sum(x[0] for x in rows) / 5.0, sum(x[1] for x in rows) / 5.0


def features(h, a):
    hs, hc = means(h["all"]); aas, aac = means(a["all"])
    hhs, hhc = means(h["home"]); aasv, aacv = means(a["away"])
    return {
        "home_avg_scored_5": hs, "home_avg_conceded_5": hc,
        "away_avg_scored_5": aas, "away_avg_conceded_5": aac,
        "home_home_avg_scored_5": hhs, "home_home_avg_conceded_5": hhc,
        "away_away_avg_scored_5": aasv, "away_away_avg_conceded_5": aacv,
        "elo_diff": 0.0,
    }


def no_vig2(a, b):
    ia, ib = 1/a, 1/b
    return ia/(ia+ib), ib/(ia+ib)


def no_vig3(a, b, c):
    x = np.array([1/a, 1/b, 1/c], float); x /= x.sum()
    return tuple(float(v) for v in x)


def walk(df, season, league, league_name):
    st = defaultdict(lambda: {"all": [], "home": [], "away": []})
    prior = []
    out = []
    for date, day in df.groupby("Date", sort=True):
        lh = np.mean([x[0] for x in prior]) if prior else None
        la = np.mean([x[1] for x in prior]) if prior else None
        for _, r in day.iterrows():
            h, a = str(r.HomeTeam), str(r.AwayTeam)
            if min(len(st[h]["all"]), len(st[a]["all"]), len(st[h]["home"]), len(st[a]["away"])) < 5:
                continue
            lamh, lama = estimate_lambdas_from_features(features(st[h], st[a]), league_home_avg=lh, league_away_avg=la)
            p = project_match_from_lambdas(lamh, lama)
            hg, ag = int(r.FTHG), int(r.FTAG); total = hg + ag
            rec = {
                "season": season, "league": league, "league_name": league_name, "date": date,
                "home": h, "away": a, "hg": hg, "ag": ag, "y_result": 0 if hg > ag else (1 if hg == ag else 2),
                "p_home": p["match_result"]["home"], "p_draw": p["match_result"]["draw"], "p_away": p["match_result"]["away"],
                "y_btts": int(hg > 0 and ag > 0), "p_btts": p["btts"]["yes"],
                "y_o15": int(total > 1.5), "p_o15": p["totals"]["1.5"]["over"],
                "y_o25": int(total > 2.5), "p_o25": p["totals"]["2.5"]["over"],
                "y_o35": int(total > 3.5), "p_o35": p["totals"]["3.5"]["over"],
                "y_ho05": int(hg > .5), "p_ho05": p["team_totals"]["home"]["0.5"]["over"],
                "y_ho15": int(hg > 1.5), "p_ho15": p["team_totals"]["home"]["1.5"]["over"],
                "y_ho25": int(hg > 2.5), "p_ho25": p["team_totals"]["home"]["2.5"]["over"],
                "y_ao05": int(ag > .5), "p_ao05": p["team_totals"]["away"]["0.5"]["over"],
                "y_ao15": int(ag > 1.5), "p_ao15": p["team_totals"]["away"]["1.5"]["over"],
                "y_ao25": int(ag > 2.5), "p_ao25": p["team_totals"]["away"]["2.5"]["over"],
            }
            oh, od, oa = sf(r.get("AvgH")), sf(r.get("AvgD")), sf(r.get("AvgA"))
            if oh and od and oa and min(oh, od, oa) > 1:
                mh, md, ma = no_vig3(oh, od, oa)
                rec.update(odd_h=oh, odd_d=od, odd_a=oa, m_home=mh, m_draw=md, m_away=ma)
            oo, ou = sf(r.get("Avg>2.5")), sf(r.get("Avg<2.5"))
            if oo and ou and min(oo, ou) > 1:
                mo, mu = no_vig2(oo, ou)
                rec.update(odd_o25=oo, odd_u25=ou, m_o25=mo, m_u25=mu)
            out.append(rec)
        # same-date outcomes only enter after every game on that date was predicted
        for _, r in day.iterrows():
            h, a, hg, ag = str(r.HomeTeam), str(r.AwayTeam), int(r.FTHG), int(r.FTAG)
            st[h]["all"].append((hg, ag)); st[h]["home"].append((hg, ag))
            st[a]["all"].append((ag, hg)); st[a]["away"].append((ag, hg)); prior.append((hg, ag))
    return out


def logloss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1-EPS); y = np.asarray(y, float)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float)-np.asarray(y, float))**2))


def ece(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); ans = 0.0
    for lo in np.arange(0, 1, .1):
        hi = lo + .1; mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any(): ans += mask.mean() * abs(p[mask].mean()-y[mask].mean())
    return float(ans)


def bs(df, y, p):
    d = df[[y, p]].dropna(); yy, pp = d[y].to_numpy(), d[p].to_numpy()
    return dict(n=len(d), actual=float(yy.mean()), pred=float(pp.mean()), acc=float(((pp>=.5)==yy).mean()), brier=brier(yy,pp), logloss=logloss(yy,pp), ece=ece(yy,pp))


def o25_cmp(df):
    d = df.dropna(subset=["y_o25","p_o25","m_o25"]); y=d.y_o25.to_numpy(); pm=d.p_o25.to_numpy(); pk=d.m_o25.to_numpy()
    return dict(n=len(d), model_brier=brier(y,pm), market_brier=brier(y,pk), model_logloss=logloss(y,pm), market_logloss=logloss(y,pk), model_ece=ece(y,pm), market_ece=ece(y,pk))


def h2h_cmp(df):
    d=df.dropna(subset=["m_home","m_draw","m_away"]); y=d.y_result.to_numpy(int)
    model=d[["p_home","p_draw","p_away"]].to_numpy(float); market=d[["m_home","m_draw","m_away"]].to_numpy(float)
    one=np.eye(3)[y]
    def m(p):
        p=np.clip(p,EPS,1); p=p/p.sum(1,keepdims=True)
        return dict(acc=float((p.argmax(1)==y).mean()), brier=float(np.mean(np.sum((p-one)**2,1))), logloss=float(-np.mean(np.log(p[np.arange(len(y)),y]))))
    return dict(n=len(d), model=m(model), market=m(market))


def tune_alpha(dev):
    d=dev.dropna(subset=["y_o25","p_o25","m_o25"]); y=d.y_o25.to_numpy(); pm=d.p_o25.to_numpy(); pk=d.m_o25.to_numpy()
    rows=[]
    for a in ALPHAS:
        p=pk+a*(pm-pk); rows.append((a,logloss(y,p),brier(y,p)))
    return min(rows,key=lambda z:(z[1],z[2]))[0], rows


def replay(df, alpha, threshold):
    d=df.dropna(subset=["odd_o25","odd_u25","m_o25","p_o25","y_o25"]).sort_values("date")
    profits=[]; evs=[]; overs=unders=wins=0
    for _,r in d.iterrows():
        po=float(r.m_o25)+alpha*(float(r.p_o25)-float(r.m_o25)); pu=1-po
        evo=po*float(r.odd_o25)-1; evu=pu*float(r.odd_u25)-1
        if evo>=evu: ev,odd,won,side=evo,float(r.odd_o25),int(r.y_o25)==1,"O"
        else: ev,odd,won,side=evu,float(r.odd_u25),int(r.y_o25)==0,"U"
        if ev < threshold: continue
        profits.append(odd-1 if won else -1); evs.append(ev); wins+=int(won); overs+=side=="O"; unders+=side=="U"
    n=len(profits); eq=peak=dd=0.0
    for x in profits: eq+=x; peak=max(peak,eq); dd=max(dd,peak-eq)
    return dict(alpha=alpha, threshold=threshold, bets=n, wins=wins, hit=wins/n if n else None, profit=float(sum(profits)), roi=float(sum(profits)/n) if n else None, avg_ev=float(np.mean(evs)) if evs else None, max_dd=dd if n else None, overs=int(overs), unders=int(unders))


def boot_ci(df, alpha, threshold, nboot=3000):
    # Bootstrap settled bet P/Ls, after strategy is frozen.
    d=df.dropna(subset=["odd_o25","odd_u25","m_o25","p_o25","y_o25"]); pl=[]
    for _,r in d.iterrows():
        po=float(r.m_o25)+alpha*(float(r.p_o25)-float(r.m_o25)); evo=po*float(r.odd_o25)-1; evu=(1-po)*float(r.odd_u25)-1
        if evo>=evu: ev,odd,won=evo,float(r.odd_o25),int(r.y_o25)==1
        else: ev,odd,won=evu,float(r.odd_u25),int(r.y_o25)==0
        if ev>=threshold: pl.append(odd-1 if won else -1)
    if len(pl)<2:return None,None
    arr=np.asarray(pl,float); rng=np.random.default_rng(20260914); r=np.empty(nboot)
    for i in range(nboot):r[i]=rng.choice(arr,len(arr),replace=True).mean()
    return tuple(float(x) for x in np.quantile(r,[.025,.975]))


def main():
    OUT.mkdir(exist_ok=True); recs=[]; failures=[]
    for s in SEASONS:
        for c,n in LEAGUES.items():
            try:
                df=load_csv(s,c); rr=walk(df,s,c,n); recs.extend(rr); print(f"DATA {s} {c}: raw={len(df)} predicted={len(rr)}")
            except Exception as e:
                failures.append(f"{s}/{c}: {e}"); print("WARNING",failures[-1])
    if not recs: raise RuntimeError("no predictions")
    r=pd.DataFrame(recs).sort_values(["season","league","date"]); r.to_csv(OUT/"goal_model_multileague_predictions.csv",index=False)
    dev=r[r.season=="2122"]; val=r[r.season=="2223"]; hold=r[r.season=="2324"]
    alpha,alpha_grid=tune_alpha(dev)
    markets={"BTTS":("y_btts","p_btts"),"O1.5":("y_o15","p_o15"),"O2.5":("y_o25","p_o25"),"O3.5":("y_o35","p_o35"),"HomeO0.5":("y_ho05","p_ho05"),"HomeO1.5":("y_ho15","p_ho15"),"HomeO2.5":("y_ho25","p_ho25"),"AwayO0.5":("y_ao05","p_ao05"),"AwayO1.5":("y_ao15","p_ao15"),"AwayO2.5":("y_ao25","p_ao25")}
    probs=[]
    for scope,df in [("ALL",r),("2122",dev),("2223",val),("2324",hold)]:
        for name,(y,p) in markets.items():probs.append(dict(scope=scope,market=name,**bs(df,y,p)))
    pd.DataFrame(probs).to_csv(OUT/"goal_model_multileague_probability_metrics.csv",index=False)
    strat=[]
    for scope,df in [("DEV",dev),("VALIDATION",val),("HOLDOUT",hold)]:
        for a in sorted(set([0,.25,.5,.75,1,alpha])):
            for t in THRESHOLDS:strat.append(dict(scope=scope,**replay(df,a,t)))
    pd.DataFrame(strat).to_csv(OUT/"goal_model_multileague_o25_replay.csv",index=False)
    hold_str=[]
    for t in THRESHOLDS:
        x=replay(hold,alpha,t); lo,hi=boot_ci(hold,alpha,t); x.update(ci_low=lo,ci_high=hi); hold_str.append(x)
    league_rows=[]
    for (c,n),df in hold.groupby(["league","league_name"]):
        for t in THRESHOLDS:league_rows.append(dict(league=c,league_name=n,**replay(df,alpha,t)))
    pd.DataFrame(hold_str).to_csv(OUT/"goal_model_multileague_holdout_strategy.csv",index=False)
    pd.DataFrame(league_rows).to_csv(OUT/"goal_model_multileague_holdout_by_league.csv",index=False)
    summary={"n":len(r),"dev":len(dev),"validation":len(val),"holdout":len(hold),"alpha":alpha,"o25_dev":o25_cmp(dev),"o25_validation":o25_cmp(val),"o25_holdout":o25_cmp(hold),"h2h_holdout":h2h_cmp(hold),"holdout_strategy":hold_str,"failures":failures}
    (OUT/"goal_model_multileague_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("=== MULTI-LEAGUE SUMMARY ===")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__":main()
