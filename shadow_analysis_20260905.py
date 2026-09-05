from __future__ import annotations

import os
from datetime import datetime, timezone

import libsql
import numpy as np
import pandas as pd


STRATEGIES = [
    ("40% + odd >=1.60", 0.40, 1.60),
    ("45% + odd >=1.60", 0.45, 1.60),
    ("50% + odd >=1.60", 0.50, 1.60),
    ("55% atual", 0.55, 1.00),
]

BINS = [
    ("40-45%", 0.40, 0.45),
    ("45-50%", 0.45, 0.50),
    ("50-55%", 0.50, 0.55),
    ("55%+", 0.55, float("inf")),
]


def connect():
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("Credenciais Turso ausentes.")
    return libsql.connect(database=url, auth_token=token)


def fetch_df(conn, sql: str) -> pd.DataFrame:
    cur = conn.execute(sql)
    rows = cur.fetchall()
    names = [x[0] for x in (cur.description or [])]
    return pd.DataFrame(rows, columns=names)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def classify_close(row):
    over = row.get("odd_over25_close")
    under = row.get("odd_under25_close")
    if pd.isna(over) or pd.isna(under) or over <= 1 or under <= 1:
        return "Sem CLOSE"
    source = str(row.get("closing_source") or "").casefold()
    if "api-football" in source:
        if "live pre-kickoff" in source:
            return "Pré-jogo"
        return "Snapshot API-Football"
    if "football-data" in source or "pós-jogo" in source or "pos-jogo" in source:
        return "Pós-jogo"
    if "the odds api" in source or "odds api" in source:
        return "Pré-jogo"
    return "Outra"


def prep(df):
    out = df.copy()
    for c in [
        "odd_over25_open","odd_under25_open","odd_over25_close","odd_under25_close",
        "clv_probability","fair_probability_open","signal","planned_stake","profit",
        "result_over25","home_goals","away_goals","closing_minutes_before_kickoff"
    ]:
        if c in out.columns:
            out[c] = num(out[c])
    out["status"] = out.get("status", "OPEN").fillna("OPEN").astype(str).str.strip().str.upper()
    valid = out.get("odd_over25_open", pd.Series(np.nan,index=out.index)).gt(1) & out.get("odd_over25_close", pd.Series(np.nan,index=out.index)).gt(1)
    out["clv_display"] = np.nan
    out.loc[valid,"clv_display"] = out.loc[valid,"odd_over25_open"] / out.loc[valid,"odd_over25_close"] - 1
    out["close_group"] = out.apply(classify_close, axis=1)
    out["fixture_date_dt"] = pd.to_datetime(out.get("fixture_date", ""), errors="coerce")
    return out


def with_shadow_profit(df):
    out=df.copy()
    out["shadow_profit"] = np.where(
        out["result_over25"].eq(1),
        out["odd_over25_open"] - 1.0,
        -1.0
    )
    return out


def bootstrap_metrics(profits, seed=20260905, n_boot=20000):
    arr=np.asarray(profits,dtype=float)
    arr=arr[np.isfinite(arr)]
    n=len(arr)
    if n == 0:
        return (np.nan,np.nan,np.nan)
    if n == 1:
        roi=arr[0]*100
        return (roi,roi,float(roi>0)*100)
    rng=np.random.default_rng(seed)
    means=np.empty(n_boot)
    chunk=1000
    pos=0
    while pos<n_boot:
        k=min(chunk,n_boot-pos)
        idx=rng.integers(0,n,size=(k,n))
        means[pos:pos+k]=arr[idx].mean(axis=1)*100
        pos+=k
    lo,hi=np.percentile(means,[2.5,97.5])
    ppos=(means>0).mean()*100
    return (float(lo),float(hi),float(ppos))


def summarize(sample):
    n=len(sample)
    if n==0:
        return {
            "n":0,"over":np.nan,"avg_odd":np.nan,"profit":np.nan,"roi":np.nan,
            "close_n":0,"clv_pos":np.nan,"clv_mean":np.nan,"clv_median":np.nan,
            "roi_lo":np.nan,"roi_hi":np.nan,"p_roi_pos":np.nan
        }
    s=with_shadow_profit(sample)
    profit=float(s["shadow_profit"].sum())
    clv=s[s["clv_display"].notna()]
    lo,hi,p=bootstrap_metrics(s["shadow_profit"].values)
    return {
        "n":n,
        "over":float(s["result_over25"].mean()*100),
        "avg_odd":float(s["odd_over25_open"].mean()),
        "profit":profit,
        "roi":profit/n*100,
        "close_n":len(clv),
        "clv_pos":float((clv["clv_display"]>0).mean()*100) if len(clv) else np.nan,
        "clv_mean":float(clv["clv_display"].mean()*100) if len(clv) else np.nan,
        "clv_median":float(clv["clv_display"].median()*100) if len(clv) else np.nan,
        "roi_lo":lo,"roi_hi":hi,"p_roi_pos":p,
    }


def fmt(x, digits=2, sign=False):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def main():
    conn=connect()
    try:
        df=fetch_df(conn, "SELECT * FROM paper_trades ORDER BY fixture_date, fixture_time, id")
    finally:
        conn.close()

    df=prep(df)
    eligible=df[
        (df["status"]=="RESOLVED")
        & df["clv_probability"].notna()
        & df["odd_over25_open"].gt(1)
        & df["result_over25"].isin([0,1])
    ].copy()

    lines=[]
    lines.append("SHADOW ANALYSIS — TURSO LIVE")
    lines.append(f"generated_utc={datetime.now(timezone.utc).replace(microsecond=0).isoformat()}")
    lines.append(f"paper_trades_total={len(df)}")
    lines.append(f"resolved_eligible={len(eligible)}")
    if len(eligible):
        lines.append(f"date_min={eligible['fixture_date_dt'].min().date()}")
        lines.append(f"date_max={eligible['fixture_date_dt'].max().date()}")
    lines.append("")

    lines.append("=== STRATEGY SUMMARY (1u per shadow bet) ===")
    for i,(name,thr,min_odd) in enumerate(STRATEGIES):
        s=eligible[(eligible["clv_probability"]>=thr)&(eligible["odd_over25_open"]>=min_odd)].copy()
        m=summarize(s)
        lines.append(
            f"{name} | n={m['n']} | over={fmt(m['over'],1)}% | odd={fmt(m['avg_odd'],3)} | "
            f"profit={fmt(m['profit'],3,True)}u | roi={fmt(m['roi'],2,True)}% | "
            f"close={m['close_n']} | clv+={fmt(m['clv_pos'],1)}% | clv_mean={fmt(m['clv_mean'],2,True)}% | "
            f"roi95=[{fmt(m['roi_lo'],2,True)}%,{fmt(m['roi_hi'],2,True)}%] | P(ROI>0)={fmt(m['p_roi_pos'],1)}%"
        )
    lines.append("")

    lines.append("=== PROBABILITY BINS, odd >=1.60 ===")
    for label,lo,hi in BINS:
        s=eligible[
            (eligible["clv_probability"]>=lo)
            & (eligible["clv_probability"]<hi)
            & (eligible["odd_over25_open"]>=1.60)
        ].copy()
        m=summarize(s)
        lines.append(
            f"{label} | n={m['n']} | over={fmt(m['over'],1)}% | odd={fmt(m['avg_odd'],3)} | "
            f"profit={fmt(m['profit'],3,True)}u | roi={fmt(m['roi'],2,True)}% | "
            f"close={m['close_n']} | clv+={fmt(m['clv_pos'],1)}% | clv_mean={fmt(m['clv_mean'],2,True)}%"
        )
    lines.append("")

    lines.append("=== CUMULATIVE THRESHOLD GRID, odd >=1.60 ===")
    for thr in [0.40,0.425,0.45,0.475,0.50,0.525,0.55,0.575,0.60,0.625,0.65]:
        s=eligible[(eligible["clv_probability"]>=thr)&(eligible["odd_over25_open"]>=1.60)].copy()
        m=summarize(s)
        lines.append(
            f">={thr:.3f} | n={m['n']} | profit={fmt(m['profit'],3,True)}u | roi={fmt(m['roi'],2,True)}% | "
            f"over={fmt(m['over'],1)}% | odd={fmt(m['avg_odd'],3)} | clv+={fmt(m['clv_pos'],1)}% | clv_mean={fmt(m['clv_mean'],2,True)}%"
        )
    lines.append("")

    candidate=eligible[(eligible["clv_probability"]>=0.40)&(eligible["odd_over25_open"]>=1.60)].copy()
    lines.append("=== 40% + odd>=1.60 BY CLOSE QUALITY ===")
    for group in ["Pré-jogo","Pós-jogo","Snapshot API-Football","Sem CLOSE","Outra"]:
        s=candidate[candidate["close_group"]==group].copy()
        if s.empty:
            continue
        m=summarize(s)
        lines.append(
            f"{group} | n={m['n']} | profit={fmt(m['profit'],3,True)}u | roi={fmt(m['roi'],2,True)}% | "
            f"over={fmt(m['over'],1)}% | clv+={fmt(m['clv_pos'],1)}% | clv_mean={fmt(m['clv_mean'],2,True)}%"
        )
    lines.append("")

    lines.append("=== 40% + odd>=1.60 BY LEAGUE ===")
    for league,s in candidate.groupby("league",dropna=False):
        m=summarize(s)
        lines.append(
            f"{league} | n={m['n']} | profit={fmt(m['profit'],3,True)}u | roi={fmt(m['roi'],2,True)}% | "
            f"over={fmt(m['over'],1)}% | odd={fmt(m['avg_odd'],3)}"
        )
    lines.append("")

    lines.append("=== CHRONOLOGICAL STABILITY, 40% + odd>=1.60 ===")
    cand=candidate.sort_values(["fixture_date_dt","id"]).copy()
    if len(cand):
        mid=len(cand)//2
        parts=[("first_half",cand.iloc[:mid]),("second_half",cand.iloc[mid:])]
        for name,s in parts:
            m=summarize(s)
            lines.append(
                f"{name} | n={m['n']} | profit={fmt(m['profit'],3,True)}u | roi={fmt(m['roi'],2,True)}% | "
                f"over={fmt(m['over'],1)}% | clv+={fmt(m['clv_pos'],1)}% | clv_mean={fmt(m['clv_mean'],2,True)}%"
            )
    lines.append("")

    high_quality=len(candidate[candidate["close_group"].isin(["Pré-jogo","Pós-jogo"]) & candidate["clv_display"].notna()])
    pre=len(candidate[(candidate["close_group"]=="Pré-jogo") & candidate["clv_display"].notna()])
    lines.append("=== VALIDATION PROGRESS ===")
    lines.append(f"candidate_games={len(candidate)}/75")
    lines.append(f"high_quality_close={high_quality}/50")
    lines.append(f"pre_game_close={pre}/30")
    lines.append("")

    real=df[(df["signal"]==1)&(df["status"]=="RESOLVED")].copy()
    if len(real):
        actual_profit=num(real.get("profit",pd.Series(dtype=float))).fillna(0).sum()
        actual_stake=num(real.get("planned_stake",pd.Series(dtype=float))).fillna(0).sum()
        lines.append("=== REAL SIGNALS ===")
        lines.append(f"resolved_signals={len(real)}")
        lines.append(f"profit={actual_profit:+.3f}u")
        lines.append(f"stake={actual_stake:.3f}u")
        lines.append(f"roi={(actual_profit/actual_stake*100 if actual_stake>0 else np.nan):+.2f}%")
        lines.append(f"over_hit={num(real['result_over25']).mean()*100:.1f}%")

    with open("shadow_analysis_20260905.txt","w",encoding="utf-8") as fh:
        fh.write("\n".join(lines)+"\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
