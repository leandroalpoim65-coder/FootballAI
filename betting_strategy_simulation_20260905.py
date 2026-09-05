from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import libsql
import numpy as np
import pandas as pd


THRESHOLDS = [0.40, 0.45, 0.50]
MIN_ODD = 1.60


def connect():
    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("Credenciais Turso ausentes.")
    return libsql.connect(database=url, auth_token=token)


def fetch_df(conn, sql: str) -> pd.DataFrame:
    cur = conn.execute(sql)
    rows = cur.fetchall()
    cols = [x[0] for x in (cur.description or [])]
    return pd.DataFrame(rows, columns=cols)


def numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def prepare(df):
    out = numeric(df, [
        "id", "clv_probability", "odd_over25_open",
        "result_over25", "profit", "planned_stake",
    ])
    out["status"] = out["status"].fillna("OPEN").astype(str).str.upper().str.strip()
    out["fixture_date"] = out["fixture_date"].astype(str).str[:10]
    out["fixture_time"] = out.get("fixture_time", "00:00").fillna("00:00").astype(str).str[:8]
    return out


def single_profit(odd, result, stake=1.0):
    return stake * (odd - 1.0) if int(result) == 1 else -stake


def double_profit(row1, row2, stake=1.0):
    won = int(row1.result_over25) == 1 and int(row2.result_over25) == 1
    product_odd = float(row1.odd_over25_open) * float(row2.odd_over25_open)
    return stake * (product_odd - 1.0) if won else -stake


def pair_same_day(sample: pd.DataFrame):
    pairs = []
    leftovers = []
    for day, group in sample.groupby("fixture_date", sort=True):
        g = group.sort_values(
            ["clv_probability", "fixture_time", "id"],
            ascending=[False, True, True],
        )
        rows = list(g.itertuples(index=False))
        i = 0
        while i + 1 < len(rows):
            pairs.append((day, rows[i], rows[i+1]))
            i += 2
        if i < len(rows):
            leftovers.append(rows[i])
    return pairs, leftovers


def max_drawdown(profits: Iterable[float]) -> float:
    curve = np.cumsum(np.asarray(list(profits), dtype=float))
    if len(curve) == 0:
        return np.nan
    curve = np.concatenate([[0.0], curve])
    peaks = np.maximum.accumulate(curve)
    dd = peaks - curve
    return float(dd.max())


def longest_losing_streak(profits: Iterable[float]) -> int:
    best = cur = 0
    for p in profits:
        if p < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def metrics(tickets: pd.DataFrame):
    if tickets.empty:
        return {}
    stake = float(tickets["stake"].sum())
    profit = float(tickets["profit"].sum())
    ordered = tickets.sort_values(["date", "order"])
    wins = int((ordered["profit"] > 0).sum())
    return {
        "tickets": len(ordered),
        "stake": stake,
        "profit": profit,
        "roi": 100 * profit / stake if stake > 0 else np.nan,
        "hit": 100 * wins / len(ordered),
        "max_dd": max_drawdown(ordered["profit"]),
        "lose_streak": longest_losing_streak(ordered["profit"]),
        "avg_ticket_odd": float(ordered["ticket_odd"].mean()),
        "profitable_days": _profitable_days(ordered),
    }


def _profitable_days(tickets):
    d = tickets.groupby("date", as_index=False)["profit"].sum()
    if d.empty:
        return np.nan
    return 100 * (d["profit"] > 0).mean()


def daily_block_bootstrap(tickets: pd.DataFrame, seed=20260905, n_boot=20000):
    if tickets.empty:
        return (np.nan, np.nan, np.nan)
    daily = tickets.groupby("date", as_index=False).agg(stake=("stake","sum"), profit=("profit","sum"))
    arr = daily[["stake","profit"]].to_numpy(float)
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s = arr[idx,0].sum()
        p = arr[idx,1].sum()
        rois[i] = 100 * p / s if s > 0 else np.nan
    lo, hi = np.nanpercentile(rois, [2.5,97.5])
    ppos = 100 * np.nanmean(rois > 0)
    return float(lo), float(hi), float(ppos)


def make_singles(sample):
    rows=[]
    for r in sample.sort_values(["fixture_date","fixture_time","id"]).itertuples(index=False):
        rows.append({
            "date": r.fixture_date,
            "order": f"{r.fixture_time}-{int(r.id):06d}",
            "stake": 1.0,
            "profit": single_profit(r.odd_over25_open, r.result_over25, 1.0),
            "ticket_odd": float(r.odd_over25_open),
            "kind": "single",
        })
    return pd.DataFrame(rows)


def make_pure_doubles(sample):
    pairs, leftovers = pair_same_day(sample)
    rows=[]
    for k,(day,a,b) in enumerate(pairs):
        rows.append({
            "date": day,
            "order": f"{min(a.fixture_time,b.fixture_time)}-D{k:04d}",
            "stake": 1.0,
            "profit": double_profit(a,b,1.0),
            "ticket_odd": float(a.odd_over25_open)*float(b.odd_over25_open),
            "kind":"double",
        })
    return pd.DataFrame(rows), len(leftovers)


def make_mixed(sample, single_stake_each, double_stake):
    pairs, leftovers = pair_same_day(sample)
    rows=[]
    n=0
    for day,a,b in pairs:
        for r in (a,b):
            rows.append({
                "date": day,
                "order": f"{r.fixture_time}-S{n:05d}",
                "stake": single_stake_each,
                "profit": single_profit(r.odd_over25_open,r.result_over25,single_stake_each),
                "ticket_odd":float(r.odd_over25_open),
                "kind":"single",
            }); n+=1
        rows.append({
            "date": day,
            "order": f"{min(a.fixture_time,b.fixture_time)}-D{n:05d}",
            "stake": double_stake,
            "profit": double_profit(a,b,double_stake),
            "ticket_odd":float(a.odd_over25_open)*float(b.odd_over25_open),
            "kind":"double",
        }); n+=1
    # leftover gets full 1u single so each eligible selection retains 1u of planned exposure budget
    for r in leftovers:
        rows.append({
            "date": r.fixture_date,
            "order": f"{r.fixture_time}-L{n:05d}",
            "stake":1.0,
            "profit":single_profit(r.odd_over25_open,r.result_over25,1.0),
            "ticket_odd":float(r.odd_over25_open),
            "kind":"leftover_single",
        }); n+=1
    return pd.DataFrame(rows), len(pairs), len(leftovers)


def strategy_line(name, tickets):
    m=metrics(tickets)
    lo,hi,p=daily_block_bootstrap(tickets)
    return (
        f"{name} | tickets={m.get('tickets',0)} | stake={m.get('stake',0):.2f}u | "
        f"profit={m.get('profit',np.nan):+.3f}u | roi={m.get('roi',np.nan):+.2f}% | "
        f"hit={m.get('hit',np.nan):.1f}% | avg_ticket_odd={m.get('avg_ticket_odd',np.nan):.3f} | "
        f"maxDD={m.get('max_dd',np.nan):.3f}u | losing_streak={m.get('lose_streak',0)} | "
        f"profitable_days={m.get('profitable_days',np.nan):.1f}% | "
        f"roi95=[{lo:+.2f}%,{hi:+.2f}%] | P(ROI>0)={p:.1f}%"
    )


def main():
    conn=connect()
    try:
        df=fetch_df(conn, "SELECT * FROM paper_trades ORDER BY fixture_date, fixture_time, id")
    finally:
        conn.close()

    df=prepare(df)
    base=df[
        (df["status"]=="RESOLVED")
        & df["clv_probability"].notna()
        & df["odd_over25_open"].ge(MIN_ODD)
        & df["result_over25"].isin([0,1])
    ].copy()

    lines=[]
    lines.append("BETTING STRATEGY SIMULATION — TURSO LIVE")
    lines.append(f"generated_utc={datetime.now(timezone.utc).replace(microsecond=0).isoformat()}")
    lines.append("Pairing rule: same day, highest clv_probability first, non-overlapping pairs; no result-based pairing.")
    lines.append("Mixed 80/20 preserves 2u exposure per pair: 0.8u each single + 0.4u double.")
    lines.append("Mixed 90/10 preserves 2u exposure per pair: 0.9u each single + 0.2u double.")
    lines.append("")

    for thr in THRESHOLDS:
        sample=base[base["clv_probability"]>=thr].copy()
        pairs,leftovers=pair_same_day(sample)
        lines.append(f"=== THRESHOLD >= {thr:.0%} + ODD >= {MIN_ODD:.2f} ===")
        lines.append(f"eligible_games={len(sample)} | pairable_doubles={len(pairs)} | leftovers={len(leftovers)} | days={sample['fixture_date'].nunique()}")
        singles=make_singles(sample)
        pure,unpaired=make_pure_doubles(sample)
        mixed80,_,_=make_mixed(sample,0.8,0.4)
        mixed90,_,_=make_mixed(sample,0.9,0.2)
        lines.append(strategy_line("SIMPLES 1u cada",singles))
        lines.append(strategy_line("DUPLAS 1u cada (só pares)",pure))
        lines.append(strategy_line("MISTA 80/20",mixed80))
        lines.append(strategy_line("MISTA 90/10",mixed90))
        lines.append("")

    # Strong double overlay: base singles at 45%, doubles only when BOTH >=50%.
    base45=base[base["clv_probability"]>=0.45].copy()
    strong=base[base["clv_probability"]>=0.50].copy()
    strong_pairs,_=pair_same_day(strong)
    # Overlay budget: 1u singles on all base45 plus +0.20u per strong double.
    overlay=make_singles(base45)
    extra=[]
    for k,(day,a,b) in enumerate(strong_pairs):
        extra.append({
            "date":day,
            "order":f"{min(a.fixture_time,b.fixture_time)}-OD{k:04d}",
            "stake":0.2,
            "profit":double_profit(a,b,0.2),
            "ticket_odd":float(a.odd_over25_open)*float(b.odd_over25_open),
            "kind":"strong_double_overlay",
        })
    if extra:
        overlay=pd.concat([overlay,pd.DataFrame(extra)],ignore_index=True)
    lines.append("=== OVERLAY: SIMPLES >=45% + DUPLA EXTRA 0.2u APENAS SE AMBOS >=50% ===")
    lines.append(f"base45_games={len(base45)} | strong50_games={len(strong)} | strong_pairs={len(strong_pairs)}")
    lines.append(strategy_line("45% singles + 0.2u strong doubles",overlay))
    lines.append("")

    # Ticket-level detail for 45 threshold pairs, so we can inspect if doubles depended on one/two lucky tickets.
    lines.append("=== 45% PAIRS DETAIL ===")
    pairs,_=pair_same_day(base45)
    for day,a,b in pairs:
        prod=float(a.odd_over25_open)*float(b.odd_over25_open)
        won=int(a.result_over25)==1 and int(b.result_over25)==1
        p=prod-1 if won else -1
        lines.append(
            f"{day} | {a.home_team} vs {a.away_team} [{a.clv_probability:.3f}@{a.odd_over25_open:.3f},{int(a.result_over25)}] + "
            f"{b.home_team} vs {b.away_team} [{b.clv_probability:.3f}@{b.odd_over25_open:.3f},{int(b.result_over25)}] | "
            f"odd={prod:.3f} | {'WIN' if won else 'LOSS'} | profit={p:+.3f}u"
        )

    with open("betting_strategy_simulation_20260905.txt","w",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
