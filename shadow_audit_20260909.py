from __future__ import annotations

import os
import math
import random
from statistics import mean
from datetime import datetime
from zoneinfo import ZoneInfo

import libsql


TZ = ZoneInfo("Europe/Lisbon")


def connect():
    return libsql.connect(
        database=os.environ["TURSO_DATABASE_URL"],
        auth_token=os.environ["TURSO_AUTH_TOKEN"],
    )


def rows_as_dicts(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [x[0] for x in (cur.description or [])]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def safe_float(v):
    try:
        x=float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def profit(row):
    result=row.get("result_over25")
    odd=safe_float(row.get("odd_over25_open"))
    if result is None or odd is None:
        return None
    return odd-1.0 if int(result)==1 else -1.0


def clv_pct(row):
    open_odd=safe_float(row.get("odd_over25_open"))
    close_odd=safe_float(row.get("odd_over25_close"))
    if not open_odd or not close_odd or open_odd<=1 or close_odd<=1:
        return None
    # Positive means our opening price beat closing price.
    return (open_odd / close_odd - 1.0) * 100.0


def source_group(row):
    s=str(row.get("closing_source") or "").casefold()
    mins=safe_float(row.get("closing_minutes_before_kickoff"))
    if not s:
        return "NO_CLOSE"
    if "recovered pre-match snapshot" in s or "snapshot football-data" in s:
        return "RECOVERED_SNAPSHOT"
    if "the odds api" in s and mins is not None and mins >= 0:
        return "TRUE_PREGAME"
    if "api-football" in s and "live pre-kickoff" in s and mins is not None and mins >= 0:
        return "TRUE_PREGAME"
    if "football-data" in s and "closing" in s:
        return "POSTGAME_FINAL"
    if mins is not None and mins >= 0:
        return "TRUE_PREGAME"
    return "OTHER_CLOSE"


def metrics(rows):
    ps=[profit(r) for r in rows]
    ps=[x for x in ps if x is not None]
    n=len(ps)
    wins=sum(1 for r in rows if r.get("result_over25") is not None and int(r["result_over25"])==1)
    avgodd=mean([float(r["odd_over25_open"]) for r in rows]) if rows else 0
    total=sum(ps)
    roi=(total/n*100) if n else 0

    clvs=[clv_pct(r) for r in rows]
    clvs=[x for x in clvs if x is not None]
    return {
        "n":n,
        "wins":wins,
        "hit":wins/n*100 if n else 0,
        "avgodd":avgodd,
        "profit":total,
        "roi":roi,
        "clv_n":len(clvs),
        "clv_pos":sum(1 for x in clvs if x>0)/len(clvs)*100 if clvs else None,
        "clv_mean":mean(clvs) if clvs else None,
    }


def bootstrap_roi(rows, iters=10000, seed=42):
    ps=[profit(r) for r in rows]
    ps=[x for x in ps if x is not None]
    if not ps:
        return None
    rng=random.Random(seed)
    vals=[]
    n=len(ps)
    for _ in range(iters):
        sample=[ps[rng.randrange(n)] for __ in range(n)]
        vals.append(sum(sample)/n*100)
    vals.sort()
    lo=vals[int(0.025*iters)]
    hi=vals[int(0.975*iters)]
    ppos=sum(1 for x in vals if x>0)/iters*100
    return lo,hi,ppos


def fmt(m):
    return (
        f"n={m['n']} | W={m['wins']} | hit={m['hit']:.1f}% | "
        f"avgodd={m['avgodd']:.3f} | P/L={m['profit']:+.3f}u | ROI={m['roi']:+.2f}% | "
        f"CLVn={m['clv_n']} | CLV+={m['clv_pos'] if m['clv_pos'] is not None else float('nan'):.1f}% | "
        f"meanCLV={m['clv_mean'] if m['clv_mean'] is not None else float('nan'):+.2f}%"
    )


def main():
    c=connect()
    try:
        tables={r[0] for r in c.execute("select name from sqlite_master where type='table'").fetchall()}
        table="paper_trades" if "paper_trades" in tables else "paper_trade"

        rows=rows_as_dicts(c, f"""
            SELECT *
            FROM {table}
            WHERE COALESCE(status,'') IN ('WON','LOST','CLOSED','OVER','UNDER','WIN','LOSS','RESOLVED')
               OR result_over25 IS NOT NULL
            ORDER BY fixture_date, COALESCE(fixture_time,''), id
        """)

        # Keep only genuinely resolved non-void rows with valid open odds/probability.
        resolved=[]
        for r in rows:
            p=safe_float(r.get("clv_probability"))
            o=safe_float(r.get("odd_over25_open"))
            res=r.get("result_over25")
            if p is None or o is None or res is None:
                continue
            resolved.append(r)

        out=[]
        out.append(f"generated_at={datetime.now(TZ).isoformat(timespec='seconds')}")
        out.append(f"resolved_total={len(resolved)}")

        for threshold in (0.40,0.45):
            cand=[r for r in resolved if float(r["clv_probability"])>=threshold and float(r["odd_over25_open"])>=1.60]
            out.append("")
            out.append(f"=== THRESHOLD {threshold:.2f} + ODD>=1.60 ===")
            out.append("TOTAL: "+fmt(metrics(cand)))

            preg=[r for r in cand if source_group(r)=="TRUE_PREGAME"]
            out.append("TRUE_PREGAME: "+fmt(metrics(preg)))

            post=[r for r in cand if source_group(r)=="POSTGAME_FINAL"]
            out.append("POSTGAME_FINAL: "+fmt(metrics(post)))

            snap=[r for r in cand if source_group(r)=="RECOVERED_SNAPSHOT"]
            out.append("RECOVERED_SNAPSHOT: "+fmt(metrics(snap)))

            noc=[r for r in cand if source_group(r)=="NO_CLOSE"]
            out.append("NO_CLOSE: "+fmt(metrics(noc)))

            for k in (25,30):
                tail=cand[-k:]
                out.append(f"LATEST_{k}: "+fmt(metrics(tail)))

            b=bootstrap_roi(cand)
            if b:
                out.append(f"BOOTSTRAP95: [{b[0]:+.2f}%, {b[1]:+.2f}%] | P(ROI>0)={b[2]:.1f}%")

        # Incremental 40-45 bin.
        bin4045=[r for r in resolved if 0.40<=float(r["clv_probability"])<0.45 and float(r["odd_over25_open"])>=1.60]
        out.append("")
        out.append("=== BIN 0.40-0.45 + ODD>=1.60 ===")
        out.append("BIN: "+fmt(metrics(bin4045)))
        for k in (20,25,30):
            out.append(f"BIN_LATEST_{k}: "+fmt(metrics(bin4045[-k:])))

        # Monthly breakdown.
        for threshold in (0.40,0.45):
            cand=[r for r in resolved if float(r["clv_probability"])>=threshold and float(r["odd_over25_open"])>=1.60]
            buckets={}
            for r in cand:
                key=str(r.get("fixture_date") or "")[:7]
                buckets.setdefault(key,[]).append(r)
            out.append("")
            out.append(f"=== MONTHLY {threshold:.2f} ===")
            for key in sorted(buckets):
                out.append(f"{key}: "+fmt(metrics(buckets[key])))

        # Chronological thirds for stability.
        for threshold in (0.40,0.45):
            cand=[r for r in resolved if float(r["clv_probability"])>=threshold and float(r["odd_over25_open"])>=1.60]
            n=len(cand)
            if n:
                cuts=[0,n//3,(2*n)//3,n]
                out.append("")
                out.append(f"=== THIRDS {threshold:.2f} ===")
                for i in range(3):
                    part=cand[cuts[i]:cuts[i+1]]
                    out.append(f"THIRD_{i+1}: "+fmt(metrics(part)))

        report="\n".join(out)
        print(report)
        open("shadow_audit_20260909.txt","w",encoding="utf-8").write(report+"\n")
    finally:
        c.close()

if __name__=="__main__":
    main()
