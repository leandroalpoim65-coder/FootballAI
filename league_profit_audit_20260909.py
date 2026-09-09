from __future__ import annotations

import os, math
from collections import defaultdict
from statistics import mean
from datetime import datetime
from zoneinfo import ZoneInfo
import libsql

TZ=ZoneInfo("Europe/Lisbon")

def sf(v):
    try: x=float(v)
    except: return None
    return x if math.isfinite(x) else None

def profit(r):
    o=sf(r.get("odd_over25_open")); res=r.get("result_over25")
    if o is None or res is None: return None
    return o-1.0 if int(res)==1 else -1.0

def clv(r):
    o=sf(r.get("odd_over25_open")); c=sf(r.get("odd_over25_close"))
    if not o or not c or o<=1 or c<=1: return None
    return (o/c-1)*100

def metrics(rows):
    ps=[profit(r) for r in rows]; ps=[x for x in ps if x is not None]
    n=len(ps); wins=sum(1 for r in rows if r.get("result_over25") is not None and int(r["result_over25"])==1)
    cs=[clv(r) for r in rows]; cs=[x for x in cs if x is not None]
    return {
      "n":n,"wins":wins,"profit":sum(ps),"roi":sum(ps)/n*100 if n else 0,
      "hit":wins/n*100 if n else 0,
      "avgodd":mean([float(r["odd_over25_open"]) for r in rows]) if rows else 0,
      "clvn":len(cs),"clvpos":sum(1 for x in cs if x>0)/len(cs)*100 if cs else None,
      "clvmean":mean(cs) if cs else None
    }

def fmt(m):
    cp="NA" if m["clvpos"] is None else f'{m["clvpos"]:.1f}%'
    cm="NA" if m["clvmean"] is None else f'{m["clvmean"]:+.2f}%'
    return f'n={m["n"]} W={m["wins"]} hit={m["hit"]:.1f}% avg={m["avgodd"]:.3f} P/L={m["profit"]:+.3f}u ROI={m["roi"]:+.2f}% CLVn={m["clvn"]} CLV+={cp} meanCLV={cm}'

c=libsql.connect(database=os.environ["TURSO_DATABASE_URL"],auth_token=os.environ["TURSO_AUTH_TOKEN"])
cur=c.execute("""
SELECT *
FROM paper_trades
WHERE result_over25 IS NOT NULL
ORDER BY fixture_date, COALESCE(fixture_time,''), id
""")
cols=[x[0] for x in cur.description]
rows=[dict(zip(cols,r)) for r in cur.fetchall()]

out=[]
out.append(f"generated={datetime.now(TZ).isoformat(timespec='seconds')}")
for th in (0.40,0.45):
    cand=[r for r in rows if sf(r.get("clv_probability")) is not None and sf(r.get("odd_over25_open")) is not None and float(r["clv_probability"])>=th and float(r["odd_over25_open"])>=1.60]
    out.append("")
    out.append(f"=== THRESHOLD {th:.2f} + ODD>=1.60 ===")
    by=defaultdict(list)
    for r in cand: by[str(r.get("league"))].append(r)
    for league, rr in sorted(by.items(), key=lambda kv:(-metrics(kv[1])["profit"], kv[0])):
        out.append(f"{league}: {fmt(metrics(rr))}")
        sep=[r for r in rr if str(r.get("fixture_date") or "").startswith("2026-09")]
        out.append(f"  SEP: {fmt(metrics(sep))}")
        tail=rr[-10:]
        out.append(f"  LAST10: {fmt(metrics(tail))}")

# contribution analysis for 40
cand40=[r for r in rows if sf(r.get("clv_probability")) is not None and sf(r.get("odd_over25_open")) is not None and float(r["clv_probability"])>=0.40 and float(r["odd_over25_open"])>=1.60]
base=metrics(cand40)
out.append("")
out.append("=== REMOVE-ONE-LEAGUE IMPACT @40 ===")
for league in sorted({str(r.get("league")) for r in cand40}):
    kept=[r for r in cand40 if str(r.get("league"))!=league]
    m=metrics(kept)
    rr=[r for r in cand40 if str(r.get("league"))==league]
    lm=metrics(rr)
    delta=m["roi"]-base["roi"]
    out.append(f"{league}: leagueROI={lm['roi']:+.2f}% leaguePL={lm['profit']:+.3f}u n={lm['n']} | ROI_without={m['roi']:+.2f}% | delta={delta:+.2f}pp")

# contribution analysis for 45
cand45=[r for r in rows if sf(r.get("clv_probability")) is not None and sf(r.get("odd_over25_open")) is not None and float(r["clv_probability"])>=0.45 and float(r["odd_over25_open"])>=1.60]
base=metrics(cand45)
out.append("")
out.append("=== REMOVE-ONE-LEAGUE IMPACT @45 ===")
for league in sorted({str(r.get("league")) for r in cand45}):
    kept=[r for r in cand45 if str(r.get("league"))!=league]
    m=metrics(kept)
    rr=[r for r in cand45 if str(r.get("league"))==league]
    lm=metrics(rr)
    delta=m["roi"]-base["roi"]
    out.append(f"{league}: leagueROI={lm['roi']:+.2f}% leaguePL={lm['profit']:+.3f}u n={lm['n']} | ROI_without={m['roi']:+.2f}% | delta={delta:+.2f}pp")

print("\n".join(out))
c.close()
