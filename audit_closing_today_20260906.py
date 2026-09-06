import os, libsql
from datetime import datetime
from zoneinfo import ZoneInfo

TODAY="2026-09-06"
db=os.environ["TURSO_DATABASE_URL"]; tok=os.environ["TURSO_AUTH_TOKEN"]
c=libsql.connect(database=db,auth_token=tok)
cur=c.execute("""
SELECT id, fixture_date, fixture_time, league, home_team, away_team,
       status, odd_over25_open, odd_over25_close, odd_under25_close,
       closing_source, closing_minutes_before_kickoff, closing_observed_at,
       api_football_fixture_id
FROM paper_trades
WHERE date(fixture_date)=date(?)
ORDER BY fixture_time,id
""",(TODAY,))
cols=[x[0] for x in cur.description]
rows=[dict(zip(cols,r)) for r in cur.fetchall()]
c.close()

def group(r):
    src=str(r.get("closing_source") or "").lower()
    over=r.get("odd_over25_close"); under=r.get("odd_under25_close")
    valid=False
    try: valid=float(over)>1 and float(under)>1
    except: pass
    if not valid: return "SEM CLOSE"
    if "the odds api" in src: return "PRE-JOGO ODDS API"
    if "api-football" in src and "live pre-kickoff" in src: return "PRE-JOGO API-FOOTBALL LIVE"
    if "api-football" in src: return "SNAPSHOT API-FOOTBALL"
    if "football-data" in src: return "POS-JOGO FOOTBALL-DATA"
    return "OUTRA CLOSE"

from collections import Counter
cnt=Counter(group(r) for r in rows)
lines=[f"AUDIT CLOSE {TODAY}",f"total={len(rows)}"]
for k,v in cnt.items(): lines.append(f"{k}={v}")
lines.append("")
for r in rows:
    lines.append(
      f"{r['fixture_time']} | {r['home_team']} vs {r['away_team']} | status={r['status']} | "
      f"group={group(r)} | close={r['odd_over25_close']} | min={r['closing_minutes_before_kickoff']} | "
      f"src={r['closing_source']} | api_fixture={r['api_football_fixture_id']}"
    )
open("audit_closing_today_20260906.txt","w",encoding="utf-8").write("\n".join(lines)+"\n")
print("\n".join(lines))
