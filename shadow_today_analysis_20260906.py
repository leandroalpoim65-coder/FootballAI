from __future__ import annotations
import os
from datetime import datetime, timezone
import libsql
import pandas as pd
import numpy as np

DATE='2026-09-06'
STRATS=[('40% + odd >=1.60',.40,1.60),('45% + odd >=1.60',.45,1.60),('50% + odd >=1.60',.50,1.60),('55% atual',.55,1.00)]

def profit(r):
    return float(r.odd_over25_open)-1 if int(r.result_over25)==1 else -1.0

def main():
    db=os.environ['TURSO_DATABASE_URL']; tok=os.environ['TURSO_AUTH_TOKEN']
    c=libsql.connect(database=db,auth_token=tok)
    cur=c.execute('SELECT * FROM paper_trades ORDER BY fixture_date, fixture_time, id')
    rows=cur.fetchall(); cols=[x[0] for x in cur.description]; c.close()
    d=pd.DataFrame(rows,columns=cols)
    for col in ['clv_probability','odd_over25_open','odd_over25_close','result_over25','signal','profit','planned_stake']:
        if col in d: d[col]=pd.to_numeric(d[col],errors='coerce')
    d['status']=d['status'].fillna('OPEN').astype(str).str.upper().str.strip()
    d['fixture_date']=d['fixture_date'].astype(str).str[:10]
    resolved=d[(d.status=='RESOLVED') & d.result_over25.isin([0,1]) & d.clv_probability.notna() & d.odd_over25_open.gt(1)].copy()
    today=resolved[resolved.fixture_date==DATE].copy()
    before=resolved[resolved.fixture_date<DATE].copy()
    lines=[f'SHADOW TODAY {DATE}',f'generated={datetime.now(timezone.utc).isoformat()}',f'today_resolved_eligible={len(today)}',f'before_resolved_eligible={len(before)}','']
    for name,t,o in STRATS:
        lines.append('=== '+name+' ===')
        for label,x in [('BEFORE',before),('TODAY',today),('TOTAL',resolved[resolved.fixture_date<=DATE])]:
            s=x[(x.clv_probability>=t)&(x.odd_over25_open>=o)].copy()
            if len(s):
                s['p']=s.apply(profit,axis=1); pr=s.p.sum(); roi=100*pr/len(s); hit=100*s.result_over25.mean()
                lines.append(f'{label}: n={len(s)} profit={pr:+.3f}u roi={roi:+.2f}% hit={hit:.1f}% avgodd={s.odd_over25_open.mean():.3f}')
            else: lines.append(f'{label}: n=0')
        lines.append('')
    lines.append('=== TODAY ELIGIBLE >=40% + odd>=1.60 ===')
    s=today[(today.clv_probability>=.40)&(today.odd_over25_open>=1.60)].sort_values(['clv_probability','fixture_time'],ascending=[False,True]).copy()
    for r in s.itertuples():
        p=profit(r)
        lines.append(f'{r.fixture_time} | {r.league} | {r.home_team} vs {r.away_team} | prob={r.clv_probability:.3f} odd={r.odd_over25_open:.3f} result={int(r.result_over25)} profit={p:+.3f}u')
    lines.append('')
    # bins today and total
    lines.append('=== BINS ===')
    for lo,hi in [(.40,.45),(.45,.50),(.50,.55),(.55,2)]:
        for label,x in [('TODAY',today),('TOTAL',resolved[resolved.fixture_date<=DATE])]:
            s=x[(x.clv_probability>=lo)&(x.clv_probability<hi)&(x.odd_over25_open>=1.60)].copy()
            if len(s):
                s['p']=s.apply(profit,axis=1)
                lines.append(f'{label} {lo:.2f}-{hi if hi<2 else 1:.2f}: n={len(s)} profit={s.p.sum():+.3f}u roi={100*s.p.sum()/len(s):+.2f}% hit={100*s.result_over25.mean():.1f}%')
    open_today=d[(d.fixture_date==DATE)&(~d.status.isin(['RESOLVED','VOID']))]
    lines+=['',f'OPEN_TODAY={len(open_today)}']
    for r in open_today.itertuples(): lines.append(f'OPEN | {r.fixture_time} | {r.home_team} vs {r.away_team} | status={r.status}')
    open('/tmp/out.txt','w').write('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__': main()
