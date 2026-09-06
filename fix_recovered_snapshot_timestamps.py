import os, libsql
from update_results_cloud import team_names_match
from ml.paper_trade import normalize_league
SRC="Football-Data fixtures / recovered pre-match snapshot"
c=libsql.connect(database=os.environ["TURSO_DATABASE_URL"],auth_token=os.environ["TURSO_AUTH_TOKEN"])
cur=c.execute("SELECT id,fixture_date,league,home_team,away_team FROM paper_trades WHERE closing_source=?",(SRC,))
rows=cur.fetchall(); updated=0
for rid,d,league,home,away in rows:
    ul=normalize_league(league)
    q=c.execute("SELECT date,league,home_team,away_team,current_observed_at FROM upcoming_matches WHERE date=?",(str(d)[:10],)).fetchall()
    m=[]
    for ud,ulg,uh,ua,obs in q:
        nulg=normalize_league(ulg)
        if nulg!=ul: continue
        if not team_names_match(ul,uh,home): continue
        if not team_names_match(ul,ua,away): continue
        if obs: m.append(obs)
    if len(m)==1:
        c.execute("UPDATE paper_trades SET closing_observed_at=? WHERE id=?",(m[0],rid)); updated+=1
print(f"UPDATED_TIMESTAMPS={updated}")
c.commit(); c.close()
