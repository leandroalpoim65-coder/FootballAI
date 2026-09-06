from __future__ import annotations
from datetime import datetime, timezone
from closing_odds_v3_cloud import APP_TIMEZONE, connect_turso, find_paper_table, table_columns, fetch_dicts, parse_fixture_datetime, valid_odd, calculate_clv
from update_results_cloud import team_names_match
from ml.paper_trade import normalize_league

SOURCE="Football-Data fixtures / recovered pre-match snapshot"

def has_close(r):
    return valid_odd(r.get("odd_over25_close")) is not None and valid_odd(r.get("odd_under25_close")) is not None

def main():
    c=connect_turso()
    table=find_paper_table(c); cols=table_columns(c,table)
    dc="fixture_date" if "fixture_date" in cols else "date"
    tc="fixture_time" if "fixture_time" in cols else "time"
    now=datetime.now(APP_TIMEZONE)
    rows=fetch_dicts(c,f"SELECT * FROM {table} WHERE UPPER(COALESCE(status,'OPEN')) != 'VOID' ORDER BY {dc},{tc},id")
    upcoming=fetch_dicts(c,"SELECT * FROM upcoming_matches ORDER BY date,time")
    updated=0; unmatched=[]; ambiguous=[]
    for r in rows:
        if has_close(r):
            continue
        dt=parse_fixture_datetime(r.get(dc),r.get(tc))
        if dt is None or dt>=now:
            continue
        league=normalize_league(r.get("league"))
        if not league:
            unmatched.append(r); continue
        candidates=[]
        for u in upcoming:
            if str(u.get("date") or "")[:10] != str(r.get(dc) or "")[:10]:
                continue
            ul=normalize_league(u.get("league"))
            if str(ul or "") != str(league):
                continue
            if not team_names_match(str(league),u.get("home_team"),r.get("home_team")):
                continue
            if not team_names_match(str(league),u.get("away_team"),r.get("away_team")):
                continue
            over=valid_odd(u.get("odd_over25_current")); under=valid_odd(u.get("odd_under25_current"))
            if over is None or under is None:
                continue
            candidates.append((u,over,under))
        if len(candidates)!=1:
            (ambiguous if len(candidates)>1 else unmatched).append(r)
            continue
        u,over,under=candidates[0]
        clv=calculate_clv(r.get("odd_over25_open"),over)
        observed=str(u.get("first_observed_at") or u.get("current_observed_at") or datetime.now(timezone.utc).replace(microsecond=0).isoformat())
        c.execute(f"""
          UPDATE {table}
          SET odd_over25_close=?, odd_under25_close=?,
              closing_observed_at=?, closing_minutes_before_kickoff=NULL,
              closing_source=?, clv=?
          WHERE id=? AND (odd_over25_close IS NULL OR odd_over25_close<=1 OR odd_under25_close IS NULL OR odd_under25_close<=1)
        """,(over,under,observed,SOURCE,clv,int(r["id"])))
        updated+=1
        print(f"FILLED id={r['id']} | {r.get(dc)} | {r.get('home_team')} vs {r.get('away_team')} | over={over:.3f} under={under:.3f}")
    c.commit()
    print(f"UPDATED={updated}")
    print(f"UNMATCHED={len(unmatched)}")
    for r in unmatched:
        print(f"UNMATCHED id={r['id']} | {r.get(dc)} | {r.get('league')} | {r.get('home_team')} vs {r.get('away_team')} | status={r.get('status')}")
    print(f"AMBIGUOUS={len(ambiguous)}")
    # final count
    left=fetch_dicts(c,f"""SELECT * FROM {table}
      WHERE UPPER(COALESCE(status,'OPEN'))!='VOID'
        AND date({dc}) < date(?)
        AND (odd_over25_close IS NULL OR odd_over25_close<=1 OR odd_under25_close IS NULL OR odd_under25_close<=1)
      ORDER BY {dc},{tc},id""",(now.date().isoformat(),))
    print(f"PAST_MISSING_AFTER={len(left)}")
    c.close()
if __name__=="__main__": main()
