from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from closing_odds_v3_cloud import (
    APP_TIMEZONE, ODDS_API_BASE_URL, LEAGUE_SPORT_KEY,
    connect_turso, find_paper_table, table_columns, fetch_dicts,
    parse_fixture_datetime, http_get_json, find_unique_odds_api_event,
    extract_odds_api_totals, valid_odd,
)
from ml.paper_trade import normalize_league

def has_close(r):
    return valid_odd(r.get("odd_over25_close")) is not None and valid_odd(r.get("odd_under25_close")) is not None

def main():
    key=os.environ["ODDS_API_KEY"].strip()
    c=connect_turso()
    table=find_paper_table(c); cols=table_columns(c,table)
    dc="fixture_date" if "fixture_date" in cols else "date"
    tc="fixture_time" if "fixture_time" in cols else "time"
    rows=fetch_dicts(c,f"SELECT * FROM {table} ORDER BY {dc},{tc},id")
    c.close()
    now=datetime.now(APP_TIMEZONE)
    missing=[]
    for r in rows:
        dt=parse_fixture_datetime(r.get(dc),r.get(tc))
        league=normalize_league(r.get("league"))
        if not dt or dt>=now or has_close(r) or not league:
            continue
        sk=LEAGUE_SPORT_KEY.get(str(league))
        if sk:
            missing.append((r,dt,str(league),sk))
    print(f"PAST_MISSING_WITH_SPORT_KEY={len(missing)}")
    for r,dt,league,sk in missing[:20]:
        print(f"MISSING id={r['id']} | {dt.isoformat()} | {league} | {r.get('home_team')} vs {r.get('away_team')} | {sk}")
    if not missing:
        return
    r,dt,league,sk=missing[-1]
    snap=(dt-timedelta(minutes=5)).astimezone(timezone.utc).replace(microsecond=0)
    q=urlencode({"apiKey":key,"regions":"eu","markets":"totals","oddsFormat":"decimal","dateFormat":"iso","date":snap.isoformat().replace("+00:00","Z")})
    url=f"{ODDS_API_BASE_URL}/historical/sports/{sk}/odds/?{q}"
    print(f"PROBE id={r['id']} {r.get('home_team')} vs {r.get('away_team')} snapshot={snap.isoformat()}")
    try:
        payload,headers=http_get_json(url)
    except Exception as e:
        print("HISTORICAL_ERROR="+str(e))
        return
    print("HISTORICAL_OK")
    print("remaining="+str(headers.get("x-requests-remaining")))
    data=payload.get("data",[]) if isinstance(payload,dict) else []
    print("events="+str(len(data)))
    ev=find_unique_odds_api_event(data,league,dt,r.get("home_team"),r.get("away_team"))
    if ev is None:
        print("MATCH_NOT_FOUND")
        return
    over,under=extract_odds_api_totals(ev)
    print(f"MATCH_FOUND over={over} under={under}")
if __name__=="__main__": main()
