from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from agent.v12_modeling import ece, probability_metrics
from agent.v12_rich_production import BASE_TRAIN_END, CALIBRATION_END, RICH_ANALYST_FEATURES, build_fixture_feature_rows, fit_frozen_rich_analyst, predict_rich_probability, safe_resolve_team
from agent_v10_universe_cloud import UNIVERSE_TABLE as V10_TABLE, canonical_fixture_key, is_safely_pre_kickoff, load_upcoming_universe
from footballai_agent_cloud import connect_turso, fetch_dicts, table_exists

TZ = ZoneInfo("Europe/Lisbon")
TABLE = "agent_v12_rich_analyst_decisions"
RUN_TABLE = "agent_v12_rich_analyst_runs"
MODEL_TABLE = "agent_v12_rich_model_registry"


def load_matches(conn):
    rows = fetch_dicts(conn, """SELECT id,date,league,home_team,away_team,home_goals,away_goals,COALESCE(odd_over25_open,odd_over25) odd_over25_open,COALESCE(odd_under25_open,odd_under25) odd_under25_open,home_shots,away_shots,home_shots_target,away_shots_target,home_corners,away_corners FROM matches ORDER BY date,id""")
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d.date, errors="coerce", format="%Y-%m-%d")
    for c in ["home_goals","away_goals","odd_over25_open","odd_under25_open","home_shots","away_shots","home_shots_target","away_shots_target","home_corners","away_corners"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["league","home_team","away_team"]: d[c] = d[c].astype("string").str.strip()
    d = d.dropna(subset=["id","date","league","home_team","away_team","home_goals","away_goals"])
    d = d.sort_values(["date","id"]).drop_duplicates("id", keep="last").drop_duplicates(["date","league","home_team","away_team"], keep="last").reset_index(drop=True)
    d["actual_over25"] = ((d.home_goals+d.away_goals)>=3).astype(int)
    return d


def ensure_tables(conn):
    conn.execute(f"CREATE TABLE IF NOT EXISTS {MODEL_TABLE} (model_version TEXT PRIMARY KEY,training_fingerprint TEXT NOT NULL,base_train_end TEXT NOT NULL,calibration_end TEXT NOT NULL,feature_count INTEGER NOT NULL,train_n INTEGER NOT NULL,calibration_n INTEGER NOT NULL,calibration_brier REAL,calibration_log_loss REAL,calibration_auc REAL,calibration_ece REAL,registered_at TEXT NOT NULL)")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (id INTEGER PRIMARY KEY AUTOINCREMENT,fixture_key TEXT NOT NULL UNIQUE,evaluated_at TEXT NOT NULL,analyst_version TEXT NOT NULL,training_fingerprint TEXT NOT NULL,fixture_date TEXT NOT NULL,fixture_time TEXT,league TEXT NOT NULL,home_team TEXT NOT NULL,away_team TEXT NOT NULL,resolved_home_team TEXT NOT NULL,resolved_away_team TEXT NOT NULL,source TEXT,odd_over25_open REAL NOT NULL,odd_under25_open REAL NOT NULL,market_probability REAL NOT NULL,rich_probability REAL NOT NULL,rich_market_delta REAL NOT NULL,counterfactual_ev REAL NOT NULL,feature_count INTEGER NOT NULL,data_quality TEXT NOT NULL,features_json TEXT NOT NULL,v10_decision TEXT,v10_vote_count INTEGER,v10_vote_signature TEXT,v10_goal_score REAL,v10_attack_score REAL,v10_ml_score REAL,v10_value_score REAL,result_over25 INTEGER,counterfactual_profit REAL,resolution_source TEXT,resolved_at TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL)""")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_date ON {TABLE}(fixture_date)")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {RUN_TABLE} (id INTEGER PRIMARY KEY AUTOINCREMENT,started_at TEXT NOT NULL,finished_at TEXT NOT NULL,analyst_version TEXT NOT NULL,training_fingerprint TEXT NOT NULL,historical_matches INTEGER NOT NULL,upcoming_seen INTEGER NOT NULL,already_frozen INTEGER NOT NULL,inserted INTEGER NOT NULL,skipped_timing INTEGER NOT NULL,skipped_data INTEGER NOT NULL,resolved_updated INTEGER NOT NULL,v10_links_backfilled INTEGER NOT NULL)""")
    conn.commit()


def register_model(conn,b,now):
    row=conn.execute(f"SELECT training_fingerprint FROM {MODEL_TABLE} WHERE model_version=?",(b.version,)).fetchone()
    if row:
        if str(row[0])!=b.training_fingerprint: raise RuntimeError("Fingerprint V12 Rich mudou; aumentar versão após revalidação.")
        return
    m=b.calibration_metrics
    conn.execute(f"INSERT INTO {MODEL_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(b.version,b.training_fingerprint,BASE_TRAIN_END.date().isoformat(),CALIBRATION_END.date().isoformat(),len(b.feature_names),b.train_n,b.calibration_n,m.get("brier"),m.get("log_loss"),m.get("auc"),b.calibration_ece,now)); conn.commit()


def sig(v):
    if not v:return None
    s="".join(x for k,x in [("goal_vote","G"),("attack_vote","A"),("ml_vote","M"),("value_vote","V")] if int(v.get(k) or 0))
    return s or "NONE"


def v10_map(conn):
    if not table_exists(conn,V10_TABLE):return {}
    q=f"SELECT fixture_key,decision,vote_count,goal_vote,attack_vote,ml_vote,value_vote,goal_score,attack_score,ml_score,value_score,result_over25 FROM {V10_TABLE}"
    return {str(r["fixture_key"]):r for r in fetch_dicts(conn,q)}


def backfill_v10(conn,vmap,now):
    n=0
    for r in fetch_dicts(conn,f"SELECT fixture_key FROM {TABLE} WHERE v10_decision IS NULL"):
        k=str(r["fixture_key"]);v=vmap.get(k)
        if not v:continue
        cur=conn.execute(f"UPDATE {TABLE} SET v10_decision=?,v10_vote_count=?,v10_vote_signature=?,v10_goal_score=?,v10_attack_score=?,v10_ml_score=?,v10_value_score=?,updated_at=? WHERE fixture_key=? AND v10_decision IS NULL",(v.get("decision"),v.get("vote_count"),sig(v),v.get("goal_score"),v.get("attack_score"),v.get("ml_score"),v.get("value_score"),now,k));n+=int(cur.rowcount or 0)
    if n:conn.commit()
    return n


def apply_result(conn,k,result,source,now):
    row=conn.execute(f"SELECT odd_over25_open FROM {TABLE} WHERE fixture_key=? AND result_over25 IS NULL",(k,)).fetchone()
    if not row:return 0
    p=float(row[0])-1 if result else -1.0
    cur=conn.execute(f"UPDATE {TABLE} SET result_over25=?,counterfactual_profit=?,resolution_source=?,resolved_at=?,updated_at=? WHERE fixture_key=? AND result_over25 IS NULL",(int(result),p,source,now,now,k));return int(cur.rowcount or 0)


def resolve(conn,vmap,now):
    n=0
    for r in fetch_dicts(conn,f"SELECT fixture_key FROM {TABLE} WHERE result_over25 IS NULL"):
        k=str(r["fixture_key"]);v=vmap.get(k)
        if v and v.get("result_over25") in (0,1):n+=apply_result(conn,k,int(v["result_over25"]),"v10",now)
    rows=fetch_dicts(conn,f"SELECT d.fixture_key,m.home_goals,m.away_goals FROM {TABLE} d JOIN matches m ON m.date=d.fixture_date AND m.league=d.league AND m.home_team=d.home_team AND m.away_team=d.away_team WHERE d.result_over25 IS NULL AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL")
    seen=set()
    for r in rows:
        k=str(r["fixture_key"])
        if k in seen:continue
        seen.add(k);n+=apply_result(conn,k,int(float(r["home_goals"])+float(r["away_goals"])>=3),"matches",now)
    if n:conn.commit()
    return n


def perf(conn):
    rows=fetch_dicts(conn,f"SELECT result_over25,rich_probability,market_probability,rich_market_delta,counterfactual_profit,v10_decision FROM {TABLE} WHERE result_over25 IN (0,1) ORDER BY fixture_date,id")
    print("V12 Rich resolvidos:",len(rows))
    if len(rows)<2:return
    f=pd.DataFrame(rows);y=f.result_over25.astype(int);rp=f.rich_probability.astype(float).to_numpy();mp=f.market_probability.astype(float).to_numpy();rm=probability_metrics(y,rp);mm=probability_metrics(y,mp)
    print("Rich:",json.dumps(rm));print("Market:",json.dumps(mm));print(f"Brier gain={mm['brier']-rm['brier']:+.6f} | LogLoss gain={mm['log_loss']-rm['log_loss']:+.6f} | ECE rich={ece(y,rp):.4f} market={ece(y,mp):.4f}")
    for d in ["BET_SHADOW","PASS"]:
        s=f[f.v10_decision==d]
        if len(s):print(f"V10 {d}: n={len(s)} over={s.result_over25.mean():.2%} rich={s.rich_probability.mean():.2%} market={s.market_probability.mean():.2%}")


def main():
    started=datetime.now(timezone.utc).replace(microsecond=0).isoformat();conn=connect_turso()
    try:
        ensure_tables(conn);matches=load_matches(conn);b=fit_frozen_rich_analyst(matches);register_model(conn,b,started)
        print(f"V12 RICH {b.version} | 73 features | train<={BASE_TRAIN_END.date()} | cal<={CALIBRATION_END.date()} | fp={b.training_fingerprint[:16]} | V10 alterado=NÃO")
        vm=v10_map(conn);back=backfill_v10(conn,vm,started);resolved=resolve(conn,vm,started);up=load_upcoming_universe(conn);existing={str(x[0]) for x in conn.execute(f"SELECT fixture_key FROM {TABLE}").fetchall()};now_local=datetime.now(TZ)
        todo=[];already=timing=skipped=0
        for r in up:
            k=str(r.get("fixture_key") or canonical_fixture_key(r.get("date"),r.get("league"),r.get("home_team"),r.get("away_team")))
            if k in existing:already+=1;continue
            if not is_safely_pre_kickoff(r,now_local):timing+=1;continue
            try:
                o=float(r.get("odd_over25_open"));u=float(r.get("odd_under25_open"))
                if o<=1 or u<=1:raise ValueError("odds")
                date=pd.Timestamp(str(r.get("date")));league=str(r.get("league") or "").strip();home=str(r.get("home_team") or "").strip();away=str(r.get("away_team") or "").strip()
                safe_resolve_team(matches,league,home,date);safe_resolve_team(matches,league,away,date)
                todo.append({"fixture_key":k,"fixture_date":date.date().isoformat(),"fixture_time":r.get("time"),"league":league,"home_team":home,"away_team":away,"source":r.get("source"),"odd_over25_open":o,"odd_under25_open":u})
            except Exception as ex:
                skipped+=1
                if skipped<=20:print("SKIP",r.get("date"),r.get("home_team"),"vs",r.get("away_team"),ex)
        prepared=build_fixture_feature_rows(matches,todo) if todo else {};inserted=0
        for r in todo:
            k=r["fixture_key"];x=prepared.get(k)
            if not x:skipped+=1;continue
            p=predict_rich_probability(b,x["features"]);m=float(x["market_probability"]);v=vm.get(k);now=datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            vals=(k,now,b.version,b.training_fingerprint,r["fixture_date"],r.get("fixture_time"),r["league"],r["home_team"],r["away_team"],x["resolved_home"],x["resolved_away"],r.get("source"),r["odd_over25_open"],r["odd_under25_open"],m,p,p-m,p*r["odd_over25_open"]-1,len(RICH_ANALYST_FEATURES),"FULL_73",json.dumps(x["features"],sort_keys=True),v.get("decision") if v else None,v.get("vote_count") if v else None,sig(v),v.get("goal_score") if v else None,v.get("attack_score") if v else None,v.get("ml_score") if v else None,v.get("value_score") if v else None,now,now)
            conn.execute(f"INSERT INTO {TABLE} (fixture_key,evaluated_at,analyst_version,training_fingerprint,fixture_date,fixture_time,league,home_team,away_team,resolved_home_team,resolved_away_team,source,odd_over25_open,odd_under25_open,market_probability,rich_probability,rich_market_delta,counterfactual_ev,feature_count,data_quality,features_json,v10_decision,v10_vote_count,v10_vote_signature,v10_goal_score,v10_attack_score,v10_ml_score,v10_value_score,created_at,updated_at) VALUES ({','.join(['?']*30)})",vals);conn.commit();inserted+=1
        finished=datetime.now(timezone.utc).replace(microsecond=0).isoformat();conn.execute(f"INSERT INTO {RUN_TABLE} (started_at,finished_at,analyst_version,training_fingerprint,historical_matches,upcoming_seen,already_frozen,inserted,skipped_timing,skipped_data,resolved_updated,v10_links_backfilled) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(started,finished,b.version,b.training_fingerprint,len(matches),len(up),already,inserted,timing,skipped,resolved,back));conn.commit()
        print(f"upcoming={len(up)} frozen={already} inserted={inserted} timing={timing} skipped={skipped} resolved={resolved} v10_backfill={back}");perf(conn)
    finally:conn.close()

if __name__=="__main__":main()
