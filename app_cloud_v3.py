from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import libsql
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

TZ = ZoneInfo("Europe/Lisbon")
THRESHOLD = 0.55
SHADOW = [0.40, 0.45, 0.50, 0.525, 0.55]

st.set_page_config(page_title="FootballAI Cloud", page_icon="⚽", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:.8rem;padding-bottom:2rem;max-width:1500px}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:.75rem 1rem;background:rgba(128,128,128,.04)}
[data-testid="stMetricLabel"]{font-weight:650}
[data-testid="stMetricValue"]{font-size:1.55rem}
@media(max-width:700px){.block-container{padding-left:.65rem;padding-right:.65rem;padding-top:.55rem}[data-testid="stMetricValue"]{font-size:1.25rem}}
</style>
""", unsafe_allow_html=True)


def setting(name: str) -> str:
    try:
        value = st.secrets[name]
    except Exception:
        value = os.getenv(name, "")
    return str(value or "").strip()


def connect():
    url, token = setting("TURSO_DATABASE_URL"), setting("TURSO_AUTH_TOKEN")
    if not url or not token:
        raise RuntimeError("TURSO_DATABASE_URL/TURSO_AUTH_TOKEN não configurados.")
    return libsql.connect(database=url, auth_token=token)


@st.cache_data(ttl=60, show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = connect()
    try:
        cur = conn.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in (cur.description or [])])
    finally:
        conn.close()


@st.cache_data(ttl=60, show_spinner=False)
def tables() -> set[str]:
    df = query("SELECT name FROM sqlite_master WHERE type='table'")
    return set(df["name"].astype(str)) if "name" in df else set()


def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{float(x)*100:.{d}f}%"


def units(x):
    return "—" if x is None or pd.isna(x) else f"{float(x):+.2f}u"


def euros(x, unit):
    return "—" if x is None or pd.isna(x) else f"€{float(x)*unit:+,.2f}".replace(",", " ")


def numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def close_group(row: pd.Series) -> str:
    over = pd.to_numeric(pd.Series([row.get("odd_over25_close")]), errors="coerce").iloc[0]
    under = pd.to_numeric(pd.Series([row.get("odd_under25_close")]), errors="coerce").iloc[0]
    if pd.isna(over) or pd.isna(under) or over <= 1 or under <= 1:
        return "Sem CLOSE"
    source = str(row.get("closing_source") or "").casefold()
    if "the odds api" in source:
        return "CLOSE pré-jogo"
    if "football-data" in source:
        return "CLOSE Football-Data"
    if "api-football" in source:
        return "Snapshot API-Football"
    return "Outra CLOSE"


BADGES = {
    "CLOSE pré-jogo": "🟢 CLOSE pré-jogo",
    "CLOSE Football-Data": "🟠 CLOSE Football-Data",
    "Snapshot API-Football": "🟡 Snapshot API-Football",
    "Sem CLOSE": "⚪ Sem CLOSE",
    "Outra CLOSE": "⚪ Outra CLOSE",
}


def prep_paper(df):
    if df.empty:
        return df
    out = numeric(df, [
        "odd_over25_open", "odd_under25_open", "odd_over25_close",
        "odd_under25_close", "clv_probability", "signal", "planned_stake",
        "profit", "result_over25", "closing_minutes_before_kickoff",
    ])
    out["status"] = out.get("status", "OPEN").fillna("OPEN").astype(str).str.upper()
    out["signal"] = out.get("signal", 0)
    out["clv_probability"] = out.get("clv_probability", np.nan)
    out["closing_source"] = out.get("closing_source", None)
    out["closing_minutes_before_kickoff"] = out.get("closing_minutes_before_kickoff", np.nan)
    valid = out["odd_over25_open"].gt(1) & out["odd_over25_close"].gt(1)
    out["clv_display"] = np.nan
    out.loc[valid, "clv_display"] = out.loc[valid, "odd_over25_open"] / out.loc[valid, "odd_over25_close"] - 1
    out["close_group"] = out.apply(close_group, axis=1)
    out["close_badge"] = out["close_group"].map(BADGES).fillna("⚪ Outra CLOSE")
    out["close_high_quality"] = out["close_group"].isin(["CLOSE pré-jogo", "CLOSE Football-Data"])
    d = out.get("fixture_date", pd.Series("", index=out.index)).astype(str)
    t = out.get("fixture_time", pd.Series("00:00", index=out.index)).fillna("00:00").astype(str)
    out["fixture_datetime"] = pd.to_datetime(d + " " + t, errors="coerce")
    return out


def prep_upcoming(df):
    if df.empty:
        return df
    out = numeric(df, ["odd_over25_open", "odd_over25_current", "active"])
    d = out.get("date", pd.Series("", index=out.index)).astype(str)
    t = out.get("time", pd.Series("00:00", index=out.index)).fillna("00:00").astype(str)
    out["fixture_datetime"] = pd.to_datetime(d + " " + t, errors="coerce")
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    out["minutes_to_kickoff"] = (out["fixture_datetime"] - now).dt.total_seconds() / 60
    observed = pd.to_datetime(out.get("current_observed_at", pd.Series(pd.NaT, index=out.index)), errors="coerce", utc=True)
    out["current_age_hours"] = (now - observed.dt.tz_convert(TZ).dt.tz_localize(None)).dt.total_seconds() / 3600
    return out


def result_badge(row):
    status = str(row.get("status") or "OPEN").strip().upper()
    if status == "VOID":
        return "⚪ VOID"
    if status != "RESOLVED":
        return "⏳"
    v = row.get("result_over25")
    return "—" if pd.isna(v) else ("✅ OVER" if int(v) == 1 else "❌ UNDER")


def paper_table(df):
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    out["Decisão"] = df["signal"].apply(lambda x: "🟢 SINAL" if int(x or 0) == 1 else "⚪ SEM SINAL")
    out["Estado"] = df["status"].map({"RESOLVED": "✅ RESOLVIDO", "OPEN": "🟡 OPEN", "VOID": "⚪ VOID / ADIADO"}).fillna(df["status"])
    out["Data"] = df.get("fixture_date", "")
    out["Hora"] = df.get("fixture_time", "")
    out["Liga"] = df.get("league", "")
    out["Jogo"] = df.get("home_team", "").astype(str) + " vs " + df.get("away_team", "").astype(str)
    out["OPEN"] = df.get("odd_over25_open", np.nan).round(3)
    out["CLOSE"] = df.get("odd_over25_close", np.nan).round(3)
    out["Tipo CLOSE"] = df["close_badge"]
    out["Min. antes"] = df.get("closing_minutes_before_kickoff", np.nan).round(0)
    out["Prob. CLV+ %"] = (df["clv_probability"] * 100).round(1)
    out["CLV %"] = (df["clv_display"] * 100).round(2)
    out["Resultado"] = df.apply(result_badge, axis=1)
    if "profit" in df:
        out["Lucro (u)"] = df["profit"].round(3)
    out["Fonte CLOSE"] = df.get("closing_source", "")
    return out.reset_index(drop=True)


def upcoming_table(df):
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    out["Data"] = df.get("date", "")
    out["Hora"] = df.get("time", "")
    out["Liga"] = df.get("league", "")
    out["Jogo"] = df.get("home_team", "").astype(str) + " vs " + df.get("away_team", "").astype(str)
    out["OPEN"] = df.get("odd_over25_open", np.nan).round(3)
    out["Atual"] = df.get("odd_over25_current", np.nan).round(3)
    out["Fonte"] = df.get("source", "")
    def cd(v):
        if pd.isna(v): return "—"
        if v < 0: return "iniciado"
        if v >= 60: return f"{int(v//60)}h {int(v%60)}m"
        return f"{max(0, int(v))} min"
    out["Faltam"] = df["minutes_to_kickoff"].apply(cd)
    return out.reset_index(drop=True)


def threshold_test(df):
    usable = df[df["clv_probability"].notna() & df["clv_display"].notna()]
    rows = []
    for t in SHADOW:
        pick = usable[usable["clv_probability"] >= t]
        n = len(pick)
        rows.append({
            "threshold": t,
            "Threshold": f"{t*100:.1f}%" if np.isclose(t, .525) else f"{t*100:.0f}%",
            "Tipo": "ATUAL" if np.isclose(t, THRESHOLD) else "SHADOW",
            "Jogos": n,
            "Cobertura %": round(100*n/len(usable), 1) if len(usable) else np.nan,
            "Prob. média %": round(100*pick["clv_probability"].mean(), 1) if n else np.nan,
            "CLV positivo %": round(100*(pick["clv_display"] > 0).mean(), 1) if n else np.nan,
            "CLV médio %": round(100*pick["clv_display"].mean(), 2) if n else np.nan,
            "CLV mediano %": round(100*pick["clv_display"].median(), 2) if n else np.nan,
        })
    return pd.DataFrame(rows)


def quality_metrics(df):
    usable = df[df["clv_probability"].notna() & df["clv_display"].notna()]
    if usable.empty:
        return np.nan, np.nan, np.nan
    y = (usable["clv_display"] > 0).astype(int).to_numpy()
    p = usable["clv_probability"].clip(1e-6, 1-1e-6).to_numpy()
    brier = float(np.mean((p-y)**2))
    logloss = float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))
    auc = np.nan
    if len(np.unique(y)) == 2:
        try: auc = float(roc_auc_score(y, p))
        except Exception: pass
    return brier, logloss, auc


try:
    tabs = tables()
except Exception as exc:
    st.error("Não foi possível ligar à Turso.")
    st.code(str(exc))
    st.stop()

paper = prep_paper(query("SELECT * FROM paper_trades ORDER BY fixture_date DESC, fixture_time DESC, id DESC")) if "paper_trades" in tabs else pd.DataFrame()
upcoming = prep_upcoming(query("SELECT * FROM upcoming_matches WHERE active=1 ORDER BY date,time,league,home_team")) if "upcoming_matches" in tabs else pd.DataFrame()
history = query("SELECT COUNT(*) total_matches, COUNT(DISTINCT league) total_leagues, MAX(date) latest_match FROM matches") if "matches" in tabs else pd.DataFrame()

st.sidebar.title("⚽ FootballAI")
page = st.sidebar.radio("Navegação", ["🏠 Resumo", "🎯 Sinais", "🧾 Paper Trades", "📅 Próximos Jogos", "📈 CLV & Performance", "⚙️ Estado do Sistema"], label_visibility="collapsed")
unit_value = st.sidebar.number_input("Valor por unidade (€)", 1.0, 1000.0, 10.0, 1.0)
leagues = sorted(paper["league"].dropna().astype(str).unique()) if len(paper) else []
selected = st.sidebar.multiselect("Filtrar ligas", leagues, default=[], placeholder="Todas as ligas")
if st.sidebar.button("🔄 Atualizar dados", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

pv, uv = paper.copy(), upcoming.copy()
if selected:
    if len(pv): pv = pv[pv["league"].isin(selected)]
    if len(uv): uv = uv[uv["league"].isin(selected)]

st.title("⚽ FootballAI Cloud")
st.caption("Paper Trade, odds, resultados e CLV — com qualidade das fontes separada.")

if page == "🏠 Resumo":
    if pv.empty:
        st.info("Ainda não existem Paper Trades.")
    else:
        signals = pv[pv["signal"] == 1]
        resolved = signals[signals["status"] == "RESOLVED"]
        profit = pd.to_numeric(resolved.get("profit", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        stake = pd.to_numeric(resolved.get("planned_stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        c = st.columns(4)
        c[0].metric("Paper Trades", len(pv)); c[1].metric("Sinais", len(signals))
        c[2].metric("OPEN", int((pv["status"]=="OPEN").sum())); c[3].metric("Resolvidos", int((pv["status"]=="RESOLVED").sum()))
        c = st.columns(4)
        c[0].metric("✅ CLOSE alta qualidade", int(pv["close_high_quality"].sum()))
        c[1].metric("🟡 Snapshots provisórias", int((pv["close_group"]=="Snapshot API-Football").sum()))
        c[2].metric("Lucro sinais", units(profit), euros(profit, unit_value))
        c[3].metric("ROI sinais", pct(profit/stake if stake > 0 else np.nan))
        st.subheader("🎯 Sinais mais recentes")
        st.dataframe(paper_table(signals.head(8)), use_container_width=True, hide_index=True)
        st.subheader("📅 Próximos jogos")
        future = uv[uv.get("minutes_to_kickoff", pd.Series(np.nan, index=uv.index)) >= 0].head(10)
        st.dataframe(upcoming_table(future), use_container_width=True, hide_index=True)

elif page == "🎯 Sinais":
    st.subheader("🎯 Sinais reais do modelo")
    st.caption("Threshold real congelado em 55%. A análise shadow não altera sinais.")
    signals = pv[pv["signal"] == 1].copy()
    mode = st.segmented_control("Estado", ["Todos", "OPEN", "RESOLVED", "VOID"], default="Todos")
    if mode in {"OPEN", "RESOLVED", "VOID"}: signals = signals[signals["status"] == mode]
    st.dataframe(paper_table(signals), use_container_width=True, hide_index=True)

elif page == "🧾 Paper Trades":
    st.subheader("🧾 Paper Trades")
    c1,c2,c3,c4 = st.columns(4)
    statuses = c1.multiselect("Estado", ["OPEN","RESOLVED","VOID"], default=[], placeholder="Todos")
    only_signals = c2.toggle("Só sinais")
    ctype = c3.selectbox("Tipo CLOSE", ["Todas","CLOSE pré-jogo","CLOSE Football-Data","Snapshot API-Football","Sem CLOSE","Outra CLOSE"])
    order = c4.selectbox("Ordenar", ["Mais recentes","Maior prob. CLV+","Maior CLV"])
    shown = pv.copy()
    if statuses: shown = shown[shown["status"].isin(statuses)]
    if only_signals: shown = shown[shown["signal"] == 1]
    if ctype != "Todas": shown = shown[shown["close_group"] == ctype]
    if order == "Maior prob. CLV+": shown = shown.sort_values("clv_probability", ascending=False)
    elif order == "Maior CLV": shown = shown.sort_values("clv_display", ascending=False)
    st.caption(f"{len(shown)} registos.")
    st.dataframe(paper_table(shown), use_container_width=True, hide_index=True)

elif page == "📅 Próximos Jogos":
    st.subheader("📅 Próximos jogos")
    future = uv[uv.get("minutes_to_kickoff", pd.Series(np.nan, index=uv.index)) >= 0].copy()
    horizon = st.selectbox("Horizonte", ["Todos","24 horas","3 dias","7 dias"])
    limits = {"24 horas":1440, "3 dias":4320, "7 dias":10080}
    if horizon in limits: future = future[future["minutes_to_kickoff"] <= limits[horizon]]
    st.dataframe(upcoming_table(future), use_container_width=True, hide_index=True)

elif page == "📈 CLV & Performance":
    st.subheader("📈 CLV & Performance")
    st.info("clv_probability = probabilidade prevista de obter CLV positivo; não é probabilidade de Over 2.5.")
    allc = pv[pv["clv_display"].notna()].copy()
    high = allc[allc["close_high_quality"]]
    pre = allc[allc["close_group"]=="CLOSE pré-jogo"]
    fd = allc[allc["close_group"]=="CLOSE Football-Data"]
    prov = allc[allc["close_group"]=="Snapshot API-Football"]
    c = st.columns(4)
    c[0].metric("CLOSE/snapshots", len(allc)); c[1].metric("✅ Alta qualidade", len(high))
    c[2].metric("🟢 The Odds API", len(pre)); c[3].metric("🟡 Provisórias", len(prov))
    choice = st.selectbox("Amostra para avaliar o modelo", ["Alta qualidade (recomendado)","The Odds API pré-jogo","Football-Data closing","Todas incluindo provisórias","Só snapshots API-Football"])
    if choice == "Alta qualidade (recomendado)": rows = high; st.success("Amostra recomendada para threshold e calibração.")
    elif choice == "The Odds API pré-jogo": rows = pre; st.success("Amostra prospetiva mais limpa.")
    elif choice == "Football-Data closing": rows = fd; st.info("Closing recuperada pós-jogo; a agregação pode diferir da OPEN.")
    elif choice == "Só snapshots API-Football": rows = prov; st.warning("Snapshots provisórias podem estar horas antes do kickoff.")
    else: rows = allc; st.warning("Vista mista. Usa 'Alta qualidade' para decisões do modelo.")
    c = st.columns(4)
    c[0].metric("Jogos", len(rows)); c[1].metric("CLV médio", pct(rows["clv_display"].mean() if len(rows) else np.nan))
    c[2].metric("CLV positivo", pct((rows["clv_display"]>0).mean() if len(rows) else np.nan)); c[3].metric("CLV mediano", pct(rows["clv_display"].median() if len(rows) else np.nan))
    brier, logloss, auc = quality_metrics(rows)
    c = st.columns(3)
    c[0].metric("Brier score", f"{brier:.3f}" if not pd.isna(brier) else "—")
    c[1].metric("Log loss", f"{logloss:.3f}" if not pd.isna(logloss) else "—")
    c[2].metric("AUC", f"{auc:.3f}" if not pd.isna(auc) else "—")
    if len(rows) < 50: st.warning("Amostra pequena: não alterar o threshold real.")
    elif len(rows) < 200: st.info("Já mostra tendências, mas ainda é cedo para mexer no threshold.")
    else: st.success("Amostra suficiente para uma avaliação inicial mais robusta.")
    if len(rows):
        bins=[-.001,.30,.35,.40,.45,.50,.55,1.001]; labels=["<30%","30–35%","35–40%","40–45%","45–50%","50–55%","55%+"]
        cal=rows.copy(); cal["Faixa"]=pd.cut(cal["clv_probability"],bins=bins,labels=labels,include_lowest=True); cal["CLV positivo"]=(cal["clv_display"]>0).astype(int)
        ct=(cal.dropna(subset=["Faixa"]).groupby("Faixa",observed=False).agg(Jogos=("id","count"),Prob=("clv_probability","mean"),Pos=("CLV positivo","mean"),CLV=("clv_display","mean")).reset_index())
        ct["Prob. prevista %"]=(ct["Prob"]*100).round(1); ct["CLV positivo %"]=(ct["Pos"]*100).round(1); ct["CLV médio %"]=(ct["CLV"]*100).round(2)
        st.markdown("#### Calibração por faixa")
        st.dataframe(ct[["Faixa","Jogos","Prob. prevista %","CLV positivo %","CLV médio %"]],use_container_width=True,hide_index=True)
        st.markdown("#### 🧪 Teste shadow de thresholds")
        tt=threshold_test(rows)
        st.dataframe(tt[["Threshold","Tipo","Jogos","Cobertura %","Prob. média %","CLV positivo %","CLV médio %","CLV mediano %"]],use_container_width=True,hide_index=True)
        st.bar_chart(tt.set_index("Threshold")[["CLV positivo %","CLV médio %"]])
    resolved = pv[(pv["signal"]==1)&(pv["status"]=="RESOLVED")]
    profit = pd.to_numeric(resolved.get("profit",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()
    stake = pd.to_numeric(resolved.get("planned_stake",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()
    c=st.columns(4)
    c[0].metric("Sinais resolvidos",len(resolved)); c[1].metric("Lucro",units(profit),euros(profit,unit_value))
    c[2].metric("ROI",pct(profit/stake if stake>0 else np.nan))
    c[3].metric("Over 2.5 acertado",pct(pd.to_numeric(resolved.get("result_over25",pd.Series(dtype=float)),errors="coerce").mean()))

else:
    st.subheader("⚙️ Estado do sistema")
    st.success("✅ Ligação à Turso ativa")
    summary = history.iloc[0] if len(history) else pd.Series(dtype=object)
    c=st.columns(4)
    c[0].metric("Jogos históricos", int(summary.get("total_matches") or 0))
    c[1].metric("Ligas históricas", int(summary.get("total_leagues") or 0))
    c[2].metric("Último histórico", str(summary.get("latest_match") or "—"))
    c[3].metric("Upcoming ativos", len(upcoming))
    c=st.columns(4)
    c[0].metric("🟢 The Odds API", int((paper["close_group"]=="CLOSE pré-jogo").sum()) if len(paper) else 0)
    c[1].metric("🟠 Football-Data", int((paper["close_group"]=="CLOSE Football-Data").sum()) if len(paper) else 0)
    c[2].metric("🟡 Provisórias", int((paper["close_group"]=="Snapshot API-Football").sum()) if len(paper) else 0)
    c[3].metric("⚪ Sem CLOSE", int((paper["close_group"]=="Sem CLOSE").sum()) if len(paper) else 0)
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    overdue = paper[(paper["status"]=="OPEN") & paper["fixture_datetime"].notna() & (paper["fixture_datetime"] < now-pd.Timedelta(hours=3))] if len(paper) else pd.DataFrame()
    stale = upcoming[upcoming["current_age_hours"].notna() & (upcoming["current_age_hours"]>24)] if len(upcoming) else pd.DataFrame()
    c=st.columns(3)
    c[0].metric("OPEN atrasados >3h",len(overdue)); c[1].metric("Upcoming stale >24h",len(stale)); c[2].metric("CLOSE alta qualidade",int(paper["close_high_quality"].sum()) if len(paper) else 0)
    if len(overdue): st.warning("Há jogos antigos ainda OPEN: verifica o workflow de resultados.")
    if len(stale): st.warning("Há upcoming sem atualização há mais de 24h.")
    if len(paper) and (paper["close_group"]=="Snapshot API-Football").any(): st.info("Snapshots API-Football são provisórias e são substituídas por Football-Data quando houver closing.")
    with st.expander("Tabelas disponíveis na Turso"): st.write(sorted(tabs))

st.divider()
st.caption("FootballAI · Paper trading apenas · " + datetime.now(TZ).strftime("Atualizado no ecrã em %d/%m/%Y às %H:%M"))
