from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import libsql
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
CURRENT_THRESHOLD = 0.55
SHADOW_THRESHOLDS = [0.40, 0.45, 0.50, 0.525, 0.55]

st.set_page_config(
    page_title="FootballAI Cloud",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container{padding-top:.8rem;padding-bottom:2rem;max-width:1500px}
    [data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:.75rem 1rem;background:rgba(128,128,128,.04)}
    [data-testid="stMetricLabel"]{font-weight:650}
    [data-testid="stMetricValue"]{font-size:1.55rem}
    @media(max-width:700px){
      .block-container{padding-left:.65rem;padding-right:.65rem;padding-top:.55rem}
      [data-testid="stMetricValue"]{font-size:1.25rem}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_setting(name: str) -> str:
    try:
        value = st.secrets[name]
    except Exception:
        value = os.getenv(name, "")
    return str(value).strip() if value is not None else ""


def connect_turso():
    url = get_setting("TURSO_DATABASE_URL")
    token = get_setting("TURSO_AUTH_TOKEN")
    if not url:
        raise RuntimeError("TURSO_DATABASE_URL não configurada.")
    if not token:
        raise RuntimeError("TURSO_AUTH_TOKEN não configurado.")
    return libsql.connect(database=url, auth_token=token)


@st.cache_data(ttl=60, show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = connect_turso()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        cols = [x[0] for x in (cur.description or [])]
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


@st.cache_data(ttl=60, show_spinner=False)
def list_tables() -> set[str]:
    df = query("SELECT name FROM sqlite_master WHERE type='table'")
    return set(df["name"].astype(str)) if "name" in df.columns else set()


def num(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}".replace(",", " ")


def pct(value, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def units(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.{decimals}f}u"


def euros(value, unit_value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"€{float(value) * unit_value:+,.2f}".replace(",", " ")


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def status_text(value) -> str:
    text = str(value or "OPEN").strip().upper()
    return text or "OPEN"


def classify_close(row: pd.Series) -> str:
    try:
        over = float(row.get("odd_over25_close"))
        under = float(row.get("odd_under25_close"))
        if over <= 1 or under <= 1:
            return "Sem CLOSE"
    except Exception:
        return "Sem CLOSE"

    source = str(row.get("closing_source") or "").casefold()

    if "the odds api" in source:
        return "CLOSE pré-jogo"
    if "football-data" in source:
        return "CLOSE Football-Data"
    if "api-football" in source:
        return "Snapshot API-Football"
    return "Outra CLOSE"


def close_badge(group: str) -> str:
    return {
        "CLOSE pré-jogo": "🟢 CLOSE pré-jogo",
        "CLOSE Football-Data": "🟠 CLOSE Football-Data",
        "Snapshot API-Football": "🟡 Snapshot API-Football",
        "Sem CLOSE": "⚪ Sem CLOSE",
        "Outra CLOSE": "⚪ Outra CLOSE",
    }.get(group, "⚪ Outra CLOSE")


def close_is_high_quality(group: str) -> bool:
    return group in {"CLOSE pré-jogo", "CLOSE Football-Data"}


def prepare_paper(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = to_numeric(
        df,
        [
            "odd_over25_open",
            "odd_under25_open",
            "odd_over25_close",
            "odd_under25_close",
            "clv_probability",
            "fair_probability_open",
            "signal",
            "planned_stake",
            "profit",
            "result_over25",
            "home_goals",
            "away_goals",
            "closing_minutes_before_kickoff",
        ],
    )

    out["status"] = (
        out["status"].apply(status_text)
        if "status" in out.columns
        else "OPEN"
    )
    out["signal"] = out.get("signal", 0)
    out["clv_probability"] = out.get("clv_probability", np.nan)
    out["closing_source"] = out.get("closing_source", None)
    out["closing_minutes_before_kickoff"] = out.get(
        "closing_minutes_before_kickoff",
        np.nan,
    )

    out["fixture_date_dt"] = pd.to_datetime(
        out.get("fixture_date", pd.Series("", index=out.index)),
        errors="coerce",
    )

    date_text = out.get(
        "fixture_date",
        pd.Series("", index=out.index),
    ).astype(str)
    time_text = out.get(
        "fixture_time",
        pd.Series("00:00", index=out.index),
    ).fillna("00:00").astype(str)

    out["fixture_datetime"] = pd.to_datetime(
        date_text + " " + time_text,
        errors="coerce",
    )

    valid = (
        out.get("odd_over25_open", pd.Series(np.nan, index=out.index)).gt(1)
        & out.get("odd_over25_close", pd.Series(np.nan, index=out.index)).gt(1)
    )

    out["clv_display"] = np.nan
    out.loc[valid, "clv_display"] = (
        out.loc[valid, "odd_over25_open"]
        / out.loc[valid, "odd_over25_close"]
        - 1
    )

    out["close_group"] = out.apply(classify_close, axis=1)
    out["close_badge"] = out["close_group"].apply(close_badge)
    out["close_high_quality"] = out["close_group"].apply(
        close_is_high_quality
    )
    return out


def prepare_upcoming(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = to_numeric(
        df,
        [
            "odd_over25_open",
            "odd_over25_current",
            "odd_under25_open",
            "odd_under25_current",
            "active",
        ],
    )

    dates = out.get("date", pd.Series("", index=out.index)).astype(str)
    times = out.get(
        "time",
        pd.Series("00:00", index=out.index),
    ).fillna("00:00").astype(str)

    out["fixture_datetime"] = pd.to_datetime(
        dates + " " + times,
        errors="coerce",
    )
    now = pd.Timestamp.now(tz=APP_TIMEZONE).tz_localize(None)
    out["minutes_to_kickoff"] = (
        out["fixture_datetime"] - now
    ).dt.total_seconds() / 60

    out["current_observed_dt"] = pd.to_datetime(
        out.get(
            "current_observed_at",
            pd.Series(pd.NaT, index=out.index),
        ),
        errors="coerce",
        utc=True,
    ).dt.tz_convert(APP_TIMEZONE).dt.tz_localize(None)

    out["current_age_hours"] = (
        now - out["current_observed_dt"]
    ).dt.total_seconds() / 3600
    return out


def result_badge(row: pd.Series) -> str:
    status = str(row.get("status") or "OPEN").strip().upper()
    if status == "VOID":
        return "⚪ VOID"
    if status != "RESOLVED":
        return "⏳"
    value = row.get("result_over25")
    if pd.isna(value):
        return "—"
    return "✅ OVER" if int(value) == 1 else "❌ UNDER"


def paper_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = pd.DataFrame(index=df.index)
    out["Decisão"] = df["signal"].apply(
        lambda x: "🟢 SINAL" if int(x or 0) == 1 else "⚪ SEM SINAL"
    )
    out["Estado"] = df["status"].map(
        {"RESOLVED": "✅ RESOLVIDO", "OPEN": "🟡 OPEN", "VOID": "⚪ VOID / ADIADO"}
    ).fillna(df["status"])
    out["Data"] = df.get("fixture_date", "")
    out["Hora"] = df.get("fixture_time", "")
    out["Liga"] = df.get("league", "")
    out["Jogo"] = (
        df.get("home_team", "").astype(str)
        + " vs "
        + df.get("away_team", "").astype(str)
    )
    out["OPEN"] = df.get("odd_over25_open", np.nan).round(3)
    out["CLOSE"] = df.get("odd_over25_close", np.nan).round(3)
    out["Tipo CLOSE"] = df["close_badge"]
    out["Min. antes"] = df.get(
        "closing_minutes_before_kickoff",
        np.nan,
    ).round(0)
    out["Prob. CLV+ %"] = (df["clv_probability"] * 100).round(1)
    out["CLV %"] = (df["clv_display"] * 100).round(2)
    out["Resultado"] = df.apply(result_badge, axis=1)
    if "profit" in df.columns:
        out["Lucro (u)"] = df["profit"].round(3)
    out["Fonte CLOSE"] = df.get("closing_source", "")
    return out.reset_index(drop=True)


def upcoming_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = pd.DataFrame(index=df.index)
    out["Data"] = df.get("date", "")
    out["Hora"] = df.get("time", "")
    out["Liga"] = df.get("league", "")
    out["Jogo"] = (
        df.get("home_team", "").astype(str)
        + " vs "
        + df.get("away_team", "").astype(str)
    )
    out["OPEN"] = df.get("odd_over25_open", np.nan).round(3)
    out["Atual"] = df.get("odd_over25_current", np.nan).round(3)
    out["Fonte"] = df.get("source", "")

    def countdown(value):
        if pd.isna(value):
            return "—"
        if value < 0:
            return "iniciado"
        if value >= 60:
            return f"{int(value // 60)}h {int(value % 60)}m"
        return f"{max(0, int(value))} min"

    out["Faltam"] = df["minutes_to_kickoff"].apply(countdown)
    return out.reset_index(drop=True)


def threshold_test(close_rows: pd.DataFrame) -> pd.DataFrame:
    usable = close_rows[
        close_rows["clv_probability"].notna()
        & close_rows["clv_display"].notna()
    ].copy()

    rows = []
    for threshold in SHADOW_THRESHOLDS:
        picked = usable[usable["clv_probability"] >= threshold]
        n = len(picked)
        rows.append(
            {
                "threshold": threshold,
                "Threshold": (
                    f"{threshold * 100:.1f}%"
                    if np.isclose(threshold, 0.525)
                    else f"{threshold * 100:.0f}%"
                ),
                "Tipo": "ATUAL" if np.isclose(threshold, CURRENT_THRESHOLD) else "SHADOW",
                "Jogos": n,
                "Cobertura %": round(100 * n / len(usable), 1) if len(usable) else np.nan,
                "CLV positivo %": round(100 * (picked["clv_display"] > 0).mean(), 1) if n else np.nan,
                "CLV médio %": round(100 * picked["clv_display"].mean(), 2) if n else np.nan,
                "CLV mediano %": round(100 * picked["clv_display"].median(), 2) if n else np.nan,
                "Prob. média %": round(100 * picked["clv_probability"].mean(), 1) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def model_quality_metrics(close_rows: pd.DataFrame) -> dict[str, float]:
    usable = close_rows[
        close_rows["clv_probability"].notna()
        & close_rows["clv_display"].notna()
    ].copy()

    if usable.empty:
        return {"brier": np.nan, "log_loss": np.nan, "auc": np.nan}

    y = (usable["clv_display"] > 0).astype(int).to_numpy()
    p = usable["clv_probability"].clip(1e-6, 1 - 1e-6).to_numpy()
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    auc = np.nan
    if len(np.unique(y)) == 2:
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = np.nan

    return {"brier": brier, "log_loss": log_loss, "auc": auc}


try:
    tables = list_tables()
except Exception as exc:
    st.error("Não foi possível ligar à Turso.")
    st.code(str(exc))
    st.stop()

paper = (
    prepare_paper(
        query(
            """
            SELECT *
            FROM paper_trades
            ORDER BY fixture_date DESC, fixture_time DESC, id DESC
            """
        )
    )
    if "paper_trades" in tables
    else pd.DataFrame()
)

upcoming = (
    prepare_upcoming(
        query(
            """
            SELECT *
            FROM upcoming_matches
            WHERE active = 1
            ORDER BY date, time, league, home_team
            """
        )
    )
    if "upcoming_matches" in tables
    else pd.DataFrame()
)

history = (
    query(
        """
        SELECT
            COUNT(*) AS total_matches,
            COUNT(DISTINCT league) AS total_leagues,
            MAX(date) AS latest_match
        FROM matches
        """
    )
    if "matches" in tables
    else pd.DataFrame()
)

st.sidebar.title("⚽ FootballAI")
st.sidebar.caption("Cloud dashboard")
page = st.sidebar.radio(
    "Navegação",
    [
        "🏠 Resumo",
        "🎯 Sinais",
        "🧾 Paper Trades",
        "📅 Próximos Jogos",
        "📈 CLV & Performance",
        "⚙️ Estado do Sistema",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
unit_value = st.sidebar.number_input(
    "Valor por unidade (€)",
    min_value=1.0,
    max_value=1000.0,
    value=10.0,
    step=1.0,
)

leagues = (
    sorted(paper["league"].dropna().astype(str).unique())
    if not paper.empty and "league" in paper.columns
    else []
)
selected_leagues = st.sidebar.multiselect(
    "Filtrar ligas",
    options=leagues,
    default=[],
    placeholder="Todas as ligas",
)

if st.sidebar.button("🔄 Atualizar dados", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Cache: 60 segundos.")

paper_view = paper.copy()
upcoming_view = upcoming.copy()
if selected_leagues:
    if not paper_view.empty:
        paper_view = paper_view[paper_view["league"].isin(selected_leagues)]
    if not upcoming_view.empty:
        upcoming_view = upcoming_view[upcoming_view["league"].isin(selected_leagues)]

st.title("⚽ FootballAI Cloud")
st.caption(
    "Paper Trade, closing odds, resultados e CLV — com qualidade das fontes separada."
)

if page == "🏠 Resumo":
    if paper_view.empty:
        st.info("Ainda não existem Paper Trades.")
    else:
        signals = paper_view[paper_view["signal"] == 1]
        resolved_signals = signals[signals["status"] == "RESOLVED"]

        profit = pd.to_numeric(
            resolved_signals.get("profit", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).sum()
        stake = pd.to_numeric(
            resolved_signals.get("planned_stake", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).sum()

        high_quality = paper_view[paper_view["close_high_quality"]]
        provisional = paper_view[
            paper_view["close_group"] == "Snapshot API-Football"
        ]

        r1 = st.columns(4)
        r1[0].metric("Paper Trades", len(paper_view))
        r1[1].metric("Sinais", len(signals))
        r1[2].metric("OPEN", int((paper_view["status"] == "OPEN").sum()))
        r1[3].metric("Resolvidos", int((paper_view["status"] == "RESOLVED").sum()))

        r2 = st.columns(4)
        r2[0].metric("✅ CLOSE alta qualidade", len(high_quality))
        r2[1].metric("🟡 Snapshots provisórias", len(provisional))
        r2[2].metric("Lucro sinais", units(profit), euros(profit, unit_value))
        r2[3].metric("ROI sinais", pct(profit / stake if stake > 0 else np.nan))

        st.subheader("🎯 Sinais mais recentes")
        st.dataframe(
            paper_table(signals.head(8)),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("📅 Próximos jogos")
        future = upcoming_view[
            upcoming_view.get(
                "minutes_to_kickoff",
                pd.Series(np.nan, index=upcoming_view.index),
            ) >= 0
        ].head(10)
        st.dataframe(
            upcoming_table(future),
            use_container_width=True,
            hide_index=True,
        )

elif page == "🎯 Sinais":
    st.subheader("🎯 Sinais reais do modelo")
    st.caption(
        "Threshold real congelado em 55%. A análise shadow não altera sinais."
    )
    signals = paper_view[paper_view["signal"] == 1].copy()

    mode = st.segmented_control(
        "Estado",
        ["Todos", "OPEN", "RESOLVED"],
        default="Todos",
    )
    if mode in {"OPEN", "RESOLVED"}:
        signals = signals[signals["status"] == mode]

    st.dataframe(
        paper_table(signals),
        use_container_width=True,
        hide_index=True,
    )

elif page == "🧾 Paper Trades":
    st.subheader("🧾 Paper Trades")

    c1, c2, c3, c4 = st.columns(4)
    statuses = c1.multiselect(
        "Estado",
        ["OPEN", "RESOLVED"],
        default=[],
        placeholder="Todos",
    )
    only_signals = c2.toggle("Só sinais")
    close_mode = c3.selectbox(
        "Tipo CLOSE",
        [
            "Todas",
            "CLOSE pré-jogo",
            "CLOSE Football-Data",
            "Snapshot API-Football",
            "Sem CLOSE",
            "Outra CLOSE",
        ],
    )
    sort_mode = c4.selectbox(
        "Ordenar",
        ["Mais recentes", "Maior prob. CLV+", "Maior CLV"],
    )

    shown = paper_view.copy()
    if statuses:
        shown = shown[shown["status"].isin(statuses)]
    if only_signals:
        shown = shown[shown["signal"] == 1]
    if close_mode != "Todas":
        shown = shown[shown["close_group"] == close_mode]

    if sort_mode == "Maior prob. CLV+":
        shown = shown.sort_values("clv_probability", ascending=False)
    elif sort_mode == "Maior CLV":
        shown = shown.sort_values("clv_display", ascending=False)

    st.caption(f"{len(shown)} registos.")
    st.dataframe(
        paper_table(shown),
        use_container_width=True,
        hide_index=True,
    )

elif page == "📅 Próximos Jogos":
    st.subheader("📅 Próximos jogos")
    future = upcoming_view[
        upcoming_view.get(
            "minutes_to_kickoff",
            pd.Series(np.nan, index=upcoming_view.index),
        ) >= 0
    ].copy()

    horizon = st.selectbox(
        "Horizonte",
        ["Todos", "24 horas", "3 dias", "7 dias"],
    )
    limits = {
        "24 horas": 24 * 60,
        "3 dias": 3 * 24 * 60,
        "7 dias": 7 * 24 * 60,
    }
    if horizon in limits:
        future = future[future["minutes_to_kickoff"] <= limits[horizon]]

    st.dataframe(
        upcoming_table(future),
        use_container_width=True,
        hide_index=True,
    )

elif page == "📈 CLV & Performance":
    st.subheader("📈 CLV & Performance")
    st.info(
        "clv_probability = probabilidade prevista de obter CLV positivo. "
        "Não é a probabilidade do jogo terminar Over 2.5."
    )

    all_close = paper_view[paper_view["clv_display"].notna()].copy()
    high_quality = all_close[all_close["close_high_quality"]].copy()
    pre = all_close[
        all_close["close_group"] == "CLOSE pré-jogo"
    ].copy()
    football_data = all_close[
        all_close["close_group"] == "CLOSE Football-Data"
    ].copy()
    provisional = all_close[
        all_close["close_group"] == "Snapshot API-Football"
    ].copy()

    m = st.columns(4)
    m[0].metric("CLOSE/snapshots", len(all_close))
    m[1].metric("✅ Alta qualidade", len(high_quality))
    m[2].metric("🟢 The Odds API", len(pre))
    m[3].metric("🟡 Provisórias", len(provisional))

    quality = st.selectbox(
        "Amostra para avaliar o modelo",
        [
            "Alta qualidade (recomendado)",
            "The Odds API pré-jogo",
            "Football-Data closing",
            "Todas incluindo provisórias",
            "Só snapshots API-Football",
        ],
    )

    if quality == "Alta qualidade (recomendado)":
        close_rows = high_quality
        st.success(
            "Esta é a amostra recomendada para avaliar threshold e calibração."
        )
    elif quality == "The Odds API pré-jogo":
        close_rows = pre
        st.success("Amostra prospetiva mais limpa: captura antes do kickoff.")
    elif quality == "Football-Data closing":
        close_rows = football_data
        st.info(
            "Closing recuperada pós-jogo. Pode ter agregação de casas diferente da OPEN."
        )
    elif quality == "Só snapshots API-Football":
        close_rows = provisional
        st.warning(
            "Snapshot provisória. Pode estar várias horas antes do kickoff e não deve ser usada sozinha para decidir o threshold."
        )
    else:
        close_rows = all_close
        if len(provisional):
            st.warning(
                "Esta vista mistura snapshots provisórias com closing odds. Usa a amostra de alta qualidade para decisões do modelo."
            )

    cm = st.columns(4)
    cm[0].metric("Jogos", len(close_rows))
    cm[1].metric(
        "CLV médio",
        pct(close_rows["clv_display"].mean() if len(close_rows) else np.nan),
    )
    cm[2].metric(
        "CLV positivo",
        pct((close_rows["clv_display"] > 0).mean() if len(close_rows) else np.nan),
    )
    cm[3].metric(
        "CLV mediano",
        pct(close_rows["clv_display"].median() if len(close_rows) else np.nan),
    )

    quality_metrics = model_quality_metrics(close_rows)
    qm = st.columns(3)
    qm[0].metric(
        "Brier score",
        f"{quality_metrics['brier']:.3f}" if not pd.isna(quality_metrics["brier"]) else "—",
    )
    qm[1].metric(
        "Log loss",
        f"{quality_metrics['log_loss']:.3f}" if not pd.isna(quality_metrics["log_loss"]) else "—",
    )
    qm[2].metric(
        "AUC",
        f"{quality_metrics['auc']:.3f}" if not pd.isna(quality_metrics["auc"]) else "—",
    )

    if len(close_rows) < 50:
        st.warning(
            "A amostra selecionada ainda é pequena. Não alteres o threshold real com base nestes números."
        )
    elif len(close_rows) < 200:
        st.info(
            "Já dá para observar tendências, mas ainda convém acumular mais dados antes de alterar o threshold real."
        )
    else:
        st.success(
            "A amostra já permite uma avaliação inicial mais robusta. Verifica também a distribuição por threshold e liga."
        )

    if len(close_rows):
        bins = [-0.001, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 1.001]
        labels = [
            "<30%",
            "30–35%",
            "35–40%",
            "40–45%",
            "45–50%",
            "50–55%",
            "55%+",
        ]

        cal = close_rows.copy()
        cal["Faixa"] = pd.cut(
            cal["clv_probability"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        cal["CLV positivo"] = (cal["clv_display"] > 0).astype(int)

        cal_table = (
            cal.dropna(subset=["Faixa"])
            .groupby("Faixa", observed=False)
            .agg(
                Jogos=("id", "count"),
                Prob=("clv_probability", "mean"),
                Pos=("CLV positivo", "mean"),
                CLV=("clv_display", "mean"),
            )
            .reset_index()
        )
        cal_table["Prob. prevista %"] = (cal_table["Prob"] * 100).round(1)
        cal_table["CLV positivo %"] = (cal_table["Pos"] * 100).round(1)
        cal_table["CLV médio %"] = (cal_table["CLV"] * 100).round(2)

        st.markdown("#### Calibração por faixa")
        st.dataframe(
            cal_table[
                [
                    "Faixa",
                    "Jogos",
                    "Prob. prevista %",
                    "CLV positivo %",
                    "CLV médio %",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 🧪 Teste shadow de thresholds")
        st.caption(
            "Compara 40%, 45%, 50%, 52.5% e 55% retrospetivamente. O threshold real continua congelado em 55%."
        )

        tt = threshold_test(close_rows)
        st.dataframe(
            tt[
                [
                    "Threshold",
                    "Tipo",
                    "Jogos",
                    "Cobertura %",
                    "Prob. média %",
                    "CLV positivo %",
                    "CLV médio %",
                    "CLV mediano %",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            tt.set_index("Threshold")[["CLV positivo %", "CLV médio %"]]
        )

        row50 = tt[np.isclose(tt["threshold"], 0.50)].iloc[0]
        row55 = tt[np.isclose(tt["threshold"], 0.55)].iloc[0]
        c = st.columns(4)
        c[0].metric("Jogos ≥50%", int(row50["Jogos"]))
        c[1].metric(
            "CLV+ ≥50%",
            f"{row50['CLV positivo %']:.1f}%" if not pd.isna(row50["CLV positivo %"]) else "—",
        )
        c[2].metric("Jogos ≥55%", int(row55["Jogos"]))
        c[3].metric(
            "CLV+ ≥55%",
            f"{row55['CLV positivo %']:.1f}%" if not pd.isna(row55["CLV positivo %"]) else "—",
        )

        st.info(
            "Para mudar o threshold procura estabilidade, não o melhor resultado de uma faixa com poucos jogos. A prioridade é acumular CLOSE de alta qualidade."
        )

    resolved_signals = paper_view[
        (paper_view["signal"] == 1)
        & (paper_view["status"] == "RESOLVED")
    ].copy()

    st.markdown("#### Resultado dos sinais reais")
    profit_series = pd.to_numeric(
        resolved_signals.get("profit", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    stake_series = pd.to_numeric(
        resolved_signals.get("planned_stake", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    profit = profit_series.sum()
    stake = stake_series.sum()

    sm = st.columns(4)
    sm[0].metric("Sinais resolvidos", len(resolved_signals))
    sm[1].metric("Lucro", units(profit), euros(profit, unit_value))
    sm[2].metric("ROI", pct(profit / stake if stake > 0 else np.nan))
    sm[3].metric(
        "Over 2.5 acertado",
        pct(
            pd.to_numeric(
                resolved_signals.get("result_over25", pd.Series(dtype=float)),
                errors="coerce",
            ).mean()
        ),
    )

elif page == "⚙️ Estado do Sistema":
    st.subheader("⚙️ Estado do sistema")
    st.success("✅ Ligação à Turso ativa")

    summary = history.iloc[0] if not history.empty else pd.Series(dtype=object)
    m = st.columns(4)
    m[0].metric("Jogos históricos", num(summary.get("total_matches")))
    m[1].metric("Ligas históricas", num(summary.get("total_leagues")))
    m[2].metric("Último histórico", str(summary.get("latest_match") or "—"))
    m[3].metric("Upcoming ativos", len(upcoming))

    st.markdown("#### Cobertura das closing odds")
    pre_count = int((paper["close_group"] == "CLOSE pré-jogo").sum()) if len(paper) else 0
    fd_count = int((paper["close_group"] == "CLOSE Football-Data").sum()) if len(paper) else 0
    provisional_count = int((paper["close_group"] == "Snapshot API-Football").sum()) if len(paper) else 0
    missing_count = int((paper["close_group"] == "Sem CLOSE").sum()) if len(paper) else 0

    c = st.columns(4)
    c[0].metric("🟢 The Odds API", pre_count)
    c[1].metric("🟠 Football-Data", fd_count)
    c[2].metric("🟡 Provisórias", provisional_count)
    c[3].metric("⚪ Sem CLOSE", missing_count)

    st.markdown("#### Saúde operacional")
    now_naive = pd.Timestamp.now(tz=APP_TIMEZONE).tz_localize(None)

    overdue_open = pd.DataFrame()
    if len(paper):
        overdue_open = paper[
            (paper["status"] == "OPEN")
            & paper["fixture_datetime"].notna()
            & (paper["fixture_datetime"] < now_naive - pd.Timedelta(hours=3))
        ]

    stale_upcoming = pd.DataFrame()
    if len(upcoming):
        stale_upcoming = upcoming[
            upcoming["current_age_hours"].notna()
            & (upcoming["current_age_hours"] > 24)
        ]

    s = st.columns(3)
    s[0].metric("OPEN atrasados >3h", len(overdue_open))
    s[1].metric("Upcoming stale >24h", len(stale_upcoming))
    s[2].metric(
        "CLOSE alta qualidade",
        int(paper["close_high_quality"].sum()) if len(paper) else 0,
    )

    if len(overdue_open):
        st.warning(
            "Existem jogos cujo kickoff passou há mais de 3 horas e continuam OPEN. Verifica o workflow de resultados."
        )
    if len(stale_upcoming):
        st.warning("Existem upcoming com observação atual há mais de 24 horas.")
    if provisional_count:
        st.info(
            "Snapshots API-Football são provisórias. O workflow de resultados tenta substituí-las por Football-Data closing após o jogo."
        )

    with st.expander("Tabelas disponíveis na Turso"):
        st.write(sorted(tables))

    st.caption(
        "A dashboard é apenas de leitura. Não altera previsões, odds, resultados ou thresholds."
    )

st.divider()
st.caption(
    "FootballAI · Paper trading apenas · "
    + datetime.now(APP_TIMEZONE).strftime(
        "Atualizado no ecrã em %d/%m/%Y às %H:%M"
    )
)
