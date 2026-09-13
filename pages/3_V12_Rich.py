from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import libsql
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="FootballAI · V12 Rich",
    page_icon="🧬",
    layout="wide",
)

LOCAL_TZ = ZoneInfo("Europe/Lisbon")
DECISIONS_TABLE = "agent_v12_rich_analyst_decisions"
RUNS_TABLE = "agent_v12_rich_analyst_runs"
MODEL_TABLE = "agent_v12_rich_model_registry"


def safe_text(value: Any, default: str = "—") -> str:
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_pct(value: Any, digits: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def setting(name: str) -> str:
    try:
        value = st.secrets[name]
    except Exception:
        value = os.getenv(name, "")
    return safe_text(value, "").strip()


def connect():
    url = setting("TURSO_DATABASE_URL")
    token = setting("TURSO_AUTH_TOKEN")
    if not url or not token:
        raise RuntimeError("TURSO_DATABASE_URL/TURSO_AUTH_TOKEN não configurados.")
    return libsql.connect(database=url, auth_token=token)


@st.cache_data(ttl=30, show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    connection = connect()
    try:
        cursor = connection.execute(sql, params)
        return pd.DataFrame(
            cursor.fetchall(),
            columns=[column[0] for column in (cursor.description or [])],
        )
    finally:
        connection.close()


@st.cache_data(ttl=30, show_spinner=False)
def table_names() -> set[str]:
    frame = query("SELECT name FROM sqlite_master WHERE type='table'")
    if frame.empty or "name" not in frame.columns:
        return set()
    return set(frame["name"].dropna().astype(str))


def load_data(tables: set[str]):
    decisions = pd.DataFrame()
    run = pd.DataFrame()
    model = pd.DataFrame()

    if DECISIONS_TABLE in tables:
        decisions = query(
            f"""
            SELECT * FROM {DECISIONS_TABLE}
            ORDER BY fixture_date DESC, fixture_time DESC, id DESC
            """
        )

    if RUNS_TABLE in tables:
        run = query(f"SELECT * FROM {RUNS_TABLE} ORDER BY id DESC LIMIT 1")

    if MODEL_TABLE in tables:
        model = query(
            f"SELECT * FROM {MODEL_TABLE} ORDER BY registered_at DESC LIMIT 1"
        )

    return decisions, run, model


def result_label(value: Any) -> str:
    if safe_int(value, -1) == 1:
        return "✅ Over"
    if safe_int(value, -1) == 0:
        return "❌ Under"
    return "⏳ Pendente"


def v12_reading(row: pd.Series) -> str:
    delta = safe_float(row.get("rich_market_delta"), 0.0)
    v10 = safe_text(row.get("v10_decision"), "")

    if delta >= 0.03:
        if v10 == "BET_SHADOW":
            return "🟢 V12 reforça"
        if v10 == "PASS":
            return "👀 V12 mais otimista"
        return "🟢 V12 > mercado"
    if delta <= -0.03:
        if v10 == "BET_SHADOW":
            return "⚠️ V12 discorda"
        if v10 == "PASS":
            return "🟢 V12 reforça PASS"
        return "🔻 V12 < mercado"
    return "⚪ Próximo do mercado"


def display_rows(frame: pd.DataFrame, include_result: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    out["Jogo"] = (
        out["home_team"].fillna("").astype(str)
        + " vs "
        + out["away_team"].fillna("").astype(str)
    )
    out["Leitura V12"] = out.apply(v12_reading, axis=1)
    if include_result:
        out["Resultado"] = out["result_over25"].apply(result_label)

    wanted = [
        "fixture_date",
        "fixture_time",
        "Jogo",
        "league",
        "v10_decision",
        "v10_vote_count",
        "v10_vote_signature",
        "rich_probability",
        "market_probability",
        "rich_market_delta",
        "odd_over25_open",
        "counterfactual_ev",
        "Leitura V12",
        "data_quality",
    ]
    if include_result:
        wanted += ["Resultado", "counterfactual_profit"]
    return out[[column for column in wanted if column in out.columns]]


def probability_metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, float | None]:
    if frame.empty or probability_column not in frame.columns:
        return {"brier": None, "log_loss": None}

    valid = frame[
        frame["result_over25"].isin([0, 1])
        & frame[probability_column].notna()
    ].copy()
    if valid.empty:
        return {"brier": None, "log_loss": None}

    y = pd.to_numeric(valid["result_over25"], errors="coerce").astype(float)
    p = pd.to_numeric(valid[probability_column], errors="coerce").clip(1e-9, 1 - 1e-9)
    brier = float(((p - y) ** 2).mean())
    log_loss = float((-(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log))).mean())
    return {"brier": brier, "log_loss": log_loss}


def group_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if frame.empty or group_column not in frame.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for group, subset in frame.groupby(group_column, dropna=False):
        subset = subset[subset["result_over25"].isin([0, 1])].copy()
        if subset.empty:
            continue
        rich = probability_metrics(subset, "rich_probability")
        market = probability_metrics(subset, "market_probability")
        rows.append(
            {
                "Grupo": safe_text(group),
                "Jogos": len(subset),
                "Over rate": float(subset["result_over25"].mean()),
                "P(V12) média": float(subset["rich_probability"].mean()),
                "P(Mercado) média": float(subset["market_probability"].mean()),
                "Brier V12": rich["brier"],
                "Brier Mercado": market["brier"],
                "Ganho Brier": (
                    market["brier"] - rich["brier"]
                    if market["brier"] is not None and rich["brier"] is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


COLUMN_CONFIG = {
    "fixture_date": st.column_config.TextColumn("Data"),
    "fixture_time": st.column_config.TextColumn("Hora"),
    "league": st.column_config.TextColumn("Liga"),
    "v10_decision": st.column_config.TextColumn("V10"),
    "v10_vote_count": st.column_config.NumberColumn("Votos", format="%d"),
    "v10_vote_signature": st.column_config.TextColumn("Assinatura"),
    "rich_probability": st.column_config.NumberColumn("P(V12)", format="%.1%%"),
    "market_probability": st.column_config.NumberColumn("P(Mercado)", format="%.1%%"),
    "rich_market_delta": st.column_config.NumberColumn("Δ V12-Mercado", format="%+.1%%"),
    "odd_over25_open": st.column_config.NumberColumn("Odd", format="%.2f"),
    "counterfactual_ev": st.column_config.NumberColumn("EV V12", format="%+.1%%"),
    "counterfactual_profit": st.column_config.NumberColumn("Lucro cf.", format="%+.2fu"),
    "data_quality": st.column_config.TextColumn("Dados"),
    "Over rate": st.column_config.NumberColumn("Over rate", format="%.1%%"),
    "P(V12) média": st.column_config.NumberColumn("P(V12) média", format="%.1%%"),
    "P(Mercado) média": st.column_config.NumberColumn("P(Mercado) média", format="%.1%%"),
    "Brier V12": st.column_config.NumberColumn("Brier V12", format="%.4f"),
    "Brier Mercado": st.column_config.NumberColumn("Brier mercado", format="%.4f"),
    "Ganho Brier": st.column_config.NumberColumn("Ganho Brier", format="%+.4f"),
}


st.title("🧬 V12 Rich Analyst")
st.caption(
    "Analista probabilístico independente com 73 features pré-jogo. Está em shadow: observa e mede; não substitui nem altera o Champion V10."
)

try:
    tables = table_names()
    decisions, run, model = load_data(tables)
except Exception as exc:
    st.error("Não foi possível carregar os dados da V12 Rich.")
    st.exception(exc)
    st.stop()

if decisions.empty:
    st.info("A V12 Rich ainda não tem previsões guardadas na base de dados.")
    st.stop()

run_row = run.iloc[0].to_dict() if not run.empty else {}
model_row = model.iloc[0].to_dict() if not model.empty else {}
today = datetime.now(LOCAL_TZ).date().isoformat()
today_rows = decisions[decisions["fixture_date"].astype(str) == today].copy()
resolved = decisions[decisions["result_over25"].isin([0, 1])].copy()
rich_metrics = probability_metrics(resolved, "rich_probability")
market_metrics = probability_metrics(resolved, "market_probability")

st.markdown("### Estado rápido")
a, b, c, d, e = st.columns(5)
a.metric("Analisados hoje", len(today_rows))
b.metric("Total congelado", len(decisions))
c.metric("Resolvidos", len(resolved))
d.metric("Versão", safe_text(decisions.iloc[0].get("analyst_version")))
e.metric("Features", safe_int(decisions.iloc[0].get("feature_count")))

if run_row:
    seen = safe_int(run_row.get("upcoming_seen"))
    inserted = safe_int(run_row.get("inserted"))
    skipped = safe_int(run_row.get("skipped_data"))
    timing = safe_int(run_row.get("skipped_timing"))
    st.caption(
        f"Última varredura: {seen} upcoming · {inserted} novos · {skipped} sem dados seguros · {timing} fora do timing pré-jogo."
    )

main_today, main_results, main_model = st.tabs(
    ["🎯 Hoje", "📈 Resultados", "🧠 Modelo"]
)

with main_today:
    st.subheader(f"Previsões de hoje · {today}")
    st.caption(
        "Δ V12-Mercado é apenas a diferença entre as duas probabilidades. A leitura usa ±3 pontos percentuais só para tornar divergências visíveis; não é um sinal de aposta."
    )
    if today_rows.empty:
        st.info("Ainda não há previsões V12 para hoje.")
    else:
        today_rows = today_rows.sort_values(
            ["v10_decision", "rich_market_delta", "fixture_time"],
            ascending=[True, False, True],
            na_position="last",
        )
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Jogos", len(today_rows))
        x2.metric(
            "V10 BET_SHADOW",
            int((today_rows["v10_decision"] == "BET_SHADOW").sum()),
        )
        x3.metric(
            "V12 ≥ mercado +3pp",
            int((pd.to_numeric(today_rows["rich_market_delta"], errors="coerce") >= 0.03).sum()),
        )
        x4.metric(
            "V12 ≤ mercado -3pp",
            int((pd.to_numeric(today_rows["rich_market_delta"], errors="coerce") <= -0.03).sum()),
        )
        st.dataframe(
            display_rows(today_rows, include_result=True),
            use_container_width=True,
            hide_index=True,
            column_config=COLUMN_CONFIG,
            height=min(760, 38 + len(today_rows) * 35),
        )

with main_results:
    st.subheader("Calibração prospectiva")
    if resolved.empty:
        st.info(
            "Ainda não há jogos V12 resolvidos. As probabilidades já estão congeladas; esta secção começa a preencher quando os resultados entrarem."
        )
    else:
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Resolvidos", len(resolved))
        r2.metric("Over real", safe_pct(resolved["result_over25"].mean()))
        r3.metric("Brier V12", f"{safe_float(rich_metrics['brier']):.4f}")
        r4.metric("Brier mercado", f"{safe_float(market_metrics['brier']):.4f}")
        brier_gain = (
            safe_float(market_metrics["brier"]) - safe_float(rich_metrics["brier"])
            if rich_metrics["brier"] is not None and market_metrics["brier"] is not None
            else None
        )
        r5.metric(
            "Ganho Brier",
            f"{brier_gain:+.4f}" if brier_gain is not None else "—",
            help="Positivo = V12 teve Brier menor (melhor) que o mercado no mesmo conjunto.",
        )

        ll_gain = (
            safe_float(market_metrics["log_loss"]) - safe_float(rich_metrics["log_loss"])
            if rich_metrics["log_loss"] is not None and market_metrics["log_loss"] is not None
            else None
        )
        st.caption(
            f"Log Loss V12: {safe_float(rich_metrics['log_loss']):.4f} · "
            f"Mercado: {safe_float(market_metrics['log_loss']):.4f} · "
            f"ganho: {ll_gain:+.4f}" if ll_gain is not None else "Log Loss ainda indisponível."
        )

        st.markdown("#### V12 dentro das decisões V10")
        grouped = group_metrics(resolved, "v10_decision")
        if not grouped.empty:
            st.dataframe(
                grouped,
                use_container_width=True,
                hide_index=True,
                column_config=COLUMN_CONFIG,
            )

        st.markdown("#### Jogos resolvidos")
        st.dataframe(
            display_rows(resolved.head(300), include_result=True),
            use_container_width=True,
            hide_index=True,
            column_config=COLUMN_CONFIG,
        )

        st.caption(
            "Lucro cf. é contrafactual: assume 1u no Over em todos os jogos mostrados. A V12 Rich não está a emitir apostas."
        )

with main_model:
    st.subheader("Versão congelada")
    if not model_row:
        st.info("O registo técnico do modelo ainda não está disponível.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Versão", safe_text(model_row.get("model_version")))
        m2.metric("Features", safe_int(model_row.get("feature_count")))
        m3.metric("Treino", safe_int(model_row.get("train_n")))
        m4.metric("Calibração", safe_int(model_row.get("calibration_n")))

        st.markdown(
            f"**Treino até:** {safe_text(model_row.get('base_train_end'))}  \n"
            f"**Calibração até:** {safe_text(model_row.get('calibration_end'))}  \n"
            f"**Fingerprint:** `{safe_text(model_row.get('training_fingerprint'))}`"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brier calibração", f"{safe_float(model_row.get('calibration_brier')):.4f}")
        c2.metric("Log Loss calibração", f"{safe_float(model_row.get('calibration_log_loss')):.4f}")
        c3.metric("AUC calibração", f"{safe_float(model_row.get('calibration_auc')):.4f}")
        c4.metric("ECE calibração", safe_pct(model_row.get("calibration_ece"), digits=2))

    st.divider()
    st.markdown("#### Última execução")
    if not run_row:
        st.info("Sem registo de execução V12.")
    else:
        u1, u2, u3, u4, u5 = st.columns(5)
        u1.metric("Histórico", safe_int(run_row.get("historical_matches")))
        u2.metric("Upcoming", safe_int(run_row.get("upcoming_seen")))
        u3.metric("Novos", safe_int(run_row.get("inserted")))
        u4.metric("Sem dados", safe_int(run_row.get("skipped_data")))
        u5.metric("Resolvidos agora", safe_int(run_row.get("resolved_updated")))
        st.caption(
            f"V10 links preenchidos: {safe_int(run_row.get('v10_links_backfilled'))} · "
            f"já congelados: {safe_int(run_row.get('already_frozen'))} · "
            f"timing: {safe_int(run_row.get('skipped_timing'))}."
        )

st.divider()
st.caption("FootballAI · V12 Rich Analyst · shadow only · Champion V10 inalterado")
