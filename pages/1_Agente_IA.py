from __future__ import annotations

import os
from typing import Any

import libsql
import pandas as pd
import streamlit as st

from agent.chat_assistant import answer_question, agent_snapshot


st.set_page_config(page_title="FootballAI · Agente IA", page_icon="🤖", layout="wide")


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


def safe_pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100:.1f}%"
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
        return pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in (cursor.description or [])])
    finally:
        connection.close()


@st.cache_data(ttl=30, show_spinner=False)
def table_names() -> set[str]:
    frame = query("SELECT name FROM sqlite_master WHERE type='table'")
    if frame.empty or "name" not in frame.columns:
        return set()
    return set(frame["name"].dropna().astype(str))


def load_agent_data(tables: set[str]):
    decisions = pd.DataFrame()
    policies = pd.DataFrame()
    learning = pd.DataFrame()
    performance = pd.DataFrame()

    if "agent_decisions" in tables:
        if "paper_trades" in tables:
            decisions = query(
                """
                SELECT d.*, p.status, p.result_over25, p.clv, p.closing_source
                FROM agent_decisions d
                LEFT JOIN paper_trades p ON p.id = d.paper_trade_id
                ORDER BY d.evaluated_at DESC, d.id DESC
                """
            )
        else:
            decisions = query("SELECT * FROM agent_decisions ORDER BY evaluated_at DESC, id DESC")

    if "agent_policies" in tables:
        policies = query(
            """
            SELECT * FROM agent_policies
            WHERE status='ACTIVE'
            ORDER BY COALESCE(activated_at, created_at) DESC, id DESC
            LIMIT 1
            """
        )

    if "agent_learning_runs" in tables:
        learning = query("SELECT * FROM agent_learning_runs ORDER BY id DESC LIMIT 1")

    if "agent_performance_runs" in tables:
        performance = query("SELECT * FROM agent_performance_runs ORDER BY id DESC LIMIT 1")

    return decisions, policies, learning, performance


st.title("🤖 Agente IA · FootballAI")
st.caption("Chat de análise ligado às decisões congeladas e aos dados guardados na Turso.")

try:
    tables = table_names()
    decisions, policies, learning, performance = load_agent_data(tables)
    snapshot = agent_snapshot(decisions, policies, learning, performance)
except Exception as exc:
    st.error("Não foi possível carregar os dados do agente.")
    st.exception(exc)
    st.stop()

policy = snapshot.get("policy") if isinstance(snapshot.get("policy"), dict) else {}
learner = snapshot.get("learner") if isinstance(snapshot.get("learner"), dict) else {}
counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}

cols = st.columns(4)
cols[0].metric("Política ativa", safe_text(policy.get("policy_version")))
cols[1].metric("Decisões", safe_int(snapshot.get("decision_total")))
cols[2].metric("QUALIFIED", safe_int(counts.get("QUALIFIED_SHADOW")))
cols[3].metric("Learner", safe_text(learner.get("decision")))

with st.expander("Estado técnico", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("CLV mínimo", safe_pct(policy.get("min_clv_probability")))
    c2.metric("EV mínimo", safe_pct(policy.get("min_expected_value")))
    c3.metric("Edge mínimo", safe_pct(policy.get("min_edge_market")))
    st.caption("O chat é apenas leitura/análise. Não altera thresholds nem executa apostas.")

with st.expander("🔎 Últimas decisões", expanded=False):
    if decisions.empty:
        st.info("Ainda não existem decisões para mostrar.")
    else:
        cols_to_show = [
            c for c in [
                "evaluated_at", "fixture_date", "league", "home_team", "away_team",
                "decision", "tier", "policy_version", "odd_over25_open",
                "clv_probability", "sports_probability", "expected_value",
                "sports_edge_market", "status", "result_over25", "clv",
            ] if c in decisions.columns
        ]
        st.dataframe(decisions[cols_to_show].head(25), use_container_width=True, hide_index=True)

if "agent_chat_messages_v2" not in st.session_state:
    st.session_state.agent_chat_messages_v2 = [
        {
            "role": "assistant",
            "content": "Estou ligado aos dados do FootballAI. Pergunta, por exemplo, **Como está o agente?**, **Qual é o threshold atual?** ou **Que ligas estão melhores?**",
        }
    ]

for message in st.session_state.agent_chat_messages_v2:
    role = safe_text(message.get("role"), "assistant")
    content = safe_text(message.get("content"), "")
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Fala com o Agente IA…")
if question:
    question_text = safe_text(question, "")
    st.session_state.agent_chat_messages_v2.append({"role": "user", "content": question_text})
    with st.chat_message("user"):
        st.markdown(question_text)

    try:
        answer = answer_question(question_text, decisions, policies, learning, performance)
        answer = safe_text(answer, "Não consegui gerar uma resposta.")
    except Exception as exc:
        answer = "O chat encontrou um erro ao analisar esta pergunta."
        with st.chat_message("assistant"):
            st.error(answer)
            st.exception(exc)
    else:
        with st.chat_message("assistant"):
            st.markdown(answer)

    st.session_state.agent_chat_messages_v2.append({"role": "assistant", "content": answer})

st.divider()
st.caption("FootballAI Agent Chat · shadow/paper only")
