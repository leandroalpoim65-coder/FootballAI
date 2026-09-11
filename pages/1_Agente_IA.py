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


def safe_num(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
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
    memory = pd.DataFrame()

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

    if "agent_memory_runs" in tables:
        memory = query("SELECT * FROM agent_memory_runs ORDER BY id DESC LIMIT 1")

    return decisions, policies, learning, performance, memory


def load_v10_data(tables: set[str]) -> pd.DataFrame:
    if "agent_v10_shadow_decisions" not in tables:
        return pd.DataFrame()
    return query(
        """
        SELECT *
        FROM agent_v10_shadow_decisions
        ORDER BY fixture_date DESC, fixture_time DESC, id DESC
        """
    )


def v10_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "total": 0,
            "bets": 0,
            "resolved": 0,
            "wins": 0,
            "hit": None,
            "profit": 0.0,
            "roi": None,
        }

    bets = frame[frame["decision"] == "BET_SHADOW"] if "decision" in frame.columns else pd.DataFrame()
    resolved = (
        bets[bets["result_over25"].isin([0, 1])]
        if not bets.empty and "result_over25" in bets.columns
        else pd.DataFrame()
    )
    wins = int(resolved["result_over25"].sum()) if not resolved.empty else 0
    profit = float(resolved["profit"].fillna(0).sum()) if not resolved.empty and "profit" in resolved.columns else 0.0
    return {
        "total": len(frame),
        "bets": len(bets),
        "resolved": len(resolved),
        "wins": wins,
        "hit": wins / len(resolved) if len(resolved) else None,
        "profit": profit,
        "roi": profit / len(resolved) if len(resolved) else None,
    }


st.title("🤖 Agente IA · FootballAI")
st.caption("Chat de análise ligado às decisões congeladas e aos dados guardados na Turso.")

try:
    tables = table_names()
    decisions, policies, learning, performance, memory = load_agent_data(tables)
    v10 = load_v10_data(tables)
    snapshot = agent_snapshot(decisions, policies, learning, performance)
except Exception as exc:
    st.error("Não foi possível carregar os dados do agente.")
    st.exception(exc)
    st.stop()

policy = snapshot.get("policy") if isinstance(snapshot.get("policy"), dict) else {}
learner = snapshot.get("learner") if isinstance(snapshot.get("learner"), dict) else {}
counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
memory_row = memory.iloc[0].to_dict() if not memory.empty else {}
v10_summary = v10_metrics(v10)

cols = st.columns(4)
cols[0].metric("Política ativa V3", safe_text(policy.get("policy_version")))
cols[1].metric("Decisões V3", safe_int(snapshot.get("decision_total")))
cols[2].metric("QUALIFIED V3", safe_int(counts.get("QUALIFIED_SHADOW")))
cols[3].metric("Learner V3", safe_text(learner.get("decision")))

with st.expander("🏆 Champion V10 · Shadow prospectivo", expanded=True):
    if v10.empty:
        st.info("Ainda não existem decisões do Champion V10 para mostrar.")
    else:
        champion_version = safe_text(v10.iloc[0].get("champion_version"))
        st.caption(
            f"Champion congelado: **{champion_version}** · Regra 3/4 votos + EV ≥ 0 · "
            "CLV é apenas diagnóstico, não é gate."
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Decisões", v10_summary["total"])
        m2.metric("BET_SHADOW", v10_summary["bets"])
        m3.metric("BET resolvidas", v10_summary["resolved"])
        m4.metric("Hit rate", safe_pct(v10_summary["hit"]))
        m5.metric("Lucro", f"{v10_summary['profit']:.2f}u")
        m6.metric("ROI", safe_pct(v10_summary["roi"]))

        tab_bets, tab_all = st.tabs(["🎯 BET_SHADOW", "📋 Todas as decisões"])

        with tab_bets:
            bet_rows = v10[v10["decision"] == "BET_SHADOW"].copy()
            if bet_rows.empty:
                st.info("Ainda não há BET_SHADOW.")
            else:
                bet_cols = [
                    c for c in [
                        "evaluated_at", "fixture_date", "fixture_time", "league",
                        "home_team", "away_team", "decision", "vote_count",
                        "odd_over25_open", "goal_score", "attack_score", "ml_score",
                        "value_score", "goal_vote", "attack_vote", "ml_vote", "value_vote",
                        "sports_probability", "expected_value", "sports_edge_market",
                        "clv_probability", "result_over25", "profit",
                    ] if c in bet_rows.columns
                ]
                st.dataframe(
                    bet_rows[bet_cols].head(100),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "goal_score": st.column_config.NumberColumn("Goal", format="%.1f"),
                        "attack_score": st.column_config.NumberColumn("Attack", format="%.1f"),
                        "ml_score": st.column_config.NumberColumn("ML", format="%.1f"),
                        "value_score": st.column_config.NumberColumn("Value", format="%.1f"),
                        "sports_probability": st.column_config.NumberColumn("Prob. ML", format="%.1%%"),
                        "expected_value": st.column_config.NumberColumn("EV", format="%.1%%"),
                        "sports_edge_market": st.column_config.NumberColumn("Edge", format="%.1%%"),
                        "clv_probability": st.column_config.NumberColumn("CLV prob.", format="%.1%%"),
                        "odd_over25_open": st.column_config.NumberColumn("Odd", format="%.2f"),
                        "profit": st.column_config.NumberColumn("Lucro", format="%.2fu"),
                    },
                )

        with tab_all:
            all_cols = [
                c for c in [
                    "evaluated_at", "fixture_date", "fixture_time", "league",
                    "home_team", "away_team", "decision", "vote_count",
                    "odd_over25_open", "goal_score", "attack_score", "ml_score",
                    "value_score", "sports_probability", "expected_value",
                    "sports_edge_market", "clv_probability", "result_over25", "profit",
                ] if c in v10.columns
            ]
            st.dataframe(
                v10[all_cols].head(200),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "goal_score": st.column_config.NumberColumn("Goal", format="%.1f"),
                    "attack_score": st.column_config.NumberColumn("Attack", format="%.1f"),
                    "ml_score": st.column_config.NumberColumn("ML", format="%.1f"),
                    "value_score": st.column_config.NumberColumn("Value", format="%.1f"),
                    "sports_probability": st.column_config.NumberColumn("Prob. ML", format="%.1%%"),
                    "expected_value": st.column_config.NumberColumn("EV", format="%.1%%"),
                    "sports_edge_market": st.column_config.NumberColumn("Edge", format="%.1%%"),
                    "clv_probability": st.column_config.NumberColumn("CLV prob.", format="%.1%%"),
                    "odd_over25_open": st.column_config.NumberColumn("Odd", format="%.2f"),
                    "profit": st.column_config.NumberColumn("Lucro", format="%.2fu"),
                },
            )

        st.caption(
            "As decisões V10 são congeladas antes do jogo. Quando o resultado fica disponível, "
            "apenas resultado e lucro são acrescentados; os scores e a decisão original não são alterados."
        )

with st.expander("🧠 Memória V5", expanded=False):
    if not memory_row:
        st.info("A memória V5 ainda não teve uma execução cloud. O primeiro run cria este resumo.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Histórico", safe_int(memory_row.get("historical_rows")))
        m2.metric("Resolvidos", safe_int(memory_row.get("resolved_rows")))
        m3.metric("OPEN sem label", safe_int(memory_row.get("open_rows")))
        m4.metric("Estado", safe_text(memory_row.get("state")))
        c1, c2, c3 = st.columns(3)
        c1.metric("ROI resolvido", safe_pct(memory_row.get("resolved_roi")))
        c2.metric("Hit rate resolvido", safe_pct(memory_row.get("resolved_hit_rate")))
        c3.metric("Drift score", safe_num(memory_row.get("drift_score")))
        st.caption(
            "Histórico = contexto desportivo. Resolvidos = aprendizagem supervisionada. "
            "OPEN = padrões/drift apenas; nunca são tratados como resultado real nem promovem uma política sozinhos."
        )

with st.expander("Estado técnico V3", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("CLV mínimo", safe_pct(policy.get("min_clv_probability")))
    c2.metric("EV mínimo", safe_pct(policy.get("min_expected_value")))
    c3.metric("Edge mínimo", safe_pct(policy.get("min_edge_market")))
    st.caption("O chat é apenas leitura/análise. Não altera thresholds nem executa apostas.")

with st.expander("🔎 Últimas decisões V3", expanded=False):
    if decisions.empty:
        st.info("Ainda não existem decisões V3 para mostrar.")
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

    if any(term in question_text.casefold() for term in ("memoria", "memory", "open", "drift", "base")) and memory_row:
        answer = (
            f"A memória V5 está em **{safe_text(memory_row.get('state'))}**: "
            f"**{safe_int(memory_row.get('historical_rows'))}** jogos históricos de contexto, "
            f"**{safe_int(memory_row.get('resolved_rows'))}** decisões resolvidas com label real e "
            f"**{safe_int(memory_row.get('open_rows'))}** jogos OPEN sem label. "
            f"ROI nos resolvidos: **{safe_pct(memory_row.get('resolved_roi'))}**; "
            f"drift score atual: **{safe_num(memory_row.get('drift_score'))}**. "
            "Os OPEN ajudam a detetar padrões e drift, mas não contam como vitória/derrota."
        )
    else:
        try:
            answer = answer_question(question_text, decisions, policies, learning, performance)
            answer = safe_text(answer, "Não consegui gerar uma resposta.")
        except Exception as exc:
            answer = "O chat encontrou um erro ao analisar esta pergunta."
            with st.chat_message("assistant"):
                st.error(answer)
                st.exception(exc)
        else:
            pass

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.agent_chat_messages_v2.append({"role": "assistant", "content": answer})

st.divider()
st.caption("FootballAI Agent Chat · shadow/paper only")
