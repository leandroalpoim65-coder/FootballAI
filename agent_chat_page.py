from __future__ import annotations

import os

import libsql
import pandas as pd
import streamlit as st

from agent.chat_assistant import answer_question, agent_snapshot


st.set_page_config(
    page_title="FootballAI · Agente IA",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container{max-width:1200px;padding-top:1rem;padding-bottom:2rem}
    [data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:.7rem 1rem}
    </style>
    """,
    unsafe_allow_html=True,
)


def setting(name: str) -> str:
    try:
        value = st.secrets[name]
    except Exception:
        value = os.getenv(name, "")
    return str(value or "").strip()


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
            columns=[item[0] for item in (cursor.description or [])],
        )
    finally:
        connection.close()


@st.cache_data(ttl=30, show_spinner=False)
def table_names() -> set[str]:
    frame = query("SELECT name FROM sqlite_master WHERE type='table'")
    if frame.empty or "name" not in frame:
        return set()
    return set(frame["name"].astype(str))


def load_agent_data(tables: set[str]):
    decisions = pd.DataFrame()
    policies = pd.DataFrame()
    learning = pd.DataFrame()
    performance = pd.DataFrame()

    if "agent_decisions" in tables:
        if "paper_trades" in tables:
            decisions = query(
                """
                SELECT
                    d.*,
                    p.status,
                    p.result_over25,
                    p.clv,
                    p.closing_source
                FROM agent_decisions d
                LEFT JOIN paper_trades p
                  ON p.id = d.paper_trade_id
                ORDER BY d.evaluated_at DESC, d.id DESC
                """
            )
        else:
            decisions = query(
                "SELECT * FROM agent_decisions ORDER BY evaluated_at DESC, id DESC"
            )

    if "agent_policies" in tables:
        policies = query(
            """
            SELECT *
            FROM agent_policies
            WHERE status='ACTIVE'
            ORDER BY COALESCE(activated_at, created_at) DESC, id DESC
            LIMIT 1
            """
        )

    if "agent_learning_runs" in tables:
        learning = query(
            "SELECT * FROM agent_learning_runs ORDER BY id DESC LIMIT 1"
        )

    if "agent_performance_runs" in tables:
        performance = query(
            "SELECT * FROM agent_performance_runs ORDER BY id DESC LIMIT 1"
        )

    return decisions, policies, learning, performance


try:
    tables = table_names()
    decisions, policies, learning, performance = load_agent_data(tables)
except Exception as exc:
    st.error("Não foi possível carregar o Agente IA a partir da Turso.")
    st.code(str(exc))
    st.stop()

snapshot = agent_snapshot(decisions, policies, learning, performance)
policy = snapshot.get("policy") or {}
learner = snapshot.get("learner") or {}
counts = snapshot.get("counts") or {}

st.title("🤖 Agente IA · FootballAI")
st.caption(
    "Conversa diretamente com a camada autónoma do projeto. "
    "As respostas abaixo são calculadas a partir da Turso e das decisões congeladas do agente."
)

cols = st.columns(4)
cols[0].metric("Política ativa", str(policy.get("policy_version") or "—"))
cols[1].metric("Decisões", int(snapshot.get("decision_total") or 0))
cols[2].metric("QUALIFIED", int(counts.get("QUALIFIED_SHADOW", 0)))
cols[3].metric("Learner", str(learner.get("decision") or "—"))

with st.expander("Estado técnico do agente", expanded=False):
    c = st.columns(3)
    c[0].metric(
        "CLV mínimo",
        f"{float(policy.get('min_clv_probability', 0))*100:.1f}%" if policy else "—",
    )
    c[1].metric(
        "EV mínimo",
        f"{float(policy.get('min_expected_value', 0))*100:.1f}%" if policy else "—",
    )
    c[2].metric(
        "Edge mínimo",
        f"{float(policy.get('min_edge_market', 0))*100:.1f}%" if policy else "—",
    )
    st.caption(
        "O chat é de leitura/análise. Não altera thresholds nem políticas por uma mensagem. "
        "As promoções continuam a passar pelo Policy Learner e pela validação temporal."
    )

with st.expander("🔎 O que o agente está a examinar", expanded=False):
    st.caption(
        "Isto mostra o rasto auditável do agente: fontes consultadas, política ativa, "
        "métricas e motivos guardados. Não expõe raciocínio interno privado."
    )
    st.write(
        {
            "Tabelas consultadas": [
                name
                for name in (
                    "agent_decisions",
                    "paper_trades",
                    "agent_policies",
                    "agent_learning_runs",
                    "agent_performance_runs",
                )
                if name in tables
            ],
            "Decisões carregadas": int(len(decisions)),
            "Política ativa": str(policy.get("policy_version") or "—"),
            "Estado learner": str(learner.get("decision") or "—"),
        }
    )

    if not decisions.empty:
        cols_to_show = [
            name
            for name in (
                "evaluated_at",
                "fixture_date",
                "league",
                "home_team",
                "away_team",
                "decision",
                "tier",
                "policy_version",
                "odd_over25_open",
                "clv_probability",
                "sports_probability",
                "expected_value",
                "sports_edge_market",
                "reasons_json",
                "status",
                "result_over25",
                "clv",
                "closing_source",
            )
            if name in decisions.columns
        ]
        st.markdown("#### Últimas decisões examinadas")
        st.dataframe(
            decisions[cols_to_show].head(25),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Ainda não existem decisões do agente para inspecionar.")

if "agent_chat_messages" not in st.session_state:
    st.session_state.agent_chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Estou ligado aos dados do FootballAI. Podes perguntar-me, por exemplo: "
                "**Como está o agente?**, **qual é a política atual?**, **o que aprendeste?**, "
                "**que ligas estão melhores/piores?** ou **porque rejeitaste [nome do jogo]?**"
            ),
        }
    ]

for message in st.session_state.agent_chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Fala com o Agente IA…")
if question:
    st.session_state.agent_chat_messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("A analisar os dados do projeto…"):
            answer = answer_question(
                question,
                decisions,
                policies,
                learning,
                performance,
            )
        st.markdown(answer)

    st.session_state.agent_chat_messages.append(
        {"role": "assistant", "content": answer}
    )

st.divider()
st.caption(
    "FootballAI Agent Chat · análise grounded na Turso · paper/shadow apenas · "
    "sem execução automática de apostas a partir do chat"
)
