from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import libsql
import pandas as pd
import streamlit as st

from agent.chat_assistant import answer_question, agent_snapshot


st.set_page_config(
    page_title="FootballAI · Agente IA",
    page_icon="🤖",
    layout="wide",
)

LOCAL_TZ = ZoneInfo("Europe/Lisbon")


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
        raise RuntimeError(
            "TURSO_DATABASE_URL/TURSO_AUTH_TOKEN não configurados."
        )
    return libsql.connect(database=url, auth_token=token)


@st.cache_data(ttl=30, show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    connection = connect()
    try:
        cursor = connection.execute(sql, params)
        return pd.DataFrame(
            cursor.fetchall(),
            columns=[c[0] for c in (cursor.description or [])],
        )
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
            decisions = query(
                "SELECT * FROM agent_decisions ORDER BY evaluated_at DESC, id DESC"
            )

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
        learning = query(
            "SELECT * FROM agent_learning_runs ORDER BY id DESC LIMIT 1"
        )

    if "agent_performance_runs" in tables:
        performance = query(
            "SELECT * FROM agent_performance_runs ORDER BY id DESC LIMIT 1"
        )

    if "agent_memory_runs" in tables:
        memory = query(
            "SELECT * FROM agent_memory_runs ORDER BY id DESC LIMIT 1"
        )

    return decisions, policies, learning, performance, memory


def load_v10_data(tables: set[str]) -> tuple[pd.DataFrame, str]:
    if "agent_v10_universe_decisions" in tables:
        return (
            query(
                """
                SELECT * FROM agent_v10_universe_decisions
                ORDER BY fixture_date DESC, fixture_time DESC, id DESC
                """
            ),
            "upcoming_matches",
        )

    if "agent_v10_shadow_decisions" in tables:
        return (
            query(
                """
                SELECT * FROM agent_v10_shadow_decisions
                ORDER BY fixture_date DESC, fixture_time DESC, id DESC
                """
            ),
            "paper_trades (legado)",
        )

    return pd.DataFrame(), "—"


def load_v10_run(tables: set[str]) -> pd.DataFrame:
    if "agent_v10_universe_runs" not in tables:
        return pd.DataFrame()
    return query(
        "SELECT * FROM agent_v10_universe_runs ORDER BY id DESC LIMIT 1"
    )


def load_v11_data(tables: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "agent_v11_outcome_runs" not in tables:
        return pd.DataFrame(), pd.DataFrame()

    runs = query(
        "SELECT * FROM agent_v11_outcome_runs ORDER BY id DESC LIMIT 1"
    )
    if runs.empty or "agent_v11_outcome_patterns" not in tables:
        return runs, pd.DataFrame()

    run_id = safe_int(runs.iloc[0].get("id"), 0)
    patterns = query(
        """
        SELECT * FROM agent_v11_outcome_patterns
        WHERE run_id=?
        ORDER BY dimension, n DESC, segment
        """,
        (run_id,),
    )
    return runs, patterns


def v10_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "decision" not in frame.columns:
        return {
            "total": 0,
            "bets": 0,
            "resolved": 0,
            "wins": 0,
            "hit": None,
            "profit": 0.0,
            "roi": None,
        }

    bets = frame[frame["decision"] == "BET_SHADOW"].copy()
    resolved = (
        bets[bets["result_over25"].isin([0, 1])]
        if "result_over25" in bets.columns
        else pd.DataFrame()
    )
    wins = int(resolved["result_over25"].sum()) if not resolved.empty else 0
    profit = (
        float(resolved["profit"].fillna(0).sum())
        if not resolved.empty and "profit" in resolved.columns
        else 0.0
    )
    return {
        "total": len(frame),
        "bets": len(bets),
        "resolved": len(resolved),
        "wins": wins,
        "hit": wins / len(resolved) if len(resolved) else None,
        "profit": profit,
        "roi": profit / len(resolved) if len(resolved) else None,
    }


def counterfactual_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0, "wins": 0, "hit": None, "profit": 0.0, "roi": None}

    valid = frame[
        frame["result_over25"].isin([0, 1])
        & frame["odd_over25_open"].notna()
    ].copy()
    if valid.empty:
        return {"n": 0, "wins": 0, "hit": None, "profit": 0.0, "roi": None}

    wins = int(valid["result_over25"].sum())
    profit = 0.0
    for _, row in valid.iterrows():
        if safe_int(row.get("result_over25")) == 1:
            profit += safe_float(row.get("odd_over25_open")) - 1.0
        else:
            profit -= 1.0
    n = len(valid)
    return {
        "n": n,
        "wins": wins,
        "hit": wins / n,
        "profit": profit,
        "roi": profit / n,
    }


def result_label(value: Any) -> str:
    if safe_int(value, -1) == 1:
        return "✅ Over"
    if safe_int(value, -1) == 0:
        return "❌ Under"
    return "⏳ Pendente"


def vote_label(row: pd.Series | dict[str, Any]) -> str:
    parts = []
    for short, field in (
        ("G", "goal_vote"),
        ("A", "attack_vote"),
        ("M", "ml_vote"),
        ("V", "value_vote"),
    ):
        if safe_int(row.get(field)) == 1:
            parts.append(short)
    return "+".join(parts) if parts else "—"


def display_games(frame: pd.DataFrame, include_result: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    out["Jogo"] = (
        out.get("home_team", pd.Series(index=out.index, dtype=str)).fillna("").astype(str)
        + " vs "
        + out.get("away_team", pd.Series(index=out.index, dtype=str)).fillna("").astype(str)
    )
    out["Votos"] = out.apply(vote_label, axis=1)
    if include_result and "result_over25" in out.columns:
        out["Resultado"] = out["result_over25"].apply(result_label)

    wanted = [
        "fixture_date",
        "fixture_time",
        "Jogo",
        "league",
        "decision",
        "Votos",
        "vote_count",
        "odd_over25_open",
        "sports_probability",
        "expected_value",
        "goal_score",
        "attack_score",
        "ml_score",
        "value_score",
    ]
    if include_result:
        wanted += ["Resultado", "profit"]
    return out[[column for column in wanted if column in out.columns]]


COLUMN_CONFIG = {
    "fixture_date": st.column_config.TextColumn("Data"),
    "fixture_time": st.column_config.TextColumn("Hora"),
    "league": st.column_config.TextColumn("Liga"),
    "decision": st.column_config.TextColumn("Decisão"),
    "vote_count": st.column_config.NumberColumn("# votos", format="%d"),
    "odd_over25_open": st.column_config.NumberColumn("Odd", format="%.2f"),
    "sports_probability": st.column_config.NumberColumn("P(Over)", format="%.1%%"),
    "expected_value": st.column_config.NumberColumn("EV", format="%.1%%"),
    "sports_edge_market": st.column_config.NumberColumn("Edge", format="%.1%%"),
    "clv_probability": st.column_config.NumberColumn("CLV prob.", format="%.1%%"),
    "goal_score": st.column_config.NumberColumn("Goal", format="%.1f"),
    "attack_score": st.column_config.NumberColumn("Attack", format="%.1f"),
    "ml_score": st.column_config.NumberColumn("ML", format="%.1f"),
    "value_score": st.column_config.NumberColumn("Value", format="%.1f"),
    "profit": st.column_config.NumberColumn("Lucro", format="%.2fu"),
}


st.title("🤖 Agente IA")
st.caption(
    "Vista operacional do Champion V10: o que está a recomendar, como está a correr e o que o V11 está a aprender."
)

try:
    tables = table_names()
    decisions, policies, learning, performance, memory = load_agent_data(tables)
    v10, v10_source = load_v10_data(tables)
    v10_run = load_v10_run(tables)
    v11_runs, v11_patterns = load_v11_data(tables)
    snapshot = agent_snapshot(decisions, policies, learning, performance)
except Exception as exc:
    st.error("Não foi possível carregar os dados do agente.")
    st.exception(exc)
    st.stop()

policy = snapshot.get("policy") if isinstance(snapshot.get("policy"), dict) else {}
learner = snapshot.get("learner") if isinstance(snapshot.get("learner"), dict) else {}
counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
memory_row = memory.iloc[0].to_dict() if not memory.empty else {}
v10_run_row = v10_run.iloc[0].to_dict() if not v10_run.empty else {}
v11_row = v11_runs.iloc[0].to_dict() if not v11_runs.empty else {}
v10_summary = v10_metrics(v10)

today = datetime.now(LOCAL_TZ).date().isoformat()
if not v10.empty and "fixture_date" in v10.columns:
    today_rows = v10[v10["fixture_date"].astype(str) == today].copy()
else:
    today_rows = pd.DataFrame()

today_bets = (
    today_rows[today_rows["decision"] == "BET_SHADOW"].copy()
    if not today_rows.empty and "decision" in today_rows.columns
    else pd.DataFrame()
)
if not today_bets.empty and "result_over25" in today_bets.columns:
    pending_bets = today_bets[~today_bets["result_over25"].isin([0, 1])].copy()
else:
    pending_bets = today_bets.copy()

resolved_bets = (
    v10[
        (v10["decision"] == "BET_SHADOW")
        & (v10["result_over25"].isin([0, 1]))
    ].copy()
    if not v10.empty
    and {"decision", "result_over25"}.issubset(v10.columns)
    else pd.DataFrame()
)
resolved_pass = (
    v10[
        (v10["decision"] == "PASS")
        & (v10["result_over25"].isin([0, 1]))
    ].copy()
    if not v10.empty
    and {"decision", "result_over25"}.issubset(v10.columns)
    else pd.DataFrame()
)
near_pass = (
    resolved_pass[
        (pd.to_numeric(resolved_pass["vote_count"], errors="coerce") == 2)
        & (pd.to_numeric(resolved_pass["expected_value"], errors="coerce") >= 0)
    ].copy()
    if not resolved_pass.empty
    and {"vote_count", "expected_value"}.issubset(resolved_pass.columns)
    else pd.DataFrame()
)
pass_summary = counterfactual_metrics(resolved_pass)
near_summary = counterfactual_metrics(near_pass)

st.markdown("### Estado rápido")
t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Jogos analisados hoje", len(today_rows))
t2.metric("BET hoje", len(today_bets))
t3.metric("BET pendentes", len(pending_bets))
t4.metric("Hit rate V10", safe_pct(v10_summary["hit"]))
t5.metric("ROI V10", safe_pct(v10_summary["roi"]))

if len(pending_bets):
    st.success(
        f"🎯 O Champion tem **{len(pending_bets)} BET_SHADOW pendente(s) hoje**. "
        "Vê os jogos na aba **Hoje**."
    )
elif len(today_bets):
    st.info("As BET_SHADOW de hoje já estão resolvidas ou não há nenhuma pendente neste momento.")
else:
    st.info("Neste momento não há BET_SHADOW para hoje. O agente continua a analisar o universo disponível.")

main_today, main_results, main_logic, main_chat, main_tech = st.tabs(
    ["🎯 Hoje", "📈 Resultados", "🧠 Como decide", "💬 Chat", "🛠️ Técnico"]
)

with main_today:
    st.subheader(f"Jogos de hoje · {today}")
    st.caption(
        "Aqui aparecem primeiro as decisões que interessam para acompanhar o dia. BET_SHADOW é apenas paper/shadow, não executa apostas reais."
    )

    if not pending_bets.empty:
        st.markdown("#### 🎯 BET_SHADOW pendentes")
        st.dataframe(
            display_games(pending_bets, include_result=False),
            use_container_width=True,
            hide_index=True,
            column_config=COLUMN_CONFIG,
        )
    elif not today_bets.empty:
        st.markdown("#### 🎯 BET_SHADOW de hoje")
        st.dataframe(
            display_games(today_bets, include_result=True),
            use_container_width=True,
            hide_index=True,
            column_config=COLUMN_CONFIG,
        )
    else:
        st.info("Sem BET_SHADOW para hoje até ao momento.")

    st.markdown("#### Cobertura do agente")
    if v10_run_row:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jogos vistos", safe_int(v10_run_row.get("upcoming_seen")))
        c2.metric("Já analisados", safe_int(v10_run_row.get("already_frozen")))
        c3.metric("Novas decisões", safe_int(v10_run_row.get("inserted")))
        c4.metric("Sem dados suficientes", safe_int(v10_run_row.get("skipped_data")))
        skipped = safe_int(v10_run_row.get("skipped_data"))
        if skipped:
            st.warning(
                f"Há **{skipped}** entradas sem histórico/features suficientes na última varredura. "
                "Esses jogos são vistos pelo agente, mas não recebem decisão até termos dados utilizáveis."
            )
    else:
        st.caption("Ainda não há registo da última varredura do universo completo.")

    with st.expander("📋 Ver todos os jogos analisados hoje", expanded=False):
        if today_rows.empty:
            st.info("Ainda não há decisões V10 de hoje.")
        else:
            st.dataframe(
                display_games(today_rows, include_result=True),
                use_container_width=True,
                hide_index=True,
                column_config=COLUMN_CONFIG,
            )

with main_results:
    st.subheader("Desempenho prospectivo do Champion V10")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("BET resolvidas", v10_summary["resolved"])
    r2.metric("Vitórias", v10_summary["wins"])
    r3.metric("Hit rate", safe_pct(v10_summary["hit"]))
    r4.metric("Lucro", f"{v10_summary['profit']:+.2f}u")
    r5.metric("ROI", safe_pct(v10_summary["roi"]))

    if resolved_bets.empty:
        st.info("Ainda não há BET_SHADOW resolvidas.")
    else:
        with st.expander("Ver BET_SHADOW resolvidas", expanded=True):
            st.dataframe(
                display_games(resolved_bets, include_result=True),
                use_container_width=True,
                hide_index=True,
                column_config=COLUMN_CONFIG,
            )

    st.markdown("#### O que ficou de fora?")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("PASS resolvidos", pass_summary["n"])
    p2.metric("Over nos PASS", safe_pct(pass_summary["hit"]))
    p3.metric("ROI se apostasse em todos", safe_pct(pass_summary["roi"]))
    p4.metric("Lucro contrafactual", f"{pass_summary['profit']:+.2f}u")
    st.caption(
        "Esta linha é contrafactual: mede o que teria acontecido se apostássemos em todos os PASS. Serve para perceber se o filtro está a deixar valor para trás."
    )

    st.markdown("#### 🔎 Quase BET · 2 votos + EV ≥ 0")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Jogos", near_summary["n"])
    q2.metric("Over rate", safe_pct(near_summary["hit"]))
    q3.metric("ROI contrafactual", safe_pct(near_summary["roi"]))
    q4.metric("Lucro contrafactual", f"{near_summary['profit']:+.2f}u")
    if not near_pass.empty:
        st.dataframe(
            display_games(near_pass, include_result=True),
            use_container_width=True,
            hide_index=True,
            column_config=COLUMN_CONFIG,
        )

    st.divider()
    st.subheader("🧠 Aprendizagem V11")
    if not v11_row:
        st.info("O V11 ainda não tem uma execução registada.")
    else:
        l1, l2, l3, l4 = st.columns(4)
        resolved_count = safe_int(v11_row.get("resolved_rows"))
        date_count = safe_int(v11_row.get("unique_dates"))
        pattern_count = safe_int(v11_row.get("pattern_count"))
        l1.metric("Jogos aprendidos", resolved_count)
        l2.metric("Dias de evidência", date_count)
        l3.metric("Padrões atuais", pattern_count)
        l4.metric("Estado", safe_text(v11_row.get("state")))

        st.caption("Amostra mínima antes de procurar um challenger: 120 jogos resolvidos e 8 datas diferentes.")
        st.progress(min(resolved_count / 120, 1.0), text=f"Jogos: {resolved_count}/120")
        st.progress(min(date_count / 8, 1.0), text=f"Datas: {date_count}/8")

        if not v11_patterns.empty:
            pass_pattern = v11_patterns[
                (v11_patterns["dimension"] == "decision")
                & (v11_patterns["segment"] == "PASS")
            ]
            if not pass_pattern.empty:
                row = pass_pattern.iloc[0]
                st.caption(
                    "O V11 está a aprender com BET e PASS. Nos PASS resolvidos do último run: "
                    f"Over rate **{safe_pct(row.get('over_rate'))}** e ROI contrafactual **{safe_pct(row.get('counterfactual_roi'))}**."
                )

with main_logic:
    st.subheader("Como o Champion V10 toma uma decisão")
    st.markdown(
        "O jogo recebe quatro opiniões independentes dentro do mesmo analisador. Cada uma dá um voto quando ultrapassa o threshold atual."
    )

    a, b, c, d = st.columns(4)
    a.markdown("**⚽ Goal · ≥ 55**\n\nPerfil de golos: expected goals, tendência Over e BTTS.")
    b.markdown("**🔥 Attack · ≥ 55**\n\nPressão ofensiva: capacidade de marcar, sofrer e qualidade do ataque.")
    c.markdown("**🤖 ML · ≥ 65**\n\nConfiança do modelo desportivo de Over 2.5.")
    d.markdown("**💶 Value · ≥ 60**\n\nValor de mercado: EV e vantagem sobre a probabilidade implícita da odd.")

    st.success("✅ BET_SHADOW = pelo menos **3 dos 4 votos** + **EV ≥ 0**")
    st.caption("CLV é guardado para diagnóstico posterior, mas não bloqueia a decisão do Champion V10.")

    candidates = today_rows.copy() if not today_rows.empty else v10.head(50).copy()
    if not candidates.empty:
        candidates["_label"] = candidates.apply(
            lambda row: f"{safe_text(row.get('home_team'))} vs {safe_text(row.get('away_team'))} · {safe_text(row.get('decision'))}",
            axis=1,
        )
        selected_label = st.selectbox(
            "Ver a explicação de um jogo",
            candidates["_label"].tolist(),
            key="agent_game_explainer",
        )
        selected = candidates[candidates["_label"] == selected_label].iloc[0]

        e1, e2, e3, e4 = st.columns(4)
        for col, label, field, threshold in (
            (e1, "Goal", "goal_score", 55.0),
            (e2, "Attack", "attack_score", 55.0),
            (e3, "ML", "ml_score", 65.0),
            (e4, "Value", "value_score", 60.0),
        ):
            value = safe_float(selected.get(field))
            col.metric(label, f"{value:.1f}", "VOTO ✅" if value >= threshold else "sem voto")

        st.markdown(
            f"**Decisão:** {safe_text(selected.get('decision'))} · "
            f"**Votos:** {safe_int(selected.get('vote_count'))}/4 ({vote_label(selected)}) · "
            f"**Odd:** {safe_num(selected.get('odd_over25_open'))} · "
            f"**P(Over):** {safe_pct(selected.get('sports_probability'))} · "
            f"**EV:** {safe_pct(selected.get('expected_value'))}"
        )

with main_chat:
    st.subheader("Fala com o agente")
    st.caption("O chat não altera thresholds e não executa apostas. Serve apenas para leitura e análise dos dados guardados.")

    if "agent_chat_messages_v3" not in st.session_state:
        st.session_state.agent_chat_messages_v3 = [
            {
                "role": "assistant",
                "content": (
                    "Posso resumir o Champion V10, os jogos de hoje, o estado da aprendizagem V11 ou os dados técnicos do agente."
                ),
            }
        ]

    for message in st.session_state.agent_chat_messages_v3:
        role = safe_text(message.get("role"), "assistant")
        content = safe_text(message.get("content"), "")
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("Pergunta ao Agente IA…")
    if question:
        question_text = safe_text(question, "")
        st.session_state.agent_chat_messages_v3.append(
            {"role": "user", "content": question_text}
        )
        with st.chat_message("user"):
            st.markdown(question_text)

        q = question_text.casefold()
        if any(term in q for term in ("v10", "champion", "hoje", "bet", "aposta")):
            answer = (
                f"Hoje tenho **{len(today_rows)}** jogos V10 analisados e **{len(today_bets)}** BET_SHADOW, "
                f"das quais **{len(pending_bets)}** estão pendentes. No acumulado prospectivo: "
                f"**{v10_summary['resolved']}** BET resolvidas, hit rate **{safe_pct(v10_summary['hit'])}**, "
                f"lucro **{v10_summary['profit']:+.2f}u** e ROI **{safe_pct(v10_summary['roi'])}**."
            )
        elif any(term in q for term in ("v11", "aprende", "aprendiz", "padr")) and v11_row:
            answer = (
                f"O V11 está em **{safe_text(v11_row.get('state'))}** com "
                f"**{safe_int(v11_row.get('resolved_rows'))}** jogos resolvidos, "
                f"**{safe_int(v11_row.get('unique_dates'))}** datas e "
                f"**{safe_int(v11_row.get('pattern_count'))}** padrões atuais. "
                "Ele aprende com BET e PASS, mas não altera automaticamente o Champion."
            )
        elif any(term in q for term in ("memoria", "memory", "drift")) and memory_row:
            answer = (
                f"A memória V5 está em **{safe_text(memory_row.get('state'))}**: "
                f"**{safe_int(memory_row.get('historical_rows'))}** jogos históricos, "
                f"**{safe_int(memory_row.get('resolved_rows'))}** resolvidos e "
                f"**{safe_int(memory_row.get('open_rows'))}** OPEN. "
                f"ROI resolvido **{safe_pct(memory_row.get('resolved_roi'))}**; "
                f"drift **{safe_num(memory_row.get('drift_score'))}**."
            )
        else:
            try:
                answer = safe_text(
                    answer_question(
                        question_text,
                        decisions,
                        policies,
                        learning,
                        performance,
                    ),
                    "Não consegui gerar uma resposta.",
                )
            except Exception:
                answer = "O chat encontrou um erro ao analisar esta pergunta."

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.agent_chat_messages_v3.append(
            {"role": "assistant", "content": answer}
        )

with main_tech:
    st.subheader("Detalhes técnicos")
    st.caption("Informação útil para diagnóstico, sem misturar com a operação diária do Champion.")

    with st.expander("🏆 Champion V10 · configuração e universo", expanded=True):
        if v10.empty:
            st.info("Ainda não existem decisões V10.")
        else:
            st.markdown(
                f"**Versão:** {safe_text(v10.iloc[0].get('champion_version'))} · "
                f"**Fonte:** {v10_source} · **Regra:** 3/4 votos + EV ≥ 0"
            )
            if v10_run_row:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Upcoming vistos", safe_int(v10_run_row.get("upcoming_seen")))
                c2.metric("Já congelados", safe_int(v10_run_row.get("already_frozen")))
                c3.metric("Novos", safe_int(v10_run_row.get("inserted")))
                c4.metric("Sem dados", safe_int(v10_run_row.get("skipped_data")))
                c5.metric("Timing", safe_int(v10_run_row.get("skipped_timing")))

            with st.expander("Todas as decisões V10", expanded=False):
                st.dataframe(
                    display_games(v10.head(500), include_result=True),
                    use_container_width=True,
                    hide_index=True,
                    column_config=COLUMN_CONFIG,
                )

    with st.expander("🧠 Memória V5", expanded=False):
        if not memory_row:
            st.info("Sem execução V5 disponível.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Histórico", safe_int(memory_row.get("historical_rows")))
            m2.metric("Resolvidos", safe_int(memory_row.get("resolved_rows")))
            m3.metric("OPEN", safe_int(memory_row.get("open_rows")))
            m4.metric("Estado", safe_text(memory_row.get("state")))

    with st.expander("🧪 Agente V3 legado", expanded=False):
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Política", safe_text(policy.get("policy_version")))
        v2.metric("Decisões", safe_int(snapshot.get("decision_total")))
        v3.metric("QUALIFIED", safe_int(counts.get("QUALIFIED_SHADOW")))
        v4.metric("Learner", safe_text(learner.get("decision")))
        c1, c2, c3 = st.columns(3)
        c1.metric("CLV mínimo", safe_pct(policy.get("min_clv_probability")))
        c2.metric("EV mínimo", safe_pct(policy.get("min_expected_value")))
        c3.metric("Edge mínimo", safe_pct(policy.get("min_edge_market")))

        if not decisions.empty:
            cols_to_show = [
                column
                for column in [
                    "evaluated_at",
                    "fixture_date",
                    "league",
                    "home_team",
                    "away_team",
                    "decision",
                    "tier",
                    "odd_over25_open",
                    "clv_probability",
                    "sports_probability",
                    "expected_value",
                    "sports_edge_market",
                    "status",
                    "result_over25",
                ]
                if column in decisions.columns
            ]
            st.dataframe(
                decisions[cols_to_show].head(50),
                use_container_width=True,
                hide_index=True,
            )

st.divider()
st.caption("FootballAI · Champion V10 + Outcome Learner V11 · shadow/paper only")
