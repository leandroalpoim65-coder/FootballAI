from __future__ import annotations

import json
import os

import libsql
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="FootballAI · Aprendizagem V11",
    page_icon="🧠",
    layout="wide",
)


def setting(name: str) -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return os.getenv(name, "").strip()


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
            columns=[c[0] for c in (cursor.description or [])],
        )
    finally:
        connection.close()


@st.cache_data(ttl=30, show_spinner=False)
def tables() -> set[str]:
    frame = query("SELECT name FROM sqlite_master WHERE type='table'")
    return set(frame["name"].astype(str)) if not frame.empty else set()


def parse_json(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def pct(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def num(value, digits=2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


st.title("🧠 V11 · Outcome Learner")
st.caption(
    "Aprendizagem supervisionada com resultados reais de todos os jogos V10 "
    "resolvidos — BET_SHADOW e PASS. O Champion V10 nunca é alterado automaticamente."
)

try:
    known = tables()
except Exception as exc:
    st.error("Não foi possível ligar à Turso.")
    st.exception(exc)
    st.stop()

if "agent_v11_outcome_runs" not in known:
    st.info("O V11 ainda não teve a primeira execução cloud.")
    st.stop()

runs = query("SELECT * FROM agent_v11_outcome_runs ORDER BY id DESC LIMIT 20")
if runs.empty:
    st.info("Ainda não existem runs V11.")
    st.stop()

latest = runs.iloc[0].to_dict()
run_id = int(latest["id"])

state = str(latest.get("state") or "—")
state_icon = {
    "WAIT_SAMPLE": "⏳",
    "KEEP_CHAMPION": "🏆",
    "CHALLENGER_READY": "🚀",
}.get(state, "🧠")

st.subheader(f"{state_icon} Estado atual: {state}")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Resolvidos", int(latest.get("resolved_rows") or 0))
m2.metric("BET resolvidos", int(latest.get("bet_rows") or 0))
m3.metric("PASS resolvidos", int(latest.get("pass_rows") or 0))
m4.metric("Datas", int(latest.get("unique_dates") or 0))
m5.metric("Treino", int(latest.get("train_rows") or 0))
m6.metric("Validação", int(latest.get("validation_rows") or 0))

if state == "WAIT_SAMPLE":
    st.info(
        "O learner está a acumular dados. Só começa a procurar challengers com "
        "pelo menos 120 jogos resolvidos e 8 datas diferentes."
    )

champion_train = parse_json(latest.get("champion_train_json"))
champion_valid = parse_json(latest.get("champion_validation_json"))
candidate_policy = parse_json(latest.get("candidate_policy_json"))
candidate_train = parse_json(latest.get("candidate_train_json"))
candidate_valid = parse_json(latest.get("candidate_validation_json"))

if candidate_policy:
    st.subheader("Challenger aprendido")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Goal ≥", num(candidate_policy.get("goal_threshold"), 0))
    p2.metric("Attack ≥", num(candidate_policy.get("attack_threshold"), 0))
    p3.metric("ML ≥", num(candidate_policy.get("ml_threshold"), 0))
    p4.metric("Value ≥", num(candidate_policy.get("value_threshold"), 0))
    p5.metric("Votos", int(candidate_policy.get("min_votes") or 0))
    p6.metric("EV mínimo", pct(candidate_policy.get("min_expected_value")))

    compare = pd.DataFrame(
        [
            {
                "Estratégia": "Champion V10",
                "Treino n": champion_train.get("n"),
                "Treino hit": champion_train.get("hit_rate"),
                "Treino ROI": champion_train.get("roi"),
                "Validação n": champion_valid.get("n"),
                "Validação hit": champion_valid.get("hit_rate"),
                "Validação ROI": champion_valid.get("roi"),
                "Validação DD": champion_valid.get("max_drawdown"),
            },
            {
                "Estratégia": "Challenger V11",
                "Treino n": candidate_train.get("n"),
                "Treino hit": candidate_train.get("hit_rate"),
                "Treino ROI": candidate_train.get("roi"),
                "Validação n": candidate_valid.get("n"),
                "Validação hit": candidate_valid.get("hit_rate"),
                "Validação ROI": candidate_valid.get("roi"),
                "Validação DD": candidate_valid.get("max_drawdown"),
            },
        ]
    )
    st.dataframe(
        compare,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Treino hit": st.column_config.NumberColumn(format="%.1%%"),
            "Treino ROI": st.column_config.NumberColumn(format="%.1%%"),
            "Validação hit": st.column_config.NumberColumn(format="%.1%%"),
            "Validação ROI": st.column_config.NumberColumn(format="%.1%%"),
            "Validação DD": st.column_config.NumberColumn(format="%.2fu"),
        },
    )

    if int(latest.get("gate_passed") or 0):
        st.success(
            "O challenger passou os gates de validação, mas continua apenas como "
            "candidato. Não substituiu automaticamente o Champion."
        )
    else:
        st.warning("O challenger não passou os gates; o Champion V10 mantém-se.")

reasons = parse_json(latest.get("reasons_json"))
if isinstance(reasons, list) and reasons:
    with st.expander("Razões / gates"):
        for reason in reasons:
            st.write(f"• {reason}")

if "agent_v11_outcome_patterns" in known:
    patterns = query(
        """
        SELECT dimension, segment, n, wins, over_rate, avg_odd,
               counterfactual_profit, counterfactual_roi
        FROM agent_v11_outcome_patterns
        WHERE run_id = ?
        ORDER BY dimension, n DESC, segment
        """,
        (run_id,),
    )

    st.subheader("Memória de padrões")
    st.caption(
        "ROI contrafactual responde à pergunta: se tivéssemos apostado 1u em todos "
        "os jogos desse padrão, incluindo os que o V10 marcou PASS, o que teria acontecido?"
    )

    if patterns.empty:
        st.info("Ainda não existem padrões resolvidos.")
    else:
        dimension = st.selectbox(
            "Analisar por",
            sorted(patterns["dimension"].unique().tolist()),
        )
        view = patterns[patterns["dimension"] == dimension].copy()
        st.dataframe(
            view,
            hide_index=True,
            use_container_width=True,
            column_config={
                "over_rate": st.column_config.NumberColumn("Over real", format="%.1%%"),
                "avg_odd": st.column_config.NumberColumn("Odd média", format="%.2f"),
                "counterfactual_profit": st.column_config.NumberColumn("Lucro CF", format="%.2fu"),
                "counterfactual_roi": st.column_config.NumberColumn("ROI CF", format="%.1%%"),
            },
        )

with st.expander("Histórico de runs V11"):
    cols = [
        c for c in [
            "id", "finished_at", "state", "resolved_rows", "unique_dates",
            "split_date", "train_rows", "validation_rows", "policies_evaluated",
            "gate_passed", "pattern_count",
        ] if c in runs.columns
    ]
    st.dataframe(runs[cols], hide_index=True, use_container_width=True)

st.caption(
    "V11 usa apenas resultados reais como label. CLV pode ser estudado depois, mas não "
    "é usado para decidir se uma previsão estava certa ou errada."
)
