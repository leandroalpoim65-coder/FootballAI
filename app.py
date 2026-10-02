from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import libsql


st.set_page_config(
    page_title="FootballAI — Análise de Jogos",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
SHADOW_TABLE = "goal_model_home_o25_shadow"


st.markdown(
    """
    <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
        .small-note { opacity: .72; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_pct(value: object, decimals: int = 2) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_odd(value: object) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def fmt_date(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return pd.Timestamp(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def load_shadow() -> pd.DataFrame:
    url = os.getenv("TURSO_DATABASE_URL")
    token = os.getenv("TURSO_AUTH_TOKEN")
    if not url or not token:
        st.error(
            "Credenciais Turso não encontradas nesta sessão do Windows. "
            "Define TURSO_DATABASE_URL e TURSO_AUTH_TOKEN e reinicia a app."
        )
        return pd.DataFrame()
    try:
        connection = libsql.connect(database=url, auth_token=token)
        try:
            return pd.read_sql_query(
                f"SELECT * FROM {SHADOW_TABLE} ORDER BY fixture_date, fixture_time",
                connection,
            )
        finally:
            connection.close()
    except Exception as exc:
        st.error(f"Não foi possível ler o Turso: {exc}")
        return pd.DataFrame()


def latest_model_rows(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "prediction_timestamp" not in data.columns:
        return data
    result = data.copy()
    result["prediction_timestamp"] = pd.to_datetime(
        result["prediction_timestamp"], errors="coerce"
    )
    # Um jogo/modelo fica representado pelo registo mais recente.
    keys = [c for c in ["fixture_key", "model_version"] if c in result.columns]
    if keys:
        result = (
            result.sort_values("prediction_timestamp")
            .drop_duplicates(keys, keep="last")
        )
    return result


def build_game_view(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    rows = []
    for fixture_key, group in data.groupby("fixture_key", sort=False):
        group = group.copy()
        def prob(model: str):
            x = group.loc[group["model_version"] == model, "probability_over"]
            return float(x.iloc[-1]) if not x.empty else None

        p2 = prob("V2_FROZEN")
        p5 = prob("V5_HOME_FROZEN")
        p6 = prob("V6_FROZEN")
        p6t = prob("V6_TRAINED")

        base = group.iloc[-1]
        probability_values = [x for x in [p2, p5, p6, p6t] if x is not None]
        consensus = sum(abs(x - (sum(probability_values) / len(probability_values))) <= 0.025 for x in probability_values) if probability_values else 0
        selected_probability = p6t if p6t is not None else p6 if p6 is not None else p5 if p5 is not None else p2

        odd = pd.to_numeric(pd.Series([base.get("odd_over_at_prediction")]), errors="coerce").iloc[0]
        market_p = pd.to_numeric(pd.Series([base.get("market_probability_over_at_prediction")]), errors="coerce").iloc[0]
        ev = pd.to_numeric(pd.Series([base.get("ev_over_at_prediction")]), errors="coerce").iloc[0]
        fair = 1.0 / selected_probability if selected_probability and 0 < selected_probability < 1 else None

        rows.append({
            "fixture_key": fixture_key,
            "data": base.get("fixture_date"),
            "hora": base.get("fixture_time"),
            "liga": base.get("league"),
            "casa": base.get("home_team"),
            "fora": base.get("away_team"),
            "odd": odd,
            "prob_modelo": selected_probability,
            "prob_mercado": market_p,
            "odd_justa": fair,
            "ev": ev,
            "v2": p2,
            "v5": p5,
            "v6": p6,
            "v6_treinado": p6t,
            "concordancia": f"{consensus}/{len(probability_values)}",
            "status": base.get("status"),
            "historico": base.get("metadata_json"),
            "_group": group,
        })

    return pd.DataFrame(rows)


st.title("⚽ FootballAI — Análise de Jogos")
st.caption(
    "Painel estatístico ligado diretamente ao Turso. "
    "Os dados servem para análise/paper trading; a decisão final é tua."
)

if st.button("🔄 Atualizar dados do Turso"):
    st.rerun()


# ------------------------------------------------------------
# SIDEBAR / FILTROS
# ------------------------------------------------------------

data = latest_model_rows(load_shadow())
view = build_game_view(data)

st.sidebar.header("Filtros")

if view.empty:
    st.sidebar.info("Sem previsões disponíveis neste momento.")
    st.warning(
        "A tabela de análise ainda não tem dados. Executa o processo de shadow "
        "do Home O2.5 para preencher `goal_model_home_o25_shadow`."
    )
    st.stop()

leagues = sorted(view["liga"].dropna().astype(str).unique().tolist())
selected_leagues = st.sidebar.multiselect(
    "Ligas",
    leagues,
    default=leagues,
)

min_ev = st.sidebar.slider(
    "EV mínimo",
    min_value=-0.20,
    max_value=0.50,
    value=0.00,
    step=0.01,
    format="%.0f%%",
)

min_odd = st.sidebar.number_input("Odd mínima", min_value=1.01, value=1.50, step=0.05)
max_odd = st.sidebar.number_input("Odd máxima", min_value=1.01, value=5.00, step=0.10)

model_choice = st.sidebar.selectbox(
    "Probabilidade usada no painel",
    ["V6_TRAINED", "V6_FROZEN", "V5_HOME_FROZEN", "V2_FROZEN"],
)

show_only_positive = st.sidebar.checkbox("Mostrar apenas EV positivo", value=False)

filtered = view.copy()
if selected_leagues:
    filtered = filtered[filtered["liga"].astype(str).isin(selected_leagues)]

filtered = filtered[
    pd.to_numeric(filtered["odd"], errors="coerce").between(min_odd, max_odd)
]

filtered = filtered[
    pd.to_numeric(filtered["ev"], errors="coerce") >= min_ev
]

if show_only_positive:
    filtered = filtered[pd.to_numeric(filtered["ev"], errors="coerce") > 0]

filtered["prob_selecionada"] = filtered[model_choice.lower().replace("_", "_")]

# As colunas internas V2/V5/V6 têm nomes simplificados.
model_column_map = {
    "V2_FROZEN": "v2",
    "V5_HOME_FROZEN": "v5",
    "V6_FROZEN": "v6",
    "V6_TRAINED": "v6_treinado",
}
filtered["prob_selecionada"] = filtered[model_column_map[model_choice]]
filtered = filtered.sort_values("ev", ascending=False, na_position="last")


# ------------------------------------------------------------
# RESUMO
# ------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jogos analisados", f"{len(filtered):,}".replace(",", " "))
positive = int((pd.to_numeric(filtered["ev"], errors="coerce") > 0).sum())
c2.metric("EV positivo", positive)
c3.metric("EV médio", fmt_pct(pd.to_numeric(filtered["ev"], errors="coerce").mean()))
c4.metric("Prob. média", fmt_pct(pd.to_numeric(filtered["prob_selecionada"], errors="coerce").mean()))
c5.metric("Modelos disponíveis", "V2 · V5 · V6")

st.divider()


# ------------------------------------------------------------
# TABELA PRINCIPAL
# ------------------------------------------------------------

st.subheader("Jogos para análise")
st.caption(
    "Ordenação por EV calculado com o modelo selecionado. "
    "Isto é uma ordenação estatística, não uma indicação automática de aposta."
)

display = filtered.copy()
display["Jogo"] = display["casa"].astype(str) + " — " + display["fora"].astype(str)
display["Data"] = display["data"].apply(fmt_date)
display["Prob. modelo"] = display["prob_selecionada"].apply(fmt_pct)
display["Mercado"] = "Casa O2.5"
display["Odd"] = display["odd"].apply(fmt_odd)
display["Odd justa"] = display["odd_justa"].apply(fmt_odd)
display["Mercado %"] = display["prob_mercado"].apply(fmt_pct)
display["EV"] = display["ev"].apply(fmt_pct)
display["Concordância"] = display["concordancia"]

table_columns = [
    "Data", "hora", "liga", "Jogo", "Mercado", "Odd",
    "Prob. modelo", "Mercado %", "Odd justa", "EV", "Concordância",
]
st.dataframe(
    display[table_columns].rename(columns={"hora": "Hora", "liga": "Liga"}),
    use_container_width=True,
    hide_index=True,
)


# ------------------------------------------------------------
# DETALHE DO JOGO
# ------------------------------------------------------------

st.subheader("Detalhe")
if filtered.empty:
    st.info("Nenhum jogo corresponde aos filtros atuais.")
else:
    options = filtered["fixture_key"].tolist()
    labels = {
        row["fixture_key"]: f"{row['casa']} — {row['fora']} · {fmt_date(row['data'])}"
        for _, row in filtered.iterrows()
    }
    selected_key = st.selectbox(
        "Seleciona um jogo",
        options,
        format_func=lambda x: labels.get(x, x),
    )
    selected = filtered[filtered["fixture_key"] == selected_key].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("V2", fmt_pct(selected["v2"]))
    d2.metric("V5", fmt_pct(selected["v5"]))
    d3.metric("V6 congelado", fmt_pct(selected["v6"]))
    d4.metric("V6 treinado", fmt_pct(selected["v6_treinado"]))

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Odd", fmt_odd(selected["odd"]))
    e2.metric("Odd justa", fmt_odd(selected["odd_justa"]))
    e3.metric("EV", fmt_pct(selected["ev"]))
    e4.metric("Mercado", fmt_pct(selected["prob_mercado"]))

    st.markdown(
        f"**{selected['casa']} — {selected['fora']}** · "
        f"{selected['liga']} · {fmt_date(selected['data'])} {selected['hora'] or ''}"
    )

    st.markdown("#### Comparação dos modelos")
    comparison = pd.DataFrame({
        "Modelo": ["V2 congelado", "V5 Home congelado", "V6 congelado", "V6 treinado"],
        "Probabilidade": [selected["v2"], selected["v5"], selected["v6"], selected["v6_treinado"]],
    })
    comparison["Probabilidade"] = comparison["Probabilidade"].apply(fmt_pct)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if selected.get("historico") and str(selected.get("historico")) not in {"nan", "None"}:
        with st.expander("Metadados / histórico usado pelo modelo"):
            st.code(str(selected["historico"]), language="json")

    st.info(
        "O painel mostra a diferença entre probabilidades estimadas e mercado, "
        "mas não valida por si só rentabilidade futura. Usa o paper trading para "
        "medir CLV, ROI, drawdown e estabilidade antes de tirar conclusões."
    )


# ------------------------------------------------------------
# NOTA TÉCNICA
# ------------------------------------------------------------

with st.expander("Como ler esta página"):
    st.write(
        "EV = probabilidade do modelo × odd − 1. A odd justa é 1 / probabilidade. "
        "A concordância mostra quantos modelos estão próximos da média das "
        "probabilidades disponíveis. O painel usa os mesmos jogos e odds para "
        "comparar V2, V5 e V6."
    )

st.divider()
st.caption(
    "FootballAI · análise estatística e paper trading · resultados históricos não garantem resultados futuros."
)
