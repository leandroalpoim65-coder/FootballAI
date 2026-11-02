from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import libsql


st.set_page_config(
    page_title="FootballAI — Match Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
SHADOW_TABLE = "goal_model_home_o25_shadow"
MODEL_ORDER = ["V2_FROZEN", "V5_HOME_FROZEN", "V6_FROZEN", "V6_TRAINED"]
MODEL_COLUMN_MAP = {
    "V2_FROZEN": "v2",
    "V5_HOME_FROZEN": "v5",
    "V6_FROZEN": "v6",
    "V6_TRAINED": "v6_treinado",
}

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1600px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
        .small-note {
            opacity: .72;
            font-size: .9rem;
        }
        .stDataFrame {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_setting(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        value = st.secrets.get(name, "") if hasattr(st, "secrets") else ""
    return str(value or "").strip()


def fmt_pct(value: object, decimals: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_odd(value: object) -> str:
    try:
        if value is None or pd.isna(value):
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


@st.cache_data(ttl=60, show_spinner=False)
def load_shadow() -> pd.DataFrame:
    url = read_setting("TURSO_DATABASE_URL")
    token = read_setting("TURSO_AUTH_TOKEN")
    if not url or not token:
        st.error(
            "Turso credentials were not found in this session. "
            "Define TURSO_DATABASE_URL and TURSO_AUTH_TOKEN and restart the app."
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
        st.error(f"Unable to read data from Turso: {exc}")
        return pd.DataFrame()


def latest_model_rows(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "prediction_timestamp" not in data.columns:
        return data

    result = data.copy()
    result["prediction_timestamp"] = pd.to_datetime(
        result["prediction_timestamp"], errors="coerce"
    )

    keys = [column for column in ["fixture_key", "model_version"] if column in result.columns]
    if keys:
        result = result.sort_values("prediction_timestamp").drop_duplicates(keys, keep="last")
    return result


def build_game_view(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    rows = []
    for fixture_key, group in data.groupby("fixture_key", sort=False):
        group = group.copy()

        def prob(model_name: str):
            model_rows = group.loc[group["model_version"] == model_name, "probability_over"]
            return float(model_rows.iloc[-1]) if not model_rows.empty else None

        p2 = prob("V2_FROZEN")
        p5 = prob("V5_HOME_FROZEN")
        p6 = prob("V6_FROZEN")
        p6t = prob("V6_TRAINED")

        base = group.iloc[-1]
        probability_values = [value for value in [p2, p5, p6, p6t] if value is not None]
        if probability_values:
            mean_probability = sum(probability_values) / len(probability_values)
            consensus = sum(abs(value - mean_probability) <= 0.025 for value in probability_values)
        else:
            consensus = 0

        selected_probability = (
            p6t if p6t is not None else p6 if p6 is not None else p5 if p5 is not None else p2
        )

        odd = pd.to_numeric(pd.Series([base.get("odd_over_at_prediction")]), errors="coerce").iloc[0]
        market_p = pd.to_numeric(
            pd.Series([base.get("market_probability_over_at_prediction")]), errors="coerce"
        ).iloc[0]
        ev = pd.to_numeric(pd.Series([base.get("ev_over_at_prediction")]), errors="coerce").iloc[0]
        fair = 1.0 / selected_probability if selected_probability and 0 < selected_probability < 1 else None

        rows.append(
            {
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
            }
        )

    return pd.DataFrame(rows)


st.title("⚽ FootballAI — Match Analysis")
st.caption(
    "Direct Turso dashboard for statistical comparison, EV analysis, and paper-trading decisions. "
    "Use it as an analysis layer, not as automatic betting advice."
)

if st.button("🔄 Refresh Turso data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    data = latest_model_rows(load_shadow())
    view = build_game_view(data)

    if view.empty:
        st.info("No forecast data is available right now.")
        st.warning(
            "The analysis table is still empty. Run the Home O2.5 shadow process to populate `goal_model_home_o25_shadow`."
        )
        st.stop()

    leagues = sorted(view["liga"].dropna().astype(str).unique().tolist())
    selected_leagues = st.multiselect("Leagues", leagues, default=leagues)

    min_ev = st.slider("Minimum EV", min_value=-0.20, max_value=0.50, value=0.00, step=0.01, format="%.0f%%")
    min_odd = st.number_input("Minimum odds", min_value=1.01, value=1.50, step=0.05)
    max_odd = st.number_input("Maximum odds", min_value=1.01, value=5.00, step=0.10)
    model_choice = st.selectbox(
        "Probability model in the dashboard",
        MODEL_ORDER,
    )
    show_only_positive = st.checkbox("Show only positive EV", value=False)

    filtered = view.copy()
    if selected_leagues:
        filtered = filtered[filtered["liga"].astype(str).isin(selected_leagues)]

    filtered = filtered[pd.to_numeric(filtered["odd"], errors="coerce").between(min_odd, max_odd)]
    filtered = filtered[pd.to_numeric(filtered["ev"], errors="coerce") >= min_ev]
    if show_only_positive:
        filtered = filtered[pd.to_numeric(filtered["ev"], errors="coerce") > 0]

    filtered = filtered.sort_values("ev", ascending=False, na_position="last")
    filtered["prob_selecionada"] = filtered[MODEL_COLUMN_MAP[model_choice]]

# Summary cards
if "filtered" not in locals():
    data = latest_model_rows(load_shadow())
    view = build_game_view(data)
    filtered = view.copy()
    if not view.empty:
        filtered = filtered[pd.to_numeric(filtered["odd"], errors="coerce").between(1.01, 5.00)]
        filtered = filtered[pd.to_numeric(filtered["ev"], errors="coerce") >= 0.00]
        filtered["prob_selecionada"] = filtered[MODEL_COLUMN_MAP[MODEL_ORDER[3]]]

if filtered.empty:
    st.warning("No games match the current filters.")
    st.stop()

summary_cols = st.columns(5)
summary_cols[0].metric("Games analyzed", f"{len(filtered):,}".replace(",", " "))
summary_cols[1].metric("Positive EV", int((pd.to_numeric(filtered["ev"], errors="coerce") > 0).sum()))
summary_cols[2].metric("Average EV", fmt_pct(pd.to_numeric(filtered["ev"], errors="coerce").mean()))
summary_cols[3].metric("Average model probability", fmt_pct(pd.to_numeric(filtered["prob_selecionada"], errors="coerce").mean()))
summary_cols[4].metric("Models available", "V2 · V5 · V6")

st.divider()

# Main view table
st.subheader("Games for analysis")
show = filtered.copy()
show["Game"] = show["casa"].astype(str) + " — " + show["fora"].astype(str)
show["Date"] = show["data"].apply(fmt_date)
show["Model probability"] = show["prob_selecionada"].apply(fmt_pct)
show["Market"] = "Home O2.5"
show["Odds"] = show["odd"].apply(fmt_odd)
show["Fair odds"] = show["odd_justa"].apply(fmt_odd)
show["Market %"] = show["prob_mercado"].apply(fmt_pct)
show["EV"] = show["ev"].apply(fmt_pct)
show["Agreement"] = show["concordancia"]

columns = [
    "Date",
    "hora",
    "liga",
    "Game",
    "Market",
    "Odds",
    "Model probability",
    "Market %",
    "Fair odds",
    "EV",
    "Agreement",
]

st.dataframe(
    show[columns].rename(columns={"hora": "Time", "liga": "League"}),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# Detail panel
st.subheader("Game detail")
options = filtered["fixture_key"].tolist()
labels = {
    row["fixture_key"]: f"{row['casa']} — {row['fora']} · {fmt_date(row['data'])}"
    for _, row in filtered.iterrows()
}
selected_key = st.selectbox(
    "Select a game",
    options,
    format_func=lambda value: labels.get(value, value),
)
selected = filtered[filtered["fixture_key"] == selected_key].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("V2", fmt_pct(selected["v2"]))
col2.metric("V5", fmt_pct(selected["v5"]))
col3.metric("V6 frozen", fmt_pct(selected["v6"]))
col4.metric("V6 trained", fmt_pct(selected["v6_treinado"]))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Odds", fmt_odd(selected["odd"]))
col2.metric("Fair odds", fmt_odd(selected["odd_justa"]))
col3.metric("EV", fmt_pct(selected["ev"]))
col4.metric("Market", fmt_pct(selected["prob_mercado"]))

st.markdown(
    f"**{selected['casa']} — {selected['fora']}** · {selected['liga']} · {fmt_date(selected['data'])} {selected['hora'] or ''}"
)

comparison = pd.DataFrame(
    {
        "Model": ["V2 frozen", "V5 home frozen", "V6 frozen", "V6 trained"],
        "Probability": [selected["v2"], selected["v5"], selected["v6"], selected["v6_treinado"]],
    }
)
comparison["Probability"] = comparison["Probability"].apply(fmt_pct)
st.dataframe(comparison, use_container_width=True, hide_index=True)

if selected.get("historico") and str(selected.get("historico")) not in {"nan", "None"}:
    with st.expander("Metadata / historical context used by the model"):
        st.code(str(selected["historico"]), language="json")

st.info(
    "This panel compares model-implied probabilities against the market and calculates EV. "
    "It does not guarantee future profitability or validate a trading strategy on its own."
)

st.divider()

with st.expander("How to read this page"):
    st.write(
        "EV = model probability × odds − 1. The fair odds is 1 / model probability. "
        "The agreement value shows how many available models are close to the mean probability. "
        "The dashboard compares V2, V5, and V6 using the same matches and odds."
    )

st.caption(
    "FootballAI · statistical analysis and paper trading · past results do not guarantee future performance."
)
