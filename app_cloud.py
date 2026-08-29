from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import libsql
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PÁGINA
# ============================================================

st.set_page_config(
    page_title="FootballAI Cloud",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TIMEZONE = ZoneInfo("Europe/Lisbon")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            background: rgba(128, 128, 128, 0.045);
        }

        [data-testid="stMetricLabel"] {
            font-weight: 650;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }

        .source-card {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin: 0.4rem 0 0.9rem 0;
            background: rgba(128, 128, 128, 0.035);
        }

        @media (max-width: 700px) {
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-top: 0.65rem;
            }

            [data-testid="stMetricValue"] {
                font-size: 1.35rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SEGREDOS / TURSO
# ============================================================

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
        raise RuntimeError(
            "TURSO_DATABASE_URL não configurada nos Secrets do Streamlit."
        )

    if not token:
        raise RuntimeError(
            "TURSO_AUTH_TOKEN não configurado nos Secrets do Streamlit."
        )

    return libsql.connect(
        database=url,
        auth_token=token,
    )


@st.cache_data(ttl=60, show_spinner=False)
def query_dataframe(
    sql: str,
    params: tuple = (),
) -> pd.DataFrame:
    connection = connect_turso()

    try:
        cursor = connection.execute(
            sql,
            params,
        )

        rows = cursor.fetchall()
        columns = [
            item[0]
            for item in (cursor.description or [])
        ]

        return pd.DataFrame(
            rows,
            columns=columns,
        )

    finally:
        connection.close()


@st.cache_data(ttl=60, show_spinner=False)
def list_tables() -> set[str]:
    frame = query_dataframe(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    )

    if frame.empty or "name" not in frame.columns:
        return set()

    return set(
        frame["name"].astype(str).tolist()
    )


# ============================================================
# FORMATAÇÃO
# ============================================================

def number(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):,.{decimals}f}".replace(
        ",",
        " ",
    )


def percentage(value, decimals: int = 1) -> str:
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

    amount = float(value) * float(unit_value)
    return f"€{amount:+,.2f}".replace(
        ",",
        " ",
    )


def safe_numeric(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def clean_status(value) -> str:
    text = str(value or "OPEN").strip().upper()
    return text if text else "OPEN"


def status_badge(value) -> str:
    return (
        "✅ RESOLVIDO"
        if clean_status(value) == "RESOLVED"
        else "🟡 OPEN"
    )


def signal_badge(value) -> str:
    try:
        return (
            "🟢 SINAL"
            if int(value or 0) == 1
            else "⚪ SEM SINAL"
        )
    except (TypeError, ValueError):
        return "⚪ SEM SINAL"


def result_badge(row: pd.Series) -> str:
    if clean_status(row.get("status")) != "RESOLVED":
        return "⏳"

    result = row.get("result_over25")

    if pd.isna(result):
        return "—"

    return "✅ OVER" if int(result) == 1 else "❌ UNDER"


def timestamp_text(value) -> str:
    if value is None or pd.isna(value):
        return "—"

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return str(value)

    try:
        local = parsed.tz_convert(APP_TIMEZONE)
        return local.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return str(value)


# ============================================================
# CLASSIFICAÇÃO DA CLOSE
# ============================================================

def classify_close_source(row: pd.Series) -> str:
    close_odd = row.get("odd_over25_close")

    try:
        valid_close = float(close_odd) > 1.0
    except (TypeError, ValueError):
        valid_close = False

    if not valid_close:
        return "⚪ Sem CLOSE"

    source = str(row.get("closing_source") or "").strip()
    source_lower = source.casefold()

    if (
        "football-data" in source_lower
        or "pós-jogo" in source_lower
        or "pos-jogo" in source_lower
    ):
        return "🟠 Recuperada pós-jogo"

    minutes = row.get("closing_minutes_before_kickoff")

    try:
        if not pd.isna(minutes) and float(minutes) >= 0:
            return "🟢 Capturada pré-jogo"
    except (TypeError, ValueError):
        pass

    if "odds api" in source_lower:
        return "🟢 Capturada pré-jogo"

    return "⚪ CLOSE sem origem classificada"


def close_group(value: str) -> str:
    text = str(value)

    if text.startswith("🟢"):
        return "Pré-jogo"

    if text.startswith("🟠"):
        return "Pós-jogo"

    if "Sem CLOSE" in text:
        return "Sem CLOSE"

    return "Outra"


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def prepare_paper(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = safe_numeric(
        frame,
        [
            "odd_over25_open",
            "odd_under25_open",
            "odd_over25_close",
            "odd_under25_close",
            "fair_probability_open",
            "clv_probability",
            "training_base_rate",
            "signal_threshold",
            "edge_over_base_rate",
            "signal",
            "planned_stake",
            "result_over25",
            "clv",
            "profit",
            "home_goals",
            "away_goals",
            "closing_minutes_before_kickoff",
        ],
    )

    if "fixture_date" in result.columns:
        result["fixture_date_dt"] = pd.to_datetime(
            result["fixture_date"],
            errors="coerce",
        )

    if "status" in result.columns:
        result["status"] = result["status"].apply(
            clean_status
        )

    valid_close = (
        result.get(
            "odd_over25_open",
            pd.Series(np.nan, index=result.index),
        ).gt(1.0)
        & result.get(
            "odd_over25_close",
            pd.Series(np.nan, index=result.index),
        ).gt(1.0)
    )

    result["clv_display"] = np.nan
    result.loc[
        valid_close,
        "clv_display",
    ] = (
        result.loc[valid_close, "odd_over25_open"]
        / result.loc[valid_close, "odd_over25_close"]
        - 1.0
    )

    result["close_type"] = result.apply(
        classify_close_source,
        axis=1,
    )

    result["close_group"] = result[
        "close_type"
    ].apply(close_group)

    return result


def prepare_upcoming(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = safe_numeric(
        frame,
        [
            "odd_over25_open",
            "odd_under25_open",
            "odd_over25_current",
            "odd_under25_current",
            "active",
        ],
    )

    date_text = result.get(
        "date",
        pd.Series("", index=result.index),
    ).astype(str)

    time_text = result.get(
        "time",
        pd.Series("00:00", index=result.index),
    ).fillna("00:00").astype(str)

    result["fixture_datetime"] = pd.to_datetime(
        date_text + " " + time_text,
        errors="coerce",
    )

    now_naive = pd.Timestamp.now(
        tz=APP_TIMEZONE
    ).tz_localize(None)

    result["minutes_to_kickoff"] = (
        result["fixture_datetime"]
        - now_naive
    ).dt.total_seconds() / 60.0

    return result


# ============================================================
# TABELAS AMIGÁVEIS
# ============================================================

def paper_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    display = pd.DataFrame(index=frame.index)

    display["Decisão"] = frame.get(
        "signal",
        pd.Series(0, index=frame.index),
    ).apply(signal_badge)

    display["Estado"] = frame.get(
        "status",
        pd.Series("OPEN", index=frame.index),
    ).apply(status_badge)

    display["Data"] = frame.get(
        "fixture_date",
        pd.Series("", index=frame.index),
    )

    display["Hora"] = frame.get(
        "fixture_time",
        pd.Series("", index=frame.index),
    )

    display["Liga"] = frame.get(
        "league",
        pd.Series("", index=frame.index),
    )

    display["Jogo"] = (
        frame.get(
            "home_team",
            pd.Series("", index=frame.index),
        ).astype(str)
        + " vs "
        + frame.get(
            "away_team",
            pd.Series("", index=frame.index),
        ).astype(str)
    )

    display["OPEN Over"] = frame.get(
        "odd_over25_open",
        pd.Series(np.nan, index=frame.index),
    ).round(3)

    display["CLOSE Over"] = frame.get(
        "odd_over25_close",
        pd.Series(np.nan, index=frame.index),
    ).round(3)

    display["Tipo CLOSE"] = frame.get(
        "close_type",
        pd.Series("⚪ Sem CLOSE", index=frame.index),
    )

    if "clv_probability" in frame.columns:
        display["Prob. CLV+ %"] = (
            frame["clv_probability"] * 100
        ).round(1)

    if "clv_display" in frame.columns:
        display["CLV %"] = (
            frame["clv_display"] * 100
        ).round(2)

    display["Resultado"] = frame.apply(
        result_badge,
        axis=1,
    )

    if "profit" in frame.columns:
        display["Lucro (u)"] = frame[
            "profit"
        ].round(3)

    return display.reset_index(drop=True)


def upcoming_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    display = pd.DataFrame(index=frame.index)

    display["Data"] = frame.get(
        "date",
        pd.Series("", index=frame.index),
    )
    display["Hora"] = frame.get(
        "time",
        pd.Series("", index=frame.index),
    )
    display["Liga"] = frame.get(
        "league",
        pd.Series("", index=frame.index),
    )
    display["Jogo"] = (
        frame.get(
            "home_team",
            pd.Series("", index=frame.index),
        ).astype(str)
        + " vs "
        + frame.get(
            "away_team",
            pd.Series("", index=frame.index),
        ).astype(str)
    )
    display["OPEN Over"] = frame.get(
        "odd_over25_open",
        pd.Series(np.nan, index=frame.index),
    ).round(3)
    display["Atual Over"] = frame.get(
        "odd_over25_current",
        pd.Series(np.nan, index=frame.index),
    ).round(3)
    display["Faltam"] = frame.get(
        "minutes_to_kickoff",
        pd.Series(np.nan, index=frame.index),
    ).apply(
        lambda value: (
            "—"
            if pd.isna(value)
            else (
                f"{int(value // 60)}h {int(value % 60)}m"
                if value >= 60
                else f"{max(0, int(value))} min"
            )
        )
    )

    return display.reset_index(drop=True)


# ============================================================
# CARREGAR BASE CLOUD
# ============================================================

try:
    tables = list_tables()
except Exception as error:
    st.error("Não foi possível ligar à base Turso.")
    st.code(str(error))
    st.info(
        "No Streamlit Community Cloud adiciona TURSO_DATABASE_URL "
        "e TURSO_AUTH_TOKEN em App settings → Secrets."
    )
    st.stop()

if "paper_trades" in tables:
    paper = prepare_paper(
        query_dataframe(
            """
            SELECT *
            FROM paper_trades
            ORDER BY
                fixture_date DESC,
                fixture_time DESC,
                id DESC
            """
        )
    )
else:
    paper = pd.DataFrame()

if "upcoming_matches" in tables:
    upcoming = prepare_upcoming(
        query_dataframe(
            """
            SELECT *
            FROM upcoming_matches
            WHERE active = 1
            ORDER BY
                date,
                time,
                league,
                home_team
            """
        )
    )
else:
    upcoming = pd.DataFrame()

if "matches" in tables:
    history_summary = query_dataframe(
        """
        SELECT
            COUNT(*) AS total_matches,
            COUNT(DISTINCT league) AS total_leagues,
            MAX(date) AS latest_match
        FROM matches
        """
    )
else:
    history_summary = pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

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

league_options = []

if not paper.empty and "league" in paper.columns:
    league_options = sorted(
        paper["league"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_leagues = st.sidebar.multiselect(
    "Filtrar ligas",
    options=league_options,
    default=[],
    placeholder="Todas as ligas",
)

if st.sidebar.button(
    "🔄 Atualizar dados",
    use_container_width=True,
):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "Dados em cache durante 60 segundos."
)


# ============================================================
# FILTRO GLOBAL
# ============================================================

paper_filtered = paper.copy()
upcoming_filtered = upcoming.copy()

if selected_leagues:
    if not paper_filtered.empty:
        paper_filtered = paper_filtered[
            paper_filtered["league"].isin(
                selected_leagues
            )
        ].copy()

    if not upcoming_filtered.empty:
        upcoming_filtered = upcoming_filtered[
            upcoming_filtered["league"].isin(
                selected_leagues
            )
        ].copy()


# ============================================================
# CABEÇALHO
# ============================================================

st.title("⚽ FootballAI Cloud")
st.caption(
    "Paper Trade, closing odds, resultados e CLV — tudo diretamente da Turso."
)


# ============================================================
# RESUMO
# ============================================================

if page == "🏠 Resumo":
    if paper_filtered.empty:
        st.warning(
            "Ainda não existem Paper Trades para apresentar."
        )
    else:
        total_trades = len(paper_filtered)
        total_signals = int(
            pd.to_numeric(
                paper_filtered.get(
                    "signal",
                    pd.Series(0, index=paper_filtered.index),
                ),
                errors="coerce",
            ).fillna(0).sum()
        )

        open_count = int(
            (paper_filtered["status"] == "OPEN").sum()
        )
        resolved_count = int(
            (paper_filtered["status"] == "RESOLVED").sum()
        )
        pregame_count = int(
            (paper_filtered["close_group"] == "Pré-jogo").sum()
        )
        recovered_count = int(
            (paper_filtered["close_group"] == "Pós-jogo").sum()
        )

        signal_resolved = paper_filtered[
            (paper_filtered["signal"] == 1)
            & (paper_filtered["status"] == "RESOLVED")
        ].copy()

        signal_profit = pd.to_numeric(
            signal_resolved.get(
                "profit",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).fillna(0).sum()

        signal_stake = pd.to_numeric(
            signal_resolved.get(
                "planned_stake",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).fillna(0).sum()

        signal_roi = (
            signal_profit / signal_stake
            if signal_stake > 0
            else np.nan
        )

        metric_row_1 = st.columns(4)
        metric_row_1[0].metric(
            "Paper Trades",
            number(total_trades),
        )
        metric_row_1[1].metric(
            "Sinais",
            number(total_signals),
        )
        metric_row_1[2].metric(
            "OPEN",
            number(open_count),
        )
        metric_row_1[3].metric(
            "Resolvidos",
            number(resolved_count),
        )

        metric_row_2 = st.columns(4)
        metric_row_2[0].metric(
            "🟢 CLOSE pré-jogo",
            number(pregame_count),
        )
        metric_row_2[1].metric(
            "🟠 CLOSE pós-jogo",
            number(recovered_count),
        )
        metric_row_2[2].metric(
            "Lucro dos sinais",
            units(signal_profit),
            euros(signal_profit, unit_value),
        )
        metric_row_2[3].metric(
            "ROI dos sinais",
            percentage(signal_roi),
        )

        st.subheader("🎯 Sinais mais recentes")

        signals = paper_filtered[
            paper_filtered["signal"] == 1
        ].head(8)

        if signals.empty:
            st.info("Ainda não existem sinais registados.")
        else:
            st.dataframe(
                paper_display(signals),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("📅 Próximos jogos")

        future = upcoming_filtered[
            upcoming_filtered.get(
                "minutes_to_kickoff",
                pd.Series(np.nan, index=upcoming_filtered.index),
            ) >= 0
        ].head(10)

        if future.empty:
            st.info(
                "Sem próximos jogos ativos neste momento."
            )
        else:
            st.dataframe(
                upcoming_display(future),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# SINAIS
# ============================================================

elif page == "🎯 Sinais":
    st.subheader("🎯 Sinais do modelo")
    st.caption(
        "Aqui aparecem apenas as entradas autorizadas pelo limiar congelado do modelo CLV."
    )

    if paper_filtered.empty:
        st.info("Sem Paper Trades.")
    else:
        signals = paper_filtered[
            paper_filtered["signal"] == 1
        ].copy()

        status_filter = st.segmented_control(
            "Estado",
            options=["Todos", "OPEN", "RESOLVED"],
            default="Todos",
        )

        if status_filter in {"OPEN", "RESOLVED"}:
            signals = signals[
                signals["status"] == status_filter
            ].copy()

        if signals.empty:
            st.info(
                "Nenhum sinal corresponde aos filtros atuais."
            )
        else:
            st.dataframe(
                paper_display(signals),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Exportar sinais CSV",
                data=signals.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="footballai_signals_cloud.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# PAPER TRADES
# ============================================================

elif page == "🧾 Paper Trades":
    st.subheader("🧾 Todos os Paper Trades")

    if paper_filtered.empty:
        st.info("Sem Paper Trades.")
    else:
        filter_columns = st.columns(4)

        status_values = filter_columns[0].multiselect(
            "Estado",
            options=["OPEN", "RESOLVED"],
            default=[],
            placeholder="Todos",
        )

        only_signals = filter_columns[1].toggle(
            "Só sinais",
            value=False,
        )

        close_filter = filter_columns[2].selectbox(
            "Tipo de CLOSE",
            [
                "Todas",
                "Pré-jogo",
                "Pós-jogo",
                "Sem CLOSE",
            ],
        )

        sort_mode = filter_columns[3].selectbox(
            "Ordenar por",
            [
                "Mais recentes",
                "Maior prob. CLV+",
                "Maior CLV observado",
            ],
        )

        displayed = paper_filtered.copy()

        if status_values:
            displayed = displayed[
                displayed["status"].isin(status_values)
            ].copy()

        if only_signals:
            displayed = displayed[
                displayed["signal"] == 1
            ].copy()

        if close_filter != "Todas":
            displayed = displayed[
                displayed["close_group"] == close_filter
            ].copy()

        if sort_mode == "Maior prob. CLV+":
            displayed = displayed.sort_values(
                "clv_probability",
                ascending=False,
                na_position="last",
            )
        elif sort_mode == "Maior CLV observado":
            displayed = displayed.sort_values(
                "clv_display",
                ascending=False,
                na_position="last",
            )
        else:
            displayed = displayed.sort_values(
                [
                    "fixture_date_dt",
                    "fixture_time",
                    "id",
                ],
                ascending=False,
                na_position="last",
            )

        st.caption(
            f"{len(displayed)} registos mostrados."
        )

        st.dataframe(
            paper_display(displayed),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Exportar Paper Trades CSV",
            data=displayed.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="footballai_paper_trades_cloud.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# UPCOMING
# ============================================================

elif page == "📅 Próximos Jogos":
    st.subheader("📅 Próximos jogos e odds")
    st.caption(
        "OPEN é a primeira odd observada; Atual é a última observação segura do downloader cloud."
    )

    if upcoming_filtered.empty:
        st.info(
            "Não existem jogos ativos na tabela upcoming_matches."
        )
    else:
        future = upcoming_filtered[
            upcoming_filtered.get(
                "minutes_to_kickoff",
                pd.Series(np.nan, index=upcoming_filtered.index),
            ) >= 0
        ].copy()

        horizon = st.selectbox(
            "Horizonte",
            [
                "Todos",
                "Próximas 24 horas",
                "Próximos 3 dias",
                "Próximos 7 dias",
            ],
        )

        horizon_minutes = {
            "Próximas 24 horas": 24 * 60,
            "Próximos 3 dias": 3 * 24 * 60,
            "Próximos 7 dias": 7 * 24 * 60,
        }.get(horizon)

        if horizon_minutes is not None:
            future = future[
                future["minutes_to_kickoff"]
                <= horizon_minutes
            ].copy()

        if future.empty:
            st.info(
                "Nenhum jogo dentro do horizonte escolhido."
            )
        else:
            st.dataframe(
                upcoming_display(future),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# CLV / PERFORMANCE
# ============================================================

elif page == "📈 CLV & Performance":
    st.subheader("📈 CLV e performance")

    st.info(
        "A clv_probability estima a probabilidade de obter CLV positivo. "
        "Não é a probabilidade de o jogo terminar Over 2.5."
    )

    if paper_filtered.empty:
        st.info("Sem dados suficientes.")
    else:
        all_close_rows = paper_filtered[
            paper_filtered["clv_display"].notna()
        ].copy()

        pregame_rows = all_close_rows[
            all_close_rows["close_group"] == "Pré-jogo"
        ].copy()

        recovered_rows = all_close_rows[
            all_close_rows["close_group"] == "Pós-jogo"
        ].copy()

        coverage_metrics = st.columns(3)
        coverage_metrics[0].metric(
            "CLOSE válidas",
            number(len(all_close_rows)),
        )
        coverage_metrics[1].metric(
            "🟢 Pré-jogo",
            number(len(pregame_rows)),
        )
        coverage_metrics[2].metric(
            "🟠 Recuperadas pós-jogo",
            number(len(recovered_rows)),
        )

        st.markdown("#### Qualidade da amostra CLV")

        close_mode = st.segmented_control(
            "Analisar",
            options=[
                "Todas",
                "🟢 Pré-jogo",
                "🟠 Pós-jogo",
            ],
            default="Todas",
        )

        if close_mode == "🟢 Pré-jogo":
            close_rows = pregame_rows.copy()
            st.success(
                "Amostra de maior qualidade: closing odds capturadas antes do kickoff."
            )
        elif close_mode == "🟠 Pós-jogo":
            close_rows = recovered_rows.copy()
            st.warning(
                "Estas CLOSE foram recuperadas posteriormente no Football-Data. "
                "São úteis para análise, mas podem usar uma agregação de casas diferente da OPEN."
            )
        else:
            close_rows = all_close_rows.copy()
            if not recovered_rows.empty:
                st.warning(
                    "A vista 'Todas' mistura CLOSE pré-jogo com CLOSE recuperadas pós-jogo. "
                    "Para decisões sobre o modelo, dá mais peso à amostra pré-jogo."
                )

        close_metrics = st.columns(4)
        close_metrics[0].metric(
            "Jogos analisados",
            number(len(close_rows)),
        )
        close_metrics[1].metric(
            "CLV médio",
            percentage(
                close_rows["clv_display"].mean()
                if not close_rows.empty
                else np.nan
            ),
        )
        close_metrics[2].metric(
            "CLV positivo",
            percentage(
                (close_rows["clv_display"] > 0).mean()
                if not close_rows.empty
                else np.nan
            ),
        )
        close_metrics[3].metric(
            "CLV mediano",
            percentage(
                close_rows["clv_display"].median()
                if not close_rows.empty
                else np.nan
            ),
        )

        if len(close_rows) < 50:
            st.warning(
                "A amostra selecionada ainda é pequena. "
                "Evita alterar o limiar do modelo apenas com base nestes jogos."
            )

        if not close_rows.empty:
            source_summary = (
                close_rows
                .groupby(
                    ["close_type", "closing_source"],
                    dropna=False,
                )
                .agg(
                    Jogos=("id", "count"),
                    CLV_medio=("clv_display", "mean"),
                    CLV_positivo=(
                        "clv_display",
                        lambda series: (series > 0).mean(),
                    ),
                )
                .reset_index()
            )

            source_summary["CLV médio %"] = (
                source_summary["CLV_medio"] * 100
            ).round(2)
            source_summary["CLV positivo %"] = (
                source_summary["CLV_positivo"] * 100
            ).round(1)
            source_summary["Origem"] = source_summary[
                "closing_source"
            ].fillna("Sem descrição")
            source_summary["Tipo"] = source_summary[
                "close_type"
            ]

            with st.expander(
                "Ver origem das closing odds",
                expanded=False,
            ):
                st.dataframe(
                    source_summary[
                        [
                            "Tipo",
                            "Origem",
                            "Jogos",
                            "CLV positivo %",
                            "CLV médio %",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            bins = [
                -0.001,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                1.001,
            ]
            labels = [
                "<30%",
                "30–35%",
                "35–40%",
                "40–45%",
                "45–50%",
                "50–55%",
                "55%+",
            ]

            calibration = close_rows.copy()
            calibration["Faixa"] = pd.cut(
                calibration["clv_probability"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )
            calibration["CLV positivo"] = (
                calibration["clv_display"] > 0
            ).astype(int)

            calibration_table = (
                calibration
                .dropna(subset=["Faixa"])
                .groupby(
                    "Faixa",
                    observed=False,
                )
                .agg(
                    Jogos=("id", "count"),
                    Prob_prevista=(
                        "clv_probability",
                        "mean",
                    ),
                    CLV_positivo=(
                        "CLV positivo",
                        "mean",
                    ),
                    CLV_medio=(
                        "clv_display",
                        "mean",
                    ),
                )
                .reset_index()
            )

            calibration_table[
                "Prob. prevista %"
            ] = (
                calibration_table["Prob_prevista"]
                * 100
            ).round(1)

            calibration_table[
                "CLV positivo %"
            ] = (
                calibration_table["CLV_positivo"]
                * 100
            ).round(1)

            calibration_table[
                "CLV médio %"
            ] = (
                calibration_table["CLV_medio"]
                * 100
            ).round(2)

            st.markdown("#### Calibração real do modelo CLV")
            st.caption(
                "Compara a probabilidade prevista de CLV+ com a percentagem real de closing odds positivas."
            )

            st.dataframe(
                calibration_table[
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

            chart = calibration_table.set_index(
                "Faixa"
            )[
                [
                    "Prob. prevista %",
                    "CLV positivo %",
                ]
            ]
            st.bar_chart(chart)

        resolved_signals = paper_filtered[
            (paper_filtered["signal"] == 1)
            & (paper_filtered["status"] == "RESOLVED")
        ].copy()

        st.markdown("#### Resultado dos sinais")

        signal_metrics = st.columns(4)
        signal_metrics[0].metric(
            "Sinais resolvidos",
            number(len(resolved_signals)),
        )

        signal_profit = pd.to_numeric(
            resolved_signals.get(
                "profit",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).fillna(0)

        signal_stakes = pd.to_numeric(
            resolved_signals.get(
                "planned_stake",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).fillna(0)

        total_profit = signal_profit.sum()
        total_stake = signal_stakes.sum()

        signal_metrics[1].metric(
            "Lucro",
            units(total_profit),
            euros(total_profit, unit_value),
        )
        signal_metrics[2].metric(
            "ROI",
            percentage(
                total_profit / total_stake
                if total_stake > 0
                else np.nan
            ),
        )
        signal_metrics[3].metric(
            "Over 2.5 acertado",
            percentage(
                pd.to_numeric(
                    resolved_signals.get(
                        "result_over25",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                ).mean()
            ),
        )

        if not resolved_signals.empty:
            curve = resolved_signals.copy()
            curve["profit_numeric"] = pd.to_numeric(
                curve["profit"],
                errors="coerce",
            ).fillna(0)
            curve = curve.sort_values(
                ["fixture_date_dt", "id"]
            )
            curve["Lucro acumulado (u)"] = (
                curve["profit_numeric"].cumsum()
            )
            curve = curve.set_index(
                "fixture_date_dt"
            )

            st.line_chart(
                curve[["Lucro acumulado (u)"]]
            )


# ============================================================
# ESTADO DO SISTEMA
# ============================================================

elif page == "⚙️ Estado do Sistema":
    st.subheader("⚙️ Estado do sistema cloud")

    st.success("✅ Ligação à Turso ativa")

    total_history = np.nan
    latest_history = "—"
    total_leagues = np.nan

    if not history_summary.empty:
        summary = history_summary.iloc[0]
        total_history = summary.get("total_matches")
        total_leagues = summary.get("total_leagues")
        latest_history = str(
            summary.get("latest_match") or "—"
        )

    system_metrics = st.columns(4)
    system_metrics[0].metric(
        "Jogos históricos",
        number(total_history),
    )
    system_metrics[1].metric(
        "Ligas históricas",
        number(total_leagues),
    )
    system_metrics[2].metric(
        "Último histórico",
        latest_history,
    )
    system_metrics[3].metric(
        "Upcoming ativos",
        number(len(upcoming)),
    )

    st.markdown("#### Cobertura das closing odds")

    if paper.empty:
        st.info("Sem Paper Trades.")
    else:
        coverage = st.columns(4)
        coverage[0].metric(
            "Total Paper Trades",
            number(len(paper)),
        )
        coverage[1].metric(
            "🟢 Pré-jogo",
            number(
                (paper["close_group"] == "Pré-jogo").sum()
            ),
        )
        coverage[2].metric(
            "🟠 Pós-jogo",
            number(
                (paper["close_group"] == "Pós-jogo").sum()
            ),
        )
        coverage[3].metric(
            "Sem CLOSE",
            number(
                (paper["close_group"] == "Sem CLOSE").sum()
            ),
        )

        source_counts = (
            paper[
                paper["clv_display"].notna()
            ]
            .groupby(
                ["close_type", "closing_source"],
                dropna=False,
            )
            .size()
            .reset_index(name="Jogos")
        )

        if not source_counts.empty:
            source_counts["Origem"] = source_counts[
                "closing_source"
            ].fillna("Sem descrição")
            source_counts["Tipo"] = source_counts[
                "close_type"
            ]

            st.dataframe(
                source_counts[
                    ["Tipo", "Origem", "Jogos"]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("#### Últimas atualizações observadas")

    health_rows = []

    if not upcoming.empty:
        for label, column in (
            (
                "Upcoming / odds atuais",
                "current_observed_at",
            ),
            (
                "Primeira observação upcoming",
                "first_observed_at",
            ),
        ):
            if column in upcoming.columns:
                values = upcoming[column].dropna()
                health_rows.append(
                    {
                        "Componente": label,
                        "Última atualização": (
                            timestamp_text(values.max())
                            if not values.empty
                            else "—"
                        ),
                    }
                )

    if not paper.empty:
        for label, column in (
            (
                "Paper Trade",
                "prediction_timestamp",
            ),
            (
                "Closing odds pré-jogo",
                "closing_observed_at",
            ),
        ):
            if column in paper.columns:
                values = paper[column].dropna()
                health_rows.append(
                    {
                        "Componente": label,
                        "Última atualização": (
                            timestamp_text(values.max())
                            if not values.empty
                            else "—"
                        ),
                    }
                )

        resolved_dates = paper.loc[
            paper["status"] == "RESOLVED",
            "fixture_date",
        ].dropna()

        health_rows.append(
            {
                "Componente": "Último jogo resolvido",
                "Última atualização": (
                    str(resolved_dates.max())
                    if not resolved_dates.empty
                    else "—"
                ),
            }
        )

    if health_rows:
        st.dataframe(
            pd.DataFrame(health_rows),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Ver tabelas disponíveis na Turso"):
        st.write(sorted(tables))

    st.caption(
        "A dashboard é apenas de leitura: não altera previsões, odds, resultados ou o modelo."
    )


st.divider()
st.caption(
    "FootballAI · Paper trading apenas · "
    + datetime.now(APP_TIMEZONE).strftime(
        "Atualizado no ecrã em %d/%m/%Y às %H:%M"
    )
)
