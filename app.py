from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="FootballAI Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = ROOT_DIR / "database" / "football.db"
REPORTS_DIR = ROOT_DIR / "reports"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ML_DIR = ROOT_DIR / "ml"
UPCOMING_PATH = (
    ROOT_DIR
    / "data"
    / "upcoming"
    / "upcoming_matches.csv"
)

PREDICTIONS_PATH = REPORTS_DIR / "clv_predictions_over25.csv"
BOOTSTRAP_SUMMARY_PATH = REPORTS_DIR / "bootstrap_clv_summary.csv"
BOOTSTRAP_LEAGUES_PATH = REPORTS_DIR / "bootstrap_clv_by_league.csv"
BOOTSTRAP_SEASONS_PATH = REPORTS_DIR / "bootstrap_clv_by_season.csv"
FEATURE_METADATA_PATH = PROCESSED_DIR / "feature_metadata.json"
MODEL_METADATA_PATH = ML_DIR / "metadata_modelo_clv_over25.json"


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.85rem;
        }

        [data-testid="stMetricLabel"] {
            font-weight: 600;
        }

        .status-card {
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
        }

        .status-ok {
            background: rgba(24, 160, 88, 0.10);
        }

        .status-warning {
            background: rgba(236, 164, 28, 0.12);
        }

        .status-danger {
            background: rgba(210, 55, 55, 0.10);
        }

        .small-note {
            opacity: 0.75;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def format_percent(
    value,
    decimals: int = 2,
) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value) * 100:.{decimals}f}%"


def format_number(
    value,
    decimals: int = 0,
) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):,.{decimals}f}".replace(
        ",",
        " ",
    )


def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for column in result.columns
    ]

    return result


@st.cache_data(
    show_spinner=False,
)
def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        low_memory=False,
    )


@st.cache_data(
    show_spinner=False,
)
def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@st.cache_data(
    show_spinner=False,
)
def list_database_tables() -> list[str]:
    if not DATABASE_PATH.exists():
        return []

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

    return [
        str(row[0])
        for row in rows
    ]


@st.cache_data(
    show_spinner=False,
)
def read_database_query(
    query: str,
) -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        return pd.read_sql_query(
            query,
            connection,
        )


def table_exists(
    table_name: str,
) -> bool:
    return table_name in list_database_tables()


def find_first_existing_table(
    candidates: list[str],
) -> str | None:
    existing = set(
        list_database_tables()
    )

    for candidate in candidates:
        if candidate in existing:
            return candidate

    return None


def save_prediction(
    match_name: str,
    market: str,
    prediction: str,
    probability: float,
) -> None:
    """
    Guarda uma previsão manual quando a tabela antiga existe.

    A dashboard não cria apostas reais.
    """
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Base de dados não encontrada: {DATABASE_PATH}"
        )

    if not table_exists(
        "historico_previsoes"
    ):
        raise RuntimeError(
            "A tabela historico_previsoes não existe."
        )

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        connection.execute(
            """
            INSERT INTO historico_previsoes (
                data_jogo,
                partida,
                mercado,
                previsao,
                probabilidade
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                date.today().isoformat(),
                match_name,
                market,
                prediction,
                float(probability),
            ),
        )

        connection.commit()


def detect_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


def prepare_prediction_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    result = normalize_columns(
        dataframe
    )

    date_column = detect_column(
        result,
        [
            "date",
            "data",
            "match_date",
        ],
    )

    if date_column is not None:
        result[date_column] = pd.to_datetime(
            result[date_column],
            errors="coerce",
        )

    numeric_candidates = [
        "probability",
        "probabilidade",
        "clv",
        "profit",
        "lucro",
        "odd_over25_open",
        "odd_over25_close",
        "opening_odd",
        "closing_odd",
        "edge",
        "signal",
    ]

    for column in numeric_candidates:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def selection_mask(
    dataframe: pd.DataFrame,
) -> pd.Series:
    signal_column = detect_column(
        dataframe,
        [
            "signal",
            "sinal",
            "selected",
            "selection",
            "selecao",
        ],
    )

    if signal_column is None:
        return pd.Series(
            True,
            index=dataframe.index,
        )

    numeric = pd.to_numeric(
        dataframe[signal_column],
        errors="coerce",
    )

    if numeric.notna().any():
        return numeric.fillna(0) > 0

    text = (
        dataframe[signal_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return text.isin(
        [
            "true",
            "yes",
            "sim",
            "selected",
            "selection",
            "sinal",
            "1",
        ]
    )


def display_status_card(
    title: str,
    message: str,
    level: str,
) -> None:
    css_class = {
        "ok": "status-ok",
        "warning": "status-warning",
        "danger": "status-danger",
    }.get(
        level,
        "status-warning",
    )

    st.markdown(
        f"""
        <div class="status-card {css_class}">
            <strong>{title}</strong><br>
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

predictions_raw = load_csv(
    PREDICTIONS_PATH
)

predictions = prepare_prediction_report(
    predictions_raw
)

bootstrap_summary = normalize_columns(
    load_csv(
        BOOTSTRAP_SUMMARY_PATH
    )
)

bootstrap_leagues = normalize_columns(
    load_csv(
        BOOTSTRAP_LEAGUES_PATH
    )
)

bootstrap_seasons = normalize_columns(
    load_csv(
        BOOTSTRAP_SEASONS_PATH
    )
)

feature_metadata = load_json(
    FEATURE_METADATA_PATH
)

model_metadata = load_json(
    MODEL_METADATA_PATH
)




def run_project_script(relative_path: str) -> tuple[bool, str]:
    """
    Executa um script do projeto com o mesmo Python/venv da dashboard.
    """
    script_path = ROOT_DIR / relative_path

    if not script_path.exists():
        return False, f"Ficheiro não encontrado: {relative_path}"

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False, f"{relative_path}: excedeu 180 segundos."
    except Exception as error:
        return False, f"{relative_path}: {error}"

    output = "\n".join(
        part.strip()
        for part in [completed.stdout, completed.stderr]
        if part and part.strip()
    )

    if completed.returncode != 0:
        return False, output or (
            f"{relative_path} terminou com código "
            f"{completed.returncode}."
        )

    return True, output or f"{relative_path}: concluído."


def run_full_update() -> tuple[bool, str]:
    """
    Pipeline completo e seguro:

      1) tenta atualizar jogos/odds;
      2) só executa capture_closing_odds.py se o CSV tiver sido
         realmente reescrito pelo downloader;
      3) só cria/congela novas previsões se o downloader tiver
         produzido dados frescos;
      4) update_results.py é executado separadamente, porque pode
         resolver jogos antigos mesmo quando não existem novos
         jogos/odds.

    O download_upcoming.py foi desenhado para NÃO reescrever
    upcoming_matches.csv quando a Odds API falha ou está
    incompleta. Portanto, a data de modificação do ficheiro é
    usada aqui como confirmação adicional de uma atualização
    válida.
    """

    logs: list[str] = []
    all_ok = True

    # ========================================================
    # 1) DOWNLOAD UPCOMING
    # ========================================================

    before_mtime: float | None = None

    if UPCOMING_PATH.exists():
        try:
            before_mtime = (
                UPCOMING_PATH
                .stat()
                .st_mtime
            )
        except OSError:
            before_mtime = None

    download_ok, download_output = run_project_script(
        "download_upcoming.py"
    )

    logs.append(
        "===== download_upcoming.py =====\n"
        f"{download_output}"
    )

    after_mtime: float | None = None

    if UPCOMING_PATH.exists():
        try:
            after_mtime = (
                UPCOMING_PATH
                .stat()
                .st_mtime
            )
        except OSError:
            after_mtime = None

    cancellation_markers = (
        "ATUALIZAÇÃO CANCELADA COM SEGURANÇA",
        "ATUALIZACAO CANCELADA COM SEGURANCA",
        "NÃO será reescrito",
        "NAO sera reescrito",
        "CSV preservado:",
    )

    download_cancelled_safely = any(
        marker in download_output
        for marker in cancellation_markers
    )

    if before_mtime is None:
        csv_was_refreshed = (
            after_mtime is not None
            and not download_cancelled_safely
        )
    else:
        csv_was_refreshed = (
            after_mtime is not None
            and after_mtime > before_mtime
            and not download_cancelled_safely
        )

    download_usable = (
        download_ok
        and csv_was_refreshed
    )

    if not download_usable:
        all_ok = False

    # ========================================================
    # 2) CAPTURE + PAPER TRADE
    # ========================================================

    if download_usable:

        capture_ok, capture_output = run_project_script(
            "capture_closing_odds.py"
        )

        logs.append(
            "===== capture_closing_odds.py =====\n"
            f"{capture_output}"
        )

        all_ok = (
            all_ok
            and capture_ok
        )

        paper_ok, paper_output = run_project_script(
            "ml/paper_trade.py"
        )

        logs.append(
            "===== ml/paper_trade.py =====\n"
            f"{paper_output}"
        )

        all_ok = (
            all_ok
            and paper_ok
        )

    else:

        logs.append(
            "===== capture_closing_odds.py =====\n"
            "IGNORADO COM SEGURANÇA.\n"
            "download_upcoming.py não produziu um "
            "upcoming_matches.csv novo e fresco.\n"
            "As closing odds antigas não serão tratadas "
            "como atuais."
        )

        logs.append(
            "===== ml/paper_trade.py =====\n"
            "IGNORADO COM SEGURANÇA.\n"
            "Sem uma atualização válida do downloader, "
            "não são criadas novas previsões."
        )

    # ========================================================
    # 3) RESULTADOS — ETAPA INDEPENDENTE
    # ========================================================
    #
    # update_results.py tem o seu próprio fail-safe.
    # Pode ser útil mesmo que o download de jogos futuros tenha
    # falhado, por isso é executado separadamente.
    # ========================================================

    results_ok, results_output = run_project_script(
        "update_results.py"
    )

    logs.append(
        "===== update_results.py =====\n"
        f"{results_output}"
    )

    results_cancelled_safely = any(
        marker in results_output
        for marker in (
            "ATUALIZAÇÃO CANCELADA COM SEGURANÇA",
            "ATUALIZACAO CANCELADA COM SEGURANCA",
        )
    )

    if (
        not results_ok
        or results_cancelled_safely
    ):
        all_ok = False

    # ========================================================
    # RESUMO DO PIPELINE
    # ========================================================

    if download_usable:

        pipeline_summary = (
            "DOWNLOAD: atualização válida e CSV fresco.\n"
            "CAPTURE: executado.\n"
            "PAPER TRADE: executado.\n"
            "RESULTADOS: executado separadamente."
        )

    else:

        pipeline_summary = (
            "DOWNLOAD: sem atualização válida do CSV.\n"
            "CAPTURE: ignorado com segurança.\n"
            "PAPER TRADE: ignorado com segurança.\n"
            "RESULTADOS: executado separadamente."
        )

    logs.insert(
        0,
        "===== RESUMO DO PIPELINE =====\n"
        f"{pipeline_summary}"
    )

    return (
        all_ok,
        "\n\n".join(
            logs
        ),
    )


# ============================================================
# CABEÇALHO
# ============================================================

st.title(
    "⚽ FootballAI — Intelligence Engine"
)

st.caption(
    "Análise de CLV, validação estatística e paper trading. "
    "Sem execução de apostas reais."
)

st.divider()


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.header(
    "Controlo"
)

st.sidebar.caption(
    "Modelo congelado para validação futura."
)

if st.sidebar.button(
    "🔄 Atualizar dados",
    use_container_width=True,
):
    with st.sidebar.status(
        "A atualizar o FootballAI...",
        expanded=True,
    ) as update_status:
        ok, update_log = run_full_update()

        if ok:
            update_status.update(
                label="Atualização concluída",
                state="complete",
                expanded=False,
            )
            st.sidebar.success(
                "Jogos, previsões e resultados atualizados."
            )
        else:
            update_status.update(
                label="Atualização concluída com avisos",
                state="error",
                expanded=True,
            )
            st.sidebar.warning(
                "Uma ou mais etapas tiveram problemas."
            )

        with st.sidebar.expander(
            "Ver relatório da atualização"
        ):
            st.code(update_log)

    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

available_leagues: list[str] = []

if (
    not predictions.empty
    and "league" in predictions.columns
):
    available_leagues = sorted(
        predictions["league"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_leagues = st.sidebar.multiselect(
    "Ligas",
    options=available_leagues,
    default=available_leagues,
)

selected_seasons: list[str] = []

if (
    not predictions.empty
    and "season" in predictions.columns
):
    season_options = sorted(
        predictions["season"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_seasons = st.sidebar.multiselect(
        "Épocas",
        options=season_options,
        default=season_options,
    )

filtered_predictions = predictions.copy()

if (
    selected_leagues
    and "league" in filtered_predictions.columns
):
    filtered_predictions = filtered_predictions[
        filtered_predictions["league"].isin(
            selected_leagues
        )
    ].copy()

if (
    selected_seasons
    and "season" in filtered_predictions.columns
):
    filtered_predictions = filtered_predictions[
        filtered_predictions["season"]
        .astype(str)
        .isin(selected_seasons)
    ].copy()

if not filtered_predictions.empty:
    filtered_selections = filtered_predictions[
        selection_mask(
            filtered_predictions
        )
    ].copy()
else:
    filtered_selections = pd.DataFrame()


# ============================================================
# ABAS
# ============================================================

(
    overview_tab,
    paper_tab,
    validation_tab,
    league_tab,
    history_tab,
    quality_tab,
) = st.tabs(
    [
        "Visão geral",
        "Paper trading",
        "Validação",
        "Ligas e épocas",
        "Histórico",
        "Qualidade dos dados",
    ]
)


# ============================================================
# VISÃO GERAL
# ============================================================

with overview_tab:
    model_threshold = (
        model_metadata.get(
            "frozen_threshold"
        )
        or model_metadata.get(
            "threshold"
        )
        or model_metadata.get(
            "limiar"
        )
        or 0.55
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Jogos processados",
        format_number(
            feature_metadata.get(
                "processed_rows"
            )
        ),
    )

    metric_columns[1].metric(
        "Features",
        format_number(
            feature_metadata.get(
                "feature_count",
                43,
            )
        ),
    )

    metric_columns[2].metric(
        "Ligas",
        (
            filtered_predictions[
                "league"
            ].nunique()
            if (
                not filtered_predictions.empty
                and "league"
                in filtered_predictions.columns
            )
            else "—"
        ),
    )

    metric_columns[3].metric(
        "Sinais OOS",
        format_number(
            len(filtered_selections)
        ),
    )

    metric_columns[4].metric(
        "Limiar congelado",
        format_percent(
            model_threshold
        ),
    )

    st.subheader(
        "Estado do modelo"
    )

    left_status, right_status = st.columns(
        2
    )

    with left_status:
        display_status_card(
            "✅ CLV validado",
            (
                "O intervalo bootstrap do CLV médio ficou "
                "totalmente acima de zero."
            ),
            "ok",
        )

    with right_status:
        display_status_card(
            "⚠ Rentabilidade não validada",
            (
                "O intervalo bootstrap do ROI inclui zero e "
                "a probabilidade de ROI positivo é baixa."
            ),
            "warning",
        )

    if not bootstrap_summary.empty:
        row = bootstrap_summary.iloc[0]

        st.subheader(
            "Resumo estatístico"
        )

        summary_columns = st.columns(4)

        summary_columns[0].metric(
            "CLV médio",
            format_percent(
                row.get(
                    "clv_mean"
                )
            ),
        )

        summary_columns[1].metric(
            "IC 95% CLV",
            (
                f"{format_percent(row.get('clv_ci_lower'))}"
                f" a "
                f"{format_percent(row.get('clv_ci_upper'))}"
            ),
        )

        summary_columns[2].metric(
            "ROI",
            format_percent(
                row.get(
                    "roi"
                )
            ),
        )

        summary_columns[3].metric(
            "Prob. ROI > 0",
            format_percent(
                row.get(
                    "probability_roi_positive"
                )
            ),
        )

    st.info(
        "Decisão operacional: manter o modelo congelado, "
        "usar stake fictícia fixa e recolher novos sinais "
        "antes de qualquer novo treino."
    )


# ============================================================
# PAPER TRADING
# ============================================================

with paper_tab:
    st.subheader(
        "Paper trading"
    )

    paper_table = find_first_existing_table(
        [
            "paper_trades",
            "paper_trade",
            "paper_trading",
            "historico_paper_trade",
        ]
    )

    if paper_table is not None:
        paper_data = read_database_query(
            f"""
            SELECT *
            FROM {paper_table}
            ORDER BY id DESC
            """
        )

        paper_data = normalize_columns(
            paper_data
        )

        top_columns = st.columns(4)

        top_columns[0].metric(
            "Registos",
            format_number(
                len(paper_data)
            ),
        )

        settled_column = detect_column(
            paper_data,
            [
                "settled",
                "resolvido",
                "closed",
                "finalizado",
                "status",
            ],
        )

        profit_column = detect_column(
            paper_data,
            [
                "profit",
                "lucro",
                "pnl",
            ],
        )

        clv_column = detect_column(
            paper_data,
            [
                "clv",
                "clv_value",
            ],
        )

        if settled_column is not None:
            settled_series = paper_data[
                settled_column
            ]

            if settled_column == "status":
                settled_count = int(
                    settled_series
                    .astype("string")
                    .str.strip()
                    .str.upper()
                    .isin(
                        [
                            "RESOLVED",
                            "SETTLED",
                            "CLOSED",
                            "FINALIZADO",
                            "RESOLVIDO",
                        ]
                    )
                    .sum()
                )
            else:
                numeric_settled = pd.to_numeric(
                    settled_series,
                    errors="coerce",
                )

                if numeric_settled.notna().any():
                    settled_count = int(
                        numeric_settled
                        .fillna(0)
                        .astype(bool)
                        .sum()
                    )
                else:
                    settled_count = int(
                        settled_series
                        .astype("string")
                        .str.strip()
                        .str.lower()
                        .isin(
                            [
                                "true",
                                "yes",
                                "sim",
                                "resolved",
                                "settled",
                                "closed",
                                "finalizado",
                                "resolvido",
                                "1",
                            ]
                        )
                        .sum()
                    )
        else:
            settled_count = 0

        top_columns[1].metric(
            "Resolvidos",
            format_number(
                settled_count
            ),
        )

        if profit_column is not None:
            profit_values = pd.to_numeric(
                paper_data[
                    profit_column
                ],
                errors="coerce",
            )

            top_columns[2].metric(
                "Lucro fictício",
                format_number(
                    profit_values.sum(),
                    2,
                ),
            )
        else:
            top_columns[2].metric(
                "Lucro fictício",
                "—",
            )

        if clv_column is not None:
            clv_values = pd.to_numeric(
                paper_data[
                    clv_column
                ],
                errors="coerce",
            )

            top_columns[3].metric(
                "CLV médio futuro",
                format_percent(
                    clv_values.mean()
                ),
            )
        else:
            top_columns[3].metric(
                "CLV médio futuro",
                "—",
            )

        # ====================================================
        # PAINEL DE DECISÕES
        # ====================================================

        paper_display = paper_data.copy()

        signal_numeric = (
            pd.to_numeric(
                paper_display["signal"],
                errors="coerce",
            ).fillna(0)
            if "signal" in paper_display.columns
            else pd.Series(
                0,
                index=paper_display.index,
                dtype=float,
            )
        )

        paper_display.insert(
            0,
            "decisao",
            signal_numeric.apply(
                lambda value: (
                    "🟢 SINAL"
                    if value > 0
                    else "⚪ SEM SINAL"
                )
            ),
        )

        if "clv_probability" in paper_display.columns:
            paper_display[
                "_sort_clv_probability"
            ] = pd.to_numeric(
                paper_display[
                    "clv_probability"
                ],
                errors="coerce",
            )

            paper_display = (
                paper_display
                .sort_values(
                    "_sort_clv_probability",
                    ascending=False,
                    na_position="last",
                )
                .drop(
                    columns=[
                        "_sort_clv_probability"
                    ]
                )
                .reset_index(drop=True)
            )

        preferred_columns = [
            "decisao",
            "fixture_date",
            "fixture_time",
            "league",
            "home_team",
            "away_team",
            "odd_over25_open",
            "odd_under25_open",
            "fair_probability_open",
            "clv_probability",
            "signal_threshold",
            "edge_over_base_rate",
            "signal",
            "planned_stake",
            "status",
            "home_goals",
            "away_goals",
            "result_over25",
            "clv",
            "profit",
        ]

        visible_columns = [
            column
            for column in preferred_columns
            if column in paper_display.columns
        ]

        st.markdown(
            "#### Decisões do modelo"
        )

        st.caption(
            "Ordenado por probabilidade de CLV do maior para o menor. "
            "🟢 SINAL = entrada autorizada pelo modelo congelado; "
            "⚪ SEM SINAL = acompanhar apenas."
        )

        st.dataframe(
            paper_display[
                visible_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Ver todos os campos técnicos"
        ):
            st.dataframe(
                paper_display,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.warning(
            "Ainda não foi encontrada uma tabela de paper trading "
            "na base de dados."
        )

        st.caption(
            "Quando o paper_trade.py começar a guardar sinais, "
            "esta área passa a apresentar os resultados automaticamente."
        )

    with st.expander(
        "Registar previsão manual antiga"
    ):
        if table_exists(
            "historico_previsoes"
        ):
            with st.form(
                "manual_prediction_form"
            ):
                match_name = st.text_input(
                    "Partida"
                )

                market = st.selectbox(
                    "Mercado",
                    [
                        "Over 2.5",
                        "BTTS",
                    ],
                )

                prediction = st.selectbox(
                    "Previsão",
                    [
                        "SIM",
                        "NÃO",
                    ],
                )

                probability = st.slider(
                    "Probabilidade",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.55,
                    step=0.01,
                )

                submitted = st.form_submit_button(
                    "Guardar previsão"
                )

            if submitted:
                if not match_name.strip():
                    st.error(
                        "Indica a partida."
                    )
                else:
                    try:
                        save_prediction(
                            match_name=match_name.strip(),
                            market=market,
                            prediction=prediction,
                            probability=probability,
                        )

                        st.success(
                            "Previsão guardada."
                        )

                        st.cache_data.clear()

                    except Exception as error:
                        st.error(
                            str(error)
                        )

        else:
            st.info(
                "A tabela historico_previsoes não existe."
            )


# ============================================================
# VALIDAÇÃO
# ============================================================

with validation_tab:
    st.subheader(
        "Validação bootstrap"
    )

    if bootstrap_summary.empty:
        st.warning(
            "Executa primeiro: python .\\ml\\bootstrap_clv.py"
        )

    else:
        row = bootstrap_summary.iloc[0]

        validation_columns = st.columns(3)

        validation_columns[0].metric(
            "Seleções",
            format_number(
                row.get(
                    "selections"
                )
            ),
        )

        validation_columns[1].metric(
            "Prob. CLV > 0",
            format_percent(
                row.get(
                    "probability_clv_mean_positive"
                )
            ),
        )

        validation_columns[2].metric(
            "Prob. ROI > 0",
            format_percent(
                row.get(
                    "probability_roi_positive"
                )
            ),
        )

        interval_data = pd.DataFrame(
            {
                "Métrica": [
                    "CLV",
                    "ROI",
                ],
                "Estimativa": [
                    row.get(
                        "clv_mean"
                    ),
                    row.get(
                        "roi"
                    ),
                ],
                "Limite inferior": [
                    row.get(
                        "clv_ci_lower"
                    ),
                    row.get(
                        "roi_ci_lower"
                    ),
                ],
                "Limite superior": [
                    row.get(
                        "clv_ci_upper"
                    ),
                    row.get(
                        "roi_ci_upper"
                    ),
                ],
            }
        )

        interval_display = (
            interval_data.copy()
        )

        for column in [
            "Estimativa",
            "Limite inferior",
            "Limite superior",
        ]:
            interval_display[column] = (
                interval_display[column]
                .apply(format_percent)
            )

        st.dataframe(
            interval_display,
            use_container_width=True,
            hide_index=True,
        )

        chart_data = interval_data.set_index(
            "Métrica"
        )[
            [
                "Estimativa",
                "Limite inferior",
                "Limite superior",
            ]
        ]

        st.bar_chart(
            chart_data
        )

        if (
            row.get(
                "clv_ci_lower",
                0,
            )
            > 0
        ):
            st.success(
                "O CLV médio apresenta evidência estatística positiva."
            )
        else:
            st.warning(
                "O intervalo do CLV ainda inclui zero."
            )

        if (
            row.get(
                "roi_ci_lower",
                0,
            )
            > 0
        ):
            st.success(
                "O ROI apresenta evidência estatística positiva."
            )
        else:
            st.warning(
                "O ROI ainda não apresenta vantagem estatística robusta."
            )


# ============================================================
# LIGAS E ÉPOCAS
# ============================================================

with league_tab:
    league_section, season_section = st.columns(
        2
    )

    with league_section:
        st.subheader(
            "Desempenho por liga"
        )

        if bootstrap_leagues.empty:
            st.info(
                "Relatório por liga indisponível."
            )
        else:
            league_display = (
                bootstrap_leagues.copy()
            )

            percentage_columns = [
                "clv_mean",
                "clv_median",
                "clv_positive_rate",
                "clv_ci_lower",
                "clv_ci_upper",
                "probability_clv_mean_positive",
                "roi",
                "roi_ci_lower",
                "roi_ci_upper",
                "probability_roi_positive",
            ]

            for column in percentage_columns:
                if column in league_display.columns:
                    league_display[column] = (
                        league_display[column]
                        .apply(format_percent)
                    )

            st.dataframe(
                league_display,
                use_container_width=True,
                hide_index=True,
            )

            if (
                "league" in bootstrap_leagues.columns
                and "clv_mean"
                in bootstrap_leagues.columns
            ):
                league_chart = (
                    bootstrap_leagues
                    .set_index(
                        "league"
                    )[
                        "clv_mean"
                    ]
                    .sort_values(
                        ascending=False
                    )
                )

                st.bar_chart(
                    league_chart
                )

    with season_section:
        st.subheader(
            "Desempenho por época"
        )

        if bootstrap_seasons.empty:
            st.info(
                "Relatório por época indisponível."
            )
        else:
            season_display = (
                bootstrap_seasons.copy()
            )

            percentage_columns = [
                "clv_mean",
                "clv_median",
                "clv_positive_rate",
                "clv_ci_lower",
                "clv_ci_upper",
                "probability_clv_mean_positive",
                "roi",
                "roi_ci_lower",
                "roi_ci_upper",
                "probability_roi_positive",
            ]

            for column in percentage_columns:
                if column in season_display.columns:
                    season_display[column] = (
                        season_display[column]
                        .apply(format_percent)
                    )

            st.dataframe(
                season_display,
                use_container_width=True,
                hide_index=True,
            )

            if (
                "season" in bootstrap_seasons.columns
                and "clv_mean"
                in bootstrap_seasons.columns
            ):
                season_chart = (
                    bootstrap_seasons
                    .set_index(
                        "season"
                    )[
                        "clv_mean"
                    ]
                )

                st.line_chart(
                    season_chart
                )


# ============================================================
# HISTÓRICO
# ============================================================

with history_tab:
    st.subheader(
        "Previsões do backtest"
    )

    if filtered_predictions.empty:
        st.info(
            "Relatório de previsões indisponível."
        )

    else:
        metrics = st.columns(4)

        metrics[0].metric(
            "Jogos avaliados",
            format_number(
                len(
                    filtered_predictions
                )
            ),
        )

        metrics[1].metric(
            "Sinais",
            format_number(
                len(
                    filtered_selections
                )
            ),
        )

        clv_column = detect_column(
            filtered_selections,
            [
                "clv",
                "clv_value",
            ],
        )

        profit_column = detect_column(
            filtered_selections,
            [
                "profit",
                "lucro",
                "pnl",
            ],
        )

        if (
            clv_column is not None
            and not filtered_selections.empty
        ):
            metrics[2].metric(
                "CLV médio",
                format_percent(
                    pd.to_numeric(
                        filtered_selections[
                            clv_column
                        ],
                        errors="coerce",
                    ).mean()
                ),
            )
        else:
            metrics[2].metric(
                "CLV médio",
                "—",
            )

        if (
            profit_column is not None
            and not filtered_selections.empty
        ):
            profit_values = pd.to_numeric(
                filtered_selections[
                    profit_column
                ],
                errors="coerce",
            )

            metrics[3].metric(
                "ROI",
                format_percent(
                    profit_values.mean()
                ),
            )
        else:
            metrics[3].metric(
                "ROI",
                "—",
            )

        only_signals = st.toggle(
            "Mostrar apenas sinais",
            value=True,
        )

        displayed_history = (
            filtered_selections
            if only_signals
            else filtered_predictions
        )

        st.dataframe(
            displayed_history,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Exportar CSV filtrado",
            data=displayed_history.to_csv(
                index=False
            ).encode(
                "utf-8-sig"
            ),
            file_name=(
                "footballai_historico_filtrado.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# QUALIDADE DOS DADOS
# ============================================================

with quality_tab:
    st.subheader(
        "Qualidade e cobertura"
    )

    if not DATABASE_PATH.exists():
        st.error(
            f"Base de dados não encontrada: {DATABASE_PATH}"
        )

    else:
        matches_summary = read_database_query(
            """
            SELECT
                COUNT(*) AS total_matches,
                COUNT(DISTINCT league) AS leagues,
                COUNT(DISTINCT season) AS seasons,
                SUM(
                    CASE
                        WHEN odd_over25_open IS NOT NULL
                         AND odd_under25_open IS NOT NULL
                        THEN 1
                        ELSE 0
                    END
                ) AS complete_opening_odds,
                SUM(
                    CASE
                        WHEN odd_over25_close IS NOT NULL
                         AND odd_under25_close IS NOT NULL
                        THEN 1
                        ELSE 0
                    END
                ) AS complete_closing_odds
            FROM matches
            """
        )

        if not matches_summary.empty:
            summary = matches_summary.iloc[0]

            quality_columns = st.columns(5)

            quality_columns[0].metric(
                "Jogos",
                format_number(
                    summary[
                        "total_matches"
                    ]
                ),
            )

            quality_columns[1].metric(
                "Ligas",
                format_number(
                    summary[
                        "leagues"
                    ]
                ),
            )

            quality_columns[2].metric(
                "Épocas",
                format_number(
                    summary[
                        "seasons"
                    ]
                ),
            )

            quality_columns[3].metric(
                "Odds abertura",
                format_number(
                    summary[
                        "complete_opening_odds"
                    ]
                ),
            )

            quality_columns[4].metric(
                "Odds fecho",
                format_number(
                    summary[
                        "complete_closing_odds"
                    ]
                ),
            )

        league_quality = read_database_query(
            """
            SELECT
                league,
                COUNT(*) AS jogos,
                SUM(
                    CASE
                        WHEN odd_over25_open IS NOT NULL
                         AND odd_under25_open IS NOT NULL
                        THEN 1
                        ELSE 0
                    END
                ) AS odds_abertura,
                SUM(
                    CASE
                        WHEN odd_over25_close IS NOT NULL
                         AND odd_under25_close IS NOT NULL
                        THEN 1
                        ELSE 0
                    END
                ) AS odds_fecho,
                COUNT(DISTINCT season) AS epocas
            FROM matches
            GROUP BY league
            ORDER BY league
            """
        )

        if not league_quality.empty:
            league_quality[
                "cobertura_abertura"
            ] = (
                league_quality[
                    "odds_abertura"
                ]
                / league_quality[
                    "jogos"
                ]
            )

            league_quality[
                "cobertura_fecho"
            ] = (
                league_quality[
                    "odds_fecho"
                ]
                / league_quality[
                    "jogos"
                ]
            )

            display_quality = (
                league_quality.copy()
            )

            display_quality[
                "cobertura_abertura"
            ] = display_quality[
                "cobertura_abertura"
            ].apply(
                format_percent
            )

            display_quality[
                "cobertura_fecho"
            ] = display_quality[
                "cobertura_fecho"
            ].apply(
                format_percent
            )

            st.dataframe(
                display_quality,
                use_container_width=True,
                hide_index=True,
            )

        duplicate_count = read_database_query(
            """
            SELECT COUNT(*) AS duplicate_groups
            FROM (
                SELECT
                    date,
                    league,
                    home_team,
                    away_team,
                    COUNT(*) AS quantity
                FROM matches
                GROUP BY
                    date,
                    league,
                    home_team,
                    away_team
                HAVING COUNT(*) > 1
            )
            """
        )

        if (
            not duplicate_count.empty
            and int(
                duplicate_count.iloc[0][
                    "duplicate_groups"
                ]
            )
            == 0
        ):
            st.success(
                "Nenhum grupo de jogos duplicados encontrado."
            )
        else:
            st.error(
                "Foram encontrados possíveis jogos duplicados."
            )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "FootballAI Pro · Uso exclusivo para investigação e "
    "paper trading. Resultados históricos não garantem "
    "resultados futuros."
)