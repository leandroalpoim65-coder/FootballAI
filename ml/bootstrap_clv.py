from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT_DIR
    / "reports"
    / "clv_predictions_over25.csv"
)

REPORTS_DIR = ROOT_DIR / "reports"

SUMMARY_CSV_PATH = (
    REPORTS_DIR
    / "bootstrap_clv_summary.csv"
)

LEAGUES_CSV_PATH = (
    REPORTS_DIR
    / "bootstrap_clv_by_league.csv"
)

SEASONS_CSV_PATH = (
    REPORTS_DIR
    / "bootstrap_clv_by_season.csv"
)

METADATA_PATH = (
    REPORTS_DIR
    / "bootstrap_clv_metadata.json"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

MIN_GROUP_SELECTIONS = 50


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_name(value: str) -> str:
    """
    Normaliza o nome de uma coluna para facilitar a deteção.
    """
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("%", "percent")
    )


def find_column(
    dataframe: pd.DataFrame,
    aliases: list[str],
) -> str | None:
    """
    Procura uma coluna usando vários nomes possíveis.
    """
    normalized_columns = {
        normalize_name(column): column
        for column in dataframe.columns
    }

    for alias in aliases:
        normalized_alias = normalize_name(alias)

        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    return None


def to_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Converte texto numérico, incluindo percentagens e vírgulas.
    """
    text = (
        series
        .astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    numeric = pd.to_numeric(
        text,
        errors="coerce",
    )

    original_text = (
        series
        .astype("string")
        .str.contains(
            "%",
            regex=False,
            na=False,
        )
    )

    numeric.loc[original_text] = (
        numeric.loc[original_text]
        / 100.0
    )

    return numeric


def to_binary(
    series: pd.Series,
) -> pd.Series:
    """
    Converte uma coluna de sinal/resultado para 0 e 1.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().any():
        return (
            numeric
            .fillna(0)
            .astype(float)
            > 0
        ).astype(int)

    text = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    true_values = {
        "1",
        "true",
        "yes",
        "y",
        "sim",
        "s",
        "selected",
        "selection",
        "selecao",
        "seleção",
        "signal",
        "sinal",
        "over",
        "over25",
    }

    return text.isin(true_values).astype(int)


def percentile_interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Calcula o intervalo percentil bootstrap.
    """
    alpha = 1.0 - confidence_level

    lower = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


def bootstrap_statistics(
    clv_values: np.ndarray,
    profit_values: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    """
    Faz bootstrap do CLV médio e do ROI.

    Cada seleção tem stake fictícia de 1 unidade.
    Assim, ROI = lucro médio por seleção.
    """
    if len(clv_values) == 0:
        raise ValueError(
            "Não existem seleções para analisar."
        )

    if len(clv_values) != len(profit_values):
        raise ValueError(
            "CLV e lucro têm tamanhos diferentes."
        )

    rng = np.random.default_rng(seed)

    selection_count = len(clv_values)

    boot_clv = np.empty(
        samples,
        dtype=float,
    )

    boot_roi = np.empty(
        samples,
        dtype=float,
    )

    for sample_number in range(samples):
        indexes = rng.integers(
            0,
            selection_count,
            size=selection_count,
        )

        boot_clv[sample_number] = (
            clv_values[indexes].mean()
        )

        boot_roi[sample_number] = (
            profit_values[indexes].mean()
        )

    clv_lower, clv_upper = percentile_interval(
        boot_clv,
        CONFIDENCE_LEVEL,
    )

    roi_lower, roi_upper = percentile_interval(
        boot_roi,
        CONFIDENCE_LEVEL,
    )

    return {
        "selections": int(selection_count),
        "clv_mean": float(clv_values.mean()),
        "clv_median": float(np.median(clv_values)),
        "clv_positive_rate": float(
            (clv_values > 0).mean()
        ),
        "clv_ci_lower": clv_lower,
        "clv_ci_upper": clv_upper,
        "roi": float(profit_values.mean()),
        "profit": float(profit_values.sum()),
        "roi_ci_lower": roi_lower,
        "roi_ci_upper": roi_upper,
        "probability_clv_mean_positive": float(
            (boot_clv > 0).mean()
        ),
        "probability_roi_positive": float(
            (boot_roi > 0).mean()
        ),
    }


def prepare_predictions(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Deteta as colunas do relatório e prepara as seleções.
    """
    signal_column = find_column(
        dataframe,
        [
            "signal",
            "sinal",
            "selected",
            "selection",
            "selecao",
            "seleção",
            "bet",
            "aposta",
        ],
    )

    clv_column = find_column(
        dataframe,
        [
            "clv",
            "clv_value",
            "clv_percent",
            "clv_pct",
            "closing_line_value",
        ],
    )

    profit_column = find_column(
        dataframe,
        [
            "profit",
            "lucro",
            "pnl",
            "resultado_financeiro",
        ],
    )

    open_odd_column = find_column(
        dataframe,
        [
            "odd_over25_open",
            "odd_abertura",
            "opening_odd",
            "open_odd",
            "odd_open",
        ],
    )

    close_odd_column = find_column(
        dataframe,
        [
            "odd_over25_close",
            "odd_fecho",
            "closing_odd",
            "close_odd",
            "odd_close",
        ],
    )

    target_column = find_column(
        dataframe,
        [
            "over25",
            "target",
            "result",
            "resultado",
            "won",
            "win",
            "vitoria",
            "vitória",
        ],
    )

    league_column = find_column(
        dataframe,
        [
            "league",
            "liga",
        ],
    )

    season_column = find_column(
        dataframe,
        [
            "season",
            "epoca",
            "época",
        ],
    )

    prepared = dataframe.copy()

    if signal_column is not None:
        prepared["_selected"] = to_binary(
            prepared[signal_column]
        )

        prepared = prepared[
            prepared["_selected"] == 1
        ].copy()

    if clv_column is not None:
        prepared["_clv"] = to_numeric(
            prepared[clv_column]
        )

    elif (
        open_odd_column is not None
        and close_odd_column is not None
    ):
        opening_odds = to_numeric(
            prepared[open_odd_column]
        )

        closing_odds = to_numeric(
            prepared[close_odd_column]
        )

        prepared["_clv"] = (
            opening_odds
            / closing_odds
            - 1.0
        )

    else:
        raise ValueError(
            "Não foi possível encontrar o CLV. "
            "O ficheiro precisa de uma coluna CLV ou das "
            "colunas de odd de abertura e de fecho."
        )

    if profit_column is not None:
        prepared["_profit"] = to_numeric(
            prepared[profit_column]
        )

    elif (
        open_odd_column is not None
        and target_column is not None
    ):
        opening_odds = to_numeric(
            prepared[open_odd_column]
        )

        target = to_binary(
            prepared[target_column]
        )

        prepared["_profit"] = np.where(
            target == 1,
            opening_odds - 1.0,
            -1.0,
        )

    else:
        raise ValueError(
            "Não foi possível encontrar ou calcular o lucro. "
            "O ficheiro precisa de uma coluna de lucro ou das "
            "colunas odd de abertura e resultado Over 2.5."
        )

    prepared = prepared.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    prepared = prepared.dropna(
        subset=[
            "_clv",
            "_profit",
        ]
    ).copy()

    detected = {
        "signal_column": signal_column,
        "clv_column": clv_column,
        "profit_column": profit_column,
        "open_odd_column": open_odd_column,
        "close_odd_column": close_odd_column,
        "target_column": target_column,
        "league_column": league_column,
        "season_column": season_column,
    }

    return prepared, detected


def analyze_groups(
    dataframe: pd.DataFrame,
    group_column: str,
    seed_offset: int,
) -> pd.DataFrame:
    """
    Executa bootstrap por liga ou época.
    """
    rows = []

    grouped = dataframe.groupby(
        group_column,
        dropna=False,
        sort=True,
    )

    for group_number, (
        group_value,
        group_data,
    ) in enumerate(grouped):

        if len(group_data) < MIN_GROUP_SELECTIONS:
            continue

        statistics = bootstrap_statistics(
            clv_values=group_data[
                "_clv"
            ].to_numpy(dtype=float),
            profit_values=group_data[
                "_profit"
            ].to_numpy(dtype=float),
            samples=BOOTSTRAP_SAMPLES,
            seed=(
                RANDOM_SEED
                + seed_offset
                + group_number
            ),
        )

        statistics["group"] = str(group_value)

        rows.append(statistics)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    ordered_columns = [
        "group",
        "selections",
        "clv_mean",
        "clv_median",
        "clv_positive_rate",
        "clv_ci_lower",
        "clv_ci_upper",
        "probability_clv_mean_positive",
        "profit",
        "roi",
        "roi_ci_lower",
        "roi_ci_upper",
        "probability_roi_positive",
    ]

    return result[ordered_columns]


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Relatório de previsões não encontrado: "
            f"{PREDICTIONS_PATH}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        low_memory=False,
    )

    print("\n==========================================")
    print(" FOOTBALLAI — BOOTSTRAP CLV E ROI")
    print("==========================================\n")

    print(
        f"Linhas no relatório : {len(predictions)}"
    )

    prepared, detected = prepare_predictions(
        predictions
    )

    print(
        f"Seleções analisadas : {len(prepared)}"
    )

    print(
        f"Amostras bootstrap  : {BOOTSTRAP_SAMPLES}"
    )

    print(
        f"Nível de confiança  : "
        f"{CONFIDENCE_LEVEL * 100:.1f}%"
    )

    print("\nColunas detetadas:")

    for key, value in detected.items():
        print(f"  {key:<20}: {value}")

    global_statistics = bootstrap_statistics(
        clv_values=prepared[
            "_clv"
        ].to_numpy(dtype=float),
        profit_values=prepared[
            "_profit"
        ].to_numpy(dtype=float),
        samples=BOOTSTRAP_SAMPLES,
        seed=RANDOM_SEED,
    )

    global_row = {
        "scope": "global",
        **global_statistics,
    }

    global_summary = pd.DataFrame(
        [global_row]
    )

    global_summary.to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )

    league_column = detected[
        "league_column"
    ]

    if league_column is not None:
        league_results = analyze_groups(
            dataframe=prepared,
            group_column=league_column,
            seed_offset=1_000,
        )

        if not league_results.empty:
            league_results = (
                league_results
                .rename(
                    columns={
                        "group": "league",
                    }
                )
                .sort_values(
                    "clv_mean",
                    ascending=False,
                )
            )

            league_results.to_csv(
                LEAGUES_CSV_PATH,
                index=False,
            )

    else:
        league_results = pd.DataFrame()

    season_column = detected[
        "season_column"
    ]

    if season_column is not None:
        season_results = analyze_groups(
            dataframe=prepared,
            group_column=season_column,
            seed_offset=2_000,
        )

        if not season_results.empty:
            season_results = (
                season_results
                .rename(
                    columns={
                        "group": "season",
                    }
                )
                .sort_values(
                    "season",
                )
            )

            season_results.to_csv(
                SEASONS_CSV_PATH,
                index=False,
            )

    else:
        season_results = pd.DataFrame()

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "predictions_path": str(
            PREDICTIONS_PATH.relative_to(
                ROOT_DIR
            )
        ),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "confidence_level": CONFIDENCE_LEVEL,
        "random_seed": RANDOM_SEED,
        "minimum_group_selections": (
            MIN_GROUP_SELECTIONS
        ),
        "rows_in_report": int(
            len(predictions)
        ),
        "selections_analyzed": int(
            len(prepared)
        ),
        "detected_columns": detected,
        "global_statistics": (
            global_statistics
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n==========================================")
    print(" RESULTADO GLOBAL")
    print("==========================================\n")

    print(
        f"Seleções                  : "
        f"{global_statistics['selections']}"
    )

    print(
        f"CLV médio                 : "
        f"{percent(global_statistics['clv_mean'])}"
    )

    print(
        f"IC 95% do CLV             : "
        f"[{percent(global_statistics['clv_ci_lower'])}, "
        f"{percent(global_statistics['clv_ci_upper'])}]"
    )

    print(
        f"Prob. CLV médio > 0       : "
        f"{percent(global_statistics['probability_clv_mean_positive'])}"
    )

    print(
        f"ROI                       : "
        f"{percent(global_statistics['roi'])}"
    )

    print(
        f"IC 95% do ROI             : "
        f"[{percent(global_statistics['roi_ci_lower'])}, "
        f"{percent(global_statistics['roi_ci_upper'])}]"
    )

    print(
        f"Prob. ROI > 0             : "
        f"{percent(global_statistics['probability_roi_positive'])}"
    )

    print(
        f"Lucro total               : "
        f"{global_statistics['profit']:.2f}"
    )

    if (
        global_statistics["clv_ci_lower"] > 0
    ):
        print(
            "\n✅ O intervalo de confiança do CLV "
            "fica totalmente acima de zero."
        )
    else:
        print(
            "\n⚠ O intervalo de confiança do CLV "
            "inclui zero."
        )

    if (
        global_statistics["roi_ci_lower"] > 0
    ):
        print(
            "✅ O intervalo de confiança do ROI "
            "fica totalmente acima de zero."
        )
    else:
        print(
            "⚠ O ROI ainda não demonstra vantagem "
            "financeira estatisticamente robusta."
        )

    print("\nRelatórios guardados:")

    print(
        "✅ "
        + str(
            SUMMARY_CSV_PATH.relative_to(
                ROOT_DIR
            )
        )
    )

    if not league_results.empty:
        print(
            "✅ "
            + str(
                LEAGUES_CSV_PATH.relative_to(
                    ROOT_DIR
                )
            )
        )

    if not season_results.empty:
        print(
            "✅ "
            + str(
                SEASONS_CSV_PATH.relative_to(
                    ROOT_DIR
                )
            )
        )

    print(
        "✅ "
        + str(
            METADATA_PATH.relative_to(
                ROOT_DIR
            )
        )
    )

    print(
        "\n⚠ Esta análise é apenas para validação "
        "e paper trading."
    )

    print(
        "⚠ Não utilizar dinheiro real com base "
        "neste resultado."
    )


if __name__ == "__main__":
    main()