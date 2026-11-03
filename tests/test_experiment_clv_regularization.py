import numpy as np
import pandas as pd

from ml.experiment_clv_regularization import (
    PRODUCTION_C,
    run_experiment,
    select_regularization_c,
)


def _synthetic_matches() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for season_index, season in enumerate(
        ["2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024"]
    ):
        for match_index in range(80):
            form = float(rng.normal())
            clv = int(rng.random() < 1 / (1 + np.exp(-form * 0.4)))
            rows.append(
                {
                    "id": season_index * 1000 + match_index,
                    "date": pd.Timestamp("2019-08-01")
                    + pd.DateOffset(years=season_index, days=match_index),
                    "season": season,
                    "league": "League A" if match_index % 2 else "League B",
                    "form": form,
                    "over25": int(rng.random() < 0.5),
                    "odd_over25_open": 1.8,
                    "odd_under25_open": 2.0,
                    "odd_over25_close": 1.7 if clv else 1.9,
                    "odd_under25_close": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_tuning_uses_only_completed_prior_validation_seasons():
    matches = _synthetic_matches()
    matches["clv_positive"] = (
        matches["odd_over25_open"] / matches["odd_over25_close"] - 1 > 0
    ).astype(int)

    chosen_c, validation_seasons = select_regularization_c(
        matches.loc[matches["season"] != "2023/2024"],
        ["form", "odd_over25_open", "odd_under25_open"],
    )

    assert chosen_c in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
    assert validation_seasons == ["2022/2023"]
    assert "2023/2024" not in validation_seasons


def test_experiment_compares_same_out_of_time_folds_and_preserves_first_fold():
    summary = run_experiment(_synthetic_matches(), ["form"])

    assert summary["data"]["test_seasons"] == ["2022/2023", "2023/2024"]
    assert summary["folds"][0]["selected_C"] == PRODUCTION_C
    assert summary["folds"][0]["tuning_seasons"] == []
    assert summary["folds"][1]["tuning_seasons"] == ["2022/2023"]
    assert summary["folds"][0]["test_matches"] == 80
    assert set(summary["pooled"]) >= {
        "selected",
        "production",
        "baseline",
        "selected_strategy",
        "production_strategy",
    }
