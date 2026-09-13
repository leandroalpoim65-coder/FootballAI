from __future__ import annotations

import pandas as pd

from agent.v12_rich_batch import build_fixture_feature_rows_resilient
from agent.v12_rich_production import BASE_TRAIN_END, CALIBRATION_END, RICH_ANALYST_FEATURES, build_fixture_feature_rows


def _history():
    rows=[]
    for i in range(10):
        home,away=("Alpha","Beta") if i%2==0 else ("Beta","Alpha")
        hg,ag=1+(i%3),i%2
        rows.append({"id":i+1,"date":pd.Timestamp("2025-01-01")+pd.Timedelta(days=7*i),"league":"Test League","home_team":home,"away_team":away,"home_goals":hg,"away_goals":ag,"actual_over25":int(hg+ag>=3),"odd_over25_open":1.9,"odd_under25_open":1.9,"home_shots":12+i%3,"away_shots":9+i%2,"home_shots_target":5+i%2,"away_shots_target":3+i%2,"home_corners":5+i%2,"away_corners":4+i%2})
    return pd.DataFrame(rows)


def _fixture(key="future-a-b", away="Beta"):
    return {"fixture_key":key,"fixture_date":"2026-01-10","league":"Test League","home_team":"Alpha","away_team":away,"odd_over25_open":1.95,"odd_under25_open":1.85}


def test_fixed_cutoffs_and_feature_count():
    assert BASE_TRAIN_END == pd.Timestamp("2025-05-31")
    assert CALIBRATION_END == pd.Timestamp("2026-05-31")
    assert len(RICH_ANALYST_FEATURES) == 73


def test_future_fixture_uses_same_73_feature_pipeline():
    result=build_fixture_feature_rows(_history(),[_fixture()])["future-a-b"]
    assert set(result["features"]) == set(RICH_ANALYST_FEATURES)
    assert 0.0 < result["market_probability"] < 1.0


def test_bad_fixture_does_not_abort_good_fixture_batch():
    good=_fixture()
    bad=_fixture("future-a-unknown", "Unknown")
    results,errors=build_fixture_feature_rows_resilient(_history(),[good,bad])
    assert "future-a-b" in results
    assert "future-a-unknown" in errors
    assert "future-a-unknown" not in results
