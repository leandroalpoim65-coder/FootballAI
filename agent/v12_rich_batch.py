from __future__ import annotations

import pandas as pd

from agent.v12_rich_production import build_fixture_feature_rows


def build_fixture_feature_rows_resilient(
    matches: pd.DataFrame,
    fixtures: list[dict],
) -> tuple[dict[str, dict], dict[str, str]]:
    """Build a batch without letting one bad fixture abort all predictions.

    The strict builder remains the single source of truth for feature generation.
    On a batch failure we bisect the batch until the offending fixture(s) are
    isolated. Successful fixtures therefore still use exactly the same 73-feature
    train/serve pipeline; failed fixtures are returned as explicit skips.
    """
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    def evaluate(batch: list[dict]) -> None:
        if not batch:
            return
        try:
            results.update(build_fixture_feature_rows(matches, batch))
            return
        except Exception as error:
            if len(batch) == 1:
                key = str(batch[0].get("fixture_key"))
                errors[key] = str(error)
                return
        midpoint = len(batch) // 2
        evaluate(batch[:midpoint])
        evaluate(batch[midpoint:])

    evaluate(list(fixtures))
    return results, errors
