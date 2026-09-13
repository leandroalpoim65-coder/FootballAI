from __future__ import annotations

import agent_v12_rich_shadow_cloud as runner
from agent.v12_rich_batch import build_fixture_feature_rows_resilient


def resilient_builder(matches, fixtures):
    results, errors = build_fixture_feature_rows_resilient(matches, fixtures)
    for index, (key, error) in enumerate(errors.items()):
        if index >= 20:
            break
        print(f"SKIP RICH {key} | {error}")
    return results


runner.build_fixture_feature_rows = resilient_builder

if __name__ == "__main__":
    runner.main()
