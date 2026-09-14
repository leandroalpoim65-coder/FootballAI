# GitHub Actions runner allocation diagnostic — 2026-09-14

## What was tested

A dedicated research workflow was created for the Goal Model multi-league backtest. The workflow was triggered multiple times with two GitHub-hosted runner labels:

- `ubuntu-latest`
- `ubuntu-22.04`

The workflow contains normal steps (checkout, Python setup, dependencies, compile, backtest, artifact upload), but none of those steps were ever started.

## Evidence

Goal Model multi-league backtest runs:

- Run `34850885388` — failed before executing steps.
- Run `34850982025` — failed before executing steps.
- Run `34851027179` — job `103998716679`, `steps=[]`, `runner_id=0`, empty runner name.
- Run `34851646434` — job `104000815740`, `steps=[]`, `runner_id=0`, empty runner name, even after changing the label to `ubuntu-22.04`.

Run 34851646434 was created at 13:49:33 UTC and completed at 13:49:35 UTC. This is before checkout or Python execution could occur.

Unrelated scheduled production/research workflows were also observed failing during the same period (Results Cloud, Closing Odds Cloud, agent workflows). This means the symptom is not isolated to the new backtest.

## Conclusion

The multi-league Python backtest has **not executed yet**. The failures happen before a GitHub-hosted runner is allocated, so these runs cannot be used as evidence of a Python/code failure.

The fixed runner-image experiment did not change the result. The workflow has therefore been restored to the standard `ubuntu-latest` label and left ready to execute once GitHub Actions runner allocation is available again.

Possible account/repository-side causes to investigate include Actions usage/billing limits, spending limits/budgets, or Actions policy/settings. No specific cause is asserted here because the run metadata only proves lack of runner allocation.

## Backtest waiting to run

`goal_model_multileague_backtest.py` is ready to test:

- 2021/22 as development/calibration data;
- 2022/23 as validation;
- 2023/24 as untouched holdout;
- Premier League, Bundesliga, La Liga, Serie A, Ligue 1, Eredivisie and Liga Portugal;
- strict same-date-safe walk-forward;
- real Football-Data opening average odds for 1X2 and O/U 2.5;
- probability metrics, no-vig market comparison, flat-stake ROI, drawdown and bootstrap ROI interval.

No multi-league ROI figure should be reported until this workflow (or an equivalent independent execution) actually runs and produces the report files.
