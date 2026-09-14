# GitHub Actions quota optimization — 2026-09-14

## Cause

The repository exhausted its monthly GitHub Actions allowance. Failed jobs showed `runner_id=0` and no executed steps, confirming that the recent failures were quota/runner-allocation failures rather than Python failures.

## Recurring workload before optimization

Approximate 30-day scheduled run count from the active recurring workflows at audit time:

- Closing Odds: 48/day -> ~1,440/month
- Results Cloud: 24/day -> ~720/month
- V10 Champion Shadow: 24/day -> ~720/month
- V11 Outcome Learner: 24/day -> ~720/month
- V12 Rich Analyst Shadow: 24/day -> ~720/month
- Upcoming Cloud: 2/day -> ~60/month
- Autonomous Agent V3: 2/day -> ~60/month
- Performance Monitor: 2/day -> ~60/month
- Policy Learner: 2/day -> ~60/month
- Memory Learner: 2/day -> ~60/month
- Broad Behavior: 1/day -> ~30/month
- Feature Lab / Historical Replay / UEFA History: weekly

Approximate total: **~4,663 scheduled workflow runs/month** before manual/push/PR runs.

## Changes applied

### Operational pipelines

- `closing_odds_cloud.yml`
  - kept 30-minute cadence in the main European football windows;
  - reduced low-value overnight runs;
  - removed a weekend schedule overlap.
- `update_results_cloud.yml`
  - changed from hourly to every 3 hours;
  - settlement/reporting can lag by up to roughly 3 hours without affecting prospective prediction integrity.
- `upcoming_cloud.yml`
  - unchanged at twice daily because it is a core prospective-data pipeline.

### Shadow / learning pipelines

- `agent_champion_v10_shadow.yml`: hourly -> twice daily, aligned with Upcoming Cloud.
- `agent_outcome_learning_v11.yml`: hourly -> daily.
- `agent_v12_rich_analyst_shadow.yml`: hourly -> daily.
- Performance Monitor + Policy Learner + Memory Learner + Broad Behavior were consolidated into `agent_daily_learning.yml` so checkout, Python setup and dependency installation happen once per day.
- Their individual workflows remain available for manual/PR diagnostics but no longer have recurring schedules.

### Research / backtests

- `goal_model_controlled_test.yml` was already manual-only.
- `goal_model_multileague_backtest.yml` is now manual-only; changing Goal Model code no longer launches an expensive historical backtest automatically.
- Feature Lab, Historical Replay and UEFA History remain weekly.
- Multi-Analyst, Formula Lab and Game Analyst Backtest have no recurring schedule.

## Estimated recurring load after optimization

Approximate 30-day run count:

- Closing Odds: ~634/month (weekly-pattern average)
- Results Cloud: ~240/month
- Upcoming Cloud: ~60/month
- Autonomous Agent V3: ~60/month
- V10 Champion Shadow: ~60/month
- V11 Outcome Learner: ~30/month
- V12 Rich Analyst Shadow: ~30/month
- Consolidated Daily Learning: ~30/month
- Weekly Feature/Historical/UEFA jobs: ~13/month combined

Approximate total: **~1,157 scheduled workflow runs/month**, about **75% fewer** than before.

Using recent observed run durations as a rough billing proxy, recurring hosted-runner consumption is expected to be around **1,800 minutes/month**. This is an estimate, not a billing guarantee; manual runs, PR checks, push-triggered CI and GitHub's billing rounding can change the final total.

## Guardrails

1. Historical backtests should remain manual unless there is a strong reason to automate them.
2. Learning jobs should not become hourly again unless evidence shows that intraday retraining materially improves decisions.
3. Closing-odds frequency should be increased only in targeted match windows, not 24/7.
4. Review Actions usage after the first full billing cycle under this schedule before adding new recurring workflows.
5. When adding a new workflow, estimate runs/month and expected runner minutes before enabling a schedule.
