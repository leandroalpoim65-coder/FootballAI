# Goal Model — Controlled Holdout EPL 2023/24

Generated from an independent, closed Premier League 2023/24 dataset with real opening average odds from Football-Data data mirrored publicly on GitHub.

## Test design

- Holdout: final 34 Premier League matches, 2024-05-02 through 2024-05-19.
- Initial pre-holdout history: 97 immediately preceding EPL matches, 2024-02-21 through 2024-04-28.
- Strict date walk-forward: all matches on a date are predicted before that date is appended to history.
- Team features: last 5 overall and last 5 home/away matches, matching Goal Model inputs.
- Goal Model config reproduced: general_weight=0.45, venue_weight=0.55, league_shrinkage=0.10, elo_adjustment_strength=0.0.
- League home/away average for shrinkage uses only the available pre-match control window, never future holdout results.
- Odds: opening `AvgH/AvgD/AvgA` and `Avg>2.5/Avg<2.5`.
- Stakes: flat 1 unit.
- This is a small diagnostic holdout, not sufficient to approve a production strategy.

## Probability quality

| Market | N | Actual rate | Mean model probability | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| BTTS Yes | 34 | 58.82% | 63.52% | 52.94% | 0.2657 | 0.7308 |
| Over 1.5 | 34 | 88.24% | 85.00% | 88.24% | 0.1019 | 0.3508 |
| Over 2.5 | 34 | 70.59% | 65.98% | 70.59% | 0.2168 | 0.6248 |
| Over 3.5 | 34 | 47.06% | 44.67% | 58.82% | 0.2491 | 0.6925 |
| Home team Over 0.5 | 34 | 82.35% | 82.71% | 82.35% | 0.1276 | 0.4041 |
| Home team Over 1.5 | 34 | 52.94% | 54.07% | 76.47% | 0.1937 | 0.5750 |
| Home team Over 2.5 | 34 | 29.41% | 28.82% | 73.53% | 0.1708 | 0.5115 |
| Away team Over 0.5 | 34 | 73.53% | 77.29% | 73.53% | 0.1964 | 0.5760 |
| Away team Over 1.5 | 34 | 47.06% | 45.64% | 55.88% | 0.2075 | 0.6008 |
| Away team Over 2.5 | 34 | 26.47% | 21.60% | 73.53% | 0.1810 | 0.5313 |

### 1X2 benchmark vs opening market

| Metric | Goal Model | No-vig opening market |
|---|---:|---:|
| Accuracy (argmax) | 67.65% | 76.47% |
| Multiclass Brier | 0.4648 | 0.3936 |
| Log loss | 0.8118 | 0.7071 |

Model mean H/D/A: 45.03% / 20.88% / 34.09%.
Actual H/D/A: 47.06% / 11.76% / 41.18%.

### Over 2.5 benchmark vs opening market

| Metric | Goal Model | No-vig opening market |
|---|---:|---:|
| Mean P(Over 2.5) | 65.98% | 68.31% |
| Actual Over 2.5 | 70.59% | 70.59% |
| Accuracy | 70.59% | 70.59% |
| Brier | 0.2168 | 0.2090 |
| Log loss | 0.6248 | 0.6095 |

The market is slightly better calibrated than Goal Model v1 on O/U 2.5 in this sample, but the gap is much smaller than in 1X2.

## O/U 2.5 betting replay with real opening average odds

Each match chooses the side (Over or Under) with the larger model EV. Thresholds are fixed before settlement; no threshold was fitted to maximize this sample.

| Rule | Bets | Profit | ROI | Wins | Avg model EV |
|---|---:|---:|---:|---:|---:|
| Always Over 2.5 | 34 | -0.54u | -1.59% | 24 | — |
| Always Under 2.5 | 34 | -3.14u | -9.24% | 10 | — |
| Model best side — all games | 34 | +0.27u | +0.79% | 15 | +13.61% |
| Best side EV > 0% | 27 | -0.57u | -2.11% | 11 | +17.84% |
| Best side EV >= 2% | 25 | -1.66u | -6.64% | 10 | +19.18% |
| Best side EV >= 5% | 21 | -3.19u | -15.19% | 7 | +22.27% |
| Best side EV >= 8% | 19 | -1.19u | -6.26% | 7 | +23.99% |
| Best side EV >= 10% | 17 | -1.57u | -9.24% | 6 | +25.79% |
| Best side EV >= 15% | 14 | +0.03u | +0.21% | 5 | +28.52% |

For EV > 0%, selections were 19 Under and 8 Over. Under selections produced +0.03u across 19 bets; Over selections produced -0.60u across 8 bets.

Bootstrap 95% interval for ROI of the EV > 0% rule is approximately -50.0% to +50.1%, showing how insufficient N=27 is for a profitability conclusion.

## 1X2 value replay warning

Using the largest model EV among Home/Draw/Away is not production-ready:

- EV > 0%: 32 bets
- Profit: -14.40u
- ROI: -45.00%
- Wins: 7/32
- Mean claimed model EV: +58.26%

This large contradiction between claimed EV and realized performance, together with the market's clearly better Brier/log loss, indicates serious 1X2 price calibration error. With `elo_adjustment_strength=0.0`, the current Poisson goal engine should track 1X2 diagnostically but must not be allowed to dominate best-value ranking.

## Interpretation

1. Goal totals are promising as a probability engine, especially O1.5 and team totals calibration, but O/U 2.5 has not demonstrated a profitable edge yet.
2. Higher raw EV thresholds do not improve ROI in this holdout. Raw Goal Model EV is therefore not yet a calibrated confidence scale.
3. 1X2 should remain diagnostic-only until score-strength / Elo / calibration improvements are validated out-of-sample.
4. Next controlled test should expand to a much larger out-of-sample set and compare calibrated Goal Model probabilities against no-vig market probabilities by league and market.
5. No strategy should be promoted from this 34-match sample alone.
