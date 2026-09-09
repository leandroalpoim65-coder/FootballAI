from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from agent.multi_analyst import AnalystScores, score_pre_match, weighted_score
from footballai_agent_cloud import connect_turso, fetch_dicts, load_sports_model, predict_sports_probability


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    goal_t: float = 0.0
    attack_t: float = 0.0
    ml_t: float = 0.0
    value_t: float = 0.0
    votes_needed: int = 0
    weights: tuple[float, float, float, float] | None = None
    weighted_t: float = 0.0


def flat_profit(result: int, odd: float) -> float:
    return odd - 1.0 if int(result) == 1 else -1.0


def build_configs() -> list[Config]:
    configs: list[Config] = []

    for goal_t in (55, 60, 65):
        for attack_t in (55, 60, 65):
            for ml_t in (50, 55, 60, 65):
                for value_t in (50, 55, 60, 65):
                    for votes_needed in (3, 4):
                        configs.append(
                            Config(
                                name=(
                                    f"VOTE{votes_needed}_G{goal_t}_A{attack_t}_"
                                    f"M{ml_t}_V{value_t}"
                                ),
                                family=f"VOTE_{votes_needed}_OF_4",
                                goal_t=goal_t,
                                attack_t=attack_t,
                                ml_t=ml_t,
                                value_t=value_t,
                                votes_needed=votes_needed,
                            )
                        )

    weight_sets = {
        "BALANCED": (0.25, 0.25, 0.25, 0.25),
        "STATS_HEAVY": (0.35, 0.35, 0.15, 0.15),
        "ML_HEAVY": (0.20, 0.20, 0.40, 0.20),
        "VALUE_HEAVY": (0.20, 0.20, 0.20, 0.40),
        "STATS_ML": (0.30, 0.25, 0.35, 0.10),
    }
    for label, weights in weight_sets.items():
        for threshold in (55, 60, 65, 70, 75):
            configs.append(
                Config(
                    name=f"W_{label}_{threshold}",
                    family="WEIGHTED",
                    weights=weights,
                    weighted_t=threshold,
                )
            )

    return configs


def qualifies(row: dict, config: Config) -> bool:
    if row["ev"] < 0.0:
        return False

    s: AnalystScores = row["analysts"]

    if config.family.startswith("VOTE_"):
        votes = sum(
            (
                s.goal_profile >= config.goal_t,
                s.attack_pressure >= config.attack_t,
                s.ml_score >= config.ml_t,
                s.value_score >= config.value_t,
            )
        )
        return votes >= config.votes_needed

    if config.family == "WEIGHTED":
        assert config.weights is not None
        return weighted_score(s, config.weights) >= config.weighted_t

    raise ValueError(f"Unknown family: {config.family}")


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "wins": 0,
            "hit": None,
            "profit": 0.0,
            "roi": None,
            "avg_odd": None,
            "max_drawdown": 0.0,
        }

    wins = sum(int(r["result"]) for r in rows)
    profit = sum(float(r["profit"]) for r in rows)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in sorted(rows, key=lambda x: (x["date"], x["id"])):
        cumulative += float(r["profit"])
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    return {
        "n": len(rows),
        "wins": wins,
        "hit": wins / len(rows),
        "profit": profit,
        "roi": profit / len(rows),
        "avg_odd": sum(float(r["odd"]) for r in rows) / len(rows),
        "max_drawdown": max_dd,
    }


def date_blocks(dates: list[str], count: int = 3) -> list[set[str]]:
    if not dates:
        return []
    blocks: list[set[str]] = []
    for i in range(count):
        start = int(len(dates) * i / count)
        end = int(len(dates) * (i + 1) / count)
        part = dates[start:end]
        if part:
            blocks.append(set(part))
    return blocks


def bootstrap_roi_by_date(rows: list[dict], samples: int = 3000) -> tuple[float | None, float | None]:
    if len(rows) < 5:
        return None, None

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["date"]].append(r)
    days = sorted(grouped)
    if len(days) < 2:
        return None, None

    rng = random.Random(42)
    values: list[float] = []
    for _ in range(samples):
        chosen = [days[rng.randrange(len(days))] for _ in days]
        sample_rows = [r for day in chosen for r in grouped[day]]
        if sample_rows:
            values.append(sum(r["profit"] for r in sample_rows) / len(sample_rows))

    if not values:
        return None, None
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return lo, hi


def main():
    print("\n==============================================")
    print(" FOOTBALLAI — MULTI-ANALYST WALK-FORWARD V10")
    print("==============================================\n")

    c = connect_turso()
    try:
        raw_rows = fetch_dicts(
            c,
            """
            SELECT id, fixture_date, fixture_time, league, home_team, away_team,
                   odd_over25_open, fair_probability_open, clv_probability,
                   features_json, result_over25
            FROM paper_trades
            WHERE UPPER(COALESCE(status,'OPEN'))='RESOLVED'
              AND result_over25 IN (0,1)
              AND odd_over25_open IS NOT NULL
              AND fair_probability_open IS NOT NULL
              AND features_json IS NOT NULL
            ORDER BY fixture_date, fixture_time, id
            """,
        )

        model, model_features, model_generated_at = load_sports_model()
        model_date = (
            str(model_generated_at)[:10]
            if model_generated_at
            else None
        )

        data: list[dict] = []
        for raw in raw_rows:
            fixture_date = str(raw.get("fixture_date") or "")
            if model_date and fixture_date < model_date:
                continue

            pre = {
                k: raw.get(k)
                for k in (
                    "id", "fixture_date", "fixture_time", "league",
                    "home_team", "away_team", "odd_over25_open",
                    "fair_probability_open", "clv_probability", "features_json",
                )
            }
            ml = float(predict_sports_probability(model, model_features, pre))
            odd = float(pre["odd_over25_open"])
            market = float(pre["fair_probability_open"])
            ev = ml * odd - 1.0
            edge = ml - market
            analysts = score_pre_match(pre["features_json"], ml, ev, edge)
            result = int(raw["result_over25"])

            data.append(
                {
                    "id": int(raw["id"]),
                    "date": fixture_date,
                    "league": raw.get("league"),
                    "home_team": raw.get("home_team"),
                    "away_team": raw.get("away_team"),
                    "odd": odd,
                    "clv": raw.get("clv_probability"),
                    "ml": ml,
                    "ev": ev,
                    "edge": edge,
                    "analysts": analysts,
                    "result": result,
                    "profit": flat_profit(result, odd),
                }
            )

        dates = sorted(set(r["date"] for r in data))
        holdout_idx = max(1, int(len(dates) * 0.80))
        dev_dates = dates[:holdout_idx]
        holdout_dates = set(dates[holdout_idx:])
        dev = [r for r in data if r["date"] in set(dev_dates)]
        holdout = [r for r in data if r["date"] in holdout_dates]

        initial_idx = max(1, int(len(dev_dates) * 0.40))
        wf_dates = dev_dates[initial_idx:]
        folds = date_blocks(wf_dates, 3)

        print(f"Modelo disponível desde : {model_date or 'desconhecido'}")
        print(f"Jogos elegíveis          : {len(data)}")
        print(f"Desenvolvimento          : {len(dev)}")
        print(f"Holdout FINAL intocado   : {len(holdout)}")
        print(f"Walk-forward folds       : {len(folds)}")
        print("CLV entra na seleção?    : NÃO\n")

        configs = build_configs()
        ranked: list[tuple[float, Config, dict, list[dict]]] = []

        for cfg in configs:
            fold_metrics: list[dict] = []
            all_selected: list[dict] = []

            for fold_dates in folds:
                fold_rows = [r for r in dev if r["date"] in fold_dates]
                bets = [r for r in fold_rows if qualifies(r, cfg)]
                fm = metrics(bets)
                fold_metrics.append(fm)
                all_selected.extend(bets)

            pooled = metrics(all_selected)
            usable_folds = [m for m in fold_metrics if m["n"] >= 5 and m["roi"] is not None]
            positive_folds = [m for m in usable_folds if m["roi"] > 0]

            if pooled["n"] < 20 or len(usable_folds) < 2 or len(positive_folds) < 2:
                continue

            fold_rois = [float(m["roi"]) for m in usable_folds]
            dispersion = statistics.pstdev(fold_rois) if len(fold_rois) > 1 else 0.0
            worst = min(fold_rois)
            robustness = (
                float(pooled["roi"])
                - 0.40 * dispersion
                - 0.15 * max(0.0, -worst)
                + min(pooled["n"], 80) / 1000.0
            )
            ranked.append((robustness, cfg, pooled, fold_metrics))

        ranked.sort(key=lambda x: x[0], reverse=True)

        print(f"Configurações testadas    : {len(configs)}")
        print(f"Elegíveis por robustez    : {len(ranked)}\n")
        print("TOP 10 APENAS NO DESENVOLVIMENTO:")
        for score, cfg, pooled, fms in ranked[:10]:
            fold_text = " / ".join(
                f"n={m['n']} roi={m['roi']:.1%}"
                for m in fms
                if m["roi"] is not None
            )
            print(
                f"{cfg.name:28s} | n={pooled['n']:3d} roi={pooled['roi']:6.1%} "
                f"dd={pooled['max_drawdown']:.2f}u robust={score:.3f} | {fold_text}"
            )

        if not ranked:
            print("\nNenhuma configuração passou os critérios de robustez.")
            return

        _, winner, dev_metric, _ = ranked[0]

        # O holdout só é aberto agora, depois do vencedor estar congelado.
        final_bets = [r for r in holdout if qualifies(r, winner)]
        final_metric = metrics(final_bets)
        ci_lo, ci_hi = bootstrap_roi_by_date(final_bets)

        low_clv = [
            r for r in final_bets
            if r.get("clv") is not None and float(r["clv"]) < 0.55
        ]
        low_clv_metric = metrics(low_clv)

        print("\n==============================================")
        print(" VENCEDOR CONGELADO -> ABERTURA DO HOLDOUT")
        print("==============================================")
        print(f"Configuração             : {winner.name}")
        print(f"Família                  : {winner.family}")
        print(f"Desenvolvimento n        : {dev_metric['n']}")
        print(f"Desenvolvimento ROI      : {dev_metric['roi']:.2%}")
        print(f"Holdout bets             : {final_metric['n']}")

        if final_metric["n"]:
            print(f"Holdout hit rate         : {final_metric['hit']:.2%}")
            print(f"Holdout lucro            : {final_metric['profit']:.2f}u")
            print(f"Holdout ROI              : {final_metric['roi']:.2%}")
            print(f"Holdout odd média        : {final_metric['avg_odd']:.2f}")
            print(f"Holdout max drawdown     : {final_metric['max_drawdown']:.2f}u")
            if ci_lo is not None and ci_hi is not None:
                print(f"ROI bootstrap 95%        : [{ci_lo:.2%}, {ci_hi:.2%}]")

        print(f"Holdout BET CLV<55       : {low_clv_metric['n']}")
        if low_clv_metric["n"]:
            print(f"ROI CLV<55              : {low_clv_metric['roi']:.2%}")

        print("\nExemplos das primeiras 15 apostas do holdout:")
        for r in final_bets[:15]:
            s = r["analysts"]
            print(
                f"{r['date']} | {r['home_team']} vs {r['away_team']} | "
                f"G={s.goal_profile:.1f} A={s.attack_pressure:.1f} "
                f"ML={s.ml_score:.1f} V={s.value_score:.1f} "
                f"odd={r['odd']:.2f} result={'WIN' if r['result'] else 'LOSS'} "
                f"profit={r['profit']:.2f}u"
            )
    finally:
        c.close()


if __name__ == "__main__":
    main()
