from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.outcome_learning import (
    CHAMPION_POLICY,
    build_patterns,
    challenger_gate,
    choose_candidate,
    metrics,
    unique_date_split,
)
from footballai_agent_cloud import connect_turso, fetch_dicts, table_exists


SOURCE_TABLE = "agent_v10_universe_decisions"
RUN_TABLE = "agent_v11_outcome_runs"
PATTERN_TABLE = "agent_v11_outcome_patterns"
MIN_RESOLVED_ROWS = 120
MIN_UNIQUE_DATES = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_tables(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            state TEXT NOT NULL,
            resolved_rows INTEGER NOT NULL,
            bet_rows INTEGER NOT NULL,
            pass_rows INTEGER NOT NULL,
            unique_dates INTEGER NOT NULL,
            split_date TEXT,
            train_rows INTEGER NOT NULL,
            validation_rows INTEGER NOT NULL,
            policies_evaluated INTEGER NOT NULL,
            champion_policy_json TEXT NOT NULL,
            candidate_policy_json TEXT,
            champion_train_json TEXT NOT NULL,
            candidate_train_json TEXT,
            champion_validation_json TEXT NOT NULL,
            candidate_validation_json TEXT,
            gate_passed INTEGER NOT NULL,
            reasons_json TEXT NOT NULL,
            pattern_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PATTERN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            dimension TEXT NOT NULL,
            segment TEXT NOT NULL,
            n INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            over_rate REAL,
            avg_odd REAL,
            counterfactual_profit REAL NOT NULL,
            counterfactual_roi REAL,
            FOREIGN KEY(run_id) REFERENCES {RUN_TABLE}(id)
        )
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{PATTERN_TABLE}_run
        ON {PATTERN_TABLE}(run_id, dimension, n DESC)
        """
    )
    connection.commit()


def load_resolved(connection) -> list[dict]:
    if not table_exists(connection, SOURCE_TABLE):
        return []

    return fetch_dicts(
        connection,
        f"""
        SELECT
            id,
            fixture_key,
            fixture_date,
            fixture_time,
            league,
            home_team,
            away_team,
            odd_over25_open,
            market_probability,
            sports_probability,
            expected_value,
            sports_edge_market,
            clv_probability,
            goal_score,
            attack_score,
            ml_score,
            value_score,
            stats_score,
            goal_vote,
            attack_vote,
            ml_vote,
            value_vote,
            vote_count,
            decision,
            result_over25,
            profit,
            resolution_source,
            resolved_at
        FROM {SOURCE_TABLE}
        WHERE result_over25 IN (0, 1)
        ORDER BY fixture_date, fixture_time, id
        """,
    )


def save_run(
    connection,
    *,
    started_at: str,
    state: str,
    rows: list[dict],
    split_date: str | None,
    train_rows: int,
    validation_rows: int,
    policies_evaluated: int,
    candidate_policy,
    champion_train,
    candidate_train,
    champion_validation,
    candidate_validation,
    gate_passed: bool,
    reasons: list[str],
    patterns: list[dict],
) -> int:
    finished_at = utc_now()
    bets = sum(str(row.get("decision")) == "BET_SHADOW" for row in rows)
    passes = sum(str(row.get("decision")) == "PASS" for row in rows)
    unique_dates = len({str(row.get("fixture_date")) for row in rows})

    cursor = connection.execute(
        f"""
        INSERT INTO {RUN_TABLE} (
            started_at, finished_at, state, resolved_rows, bet_rows, pass_rows,
            unique_dates, split_date, train_rows, validation_rows,
            policies_evaluated, champion_policy_json, candidate_policy_json,
            champion_train_json, candidate_train_json,
            champion_validation_json, candidate_validation_json,
            gate_passed, reasons_json, pattern_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            started_at,
            finished_at,
            state,
            len(rows),
            bets,
            passes,
            unique_dates,
            split_date,
            train_rows,
            validation_rows,
            policies_evaluated,
            json.dumps(CHAMPION_POLICY.to_dict(), sort_keys=True),
            json.dumps(candidate_policy.to_dict(), sort_keys=True)
            if candidate_policy is not None else None,
            json.dumps(champion_train.to_dict(), sort_keys=True),
            json.dumps(candidate_train.to_dict(), sort_keys=True)
            if candidate_train is not None else None,
            json.dumps(champion_validation.to_dict(), sort_keys=True),
            json.dumps(candidate_validation.to_dict(), sort_keys=True)
            if candidate_validation is not None else None,
            int(gate_passed),
            json.dumps(reasons, ensure_ascii=False),
            len(patterns),
        ),
    )
    run_id = int(cursor.lastrowid)

    for pattern in patterns:
        connection.execute(
            f"""
            INSERT INTO {PATTERN_TABLE} (
                run_id, dimension, segment, n, wins, over_rate, avg_odd,
                counterfactual_profit, counterfactual_roi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                pattern["dimension"],
                pattern["segment"],
                int(pattern["n"]),
                int(pattern["wins"]),
                pattern["over_rate"],
                pattern["avg_odd"],
                float(pattern["counterfactual_profit"]),
                pattern["counterfactual_roi"],
            ),
        )

    connection.commit()
    return run_id


def pct(value) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def print_metric(label: str, metric) -> None:
    print(
        f"{label:25s}: n={metric.n:3d} "
        f"hit={pct(metric.hit_rate):>7s} roi={pct(metric.roi):>7s} "
        f"profit={metric.profit:+.2f}u dd={metric.max_drawdown:.2f}u "
        f"L={metric.longest_losing_streak}"
    )


def _counterfactual_profit(row: dict) -> float:
    odd = float(row.get("odd_over25_open") or 0.0)
    return odd - 1.0 if int(row.get("result_over25") or 0) == 1 else -1.0


def _failed_votes(row: dict) -> list[str]:
    failed: list[str] = []
    if float(row.get("goal_score") or 0.0) < CHAMPION_POLICY.goal_threshold:
        failed.append("G")
    if float(row.get("attack_score") or 0.0) < CHAMPION_POLICY.attack_threshold:
        failed.append("A")
    if float(row.get("ml_score") or 0.0) < CHAMPION_POLICY.ml_threshold:
        failed.append("M")
    if float(row.get("value_score") or 0.0) < CHAMPION_POLICY.value_threshold:
        failed.append("V")
    return failed


def print_resolved_audit(rows: list[dict], patterns: list[dict]) -> None:
    pass_rows = [row for row in rows if str(row.get("decision")) == "PASS"]
    bet_rows = [row for row in rows if str(row.get("decision")) == "BET_SHADOW"]

    decision_patterns = {
        str(pattern.get("segment")): pattern
        for pattern in patterns
        if str(pattern.get("dimension")) == "decision"
    }
    pass_pattern = decision_patterns.get("PASS")

    print("\n----------------------------------------------")
    print(" AUDITORIA PROSPECTIVA — BET vs PASS")
    print("----------------------------------------------")
    print(f"BET_SHADOW resolvidas    : {len(bet_rows)}")
    print(f"PASS resolvidos          : {len(pass_rows)}")

    if pass_pattern is not None:
        n = int(pass_pattern["n"])
        overs = int(pass_pattern["wins"])
        unders = n - overs
        print(f"PASS que deram Over 2.5  : {overs}")
        print(f"PASS que deram Under 2.5 : {unders}")
        print(f"Over rate dos PASS       : {pct(pass_pattern['over_rate'])}")
        print(f"Odd média dos PASS       : {float(pass_pattern['avg_odd']):.2f}")
        print(
            "Lucro se apostasse PASS  : "
            f"{float(pass_pattern['counterfactual_profit']):+.2f}u"
        )
        print(
            "ROI se apostasse PASS    : "
            f"{pct(pass_pattern['counterfactual_roi'])}"
        )

    near_vote = []
    ev_gate_only = []
    for row in pass_rows:
        ev = float(row.get("expected_value") or 0.0)
        votes = int(row.get("vote_count") or 0)
        if ev >= CHAMPION_POLICY.min_expected_value and votes == CHAMPION_POLICY.min_votes - 1:
            near_vote.append(row)
        if ev < CHAMPION_POLICY.min_expected_value and votes >= CHAMPION_POLICY.min_votes:
            ev_gate_only.append(row)

    def print_subset(label: str, subset: list[dict]) -> None:
        overs = sum(int(row.get("result_over25") or 0) for row in subset)
        profit = sum(_counterfactual_profit(row) for row in subset)
        roi = profit / len(subset) if subset else None
        print(f"\n{label}")
        print(f"  Jogos                  : {len(subset)}")
        print(f"  Overs / Unders         : {overs} / {len(subset) - overs}")
        print(f"  Over rate              : {pct(overs / len(subset) if subset else None)}")
        print(f"  Lucro contrafactual    : {profit:+.2f}u")
        print(f"  ROI contrafactual      : {pct(roi)}")
        for row in subset:
            result = "OVER" if int(row.get("result_over25") or 0) else "UNDER"
            failed = "+".join(_failed_votes(row)) or "EV"
            print(
                "  - "
                f"{row.get('fixture_date')} | {row.get('home_team')} vs {row.get('away_team')} "
                f"| {result} | odd={float(row.get('odd_over25_open') or 0):.2f} "
                f"| votes={int(row.get('vote_count') or 0)} "
                f"| falhou={failed} "
                f"| G={float(row.get('goal_score') or 0):.1f} "
                f"A={float(row.get('attack_score') or 0):.1f} "
                f"M={float(row.get('ml_score') or 0):.1f} "
                f"V={float(row.get('value_score') or 0):.1f} "
                f"EV={float(row.get('expected_value') or 0):+.3f}"
            )

    print_subset("PASS a 1 voto da BET (EV >= 0)", near_vote)
    print_subset("PASS bloqueados apenas pelo EV (< 0)", ev_gate_only)

    print("\nTodos os PASS resolvidos:")
    for row in pass_rows:
        result = "OVER" if int(row.get("result_over25") or 0) else "UNDER"
        failed = "+".join(_failed_votes(row)) or "EV"
        print(
            "  - "
            f"{row.get('fixture_date')} | {row.get('home_team')} vs {row.get('away_team')} "
            f"| {result} | odd={float(row.get('odd_over25_open') or 0):.2f} "
            f"| votes={int(row.get('vote_count') or 0)} "
            f"| falhou={failed} "
            f"| G={float(row.get('goal_score') or 0):.1f} "
            f"A={float(row.get('attack_score') or 0):.1f} "
            f"M={float(row.get('ml_score') or 0):.1f} "
            f"V={float(row.get('value_score') or 0):.1f} "
            f"EV={float(row.get('expected_value') or 0):+.3f}"
        )


def main() -> None:
    started_at = utc_now()
    print("\n==============================================")
    print(" FOOTBALLAI — V11 OUTCOME LEARNER")
    print("==============================================\n")
    print("Verdade supervisionada : resultado Over 2.5 real")
    print("Aprende com            : BET_SHADOW + PASS")
    print("CLV como label/gate?   : NÃO")
    print("Auto-altera Champion?  : NÃO\n")

    connection = connect_turso()
    try:
        ensure_tables(connection)
        rows = load_resolved(connection)
        patterns = build_patterns(rows)
        unique_dates = len({str(row.get("fixture_date")) for row in rows})

        print(f"Jogos V10 resolvidos    : {len(rows)}")
        print(f"Datas únicas resolvidas : {unique_dates}")
        print(f"Padrões calculados      : {len(patterns)}")
        print_resolved_audit(rows, patterns)

        empty_metric = metrics([], CHAMPION_POLICY)

        if len(rows) < MIN_RESOLVED_ROWS or unique_dates < MIN_UNIQUE_DATES:
            reasons = []
            if len(rows) < MIN_RESOLVED_ROWS:
                reasons.append(
                    f"resolved_rows={len(rows)} < {MIN_RESOLVED_ROWS}"
                )
            if unique_dates < MIN_UNIQUE_DATES:
                reasons.append(
                    f"unique_dates={unique_dates} < {MIN_UNIQUE_DATES}"
                )

            run_id = save_run(
                connection,
                started_at=started_at,
                state="WAIT_SAMPLE",
                rows=rows,
                split_date=None,
                train_rows=0,
                validation_rows=0,
                policies_evaluated=0,
                candidate_policy=None,
                champion_train=empty_metric,
                candidate_train=None,
                champion_validation=empty_metric,
                candidate_validation=None,
                gate_passed=False,
                reasons=reasons,
                patterns=patterns,
            )
            print(f"\nEstado                 : WAIT_SAMPLE")
            print(f"Run V11                : {run_id}")
            for reason in reasons:
                print(f"- {reason}")
            return

        train, validation, split_date = unique_date_split(rows, 0.70)
        champion_train = metrics(train, CHAMPION_POLICY)
        champion_validation = metrics(validation, CHAMPION_POLICY)
        candidate_policy, candidate_train, policies_evaluated = choose_candidate(train)

        print(f"\nSplit validação         : {split_date}")
        print(f"Treino / validação      : {len(train)} / {len(validation)}")
        print(f"Políticas testadas      : {policies_evaluated}")
        print_metric("Champion treino", champion_train)
        print_metric("Champion validação", champion_validation)

        if candidate_policy is None or candidate_train is None:
            run_id = save_run(
                connection,
                started_at=started_at,
                state="KEEP_CHAMPION",
                rows=rows,
                split_date=split_date,
                train_rows=len(train),
                validation_rows=len(validation),
                policies_evaluated=policies_evaluated,
                candidate_policy=None,
                champion_train=champion_train,
                candidate_train=None,
                champion_validation=champion_validation,
                candidate_validation=None,
                gate_passed=False,
                reasons=["no_positive_train_candidate"],
                patterns=patterns,
            )
            print("Estado                  : KEEP_CHAMPION")
            print(f"Run V11                 : {run_id}")
            return

        candidate_validation = metrics(validation, candidate_policy)
        passed, reasons = challenger_gate(
            champion_validation,
            candidate_validation,
        )
        state = "CHALLENGER_READY" if passed else "KEEP_CHAMPION"

        print("\nCandidate congelado no treino:")
        print(json.dumps(candidate_policy.to_dict(), indent=2, sort_keys=True))
        print_metric("Candidate treino", candidate_train)
        print_metric("Candidate validação", candidate_validation)
        print(f"Gate challenger         : {'PASSOU' if passed else 'NÃO PASSOU'}")
        if reasons:
            for reason in reasons:
                print(f"- {reason}")

        run_id = save_run(
            connection,
            started_at=started_at,
            state=state,
            rows=rows,
            split_date=split_date,
            train_rows=len(train),
            validation_rows=len(validation),
            policies_evaluated=policies_evaluated,
            candidate_policy=candidate_policy,
            champion_train=champion_train,
            candidate_train=candidate_train,
            champion_validation=champion_validation,
            candidate_validation=candidate_validation,
            gate_passed=passed,
            reasons=reasons,
            patterns=patterns,
        )
        print(f"\nEstado final            : {state}")
        print(f"Run V11                 : {run_id}")
        print("Nota: nenhuma política foi promovida automaticamente.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
