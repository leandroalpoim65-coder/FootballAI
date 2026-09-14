from __future__ import annotations

import goal_model_performance_report_v2_cloud as v2


report = v2.report
DIAGNOSTIC_ONLY_MARKETS = {"h2h"}


def best_rankable_per_fixture(rows: list[dict]) -> list[dict]:
    selected: dict[int, dict] = {}
    for row in rows:
        if str(row.get("market") or "") in DIAGNOSTIC_ONLY_MARKETS:
            continue
        shadow_id = int(row.get("shadow_id") or 0)
        current = selected.get(shadow_id)
        if current is None or float(row["ev"]) > float(current["ev"]):
            selected[shadow_id] = row
    return list(selected.values())


def build_summaries_v3(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []

    # Overall decision metrics exclude markets explicitly held as diagnostic.
    best_rows = best_rankable_per_fixture(rows)
    best_positive = [row for row in best_rows if float(row["ev"]) > 0.0]
    positive_rows = [row for row in rows if float(row["ev"]) > 0.0]

    # Family comparisons still include 1X2 so we can monitor whether future
    # calibration improves it, but 1X2 cannot contaminate overall selection.
    family_rows = v2.best_market_family_per_fixture(rows)
    family_positive = [row for row in family_rows if float(row["ev"]) > 0.0]

    report.add_summary(
        summaries,
        "OVERALL",
        "best_rankable_all",
        "Best rankable market per fixture — 1X2 diagnostic-only",
        best_rows,
    )
    report.add_summary(
        summaries,
        "OVERALL",
        "best_rankable_positive",
        "Best rankable market per fixture — EV > 0%",
        best_positive,
    )
    report.add_summary(
        summaries,
        "OVERALL",
        "all_market_lines_positive",
        "Every market/line candidate — EV > 0% (includes diagnostics)",
        positive_rows,
    )
    report.add_summary(
        summaries,
        "OVERALL",
        "best_per_market_family_positive",
        "Best line per market family/fixture — EV > 0%",
        family_positive,
    )

    for label, group_rows in report.grouped(
        family_positive,
        lambda row: row["market_group"],
    ).items():
        report.add_summary(summaries, "MARKET", label, label, group_rows)

    for label, group_rows in report.grouped(
        positive_rows,
        lambda row: row["market_label"],
    ).items():
        report.add_summary(summaries, "MARKET_LINE", label, label, group_rows)

    for label, group_rows in report.grouped(
        best_positive,
        lambda row: row["league"],
    ).items():
        report.add_summary(summaries, "LEAGUE_BEST", label, label, group_rows)

    for label, group_rows in report.grouped(
        best_positive,
        lambda row: report.odd_bucket(float(row["odd"])),
    ).items():
        report.add_summary(
            summaries,
            "ODD_BUCKET_BEST",
            label,
            label,
            group_rows,
        )

    for label, group_rows in report.grouped(
        best_rows,
        lambda row: report.ev_bucket(float(row["ev"])),
    ).items():
        report.add_summary(
            summaries,
            "EV_BUCKET_BEST",
            label,
            label,
            group_rows,
        )

    for threshold in (0.00, 0.02, 0.05, 0.08, 0.10, 0.15):
        threshold_rows = [
            row for row in best_rows if float(row["ev"]) >= threshold
        ]
        report.add_summary(
            summaries,
            "EV_THRESHOLD_BEST",
            f"{threshold:.4f}",
            f"Rankable EV >= {threshold:.0%}",
            threshold_rows,
        )

    for market_name, group_rows in report.grouped(
        family_rows,
        lambda row: row["market_group"],
    ).items():
        for threshold in (0.00, 0.05, 0.08, 0.10, 0.15):
            threshold_rows = [
                row for row in group_rows if float(row["ev"]) >= threshold
            ]
            report.add_summary(
                summaries,
                "MARKET_EV_THRESHOLD",
                f"{market_name}|{threshold:.4f}",
                f"{market_name} | EV >= {threshold:.0%}",
                threshold_rows,
            )

    return summaries


report.build_summaries = build_summaries_v3


if __name__ == "__main__":
    report.main()
