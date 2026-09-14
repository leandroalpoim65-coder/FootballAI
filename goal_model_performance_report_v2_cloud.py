from __future__ import annotations

import goal_model_performance_report_cloud as report


def best_market_family_per_fixture(rows: list[dict]) -> list[dict]:
    """
    Para comparar famílias de mercado de forma justa, mantém apenas a
    oportunidade com maior EV de cada família em cada jogo.

    Exemplo: se Goals Totals tiver 2.0, 2.25, 2.5 e 2.75, conta apenas a
    melhor linha desse jogo no resumo da família. O relatório MARKET_LINE
    continua a avaliar cada linha individual separadamente.
    """
    selected: dict[tuple[int, str], dict] = {}

    for row in rows:
        key = (
            int(row.get("shadow_id") or 0),
            str(row.get("market_group") or ""),
        )
        current = selected.get(key)
        if current is None or float(row["ev"]) > float(current["ev"]):
            selected[key] = row

    return list(selected.values())


def build_summaries_v2(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []

    best_rows = [
        row
        for row in rows
        if int(row.get("is_best_overall") or 0) == 1
    ]
    best_positive = [
        row
        for row in best_rows
        if float(row["ev"]) > 0.0
    ]
    positive_rows = [
        row
        for row in rows
        if float(row["ev"]) > 0.0
    ]

    family_rows = best_market_family_per_fixture(rows)
    family_positive = [
        row
        for row in family_rows
        if float(row["ev"]) > 0.0
    ]

    report.add_summary(
        summaries,
        "OVERALL",
        "best_all",
        "Best market per fixture — all EV",
        best_rows,
    )
    report.add_summary(
        summaries,
        "OVERALL",
        "best_positive",
        "Best market per fixture — EV > 0%",
        best_positive,
    )
    report.add_summary(
        summaries,
        "OVERALL",
        "all_market_lines_positive",
        "Every market/line candidate — EV > 0%",
        positive_rows,
    )
    report.add_summary(
        summaries,
        "OVERALL",
        "best_per_market_family_positive",
        "Best line per market family/fixture — EV > 0%",
        family_positive,
    )

    # Comparação justa entre famílias: uma seleção por família e por jogo.
    for label, group_rows in report.grouped(
        family_positive,
        lambda row: row["market_group"],
    ).items():
        report.add_summary(
            summaries,
            "MARKET",
            label,
            label,
            group_rows,
        )

    # Aqui cada linha individual continua a ser analisada isoladamente.
    for label, group_rows in report.grouped(
        positive_rows,
        lambda row: row["market_label"],
    ).items():
        report.add_summary(
            summaries,
            "MARKET_LINE",
            label,
            label,
            group_rows,
        )

    for label, group_rows in report.grouped(
        best_positive,
        lambda row: row["league"],
    ).items():
        report.add_summary(
            summaries,
            "LEAGUE_BEST",
            label,
            label,
            group_rows,
        )

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
            row
            for row in best_rows
            if float(row["ev"]) >= threshold
        ]
        report.add_summary(
            summaries,
            "EV_THRESHOLD_BEST",
            f"{threshold:.4f}",
            f"EV >= {threshold:.0%}",
            threshold_rows,
        )

    # Threshold por família também usa apenas a melhor linha da família/jogo.
    for market_name, group_rows in report.grouped(
        family_rows,
        lambda row: row["market_group"],
    ).items():
        for threshold in (0.00, 0.05, 0.08, 0.10, 0.15):
            threshold_rows = [
                row
                for row in group_rows
                if float(row["ev"]) >= threshold
            ]
            report.add_summary(
                summaries,
                "MARKET_EV_THRESHOLD",
                f"{market_name}|{threshold:.4f}",
                f"{market_name} | EV >= {threshold:.0%}",
                threshold_rows,
            )

    return summaries


report.build_summaries = build_summaries_v2


if __name__ == "__main__":
    report.main()
