from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


def normalize(text: Any) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def pct(value: Any, digits: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def num(value: Any, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def first_row(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def agent_snapshot(
    decisions: pd.DataFrame,
    policies: pd.DataFrame,
    learning: pd.DataFrame,
    performance: pd.DataFrame,
) -> dict[str, Any]:
    active = first_row(policies)
    learner = first_row(learning)
    perf = first_row(performance)

    counts: dict[str, int] = {}
    if decisions is not None and not decisions.empty and "decision" in decisions:
        counts = {
            str(key): int(value)
            for key, value in decisions["decision"].fillna("UNKNOWN").value_counts().items()
        }

    return {
        "policy": active,
        "learner": learner,
        "performance": perf,
        "counts": counts,
        "decision_total": int(len(decisions)) if decisions is not None else 0,
    }


def _policy_answer(snapshot: dict[str, Any]) -> str:
    p = snapshot.get("policy") or {}
    if not p:
        return "Ainda não encontro uma política ativa na Turso."

    return (
        f"A política ativa é **{p.get('policy_version', '—')}**. "
        f"Os mínimos atuais são **CLV {pct(p.get('min_clv_probability'))}**, "
        f"**EV {pct(p.get('min_expected_value'))}** e "
        f"**edge {pct(p.get('min_edge_market'))}**. "
        f"A faixa de odds permitida é **{num(p.get('min_odd'), 2)}–{num(p.get('max_odd'), 2)}**."
    )


def _learning_answer(snapshot: dict[str, Any]) -> str:
    row = snapshot.get("learner") or {}
    if not row:
        return "O learner ainda não tem uma execução registada."

    decision = str(row.get("decision") or "—")
    resolved = int(row.get("resolved_rows") or 0)
    train = int(row.get("train_rows") or 0)
    validation = int(row.get("validation_rows") or 0)
    promoted = row.get("promoted_policy_version") or "nenhuma"

    return (
        f"O learner está em **{decision}**. Na última execução tinha "
        f"**{resolved} jogos resolvidos**, com **{train}** no treino temporal e "
        f"**{validation}** na validação. Política promovida nessa execução: "
        f"**{promoted}**. Enquanto a evidência não passar os gates estatísticos, "
        "ele mantém a política atual."
    )


def _performance_answer(snapshot: dict[str, Any]) -> str:
    row = snapshot.get("performance") or {}
    if not row:
        return "O Performance Monitor ainda não tem uma execução disponível."

    return (
        f"Último monitor: estado **{row.get('monitor_state', '—')}**, "
        f"**{int(row.get('resolved_agent_sample') or 0)}** decisões resolvidas, "
        f"**{int(row.get('qualified_sample') or 0)} QUALIFIED**. "
        f"ROI original: **{pct(row.get('original_roi'))}**; "
        f"ROI QUALIFIED: **{pct(row.get('qualified_roi'))}**; "
        f"lift: **{pct(row.get('roi_lift'))}**. "
        f"Retenção de vencedoras: **{pct(row.get('winner_retention'))}**; "
        f"rejeição de perdedoras: **{pct(row.get('loser_rejection'))}**."
    )


def _decision_summary(snapshot: dict[str, Any]) -> str:
    counts = snapshot.get("counts") or {}
    total = int(snapshot.get("decision_total") or 0)
    return (
        f"Tenho **{total} decisões** registadas: "
        f"**{counts.get('QUALIFIED_SHADOW', 0)} QUALIFIED**, "
        f"**{counts.get('WATCH', 0)} WATCH** e "
        f"**{counts.get('REJECT', 0)} REJECT**."
    )


def _league_answer(decisions: pd.DataFrame) -> str:
    if decisions is None or decisions.empty or "league" not in decisions:
        return "Ainda não há decisões suficientes para comparar ligas."

    resolved = decisions.copy()
    if "result_over25" not in resolved or "odd_over25_open" not in resolved:
        return "Ainda não tenho resultados e odds suficientes para calcular ROI por liga."

    resolved["result_over25"] = pd.to_numeric(resolved["result_over25"], errors="coerce")
    resolved["odd_over25_open"] = pd.to_numeric(resolved["odd_over25_open"], errors="coerce")
    resolved = resolved[
        resolved["result_over25"].isin([0, 1])
        & resolved["odd_over25_open"].gt(1)
    ].copy()

    if resolved.empty:
        return "As decisões do agente ainda não têm resultados resolvidos suficientes para ROI por liga."

    resolved["unit_profit"] = resolved.apply(
        lambda row: row["odd_over25_open"] - 1.0 if row["result_over25"] == 1 else -1.0,
        axis=1,
    )
    grouped = (
        resolved.groupby("league", dropna=False)
        .agg(Jogos=("unit_profit", "size"), ROI=("unit_profit", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["Jogos"] >= 5].sort_values("ROI", ascending=False)
    if grouped.empty:
        return "Ainda não há nenhuma liga com pelo menos 5 decisões resolvidas; prefiro não tirar conclusões cedo demais."

    best = grouped.head(3)
    worst = grouped.tail(3).sort_values("ROI")
    best_text = ", ".join(
        f"{row.league} ({int(row.Jogos)} jogos, ROI {pct(row.ROI)})"
        for row in best.itertuples()
    )
    worst_text = ", ".join(
        f"{row.league} ({int(row.Jogos)} jogos, ROI {pct(row.ROI)})"
        for row in worst.itertuples()
    )
    return f"Melhores ligas com amostra mínima: **{best_text}**. Mais fracas: **{worst_text}**."


def _find_game(question: str, decisions: pd.DataFrame) -> pd.Series | None:
    if decisions is None or decisions.empty:
        return None

    q = normalize(question)
    best_index = None
    best_score = 0

    for index, row in decisions.iterrows():
        home = normalize(row.get("home_team"))
        away = normalize(row.get("away_team"))
        score = 0
        if home and home in q:
            score += 2
        if away and away in q:
            score += 2
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None or best_score < 2:
        return None
    return decisions.loc[best_index]


def _game_answer(row: pd.Series) -> str:
    reasons = str(row.get("reasons_json") or "")
    return (
        f"**{row.get('home_team', '—')} vs {row.get('away_team', '—')}**: "
        f"decisão **{row.get('decision', '—')} / {row.get('tier', '—')}**, "
        f"política **{row.get('policy_version', '—')}**. "
        f"CLV probability **{pct(row.get('clv_probability'))}**, "
        f"sports probability **{pct(row.get('sports_probability'))}**, "
        f"EV **{pct(row.get('expected_value'))}**, edge **{pct(row.get('sports_edge_market'))}**, "
        f"odd OPEN **{num(row.get('odd_over25_open'), 3)}**. "
        + (f"Motivos guardados: `{reasons}`" if reasons else "")
    )


def answer_question(
    question: str,
    decisions: pd.DataFrame,
    policies: pd.DataFrame,
    learning: pd.DataFrame,
    performance: pd.DataFrame,
) -> str:
    snapshot = agent_snapshot(decisions, policies, learning, performance)
    q = normalize(question)

    game = _find_game(question, decisions)
    if game is not None and any(word in q for word in ("porque", "jogo", "rejeit", "watch", "qualific", "decis")):
        return _game_answer(game)

    if any(word in q for word in ("threshold", "politica", "limiar", "clv minimo", "ev minimo", "edge minimo")):
        return _policy_answer(snapshot)

    if any(word in q for word in ("aprend", "learner", "treino", "validacao", "promov")):
        return _learning_answer(snapshot)

    if any(word in q for word in ("roi", "performance", "resultado", "retencao", "perdedor")):
        if any(word in q for word in ("liga", "ligas")):
            return _league_answer(decisions)
        return _performance_answer(snapshot)

    if any(word in q for word in ("liga", "ligas")):
        return _league_answer(decisions)

    if any(word in q for word in ("quant", "decis", "qualified", "watch", "reject", "estado")):
        return _decision_summary(snapshot) + " " + _learning_answer(snapshot)

    return (
        _decision_summary(snapshot)
        + " "
        + _policy_answer(snapshot)
        + " "
        + _learning_answer(snapshot)
        + " Podes perguntar-me por ROI, ligas, thresholds, aprendizagem ou pelo motivo de uma decisão específica."
    )
