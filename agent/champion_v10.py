from __future__ import annotations

from dataclasses import dataclass

from agent.multi_analyst import AnalystScores


CHAMPION_VERSION = "v10-vote3-g55-a55-m65-v60"
GOAL_THRESHOLD = 55.0
ATTACK_THRESHOLD = 55.0
ML_THRESHOLD = 65.0
VALUE_THRESHOLD = 60.0
MIN_VOTES = 3
MIN_EXPECTED_VALUE = 0.0


@dataclass(frozen=True)
class ChampionDecision:
    decision: str
    vote_count: int
    goal_vote: bool
    attack_vote: bool
    ml_vote: bool
    value_vote: bool


def evaluate_champion(
    scores: AnalystScores,
    expected_value: float,
) -> ChampionDecision:
    goal_vote = scores.goal_profile >= GOAL_THRESHOLD
    attack_vote = scores.attack_pressure >= ATTACK_THRESHOLD
    ml_vote = scores.ml_score >= ML_THRESHOLD
    value_vote = scores.value_score >= VALUE_THRESHOLD

    vote_count = sum((goal_vote, attack_vote, ml_vote, value_vote))
    qualifies = (
        expected_value >= MIN_EXPECTED_VALUE
        and vote_count >= MIN_VOTES
    )

    return ChampionDecision(
        decision="BET_SHADOW" if qualifies else "PASS",
        vote_count=vote_count,
        goal_vote=goal_vote,
        attack_vote=attack_vote,
        ml_vote=ml_vote,
        value_vote=value_vote,
    )
