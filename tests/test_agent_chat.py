import unittest

import pandas as pd

from agent.chat_assistant import answer_question


class AgentChatTests(unittest.TestCase):
    def setUp(self):
        self.policies = pd.DataFrame([
            {
                "policy_version": "agent-v1.0",
                "min_clv_probability": 0.55,
                "min_expected_value": 0.03,
                "min_edge_market": 0.02,
                "min_odd": 1.30,
                "max_odd": 4.00,
            }
        ])
        self.learning = pd.DataFrame([
            {
                "decision": "WAIT",
                "resolved_rows": 0,
                "train_rows": 0,
                "validation_rows": 0,
                "promoted_policy_version": None,
            }
        ])
        self.performance = pd.DataFrame([
            {
                "monitor_state": "COLD_START",
                "resolved_agent_sample": 0,
                "qualified_sample": 0,
                "original_roi": None,
                "qualified_roi": None,
                "roi_lift": None,
                "winner_retention": None,
                "loser_rejection": None,
            }
        ])
        self.decisions = pd.DataFrame([
            {
                "home_team": "Stevenage",
                "away_team": "Luton",
                "league": "League One",
                "decision": "REJECT",
                "tier": "CLV_REJECT",
                "policy_version": "agent-v1.0",
                "clv_probability": 0.377,
                "sports_probability": 0.61,
                "expected_value": 0.093,
                "sports_edge_market": 0.04,
                "odd_over25_open": 1.79,
                "reasons_json": '["CLV abaixo do limiar"]',
                "result_over25": None,
            }
        ])

    def test_policy_question_uses_active_policy(self):
        answer = answer_question(
            "Qual é o threshold atual?",
            self.decisions,
            self.policies,
            self.learning,
            self.performance,
        )
        self.assertIn("agent-v1.0", answer)
        self.assertIn("55.0%", answer)

    def test_learning_question_reports_wait(self):
        answer = answer_question(
            "O que aprendeste até agora?",
            self.decisions,
            self.policies,
            self.learning,
            self.performance,
        )
        self.assertIn("WAIT", answer)
        self.assertIn("0 jogos resolvidos", answer)

    def test_game_question_explains_frozen_decision(self):
        answer = answer_question(
            "Porque rejeitaste Stevenage vs Luton?",
            self.decisions,
            self.policies,
            self.learning,
            self.performance,
        )
        self.assertIn("REJECT", answer)
        self.assertIn("37.7%", answer)
        self.assertIn("9.3%", answer)


if __name__ == "__main__":
    unittest.main()
