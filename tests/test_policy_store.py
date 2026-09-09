import sqlite3
import unittest

from agent.decision_engine import (
    DEFAULT_POLICY,
    PolicyConfig,
)
from agent.policy_store import (
    activate_policy,
    ensure_policy_table,
    load_active_policy,
)


class PolicyStoreTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(
            ":memory:"
        )

    def tearDown(self):
        self.connection.close()

    def test_bootstraps_default_policy(self):
        ensure_policy_table(
            self.connection
        )

        active = load_active_policy(
            self.connection
        )

        self.assertEqual(
            active.version,
            DEFAULT_POLICY.version,
        )

    def test_activation_retires_previous_policy(self):
        ensure_policy_table(
            self.connection
        )

        candidate = PolicyConfig(
            version="agent-auto-test",
            min_clv_probability=0.60,
            min_expected_value=0.05,
            min_edge_market=0.03,
        )

        activate_policy(
            self.connection,
            candidate,
            source="test",
            parent_version=DEFAULT_POLICY.version,
            evidence={"ok": True},
        )

        active = load_active_policy(
            self.connection
        )

        self.assertEqual(
            active.version,
            "agent-auto-test",
        )

        old_status = self.connection.execute(
            """
            SELECT status
            FROM agent_policies
            WHERE policy_version = ?
            """,
            (
                DEFAULT_POLICY.version,
            ),
        ).fetchone()[0]

        self.assertEqual(
            old_status,
            "RETIRED",
        )


if __name__ == "__main__":
    unittest.main()
