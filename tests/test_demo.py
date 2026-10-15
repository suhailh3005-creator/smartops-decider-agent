from __future__ import annotations

import unittest

from src.demo import run_demo
from src.decider_client import DecisionStatus, IncidentTool, ToolDecision


class FixedDecider:
    def decide(self, incident: str) -> ToolDecision:
        return ToolDecision(
            tool=IncidentTool.LOGS,
            confidence=0.96,
            probabilities={
                "LOGS": 0.97,
                "METRICS": 0.01,
                "RUNBOOK": 0.01,
                "ESCALATE": 0.01,
            },
            status=DecisionStatus.SELECTED,
            detail="test decision",
        )


class DemoTests(unittest.TestCase):
    def test_default_demo_path_does_not_generate_a_bedrock_report(self) -> None:
        outcome = run_demo("checkout failure", decider=FixedDecider())

        self.assertEqual(outcome.execution.result.tool, IncidentTool.LOGS)
        self.assertIsNone(outcome.report)

    def test_bedrock_opt_in_requires_an_explicit_model_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "SMARTOPS_BEDROCK_MODEL_ID"):
            run_demo("checkout failure", bedrock_enabled=True, decider=FixedDecider())


if __name__ == "__main__":
    unittest.main()
