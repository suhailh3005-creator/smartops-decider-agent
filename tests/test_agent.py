from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.agent import SmartOpsAgent
from src.bedrock_report import EvidenceClaim, EvidenceItem, IncidentReport
from src.decider_client import DecisionStatus, IncidentTool, ToolDecision
from src.tools import TOOL_HANDLERS, query_metrics


class FixedDecider:
    def __init__(self, decision: ToolDecision) -> None:
        self.decision = decision

    def decide(self, incident: str) -> ToolDecision:
        return self.decision


def decision(tool: IncidentTool) -> ToolDecision:
    other_probability = 0.04 / (len(IncidentTool) - 1)
    probabilities = {candidate.value: other_probability for candidate in IncidentTool}
    probabilities[tool.value] = 0.96
    return ToolDecision(
        tool=tool,
        confidence=(len(IncidentTool) * 0.96 - 1.0) / (len(IncidentTool) - 1),
        probabilities=probabilities,
        status=(
            DecisionStatus.MODEL_ESCALATION
            if tool is IncidentTool.ESCALATE
            else DecisionStatus.SELECTED
        ),
        detail="test decision",
    )


class SmartOpsAgentTests(unittest.TestCase):
    def test_executes_only_the_selected_mock_tool(self) -> None:
        handlers = {tool: Mock(wraps=handler) for tool, handler in TOOL_HANDLERS.items()}
        incident = "payment errors"

        with patch.dict(TOOL_HANDLERS, handlers):
            outcome = SmartOpsAgent(FixedDecider(decision(IncidentTool.METRICS))).investigate(
                incident
            )

        handlers[IncidentTool.LOGS].assert_not_called()
        handlers[IncidentTool.METRICS].assert_called_once_with(incident)
        handlers[IncidentTool.RUNBOOK].assert_not_called()
        self.assertIsNotNone(outcome.execution.result)
        self.assertEqual(outcome.execution.result.tool, IncidentTool.METRICS)

    def test_escalation_does_not_execute_any_mock_tool(self) -> None:
        handlers = {tool: Mock(wraps=handler) for tool, handler in TOOL_HANDLERS.items()}
        report_generator = Mock()

        with patch.dict(TOOL_HANDLERS, handlers):
            outcome = SmartOpsAgent(
                FixedDecider(decision(IncidentTool.ESCALATE)), report_generator
            ).investigate("unclear incident")

        for handler in handlers.values():
            handler.assert_not_called()
        report_generator.generate.assert_not_called()
        self.assertIsNone(outcome.execution.result)

    def test_report_generator_receives_only_the_selected_tool_evidence(self) -> None:
        incident = "payment latency increased"
        result = query_metrics(incident)
        report = IncidentReport(
            incident=incident,
            tool=IncidentTool.METRICS,
            summary=EvidenceClaim(text="Latency increased.", evidence_ids=["E1"]),
            findings=[],
            recommended_actions=[],
            evidence=[
                EvidenceItem(id="E1", source=result.source, observation=result.observations[0])
            ],
        )
        report_generator = Mock()
        report_generator.generate.return_value = report

        outcome = SmartOpsAgent(
            FixedDecider(decision(IncidentTool.METRICS)), report_generator
        ).investigate(incident)

        report_generator.generate.assert_called_once_with(incident, result)
        self.assertIs(outcome.report, report)

    def test_empty_incident_is_rejected_before_decision(self) -> None:
        decider = Mock()

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            SmartOpsAgent(decider).investigate(" ")

        decider.decide.assert_not_called()


if __name__ == "__main__":
    unittest.main()
