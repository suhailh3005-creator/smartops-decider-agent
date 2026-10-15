from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.decider_client import (
    DEFAULT_CHECKPOINT,
    DecisionStatus,
    IncidentTool,
    StrandsDeciderClient,
)


def make_response(
    choice: str = "LOGS",
    confidence: float = 0.95,
    probabilities: dict[str, float] | None = None,
) -> SimpleNamespace:
    if probabilities is None:
        top_probability = (3 * confidence + 1) / 4
        remainder = (1 - top_probability) / 3
        probabilities = {
            "LOGS": top_probability if choice == "LOGS" else remainder,
            "METRICS": top_probability if choice == "METRICS" else remainder,
            "RUNBOOK": top_probability if choice == "RUNBOOK" else remainder,
            "ESCALATE": top_probability if choice == "ESCALATE" else remainder,
        }
    return SimpleNamespace(
        answers={
            "incident_tool": SimpleNamespace(
                type="choice",
                choice=choice,
                confidence=confidence,
                probabilities=probabilities,
            )
        }
    )


class FakeEngine:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def ask(self, state: str, questions: dict[str, object]) -> object:
        self.calls.append((state, questions))
        return self.response


class StrandsDeciderClientTests(unittest.TestCase):
    def test_confident_choice_is_returned_with_probability_evidence(self) -> None:
        engine = FakeEngine(make_response())
        client = StrandsDeciderClient(engine=engine)

        decision = client.decide("checkout is returning payment errors")

        self.assertEqual(decision.tool, IncidentTool.LOGS)
        self.assertEqual(decision.status, DecisionStatus.SELECTED)
        self.assertAlmostEqual(decision.probabilities["LOGS"], 0.9625)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(set(engine.calls[0][1]), {"incident_tool"})

    def test_low_confidence_selection_escalates_instead_of_running_a_tool(self) -> None:
        engine = FakeEngine(make_response(confidence=0.89))
        client = StrandsDeciderClient(engine=engine, minimum_confidence=0.9)

        decision = client.decide("unclear service outage")

        self.assertEqual(decision.tool, IncidentTool.ESCALATE)
        self.assertEqual(decision.status, DecisionStatus.LOW_CONFIDENCE)
        self.assertIn("below", decision.detail)

    def test_model_selected_escalation_is_preserved(self) -> None:
        engine = FakeEngine(
            make_response(
                choice="ESCALATE",
                confidence=0.9333,
                probabilities={
                    "LOGS": 0.02,
                    "METRICS": 0.02,
                    "RUNBOOK": 0.01,
                    "ESCALATE": 0.95,
                },
            )
        )

        decision = StrandsDeciderClient(engine=engine).decide("possible data exposure")

        self.assertEqual(decision.tool, IncidentTool.ESCALATE)
        self.assertEqual(decision.status, DecisionStatus.MODEL_ESCALATION)

    def test_inconsistent_choice_and_probabilities_fail_closed(self) -> None:
        engine = FakeEngine(
            make_response(
                choice="METRICS",
                confidence=0.95,
                probabilities={
                    "LOGS": 0.9625,
                    "METRICS": 0.0125,
                    "RUNBOOK": 0.0125,
                    "ESCALATE": 0.0125,
                },
            )
        )

        with self.assertLogs("src.decider_client", level="ERROR"):
            decision = StrandsDeciderClient(engine=engine).decide("unclear incident")

        self.assertEqual(decision.tool, IncidentTool.ESCALATE)
        self.assertEqual(decision.status, DecisionStatus.MODEL_ERROR)

    def test_malformed_response_fails_closed_with_visible_fallback(self) -> None:
        engine = FakeEngine(SimpleNamespace(answers={}))
        client = StrandsDeciderClient(engine=engine)

        with self.assertLogs("src.decider_client", level="ERROR"):
            decision = client.decide("service unavailable")

        self.assertEqual(decision.tool, IncidentTool.ESCALATE)
        self.assertEqual(decision.status, DecisionStatus.MODEL_ERROR)
        self.assertIn("ValueError", decision.detail)

    def test_engine_failure_fails_closed(self) -> None:
        class BrokenEngine:
            def ask(self, state: str, questions: dict[str, object]) -> object:
                raise OSError("model unavailable")

        with self.assertLogs("src.decider_client", level="ERROR"):
            decision = StrandsDeciderClient(engine=BrokenEngine()).decide("API outage")

        self.assertEqual(decision.status, DecisionStatus.MODEL_ERROR)
        self.assertEqual(decision.tool, IncidentTool.ESCALATE)
        self.assertIn("OSError", decision.detail)

    def test_engine_loader_is_explicitly_cpu_only(self) -> None:
        sentinel = object()
        with patch("strands_decider.infer.load_engine", return_value=sentinel) as loader:
            client = StrandsDeciderClient(checkpoint=DEFAULT_CHECKPOINT)

            self.assertIs(client._get_engine(), sentinel)

        loader.assert_called_once_with(DEFAULT_CHECKPOINT, device="cpu")

    def test_empty_incident_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            StrandsDeciderClient(engine=FakeEngine(make_response())).decide("  ")


if __name__ == "__main__":
    unittest.main()
