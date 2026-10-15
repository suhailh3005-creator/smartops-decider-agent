"""Typed interface to the Strands Decider model."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"
DEFAULT_MINIMUM_CONFIDENCE = 0.9
QUESTION_NAME = "incident_tool"


class IncidentTool(str, Enum):
    LOGS = "LOGS"
    METRICS = "METRICS"
    RUNBOOK = "RUNBOOK"
    ESCALATE = "ESCALATE"


class DecisionStatus(str, Enum):
    SELECTED = "selected"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_ESCALATION = "model_escalation"
    MODEL_ERROR = "model_error"


class ToolDecision(BaseModel):
    tool: IncidentTool
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    status: DecisionStatus
    detail: str

    @model_validator(mode="after")
    def validate_probabilities(self) -> ToolDecision:
        if any(
            not math.isfinite(probability) or not 0.0 <= probability <= 1.0
            for probability in self.probabilities.values()
        ):
            raise ValueError("decision probabilities must be finite values in [0, 1]")
        if self.probabilities and not math.isclose(
            sum(self.probabilities.values()), 1.0, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise ValueError("decision probabilities must sum to 1")
        if self.probabilities:
            expected_confidence = (
                len(self.probabilities) * max(self.probabilities.values()) - 1.0
            ) / (len(self.probabilities) - 1)
            if not math.isclose(
                self.confidence, expected_confidence, rel_tol=0.0, abs_tol=1e-3
            ):
                raise ValueError("decision confidence does not match its probabilities")
        if (
            self.status is DecisionStatus.SELECTED
            and self.tool is IncidentTool.ESCALATE
        ):
            raise ValueError("a selected tool decision cannot be ESCALATE")
        if (
            self.status is not DecisionStatus.SELECTED
            and self.tool is not IncidentTool.ESCALATE
        ):
            raise ValueError("non-selected decisions must route to ESCALATE")
        return self


class DecisionProvider(Protocol):
    def decide(self, incident: str) -> ToolDecision: ...


class DeciderEngine(Protocol):
    def ask(self, state: str, questions: dict[str, Any]) -> Any: ...


class StrandsDeciderClient:
    """Ask Strands Decider to select one incident tool.

    A Hub checkpoint identifier is accepted by the installed loader. Loading it
    can download model weights; callers can instead pass an already-cached local
    checkpoint path. The engine is always constructed with the CPU device.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
        engine: DeciderEngine | None = None,
    ) -> None:
        if not math.isfinite(minimum_confidence) or not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be finite and in [0, 1]")
        self.checkpoint = checkpoint
        self.minimum_confidence = minimum_confidence
        self._engine = engine

    def _get_engine(self) -> DeciderEngine:
        if self._engine is None:
            from strands_decider.infer import load_engine

            self._engine = load_engine(self.checkpoint, device="cpu")
        return self._engine

    @staticmethod
    def _choice_question() -> Any:
        from strands_decider.schema import ChoiceQuestion

        return ChoiceQuestion(
            instructions=(
                "Choose the single incident investigation action that should be "
                "performed next. Select ESCALATE when the incident is unclear, "
                "high-risk, or needs human judgment."
            ),
            criteria={
                IncidentTool.LOGS.value: "Inspect relevant application and service logs.",
                IncidentTool.METRICS.value: "Inspect service health and performance metrics.",
                IncidentTool.RUNBOOK.value: "Consult the operational runbook for a known procedure.",
                IncidentTool.ESCALATE.value: "Hand the incident to a human responder.",
            },
        )

    @staticmethod
    def _parse_answer(response: Any) -> tuple[IncidentTool, float, dict[str, float]]:
        answers = getattr(response, "answers", None)
        if not isinstance(answers, Mapping) or QUESTION_NAME not in answers:
            raise ValueError("Decider response is missing the incident tool answer")

        answer = answers[QUESTION_NAME]
        if getattr(answer, "type", None) != "choice":
            raise ValueError("Decider returned a non-choice answer for tool selection")

        try:
            selected_tool = IncidentTool(answer.choice)
        except (AttributeError, ValueError) as exc:
            raise ValueError("Decider selected an unknown incident tool") from exc

        raw_probabilities = getattr(answer, "probabilities", None)
        if not isinstance(raw_probabilities, Mapping):
            raise ValueError("Decider response is missing choice probabilities")
        if set(raw_probabilities) != {tool.value for tool in IncidentTool}:
            raise ValueError("Decider response must include probabilities for every tool")

        probabilities: dict[str, float] = {}
        for label, value in raw_probabilities.items():
            probability = float(value)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("Decider returned an invalid choice probability")
            probabilities[str(label)] = probability
        if not math.isclose(
            sum(probabilities.values()), 1.0, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise ValueError("Decider choice probabilities do not sum to 1")

        confidence = float(answer.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Decider returned an invalid confidence")
        most_likely = max(probabilities, key=probabilities.__getitem__)
        if selected_tool.value != most_likely:
            raise ValueError("Decider choice does not match its highest probability")
        expected_confidence = (
            len(probabilities) * max(probabilities.values()) - 1.0
        ) / (len(probabilities) - 1)
        if not math.isclose(
            confidence, expected_confidence, rel_tol=0.0, abs_tol=1e-3
        ):
            raise ValueError("Decider confidence does not match its probabilities")
        return selected_tool, confidence, probabilities

    def decide(self, incident: str) -> ToolDecision:
        if not incident.strip():
            raise ValueError("incident must not be empty")

        try:
            response = self._get_engine().ask(
                incident, {QUESTION_NAME: self._choice_question()}
            )
            tool, confidence, probabilities = self._parse_answer(response)
        except Exception as exc:
            logger.exception("Strands Decider inference failed; escalating incident")
            return ToolDecision(
                tool=IncidentTool.ESCALATE,
                confidence=0.0,
                status=DecisionStatus.MODEL_ERROR,
                detail=(
                    "Decider inference failed "
                    f"({type(exc).__name__}); incident routed for human escalation."
                ),
            )

        if tool is IncidentTool.ESCALATE:
            return ToolDecision(
                tool=tool,
                confidence=confidence,
                probabilities=probabilities,
                status=DecisionStatus.MODEL_ESCALATION,
                detail="The Decider selected human escalation.",
            )
        if confidence < self.minimum_confidence:
            return ToolDecision(
                tool=IncidentTool.ESCALATE,
                confidence=confidence,
                probabilities=probabilities,
                status=DecisionStatus.LOW_CONFIDENCE,
                detail=(
                    f"Decider confidence {confidence:.3f} is below the "
                    f"{self.minimum_confidence:.3f} action threshold."
                ),
            )
        return ToolDecision(
            tool=tool,
            confidence=confidence,
            probabilities=probabilities,
            status=DecisionStatus.SELECTED,
            detail=f"The Decider selected {tool.value} above the confidence threshold.",
        )
