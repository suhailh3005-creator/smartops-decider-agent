"""Incident triage flow: choose one tool, then execute only that tool."""

from __future__ import annotations

from pydantic import BaseModel

from .bedrock_report import IncidentReport, ReportGenerator
from .decider_client import DecisionProvider, StrandsDeciderClient
from .tools import ToolExecution, execute_selected_tool


class IncidentOutcome(BaseModel):
    incident: str
    execution: ToolExecution
    report: IncidentReport | None = None


class SmartOpsAgent:
    def __init__(
        self,
        decider: DecisionProvider | None = None,
        report_generator: ReportGenerator | None = None,
    ) -> None:
        self.decider = decider or StrandsDeciderClient()
        self.report_generator = report_generator

    def investigate(self, incident: str) -> IncidentOutcome:
        if not incident.strip():
            raise ValueError("incident must not be empty")
        decision = self.decider.decide(incident)
        execution = execute_selected_tool(decision, incident)
        report = None
        if execution.result is not None and self.report_generator is not None:
            report = self.report_generator.generate(incident, execution.result)
        return IncidentOutcome(incident=incident, execution=execution, report=report)