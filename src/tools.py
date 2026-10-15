"""Deterministic mock incident tools used by the local SmartOps demo."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from .decider_client import IncidentTool, ToolDecision


class ToolResult(BaseModel):
    tool: IncidentTool
    source: str
    query: str
    observations: list[str]


class ToolExecution(BaseModel):
    decision: ToolDecision
    result: ToolResult | None


def query_logs(incident: str) -> ToolResult:
    return ToolResult(
        tool=IncidentTool.LOGS,
        source="mock application logs",
        query=incident,
        observations=[
            "14:03:11 ERROR checkout-api: payment provider returned HTTP 502",
            "14:03:15 WARN checkout-api: retry scheduled after provider timeout",
        ],
    )


def query_metrics(incident: str) -> ToolResult:
    return ToolResult(
        tool=IncidentTool.METRICS,
        source="mock service metrics",
        query=incident,
        observations=[
            "checkout-api 5xx rate: 18% over the last 5 minutes (baseline: 0.4%)",
            "payment-provider request p95 latency: 4.2 seconds (baseline: 0.6 seconds)",
        ],
    )


def query_runbook(incident: str) -> ToolResult:
    return ToolResult(
        tool=IncidentTool.RUNBOOK,
        source="mock operational runbook",
        query=incident,
        observations=[
            "RB-PAY-04: For payment-provider 5xx errors, verify provider health before retrying.",
            "Do not replay failed payment requests until idempotency status is confirmed.",
        ],
    )


TOOL_HANDLERS: dict[IncidentTool, Callable[[str], ToolResult]] = {
    IncidentTool.LOGS: query_logs,
    IncidentTool.METRICS: query_metrics,
    IncidentTool.RUNBOOK: query_runbook,
}


def execute_selected_tool(decision: ToolDecision, incident: str) -> ToolExecution:
    """Execute only a confident tool selection; escalation executes no mock tool."""
    if decision.tool is IncidentTool.ESCALATE:
        return ToolExecution(decision=decision, result=None)

    result = TOOL_HANDLERS[IncidentTool(decision.tool)](incident)
    return ToolExecution(decision=decision, result=result)