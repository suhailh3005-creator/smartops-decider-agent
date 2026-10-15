"""Run SmartOps locally with mock incident tools and optional Bedrock reporting."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from .agent import IncidentOutcome, SmartOpsAgent
from .bedrock_report import BedrockReportGenerator, ReportGenerator
from .decider_client import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MINIMUM_CONFIDENCE,
    DecisionProvider,
    StrandsDeciderClient,
)

DEFAULT_INCIDENT = (
    "I need the exact application error messages and timestamps for failed checkout "
    "requests. Which source should I query?"
)


def run_demo(
    incident: str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
    bedrock_enabled: bool = False,
    bedrock_model_id: str | None = None,
    region_name: str | None = None,
    decider: DecisionProvider | None = None,
    report_generator: ReportGenerator | None = None,
) -> IncidentOutcome:
    if bedrock_enabled and not bedrock_model_id:
        raise ValueError("set --bedrock-model-id or SMARTOPS_BEDROCK_MODEL_ID to use Bedrock")
    if bedrock_enabled and report_generator is None:
        report_generator = BedrockReportGenerator(
            bedrock_model_id or "", region_name=region_name
        )

    decision_provider = decider or StrandsDeciderClient(
        checkpoint=checkpoint,
        minimum_confidence=minimum_confidence,
    )
    return SmartOpsAgent(
        decision_provider, report_generator=report_generator
    ).investigate(incident)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use Strands Decider to select one mock incident tool."
    )
    parser.add_argument(
        "--incident",
        default=DEFAULT_INCIDENT,
        help="Incident description (the default is a clear log-query example).",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("SMARTOPS_DECIDER_CHECKPOINT", DEFAULT_CHECKPOINT),
        help="Local Decider checkpoint or Hugging Face checkpoint ID.",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=os.getenv(
            "SMARTOPS_MINIMUM_CONFIDENCE",
            str(DEFAULT_MINIMUM_CONFIDENCE),
        ),
        help=f"Confidence required to run a mock tool (default: {DEFAULT_MINIMUM_CONFIDENCE}).",
    )
    parser.add_argument(
        "--bedrock",
        action="store_true",
        help="Opt in to an AWS Bedrock report when a mock tool returns evidence.",
    )
    parser.add_argument(
        "--bedrock-model-id",
        default=os.getenv("SMARTOPS_BEDROCK_MODEL_ID"),
        help="Bedrock model ID (or set SMARTOPS_BEDROCK_MODEL_ID).",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        help="AWS region (or use the standard AWS SDK region configuration).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        outcome = run_demo(
            args.incident,
            checkpoint=args.checkpoint,
            minimum_confidence=args.minimum_confidence,
            bedrock_enabled=args.bedrock,
            bedrock_model_id=args.bedrock_model_id,
            region_name=args.region,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(outcome.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
