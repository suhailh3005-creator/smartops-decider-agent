# Incident Tool Selection

SmartOps demonstrates confidence-aware tool selection with the Strands Decider 2B
model. The decision is typed as `LOGS`, `METRICS`, `RUNBOOK`, or `ESCALATE`;
only a confident non-escalation choice executes its corresponding mock tool.
Low confidence, malformed output, and inference errors fail closed to escalation.

The implementation is in `src/decider_client.py`, `src/tools.py`, and
`src/agent.py`. The optional Bedrock report adapter is in `src/bedrock_report.py`.
See the repository [README](../../README.md) for setup, the local demo, and the
explicit opt-in required for live Bedrock inference.
