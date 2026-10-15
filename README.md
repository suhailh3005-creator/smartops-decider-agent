# SmartOps Decider Agent

SmartOps uses the **Strands Decider 2B** model to choose one next incident action:
`LOGS`, `METRICS`, `RUNBOOK`, or `ESCALATE`. A confident selection runs exactly
one deterministic mock tool. If the choice is uncertain or inference fails,
SmartOps escalates and runs no tool. When explicitly enabled, Amazon Bedrock
generates a typed incident report from that tool's evidence.

## Local setup

This project was developed with the repository's existing `.venv`. Activate it
and run the focused test suite:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The direct dependencies are listed in `requirements.txt`. Installing
`strands-decider` also installs its PyTorch/Transformers dependencies; use the
existing environment when it is already provisioned.

The default checkpoint is
`StrandsAgents/strands-decider-2B-hobson-v19`. The first uncached inference
downloads the checkpoint and its Qwen base weights (about 4.5 GB) from Hugging
Face. Point `SMARTOPS_DECIDER_CHECKPOINT` at a local checkpoint directory to
avoid using a Hub identifier. The model loader is always given `device="cpu"`;
the demo does not intentionally enable or select a GPU.

## Run the demo

```bash
python -m src.demo
```

The default incident asks for exact application error messages, so the model
should select the mock LOGS tool. The result is JSON containing the typed
decision, confidence, option probabilities, and the selected tool's observations.
Try another incident with:

```bash
python -m src.demo --incident "The alert is about p95 latency; I need its metric time series."
```

`--minimum-confidence` (default `0.9`) controls when the selected mock tool is
allowed to run. The threshold follows the Decider's documented conservative
confidence convention. Below it, the decision changes to `ESCALATE`; model or
response errors also produce an explicit `ESCALATE` result. Do not lower this
threshold for production without evaluating it on representative incidents.

## Bedrock incident reports

Bedrock is **not called by the default demo or test suite**. To opt in to a live
report request, configure the AWS SDK credentials and region in the normal AWS
way, then provide a Bedrock model ID:

```bash
export AWS_REGION=us-west-2
export SMARTOPS_BEDROCK_MODEL_ID=your-bedrock-model-id
python -m src.demo --bedrock
```

Alternatively, pass `--region` and `--bedrock-model-id`. The request is made only
when the Decider selects a tool above the confidence threshold and that mock tool
returns evidence. On escalation there is no tool evidence, so no Bedrock request
is made. Bedrock requires `bedrock:InvokeModel` permission. The incident text and
selected tool observations are sent to the chosen Bedrock model; review that data
flow and obtain any required approval before using real incident content.

The report is typed and each summary, finding, or recommendation must cite
evidence IDs supplied by the mock tool. Unknown evidence references, malformed
responses, and service errors are surfaced rather than converted into a
success-shaped report. Tests inject a fake Converse client and do not contact
AWS.

## Project layout

- `src/decider_client.py` — typed Strands Decider choice, confidence checks, and safe escalation.
- `src/tools.py` — deterministic LOGS, METRICS, and RUNBOOK mock tools.
- `src/agent.py` — selected-tool execution and optional report generation.
- `src/bedrock_report.py` — typed, evidence-referenced report generation through Bedrock Converse.
- `src/demo.py` — local CLI demo; Bedrock use is opt-in.
- `tests/` — offline unit tests for decisions, tool execution, reports, and demo gating.

The Bedrock adapter uses the AWS `bedrock-runtime` Converse API documented in
the [AWS API reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html).
