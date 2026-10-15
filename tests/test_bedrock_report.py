from __future__ import annotations

import json
import unittest

from src.bedrock_report import BedrockReportGenerator, ReportResponseError
from src.tools import ToolResult, query_logs


def converse_response(text: str) -> dict[str, object]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        }
    }


class FakeConverseClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.response


class BedrockReportGeneratorTests(unittest.TestCase):
    def test_generates_typed_report_with_only_supplied_evidence(self) -> None:
        response_text = json.dumps(
            {
                "summary": {
                    "text": "The payment provider returned an HTTP 502.",
                    "evidence_ids": ["E1"],
                },
                "findings": [
                    {
                        "text": "A retry was scheduled after the provider timeout.",
                        "evidence_ids": ["E2"],
                    }
                ],
                "recommended_actions": [],
            }
        )
        client = FakeConverseClient(converse_response(response_text))
        generator = BedrockReportGenerator(
            "test-model", region_name="us-west-2", client=client
        )
        result = query_logs("checkout payment failures")

        report = generator.generate("checkout payment failures", result)

        self.assertEqual(report.tool, result.tool)
        self.assertEqual(report.incident, "checkout payment failures")
        self.assertEqual(report.summary.evidence_ids, ["E1"])
        self.assertEqual(report.evidence[0].observation, result.observations[0])
        request = client.calls[0]
        self.assertEqual(request["modelId"], "test-model")
        self.assertEqual(request["inferenceConfig"], {"temperature": 0.0, "maxTokens": 1200})
        prompt = request["messages"][0]["content"][0]["text"]
        self.assertEqual(json.loads(prompt)["evidence"][0]["id"], "E1")

    def test_rejects_claims_that_cite_unknown_evidence(self) -> None:
        response_text = json.dumps(
            {
                "summary": {
                    "text": "Payments are failing globally.",
                    "evidence_ids": ["E99"],
                },
                "findings": [],
                "recommended_actions": [],
            }
        )
        generator = BedrockReportGenerator(
            "test-model", client=FakeConverseClient(converse_response(response_text))
        )

        with self.assertRaisesRegex(ReportResponseError, "invalid or ungrounded"):
            generator.generate("payment incident", query_logs("payment incident"))

    def test_rejects_malformed_bedrock_response(self) -> None:
        generator = BedrockReportGenerator(
            "test-model", client=FakeConverseClient({"output": {}})
        )

        with self.assertRaisesRegex(ReportResponseError, "missing output message"):
            generator.generate("payment incident", query_logs("payment incident"))

    def test_requires_tool_evidence_before_calling_bedrock(self) -> None:
        client = FakeConverseClient(converse_response("{}"))
        generator = BedrockReportGenerator("test-model", client=client)
        empty_result = ToolResult(
            tool=query_logs("incident").tool,
            source="empty mock",
            query="incident",
            observations=[],
        )

        with self.assertRaisesRegex(ValueError, "no evidence"):
            generator.generate("incident", empty_result)

        self.assertEqual(client.calls, [])

    def test_does_not_construct_aws_client_until_generation(self) -> None:
        generator = BedrockReportGenerator("test-model")

        self.assertIsNone(generator._client)


if __name__ == "__main__":
    unittest.main()
