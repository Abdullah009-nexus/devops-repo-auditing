import io
import json
import os
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch

import agent
from tools import GitHubRepository, RepositoryAddress


def completion_message(message):
    return {"choices": [{"message": message}]}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.repository = GitHubRepository.__new__(GitHubRepository)
        self.repository.address = RepositoryAddress.parse(
            "https://github.com/example/project"
        )
        self.repository.files = {"config.py": {"type": "blob", "size": 10}}
        self.repository._read_cache = {"config.py": "setting = True\n"}
        self.repository._inspected_paths = {"config.py"}

    def test_agent_performs_follow_up_tool_calls_before_reporting(self):
        first = completion_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-search",
                        "type": "function",
                        "function": {
                            "name": "search_code",
                            "arguments": json.dumps({"pattern": "debug"}),
                        },
                    }
                ],
            }
        )
        second = completion_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-read",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "config.py"}),
                        },
                    }
                ],
            }
        )
        final_report = {
            "repository": self.repository.address.url,
            "violations": [],
        }
        final = completion_message(
            {"role": "assistant", "content": json.dumps(final_report)}
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}),
            patch.object(agent, "_chat_completion", side_effect=[first, second, final]) as chat,
            patch("agent.RepositoryTools.call", side_effect=[
                {"pattern": "debug", "matches": []},
                {"path": "config.py", "content": "1: setting = True"},
            ]) as tool_call,
        ):
            result = agent.run_agent(self.repository)
        self.assertEqual(result["violations"], [])
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(tool_call.call_count, 2)
        final_messages = chat.call_args_list[-1].args[0]
        self.assertEqual(final_messages[-1]["role"], "tool")
        self.assertIn('"matches": []', final_messages[-3]["content"])

    def test_report_rejects_uninspected_file_evidence(self):
        report = {
            "repository": self.repository.address.url,
            "violations": [
                {
                    "rule_id": "CUSTOM-001",
                    "category": "security",
                    "severity": "high",
                    "file": "unknown.py",
                    "line": 1,
                    "description": "Unsupported issue",
                    "evidence": "Not observed",
                    "recommendation": "Investigate",
                }
            ],
        }
        with self.assertRaises(agent.AgentError):
            agent.validate_report(json.dumps(report), self.repository)

    def test_report_redacts_secrets_and_deduplicates(self):
        finding = {
            "rule_id": "CUSTOM-001",
            "category": "security",
            "severity": "high",
            "file": "config.py",
            "line": 1,
            "description": "API token is exposed",
            "evidence": "TOKEN = supersecretvalue",
            "recommendation": "Use a secret manager",
        }
        report = {
            "repository": self.repository.address.url,
            "violations": [finding, finding],
        }
        validated = agent.validate_report(json.dumps(report), self.repository)
        self.assertEqual(len(validated["violations"]), 1)
        self.assertNotIn("supersecretvalue", json.dumps(validated))

    def test_report_requires_exact_evidence_for_known_audit_rules(self):
        self.repository._read_cache["config.py"] = (
            'API_TOKEN = "local_test_token_value_12345"\n'
        )
        report = {
            "repository": self.repository.address.url,
            "violations": [
                {
                    "rule_id": "SEC-001",
                    "category": "security",
                    "severity": "high",
                    "file": "config.py",
                    "line": 1,
                    "description": "Credential hardcoded in source.",
                    "evidence": "API_TOKEN = [REDACTED]",
                    "recommendation": "Use environment-based secret management.",
                }
            ],
        }
        result = agent.validate_report(json.dumps(report), self.repository)
        self.assertEqual(result["violations"][0]["rule_id"], "SEC-001")
        report["violations"][0]["evidence"] = "This file has a security problem."
        with self.assertRaisesRegex(agent.AgentError, "deterministic audit rule"):
            agent.validate_report(json.dumps(report), self.repository)

    def test_rejects_invalid_report_json(self):
        with self.assertRaises(agent.AgentError):
            agent.validate_report("not json", self.repository)

    def test_tool_result_returns_validation_errors_to_model(self):
        result = agent._tool_result(
            agent.RepositoryTools(self.repository),
            "read_file",
            json.dumps({"path": "config.py", "unexpected": True}),
        )
        self.assertIn("unsupported arguments", json.loads(result)["error"])

    def test_agent_rejects_empty_provider_choices(self):
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"},
            ),
            patch.object(agent, "_chat_completion", return_value={"choices": []}),
            self.assertRaisesRegex(agent.AgentError, "completion choice"),
        ):
            agent.run_agent(self.repository)

    def test_agent_enforces_tool_call_limit_before_dispatch(self):
        message = completion_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                    for index in range(agent.MAX_TOOL_CALLS + 1)
                ],
            }
        )
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"},
            ),
            patch.object(agent, "_chat_completion", return_value=message),
            patch("agent.RepositoryTools.call") as tool_call,
            self.assertRaisesRegex(agent.AgentError, "inspection limit"),
        ):
            agent.run_agent(self.repository)
        tool_call.assert_not_called()

    def test_agent_enforces_reasoning_turn_limit(self):
        tool_response = completion_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-list",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            }
        )
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"},
            ),
            patch.object(
                agent, "_chat_completion", return_value=tool_response
            ) as chat,
            patch("agent.RepositoryTools.call", return_value={"files": []}),
            patch.object(agent, "MAX_AGENT_TURNS", 2),
            self.assertRaisesRegex(agent.AgentError, "reasoning limit"),
        ):
            agent.run_agent(self.repository)
        self.assertEqual(chat.call_count, 2)

    @patch("agent.urllib.request.urlopen")
    def test_provider_http_failure_is_explicit(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError(
            "https://api.example.com/chat/completions",
            429,
            "rate limited",
            {},
            io.BytesIO(b"provider error details must not be exposed"),
        )
        try:
            with self.assertRaisesRegex(agent.AgentError, "HTTP 429"):
                agent._chat_completion(
                    [],
                    api_key="test-key",
                    model="test-model",
                    base_url="https://api.example.com/v1",
                )
        finally:
            urlopen.side_effect.close()

    @patch("agent.urllib.request.urlopen")
    def test_provider_timeout_is_explicit(self, urlopen):
        urlopen.side_effect = TimeoutError()
        with self.assertRaisesRegex(agent.AgentError, "timed out"):
            agent._chat_completion(
                [],
                api_key="test-key",
                model="test-model",
                base_url="https://api.example.com/v1",
            )

    @patch("agent.urllib.request.urlopen")
    def test_invalid_provider_json_is_explicit(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = b"not json"
        with self.assertRaisesRegex(agent.AgentError, "invalid JSON"):
            agent._chat_completion(
                [],
                api_key="test-key",
                model="test-model",
                base_url="https://api.example.com/v1",
            )

    def test_report_writer_creates_valid_json(self):
        report = {"repository": self.repository.address.url, "violations": []}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "report.json"
            agent._write_report(str(output), report)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_cli_returns_nonzero_for_repository_failure(self):
        with (
            patch.object(
                agent.RepositoryAddress,
                "parse",
                side_effect=agent.RepositoryError("repository unavailable"),
            ),
            patch("agent.sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            result = agent.main(
                ["--repo", "https://github.com/example/project", "--output", "report.json"]
            )
        self.assertEqual(result, 1)
        self.assertIn("repository unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
