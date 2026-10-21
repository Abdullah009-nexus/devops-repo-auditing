import json
import os
import unittest
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
                    "rule_id": "SEC-001",
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
            "rule_id": "SEC-001",
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

    def test_rejects_invalid_report_json(self):
        with self.assertRaises(agent.AgentError):
            agent.validate_report("not json", self.repository)


if __name__ == "__main__":
    unittest.main()
