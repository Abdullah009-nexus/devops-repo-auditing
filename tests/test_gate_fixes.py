import contextlib
import email.message
import http.client
import io
import json
import os
import tempfile
import unittest
import urllib.request
import urllib.response
from pathlib import Path
from unittest.mock import patch

import agent
from auditor import llm_client
from auditor.findings import (
    FINISH_AUDIT,
    REPORT_FINDING,
    FindingStore,
    build_finding_tools,
)
from auditor.llm_client import LLMClient, LLMError
from auditor.loop import AuditAgent
from auditor.prompts import load_prompt
from auditor.settings import load_settings
from auditor.trace import Tracer
from tests.support import default_settings
from tests.test_llm_client import ENVIRON, GOOD, response
from tests.test_loop import (
    FINDING,
    TOOL_CALL,
    FakeClient,
    LoopTestCase,
    call,
    finish,
    make_repository,
    reply,
    tool_replies,
    with_limits,
)


class RedirectTests(unittest.TestCase):
    def make_client(self):
        return LLMClient(default_settings().llm, ENVIRON, sleep=lambda seconds: None)

    def test_a_redirect_is_refused_not_followed(self):
        class Redirecting(urllib.request.HTTPSHandler):
            def https_open(self, req):
                headers = email.message.Message()
                headers["Location"] = "https://elsewhere.example.com/v1"
                reply_ = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, 302)
                reply_.msg = "Found"
                return reply_

        opener = urllib.request.build_opener(llm_client._NoRedirect, Redirecting)
        with patch.object(llm_client, "_opener", opener):
            with self.assertRaisesRegex(LLMError, "redirected the request; refusing to resend"):
                self.make_client().chat([], [])

    def test_the_provider_opener_carries_the_refusing_handler(self):
        self.assertTrue(any(isinstance(h, llm_client._NoRedirect) for h in llm_client._opener.handlers))


class TransientFailureTests(unittest.TestCase):
    def test_connection_reset_and_incomplete_read_are_retried(self):
        sleeps = []
        client = LLMClient(default_settings().llm, ENVIRON, sleep=sleeps.append)
        failures = [ConnectionResetError(), http.client.IncompleteRead(b"par"), response()]
        with patch("auditor.llm_client._opener.open", side_effect=failures) as opener:
            message = client.chat([], [])
        self.assertEqual(message["content"], "ok")
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(len(sleeps), 2)

    def test_persistent_reset_ends_with_a_fixed_message(self):
        settings = default_settings().llm
        client = LLMClient(settings, ENVIRON, sleep=lambda seconds: None)
        with patch("auditor.llm_client._opener.open", side_effect=ConnectionResetError("secret")):
            with self.assertRaisesRegex(LLMError, "connection failed") as caught:
                client.chat([], [])
        self.assertNotIn("secret", str(caught.exception))


class UsageTests(unittest.TestCase):
    def test_last_usage_holds_the_providers_usage_dict(self):
        body = json.dumps(
            {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 7}}
        ).encode()
        client = LLMClient(default_settings().llm, ENVIRON)
        with patch("auditor.llm_client._opener.open", return_value=response(body)):
            client.chat([], [])
            self.assertEqual(client.last_usage, {"total_tokens": 7})
        with patch("auditor.llm_client._opener.open", return_value=response(GOOD)):
            client.chat([], [])
        self.assertIsNone(client.last_usage)

    def test_loop_records_usage_in_the_trace(self):
        settings = default_settings()
        client = FakeClient(reply(finish()))
        client.last_usage = {"total_tokens": 7}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            AuditAgent(make_repository(settings), settings, client, Tracer(path, 500)).run()
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        usage = [e["data"]["usage"] for e in events if e["kind"] == "llm_response"]
        self.assertEqual(usage, [{"total_tokens": 7}])


class FinalizeClockTests(LoopTestCase):
    def test_finalize_stops_when_its_window_is_spent_and_says_so(self):
        settings = with_limits(self.settings, max_wall_seconds=100, finalize_wall_seconds=100, finalize_turns=3)
        now = [0.0]

        def slow(message):
            def respond():
                now[0] += 1000.0
                return message

            return respond

        with patch(TOOL_CALL, return_value={"files": []}):
            self.run_loop(
                slow(reply(call("list_files", {}, "a"))),
                slow(reply(call("list_files", {}, "b"))),
                settings=settings,
                clock=lambda: now[0],
            )
        self.assertEqual(len(self.client.requests), 2)

    def test_a_budget_stop_traces_a_finalize_event(self):
        settings = with_limits(self.settings, max_turns=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            client = FakeClient(reply(call("list_files", {}, "a")), reply(finish()))
            with patch(TOOL_CALL, return_value={"files": []}):
                AuditAgent(self.repository, settings, client, Tracer(path, 500)).run()
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        finalize = [e for e in events if e["kind"] == "finalize"]
        self.assertEqual(finalize[0]["data"]["reason"], "turn_limit")


class FindingLimitTests(LoopTestCase):
    def test_over_long_text_field_is_rejected_with_a_shorten_hint(self):
        limit = self.settings.agent.max_finding_field_chars
        report = self.run_loop(
            reply(call(REPORT_FINDING, {**FINDING, "evidence": "x" * (limit + 1)}, "long")),
            reply(finish()),
        )
        verdict = tool_replies(self.client.requests[1][0])[0]
        self.assertFalse(verdict["accepted"])
        self.assertIn("shorten", verdict["reason"])
        self.assertEqual(report["violations"], [])

    def test_exact_environment_secrets_are_masked_in_finding_text(self):
        store = FindingStore(
            self.repository, self.settings.agent, environ={"GITHUB_TOKEN": "opaque-cred-77"}
        )
        decision = store.add({**FINDING, "description": "uses opaque-cred-77 directly"})
        self.assertTrue(decision.accepted)
        self.assertNotIn("opaque-cred-77", decision.finding.description)
        self.assertIn("[REDACTED]", decision.finding.description)


class MessageHygieneTests(LoopTestCase):
    def test_resent_assistant_message_keeps_only_standard_fields(self):
        noisy = reply({**call("list_files", {}, "a"), "extra_content": {"sig": "x"}})
        noisy["reasoning"] = "private notes"
        with patch(TOOL_CALL, return_value={"files": []}):
            self.run_loop(noisy, reply(finish()))
        resent = self.client.requests[1][0][2]
        self.assertEqual(set(resent), {"role", "content", "tool_calls"})
        self.assertEqual(set(resent["tool_calls"][0]), {"id", "type", "function"})

    def test_text_reply_limit_comes_from_settings(self):
        settings = with_limits(self.settings, max_text_replies=3)
        self.run_loop(
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "assistant", "content": "c"},
            settings=settings,
        )
        self.assertEqual(len(self.client.requests), 3)

    def test_omitted_file_and_line_are_an_absence_finding(self):
        absence = {
            key: value for key, value in FINDING.items() if key not in ("file", "line")
        }
        report = self.run_loop(
            reply(call("list_files", {}, "l")),
            reply(call(REPORT_FINDING, absence, "a")),
            reply(finish()),
        )
        self.assertEqual(len(report["violations"]), 1)
        self.assertIsNone(report["violations"][0]["file"])
        self.assertIsNone(report["violations"][0]["line"])


class SchemaAndPromptTests(unittest.TestCase):
    def test_report_finding_schema_has_no_type_arrays_and_optional_location(self):
        schema = build_finding_tools()[0]["function"]["parameters"]
        for name, spec in schema["properties"].items():
            self.assertIsInstance(spec["type"], str, name)
        self.assertEqual(schema["properties"]["file"]["type"], "string")
        self.assertEqual(schema["properties"]["line"]["type"], "integer")
        self.assertNotIn("file", schema["required"])
        self.assertNotIn("line", schema["required"])
        self.assertEqual(FINISH_AUDIT, build_finding_tools()[1]["function"]["name"])

    def test_system_prompt_demands_verbatim_audit_rule_findings(self):
        text = load_prompt(
            "system", repository_url="u", rule_catalog="", max_tool_calls=1, max_turns=1
        )
        self.assertIn("exactly as the tool returned them", text)
        self.assertIn("multi-line evidence", text)


class TruncationTests(LoopTestCase):
    def test_truncated_result_never_exceeds_the_limit(self):
        limit = self.settings.agent.max_tool_output_chars
        for blob in ("x" * limit * 2, '"\\' * limit):
            with patch(TOOL_CALL, return_value={"blob": blob}):
                self.run_loop(reply(call("list_files", {}, "big")), reply(finish()))
            content = self.client.requests[1][0][-1]["content"]
            self.assertLessEqual(len(content), limit)
            self.assertTrue(json.loads(content)["result_truncated"])


class OutputPathTests(unittest.TestCase):
    ARGV = ["--repo", "https://github.com/example/project"]

    def run_cli(self, output):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m"}),
            patch.object(agent, "GitHubRepository") as repository,
            patch.object(agent, "LLMClient") as client,
            patch("agent.sys.stderr", new_callable=io.StringIO) as stderr,
            patch("agent.sys.stdout", new_callable=io.StringIO),
            contextlib.chdir(self.directory),
        ):
            code = agent.main([*self.ARGV, "--output", output])
        return code, stderr.getvalue(), repository, client

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_unusable_parent_fails_before_the_repository_or_model_is_touched(self):
        (self.directory / "file").write_text("x", encoding="utf-8")
        code, stderr, repository, client = self.run_cli("file/report.json")
        self.assertEqual(code, 1)
        self.assertTrue(stderr.startswith("Error: "))
        repository.assert_not_called()

    def test_output_that_is_a_directory_fails_early(self):
        (self.directory / "out").mkdir()
        code, stderr, repository, _ = self.run_cli("out")
        self.assertEqual(code, 1)
        self.assertIn("directory", stderr)
        repository.assert_not_called()

    def test_missing_parent_directory_is_created_before_the_audit(self):
        with patch.object(agent, "run_agent", return_value={"repository": "r", "violations": []}):
            code, _, repository, _ = self.run_cli("new/dir/report.json")
        self.assertTrue((self.directory / "new" / "dir").is_dir())
        repository.assert_called_once()


class SettingsTests(unittest.TestCase):
    def test_new_agent_keys_have_defaults(self):
        agent_settings = load_settings().agent
        self.assertEqual(agent_settings.max_finding_field_chars, 2000)
        self.assertEqual(agent_settings.max_text_replies, 2)


if __name__ == "__main__":
    unittest.main()
