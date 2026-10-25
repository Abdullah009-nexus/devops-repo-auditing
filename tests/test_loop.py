import copy
import dataclasses
import json
import unittest
from unittest.mock import patch

from auditor.findings import FINISH_AUDIT, REPORT_FINDING
from auditor.llm_client import LLMError
from auditor.loop import AgentError, AuditAgent
from auditor.report import validate_report_shape
from auditor.repository import GitHubApiSource, RepositoryMetadata
from auditor.schema import REQUIRED_FIELDS
from auditor.trace import Tracer
from tests.support import default_settings
from tools import GitHubRepository, RepositoryAddress


FINDING = {
    "rule_id": "SEC-004",
    "category": "security",
    "severity": "medium",
    "file": "config.py",
    "line": 1,
    "description": "Debug flag enabled",
    "evidence": "setting = True",
    "recommendation": "Disable it",
    "quote": "setting = True",
}

TOOL_CALL = "auditor.loop.RepositoryTools.call"


def make_repository(settings, files=None):
    address = RepositoryAddress.parse("https://github.com/example/project", settings.github)
    tree = {"tree": [{"type": "blob", "path": "config.py", "size": 10}]}
    with patch("auditor.repository._request_json", return_value=tree):
        source = GitHubApiSource(address, RepositoryMetadata("main", 1), settings)
    repository = GitHubRepository(address, settings, source=source)
    repository._read_cache = files or {"config.py": "setting = True\n"}
    repository._inspected_paths = set(repository._read_cache)
    return repository


def with_limits(settings, **changes):
    return dataclasses.replace(settings, agent=dataclasses.replace(settings.agent, **changes))


def call(name, arguments, call_id):
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def reply(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def text(content):
    return {"role": "assistant", "content": content}


def finish(call_id="finish"):
    return call(FINISH_AUDIT, {"summary": "done"}, call_id)


class FakeClient:
    """Plays back scripted assistant messages and records what each request carried."""

    model = "fake-model"
    last_usage = None

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []

    def chat(self, messages, tools, tool_choice="auto"):
        self.requests.append((copy.deepcopy(messages), [t["function"]["name"] for t in tools]))
        if not self._responses:
            raise AssertionError("the model was called more often than the script allows")
        response = self._responses.pop(0)
        if callable(response):
            response = response()
        if isinstance(response, Exception):
            raise response
        return response


def tool_replies(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


class LoopTestCase(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings()
        self.repository = make_repository(self.settings)

    def run_loop(self, *responses, settings=None, clock=None):
        self.client = FakeClient(*responses)
        kwargs = {"clock": clock} if clock else {}
        agent = AuditAgent(
            self.repository, settings or self.settings, self.client, Tracer(None, 500), **kwargs
        )
        report = agent.run()
        validate_report_shape(report)
        return report

    def assertPaired(self, messages):
        asked = [c["id"] for m in messages for c in m.get("tool_calls") or []]
        answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        self.assertEqual(asked, answered)


class InvestigationTests(LoopTestCase):
    def test_follow_up_tool_calls_precede_the_report(self):
        settings = dataclasses.replace(
            self.settings,
            context=dataclasses.replace(self.settings.context, keep_recent_tool_results=1),
        )
        with patch(
            TOOL_CALL,
            side_effect=[
                {"pattern": "debug", "matches": [], "padding": "x" * 300},
                {"path": "config.py", "content": "1: setting = True"},
            ],
        ) as tools:
            report = self.run_loop(
                reply(call("search_code", {"pattern": "debug"}, "c1")),
                reply(call("read_file", {"path": "config.py"}, "c2")),
                reply(call(REPORT_FINDING, FINDING, "c3")),
                reply(finish()),
                settings=settings,
            )
        self.assertEqual(len(report["violations"]), 1)
        self.assertEqual(tools.call_count, 2)
        self.assertEqual(len(self.client.requests), 4)
        first_offer = self.client.requests[0][1]
        self.assertIn("list_files", first_offer)
        self.assertIn(REPORT_FINDING, first_offer)
        self.assertIn(FINISH_AUDIT, first_offer)
        third = self.client.requests[2][0]
        self.assertEqual(third[-1]["role"], "tool")
        self.assertEqual(third[-1]["tool_call_id"], "c2")
        for messages, _ in self.client.requests:
            self.assertPaired(messages)
        # With one result kept in full, the search result is already a stub.
        self.assertTrue(json.loads(third[3]["content"])["compacted"])

    def test_parallel_calls_are_answered_in_order_with_matching_ids(self):
        with patch(TOOL_CALL, return_value={"files": ["config.py"]}):
            self.run_loop(
                reply(
                    call("list_files", {}, "first"),
                    call(REPORT_FINDING, FINDING, "second"),
                    call("list_files", {}, "third"),
                ),
                reply(finish()),
            )
        messages = self.client.requests[1][0]
        replies = [m for m in messages if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in replies], ["first", "second", "third"])
        self.assertTrue(json.loads(replies[1]["content"])["accepted"])

    def test_unknown_tool_and_bad_arguments_go_back_as_errors(self):
        report = self.run_loop(
            reply(
                call("rm_rf", {}, "a"),
                call("read_file", "{not json", "b"),
                call("read_file", "[1]", "c"),
                call("read_file", {"path": "config.py", "unexpected": True}, "d"),
            ),
            reply(finish()),
        )
        replies = tool_replies(self.client.requests[1][0])
        self.assertEqual(len(replies), 4)
        self.assertIn("Unsupported inspection tool", replies[0]["error"])
        self.assertIn("not valid JSON", replies[1]["error"])
        self.assertIn("must be a JSON object", replies[2]["error"])
        self.assertIn("unsupported arguments", replies[3]["error"])
        self.assertEqual(report["violations"], [])

    def test_oversized_tool_output_is_cut_to_the_limit(self):
        limit = self.settings.agent.max_tool_output_chars
        with patch(TOOL_CALL, return_value={"blob": "x" * limit * 2}):
            self.run_loop(reply(call("list_files", {}, "big")), reply(finish()))
        content = self.client.requests[1][0][-1]["content"]
        self.assertLessEqual(len(content), limit)
        self.assertTrue(json.loads(content)["result_truncated"])

    def test_tool_call_without_an_id_is_dropped_and_the_model_is_nudged(self):
        broken = {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "function": {"name": "list_files", "arguments": "{}"}}
        ]}
        report = self.run_loop(broken, reply(finish()))
        self.assertEqual(report["violations"], [])
        nudge = self.client.requests[1][0][-1]
        self.assertEqual(nudge["role"], "user")
        self.assertIn("finish_audit", nudge["content"])

    def test_finish_with_no_findings_gives_an_empty_valid_report(self):
        report = self.run_loop(reply(finish()))
        self.assertEqual(
            report, {"repository": "https://github.com/example/project", "violations": []}
        )


class FindingFlowTests(LoopTestCase):
    def test_rejected_finding_is_explained_and_can_be_corrected(self):
        wrong = {**FINDING, "file": "unknown.py"}
        report = self.run_loop(
            reply(call(REPORT_FINDING, wrong, "bad")),
            reply(call(REPORT_FINDING, FINDING, "good")),
            reply(finish()),
        )
        first = tool_replies(self.client.requests[1][0])[0]
        self.assertFalse(first["accepted"])
        self.assertIn("unknown.py", first["reason"])
        self.assertEqual(len(report["violations"]), 1)
        self.assertEqual(report["violations"][0]["file"], "config.py")

    def test_malformed_finding_arguments_are_a_rejection(self):
        report = self.run_loop(reply(call(REPORT_FINDING, "{broken", "bad")), reply(finish()))
        verdict = tool_replies(self.client.requests[1][0])[0]
        self.assertFalse(verdict["accepted"])
        self.assertIn("not valid JSON", verdict["reason"])
        self.assertEqual(report["violations"], [])

    def test_report_redacts_secrets_and_deduplicates(self):
        leaky = {**FINDING, "evidence": "TOKEN = supersecretvalue"}
        report = self.run_loop(
            reply(call(REPORT_FINDING, leaky, "one"), call(REPORT_FINDING, leaky, "two")),
            reply(finish()),
        )
        self.assertEqual(len(report["violations"]), 1)
        self.assertNotIn("supersecretvalue", json.dumps(report))
        verdicts = tool_replies(self.client.requests[1][0])
        self.assertEqual([v["accepted"] for v in verdicts], [True, False])

    def test_known_audit_rules_need_the_tools_exact_evidence(self):
        self.repository._read_cache["config.py"] = 'API_TOKEN = "local_test_token_value_12345"\n'
        sec = {
            **FINDING,
            "rule_id": "SEC-001",
            "severity": "critical",
            "description": "Credential hardcoded in source.",
            "evidence": "API_TOKEN = [REDACTED]",
        }
        vague = {**sec, "evidence": "This file has a security problem."}
        report = self.run_loop(
            reply(call(REPORT_FINDING, vague, "vague")),
            reply(call(REPORT_FINDING, sec, "exact")),
            reply(finish()),
        )
        self.assertIn("deterministic audit rule", tool_replies(self.client.requests[1][0])[0]["reason"])
        self.assertEqual([v["rule_id"] for v in report["violations"]], ["SEC-001"])

    def test_report_has_only_contract_keys_and_no_quote(self):
        report = self.run_loop(
            reply(call(REPORT_FINDING, {**FINDING, "quote": "setting = True"}, "q")),
            reply(finish()),
        )
        self.assertEqual(set(report), {"repository", "violations"})
        self.assertEqual(set(report["violations"][0]), set(REQUIRED_FIELDS))
        self.assertNotIn("quote", json.dumps(report))


class TextReplyTests(LoopTestCase):
    def test_json_report_in_a_message_is_checked_finding_by_finding(self):
        content = json.dumps(
            {
                "repository": self.repository.address.url,
                "violations": [FINDING, {"rule_id": "INCOMPLETE"}],
            }
        )
        report = self.run_loop(text(content))
        self.assertEqual(len(report["violations"]), 1)
        self.assertEqual(len(self.client.requests), 1)

    def test_fenced_json_report_is_accepted(self):
        content = "```json\n" + json.dumps({"violations": [FINDING]}) + "\n```"
        self.assertEqual(len(self.run_loop(text(content))["violations"]), 1)

    def test_plain_text_is_nudged_once_then_the_run_continues(self):
        report = self.run_loop(text("Looks fine to me."), reply(finish()))
        self.assertEqual(report["violations"], [])
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(self.client.requests[1][0][-2]["role"], "assistant")
        self.assertEqual(self.client.requests[1][0][-2]["content"], "Looks fine to me.")

    def test_two_plain_replies_in_a_row_end_the_run(self):
        report = self.run_loop(text("hmm"), text("still nothing"))
        self.assertEqual(report["violations"], [])
        self.assertEqual(len(self.client.requests), 2)

    def test_null_content_is_sent_back_as_an_empty_string(self):
        self.run_loop({"role": "assistant", "content": None}, reply(finish()))
        self.assertEqual(self.client.requests[1][0][-2]["content"], "")


class BudgetTests(LoopTestCase):
    def finalize_reply(self):
        return reply(call(REPORT_FINDING, FINDING, "late"), finish())

    def test_turn_limit_leads_to_a_finalize_turn_and_a_valid_report(self):
        settings = with_limits(self.settings, max_turns=2)
        with patch(TOOL_CALL, return_value={"files": []}):
            report = self.run_loop(
                reply(call("list_files", {}, "a")),
                reply(call("list_files", {}, "b")),
                self.finalize_reply(),
                settings=settings,
            )
        self.assertEqual(len(report["violations"]), 1)
        offered = self.client.requests[2][1]
        self.assertEqual(offered, [REPORT_FINDING, FINISH_AUDIT])
        prompt = self.client.requests[2][0][-1]
        self.assertEqual(prompt["role"], "user")
        self.assertIn("budget", prompt["content"])
        self.assertIn("Findings accepted so far: 0", prompt["content"])

    def test_tool_call_limit_answers_the_excess_calls_and_finalizes(self):
        settings = with_limits(self.settings, max_tool_calls=2)
        with patch(TOOL_CALL, return_value={"files": []}) as tools:
            report = self.run_loop(
                reply(*(call("list_files", {}, f"c{i}") for i in range(3))),
                reply(finish()),
                settings=settings,
            )
        self.assertEqual(tools.call_count, 2)
        replies = tool_replies(self.client.requests[1][0])
        self.assertEqual(len(replies), 3)
        self.assertIn("budget", replies[2]["error"])
        self.assertEqual(self.client.requests[1][1], [REPORT_FINDING, FINISH_AUDIT])
        self.assertEqual(report["violations"], [])

    def test_wall_time_limit_leads_to_finalize(self):
        settings = with_limits(self.settings, max_wall_seconds=100)
        now = [0.0]

        def slow_turn():
            now[0] = 1000.0
            return reply(call("list_files", {}, "slow"))

        with patch(TOOL_CALL, return_value={"files": []}):
            report = self.run_loop(
                slow_turn, self.finalize_reply(), settings=settings, clock=lambda: now[0]
            )
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(self.client.requests[1][1], [REPORT_FINDING, FINISH_AUDIT])
        self.assertEqual(len(report["violations"]), 1)

    def test_finalize_turns_are_bounded_and_inspection_is_refused(self):
        settings = with_limits(self.settings, max_turns=1, finalize_turns=2)
        with patch(TOOL_CALL, return_value={"files": []}) as tools:
            report = self.run_loop(
                reply(call("list_files", {}, "a")),
                reply(call("list_files", {}, "b")),
                reply(call("read_file", {"path": "config.py"}, "c")),
                settings=settings,
            )
        self.assertEqual(len(self.client.requests), 3)
        self.assertEqual(tools.call_count, 1)
        refused = tool_replies(self.client.requests[2][0])[-1]
        self.assertIn("inspection budget", refused["error"])
        self.assertEqual(report["violations"], [])


class ProviderFailureTests(LoopTestCase):
    def test_exhausted_provider_retries_abort_the_run(self):
        failure = LLMError("LLM provider request failed with HTTP 429.")
        with patch(TOOL_CALL, return_value={"files": []}):
            with self.assertRaisesRegex(AgentError, "HTTP 429"):
                self.run_loop(reply(call("list_files", {}, "a")), failure)

    def test_empty_provider_choices_are_reported(self):
        failure = LLMError("LLM provider response did not contain a completion choice.")
        with self.assertRaisesRegex(AgentError, "completion choice"):
            self.run_loop(failure)


if __name__ == "__main__":
    unittest.main()
