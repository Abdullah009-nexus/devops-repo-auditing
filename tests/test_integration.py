"""End to end: scripted model -> AuditAgent -> real RepositoryTools on a fixture directory."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent

from auditor.findings import FINISH_AUDIT, REPORT_FINDING
from auditor.loop import AuditAgent
from auditor.report import validate_report_shape, write_report
from auditor.repository import LocalDirSource
from auditor.schema import REQUIRED_FIELDS, SEVERITIES
from auditor.trace import Tracer
from tests.support import default_settings
from tests.test_loop import FakeClient, call, reply
from tools import GitHubRepository, RepositoryAddress

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_SECRET = "local_test_token_value_12345"
WORKFLOW = ".github/workflows/unsafe.yml"
REPO_URL = "https://github.com/example/project"

SECRET_FINDING = {
    "rule_id": "SEC-001",
    "category": "security",
    "severity": "high",
    "file": "config.py",
    "line": 1,
    "description": "A credential-like value is assigned directly in a repository file.",
    "evidence": "API_TOKEN = [REDACTED]",
    "recommendation": "Load credentials from an environment variable.",
}
PERMISSIONS_FINDING = {
    "rule_id": "CI-001",
    "category": "security",
    "severity": "high",
    "file": WORKFLOW,
    "line": 5,
    "description": "The workflow grants write access to every token scope.",
    "evidence": "permissions: write-all",
    "recommendation": "Declare only the permissions the job needs.",
}
CHECKOUT_FINDING = {
    "rule_id": "CI-002",
    "category": "security",
    "severity": "high",
    "file": WORKFLOW,
    "line": 12,
    "description": "A pull_request_target workflow checks out pull-request head code.",
    # The known-rule check compares evidence verbatim, indentation and related-line note included.
    "evidence": (
        "          ref: ${{ github.event.pull_request.head.sha }}\n"
        "Related workflow evidence: pull_request_target trigger, actions/checkout, "
        "and a project test/build command."
    ),
    "recommendation": "Use pull_request, or keep untrusted code out of the privileged workflow.",
}
EVENT_KINDS = {"run_start", "llm_response", "tool_call", "finding", "run_end"}


class FixtureAuditTests(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.trace = Path(directory.name) / "run.jsonl"
        self.output = Path(directory.name) / "report.json"

    def audit(self, fixture, *responses):
        address = RepositoryAddress.parse(REPO_URL, self.settings.github)
        source = LocalDirSource(FIXTURES / fixture, self.settings, "main")
        self.client = FakeClient(*responses)
        with GitHubRepository(address, self.settings, source) as repository:
            report = AuditAgent(
                repository, self.settings, self.client, Tracer(self.trace, 500)
            ).run()
        validate_report_shape(report)
        return report

    def trace_events(self):
        return [json.loads(line) for line in self.trace.read_text(encoding="utf-8").splitlines()]

    def test_defective_fixture_yields_exactly_the_three_expected_findings(self):
        report = self.audit(
            "audit-defective",
            reply(call("list_files", {}, "a")),
            reply(
                call("read_file", {"path": "config.py"}, "b"),
                call("read_file", {"path": WORKFLOW}, "c"),
            ),
            reply(call("audit_inspected_files", {}, "d")),
            reply(
                call(REPORT_FINDING, SECRET_FINDING, "e"),
                call(REPORT_FINDING, PERMISSIONS_FINDING, "f"),
                call(REPORT_FINDING, CHECKOUT_FINDING, "g"),
            ),
            reply(call(FINISH_AUDIT, {"summary": "done"}, "h")),
        )
        self.assertEqual(report["repository"], REPO_URL)
        found = {(v["rule_id"], v["file"], v["line"]) for v in report["violations"]}
        self.assertEqual(
            found,
            {("SEC-001", "config.py", 1), ("CI-001", WORKFLOW, 5), ("CI-002", WORKFLOW, 12)},
        )
        self.assertEqual(len(report["violations"]), 3)

        events = self.trace_events()
        self.assertEqual({event["kind"] for event in events}, EVENT_KINDS)
        tool_names = [e["data"]["name"] for e in events if e["kind"] == "tool_call"]
        self.assertIn("audit_inspected_files", tool_names)
        accepted = [e for e in events if e["kind"] == "finding" and e["data"]["accepted"]]
        self.assertEqual(len(accepted), 3)
        self.assertNotIn(FAKE_SECRET, self.trace.read_text(encoding="utf-8"))

        write_report(str(self.output), report)
        self.assertNotIn(FAKE_SECRET, self.output.read_text(encoding="utf-8"))
        for request, _ in self.client.requests:
            self.assertNotIn(FAKE_SECRET, json.dumps(request))

    def test_healthy_fixture_yields_no_violations(self):
        report = self.audit(
            "audit-healthy",
            reply(call("list_files", {}, "a")),
            reply(
                call("read_file", {"path": "config.py"}, "b"),
                call("read_file", {"path": ".github/workflows/test.yml"}, "c"),
            ),
            reply(call("audit_inspected_files", {}, "d")),
            reply(call(FINISH_AUDIT, {"summary": "nothing to report"}, "e")),
        )
        self.assertEqual(report, {"repository": REPO_URL, "violations": []})
        kinds = {event["kind"] for event in self.trace_events()}
        self.assertEqual(kinds, EVENT_KINDS - {"finding"})

    def test_written_report_follows_the_pdf_contract_exactly(self):
        report = self.audit(
            "audit-defective",
            reply(call("read_file", {"path": "config.py"}, "a")),
            reply(
                call(REPORT_FINDING, SECRET_FINDING, "b"),
                call(FINISH_AUDIT, {"summary": "done"}, "c"),
            ),
        )
        write_report(str(self.output), report)
        written = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(set(written), {"repository", "violations"})
        self.assertTrue(written["violations"])
        for violation in written["violations"]:
            self.assertEqual(set(violation), set(REQUIRED_FIELDS))
            self.assertEqual(len(violation), 8)
            self.assertIn(violation["severity"], SEVERITIES)
            self.assertNotIn("quote", violation)
            if violation["file"] is None:
                self.assertIsNone(violation["line"])


class CliOnLocalFixtureTests(unittest.TestCase):
    """X05: the real CLI entry point over a directory-backed repository and a scripted model."""

    def test_main_writes_a_valid_report_and_a_trace(self):
        settings = default_settings()
        real_repository = GitHubRepository
        client = FakeClient(
            reply(call("read_file", {"path": "config.py"}, "a")),
            reply(
                call(REPORT_FINDING, SECRET_FINDING, "b"),
                call(FINISH_AUDIT, {"summary": "done"}, "c"),
            ),
        )

        def open_fixture(address, repo_settings):
            source = LocalDirSource(FIXTURES / "audit-defective", repo_settings, "main")
            return real_repository(address, repo_settings, source)

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.chdir(directory))
            stack.enter_context(
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"})
            )
            stack.enter_context(patch.object(agent, "GitHubRepository", open_fixture))
            stack.enter_context(patch.object(agent, "LLMClient", return_value=client))
            stack.enter_context(patch("agent.sys.stdout", new_callable=io.StringIO))
            output = Path(directory) / "report.json"
            trace = Path(directory) / "run.jsonl"
            result = agent.main(["--repo", REPO_URL, "--output", str(output), "--trace", str(trace)])
            report = json.loads(output.read_text(encoding="utf-8"))
            validate_report_shape(report)
            self.assertEqual(result, 0)
            self.assertEqual(report["repository"], REPO_URL)
            self.assertEqual([v["rule_id"] for v in report["violations"]], ["SEC-001"])
            self.assertTrue(trace.is_file())
            self.assertNotIn(FAKE_SECRET, output.read_text(encoding="utf-8"))
            self.assertNotIn(FAKE_SECRET, trace.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
