import contextlib
import io
import json
import os
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import agent
from auditor.findings import FINISH_AUDIT, REPORT_FINDING
from auditor.llm_client import LLMClient
from auditor.loop import AgentError, AuditAgent
from auditor.trace import Tracer
from tests.support import default_settings
from tests.test_loop import FINDING, FakeClient, call, make_repository, reply
from tests.test_repository import FakeGit

ENVIRON = {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}
REPO_URL = "https://github.com/example/project"


class ProviderFailureTests(unittest.TestCase):
    """The real client behind the loop: transport failures stay explicit and fatal."""

    def audit_with(self, urlopen):
        settings = default_settings()
        client = LLMClient(settings.llm, ENVIRON, sleep=lambda seconds: None)
        with patch("auditor.llm_client._opener.open", urlopen):
            AuditAgent(make_repository(settings), settings, client, Tracer(None, 500)).run()

    def test_http_failure_is_explicit_and_hides_the_provider_body(self):
        error = urllib.error.HTTPError(
            "https://api.example.com/chat/completions",
            429,
            "rate limited",
            {},
            io.BytesIO(b"provider error details must not be exposed"),
        )
        self.addCleanup(error.close)
        with self.assertRaisesRegex(AgentError, "HTTP 429") as caught:
            self.audit_with(Mock(side_effect=error))
        self.assertNotIn("details", str(caught.exception))

    def test_timeout_is_explicit(self):
        with self.assertRaisesRegex(AgentError, "timed out"):
            self.audit_with(Mock(side_effect=TimeoutError()))

    def test_invalid_provider_json_is_explicit(self):
        urlopen = MagicMock()
        urlopen.return_value.__enter__.return_value.read.return_value = b"not json"
        with self.assertRaisesRegex(AgentError, "invalid JSON"):
            self.audit_with(urlopen)


class CliTests(unittest.TestCase):
    ARGV = ["--repo", REPO_URL, "--output", "report.json"]

    def setUp(self):
        self.workspaces = []
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.chdir(directory.name))
        stack.enter_context(patch.dict(os.environ, ENVIRON))
        stack.enter_context(
            patch(
                "auditor.repository._request_json",
                return_value={"private": False, "default_branch": "main", "size": 1},
            )
        )
        self.git = FakeGit({"app.py": b"print('x')\n"})
        stack.enter_context(patch("auditor.repository.subprocess.Popen", side_effect=self._clone))
        stack.enter_context(
            patch("urllib.request.urlopen", side_effect=AssertionError("network used"))
        )
        stack.enter_context(
            patch("auditor.llm_client._opener.open", side_effect=AssertionError("network used"))
        )
        self.stdout = stack.enter_context(patch("agent.sys.stdout", new_callable=io.StringIO))
        self.stderr = stack.enter_context(patch("agent.sys.stderr", new_callable=io.StringIO))
        self.directory = Path(directory.name)
        self.output = self.directory / "report.json"

    def _clone(self, args, **kwargs):
        if "clone" in args:
            self.workspaces.append(Path(args[-1]).parent)
        return self.git(args, **kwargs)

    def _run_agent_that(self, outcome):
        def run_agent(repository, settings, client, tracer):
            self.assertEqual(repository.source.backend, "clone")
            self.assertTrue(repository.source.root.is_dir())
            return outcome(repository)

        return patch.object(agent, "run_agent", side_effect=run_agent)

    def test_clone_is_removed_after_a_successful_audit(self):
        report = {"repository": REPO_URL, "violations": []}
        with self._run_agent_that(lambda repository: report):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), report)
        self.assertEqual(len(self.workspaces), 1)
        self.assertFalse(self.workspaces[0].exists())

    def test_clone_is_removed_when_the_audit_fails(self):
        def fail(repository):
            raise AgentError("provider unavailable")

        with self._run_agent_that(fail):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 1)
        self.assertIn("provider unavailable", self.stderr.getvalue())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.workspaces[0].exists())

    def test_clone_is_removed_and_the_error_reported_when_something_unforeseen_fails(self):
        def crash(repository):
            raise RuntimeError("unexpected")

        with self._run_agent_that(crash):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 1)
        self.assertIn("Error: unexpected RuntimeError: unexpected", self.stderr.getvalue())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.workspaces[0].exists())

    def test_report_that_breaks_the_contract_is_not_written(self):
        with self._run_agent_that(lambda repository: {"repository": REPO_URL}):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 1)
        self.assertFalse(self.output.exists())

    def test_unwritable_output_exits_1(self):
        (self.directory / "blocked").write_text("a file", encoding="utf-8")
        report = {"repository": REPO_URL, "violations": []}
        with self._run_agent_that(lambda repository: report):
            result = agent.main(["--repo", REPO_URL, "--output", "blocked/report.json"])
        self.assertEqual(result, 1)
        self.assertTrue(self.stderr.getvalue().startswith("Error: "))

    def test_missing_api_key_exits_2_before_any_clone(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 2)
        self.assertIn("OPENAI_API_KEY", self.stderr.getvalue())
        self.assertEqual(self.workspaces, [])

    def test_missing_model_exits_2(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": ""}):
            self.assertEqual(agent.main(self.ARGV), 2)
        self.assertIn("OPENAI_MODEL", self.stderr.getvalue())

    def test_invalid_repository_url_exits_2(self):
        result = agent.main(["--repo", "https://example.com/not/github", "--output", "report.json"])
        self.assertEqual(result, 2)
        self.assertTrue(self.stderr.getvalue().startswith("Error: "))
        self.assertEqual(self.workspaces, [])

    def test_unreachable_repository_exits_1(self):
        failure = agent.RepositoryError("repository unavailable")
        with patch.object(agent, "GitHubRepository", side_effect=failure):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 1)
        self.assertIn("repository unavailable", self.stderr.getvalue())

    def test_argparse_rejects_missing_arguments_with_exit_2(self):
        with self.assertRaises(SystemExit) as caught:
            agent.main(["--repo", REPO_URL])
        self.assertEqual(caught.exception.code, 2)

    def test_provider_failure_mid_run_exits_1_and_writes_no_report(self):
        client = FakeClient(
            reply(call("list_files", {}, "a")),
            agent.LLMError("LLM provider request failed with HTTP 503."),
        )
        with patch.object(agent, "LLMClient", return_value=client):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 1)
        self.assertIn("HTTP 503", self.stderr.getvalue())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.workspaces[0].exists())

    def test_invalid_config_exits_2_before_any_clone(self):
        with patch.dict(os.environ, {"AUDITOR_AGENT_MAX_TURNS": "many"}):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 2)
        self.assertIn("AUDITOR_AGENT_MAX_TURNS", self.stderr.getvalue())
        self.assertEqual(self.workspaces, [])

    def test_unreadable_config_file_exits_2(self):
        result = agent.main([*self.ARGV, "--config", str(self.directory / "missing.toml")])
        self.assertEqual(result, 2)
        self.assertEqual(self.workspaces, [])

    def test_provider_retries_exhausted_exits_1_and_writes_no_report(self):
        failure = urllib.error.HTTPError(
            "https://api.example.com/chat/completions", 503, "unavailable", {}, io.BytesIO(b"")
        )
        self.addCleanup(failure.close)
        with (
            patch("auditor.llm_client.time.sleep"),
            patch("auditor.llm_client._opener.open", Mock(side_effect=failure)),
        ):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 1)
        self.assertIn("HTTP 503", self.stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_completed_audit_with_no_findings_exits_0_with_an_empty_array(self):
        client = FakeClient(reply(call(FINISH_AUDIT, {"summary": "nothing found"}, "a")))
        with patch.object(agent, "LLMClient", return_value=client):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(self.output.read_text(encoding="utf-8")),
            {"repository": REPO_URL, "violations": []},
        )

    def test_full_run_writes_a_contract_valid_report_and_a_trace(self):
        client = FakeClient(
            reply(call("list_files", {}, "a")),
            reply(call("read_file", {"path": "app.py"}, "b")),
            reply(
                call(
                    REPORT_FINDING,
                    {**FINDING, "file": "app.py", "line": 1, "evidence": "print('x')", "quote": "print('x')"},
                    "c",
                ),
                call(FINISH_AUDIT, {"summary": "done"}, "d"),
            ),
        )
        with patch.object(agent, "LLMClient", return_value=client):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 0)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(set(report), {"repository", "violations"})
        self.assertEqual(report["violations"][0]["file"], "app.py")
        traces = list((self.directory / "logs").glob("example-project-*.jsonl"))
        self.assertEqual(len(traces), 1)

    def test_trace_option_overrides_the_default_location(self):
        client = FakeClient(reply(call(FINISH_AUDIT, {"summary": "done"}, "a")))
        target = self.directory / "custom" / "run.jsonl"
        with patch.object(agent, "LLMClient", return_value=client):
            result = agent.main([*self.ARGV, "--trace", str(target)])
        self.assertEqual(result, 0)
        self.assertTrue(target.is_file())
        self.assertFalse((self.directory / "logs").exists())

    def test_disabled_trace_setting_writes_no_logs_directory(self):
        client = FakeClient(reply(call(FINISH_AUDIT, {"summary": "done"}, "a")))
        with (
            patch.dict(os.environ, {"AUDITOR_TRACE_ENABLED": "false"}),
            patch.object(agent, "LLMClient", return_value=client),
        ):
            result = agent.main(self.ARGV)
        self.assertEqual(result, 0)
        self.assertFalse((self.directory / "logs").exists())

    def test_verbose_logs_turns_and_tool_names_to_stderr(self):
        client = FakeClient(reply(call("list_files", {}, "a")), reply(call(FINISH_AUDIT, {}, "b")))
        with patch.object(agent, "LLMClient", return_value=client):
            result = agent.main([*self.ARGV, "--verbose"])
        self.assertEqual(result, 0)
        self.assertIn("turn 1: list_files", self.stderr.getvalue())

    def test_quiet_run_prints_no_progress(self):
        client = FakeClient(reply(call(FINISH_AUDIT, {}, "a")))
        with patch.object(agent, "LLMClient", return_value=client):
            agent.main(self.ARGV)
        self.assertNotIn("turn 1", self.stderr.getvalue())


class ReportWriterTests(unittest.TestCase):
    def test_report_writer_creates_valid_json(self):
        report = {"repository": REPO_URL, "violations": []}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "report.json"
            agent.write_report(str(output), report)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertFalse(output.with_name(output.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()


class UnexpectedFailureTests(unittest.TestCase):
    def test_an_unforeseen_exception_becomes_one_error_line_with_exit_1(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            with (
                patch.object(agent, "run_agent", side_effect=ZeroDivisionError("boom")),
                patch.object(agent, "GitHubRepository"),
                patch.object(agent, "LLMClient"),
                contextlib.redirect_stderr(stderr),
            ):
                code = agent.main(
                    ["--repo", "https://github.com/o/r", "--output", str(Path(directory) / "r.json")]
                )
        self.assertEqual(code, 1)
        self.assertIn("Error: unexpected ZeroDivisionError: boom", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


class InterruptTests(unittest.TestCase):
    def test_ctrl_c_exits_cleanly_without_a_traceback(self):
        stderr = io.StringIO()
        with (
            patch.object(agent, "_audit", side_effect=KeyboardInterrupt),
            contextlib.redirect_stderr(stderr),
        ):
            code = agent.main(["--repo", "https://github.com/o/r", "--output", "report.json"])
        self.assertEqual(code, 130)
        self.assertIn("Interrupted", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
