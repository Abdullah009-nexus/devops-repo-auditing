import contextlib
import io
import json
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from auditor.findings import REPORT_FINDING
from auditor.loop import AuditAgent
from auditor.settings import TraceSettings
from auditor.trace import Tracer, trace_path
from tests.support import default_settings
from tests.test_loop import (
    FINDING,
    FakeClient,
    call,
    finish,
    make_repository,
    reply,
    with_limits,
)
from tools import RepositoryAddress

API_KEY = "zz-opaque-provider-key-9"
GITHUB_TOKEN_VALUE = "ghp_" + "a1B2c3D4e5" * 3
OPAQUE_TOKEN = "opaque-github-credential-77"


def read_events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class TracerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_each_line_has_timestamp_turn_kind_and_data(self):
        path = self.directory / "nested" / "run.jsonl"
        tracer = Tracer(path, 500, environ={})
        tracer.event("run_start", 0, repository="https://github.com/o/r")
        (event,) = read_events(path)
        self.assertEqual(set(event), {"ts", "turn", "kind", "data"})
        self.assertEqual(event["kind"], "run_start")
        self.assertEqual(datetime.fromisoformat(event["ts"]).utcoffset().total_seconds(), 0)

    def test_pattern_and_exact_secrets_are_masked_in_nested_values(self):
        path = self.directory / "run.jsonl"
        tracer = Tracer(
            path, 500, environ={"OPENAI_API_KEY": API_KEY, "GITHUB_TOKEN": OPAQUE_TOKEN}
        )
        tracer.event(
            "tool_call",
            3,
            name="search_code",
            arguments={"pattern": GITHUB_TOKEN_VALUE, "paths": [API_KEY, OPAQUE_TOKEN]},
            error=f"failed with {API_KEY}",
        )
        raw = path.read_text(encoding="utf-8")
        for secret in (GITHUB_TOKEN_VALUE, API_KEY, OPAQUE_TOKEN):
            self.assertNotIn(secret, raw)
        self.assertIn("[REDACTED]", raw)

    def test_long_strings_are_cut(self):
        path = self.directory / "run.jsonl"
        Tracer(path, 20, environ={}).event("finding", 1, reason="r" * 100)
        reason = read_events(path)[0]["data"]["reason"]
        self.assertEqual(reason, "r" * 20 + "...")

    def test_no_path_means_no_output(self):
        Tracer(None, 500, environ={}).event("run_start", 0)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_write_failure_warns_once_and_never_raises(self):
        blocker = self.directory / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        tracer = Tracer(blocker / "run.jsonl", 500, environ={})
        with patch("auditor.trace.sys.stderr", new_callable=io.StringIO) as stderr:
            tracer.event("run_start", 0)
            tracer.event("run_end", 1)
        self.assertEqual(stderr.getvalue().count("Warning"), 1)


class TracePathTests(unittest.TestCase):
    def test_default_name_has_owner_repo_and_utc_stamp(self):
        settings = default_settings()
        address = RepositoryAddress.parse("https://github.com/octo/widget", settings.github)
        moment = datetime(2026, 10, 25, 9, 8, 7, tzinfo=timezone.utc)
        path = trace_path(settings.trace, address, moment)
        self.assertEqual(path, Path(settings.trace.directory) / "octo-widget-20261025T090807Z.jsonl")

    def test_disabled_tracing_has_no_path(self):
        settings = default_settings()
        address = RepositoryAddress.parse("https://github.com/octo/widget", settings.github)
        off = TraceSettings(enabled=False, directory="logs", max_string_chars=500)
        self.assertIsNone(trace_path(off, address))


class RunTraceTests(unittest.TestCase):
    def setUp(self):
        self.settings = with_limits(default_settings(), max_turns=2)
        self.repository = make_repository(self.settings)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "run.jsonl"

    def run_with_trace(self, *responses, environ=None):
        tracer = Tracer(self.path, 500, environ=environ or {})
        client = FakeClient(*responses)
        return AuditAgent(self.repository, self.settings, client, tracer).run()

    def test_a_full_run_writes_every_event_kind(self):
        bad = {**FINDING, "file": "missing.py"}
        with patch("auditor.loop.RepositoryTools.call", return_value={"files": []}):
            self.run_with_trace(
                reply(call("list_files", {}, "a"), call(REPORT_FINDING, bad, "bad")),
                reply(call("list_files", {}, "b")),
                reply(call(REPORT_FINDING, FINDING, "good"), finish()),
            )
        events = read_events(self.path)
        kinds = [event["kind"] for event in events]
        self.assertEqual(kinds[0], "run_start")
        self.assertEqual(kinds[-1], "run_end")
        self.assertEqual(
            set(kinds), {"run_start", "llm_response", "tool_call", "finding", "finalize", "run_end"}
        )
        start = events[0]["data"]
        self.assertEqual(start["model"], "fake-model")
        self.assertEqual(start["backend"], "api")
        self.assertEqual(start["budgets"]["max_turns"], 2)
        findings = [e["data"] for e in events if e["kind"] == "finding"]
        self.assertEqual([f["accepted"] for f in findings], [False, True])
        self.assertIn("missing.py", findings[0]["reason"])
        end = events[-1]["data"]
        self.assertEqual((end["status"], end["accepted"], end["rejected"]), ("turn_limit", 1, 1))
        tool_event = next(e["data"] for e in events if e["kind"] == "tool_call")
        self.assertEqual(set(tool_event), {"name", "arguments", "result_chars", "truncated", "error"})

    def test_secrets_in_arguments_output_and_environment_never_reach_the_file(self):
        output = {"content": f"TOKEN = {GITHUB_TOKEN_VALUE}", "again": API_KEY}
        with patch("auditor.loop.RepositoryTools.call", return_value=output):
            self.run_with_trace(
                reply(
                    call("search_code", {"pattern": GITHUB_TOKEN_VALUE}, "a"),
                    call("read_file", {"path": f"{API_KEY}.txt"}, "b"),
                ),
                reply(finish()),
                environ={"OPENAI_API_KEY": API_KEY},
            )
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn(GITHUB_TOKEN_VALUE, raw)
        self.assertNotIn(API_KEY, raw)
        self.assertNotIn("TOKEN =", raw)

    def test_unwritable_trace_does_not_stop_the_audit(self):
        blocker = self.path.parent / "blocker"
        blocker.write_text("x", encoding="utf-8")
        tracer = Tracer(blocker / "run.jsonl", 500, environ={})
        client = FakeClient(reply(call(REPORT_FINDING, FINDING, "good"), finish()))
        with contextlib.redirect_stderr(io.StringIO()):
            report = AuditAgent(self.repository, self.settings, client, tracer).run()
        self.assertEqual(len(report["violations"]), 1)


if __name__ == "__main__":
    unittest.main()
