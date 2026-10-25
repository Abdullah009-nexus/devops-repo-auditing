"""Phase 1 gate: token containment, metadata/workflow redaction, dispatcher validation."""

import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from auditor import repository as repository_module
from auditor.repository import RepositoryError, RepositoryMetadata, open_source
from tests.support import default_settings
from tests.test_repository import ADDRESS, METADATA, FakeGit
from tests.test_tool_upgrades import api_repository, serve
from tools import GitHubRepository, RepositoryTools

GATE_TOKEN = "ghp_" + "z" * 36
GATE_SECRET = "ghp_" + "y" * 36


class TokenContainmentTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"GITHUB_TOKEN": GATE_TOKEN})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_tool_output_contains_the_token(self):
        files = {"a.py": 5, ".github/workflows/ci.yml": 5}
        repository = api_repository(files)
        contents = {"a.py": "x = 1\n", ".github/workflows/ci.yml": "on: push\njobs:\n  b: {}\n"}
        tools = RepositoryTools(repository)
        calls = (
            ("list_files", {}), ("read_file", {"path": "a.py"}),
            ("search_code", {"pattern": "x"}), ("get_repo_metadata", {}),
            ("list_workflows", {}), ("audit_inspected_files", {}),
        )
        with serve(repository, contents):
            for name, arguments in calls:
                with self.subTest(tool=name):
                    self.assertNotIn(GATE_TOKEN, json.dumps(tools.call(name, arguments)))

    def test_a_token_echoed_in_a_transport_error_does_not_reach_the_message(self):
        error = urllib.error.URLError(f"proxy rejected Bearer {GATE_TOKEN}")
        with patch("auditor.repository.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RepositoryError) as caught:
                repository_module._request_bytes(
                    "https://api.github.com/repos/x", default_settings().github,
                    accept="x", limit=10,
                )
        self.assertNotIn(GATE_TOKEN, str(caught.exception))

    def test_http_denied_message_never_contains_the_token(self):
        error = urllib.error.HTTPError(
            f"https://api.github.com/?t={GATE_TOKEN}", 403, "denied", {}, None
        )
        with patch("auditor.repository.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RepositoryError) as caught:
                repository_module._request_bytes(
                    "https://api.github.com/repos/x", default_settings().github,
                    accept="x", limit=10,
                )
        self.assertNotIn(GATE_TOKEN, str(caught.exception))

    def test_fallback_reason_and_listing_never_contain_the_token(self):
        fake = FakeGit({"a.py": b"x\n"}, statuses={"clone": 128})
        settings = default_settings()
        with (
            patch("auditor.repository._request_json",
                  side_effect=[METADATA, {"truncated": False, "tree": []}]),
            patch("auditor.repository.subprocess.Popen", fake),
        ):
            source = open_source(ADDRESS, settings)
        listing = GitHubRepository(ADDRESS, settings, source=source).list_files()
        self.assertIn("fallback_reason", listing)
        self.assertNotIn(GATE_TOKEN, json.dumps(listing))


class MetadataAndWorkflowRedactionTests(unittest.TestCase):
    def test_metadata_text_fields_are_redacted(self):
        metadata = RepositoryMetadata(
            "main", 1, visibility=f"password={GATE_SECRET}",
            license_spdx_id=GATE_SECRET, pushed_at=GATE_SECRET,
        )
        result = api_repository({"a.py": 1}, metadata=metadata).get_repo_metadata()
        self.assertNotIn(GATE_SECRET, json.dumps(result))

    def test_workflow_scan_output_is_redacted(self):
        path = ".github/workflows/ci.yml"
        text = (
            f"on: push\npermissions: {GATE_SECRET}\n"
            f"jobs:\n  build:\n    env:\n      API_KEY: {GATE_SECRET}\n"
        )
        repository = api_repository({path: len(text)})
        with serve(repository, {path: text}):
            result = repository.list_workflows()
        self.assertEqual(len(result["workflows"]), 1)
        self.assertNotIn(GATE_SECRET, json.dumps(result))


class DispatcherValidationTests(unittest.TestCase):
    def setUp(self):
        self.tools = RepositoryTools(api_repository({"a.py": 1}))

    def assertRejected(self, name, arguments):
        with self.subTest(name=name, arguments=arguments):
            with self.assertRaises(RepositoryError):
                self.tools.call(name, arguments)

    def test_every_tool_rejects_an_unknown_argument(self):
        for name, base in (
            ("list_files", {}), ("read_file", {"path": "a.py"}),
            ("search_code", {"pattern": "x"}), ("git_log", {}),
            ("get_repo_metadata", {}), ("list_workflows", {}),
            ("audit_inspected_files", {}),
        ):
            self.assertRejected(name, {**base, "shell": "id"})

    def test_wrong_types_are_rejected_for_every_typed_argument(self):
        for name, arguments in (
            ("list_files", {"path_prefix": 1}), ("list_files", {"glob": ["*.py"]}),
            ("read_file", {"path": ["a.py"]}),
            ("read_file", {"path": "a.py", "start_line": "1"}),
            ("read_file", {"path": "a.py", "end_line": 1.5}),
            ("search_code", {"pattern": 5}),
            ("search_code", {"pattern": "x", "path_glob": 3}),
            ("git_log", {"max_commits": "5"}), ("git_log", {"path": 7}),
        ):
            self.assertRejected(name, arguments)

    def test_a_boolean_is_never_accepted_for_an_integer(self):
        for name, arguments in (
            ("read_file", {"path": "a.py", "start_line": True}),
            ("read_file", {"path": "a.py", "end_line": False}),
            ("git_log", {"max_commits": True}),
        ):
            self.assertRejected(name, arguments)

    def test_required_arguments_missing_or_null_are_rejected(self):
        self.assertRejected("read_file", {"path": None})
        self.assertRejected("search_code", {})

    def test_null_optional_arguments_are_treated_as_absent(self):
        result = self.tools.call("list_files", {"path_prefix": None, "glob": None})
        self.assertEqual(result["files"], ["a.py"])


if __name__ == "__main__":
    unittest.main()
