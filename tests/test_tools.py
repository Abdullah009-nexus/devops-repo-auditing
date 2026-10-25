import dataclasses
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from auditor.repository import GitHubApiSource, RepositoryMetadata, _request_bytes
from tools import (
    GitHubRepository,
    RepositoryAddress,
    RepositoryError,
    RepositoryTools,
    build_tool_schemas,
    redact_secrets,
)
from tests.support import default_settings


def api_repository(files):
    """A repository backed by the API source, with the tree request mocked."""
    settings = default_settings()
    address = RepositoryAddress.parse("https://github.com/example/project", default_settings().github)
    tree = {
        "truncated": False,
        "tree": [{"type": "blob", "path": path, "size": size} for path, size in files.items()],
    }
    with patch("auditor.repository._request_json", return_value=tree):
        source = GitHubApiSource(address, RepositoryMetadata("main", 1), settings)
    return GitHubRepository(address, settings, source=source)


# Built in code from obviously fake parts; nothing credential-shaped is committed.
FAKE_VALUE = "fake" + "q" * 14
FAKE_GITHUB_TOKEN = "ghp_" + "q" * 36
# One line per key style the old pattern missed or that real repositories use.
PREFIXED_SECRET_LINES = (
    f"OPENAI_API_KEY={FAKE_VALUE}",
    f"DB_PASSWORD={FAKE_VALUE}",
    f"GITHUB_TOKEN={FAKE_VALUE}",
    f'dbPassword = "{FAKE_VALUE}"',
    f'{{"user": "admin", "apiKey": "{FAKE_VALUE}"}}',
    f"  client_secret: {FAKE_VALUE}",
)


class RepositoryAddressTests(unittest.TestCase):
    def test_normalizes_repository_url(self):
        address = RepositoryAddress.parse("https://github.com/example/project.git/", default_settings().github)
        self.assertEqual(address.url, "https://github.com/example/project")
        self.assertEqual((address.owner, address.name), ("example", "project"))

    def test_rejects_non_repository_urls(self):
        invalid = (
            "http://github.com/example/project",
            "https://github.com/example/project/tree/main",
            "https://user:password@github.com/example/project",
            "https://example.com/example/project",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(RepositoryError):
                RepositoryAddress.parse(value, default_settings().github)


class SecretRedactionTests(unittest.TestCase):
    def test_redacts_assignments_tokens_and_private_keys(self):
        content = (
            'api_key = "supersecretvalue"\n'
            'API_TOKEN = "local_test_token_value_12345"\n'
            'password = "short secret with spaces"\n'
            "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789\n"
            "-----BEGIN PRIVATE KEY-----\nprivate-material\n"
            "-----END PRIVATE KEY-----"
        )
        redacted = redact_secrets(content)
        self.assertNotIn("supersecretvalue", redacted)
        self.assertNotIn("local_test_token_value_12345", redacted)
        self.assertNotIn("short secret with spaces", redacted)
        self.assertNotIn("spaces", redacted)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz", redacted)
        self.assertNotIn("private-material", redacted)
        self.assertIn("[REDACTED]", redacted)


class RepositoryToolsTests(unittest.TestCase):
    def setUp(self):
        self.repository = api_repository(
            {"config.py": 64, "README.md": 20, "image.png": 4}
        )

    @patch("auditor.repository._request_bytes")
    def test_read_file_redacts_secret_and_tracks_observation(self, request_bytes):
        request_bytes.return_value = (
            b'API_KEY = "supersecretvalue"\nprint("safe")\n'
        )
        result = self.repository.read_file("config.py")
        self.assertIn("1: API_KEY = [REDACTED]", result["content"])
        self.assertNotIn("supersecretvalue", result["content"])
        self.assertTrue(self.repository.has_inspected_path("config.py"))
        self.assertTrue(self.repository.validate_finding_location("config.py", 2))
        self.assertFalse(self.repository.validate_finding_location("config.py", 3))

    def test_read_file_rejects_path_traversal_and_missing_files(self):
        for path in ("../secrets", "/etc/passwd", "missing.py"):
            with self.subTest(path=path), self.assertRaises(RepositoryError):
                self.repository.read_file(path)

    @patch("auditor.repository._request_bytes")
    def test_search_uses_literal_case_insensitive_matching(self, request_bytes):
        request_bytes.return_value = b"Hardcoded value\n"
        result = self.repository.search_code("hardcoded")
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(
            {match["file"] for match in result["matches"]},
            {"config.py", "README.md"},
        )
        self.assertNotIn("image.png", {item["file"] for item in result["matches"]})

    def assertNoSecretIn(self, text):
        # The message must not echo the value the assertion is guarding.
        self.assertFalse(FAKE_VALUE in text, "a secret value reached tool output")
        self.assertFalse(FAKE_GITHUB_TOKEN in text, "a token reached tool output")

    @patch("auditor.repository._request_bytes")
    def test_read_file_redacts_prefixed_camel_case_json_and_yaml_keys(self, request_bytes):
        request_bytes.return_value = "\n".join(PREFIXED_SECRET_LINES).encode("utf-8")
        result = self.repository.read_file("config.py")
        self.assertNoSecretIn(json.dumps(result))
        rendered = result["content"].splitlines()
        self.assertEqual(len(rendered), len(PREFIXED_SECRET_LINES))
        for index, line in enumerate(rendered):
            with self.subTest(index=index):
                self.assertIn("[REDACTED]", line)
        self.assertEqual(rendered[1], "2: DB_PASSWORD=[REDACTED]")

    @patch("auditor.repository._request_bytes")
    def test_search_code_cannot_find_a_secret_by_its_value(self, request_bytes):
        request_bytes.return_value = "\n".join(PREFIXED_SECRET_LINES).encode("utf-8")
        self.assertEqual(self.repository.search_code(FAKE_VALUE)["matches"], [])
        result = self.repository.search_code("[REDACTED]")
        self.assertNoSecretIn(json.dumps(result["matches"]))
        self.assertEqual(len(result["matches"]), 2 * len(PREFIXED_SECRET_LINES))
        for match in result["matches"]:
            with self.subTest(file=match["file"], line=match["line"]):
                self.assertIn("[REDACTED]", match["text"])

    @patch("auditor.repository._request_bytes")
    def test_git_log_redacts_secrets_in_commit_messages(self, request_bytes):
        message = (
            f"Set DB_PASSWORD={FAKE_VALUE} for staging\n\n"
            f"Also rotated GITHUB_TOKEN: {FAKE_GITHUB_TOKEN}"
        )
        request_bytes.return_value = json.dumps(
            [
                {
                    "sha": "0123456789abcdef0123",
                    "commit": {
                        "author": {"date": "2026-10-25T00:00:00Z"},
                        "message": message,
                    },
                }
            ]
        ).encode("utf-8")
        result = self.repository.git_log()
        self.assertNoSecretIn(json.dumps(result))
        self.assertEqual(len(result["commits"]), 1)
        redacted = result["commits"][0]["message"]
        self.assertIn("Set DB_PASSWORD=[REDACTED] for staging", redacted)
        self.assertIn("GITHUB_TOKEN: [REDACTED]", redacted)

    def test_dispatch_rejects_unknown_tools(self):
        dispatcher = RepositoryTools(self.repository)
        with self.assertRaises(RepositoryError):
            dispatcher.call("run_shell", {})

    def test_dispatch_rejects_missing_unexpected_and_wrong_type_arguments(self):
        dispatcher = RepositoryTools(self.repository)
        invalid_calls = (
            ("read_file", {}),
            ("read_file", {"path": "config.py", "execute": True}),
            ("read_file", {"path": 42}),
            ("search_code", {"pattern": "secret", "regex": "yes"}),
            ("search_code", {"pattern": "secret", "case": True}),
            ("list_files", {"depth": 2}),
        )
        for name, arguments in invalid_calls:
            with self.subTest(name=name, arguments=arguments):
                with self.assertRaises(RepositoryError):
                    dispatcher.call(name, arguments)

    def test_audit_tool_checks_only_files_already_read(self):
        self.repository._read_cache["config.py"] = 'API_TOKEN = "local_test_token_value"\n'
        self.repository._inspected_paths.add("config.py")
        dispatcher = RepositoryTools(self.repository)
        result = dispatcher.call("audit_inspected_files", {})
        self.assertEqual(result["checked_paths"], ["config.py"])
        self.assertEqual([item["rule_id"] for item in result["findings"]], ["SEC-001"])

    def test_dynamic_tool_schema_exposes_audit_checks(self):
        names = {
            item["function"]["name"]
            for item in build_tool_schemas(default_settings().git_log)
        }
        self.assertIn("audit_inspected_files", names)


class GitHubRequestTests(unittest.TestCase):
    def setUp(self):
        self.github = default_settings().github

    @patch("auditor.repository.urllib.request.urlopen")
    def test_http_errors_are_reported_without_response_bodies(self, urlopen):
        for status, expected in ((403, "rate limits"), (404, "not found")):
            urlopen.side_effect = urllib.error.HTTPError(
                "https://api.github.com/repos/example/missing",
                status,
                "failure",
                {},
                io.BytesIO(b"error details must not be exposed"),
            )
            try:
                with self.subTest(status=status), self.assertRaisesRegex(
                    RepositoryError, expected
                ):
                    _request_bytes(
                        "https://api.github.com/repos/example/missing",
                        self.github,
                        accept="application/json",
                        limit=10,
                    )
            finally:
                urlopen.side_effect.close()

    @patch("auditor.repository.urllib.request.urlopen")
    def test_timeout_is_reported(self, urlopen):
        urlopen.side_effect = TimeoutError()
        with self.assertRaisesRegex(RepositoryError, "timed out"):
            _request_bytes(
                "https://api.github.com/repos/example/project",
                self.github,
                accept="application/json",
                limit=10,
            )

    @patch("auditor.repository.urllib.request.urlopen")
    def test_url_error_is_reported(self, urlopen):
        urlopen.side_effect = urllib.error.URLError("connection failed")
        with self.assertRaisesRegex(RepositoryError, "connection failed"):
            _request_bytes(
                "https://api.github.com/repos/example/project",
                self.github,
                accept="application/json",
                limit=10,
            )

    @patch("auditor.repository.urllib.request.urlopen")
    def test_response_size_limit_is_enforced(self, urlopen):
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = b"123456"
        with self.assertRaisesRegex(RepositoryError, "size limit"):
            _request_bytes(
                "https://api.github.com/repos/example/project",
                self.github,
                accept="application/json",
                limit=5,
            )

    def test_tree_truncation_is_reported_to_the_agent(self):
        address = RepositoryAddress.parse("https://github.com/example/project", default_settings().github)
        settings = default_settings()
        settings = dataclasses.replace(
            settings, repository=dataclasses.replace(settings.repository, backend="api")
        )
        with patch(
            "auditor.repository._request_json",
            side_effect=[
                {"private": False, "default_branch": "main"},
                {
                    "truncated": True,
                    "tree": [{"type": "blob", "path": "README.md", "size": 8}],
                },
            ],
        ):
            repository = GitHubRepository(address, settings)
        result = repository.list_files()
        self.assertTrue(result["truncated"])
        self.assertIn("absence of a path is not evidence", result["notice"])


if __name__ == "__main__":
    unittest.main()
