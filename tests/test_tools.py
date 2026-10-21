import unittest
from unittest.mock import patch

from tools import (
    GitHubRepository,
    RepositoryAddress,
    RepositoryError,
    RepositoryTools,
    redact_secrets,
)


class RepositoryAddressTests(unittest.TestCase):
    def test_normalizes_repository_url(self):
        address = RepositoryAddress.parse("https://github.com/example/project.git/")
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
                RepositoryAddress.parse(value)


class SecretRedactionTests(unittest.TestCase):
    def test_redacts_assignments_tokens_and_private_keys(self):
        content = (
            'api_key = "supersecretvalue"\n'
            'password = "short secret with spaces"\n'
            "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789\n"
            "-----BEGIN PRIVATE KEY-----\nprivate-material\n"
            "-----END PRIVATE KEY-----"
        )
        redacted = redact_secrets(content)
        self.assertNotIn("supersecretvalue", redacted)
        self.assertNotIn("short secret with spaces", redacted)
        self.assertNotIn("spaces", redacted)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz", redacted)
        self.assertNotIn("private-material", redacted)
        self.assertIn("[REDACTED]", redacted)


class RepositoryToolsTests(unittest.TestCase):
    def setUp(self):
        self.repository = GitHubRepository.__new__(GitHubRepository)
        self.repository.address = RepositoryAddress.parse(
            "https://github.com/example/project"
        )
        self.repository.branch = "main"
        self.repository.tree_truncated = False
        self.repository.list_truncated = False
        self.repository.files = {
            "config.py": {"type": "blob", "path": "config.py", "size": 64},
            "README.md": {"type": "blob", "path": "README.md", "size": 20},
            "image.png": {"type": "blob", "path": "image.png", "size": 4},
        }
        self.repository._read_cache = {}
        self.repository._inspected_paths = set()

    @patch("tools._request_bytes")
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

    @patch("tools._request_bytes")
    def test_search_uses_literal_case_insensitive_matching(self, request_bytes):
        request_bytes.return_value = b"Hardcoded value\n"
        result = self.repository.search_code("hardcoded")
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(
            {match["file"] for match in result["matches"]},
            {"config.py", "README.md"},
        )
        self.assertNotIn("image.png", {item["file"] for item in result["matches"]})

    def test_dispatch_rejects_unknown_tools(self):
        dispatcher = RepositoryTools(self.repository)
        with self.assertRaises(RepositoryError):
            dispatcher.call("run_shell", {})


if __name__ == "__main__":
    unittest.main()
