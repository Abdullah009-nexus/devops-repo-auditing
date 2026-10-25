import time
import unittest

from auditor.repository import LocalDirSource
from tests.support import default_settings
from tests.test_repository import FIXTURE
from tools import GitHubRepository


class NestedRepeatGuardTests(unittest.TestCase):
    def setUp(self):
        settings = default_settings()
        self.repository = GitHubRepository(
            None, settings, LocalDirSource(FIXTURE, settings, "main")
        )

    def test_refuses_nested_repetition_without_scanning(self):
        for pattern in ("(a+)+$", r"(\w+\s?)*x", "(a{1,})+b", "(a*)*b"):
            with self.subTest(pattern=pattern):
                started = time.perf_counter()
                result = self.repository.search_code(pattern, regex=True)
                self.assertIn("refused", result["error"])
                self.assertNotIn("matches", result)
                self.assertLess(time.perf_counter() - started, 1.0)

    def test_refuses_repeated_alternation_with_a_hint(self):
        for pattern in ("(a|aa)+$", "(a|b)*c", "(x|xy){1,}"):
            with self.subTest(pattern=pattern):
                result = self.repository.search_code(pattern, regex=True)
                self.assertIn("refused", result["error"])
                self.assertIn("separately", result["error"])
                self.assertNotIn("matches", result)

    def test_allows_ordinary_groups_and_literal_mode(self):
        for pattern in ("(ab)+", r"(\d{3})+", r"uses:\s+\S+@(main|master)", "(ab|cd){2}"):
            with self.subTest(pattern=pattern):
                self.assertNotIn("error", self.repository.search_code(pattern, regex=True))
        # The guard only concerns regex mode; the same text as a literal is harmless.
        self.assertNotIn("error", self.repository.search_code("(a+)+", regex=False))


if __name__ == "__main__":
    unittest.main()
