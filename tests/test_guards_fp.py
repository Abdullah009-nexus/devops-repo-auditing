"""Catalog, severity, grounding, false-positive and duplicate checks in FindingStore."""

import unittest
from unittest.mock import patch

from auditor.findings import FindingStore
from auditor.repository import GitHubApiSource, RepositoryMetadata
from tests.support import default_settings
from tools import GitHubRepository, RepositoryAddress

FILES = {
    "config.py": "setting = True\nother = 2\nthree = 3\nfour = 4\nfive = 5\nsix = 6\nseven = 7\n",
    "tests/fixtures/app/settings.py": 'API_TOKEN = "abc12345xyz"\n',
    ".env.example": "KEY=value\n",
    "vendor/lib/x.py": "x = 1\n",
    "src/vendorish/y.py": "y = 1\n",
    "README.md": 'password = "hunter2hunter2"\n',
    "docs/guide.md": "guide = 1\n",
    "listed_only.py": "unread = 1\n",
}
UNREAD = {"listed_only.py"}


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings()
        address = RepositoryAddress.parse("https://github.com/example/project", self.settings.github)
        tree = {"tree": [{"type": "blob", "path": path, "size": 10} for path in FILES]}
        with patch("auditor.repository._request_json", return_value=tree):
            source = GitHubApiSource(address, RepositoryMetadata("main", 1), self.settings)
        self.repository = GitHubRepository(address, self.settings, source=source)
        self.repository._read_cache = {p: c for p, c in FILES.items() if p not in UNREAD}
        self.repository._inspected_paths = set(self.repository._read_cache)
        self.store = FindingStore(self.repository, self.settings.agent)

    def candidate(self, **overrides):
        values = {
            "rule_id": "QUA-002",
            "category": "code_quality",
            "severity": "low",
            "file": "config.py",
            "line": 1,
            "description": "Something is wrong",
            "evidence": "setting = True",
            "recommendation": "Change it",
            "quote": "setting = True",
        }
        values.update(overrides)
        return values

    def assertRejected(self, decision, *fragments):
        self.assertFalse(decision.accepted, decision.reason)
        for fragment in fragments:
            self.assertIn(fragment, decision.reason)


class CatalogTests(StoreCase):
    def test_unknown_rule_is_rejected_with_catalog_hint(self):
        self.assertRejected(self.store.add(self.candidate(rule_id="XYZ-001")), "catalog", "-099")

    def test_known_rule_is_accepted(self):
        self.assertTrue(self.store.add(self.candidate()).accepted)

    def test_category_must_match_the_rule(self):
        decision = self.store.add(self.candidate(category="security"))
        self.assertRejected(decision, "QUA-002", "code_quality")

    def test_category_alias_matching_the_rule_is_accepted(self):
        decision = self.store.add(self.candidate(category="Code Quality"))
        self.assertTrue(decision.accepted, decision.reason)


class SeverityTests(StoreCase):
    def test_two_steps_from_default_is_rejected(self):
        decision = self.store.add(self.candidate(severity="high"))
        self.assertRejected(decision, "default for QUA-002 is low", "neighbouring")

    def test_one_step_from_default_is_accepted(self):
        self.assertTrue(self.store.add(self.candidate(severity="medium")).accepted)

    def test_default_is_accepted(self):
        self.assertTrue(self.store.add(self.candidate(severity="low")).accepted)

    def test_critical_rule_may_drop_one_level_only(self):
        sec = dict(rule_id="SEC-002", category="security", file="config.py")
        self.assertTrue(self.store.add(self.candidate(severity="high", **sec)).accepted)
        other = FindingStore(self.repository, self.settings.agent)
        self.assertRejected(other.add(self.candidate(severity="medium", **sec)), "default for SEC-002")


class GroundingTests(StoreCase):
    def test_located_finding_without_quote_is_rejected(self):
        candidate = self.candidate()
        del candidate["quote"]
        self.assertRejected(self.store.add(candidate), "quote is required", "lines 1-3")

    def test_quote_within_window_is_accepted_with_whitespace_differences(self):
        decision = self.store.add(self.candidate(line=3, quote="setting   =\n True"))
        self.assertTrue(decision.accepted, decision.reason)

    def test_quote_missing_from_window_is_rejected_naming_lines(self):
        decision = self.store.add(self.candidate(line=1, quote="seven = 7"))
        self.assertRejected(decision, "lines 1-3", "config.py")

    def test_quote_edge_of_window_is_accepted(self):
        self.assertTrue(self.store.add(self.candidate(line=5, quote="three = 3")).accepted)
        other = FindingStore(self.repository, self.settings.agent)
        self.assertRejected(other.add(self.candidate(line=6, quote="setting = True")))

    def test_strict_rule_needs_no_quote(self):
        self.repository._read_cache["config.py"] = 'API_TOKEN = "local_test_token_value_12345"\n'
        candidate = self.candidate(
            rule_id="SEC-001", category="security", severity="critical", evidence="API_TOKEN = [REDACTED]"
        )
        del candidate["quote"]
        self.assertTrue(self.store.add(candidate).accepted)

    def test_absence_needs_a_complete_listing(self):
        absent = self.candidate(rule_id="CI-003", category="ci_cd", severity="medium", file=None, line=None)
        self.assertRejected(self.store.add(absent), "list_files", "tree_truncated")
        self.repository.list_files()
        self.assertTrue(self.store.add(absent).accepted)

    def test_absence_is_rejected_after_a_truncated_listing(self):
        self.repository.source.truncated = True
        self.repository.list_files()
        absent = self.candidate(rule_id="CI-003", category="ci_cd", severity="medium", file=None, line=None)
        self.assertRejected(self.store.add(absent), "tree_truncated")

    def test_path_only_finding_needs_the_path_in_the_listing(self):
        listed = self.candidate(rule_id="CFG-002", category="configuration", file="listed_only.py", line=None)
        self.assertTrue(self.store.add(listed).accepted)
        missing = dict(listed, file="nope.py")
        self.assertRejected(self.store.add(missing), "nope.py")


class FalsePositiveGuardTests(StoreCase):
    def sec(self, path, **overrides):
        values = dict(
            rule_id="SEC-004", category="security", severity="medium", file=path, line=1, quote="="
        )
        values.update(overrides)
        return self.candidate(**values)

    def test_fixture_path_rejected_for_sec_rules(self):
        decision = self.store.add(self.sec("tests/fixtures/app/settings.py"))
        self.assertRejected(decision, "fixture")

    def test_fixture_path_allowed_for_known_token_format(self):
        decision = self.store.add(
            self.sec("tests/fixtures/app/settings.py", rule_id="SEC-002", severity="critical")
        )
        self.assertTrue(decision.accepted, decision.reason)

    def test_fixture_path_allowed_for_non_sec_rules(self):
        decision = self.candidate(file="tests/fixtures/app/settings.py", quote="API_TOKEN")
        self.assertTrue(self.store.add(decision).accepted)

    def test_template_file_rejected(self):
        decision = self.store.add(self.candidate(file=".env.example", quote="KEY"))
        self.assertRejected(decision, "template")

    def test_non_template_file_with_similar_name_allowed(self):
        self.repository._read_cache["examples.py"] = "KEY = 1\n"
        self.repository._inspected_paths.add("examples.py")
        self.assertTrue(self.store.add(self.candidate(file="examples.py", quote="KEY")).accepted)

    def test_vendored_path_rejected(self):
        self.assertRejected(self.store.add(self.candidate(file="vendor/lib/x.py", quote="x")), "vendored")

    def test_similar_directory_name_is_not_vendored(self):
        self.assertTrue(self.store.add(self.candidate(file="src/vendorish/y.py", quote="y")).accepted)

    def test_nested_vendor_directory_rejected(self):
        self.repository._read_cache["app/node_modules/z.js"] = "z\n"
        self.repository._inspected_paths.add("app/node_modules/z.js")
        decision = self.store.add(self.candidate(file="app/node_modules/z.js", quote="z"))
        self.assertRejected(decision, "vendored")

    def test_documentation_rejected_for_sec_and_cfg(self):
        for path in ("README.md", "docs/guide.md"):
            decision = FindingStore(self.repository, self.settings.agent).add(
                self.sec(path, quote="=")
            )
            self.assertRejected(decision, "documentation")

    def test_documentation_allowed_for_other_rules(self):
        self.assertTrue(self.store.add(self.candidate(file="README.md", quote="password")).accepted)

    def test_remote_setting_claim_rejected(self):
        decision = self.store.add(
            self.candidate(description="The main branch has no Branch Protection rules")
        )
        self.assertRejected(decision, "remote settings can't be inferred from files")

    def test_required_reviewers_via_codeowners_is_not_a_remote_claim(self):
        decision = self.store.add(
            self.candidate(
                rule_id="COL-002",
                category="collaboration",
                description="No required reviewers are defined via CODEOWNERS",
            )
        )
        self.assertTrue(decision.accepted, decision.reason)

    def test_negated_branch_protection_claim_rejected(self):
        decision = self.store.add(self.candidate(description="branch protection is not enabled"))
        self.assertRejected(decision, "remote settings can't be inferred from files")

    def test_ordinary_description_allowed(self):
        self.assertTrue(self.store.add(self.candidate(description="Bare except hides errors")).accepted)


class DuplicateTests(StoreCase):
    def test_same_location_different_rule_is_a_duplicate(self):
        self.assertTrue(self.store.add(self.candidate(rule_id="QUA-002")).accepted)
        decision = self.store.add(self.candidate(rule_id="QUA-099"))
        self.assertTrue(decision.duplicate)
        self.assertFalse(decision.accepted)
        self.assertIn("QUA-002 at config.py:1", decision.reason)
        self.assertEqual(len(self.store.findings), 1)

    def test_different_line_is_not_a_duplicate(self):
        self.store.add(self.candidate())
        self.assertTrue(self.store.add(self.candidate(line=2, quote="other = 2")).accepted)

    def test_higher_severity_replaces_the_kept_finding(self):
        sec = dict(category="security", file="config.py")
        self.store.add(self.candidate(rule_id="SEC-099", severity="low", **sec))
        decision = self.store.add(self.candidate(rule_id="SEC-004", severity="medium", **sec))
        self.assertTrue(decision.duplicate)
        self.assertEqual([(f.rule_id, f.severity) for f in self.store.findings], [("SEC-004", "medium")])

    def test_lower_severity_does_not_replace(self):
        sec = dict(category="security", file="config.py")
        self.store.add(self.candidate(rule_id="SEC-004", severity="medium", **sec))
        self.store.add(self.candidate(rule_id="SEC-099", severity="low", **sec))
        self.assertEqual([f.rule_id for f in self.store.findings], ["SEC-004"])

    def test_same_absence_rule_is_a_duplicate(self):
        self.repository.list_files()
        absent = dict(file=None, line=None, category="ci_cd", severity="medium")
        self.assertTrue(self.store.add(self.candidate(rule_id="CI-003", **absent)).accepted)
        decision = self.store.add(self.candidate(rule_id="CI-003", **absent))
        self.assertTrue(decision.duplicate)

    def test_distinct_absence_rules_in_one_family_are_kept(self):
        self.repository.list_files()
        absent = dict(file=None, line=None, category="repository_hygiene", severity="low")
        self.assertTrue(self.store.add(self.candidate(rule_id="HYG-001", **absent)).accepted)
        self.assertTrue(self.store.add(self.candidate(rule_id="HYG-002", **absent)).accepted)

    def test_absence_findings_in_other_families_are_kept(self):
        self.repository.list_files()
        self.store.add(self.candidate(rule_id="CI-003", file=None, line=None, category="ci_cd", severity="medium"))
        other = self.candidate(
            rule_id="HYG-001", file=None, line=None, category="repository_hygiene", severity="low"
        )
        self.assertTrue(self.store.add(other).accepted)


class LongLineTests(unittest.TestCase):
    def test_sec001_skips_lines_beyond_redaction_limit(self):
        from auditor.checks import audit_inspected_files
        from auditor.redaction import packaged_secret_rules

        limit = packaged_secret_rules().max_line_chars
        long_line = 'API_TOKEN = "local_test_token_value_12345"' + " " * limit + "\n"
        short_line = 'API_TOKEN = "local_test_token_value_12345"\n'
        found = audit_inspected_files({"a.py": long_line + short_line}, rule_ids=["SEC-001"])["findings"]
        self.assertEqual([f["line"] for f in found], [2])


if __name__ == "__main__":
    unittest.main()
