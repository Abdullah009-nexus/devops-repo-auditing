import dataclasses
import unittest
from unittest.mock import patch

from auditor.findings import FindingStore
from auditor.repository import GitHubApiSource, RepositoryMetadata
from tests.support import default_settings
from tools import GitHubRepository, RepositoryAddress


class FindingStoreTests(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings()
        address = RepositoryAddress.parse("https://github.com/example/project", self.settings.github)
        tree = {"tree": [{"type": "blob", "path": "config.py", "size": 10}]}
        with patch("auditor.repository._request_json", return_value=tree):
            source = GitHubApiSource(address, RepositoryMetadata("main", 1), self.settings)
        self.repository = GitHubRepository(address, self.settings, source=source)
        self.repository._read_cache = {"config.py": "setting = True\nother = 2\n"}
        self.repository._inspected_paths = {"config.py"}
        self.store = FindingStore(self.repository, self.settings.agent)

    def candidate(self, **overrides):
        values = {
            "rule_id": "SEC-004",
            "category": "security",
            "severity": "medium",
            "file": "config.py",
            "line": 1,
            "description": "Something is wrong",
            "evidence": "setting = True",
            "recommendation": "Change it",
            "quote": "setting = True",
        }
        values.update(overrides)
        return values

    def test_valid_finding_is_accepted(self):
        decision = self.store.add(self.candidate())
        self.assertTrue(decision.accepted)
        self.assertIsNone(decision.reason)
        self.assertEqual(self.store.findings, [decision.finding])

    def test_category_is_normalised(self):
        decision = self.store.add(
            self.candidate(rule_id="CI-099", category="CI-CD", severity="low")
        )
        self.assertEqual(decision.finding.category, "ci_cd")

    def test_unknown_category_is_rejected_with_options(self):
        decision = self.store.add(self.candidate(category="performance"))
        self.assertFalse(decision.accepted)
        self.assertIn("repository_hygiene", decision.reason)

    def test_rejection_never_raises_for_malformed_input(self):
        for bad in ("text", None, 5, [], {}):
            self.assertFalse(self.store.add(bad).accepted)

    def test_missing_fields_are_named(self):
        candidate = self.candidate()
        del candidate["recommendation"]
        decision = self.store.add(candidate)
        self.assertFalse(decision.accepted)
        self.assertIn("recommendation", decision.reason)

    def test_bad_types_and_severity_are_rejected(self):
        cases = (
            ("severity", "urgent"),
            ("line", True),
            ("line", 0),
            ("line", "1"),
            ("file", 3),
            ("description", " "),
        )
        for field, value in cases:
            self.assertFalse(self.store.add(self.candidate(**{field: value})).accepted, field)

    def test_line_without_file_is_rejected(self):
        decision = self.store.add(self.candidate(file=None, line=1))
        self.assertFalse(decision.accepted)
        self.assertIn("without a file", decision.reason)

    def test_unread_file_tells_model_to_read_it(self):
        decision = self.store.add(self.candidate(file="unknown.py", line=None))
        self.assertFalse(decision.accepted)
        self.assertIn("call read_file on unknown.py first", decision.reason)

    def test_line_beyond_read_content_is_rejected(self):
        decision = self.store.add(self.candidate(line=40))
        self.assertFalse(decision.accepted)
        self.assertIn("line 40", decision.reason)

    def test_invalid_path_is_rejected_not_raised(self):
        decision = self.store.add(self.candidate(file="../etc/passwd"))
        self.assertFalse(decision.accepted)

    def test_absence_finding_with_null_file_and_line_is_accepted(self):
        self.repository.list_files()
        decision = self.store.add(
            self.candidate(file=None, line=None, rule_id="SEC-099", severity="low")
        )
        self.assertTrue(decision.accepted)

    def test_audit_rule_requires_matching_evidence(self):
        self.repository._read_cache["config.py"] = 'API_TOKEN = "local_test_token_value_12345"\n'
        good = self.candidate(rule_id="SEC-001", severity="critical", evidence="API_TOKEN = [REDACTED]")
        self.assertTrue(self.store.add(good).accepted)
        other = FindingStore(self.repository, self.settings.agent)
        decision = other.add(dict(good, evidence="This file has a security problem."))
        self.assertFalse(decision.accepted)
        self.assertIn("deterministic audit rule", decision.reason)

    def test_audit_evidence_matches_after_whitespace_normalisation(self):
        self.repository._read_cache["config.py"] = 'API_TOKEN = "local_test_token_value_12345"\n'
        decision = self.store.add(
            self.candidate(rule_id="SEC-001", severity="critical", evidence="  API_TOKEN   =  [REDACTED] ")
        )
        self.assertTrue(decision.accepted)

    def test_audit_rule_without_location_is_rejected(self):
        decision = self.store.add(self.candidate(rule_id="SEC-001", file=None, line=None))
        self.assertFalse(decision.accepted)

    def test_duplicates_are_flagged_and_kept_once(self):
        self.store.add(self.candidate())
        decision = self.store.add(self.candidate(description="Reworded"))
        self.assertFalse(decision.accepted)
        self.assertTrue(decision.duplicate)
        self.assertEqual(len(self.store.findings), 1)

    def test_finding_limit(self):
        agent = dataclasses.replace(self.settings.agent, max_report_findings=1)
        store = FindingStore(self.repository, agent)
        self.assertTrue(store.add(self.candidate()).accepted)
        decision = store.add(
            self.candidate(rule_id="SEC-099", severity="low", line=2, quote="other = 2")
        )
        self.assertFalse(decision.accepted)
        self.assertIn("maximum of 1", decision.reason)

    def test_text_fields_are_redacted(self):
        decision = self.store.add(self.candidate(evidence="TOKEN = supersecretvalue"))
        self.assertNotIn("supersecretvalue", decision.finding.evidence)

    def test_quote_stays_on_the_decision_only(self):
        decision = self.store.add(self.candidate(quote="setting = True"))
        self.assertEqual(decision.quote, "setting = True")
        self.assertNotIn("quote", decision.finding.to_report_dict())
        self.assertFalse(hasattr(decision.finding, "quote"))

    def test_findings_keep_acceptance_order(self):
        self.store.add(self.candidate(rule_id="SEC-004"))
        self.store.add(self.candidate(rule_id="SEC-099", severity="low", line=2, quote="other = 2"))
        self.assertEqual([f.rule_id for f in self.store.findings], ["SEC-004", "SEC-099"])


if __name__ == "__main__":
    unittest.main()
