import unittest
from pathlib import Path

from audit import audit_inspected_files, supports_finding


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def read_fixture(name):
    root = FIXTURE_ROOT / name
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    }


class AuditRuleTests(unittest.TestCase):
    def test_defective_fixture_reports_supported_findings_without_secret_values(self):
        fixture = read_fixture("audit-defective")
        results = audit_inspected_files(fixture)
        findings = results["findings"]
        expected = {
            ("SEC-001", "config.py", 1),
            ("CI-001", ".github/workflows/unsafe.yml", 5),
            ("CI-002", ".github/workflows/unsafe.yml", 12),
        }
        actual = {
            (finding["rule_id"], finding["file"], finding["line"])
            for finding in findings
        }
        self.assertEqual(actual, expected)
        serialized = repr(findings)
        self.assertNotIn("local_test_token_value_12345", serialized)
        for finding in findings:
            self.assertIsInstance(finding["line"], int)
            self.assertTrue(finding["evidence"])
            self.assertTrue(
                supports_finding(
                    finding["rule_id"],
                    finding["file"],
                    finding["line"],
                    finding["evidence"],
                    fixture,
                )
            )
        precision = len(actual & expected) / len(actual)
        recall = len(actual & expected) / len(expected)
        self.assertEqual((precision, recall), (1.0, 1.0))

    def test_healthy_fixture_has_no_findings(self):
        results = audit_inspected_files(read_fixture("audit-healthy"))
        self.assertEqual(results["findings"], [])
        self.assertEqual(results["checked_files"], 2)

    def test_empty_or_uninspected_files_are_not_treated_as_findings(self):
        results = audit_inspected_files({})
        self.assertEqual(results["findings"], [])
        self.assertEqual(results["checked_files"], 0)
        self.assertIn("partial evidence", results["scope_note"])

    def test_rule_evidence_must_match_the_inspected_source(self):
        fixture = read_fixture("audit-defective")
        finding = audit_inspected_files(fixture)["findings"][0]
        self.assertTrue(
            supports_finding(
                finding["rule_id"],
                finding["file"],
                finding["line"],
                finding["evidence"],
                fixture,
            )
        )
        self.assertFalse(
            supports_finding(
                finding["rule_id"],
                finding["file"],
                finding["line"],
                "Unsupported claim",
                fixture,
            )
        )


if __name__ == "__main__":
    unittest.main()
