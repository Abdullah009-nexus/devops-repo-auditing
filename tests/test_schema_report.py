import json
import tempfile
import unittest
from pathlib import Path

from auditor.report import ReportError, build_report, validate_report_shape, write_report
from auditor.schema import CATEGORIES, REQUIRED_FIELDS, SEVERITIES, Finding, normalize_category


def make_finding(**overrides):
    values = {
        "rule_id": "CUSTOM-001",
        "category": "security",
        "severity": "high",
        "file": "app.py",
        "line": 3,
        "description": "desc",
        "evidence": "x = 1",
        "recommendation": "fix it",
    }
    values.update(overrides)
    return Finding(**values)


class SchemaTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(SEVERITIES, ("critical", "high", "medium", "low"))
        self.assertEqual(len(REQUIRED_FIELDS), 8)
        self.assertEqual(len(CATEGORIES), 9)

    def test_category_variants_are_normalised(self):
        for spelling in ("CI-CD", "ci cd", "Ci_Cd", " ci_cd "):
            self.assertEqual(normalize_category(spelling), "ci_cd")
        self.assertEqual(normalize_category("Build Release"), "build_release")
        self.assertEqual(normalize_category("Repository-Hygiene"), "repository_hygiene")

    def test_unknown_category_is_none(self):
        self.assertIsNone(normalize_category("performance"))

    def test_to_report_dict_has_exactly_the_contract_fields(self):
        self.assertEqual(list(make_finding().to_report_dict()), list(REQUIRED_FIELDS))

    def test_finding_is_frozen(self):
        with self.assertRaises(Exception):
            make_finding().file = "other.py"


class ReportTests(unittest.TestCase):
    URL = "https://github.com/o/r"

    def test_build_report_has_two_keys(self):
        report = build_report(self.URL, [make_finding()])
        self.assertEqual(set(report), {"repository", "violations"})
        self.assertEqual(report["violations"][0]["file"], "app.py")
        validate_report_shape(report)

    def test_empty_violations_are_valid(self):
        validate_report_shape(build_report(self.URL, []))

    def test_extra_top_level_key_is_rejected(self):
        report = build_report(self.URL, [])
        report["summary"] = "x"
        with self.assertRaises(ReportError):
            validate_report_shape(report)

    def test_extra_violation_field_is_rejected(self):
        report = build_report(self.URL, [make_finding()])
        report["violations"][0]["quote"] = "x = 1"
        with self.assertRaises(ReportError):
            validate_report_shape(report)

    def test_bad_values_are_rejected(self):
        cases = (("severity", "urgent"), ("line", 0), ("line", True), ("file", 4), ("evidence", " "))
        for field, value in cases:
            report = build_report(self.URL, [make_finding()])
            report["violations"][0][field] = value
            with self.assertRaises(ReportError, msg=field):
                validate_report_shape(report)

    def test_null_file_and_line_are_valid(self):
        validate_report_shape(build_report(self.URL, [make_finding(file=None, line=None)]))

    def test_report_error_is_a_value_error(self):
        self.assertTrue(issubclass(ReportError, ValueError))

    def test_write_report_is_atomic_and_creates_parents(self):
        report = build_report(self.URL, [])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.json"
            write_report(output, report)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertFalse(output.with_name("report.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
