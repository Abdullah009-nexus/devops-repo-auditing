import copy
import re
import unittest

from auditor import rules
from auditor.schema import CATEGORIES
from auditor.settings import ConfigError


def document():
    return copy.deepcopy(
        {
            "rules": [
                {
                    "id": "SEC-001",
                    "category": "security",
                    "default_severity": "critical",
                    "title": "Hard-coded credential",
                    "evidence": "the line",
                    "recommendation": "use env",
                    "acceptable_alternatives": ["env lookups"],
                    "detector": "demo",
                }
            ],
            "detectors": {"demo": {}},
            "audit": {"strict_rule_ids": ["SEC-001"], "prompt_alternatives_per_rule": 3},
        }
    )


class CatalogTests(unittest.TestCase):
    def test_packaged_catalog_is_valid_and_complete(self):
        catalog = rules.load_catalog()
        ids = [rule.id for rule in catalog.rules]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 40)
        for category in CATEGORIES:
            self.assertTrue(
                any(rule.category == category and rule.id.endswith("-099") for rule in catalog.rules),
                category,
            )
        self.assertEqual(rules.get_rule("SEC-001").default_severity, "critical")

    def test_known_rule(self):
        self.assertTrue(rules.known_rule("CI-004"))
        self.assertFalse(rules.known_rule("XYZ-123"))

    def test_prompt_table_lists_every_id_and_stays_small(self):
        text = rules.render_catalog_for_prompt()
        for rule in rules.load_catalog().rules:
            self.assertIn(rule.id, text)
        self.assertLess(len(text), 6000)

    def test_valid_document_parses(self):
        self.assertEqual(rules.parse_catalog(document()).rules[0].id, "SEC-001")

    def test_rejects_bad_documents(self):
        def broken(change):
            doc = document()
            change(doc)
            return doc

        cases = {
            "duplicate": lambda d: d["rules"].append(dict(d["rules"][0])),
            "id format": lambda d: d["rules"][0].update(id="sec-1"),
            "category": lambda d: d["rules"][0].update(category="nope"),
            "severity": lambda d: d["rules"][0].update(default_severity="urgent"),
            "recommendation": lambda d: d["rules"][0].update(recommendation=" "),
            "detector": lambda d: d["rules"][0].update(detector="missing"),
            "unknown field": lambda d: d["rules"][0].update(extra=1),
            "strict id": lambda d: d["audit"].update(strict_rule_ids=["ZZZ-999"]),
        }
        for name, change in cases.items():
            with self.subTest(name), self.assertRaises(ConfigError):
                rules.parse_catalog(broken(change))

    def test_ids_match_the_report_format(self):
        for rule in rules.load_catalog().rules:
            self.assertRegex(rule.id, re.compile(r"^[A-Z]{2,3}-\d{3}$"))


if __name__ == "__main__":
    unittest.main()
