import tomllib
import unittest
from pathlib import Path
from string import Template

from auditor.prompts import PROMPT_DIR, load_prompt

ROOT = Path(__file__).resolve().parents[1]


class PromptTests(unittest.TestCase):
    def render_system(self):
        return load_prompt(
            "system",
            repository_url="https://github.com/o/r",
            rule_catalog="",
            max_tool_calls=32,
            max_turns=8,
        )

    def test_system_placeholders_render(self):
        text = self.render_system()
        self.assertIn("https://github.com/o/r", text)
        self.assertIn("32 tool calls", text)
        self.assertIn("8 model turns", text)

    def test_no_placeholder_is_left_unrendered(self):
        self.assertNotRegex(self.render_system(), r"\$[A-Za-z_{]")
        self.assertNotRegex(load_prompt("finalize"), r"\$[A-Za-z_{]")

    def test_system_prompt_ends_with_finding_tools_not_json(self):
        text = self.render_system()
        self.assertIn("report_finding", text)
        self.assertIn("finish_audit", text)
        self.assertNotIn('"violations":[', text)

    def test_missing_value_raises(self):
        with self.assertRaises(KeyError):
            load_prompt("system", repository_url="x")

    def test_system_prompt_declares_its_placeholders(self):
        text = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
        self.assertEqual(
            set(Template(text).get_identifiers()),
            {"repository_url", "rule_catalog", "max_tool_calls", "max_turns"},
        )

    def test_finalize_prompt_mentions_evidence(self):
        self.assertIn("evidence", load_prompt("finalize"))

    def test_package_data_lists_prompts(self):
        with (ROOT / "pyproject.toml").open("rb") as handle:
            package_data = tomllib.load(handle)["tool"]["setuptools"]["package-data"]
        self.assertIn("prompts/*.txt", package_data["auditor"])


if __name__ == "__main__":
    unittest.main()
