import os
import re
import shutil
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

from auditor.redaction import redact_secrets


ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_ASSIGNMENT = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*)=(?P<value>.*)$")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


def _is_placeholder(value: str) -> bool:
    return value == "" or "your-" in value or value.startswith("https://")


class ImportLayoutTests(unittest.TestCase):
    def test_modules_import_in_a_fresh_interpreter(self):
        for module in ("auditor.redaction", "auditor.checks", "auditor.settings", "tools", "agent"):
            with self.subTest(module=module):
                result = _run([sys.executable, "-B", "-c", f"import {module}"])
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_checks_do_not_import_tools_or_agent(self):
        script = (
            "import sys, auditor.checks; "
            "print(','.join(m for m in ('tools', 'agent') if m in sys.modules))"
        )
        result = _run([sys.executable, "-B", "-c", script])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


class ProjectFilesTests(unittest.TestCase):
    def test_legacy_audit_module_is_gone_from_repo_root(self):
        self.assertFalse((ROOT / "audit.py").exists())
        self.assertFalse((ROOT / "audit").exists())
        self.assertTrue((ROOT / "auditor" / "checks.py").is_file())

    def test_requirements_file_lists_no_packages(self):
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        packages = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        self.assertEqual(packages, [])


class EnvExampleTests(unittest.TestCase):
    def setUp(self):
        self.lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        self.assignments = {}
        for line in self.lines:
            match = PLACEHOLDER_ASSIGNMENT.match(line.strip())
            if match:
                self.assignments[match["key"]] = match["value"]

    def test_declares_the_documented_variables(self):
        for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "GITHUB_TOKEN"):
            self.assertIn(key, self.assignments)

    def test_values_are_placeholders_only(self):
        for key, value in self.assignments.items():
            with self.subTest(key=key):
                self.assertTrue(_is_placeholder(value), f"{key} does not look like a placeholder")

    def test_redaction_changes_only_placeholder_lines(self):
        redacted = redact_secrets("\n".join(self.lines)).splitlines()
        self.assertEqual(len(redacted), len(self.lines))
        for original, cleaned in zip(self.lines, redacted):
            if original == cleaned:
                continue
            with self.subTest(line=original):
                match = PLACEHOLDER_ASSIGNMENT.match(original.strip())
                self.assertIsNotNone(match)
                self.assertTrue(_is_placeholder(match["value"]))


class GitignoreTests(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / ".gitignore").read_text(encoding="utf-8")

    def _require_git(self):
        if shutil.which("git") is None or not (ROOT / ".git").exists():
            self.skipTest("git is not available")

    def test_ignores_local_env_and_run_reports(self):
        self._require_git()
        for path in (".env", "report.json"):
            with self.subTest(path=path):
                # --no-index so tracked state doesn't matter; -v so the match must come
                # from .gitignore and not from a personal global excludes file.
                result = _run(["git", "check-ignore", "--no-index", "-v", path])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.startswith(".gitignore:"), result.stdout)

    def test_keeps_env_template_and_sample_report_tracked(self):
        self._require_git()
        for path in (".env.example", "sample_report.json"):
            with self.subTest(path=path):
                result = _run(["git", "check-ignore", "--no-index", "-q", path])
                self.assertEqual(result.returncode, 1, f"{path} is ignored")

    def test_does_not_publish_the_local_only_policy(self):
        # Working notes are excluded locally, not here: a blanket rule in the
        # public .gitignore would also hide README.md from contributors.
        for pattern in ("*.md", "*.pdf"):
            with self.subTest(pattern=pattern):
                self.assertNotIn(pattern, self.text)


class PyprojectTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / "pyproject.toml").open("rb") as handle:
            self.project = tomllib.load(handle)["project"]

    def test_requires_python_311_without_dependencies(self):
        self.assertEqual(self.project["requires-python"], ">=3.11")
        self.assertEqual(self.project["dependencies"], [])

    def test_console_script_points_to_agent_main(self):
        self.assertIn("agent:main", self.project["scripts"].values())

    def test_packaged_defaults_are_shipped_with_the_package(self):
        with (ROOT / "pyproject.toml").open("rb") as handle:
            package_data = tomllib.load(handle)["tool"]["setuptools"]["package-data"]
        self.assertIn("config/*.toml", package_data["auditor"])
        self.assertTrue((ROOT / "auditor" / "config" / "defaults.toml").is_file())


if __name__ == "__main__":
    unittest.main()
