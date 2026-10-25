import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Tunables belong in auditor/config/defaults.toml; every module reads them from Settings.
# schema.py holds the PDF contract constants and is the only exempt module.
CONTRACT_MODULE = "schema.py"
# Fixed by an external protocol, not a choice of ours, so they are not tunables.
PROTOCOL_CONSTANTS = {
    ("llm_client.py", "_RETRYABLE_STATUS"),  # HTTP statuses that signal a transient failure
    ("repository.py", "_OBJECT_ID_LENGTHS"),  # git SHA-1 / SHA-256 hex lengths
    ("repository.py", "_BYTES_PER_MB"),  # unit conversion
    ("settings.py", "_SHA_HEX_LENGTH"),  # full SHA-1 length
    ("settings.py", "_GITHUB_MAX_PER_PAGE"),  # GitHub API page-size ceiling
}
SETTINGS_DRIVEN_MODULES = (
    "agent.py",
    "tools.py",
    *(
        f"auditor/{path.name}"
        for path in sorted((ROOT / "auditor").glob("*.py"))
        if path.name != CONTRACT_MODULE
    ),
)


def numeric_constants(source: str) -> list[str]:
    """Names of module-level UPPER_CASE assignments whose value contains a number."""
    found = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if not any(name.isupper() for name in names):
            continue
        # bool is an int subclass, but a flag is not a tunable limit.
        if any(
            isinstance(item, ast.Constant)
            and isinstance(item.value, (int, float))
            and not isinstance(item.value, bool)
            for item in ast.walk(value)
        ):
            found.extend(name for name in names if name.isupper())
    return found


class NumericConstantGuardTests(unittest.TestCase):
    def test_detects_numeric_constants_and_ignores_others(self):
        source = (
            "MAX_TURNS = 8\n"
            "LIMIT: int = 256 * 1024\n"
            "TIMEOUT = -1.5\n"
            "SUFFIXES = ('.yml', 3)\n"
            "PROMPT = 'text'\n"
            "ENABLED = True\n"
            "local_value = 10\n"
            "def helper():\n"
            "    INNER = 5\n"
        )
        self.assertEqual(numeric_constants(source), ["MAX_TURNS", "LIMIT", "TIMEOUT", "SUFFIXES"])

    def test_settings_driven_modules_have_no_numeric_constants(self):
        for name in SETTINGS_DRIVEN_MODULES:
            with self.subTest(module=name):
                source = (ROOT / name).read_text(encoding="utf-8")
                module = Path(name).name
                found = [c for c in numeric_constants(source) if (module, c) not in PROTOCOL_CONSTANTS]
                self.assertEqual(found, [])

    def test_guard_covers_every_auditor_module(self):
        self.assertIn("auditor/loop.py", SETTINGS_DRIVEN_MODULES)
        self.assertIn("auditor/repository.py", SETTINGS_DRIVEN_MODULES)
        self.assertNotIn("auditor/schema.py", SETTINGS_DRIVEN_MODULES)


def _words(name: str) -> set[str]:
    return {name, name.replace("-", "_"), name.replace("-", " ")}


def seeded_names() -> set[str]:
    """Eval case ids and fixture directory names; none may leak into the product."""
    names: set[str] = set()
    for base in (ROOT / "eval" / "cases", ROOT / "tests" / "fixtures"):
        names.update(path.name for path in base.iterdir() if path.is_dir())
    return names


def expected_texts() -> set[str]:
    """Descriptive text from eval expectations, if any case carries it."""
    texts: set[str] = set()
    for path in (ROOT / "eval" / "cases").glob("*/expected.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for violation in data.get("violations", []):
            for key in ("description", "evidence", "recommendation"):
                if len(violation.get(key) or "") >= 12:
                    texts.add(violation[key])
    return texts


class NoHardCodedAnswersTests(unittest.TestCase):
    def product_files(self) -> list[Path]:
        files = [ROOT / "agent.py", ROOT / "tools.py"]
        files += sorted((ROOT / "auditor").glob("*.py"))
        files += sorted((ROOT / "auditor" / "prompts").glob("*.txt"))
        files += sorted((ROOT / "auditor" / "config").glob("*.toml"))
        return files

    def test_name_lists_are_not_empty(self):
        names = seeded_names()
        self.assertIn("a1-hardcoded-db-password", names)
        self.assertIn("audit-defective", names)

    def test_no_eval_or_fixture_name_appears_in_product_files(self):
        for path in self.product_files():
            text = path.read_text(encoding="utf-8").lower()
            for name in seeded_names():
                for variant in _words(name.lower()):
                    with self.subTest(file=path.name, name=variant):
                        self.assertNotIn(variant, text)

    def test_no_expected_finding_text_appears_in_product_files(self):
        for path in self.product_files():
            text = path.read_text(encoding="utf-8")
            for expected in expected_texts():
                with self.subTest(file=path.name):
                    self.assertNotIn(expected, text)


if __name__ == "__main__":
    unittest.main()
