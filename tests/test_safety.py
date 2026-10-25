"""Project-wide safety guards: subprocess use (G04) and secret containment (G05)."""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auditor.findings import FINISH_AUDIT
from auditor.loop import AuditAgent
from auditor.report import write_report
from auditor.repository import LocalDirSource, git_environment
from auditor.trace import Tracer
from tests.support import default_settings
from tests.test_loop import FakeClient, call, reply
from tools import GitHubRepository, RepositoryAddress

ROOT = Path(__file__).resolve().parent.parent
SUBPROCESS_OWNER = ROOT / "auditor" / "repository.py"
BANNED_CALLS = {"eval", "exec", "system", "popen"}
ALLOWED_SUBPROCESS = {"run", "Popen"}
# Built from parts so no credential-shaped literal sits in the source.
FAKE_TOKEN = "ghp_" + "SafetyGuard" + "Token" + "0123456789abcdefABCD"
FAKE_KEY = "sk-" + "SafetyGuard" + "Key" + "0123456789abcdefABCD"


def sources():
    paths = [ROOT / "tools.py", ROOT / "agent.py", *sorted((ROOT / "auditor").rglob("*.py"))]
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in paths]


def callee(node):
    """(name, owner) of a call target such as subprocess.run or eval."""
    if isinstance(node, ast.Attribute):
        return node.attr, node.value.id if isinstance(node.value, ast.Name) else None
    if isinstance(node, ast.Name):
        return node.id, None
    return None, None


class SubprocessGuardTests(unittest.TestCase):
    def test_no_shell_execution_or_dynamic_code_anywhere(self):
        for path, tree in sources():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name, _ = callee(node.func)
                self.assertNotIn(name, BANNED_CALLS, f"{path.name}:{node.lineno}")
                for keyword in node.keywords:
                    if keyword.arg == "shell":
                        self.assertFalse(
                            isinstance(keyword.value, ast.Constant) and keyword.value.value,
                            f"shell=True at {path.name}:{node.lineno}",
                        )

    def test_subprocess_is_called_only_by_the_repository_module_with_list_argv(self):
        used = 0
        for path, tree in sources():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name, owner = callee(node.func)
                # Exception classes such as TimeoutExpired are not process launches.
                if owner != "subprocess" or name[:1].isupper() and name != "Popen":
                    continue
                used += 1
                where = f"{path.name}:{node.lineno}"
                self.assertEqual(path, SUBPROCESS_OWNER, where)
                self.assertIn(name, ALLOWED_SUBPROCESS, where)
                self.assertTrue(node.args, f"argv must be positional at {where}")
                self.assertNotIsInstance(node.args[0], (ast.Constant, ast.JoinedStr), where)
        self.assertGreater(used, 0)

    def test_subprocess_is_not_imported_elsewhere(self):
        for path, tree in sources():
            if path == SUBPROCESS_OWNER:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = {alias.name for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    names = {node.module}
                else:
                    continue
                self.assertNotIn("subprocess", names, path.name)


class SecretContainmentTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN, "OPENAI_API_KEY": FAKE_KEY})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = default_settings()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_git_environment_carries_the_token_only_inside_the_auth_header(self):
        environment = git_environment(self.settings.github.clone_root)
        for key, value in environment.items():
            self.assertNotIn(FAKE_KEY, value, key)
            self.assertNotIn(FAKE_TOKEN, value, key)
        self.assertTrue(environment["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic "))

    def test_tool_output_trace_and_report_never_contain_the_secrets(self):
        repo = self.directory / "repo"
        repo.mkdir()
        (repo / "app.py").write_text(
            f'API_TOKEN = "{FAKE_TOKEN}"\nOPENAI_KEY = "{FAKE_KEY}"\n', encoding="utf-8"
        )
        trace = self.directory / "run.jsonl"
        output = self.directory / "report.json"
        address = RepositoryAddress.parse(
            "https://github.com/example/project", self.settings.github
        )
        source = LocalDirSource(repo, self.settings, "main")
        client = FakeClient(
            reply(call("list_files", {}, "a")),
            reply(call("read_file", {"path": "app.py"}, "b")),
            reply(call("search_code", {"pattern": "API_TOKEN"}, "c")),
            reply(call("audit_inspected_files", {}, "e")),
            reply(call(FINISH_AUDIT, {"summary": "done"}, "f")),
        )
        with GitHubRepository(address, self.settings, source) as repository:
            outputs = [
                json.dumps(repository.read_file("app.py")),
                json.dumps(repository.search_code("API_TOKEN")),
                json.dumps(repository.search_code(FAKE_TOKEN)),
                json.dumps(repository.search_code("KEY", regex=True)),
            ]
            report = AuditAgent(repository, self.settings, client, Tracer(trace, 500)).run()
        write_report(str(output), report)
        outputs.append(trace.read_text(encoding="utf-8"))
        outputs.append(output.read_text(encoding="utf-8"))
        outputs.extend(json.dumps(request) for request, _ in client.requests)
        for text in outputs:
            self.assertNotIn(FAKE_TOKEN, text)
            self.assertNotIn(FAKE_KEY, text)


if __name__ == "__main__":
    unittest.main()
