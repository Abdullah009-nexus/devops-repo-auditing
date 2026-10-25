import contextlib
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent
from auditor import settings as settings_module
from auditor.settings import ConfigError, load_env_file, load_settings


def _clean_environment(test: unittest.TestCase) -> None:
    patcher = patch.dict(os.environ)
    patcher.start()
    test.addCleanup(patcher.stop)
    for name in [key for key in os.environ if key.upper().startswith("AUDITOR_")]:
        del os.environ[name]


class _TempDirTestCase(unittest.TestCase):
    def setUp(self):
        _clean_environment(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path


class PrecedenceTests(_TempDirTestCase):
    def test_packaged_defaults_keep_current_limits(self):
        settings = load_settings()
        self.assertEqual(settings.agent.max_turns, 30)
        self.assertEqual(settings.agent.max_tool_calls, 80)
        self.assertEqual(settings.repository.max_file_bytes, 256 * 1024)
        self.assertEqual(settings.git_log.default_commits, 10)

    def test_efficiency_defaults(self):
        settings = load_settings()
        self.assertEqual(settings.context.keep_recent_tool_results, 12)
        self.assertEqual(settings.context.max_stub_args_chars, 200)
        self.assertEqual(settings.agent.ledger_max_paths, 60)
        self.assertEqual(settings.agent.ledger_max_searches, 15)
        self.assertEqual(settings.agent.max_repeat_replies, 3)

    def test_searchable_suffixes_have_no_duplicates(self):
        suffixes = load_settings().search.searchable_suffixes
        self.assertEqual(len(suffixes), len(set(suffixes)))
        self.assertIn(".yml", suffixes)

    def test_config_file_overrides_defaults_key_by_key(self):
        config = self.write("override.toml", "[agent]\nmax_turns = 20\n")
        settings = load_settings(config)
        self.assertEqual(settings.agent.max_turns, 20)
        self.assertEqual(settings.agent.max_tool_calls, 80)

    def test_environment_overrides_config_file(self):
        config = self.write("override.toml", "[agent]\nmax_turns = 20\nmax_tool_calls = 50\n")
        with patch.dict(os.environ, {"AUDITOR_AGENT_MAX_TURNS": "40"}, clear=False):
            settings = load_settings(config)
        self.assertEqual(settings.agent.max_turns, 40)
        self.assertEqual(settings.agent.max_tool_calls, 50)

    def test_environment_reaches_sections_with_underscores(self):
        with patch.dict(os.environ, {"AUDITOR_GIT_LOG_MAX_COMMITS": "25"}, clear=False):
            self.assertEqual(load_settings().git_log.max_commits, 25)


class ValidationTests(_TempDirTestCase):
    def test_missing_default_key_names_the_key(self):
        text = settings_module.DEFAULTS_PATH.read_text(encoding="utf-8")
        trimmed = re.sub(r"^max_turns = .*\n", "", text, count=1, flags=re.MULTILINE)
        self.assertNotEqual(trimmed, text)
        defaults = self.write("defaults.toml", trimmed)
        with (
            patch.object(settings_module, "DEFAULTS_PATH", defaults),
            self.assertRaisesRegex(ConfigError, r"missing setting agent\.max_turns"),
        ):
            load_settings()

    def test_missing_default_section_is_rejected(self):
        text = settings_module.DEFAULTS_PATH.read_text(encoding="utf-8")
        defaults = self.write("defaults.toml", text.split("[llm]")[0])
        with (
            patch.object(settings_module, "DEFAULTS_PATH", defaults),
            self.assertRaisesRegex(ConfigError, r"missing section \[llm\]"),
        ):
            load_settings()

    def test_unknown_key_and_section_are_rejected(self):
        cases = {
            "[agent]\nmax_turn = 3\n": r"agent\.max_turn",
            "[agents]\nmax_turns = 3\n": r"\[agents\]",
            "agent = 3\n": r"\[agent\] must be a table",
        }
        for text, message in cases.items():
            config = self.write("bad.toml", text)
            with self.subTest(text=text), self.assertRaisesRegex(ConfigError, message):
                load_settings(config)

    def test_wrong_types_are_rejected(self):
        cases = {
            '[agent]\nmax_turns = "eight"\n': r"agent\.max_turns must be an integer",
            "[agent]\nmax_turns = true\n": r"agent\.max_turns must be an integer",
            "[agent]\nmax_turns = 8.5\n": r"agent\.max_turns must be an integer",
            "[github]\nuser_agent = 7\n": r"github\.user_agent must be a non-empty string",
            '[search]\nsearchable_suffixes = ".py"\n': r"search\.searchable_suffixes",
            "[search]\nsearchable_suffixes = [1]\n": r"search\.searchable_suffixes",
            '[llm]\nrequest_timeout_seconds = "60"\n': r"llm\.request_timeout_seconds must be a number",
        }
        for text, message in cases.items():
            config = self.write("bad.toml", text)
            with self.subTest(text=text), self.assertRaisesRegex(ConfigError, message):
                load_settings(config)

    def test_out_of_range_values_are_rejected(self):
        cases = {
            "[agent]\nmax_turns = 0\n": r"agent\.max_turns must be at least 1",
            "[agent]\nledger_max_paths = 0\n": r"agent\.ledger_max_paths must be at least 1",
            "[agent]\nmax_repeat_replies = -1\n": r"agent\.max_repeat_replies must be at least 0",
            "[context]\nmax_stub_args_chars = 0\n": r"context\.max_stub_args_chars must be at least 1",
            "[git_log]\nsha_length = 41\n": r"git_log\.sha_length must be at most 40",
            '[search]\nsearchable_suffixes = [".yml", ".yml"]\n': r"duplicate",
            '[github]\napi_root = "http://api.github.com"\n': r"github\.api_root must be an HTTPS URL",
            '[llm]\ndefault_base_url = "https://user:pw@example.com/v1"\n': r"llm\.default_base_url",
        }
        for text, message in cases.items():
            config = self.write("bad.toml", text)
            with self.subTest(text=text), self.assertRaisesRegex(ConfigError, message):
                load_settings(config)

    def test_unreadable_or_invalid_config_file(self):
        with self.assertRaisesRegex(ConfigError, "Cannot read config file"):
            load_settings(self.directory / "missing.toml")
        with self.assertRaisesRegex(ConfigError, "Invalid TOML"):
            load_settings(self.write("broken.toml", "[agent\n"))


class EnvironmentOverrideTests(_TempDirTestCase):
    def test_coerces_int_float_str_and_list(self):
        overrides = {
            "AUDITOR_AGENT_MAX_TURNS": " 12 ",
            "AUDITOR_LLM_REQUEST_TIMEOUT_SECONDS": "7.5",
            "AUDITOR_GITHUB_USER_AGENT": "custom-agent/2.0",
            "AUDITOR_SEARCH_SEARCHABLE_FILENAMES": "dockerfile, makefile ,procfile",
        }
        with patch.dict(os.environ, overrides, clear=False):
            settings = load_settings()
        self.assertEqual(settings.agent.max_turns, 12)
        self.assertEqual(settings.llm.request_timeout_seconds, 7.5)
        self.assertEqual(settings.github.user_agent, "custom-agent/2.0")
        self.assertEqual(
            settings.search.searchable_filenames, ("dockerfile", "makefile", "procfile")
        )

    def test_coerces_boolean_words(self):
        for raw, expected in (("true", True), ("YES", True), ("1", True),
                              ("false", False), ("No", False), ("0", False)):
            with self.subTest(raw=raw):
                self.assertIs(settings_module._coerce_env("AUDITOR_X", raw, bool), expected)
        with self.assertRaisesRegex(ConfigError, "AUDITOR_X must be one of"):
            settings_module._coerce_env("AUDITOR_X", "maybe", bool)

    def test_invalid_values_are_rejected_without_echoing_them(self):
        cases = {
            "AUDITOR_AGENT_MAX_TURNS": "lots",
            "AUDITOR_LLM_REQUEST_TIMEOUT_SECONDS": "soon",
        }
        for name, raw in cases.items():
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: raw}, clear=False),
                self.assertRaisesRegex(ConfigError, name) as caught,
            ):
                load_settings()
            self.assertNotIn(raw, str(caught.exception))

    def test_out_of_range_override_is_rejected(self):
        with (
            patch.dict(os.environ, {"AUDITOR_AGENT_MAX_TURNS": "0"}, clear=False),
            self.assertRaisesRegex(ConfigError, r"agent\.max_turns must be at least 1"),
        ):
            load_settings()

    def test_empty_list_item_is_rejected(self):
        with (
            patch.dict(os.environ, {"AUDITOR_SEARCH_SEARCHABLE_SUFFIXES": ".py,"}, clear=False),
            self.assertRaisesRegex(ConfigError, r"search\.searchable_suffixes"),
        ):
            load_settings()

    def test_unknown_auditor_variable_is_rejected(self):
        with (
            patch.dict(os.environ, {"AUDITOR_AGENT_MAX_TURN": "3"}, clear=False),
            self.assertRaisesRegex(ConfigError, "AUDITOR_AGENT_MAX_TURN"),
        ):
            load_settings()


class EnvFileTests(_TempDirTestCase):
    def test_reads_comments_quotes_and_export(self):
        env_file = self.write(
            ".env",
            "# provider settings\n"
            "\n"
            "DOTENV_TEST_PLAIN=plain-value\n"
            'DOTENV_TEST_DOUBLE="double quoted"\n'
            "DOTENV_TEST_SINGLE='single quoted'\n"
            "export DOTENV_TEST_EXPORTED=exported\n"
            "DOTENV_TEST_EMPTY=\n"
            "DOTENV_TEST_URL=https://example.com/v1?a=b\n",
        )
        load_env_file(env_file)
        self.assertEqual(os.environ["DOTENV_TEST_PLAIN"], "plain-value")
        self.assertEqual(os.environ["DOTENV_TEST_DOUBLE"], "double quoted")
        self.assertEqual(os.environ["DOTENV_TEST_SINGLE"], "single quoted")
        self.assertEqual(os.environ["DOTENV_TEST_EXPORTED"], "exported")
        self.assertEqual(os.environ["DOTENV_TEST_EMPTY"], "")
        self.assertEqual(os.environ["DOTENV_TEST_URL"], "https://example.com/v1?a=b")

    def test_does_not_override_real_environment(self):
        env_file = self.write(".env", "DOTENV_TEST_KEY=from-file\n")
        with patch.dict(os.environ, {"DOTENV_TEST_KEY": "from-shell"}, clear=False):
            load_env_file(env_file)
            self.assertEqual(os.environ["DOTENV_TEST_KEY"], "from-shell")

    def test_malformed_line_reports_line_number_but_not_content(self):
        env_file = self.write(".env", "DOTENV_TEST_OK=1\nsk-not-an-assignment\n")
        with self.assertRaisesRegex(ConfigError, "line 2") as caught:
            load_env_file(env_file)
        self.assertNotIn("sk-not-an-assignment", str(caught.exception))

    def test_malformed_key_does_not_echo_the_value(self):
        cases = (
            "OPENAI_API_KEY sk-fake-0000000000000000\n",
            "BAD KEY=sk-fake-0000000000000000\n",
            "1BAD=sk-fake-0000000000000000\n",
            "=sk-fake-0000000000000000\n",
        )
        for text in cases:
            env_file = self.write(".env", text)
            with self.subTest(text=text), self.assertRaisesRegex(ConfigError, "line 1") as caught:
                load_env_file(env_file)
            self.assertNotIn("sk-fake", str(caught.exception))

    def test_blank_and_comment_lines_between_assignments_are_skipped(self):
        env_file = self.write(
            ".env",
            "   \n# DOTENV_TEST_COMMENTED=yes\nDOTENV_TEST_AFTER=value\n\t\n  # indented comment\n",
        )
        load_env_file(env_file)
        self.assertEqual(os.environ["DOTENV_TEST_AFTER"], "value")
        self.assertNotIn("DOTENV_TEST_COMMENTED", os.environ)


class CliConfigTests(_TempDirTestCase):
    def _run_cli(self, *extra: str) -> tuple[int, str]:
        with (
            contextlib.chdir(self.directory),
            patch("agent.sys.stderr", new_callable=io.StringIO) as stderr,
            patch.object(agent, "GitHubRepository") as repository,
            patch("urllib.request.urlopen", side_effect=AssertionError("network used")) as urlopen,
        ):
            code = agent.main(
                ["--repo", "https://github.com/example/project", "--output", "report.json", *extra]
            )
        repository.assert_not_called()
        urlopen.assert_not_called()
        self.assertFalse((self.directory / "report.json").exists())
        return code, stderr.getvalue()

    def test_help_lists_config_option(self):
        self.assertIn("--config", agent.build_parser().format_help())

    def test_bad_config_file_exits_with_code_2(self):
        config = self.write("bad.toml", "[agent]\nmax_turns = 0\n")
        code, stderr = self._run_cli("--config", str(config))
        self.assertEqual(code, 2)
        self.assertTrue(stderr.startswith("Error: "))
        self.assertIn("agent.max_turns", stderr)

    def test_missing_config_file_exits_with_code_2(self):
        code, stderr = self._run_cli("--config", str(self.directory / "missing.toml"))
        self.assertEqual(code, 2)
        self.assertIn("Cannot read config file", stderr)

    def test_env_file_in_working_directory_is_loaded_before_settings(self):
        self.write(".env", "AUDITOR_AGENT_MAX_TURNS=0\n")
        code, stderr = self._run_cli()
        self.assertEqual(code, 2)
        self.assertIn("agent.max_turns", stderr)


if __name__ == "__main__":
    unittest.main()
