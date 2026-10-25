import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auditor import repository as repository_module
from auditor.repository import (
    GitHubApiSource,
    LocalDirSource,
    RepositoryAddress,
    RepositoryError,
    RepositoryMetadata,
    git_environment,
)
from auditor.settings import ConfigError, load_env_file, load_settings
from tests.support import default_settings
from tests.test_repository import FIXTURE, METADATA, FakeGit, log_record, open_with
from tools import GitHubRepository, RepositoryTools, build_tool_schemas

ADDRESS = RepositoryAddress.parse("https://github.com/example/project", default_settings().github)
FAKE_VALUE = "fake" + "q" * 14
FAKE_TOKEN = "ghp_" + "q" * 36
SHA_A = "a" * 40
SHA_B = "b" * 40


def with_section(settings, section, **changes):
    return dataclasses.replace(
        settings, **{section: dataclasses.replace(getattr(settings, section), **changes)}
    )


def api_repository(files, settings=None, metadata=None):
    settings = settings or default_settings()
    tree = {
        "truncated": False,
        "tree": [{"type": "blob", "path": path, "size": size} for path, size in files.items()],
    }
    with patch("auditor.repository._request_json", return_value=tree):
        source = GitHubApiSource(ADDRESS, metadata or RepositoryMetadata("main", 1), settings)
    return GitHubRepository(ADDRESS, settings, source=source)


def serve(repository, contents):
    """Patch the raw-file request so each path returns its text."""
    def raw(url, github, *, accept, limit):
        data = contents[url.split("/main/", 1)[1]].encode("utf-8")
        if len(data) > limit:
            raise RepositoryError(f"File exceeds the {limit}-byte inspection limit")
        return data
    return patch("auditor.repository._request_bytes", side_effect=raw)


class ListFilesTests(unittest.TestCase):
    def setUp(self):
        self.files = {
            ".github/workflows/ci.yml": 10,
            ".github/workflows/cd.yaml": 10,
            "src/app.py": 10,
            "src/util/io.py": 10,
            "README.md": 10,
        }

    def test_prefix_and_glob_filter_and_report_totals(self):
        repository = api_repository(self.files)
        by_prefix = repository.list_files(path_prefix="src/")
        self.assertEqual(by_prefix["files"], ["src/app.py", "src/util/io.py"])
        self.assertEqual((by_prefix["total_files"], by_prefix["returned"]), (2, 2))
        self.assertFalse(by_prefix["truncated"])
        both = repository.list_files(path_prefix="src/", glob="*.py")
        self.assertEqual(both["total_files"], 2)
        by_glob = repository.list_files(glob="*.yml")
        self.assertEqual(by_glob["files"], [".github/workflows/ci.yml"])
        self.assertEqual(by_glob["backend"], "api")
        self.assertEqual(by_glob["default_branch"], "main")

    def test_entry_cap_marks_the_listing_truncated(self):
        settings = with_section(default_settings(), "repository", max_list_entries=2)
        result = api_repository(self.files, settings).list_files()
        self.assertEqual(result["returned"], 2)
        self.assertEqual(result["total_files"], len(self.files))
        self.assertTrue(result["truncated"])
        self.assertFalse(result["tree_truncated"])
        self.assertIn("path_prefix", result["notice"])

    def test_unsafe_prefix_is_refused(self):
        repository = api_repository(self.files)
        for prefix in ("../x", "/etc", "a//b", ""):
            with self.subTest(prefix=prefix), self.assertRaises(RepositoryError):
                repository.list_files(path_prefix=prefix)

    def test_no_arguments_still_works(self):
        self.assertEqual(api_repository(self.files).list_files()["total_files"], 5)


class ReadFileRangeTests(unittest.TestCase):
    def setUp(self):
        self.text = "".join(f"line {number}\n" for number in range(1, 11))
        self.text += f"API_KEY = {FAKE_VALUE}\n"

    def read(self, repository, **arguments):
        with serve(repository, {"a.py": self.text}):
            return repository.read_file("a.py", **arguments)

    def test_range_numbers_and_redaction_match_the_whole_file_view(self):
        repository = api_repository({"a.py": len(self.text)})
        whole = self.read(repository)
        part = self.read(repository, start_line=9, end_line=11)
        self.assertEqual(part["content"].splitlines(), whole["content"].splitlines()[8:11])
        self.assertEqual((part["start_line"], part["end_line"]), (9, 11))
        self.assertEqual(part["line_count"], whole["line_count"])
        self.assertFalse(part["truncated"])
        self.assertNotIn(FAKE_VALUE, part["content"])

    def test_line_cap_truncates_and_names_the_next_line(self):
        settings = with_section(default_settings(), "repository", max_read_lines=4)
        repository = api_repository({"a.py": len(self.text)}, settings)
        result = self.read(repository)
        self.assertEqual((result["start_line"], result["end_line"]), (1, 4))
        self.assertTrue(result["truncated"])
        self.assertEqual(result["next_start_line"], 5)
        self.assertEqual(result["line_count"], 11)

    def test_invalid_ranges_are_refused(self):
        repository = api_repository({"a.py": len(self.text)})
        for arguments in ({"start_line": 0}, {"start_line": 5, "end_line": 2}, {"start_line": 99}):
            with self.subTest(arguments=arguments), self.assertRaises(RepositoryError):
                self.read(repository, **arguments)

    def test_file_over_the_whole_file_limit_is_readable_by_range_up_to_the_hard_cap(self):
        settings = with_section(
            default_settings(), "repository", max_file_bytes=50, max_range_file_bytes=500
        )
        repository = api_repository({"a.py": len(self.text)}, settings)
        with self.assertRaisesRegex(RepositoryError, "start_line/end_line"):
            self.read(repository)
        result = self.read(repository, start_line=1, end_line=2)
        self.assertEqual(result["content"], "1: line 1\n2: line 2")
        too_big = with_section(settings, "repository", max_range_file_bytes=60)
        with self.assertRaisesRegex(RepositoryError, "inspection limit"):
            self.read(api_repository({"a.py": len(self.text)}, too_big), start_line=1)

    def test_dispatch_validates_range_argument_types(self):
        tools = RepositoryTools(api_repository({"a.py": 5}))
        for arguments in ({"path": "a.py", "start_line": "1"}, {"path": "a.py", "start_line": True}):
            with self.subTest(arguments=arguments), self.assertRaises(RepositoryError):
                tools.call("read_file", arguments)


class SearchUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.text = (
            f"password = {FAKE_VALUE}\n"
            "uses: actions/checkout@v3\n"
            "image: nginx:latest\n"
        )
        self.repository = api_repository({"a.yml": 1, "b.py": 1})

    def search(self, *args, **kwargs):
        with serve(self.repository, {"a.yml": self.text, "b.py": "nginx:latest\n"}):
            return self.repository.search_code(*args, **kwargs)

    def test_regex_mode_and_path_glob(self):
        result = self.search(r"nginx:(latest|\d+)", regex=True, path_glob="*.yml")
        self.assertEqual([(m["file"], m["line"]) for m in result["matches"]], [("a.yml", 3)])
        self.assertEqual(result["mode"], "regex")
        self.assertEqual(result["lines_skipped"], 0)

    def test_literal_mode_does_not_interpret_regex(self):
        self.assertEqual(self.search("nginx:(latest|1)")["matches"], [])

    def test_invalid_regex_is_a_tool_error_not_an_exception(self):
        result = self.search("(unclosed", regex=True)
        self.assertIn("Invalid regular expression", result["error"])
        self.assertNotIn("matches", result)

    def test_long_lines_are_skipped_in_regex_mode_and_counted(self):
        settings = with_section(default_settings(), "search", max_scan_line_chars=20)
        repository = api_repository({"a.yml": 1}, settings)
        with serve(repository, {"a.yml": "short\n" + "x" * 50 + "\n"}):
            result = repository.search_code("x+", regex=True)
            literal = repository.search_code("xxx")
        self.assertEqual(result["lines_skipped"], 1)
        self.assertEqual(result["matches"], [])
        self.assertIn("not searched", result["notice"])
        self.assertEqual(len(literal["matches"]), 1)

    def test_match_text_is_capped(self):
        settings = with_section(default_settings(), "search", max_line_chars=10)
        repository = api_repository({"a.yml": 1}, settings)
        with serve(repository, {"a.yml": "needle " + "y" * 40 + "\n"}):
            match = repository.search_code("needle")["matches"][0]
        self.assertEqual(len(match["text"]), 10)
        self.assertTrue(match["text_truncated"])

    def test_a_pattern_matching_a_secret_value_finds_nothing(self):
        for regex in (False, True):
            with self.subTest(regex=regex):
                self.assertEqual(self.search(FAKE_VALUE[:8], regex=regex)["matches"], [])
        found = self.search("password")
        self.assertIn("[REDACTED]", found["matches"][0]["text"])

    def test_the_pattern_echo_is_redacted(self):
        result = self.search(f"password = {FAKE_VALUE}")
        self.assertNotIn(FAKE_VALUE, json.dumps(result))
        token_result = self.search(FAKE_TOKEN)
        self.assertNotIn(FAKE_TOKEN, json.dumps(token_result))

    def test_redaction_runs_once_per_file_across_searches(self):
        with (
            serve(self.repository, {"a.yml": self.text, "b.py": "x\n"}),
            patch("tools.redacted_lines", wraps=__import__("tools").redacted_lines) as spy,
        ):
            self.repository.search_code("uses")
            self.repository.search_code("image")
        self.assertEqual(spy.call_count, 2)


class GitLogUpgradeTests(unittest.TestCase):
    def clone_repository(self, log, settings=None):
        fake = FakeGit({"a.py": b"x\n"}, log=log.encode("utf-8"))
        settings = settings or default_settings()
        source = open_with(fake, settings)
        repository = GitHubRepository(ADDRESS, settings, source)
        self.addCleanup(repository.close)
        return repository, fake

    def history(self):
        return (
            log_record(SHA_A, "2026-10-25T10:00:00+00:00", "one", author="Ann", files=("a.py", "b.py"))
            + log_record(SHA_B, "2026-10-24T10:00:00+00:00", "two", author="Ann")
            + log_record("c" * 40, "2026-10-23T10:00:00+00:00", "three", author="Bob")
        )

    def test_clone_log_counts_authors_and_lists_changed_files_without_emails(self):
        repository, fake = self.clone_repository(self.history())
        with patch("auditor.repository.subprocess.Popen", fake):
            result = repository.git_log()
        self.assertEqual(result["author_count"], 2)
        self.assertEqual(result["commits"][0]["files"], ["a.py", "b.py"])
        self.assertNotIn("@", json.dumps(result))
        args = fake.commands("log")[0]
        self.assertNotIn("%ae", " ".join(args))
        self.assertNotIn("%aE", " ".join(args))

    def test_changed_files_are_capped_per_commit(self):
        settings = with_section(default_settings(), "git_log", max_files_per_commit=1)
        repository, fake = self.clone_repository(self.history(), settings)
        with patch("auditor.repository.subprocess.Popen", fake):
            first = repository.git_log()["commits"][0]
        self.assertEqual(first["files"], ["a.py"])
        self.assertEqual(first["files_omitted"], 1)

    def test_max_commits_is_clamped_and_passed_to_git(self):
        repository, fake = self.clone_repository(self.history())
        cap = repository.settings.git_log.max_commits
        with patch("auditor.repository.subprocess.Popen", fake):
            result = repository.git_log(max_commits=10_000)
            repository.git_log(max_commits=-3)
        self.assertEqual(result["max_commits"], cap)
        counts = [args[args.index("-n") + 1] for args in fake.commands("log")]
        self.assertEqual(counts, [str(cap), "1"])

    def test_path_filter_goes_after_the_double_dash(self):
        repository, fake = self.clone_repository(self.history())
        with patch("auditor.repository.subprocess.Popen", fake):
            result = repository.git_log(path="deploy/run.sh")
        args = fake.commands("log")[0]
        self.assertEqual(args[-2:], ["--", "deploy/run.sh"])
        self.assertEqual(result["path"], "deploy/run.sh")
        self.assertEqual(fake.calls[-1][1]["env"]["GIT_LITERAL_PATHSPECS"], "1")
        with self.assertRaises(RepositoryError):
            repository.git_log(path="../etc/passwd")

    def test_api_log_sends_the_path_query_and_has_no_changed_files(self):
        repository = api_repository({"a.py": 1})
        body = json.dumps([{
            "sha": SHA_A,
            "commit": {"author": {"date": "2026-10-25T00:00:00Z", "name": "Ann", "email": "ann@example.invalid"},
                       "message": "hello"},
        }]).encode("utf-8")
        with patch("auditor.repository._request_bytes", return_value=body) as request:
            result = repository.git_log(max_commits=5, path="a b.py")
        url = request.call_args.args[0]
        self.assertIn("per_page=5", url)
        self.assertIn("path=a+b.py", url)
        self.assertEqual(result["author_count"], 1)
        self.assertNotIn("files", result["commits"][0])
        self.assertNotIn("ann@example", json.dumps(result))
        self.assertIn("clone backend", result["notice"])

    def test_message_is_redacted_before_it_is_truncated(self):
        message = f"DB_PASSWORD={FAKE_VALUE}"
        cut = len("DB_PASSWORD=") + 6
        settings = with_section(default_settings(), "git_log", message_max_chars=cut)
        log = log_record(SHA_A, "2026-10-25T10:00:00+00:00", message)
        repository, fake = self.clone_repository(log, settings)
        with patch("auditor.repository.subprocess.Popen", fake):
            result = repository.git_log()
        text = result["commits"][0]["message"]
        self.assertEqual(len(text), cut)
        self.assertNotIn("fakeq", text)
        self.assertTrue(result["commits"][0]["message_truncated"])

    def test_local_source_says_there_is_no_history(self):
        source = LocalDirSource(FIXTURE, default_settings(), "main")
        result = GitHubRepository(ADDRESS, default_settings(), source).git_log()
        self.assertEqual(result["commits"], [])
        self.assertIn("not available", result["notice"])

    def test_schema_and_dispatch_expose_the_new_arguments(self):
        settings = default_settings()
        schemas = {item["function"]["name"]: item["function"] for item in build_tool_schemas(settings.git_log)}
        self.assertEqual(schemas["git_log"]["parameters"]["properties"]["max_commits"]["maximum"],
                         settings.git_log.max_commits)
        self.assertIn("regex", schemas["search_code"]["parameters"]["properties"])
        self.assertIn("get_repo_metadata", schemas)
        self.assertIn("list_workflows", schemas)
        self.assertEqual(schemas["search_code"]["parameters"]["required"], ["pattern"])
        tools = RepositoryTools(api_repository({"a.py": 1}))
        with self.assertRaises(RepositoryError):
            tools.call("git_log", {"max_commits": "5"})

    def test_settings_bound_commits_to_the_github_page_size(self):
        with patch.dict(os.environ, {"AUDITOR_GIT_LOG_MAX_COMMITS": "101"}):
            with self.assertRaisesRegex(ConfigError, "git_log.max_commits must be at most 100"):
                load_settings()
        with patch.dict(os.environ, {"AUDITOR_GIT_LOG_DEFAULT_COMMITS": "60"}):
            with self.assertRaisesRegex(ConfigError, "default_commits"):
                load_settings()
        with patch.dict(os.environ, {"AUDITOR_REPOSITORY_MAX_RANGE_FILE_BYTES": "10"}):
            with self.assertRaisesRegex(ConfigError, "max_range_file_bytes"):
                load_settings()


class MetadataAndWorkflowTests(unittest.TestCase):
    def test_metadata_comes_from_the_fetched_values(self):
        metadata = RepositoryMetadata(
            "main", 42, archived=False, fork=True, private=False, visibility="public",
            license_spdx_id="MIT", pushed_at="2026-10-01T00:00:00Z",
        )
        result = api_repository({"a.py": 1}, metadata=metadata).get_repo_metadata()
        self.assertTrue(result["available"])
        self.assertEqual(result["license_spdx_id"], "MIT")
        self.assertIs(result["fork"], True)
        self.assertEqual(result["size_kb"], 42)

    def test_fetch_metadata_extracts_the_extra_fields(self):
        raw = dict(
            METADATA, archived=True, fork=False, visibility="public",
            license={"spdx_id": "Apache-2.0"}, pushed_at="2026-10-01T00:00:00Z",
        )
        with patch("auditor.repository._request_json", return_value=raw):
            metadata = repository_module.fetch_metadata(ADDRESS, default_settings().github)
        self.assertEqual((metadata.archived, metadata.license_spdx_id), (True, "Apache-2.0"))

    def test_local_source_reports_that_metadata_is_missing(self):
        source = LocalDirSource(FIXTURE, default_settings(), "main")
        result = GitHubRepository(ADDRESS, default_settings(), source).get_repo_metadata()
        self.assertFalse(result["available"])
        self.assertEqual(result["default_branch"], "main")
        self.assertIn("not available", result["notice"])

    def test_metadata_strings_are_bounded(self):
        settings = with_section(default_settings(), "github", metadata_max_chars=5)
        metadata = RepositoryMetadata("main", 1, license_spdx_id="L" * 50)
        result = api_repository({"a.py": 1}, settings, metadata).get_repo_metadata()
        self.assertEqual(result["license_spdx_id"], "LLLLL")

    def test_workflow_scan_reports_triggers_permissions_and_jobs_with_lines(self):
        source = LocalDirSource(FIXTURE, default_settings(), "main")
        result = GitHubRepository(ADDRESS, default_settings(), source).list_workflows()
        self.assertEqual(result["unreadable_files"], [])
        workflow = result["workflows"][0]
        self.assertEqual(workflow["file"], ".github/workflows/unsafe.yml")
        self.assertEqual(workflow["on_line"], 2)
        self.assertEqual(workflow["triggers"], [{"name": "pull_request_target", "line": 3}])
        self.assertEqual(workflow["permissions"], [{"line": 5, "text": "permissions: write-all"}])
        self.assertEqual(workflow["jobs"], [{"id": "validate", "line": 7}])

    def test_inline_trigger_lists_and_job_level_permissions(self):
        text = (
            "on: [push, pull_request]\n"
            "jobs:\n"
            "  build:\n"
            "    permissions:\n"
            "      contents: read\n"
            "    steps: []\n"
            "  deploy:\n"
            "    steps: []\n"
        )
        repository = api_repository({".github/workflows/ci.yaml": len(text)})
        with serve(repository, {".github/workflows/ci.yaml": text}):
            workflow = repository.list_workflows()["workflows"][0]
        self.assertEqual([item["name"] for item in workflow["triggers"]], ["push", "pull_request"])
        self.assertEqual(workflow["permissions"][0]["line"], 4)
        self.assertEqual([job["id"] for job in workflow["jobs"]], ["build", "deploy"])

    def test_oversized_workflow_is_reported_unreadable_not_skipped(self):
        settings = with_section(default_settings(), "repository", max_file_bytes=10, max_range_file_bytes=20)
        repository = api_repository({".github/workflows/big.yml": 5000, "README.md": 1}, settings)
        result = repository.list_workflows()
        self.assertEqual(result["workflows"], [])
        self.assertEqual(result["unreadable_files"][0]["file"], ".github/workflows/big.yml")

    def test_files_beyond_the_workflow_cap_are_named(self):
        settings = with_section(default_settings(), "workflows", max_files=1)
        files = {f".github/workflows/{name}.yml": 5 for name in ("a", "b")}
        repository = api_repository(files, settings)
        with serve(repository, {path: "on: push\n" for path in files}):
            result = repository.list_workflows()
        self.assertEqual(len(result["workflows"]), 1)
        self.assertEqual(result["unreadable_files"][0]["file"], ".github/workflows/b.yml")

    def test_new_tools_reject_arguments(self):
        tools = RepositoryTools(api_repository({"a.py": 1}))
        for name in ("get_repo_metadata", "list_workflows"):
            with self.subTest(name=name), self.assertRaises(RepositoryError):
                tools.call(name, {"depth": 1})


class TokenTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"GITHUB_TOKEN": FAKE_TOKEN})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.github = default_settings().github

    def request(self, url):
        with patch("auditor.repository.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b"{}"
            repository_module._request_bytes(url, self.github, accept="x", limit=10)
        return urlopen.call_args.args[0]

    def test_bearer_header_is_sent_to_the_api_host_only(self):
        api = self.request("https://api.github.com/repos/example/project")
        self.assertEqual(api.get_header("Authorization"), f"Bearer {FAKE_TOKEN}")
        for url in (
            "https://raw.githubusercontent.com/example/project/main/a.py",
            "https://example.com/repos/x",
            "http://api.github.com/repos/x",
            "https://api.github.com.evil.example/repos/x",
        ):
            with self.subTest(url=url):
                self.assertIsNone(self.request(url).get_header("Authorization"))

    def test_no_header_without_a_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "  "}):
            self.assertIsNone(self.request("https://api.github.com/repos/x").get_header("Authorization"))

    def test_clone_env_carries_the_header_but_argv_and_other_commands_do_not(self):
        fake = FakeGit({"a.py": b"x\n"}, log=b"")
        source = open_with(fake)
        self.addCleanup(source.close)
        with patch("auditor.repository.subprocess.Popen", fake):
            source.read_bytes("a.py", 100)
        clone_env = fake.calls[0][1]["env"]
        self.assertEqual(clone_env["GIT_CONFIG_COUNT"], "1")
        self.assertEqual(clone_env["GIT_CONFIG_KEY_0"], "http.https://github.com/.extraheader")
        self.assertTrue(clone_env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic "))
        self.assertNotIn(FAKE_TOKEN, clone_env["GIT_CONFIG_VALUE_0"])
        for args, kwargs in fake.calls:
            self.assertNotIn(FAKE_TOKEN, " ".join(args))
            if args is not fake.calls[0][0]:
                self.assertNotIn("GIT_CONFIG_COUNT", kwargs["env"])

    def test_inherited_git_config_variables_are_still_dropped(self):
        with patch.dict(os.environ, {"GIT_CONFIG_COUNT": "5", "GIT_CONFIG_KEY_0": "core.x"}):
            environment = git_environment()
        self.assertNotIn("GIT_CONFIG_COUNT", environment)

    def test_error_text_never_contains_the_token(self):
        import urllib.error
        with patch("auditor.repository.urllib.request.urlopen",
                   side_effect=urllib.error.URLError(f"failed Bearer {FAKE_TOKEN}")):
            with self.assertRaises(RepositoryError) as caught:
                repository_module._request_bytes(
                    "https://api.github.com/repos/x", self.github, accept="x", limit=10
                )
        self.assertNotIn(FAKE_TOKEN, str(caught.exception))


class EnvFileUpgradeTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def load(self, data: bytes):
        path = self.directory / ".env"
        path.write_bytes(data)
        load_env_file(path)

    def test_inline_comments_are_stripped_from_unquoted_values_only(self):
        self.load(
            b"UPG_A=value # note\n"
            b"UPG_B=https://example.com/a#frag\n"
            b'UPG_C="keep # this" # but not this\n'
            b"UPG_D='single # kept'\n"
            b"UPG_E=  # only a comment\n"
            b'UPG_F="unterminated # value\n'
        )
        self.assertEqual(os.environ["UPG_A"], "value")
        self.assertEqual(os.environ["UPG_B"], "https://example.com/a#frag")
        self.assertEqual(os.environ["UPG_C"], "keep # this")
        self.assertEqual(os.environ["UPG_D"], "single # kept")
        self.assertEqual(os.environ["UPG_E"], "")
        self.assertEqual(os.environ["UPG_F"], '"unterminated # value')

    def test_a_byte_order_mark_does_not_break_the_first_key(self):
        self.load(b"\xef\xbb\xbfUPG_FIRST=1\nUPG_SECOND=2\n")
        self.assertEqual(os.environ["UPG_FIRST"], "1")
        self.assertEqual(os.environ["UPG_SECOND"], "2")


if __name__ == "__main__":
    unittest.main()
