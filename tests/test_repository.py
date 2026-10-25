import dataclasses
import glob
import hashlib
import io
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auditor import repository as repository_module
from auditor.repository import (
    CloneSource,
    GitHubApiSource,
    LocalDirSource,
    RepositoryAddress,
    RepositoryError,
    RepositoryMetadata,
    _remove_tree,
    clone_command,
    git_environment,
    open_source,
    parse_git_log,
    parse_ls_tree,
)
from auditor.settings import ConfigError, load_settings
from tests.support import default_settings
from tools import GitHubRepository


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "audit-defective"
HEALTHY_FIXTURE = FIXTURE.with_name("audit-healthy")
ADDRESS = RepositoryAddress.parse("https://github.com/example/project", default_settings().github)
METADATA = {"private": False, "default_branch": "main", "size": 2048}
GIT_COMMANDS = ("clone", "ls-tree", "cat-file", "log")
SHA_A = "a" * 40
SHA_B = "b" * 40


def with_repository(settings, **changes):
    return dataclasses.replace(
        settings, repository=dataclasses.replace(settings.repository, **changes)
    )


def fixture_tree(root):
    return {
        "truncated": False,
        "tree": [
            {"type": "blob", "path": path.relative_to(root).as_posix(), "size": path.stat().st_size}
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ],
    }


def log_record(sha, date, message, author="Dev", files=()):
    """One commit as `git log --name-only` prints it in the clone source's format."""
    names = "".join(f"{name}\n" for name in files)
    return f"\x1e{sha}\x1f{date}\x1f{author}\x1f{message}\n\x1f\n\n{names}"


def tree_line(mode, kind, object_id, size, path):
    return f"{mode} {kind} {object_id} {size:>7}\t{path}".encode("utf-8") + b"\0"


class FakeProcess:
    # No such process, so the POSIX process-group kill in _stop finds nothing.
    pid = 2**31 - 1

    def __init__(self, output, returncode, hangs):
        self.stdout = None if output is None else io.BytesIO(output)
        self.returncode = returncode
        self.hangs = hangs
        self.wait_timeouts = []
        self.killed = False

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.hangs and timeout is not None:
            raise subprocess.TimeoutExpired("git", timeout)
        return self.returncode

    def kill(self):
        self.killed = True


class FakeGit:
    """Stands in for subprocess.Popen and answers the git commands a clone source runs."""

    def __init__(self, files=None, *, log=b"", errors=None, hangs=(), statuses=None, extra_tree=b""):
        self.files = {path: content for path, content in (files or {}).items()}
        self.log = log
        self.errors = errors or {}
        self.hangs = set(hangs)
        self.statuses = statuses or {}
        self.extra_tree = extra_tree
        self.calls = []
        self.processes = []

    @staticmethod
    def object_id(content):
        return hashlib.sha1(content).hexdigest()

    def tree(self):
        lines = [
            tree_line("100644", "blob", self.object_id(content), len(content), path)
            for path, content in sorted(self.files.items())
        ]
        return b"".join(lines) + self.extra_tree

    def commands(self, name):
        return [args for args, _ in self.calls if name in args]

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        command = next(item for item in args if item in GIT_COMMANDS)
        if command in self.errors:
            raise self.errors[command]
        output = b""
        if command == "clone":
            # What a --no-checkout clone leaves behind: a .git directory and nothing else.
            pack = Path(args[-1]) / ".git" / "objects" / "pack"
            pack.mkdir(parents=True)
            (pack / "pack-fake.pack").write_bytes(b"PACK")
        elif command == "ls-tree":
            output = self.tree()
        elif command == "cat-file":
            blobs = {self.object_id(content): content for content in self.files.values()}
            output = blobs[args[-1]]
        elif command == "log":
            output = self.log
        piped = kwargs.get("stdout") == subprocess.PIPE
        process = FakeProcess(
            output if piped else None, self.statuses.get(command, 0), command in self.hangs
        )
        self.processes.append(process)
        return process


def open_with(fake, settings=None, metadata=METADATA):
    with (
        patch("auditor.repository._request_json", return_value=metadata),
        patch("auditor.repository.subprocess.Popen", fake),
    ):
        return open_source(ADDRESS, settings or default_settings())


class CloneCommandTests(unittest.TestCase):
    def test_clone_command_is_an_argument_list_with_every_hardening_flag(self):
        hooks = Path("hooks-dir")
        destination = Path("dest-dir")
        command = clone_command(
            "git", hooks, "https://github.com/example/project.git", "main", 50, destination
        )
        self.assertEqual(
            command,
            [
                "git",
                "-c", f"core.hooksPath={hooks}",
                "-c", "core.symlinks=false",
                "-c", "core.fsmonitor=false",
                "-c", "core.protectNTFS=true",
                "-c", "core.protectHFS=true",
                "-c", "protocol.allow=never",
                "-c", "protocol.https.allow=always",
                "-c", "credential.helper=",
                "-c", "core.askPass=",
                "clone",
                "--no-checkout",
                "--depth", "50",
                "--single-branch",
                "--branch", "main",
                "--no-tags",
                "--no-recurse-submodules",
                "--quiet",
                "--",
                "https://github.com/example/project.git",
                str(destination),
            ],
        )

    def test_git_environment_disables_prompts_lfs_and_machine_config(self):
        inherited = {"GIT_DIR": "/elsewhere", "GIT_CONFIG_PARAMETERS": "'core.hooksPath'='x'"}
        with patch.dict(os.environ, inherited):
            environment = git_environment()
        self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(environment["GIT_LFS_SKIP_SMUDGE"], "1")
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertNotIn("GIT_ASKPASS", environment)
        self.assertNotIn("SSH_ASKPASS", environment)
        self.assertNotIn("GIT_DIR", environment)
        self.assertNotIn("GIT_CONFIG_PARAMETERS", environment)
        self.assertIn("PATH", {key.upper() for key in environment})

    def test_git_environment_is_an_allowlist(self):
        # Built from parts so the literal is not a credential-shaped string in the source.
        secret = "sk-" + "unit" + "test" + "0123456789"
        extra = {
            "OPENAI_API_KEY": secret,
            "HTTPS_PROXY": "http://proxy.invalid:3128",
            "SSH_ASKPASS": "/bin/prompt",
            "GIT_ASKPASS": "ask",
        }
        with patch.dict(os.environ, extra):
            environment = git_environment()
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn(secret, "".join(environment.values()))
        self.assertEqual(environment["HTTPS_PROXY"], "http://proxy.invalid:3128")
        self.assertNotIn("SSH_ASKPASS", environment)
        self.assertNotIn("GIT_ASKPASS", environment)

    def test_token_config_is_added_on_top_of_the_allowlist(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "tok" + "en123", "OPENAI_API_KEY": "k"}):
            environment = git_environment("https://github.com")
        self.assertEqual(environment["GIT_CONFIG_COUNT"], "1")
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("GITHUB_TOKEN", environment)

    def test_clone_runs_without_a_shell_and_with_a_timeout(self):
        fake = FakeGit({"app.py": b"print('x')\n"})
        settings = default_settings()
        source = open_with(fake, settings)
        try:
            self.assertEqual(source.backend, "clone")
            self.assertEqual(source.files, {"app.py": len(b"print('x')\n")})
            args, kwargs = fake.calls[0]
            self.assertIsInstance(args, list)
            self.assertNotIn("shell", kwargs)
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            # No pipes on the clone: a lingering helper cannot hold the deadline open.
            self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
            timeout = fake.processes[0].wait_timeouts[0]
            self.assertTrue(0 < timeout <= settings.repository.clone_timeout_seconds)
            self.assertEqual(args[-2], "https://github.com/example/project.git")
            self.assertEqual(args[args.index("--depth") + 1], str(settings.repository.clone_depth))
            hooks_dir = Path(args[2].split("=", 1)[1])
            self.assertTrue(hooks_dir.is_dir())
            self.assertEqual(list(hooks_dir.iterdir()), [])
        finally:
            source.close()

    def test_nothing_from_the_tree_is_written_to_disk(self):
        fake = FakeGit({"app.py": b"x\n", "deploy/run.sh": b"y\n"})
        source = open_with(fake)
        self.addCleanup(source.close)
        self.assertEqual([item.name for item in source.root.iterdir()], [".git"])
        with patch("auditor.repository.subprocess.Popen", fake):
            self.assertEqual(source.read_bytes("deploy/run.sh", 100), b"y\n")


class CloneTreeTests(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings()

    def test_reads_blobs_by_object_id_with_a_bounded_pipe(self):
        fake = FakeGit({"config.py": b"DEBUG = True\n"})
        source = open_with(fake)
        self.addCleanup(source.close)
        with patch("auditor.repository.subprocess.Popen", fake):
            self.assertEqual(source.read_bytes("config.py", 1024), b"DEBUG = True\n")
        args, kwargs = fake.calls[-1]
        self.assertEqual(args[-3:], ["cat-file", "blob", FakeGit.object_id(b"DEBUG = True\n")])
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertIn("protocol.allow=never", args)

    def test_unknown_and_oversized_paths_are_refused_without_running_git(self):
        fake = FakeGit({"big.txt": b"x" * 50})
        source = open_with(fake)
        self.addCleanup(source.close)
        before = len(fake.calls)
        with self.assertRaisesRegex(RepositoryError, "not present"):
            source.read_bytes("../etc/passwd", 1024)
        with self.assertRaisesRegex(RepositoryError, "inspection limit"):
            source.read_bytes("big.txt", 10)
        self.assertEqual(len(fake.calls), before)

    def test_symlinks_and_submodules_are_not_listed(self):
        extra = (
            tree_line("120000", "blob", SHA_A, 11, "link-to-passwd")
            + tree_line("160000", "commit", SHA_B, "-", "vendor/lib")
        )
        source = open_with(FakeGit({"a.py": b"x\n"}, extra_tree=extra))
        self.addCleanup(source.close)
        self.assertEqual(sorted(source.files), ["a.py"])

    def test_file_budget_falls_back_to_the_api(self):
        settings = with_repository(self.settings, max_checkout_files=1)
        fake = FakeGit({"a.py": b"1\n", "b.py": b"2\n"})
        with patch("auditor.repository._request_json", side_effect=[METADATA, {"tree": []}]):
            with patch("auditor.repository.subprocess.Popen", fake):
                source = open_source(ADDRESS, settings)
        self.assertEqual(source.backend, "api")
        self.assertEqual(source.fallback_reason, "repository exceeds repository.max_checkout_files")
        workspace = Path(fake.commands("clone")[0][-1]).parent
        self.assertFalse(workspace.exists())

    def test_size_budget_falls_back_to_the_api(self):
        settings = with_repository(self.settings, max_checkout_mb=1)
        huge = tree_line("100644", "blob", SHA_A, 2 * 1024 * 1024, "data.bin")
        with patch("auditor.repository._request_json", side_effect=[METADATA, {"tree": []}]):
            with patch("auditor.repository.subprocess.Popen", FakeGit(extra_tree=huge)):
                source = open_source(ADDRESS, settings)
        self.assertEqual(source.fallback_reason, "repository exceeds repository.max_checkout_mb")

    def test_oversized_tree_listing_falls_back_to_the_api(self):
        settings = with_repository(self.settings, max_tree_listing_bytes=10)
        with patch("auditor.repository._request_json", side_effect=[METADATA, {"tree": []}]):
            with patch("auditor.repository.subprocess.Popen", FakeGit({"a.py": b"x\n"})):
                source = open_source(ADDRESS, settings)
        self.assertEqual(source.fallback_reason, "git ls-tree output exceeded the configured limit")

    def test_listing_respects_max_listed_files(self):
        settings = with_repository(self.settings, max_listed_files=1)
        source = open_with(FakeGit({"a.py": b"1\n", "b.py": b"2\n"}), settings)
        self.addCleanup(source.close)
        self.assertEqual(list(source.files), ["a.py"])
        self.assertTrue(source.truncated)
        with self.assertRaisesRegex(RepositoryError, "not present"):
            source.read_bytes("b.py", 100)


class LsTreeParsingTests(unittest.TestCase):
    def test_keeps_regular_blobs_with_spaces_and_tabs_in_paths(self):
        output = (
            tree_line("100644", "blob", SHA_A, 12, "docs/read me.md")
            + tree_line("100755", "blob", SHA_B, 3, "bin/run\tit.sh")
        )
        self.assertEqual(
            [(blob.path, blob.object_id, blob.size) for blob in parse_ls_tree(output)],
            [("docs/read me.md", SHA_A, 12), ("bin/run\tit.sh", SHA_B, 3)],
        )

    def test_skips_links_submodules_and_malformed_records(self):
        output = (
            tree_line("120000", "blob", SHA_A, 5, "link")
            + tree_line("160000", "commit", SHA_B, "-", "sub")
            + tree_line("100644", "blob", "not-hex", 1, "bad-id")
            + b"garbage without a tab\0"
            + tree_line("100644", "blob", SHA_A, 1, "ok")
        )
        self.assertEqual([blob.path for blob in parse_ls_tree(output)], ["ok"])


class OpenSourceTests(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings()

    def _open_with_failing_clone(self, failure):
        if isinstance(failure, subprocess.TimeoutExpired):
            fake = FakeGit(hangs={"clone"})
        elif isinstance(failure, BaseException):
            fake = FakeGit(errors={"clone": failure})
        else:
            fake = FakeGit(statuses={"clone": failure})
        tree = {"truncated": False, "tree": []}
        with (
            patch("auditor.repository._request_json", side_effect=[METADATA, tree]),
            patch("auditor.repository.subprocess.Popen", fake),
        ):
            source = open_source(ADDRESS, self.settings)
        self.assertEqual(source.backend, "api")
        destination = Path(fake.commands("clone")[0][-1])
        self.assertFalse(destination.parent.exists(), "the failed clone was not removed")
        return source

    def test_missing_git_falls_back_to_the_api(self):
        source = self._open_with_failing_clone(FileNotFoundError("git"))
        self.assertEqual(source.fallback_reason, "git executable not found")

    def test_clone_timeout_falls_back_to_the_api(self):
        source = self._open_with_failing_clone(subprocess.TimeoutExpired("git", 120))
        self.assertIn("timed out", source.fallback_reason)

    def test_failed_clone_falls_back_without_exposing_stderr(self):
        source = self._open_with_failing_clone(128)
        self.assertEqual(source.fallback_reason, "git clone exited with status 128")
        repository = GitHubRepository(ADDRESS, self.settings, source=source)
        listing = repository.list_files()
        self.assertEqual(listing["backend"], "api")
        self.assertEqual(listing["fallback_reason"], "git clone exited with status 128")

    def test_api_backend_never_runs_git(self):
        settings = with_repository(self.settings, backend="api")
        tree = {"truncated": False, "tree": [{"type": "blob", "path": "a.py", "size": 1}]}
        with (
            patch("auditor.repository._request_json", side_effect=[METADATA, tree]),
            patch("auditor.repository.subprocess.Popen") as popen,
        ):
            source = open_source(ADDRESS, settings)
        popen.assert_not_called()
        self.assertEqual(source.backend, "api")
        self.assertIsNone(source.fallback_reason)
        self.assertNotIn("fallback_reason", GitHubRepository(ADDRESS, settings, source).list_files())

    def test_oversized_repository_is_refused_before_cloning(self):
        too_big = dict(METADATA, size=(self.settings.repository.max_repo_size_mb + 1) * 1024)
        with (
            patch("auditor.repository._request_json", return_value=too_big),
            patch("auditor.repository.subprocess.Popen") as popen,
            self.assertRaisesRegex(RepositoryError, "max_repo_size_mb"),
        ):
            open_source(ADDRESS, self.settings)
        popen.assert_not_called()

    def test_git_that_cannot_start_falls_back_to_the_api(self):
        source = self._open_with_failing_clone(PermissionError("access denied"))
        self.assertEqual(source.fallback_reason, "git could not be started")

    def test_repository_exactly_at_the_size_limit_is_cloned(self):
        at_limit = dict(METADATA, size=self.settings.repository.max_repo_size_mb * 1024)
        fake = FakeGit({"a.py": b"x\n"})
        source = open_with(fake, self.settings, at_limit)
        self.addCleanup(source.close)
        self.assertEqual(source.backend, "clone")
        self.assertEqual(len(fake.commands("clone")), 1)

    def test_unknown_size_uses_the_api_instead_of_cloning(self):
        unknown = {"private": False, "default_branch": "main"}
        with (
            patch("auditor.repository._request_json", side_effect=[unknown, {"tree": []}]),
            patch("auditor.repository.subprocess.Popen") as popen,
        ):
            source = open_source(ADDRESS, self.settings)
        popen.assert_not_called()
        self.assertEqual(source.fallback_reason, "repository size unknown")

    def test_private_repository_is_refused(self):
        with (
            patch("auditor.repository._request_json", return_value=dict(METADATA, private=True)),
            self.assertRaisesRegex(RepositoryError, "public"),
        ):
            open_source(ADDRESS, self.settings)

    def test_falls_back_even_when_cleanup_of_the_failed_clone_fails(self):
        # The simulated cleanup failure leaves the workspace behind; remove what this test made.
        pattern = os.path.join(tempfile.gettempdir(), self.settings.repository.temp_dir_prefix + "*")
        existing = set(glob.glob(pattern))
        self.addCleanup(
            lambda: [shutil.rmtree(d, ignore_errors=True) for d in set(glob.glob(pattern)) - existing]
        )
        tree = {"truncated": False, "tree": []}
        with (
            patch("auditor.repository._request_json", side_effect=[METADATA, tree]),
            patch("auditor.repository.subprocess.Popen", FakeGit(statuses={"clone": 128})),
            patch("auditor.repository.shutil.rmtree", side_effect=PermissionError("in use")),
        ):
            source = open_source(ADDRESS, self.settings)
        self.assertEqual(source.backend, "api")
        self.assertEqual(source.fallback_reason, "git clone exited with status 128")


class GitProcessTests(unittest.TestCase):
    def test_output_over_the_limit_stops_git(self):
        process = FakeProcess(b"x" * 500, 0, hangs=False)
        with (
            patch("auditor.repository.subprocess.Popen", return_value=process),
            patch("auditor.repository.os.name", "nt"),
            self.assertRaisesRegex(RepositoryError, "cat-file output exceeded"),
        ):
            repository_module._run_git(["git", "cat-file"], "cat-file", 5, output_limit=20)
        self.assertTrue(process.killed)
        self.assertTrue(process.stdout.closed)

    @unittest.skipIf(os.name == "nt", "process groups are POSIX only")
    def test_timeout_kills_the_whole_process_group_on_posix(self):
        process = FakeProcess(None, 0, hangs=True)
        with (
            patch("auditor.repository.subprocess.Popen", return_value=process),
            patch("auditor.repository.os.killpg") as killpg,
            self.assertRaisesRegex(RepositoryError, "timed out"),
        ):
            repository_module._run_git(["git", "clone"], "clone", 1)
        killpg.assert_called_once_with(process.pid, repository_module.signal.SIGKILL)

    def test_timeout_kills_git_on_windows(self):
        process = FakeProcess(None, 0, hangs=True)
        with (
            patch("auditor.repository.subprocess.Popen", return_value=process),
            patch("auditor.repository.os.name", "nt"),
            self.assertRaisesRegex(RepositoryError, "timed out"),
        ):
            repository_module._run_git(["git", "clone"], "clone", 1)
        self.assertTrue(process.killed)


class CloneCleanupTests(unittest.TestCase):
    def test_close_removes_read_only_files(self):
        source = open_with(FakeGit({"a.py": b"x\n"}))
        packed = source.root / ".git" / "objects" / "pack" / "pack-fake.pack"
        os.chmod(packed, stat.S_IREAD)
        workspace = source.root.parent
        source.close()
        self.assertFalse(workspace.exists())
        self.assertIsNone(source.cleanup_note)
        source.close()

    def test_context_manager_closes_the_clone_on_error(self):
        with (
            patch("auditor.repository._request_json", return_value=METADATA),
            patch("auditor.repository.subprocess.Popen", FakeGit({"a.py": b"x = 1\n"})),
        ):
            repository = GitHubRepository(ADDRESS, default_settings())
        workspace = repository.source.root.parent
        with self.assertRaises(RuntimeError), repository:
            raise RuntimeError("audit failed")
        self.assertFalse(workspace.exists())

    def test_failed_cleanup_is_retried_once_then_noted_without_raising(self):
        source = open_with(FakeGit({"a.py": b"x\n"}))
        self.addCleanup(_remove_tree, source.root.parent)
        with patch("auditor.repository.shutil.rmtree", side_effect=PermissionError("in use")) as rmtree:
            source.close()
        self.assertEqual(rmtree.call_count, 2)
        self.assertEqual(source.cleanup_note, "The temporary clone could not be fully removed.")

    def test_read_only_bits_are_cleared_only_on_windows(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        for name, expect_handler in (("posix", False), ("nt", True)):
            with (
                self.subTest(os_name=name),
                patch("auditor.repository.os.name", name),
                patch("auditor.repository.shutil.rmtree") as rmtree,
            ):
                repository_module._rmtree(directory)
            _, kwargs = rmtree.call_args
            self.assertEqual(bool(kwargs), expect_handler)


class CheckoutReadTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.root = self.directory / "checkout"
        self.root.mkdir()
        (self.root / "inside.txt").write_bytes(b"inside\n")
        (self.directory / "outside.txt").write_bytes(b"outside\n")
        self.settings = default_settings()

    def source(self):
        return LocalDirSource(self.root, self.settings, "main")

    def test_rejects_traversal_and_absolute_paths(self):
        source = self.source()
        for path in ("../outside.txt", "a/../../outside.txt", "/etc/passwd", "C:/Windows/win.ini",
                     "..\\outside.txt", str(self.directory / "outside.txt"), ""):
            with self.subTest(path=path), self.assertRaises(RepositoryError):
                source.read_bytes(path, 1024)

    def test_reads_a_regular_file_within_the_limit(self):
        self.assertEqual(self.source().read_bytes("inside.txt", 1024), b"inside\n")
        with self.assertRaisesRegex(RepositoryError, "inspection limit"):
            self.source().read_bytes("inside.txt", 3)

    def test_refuses_symlinks_that_point_outside(self):
        try:
            (self.root / "link.txt").symlink_to(self.directory / "outside.txt")
            (self.root / "linkdir").symlink_to(self.directory, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("this OS account cannot create symbolic links")
        source = self.source()
        self.assertEqual(sorted(source.files), ["inside.txt"])
        for path in ("link.txt", "linkdir/outside.txt"):
            with self.subTest(path=path), self.assertRaises(RepositoryError):
                source.read_bytes(path, 1024)

    @unittest.skipUnless(os.name == "nt", "directory junctions exist only on Windows")
    def test_refuses_a_junction_that_points_outside(self):
        import _winapi

        # Junctions need no special privilege, unlike symlinks, and is_symlink() is
        # False for them, so only the resolve check stands between them and the disk.
        junction = self.root / "linkdir"
        _winapi.CreateJunction(str(self.directory), str(junction))
        self.addCleanup(os.rmdir, junction)
        source = self.source()
        self.assertEqual(sorted(source.files), ["inside.txt"])
        with self.assertRaisesRegex(RepositoryError, "leaves the repository checkout"):
            source.read_bytes("linkdir/outside.txt", 1024)

    def test_refuses_a_path_whose_component_is_a_link(self):
        # Runs everywhere, including Windows accounts that cannot create real links.
        with patch.object(Path, "is_symlink", lambda path: path.name == "inside.txt"):
            with self.assertRaisesRegex(RepositoryError, "symbolic link"):
                self.source().read_bytes("inside.txt", 1024)

    def test_listing_respects_max_listed_files(self):
        (self.root / "second.txt").write_bytes(b"2\n")
        settings = with_repository(self.settings, max_listed_files=1)
        source = LocalDirSource(self.root, settings, "main")
        self.assertEqual(len(source.files), 1)
        self.assertTrue(source.truncated)


class LocalDirSourceTests(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings()

    def test_lists_and_reads_the_defective_fixture(self):
        source = LocalDirSource(FIXTURE, self.settings, "main")
        self.assertEqual(
            sorted(source.files), [".github/workflows/unsafe.yml", "config.py"]
        )
        self.assertFalse(source.truncated)
        self.assertEqual(source.backend, "local")
        self.assertEqual(source.commits(5), [])
        self.assertEqual(
            source.read_bytes("config.py", 4096), (FIXTURE / "config.py").read_bytes()
        )

    def assertSameToolResults(self, fixture, paths, patterns):
        local = GitHubRepository(ADDRESS, self.settings, LocalDirSource(fixture, self.settings, "main"))

        def raw_file(url, github, *, accept, limit):
            relative = url.split("/main/", 1)[1]
            return (fixture / relative).read_bytes()

        with patch("auditor.repository._request_json", return_value=fixture_tree(fixture)):
            api_source = GitHubApiSource(ADDRESS, RepositoryMetadata("main", 1), self.settings)
        api = GitHubRepository(ADDRESS, self.settings, api_source)
        with patch("auditor.repository._request_bytes", side_effect=raw_file):
            self.assertEqual(local.list_files()["files"], api.list_files()["files"])
            for path in paths:
                with self.subTest(path=path):
                    self.assertEqual(local.read_file(path), api.read_file(path))
            for pattern in patterns:
                with self.subTest(pattern=pattern):
                    self.assertEqual(local.search_code(pattern), api.search_code(pattern))

    def test_tools_give_the_same_results_over_local_and_api_sources(self):
        self.assertSameToolResults(
            FIXTURE,
            (".github/workflows/unsafe.yml", "config.py"),
            ("password", "pull_request_target", "absent-text"),
        )

    def test_tools_give_the_same_results_on_the_healthy_fixture(self):
        self.assertSameToolResults(
            HEALTHY_FIXTURE,
            (".github/workflows/test.yml", "config.py"),
            ("API_TOKEN", "pull_request", "absent-text"),
        )

    def test_clone_source_gives_the_same_tool_results_as_the_local_source(self):
        files = {
            path.relative_to(FIXTURE).as_posix(): path.read_bytes()
            for path in FIXTURE.rglob("*")
            if path.is_file()
        }
        clone = GitHubRepository(ADDRESS, self.settings, open_with(FakeGit(files)))
        self.addCleanup(clone.close)
        local = GitHubRepository(ADDRESS, self.settings, LocalDirSource(FIXTURE, self.settings, "main"))
        with patch("auditor.repository.subprocess.Popen", FakeGit(files)):
            self.assertEqual(clone.list_files()["files"], local.list_files()["files"])
            for path in files:
                with self.subTest(path=path):
                    self.assertEqual(clone.read_file(path), local.read_file(path))
            self.assertEqual(clone.search_code("password"), local.search_code("password"))


class ListFilesBackendTests(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings()

    def test_clone_listing_names_the_backend_without_a_fallback_reason(self):
        with (
            patch("auditor.repository._request_json", return_value=METADATA),
            patch("auditor.repository.subprocess.Popen", FakeGit({"a.py": b"x\n"})),
        ):
            repository = GitHubRepository(ADDRESS, self.settings)
        with repository:
            listing = repository.list_files()
        self.assertEqual(listing["backend"], "clone")
        self.assertNotIn("fallback_reason", listing)
        self.assertEqual(listing["files"], ["a.py"])

    def test_local_listing_names_the_backend(self):
        source = LocalDirSource(FIXTURE, self.settings, "main")
        listing = GitHubRepository(ADDRESS, self.settings, source).list_files()
        self.assertEqual(listing["backend"], "local")
        self.assertNotIn("fallback_reason", listing)

    def test_api_fallback_listing_carries_the_reason(self):
        tree = {"truncated": False, "tree": [{"type": "blob", "path": "a.py", "size": 1}]}
        with (
            patch("auditor.repository._request_json", side_effect=[METADATA, tree]),
            patch("auditor.repository.subprocess.Popen", side_effect=FileNotFoundError("git")),
        ):
            repository = GitHubRepository(ADDRESS, self.settings)
        listing = repository.list_files()
        self.assertEqual(listing["backend"], "api")
        self.assertEqual(listing["fallback_reason"], "git executable not found")
        self.assertEqual(listing["files"], ["a.py"])


class GitLogParsingTests(unittest.TestCase):
    def test_parses_separator_delimited_records(self):
        output = (
            log_record(SHA_A, "2026-10-25T10:00:00+05:00", "Fix deploy\n\nLonger body",
                       files=("deploy/run.sh", "README.md"))
            + log_record(SHA_B, "2026-10-24T09:00:00+00:00", "Initial commit", author="")
        )
        self.assertEqual(
            parse_git_log(output),
            [
                {"sha": SHA_A, "date": "2026-10-25T10:00:00+05:00", "author": "Dev",
                 "message": "Fix deploy\n\nLonger body",
                 "files": ["deploy/run.sh", "README.md"]},
                {"sha": SHA_B, "date": "2026-10-24T09:00:00+00:00", "author": None,
                 "message": "Initial commit", "files": []},
            ],
        )

    def test_empty_output_has_no_commits(self):
        self.assertEqual(parse_git_log(""), [])

    def test_drops_records_without_a_valid_sha_or_date(self):
        output = (
            log_record("not-a-sha", "2026-10-25T10:00:00+00:00", "bad sha")
            + log_record(SHA_A, "yesterday", "bad date")
            + f"\x1e{SHA_B}\x1f2026-10-25T10:00:00+00:00\x1fDev\x1fno closing separator\n"
            + log_record("c" * 64, "2026-10-25T10:00:00+00:00", "sha256 repository")
        )
        self.assertEqual([item["message"] for item in parse_git_log(output)], ["sha256 repository"])

    def test_commits_runs_hardened_git_log(self):
        log = log_record(SHA_A, "2026-10-25T10:00:00+00:00", "hello").encode("utf-8")
        fake = FakeGit({"a.py": b"x\n"}, log=log)
        source = open_with(fake)
        with patch("auditor.repository.subprocess.Popen", fake):
            try:
                commits = source.commits(7)
            finally:
                source.close()
        self.assertEqual([commit["message"] for commit in commits], ["hello"])
        args, kwargs = next(call for call in fake.calls if "log" in call[0])
        self.assertIn("protocol.allow=never", args)
        self.assertEqual(args[args.index("-n") + 1], "7")
        self.assertEqual(args[args.index("-C") + 1], str(source.root))
        self.assertEqual(kwargs["env"]["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)


class RepositoryAddressNameTests(unittest.TestCase):
    def setUp(self):
        self.github = default_settings().github

    def test_accepts_github_style_names(self):
        for url in ("https://github.com/a/b", "https://github.com/my-org/repo.name_v2.git"):
            with self.subTest(url=url):
                RepositoryAddress.parse(url, self.github)

    def test_rejects_names_github_would_not_allow(self):
        invalid = (
            "https://github.com/-org/repo",
            "https://github.com/org_name/repo",
            "https://github.com/" + "o" * 40 + "/repo",
            "https://github.com/org/..",
            "https://github.com/org/.",
            "https://github.com/org/re%2Fpo",
            "https://github.com/org/re po",
            "https://github.com/org/" + "r" * 101,
            "https://github.com/org/ext::sh",
        )
        for url in invalid:
            with self.subTest(url=url), self.assertRaisesRegex(RepositoryError, "valid GitHub name"):
                RepositoryAddress.parse(url, self.github)

    def test_api_raw_and_clone_urls_share_the_same_path(self):
        settings = default_settings()
        self.assertEqual(ADDRESS.path, "example/project")
        self.assertEqual(
            repository_module.clone_url(settings.github.clone_root, ADDRESS),
            "https://github.com/example/project.git",
        )
        self.assertEqual(
            repository_module._api_repo_url(ADDRESS, settings.github),
            "https://api.github.com/repos/example/project",
        )


class RepositorySettingsTests(unittest.TestCase):
    def test_backend_must_be_clone_or_api(self):
        with patch.dict(os.environ, {"AUDITOR_REPOSITORY_BACKEND": "ssh"}):
            with self.assertRaisesRegex(ConfigError, "repository.backend"):
                load_settings()

    def test_clone_root_must_be_https(self):
        with patch.dict(os.environ, {"AUDITOR_GITHUB_CLONE_ROOT": "file:///tmp"}):
            with self.assertRaisesRegex(ConfigError, "github.clone_root"):
                load_settings()

    def test_clone_depth_must_be_positive(self):
        with patch.dict(os.environ, {"AUDITOR_REPOSITORY_CLONE_DEPTH": "0"}):
            with self.assertRaisesRegex(ConfigError, "repository.clone_depth"):
                load_settings()

    def test_temp_dir_prefix_must_not_be_path_like(self):
        for prefix in ("../x", "a/b", "a\\b", "C:x", "x..y"):
            with self.subTest(prefix=prefix):
                with patch.dict(os.environ, {"AUDITOR_REPOSITORY_TEMP_DIR_PREFIX": prefix}):
                    with self.assertRaisesRegex(ConfigError, "repository.temp_dir_prefix"):
                        load_settings()

    def test_name_patterns_must_be_valid_regular_expressions(self):
        with patch.dict(os.environ, {"AUDITOR_GITHUB_OWNER_PATTERN": "[a-"}):
            with self.assertRaisesRegex(ConfigError, "github.owner_pattern"):
                load_settings()


@unittest.skipIf(shutil.which("git") is None, "git is not installed")
class RealGitTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(_remove_tree, self.directory)
        self.origin = self.directory / "origin"
        self.hooks = self.directory / "hooks"
        self.hooks.mkdir()
        self._git("init", "--quiet", "-b", "main", str(self.origin))
        (self.origin / "README").write_bytes(b"local\n")
        self._git("-C", str(self.origin), "add", "README")
        # Index entries for a link and a submodule, made without touching the filesystem.
        link_target = self._git("-C", str(self.origin), "hash-object", "-w", "--stdin", input=b"/etc/passwd")
        self._git(
            "-C", str(self.origin), "update-index", "--add",
            "--cacheinfo", f"120000,{link_target},link",
            "--cacheinfo", f"160000,{'1' * 40},vendor/sub",
        )
        self._git(
            "-C", str(self.origin), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
            "commit", "--quiet", "-m", "init",
        )

    def _git(self, *args, input=None):
        result = subprocess.run(
            ["git", "-c", f"core.hooksPath={self.hooks}", *args],
            env=git_environment(), check=True, capture_output=True, timeout=60, input=input,
        )
        return result.stdout.decode("ascii").strip()

    @staticmethod
    def _without_allowlist(command):
        flag = command.index("protocol.allow=never")
        del command[flag - 1 : flag + 1]
        return command

    def test_hardened_clone_refuses_the_file_transport(self):
        destination = self.directory / "refused"
        command = clone_command(
            "git", self.hooks, self.origin.as_uri(), "main", 1, destination
        )
        result = subprocess.run(command, env=git_environment(), capture_output=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((destination / ".git").exists())

        allowed = destination.with_name("allowed")
        control = self._without_allowlist(
            clone_command("git", self.hooks, self.origin.as_uri(), "main", 1, allowed)
        )
        result = subprocess.run(control, env=git_environment(), capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, "the control clone without the flag should work")
        self.assertTrue((allowed / ".git").is_dir())

    def test_clone_source_lists_and_reads_a_real_repository_without_a_checkout(self):
        real_clone_command = clone_command
        settings = default_settings()

        def local_clone(*args):
            return self._without_allowlist(real_clone_command(*args))

        with (
            patch("auditor.repository.clone_command", side_effect=local_clone),
            patch("auditor.repository.clone_url", return_value=self.origin.as_uri()),
        ):
            source = CloneSource(ADDRESS, RepositoryMetadata("main", 1), settings)
        self.addCleanup(source.close)
        self.assertEqual(source.files, {"README": len(b"local\n")})
        self.assertEqual([item.name for item in source.root.iterdir()], [".git"])
        self.assertEqual(source.read_bytes("README", 100), b"local\n")
        commits = source.commits(5)
        self.assertEqual([commit["message"] for commit in commits], ["init"])
        self.assertEqual(commits[0]["author"], "test")
        self.assertIn("README", commits[0]["files"])
        self.assertEqual(len(commits[0]["sha"]), 40)
        workspace = source.root.parent
        source.close()
        self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()
