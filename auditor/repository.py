"""Read-only sources for a public GitHub repository: a hardened shallow clone or the GitHub API.

Nothing here executes repository code. ``git`` is the only subprocess; it gets an
argument list, a deadline and configuration that disables hooks, submodules, LFS
filters, credential helpers and every transport except HTTPS. The clone is never
checked out: files are listed with ``git ls-tree`` and read with ``git cat-file``,
so repository paths never reach the local filesystem.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any, Protocol

from auditor.redaction import redact_secrets
from auditor.settings import GitHubSettings, RepositorySettings, Settings, load_settings


# %x1f and %x1e are unit/record separators, which never occur in a sha or an ISO date
# and are vanishingly rare in commit messages. The record separator comes first and
# the message is closed by a field separator, because --name-only prints the changed
# paths after the formatted text. %an is the author name; the email is never read.
_LOG_FORMAT = "--format=%x1e%H%x1f%aI%x1f%an%x1f%B%x1f"
_FIELD_SEPARATOR = "\x1f"
_RECORD_SEPARATOR = "\x1e"
# SHA-1 and SHA-256 object names, as git prints them.
_OBJECT_ID_LENGTHS = frozenset({40, 64})
_HEX_DIGITS = frozenset("0123456789abcdef")
# Plain and executable files. Symlinks (120000) only hold a link target and
# submodules (160000) have no content in this repository, so neither is listed.
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})
_BYTES_PER_MB = 1024 * 1024
_CLEANUP_NOTE = "The temporary clone could not be fully removed."
_TOKEN_VARIABLE = "GITHUB_TOKEN"


class RepositoryError(Exception):
    """Raised when a repository cannot be inspected safely."""


class _GitFailure(RepositoryError):
    """git could not run, failed or went over a budget; the message never includes stderr."""


@dataclass(frozen=True)
class RepositoryAddress:
    owner: str
    name: str
    url: str

    @classmethod
    def parse(cls, value: str, github: GitHubSettings) -> "RepositoryAddress":
        parsed = urllib.parse.urlsplit(value.strip())
        if (
            parsed.scheme.lower() != "https"
            or parsed.hostname != "github.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise RepositoryError(
                "Repository must be a public HTTPS URL in the form "
                "https://github.com/owner/repository."
            )
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 2:
            raise RepositoryError(
                "Repository URL must identify exactly one GitHub owner and repository."
            )
        owner, name = parts
        if name.endswith(".git"):
            name = name[:-4]
        # Owner and name end up in API, raw and clone URLs; GitHub's own naming
        # rules keep anything URL- or path-significant out of them.
        if (
            not re.fullmatch(github.owner_pattern, owner)
            or not re.fullmatch(github.name_pattern, name)
            or name in {".", ".."}
        ):
            raise RepositoryError("Repository owner or name is not a valid GitHub name.")
        canonical = f"https://github.com/{owner}/{name}"
        return cls(owner=owner, name=name, url=canonical)

    @property
    def path(self) -> str:
        """``owner/name``, quoted the same way for every URL built from it."""
        owner = urllib.parse.quote(self.owner, safe="")
        name = urllib.parse.quote(self.name, safe="")
        return f"{owner}/{name}"


@dataclass(frozen=True)
class RepositoryMetadata:
    default_branch: str
    size_kib: int | None
    archived: bool | None = None
    fork: bool | None = None
    private: bool | None = None
    visibility: str | None = None
    license_spdx_id: str | None = None
    pushed_at: str | None = None


@dataclass(frozen=True)
class TreeBlob:
    path: str
    object_id: str
    size: int


class RepositorySource(Protocol):
    """Raw, unredacted access to one repository snapshot; callers redact before output."""

    default_branch: str
    metadata: RepositoryMetadata | None
    files: dict[str, int | None]
    truncated: bool
    backend: str
    fallback_reason: str | None
    cleanup_note: str | None

    def read_bytes(self, path: str, limit: int) -> bytes: ...

    def commits(self, max_commits: int, path: str | None = None) -> list[dict[str, Any]]:
        """Newest first: sha, date, author name or None, message, and the changed
        paths as ``files`` (None where the backend cannot list them)."""
        ...

    def close(self) -> None: ...


def open_source(address: RepositoryAddress, settings: Settings) -> RepositorySource:
    """Clone when configured and possible, otherwise use the API.

    Raises RepositoryError for a private or oversized repository. A missing git, a
    timeout, a failed clone or a tree over budget falls back to the API with
    ``fallback_reason`` set.
    """
    metadata = fetch_metadata(address, settings.github)
    if settings.repository.backend == "api":
        return GitHubApiSource(address, metadata, settings)
    if metadata.size_kib is None:
        return GitHubApiSource(
            address, metadata, settings, fallback_reason="repository size unknown"
        )
    _check_size(metadata.size_kib, settings.repository.max_repo_size_mb)
    try:
        return CloneSource(address, metadata, settings)
    except _GitFailure as exc:
        return GitHubApiSource(address, metadata, settings, fallback_reason=str(exc))


def fetch_metadata(address: RepositoryAddress, github: GitHubSettings) -> RepositoryMetadata:
    metadata = _request_json(_api_repo_url(address, github), github)
    if metadata.get("private") is True:
        raise RepositoryError("Only public GitHub repositories are supported.")
    default_branch = metadata.get("default_branch")
    if not isinstance(default_branch, str) or not default_branch:
        raise RepositoryError("GitHub repository has no discoverable default branch.")
    size = metadata.get("size")
    valid_size = isinstance(size, int) and not isinstance(size, bool) and size >= 0
    license_info = metadata.get("license")
    return RepositoryMetadata(
        default_branch,
        size if valid_size else None,
        archived=_flag(metadata.get("archived")),
        fork=_flag(metadata.get("fork")),
        private=_flag(metadata.get("private")),
        visibility=_text(metadata.get("visibility")),
        license_spdx_id=(
            _text(license_info.get("spdx_id")) if isinstance(license_info, dict) else None
        ),
        pushed_at=_text(metadata.get("pushed_at")),
    )


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _check_size(size_kib: int, max_repo_size_mb: int) -> None:
    # GitHub reports size in KiB. Too big to clone is a property of the repository,
    # not a transient failure, so this does not fall back to the API.
    if size_kib > max_repo_size_mb * 1024:
        raise RepositoryError(
            f"Repository is about {size_kib // 1024} MiB, above repository.max_repo_size_mb "
            f"({max_repo_size_mb}); set repository.backend = \"api\" to audit it without cloning."
        )


def _api_repo_url(address: RepositoryAddress, github: GitHubSettings) -> str:
    return f"{github.api_root}/repos/{address.path}"


def github_token() -> str | None:
    """GITHUB_TOKEN as set at call time, so a value loaded from .env is seen."""
    token = os.environ.get(_TOKEN_VARIABLE, "").strip()
    return token or None


def _is_api_url(url: str, github: GitHubSettings) -> bool:
    target = urllib.parse.urlsplit(url)
    api = urllib.parse.urlsplit(github.api_root)
    return target.scheme == "https" and target.netloc.lower() == api.netloc.lower()


def _safe_reason(reason: object, token: str | None) -> str:
    # The reason text comes from the network stack or a proxy and can echo request
    # headers; the exact token is masked even if it has no recognisable shape.
    text = str(reason)
    if token is not None:
        text = text.replace(token, "[REDACTED TOKEN]")
    return redact_secrets(text)


def _request_bytes(url: str, github: GitHubSettings, *, accept: str, limit: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": github.user_agent,
        },
    )
    token = github_token()
    # Only the API host gets the token; raw file hosts do not need it. Unredirected,
    # because urllib copies ordinary headers to wherever a redirect points.
    if token is not None and _is_api_url(url, github):
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=github.request_timeout_seconds) as response:
            data = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RepositoryError("GitHub resource was not found or is not public.") from exc
        if exc.code in {403, 429}:
            hint = "" if token is not None else f" Setting {_TOKEN_VARIABLE} raises the limit."
            raise RepositoryError(
                "GitHub denied the request; check API rate limits and repository access."
                + hint
            ) from exc
        raise RepositoryError(f"GitHub request failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RepositoryError(f"GitHub request failed: {_safe_reason(exc.reason, token)}.") from exc
    except TimeoutError as exc:
        raise RepositoryError("GitHub request failed: timed out.") from exc
    if len(data) > limit:
        raise RepositoryError("GitHub response exceeded the configured size limit.")
    return data


def _request_json(url: str, github: GitHubSettings) -> dict[str, Any]:
    raw = _request_bytes(
        url,
        github,
        accept="application/vnd.github+json",
        limit=github.max_response_bytes,
    )
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RepositoryError("GitHub returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RepositoryError("GitHub returned an unexpected response.")
    return parsed


class GitHubApiSource:
    """Tree from the GitHub API, file bodies from raw.githubusercontent.com."""

    backend = "api"

    def __init__(
        self,
        address: RepositoryAddress,
        metadata: RepositoryMetadata,
        settings: Settings,
        *,
        fallback_reason: str | None = None,
    ):
        self.address = address
        self.metadata: RepositoryMetadata | None = metadata
        self.default_branch = metadata.default_branch
        self.fallback_reason = fallback_reason
        self.cleanup_note: str | None = None
        self._github = settings.github
        encoded_branch = urllib.parse.quote(self.default_branch, safe="")
        tree_url = f"{_api_repo_url(address, self._github)}/git/trees/{encoded_branch}?recursive=1"
        tree = _request_json(tree_url, self._github)
        entries = tree.get("tree")
        if not isinstance(entries, list):
            raise RepositoryError("GitHub did not return a repository file tree.")
        max_listed_files = settings.repository.max_listed_files
        self.files: dict[str, int | None] = {}
        for entry in entries[:max_listed_files]:
            if (
                isinstance(entry, dict)
                and entry.get("type") == "blob"
                and isinstance(entry.get("path"), str)
            ):
                size = entry.get("size")
                self.files[entry["path"]] = size if isinstance(size, int) else None
        self.truncated = tree.get("truncated") is True or len(entries) > max_listed_files

    def read_bytes(self, path: str, limit: int) -> bytes:
        encoded_path = urllib.parse.quote(path, safe="/")
        encoded_branch = urllib.parse.quote(self.default_branch, safe="")
        url = f"{self._github.raw_root}/{self.address.path}/{encoded_branch}/{encoded_path}"
        return _request_bytes(url, self._github, accept="application/octet-stream", limit=limit)

    def commits(self, max_commits: int, path: str | None = None) -> list[dict[str, Any]]:
        query = {"per_page": str(max_commits)}
        if path is not None:
            query["path"] = path
        url = f"{_api_repo_url(self.address, self._github)}/commits?{urllib.parse.urlencode(query)}"
        raw = _request_bytes(
            url,
            self._github,
            accept="application/vnd.github+json",
            limit=self._github.commit_history_bytes,
        )
        try:
            commits = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RepositoryError("GitHub returned invalid commit history JSON.") from exc
        if not isinstance(commits, list):
            raise RepositoryError("GitHub returned an unexpected commit history response.")
        parsed = []
        for commit in commits:
            if not isinstance(commit, dict):
                continue
            details = commit.get("commit")
            if not isinstance(details, dict):
                continue
            author = details.get("author")
            if not isinstance(author, dict):
                author = {}
            parsed.append(
                {
                    "sha": str(commit.get("sha", "")),
                    "date": author.get("date"),
                    "author": _text(author.get("name")),
                    "message": str(details.get("message", "")),
                    # The list endpoint has no changed paths; that would cost a request per commit.
                    "files": None,
                }
            )
        return parsed

    def close(self) -> None:
        """Nothing to release; present so every source can be closed the same way."""


def hardened_git_config(hooks_dir: Path) -> list[str]:
    """``-c`` options placed before every git subcommand we run."""
    return [
        # An empty hooks directory: no hook from a template or user config can run.
        "-c", f"core.hooksPath={hooks_dir}",
        # Nothing is checked out, but if anything ever is, links stay plain files.
        "-c", "core.symlinks=false",
        # Never start an fsmonitor daemon or command.
        "-c", "core.fsmonitor=false",
        # Refuse tree entries that are dangerous on NTFS or HFS+ (.git aliases, 8.3 names).
        "-c", "core.protectNTFS=true",
        "-c", "core.protectHFS=true",
        # Transport allowlist: only HTTPS, so no file://, ext:: or ssh redirect is followed.
        "-c", "protocol.allow=never",
        "-c", "protocol.https.allow=always",
        # An empty value clears inherited helpers, so no stored credential is sent.
        "-c", "credential.helper=",
        # No askpass program either, so a credential request cannot spawn one or print a prompt.
        "-c", "core.askPass=",
    ]


def clone_command(
    git_executable: str,
    hooks_dir: Path,
    url: str,
    branch: str,
    depth: int,
    destination: Path,
) -> list[str]:
    return [
        git_executable,
        *hardened_git_config(hooks_dir),
        "clone",
        # Objects only: repository paths never become files on the local disk.
        "--no-checkout",
        # Shallow and single-branch: the audit needs the current tree and recent history only.
        "--depth", str(depth),
        "--single-branch",
        "--branch", branch,
        "--no-tags",
        # Submodules are other repositories, possibly on other hosts and transports.
        "--no-recurse-submodules",
        "--quiet",
        # Nothing after "--" can be read as an option, whatever the URL or path contains.
        "--",
        url,
        str(destination),
    ]


def git_environment(
    clone_root: str | None = None, allowlist: tuple[str, ...] | None = None
) -> dict[str, str]:
    """Environment for every git subprocess; with ``clone_root``, GITHUB_TOKEN is added
    as an HTTP header scoped to that host."""
    if allowlist is None:
        allowlist = load_settings().repository.git_env_allowlist
    # An allowlist rather than a filter: API keys and any other secret in the parent
    # environment must never reach git or a helper it might start. This also drops
    # inherited GIT_* variables (GIT_DIR, GIT_CONFIG_PARAMETERS, ...) and SSH_ASKPASS.
    allowed = {name.upper() for name in allowlist}
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment.update(
        {
            # Fail instead of waiting for a username on a missing or private repository.
            "GIT_TERMINAL_PROMPT": "0",
            # Do not download LFS objects through the smudge filter.
            "GIT_LFS_SKIP_SMUDGE": "1",
            # The machine's own git config (url.insteadOf, helpers, hooks) must not apply.
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            # A path given to `git log --` is a plain path, never pathspec magic or a glob.
            "GIT_LITERAL_PATHSPECS": "1",
        }
    )
    token = github_token() if clone_root is not None else None
    if token is not None:
        # Config from the environment keeps the token out of argv, which other local
        # users can read, and the URL scope keeps it to the clone host. Added after
        # the GIT_* filter above, which would otherwise drop it.
        credentials = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
        environment.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": f"http.{clone_root.rstrip('/')}/.extraheader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credentials}",
            }
        )
    return environment


def clone_url(clone_root: str, address: RepositoryAddress) -> str:
    return f"{clone_root.rstrip('/')}/{address.path}.git"


def parse_ls_tree(output: bytes) -> list[TreeBlob]:
    """Regular-file blobs from ``git ls-tree -r -l -z`` output, in git's order."""
    blobs = []
    for record in output.split(b"\0"):
        meta, separator, raw_path = record.partition(b"\t")
        fields = meta.decode("ascii", errors="replace").split()
        if not separator or len(fields) != 4:
            continue
        mode, kind, object_id, size = fields
        if (
            kind != "blob"
            or mode not in _REGULAR_FILE_MODES
            or not _is_object_id(object_id)
            or not size.isdigit()
        ):
            continue
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError:
            # Such a path could not be named in a tool call or a report anyway.
            continue
        blobs.append(TreeBlob(path, object_id, int(size)))
    return blobs


def parse_git_log(output: str) -> list[dict[str, Any]]:
    """Commits from ``git log --name-only`` output in ``_LOG_FORMAT``.

    Records without a valid sha and date, or without the field separator that
    closes the message, are dropped.
    """
    commits = []
    for record in output.split(_RECORD_SEPARATOR):
        fields = record.split(_FIELD_SEPARATOR, 3)
        if len(fields) != 4:
            continue
        sha, date, author, rest = fields
        # rpartition: a separator inside the message must not be taken for its end.
        message, closed, paths = rest.rpartition(_FIELD_SEPARATOR)
        if not closed or not _is_object_id(sha) or not _is_iso_date(date):
            continue
        commits.append(
            {
                "sha": sha,
                "date": date,
                "author": author or None,
                "message": message.rstrip("\n"),
                "files": [line for line in paths.split("\n") if line],
            }
        )
    return commits


def _is_object_id(value: str) -> bool:
    return len(value) in _OBJECT_ID_LENGTHS and set(value) <= _HEX_DIGITS


def _is_iso_date(value: str) -> bool:
    try:
        datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _run_git(
    args: list[str],
    action: str,
    timeout: float,
    output_limit: int | None = None,
    *,
    environment: dict[str, str] | None = None,
) -> bytes:
    """Run git with a deadline; stdout is read (at most ``output_limit`` bytes) only when asked."""
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(
            args,
            env=environment if environment is not None else git_environment(),
            stdin=subprocess.DEVNULL,
            # No pipe unless the output is needed: a pipe inherited by a leftover
            # git-remote-https would otherwise keep the deadline from firing on Windows.
            stdout=subprocess.PIPE if output_limit is not None else subprocess.DEVNULL,
            # Never read: it can echo URLs and credential prompts.
            stderr=subprocess.DEVNULL,
            # A process group of its own (POSIX), so a timeout also stops git's helpers.
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise _GitFailure("git executable not found") from exc
    except OSError as exc:
        raise _GitFailure("git could not be started") from exc
    try:
        output = b""
        if output_limit is not None and process.stdout is not None:
            read = _read_bounded(process.stdout, output_limit, deadline)
            if read is None:
                raise subprocess.TimeoutExpired(args, timeout)
            output = read
            if len(output) > output_limit:
                _stop(process)
                raise _GitFailure(f"git {action} output exceeded the configured limit")
        returncode = process.wait(timeout=max(deadline - time.monotonic(), 0))
    except subprocess.TimeoutExpired as exc:
        _stop(process)
        raise _GitFailure(f"git {action} timed out after {timeout:g} seconds") from exc
    finally:
        if process.stdout is not None:
            process.stdout.close()
    if returncode != 0:
        raise _GitFailure(f"git {action} exited with status {returncode}")
    return output


def _read_bounded(stream: IO[bytes], limit: int, deadline: float) -> bytes | None:
    """At most ``limit + 1`` bytes, or None if the stream is still open at the deadline."""
    # A thread, because a blocking read has no timeout of its own.
    chunks: list[bytes] = []
    reader = threading.Thread(target=lambda: chunks.append(stream.read(limit + 1)), daemon=True)
    reader.start()
    reader.join(max(deadline - time.monotonic(), 0))
    return None if reader.is_alive() or not chunks else chunks[0]


def _stop(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The group is already gone; wait() below still reaps the child.
            pass
    process.wait()


class CloneSource:
    """A hardened shallow clone without a checkout; ``close()`` removes its temporary directory."""

    backend = "clone"

    def __init__(self, address: RepositoryAddress, metadata: RepositoryMetadata, settings: Settings):
        repository = settings.repository
        self.metadata: RepositoryMetadata | None = metadata
        self.default_branch = metadata.default_branch
        self.fallback_reason: str | None = None
        self.cleanup_note: str | None = None
        self._git_executable = repository.git_executable
        self._timeout = repository.clone_timeout_seconds
        self._max_log_bytes = settings.git_log.max_output_bytes
        self._workspace = Path(tempfile.mkdtemp(prefix=repository.temp_dir_prefix))
        self._hooks_dir = self._workspace / "hooks"
        self.root = self._workspace / "checkout"
        try:
            self._hooks_dir.mkdir()
            command = clone_command(
                self._git_executable,
                self._hooks_dir,
                clone_url(settings.github.clone_root, address),
                self.default_branch,
                repository.clone_depth,
                self.root,
            )
            # Only the clone talks to the network, so only it gets the token.
            _run_git(
                command,
                "clone",
                self._timeout,
                environment=git_environment(
                    settings.github.clone_root, repository.git_env_allowlist
                ),
            )
            self._load_tree(repository)
        except BaseException:
            self.close()
            raise

    def _load_tree(self, repository: RepositorySettings) -> None:
        listing = self._git(
            "ls-tree", ["-r", "-l", "-z", "HEAD"], repository.max_tree_listing_bytes
        )
        blobs = parse_ls_tree(listing)
        # GitHub's packed size can be stale or zero, so the real tree is checked too.
        if len(blobs) > repository.max_checkout_files:
            raise _GitFailure("repository exceeds repository.max_checkout_files")
        if sum(blob.size for blob in blobs) > repository.max_checkout_mb * _BYTES_PER_MB:
            raise _GitFailure("repository exceeds repository.max_checkout_mb")
        listed = blobs[: repository.max_listed_files]
        self.files: dict[str, int | None] = {blob.path: blob.size for blob in listed}
        self._object_ids = {blob.path: blob.object_id for blob in listed}
        self.truncated = len(blobs) > len(listed)

    def _git(self, action: str, arguments: list[str], output_limit: int) -> bytes:
        command = [
            self._git_executable,
            *hardened_git_config(self._hooks_dir),
            "-C", str(self.root),
            action,
            *arguments,
        ]
        return _run_git(command, action, self._timeout, output_limit)

    def read_bytes(self, path: str, limit: int) -> bytes:
        object_id = self._object_ids.get(path)
        if object_id is None:
            raise RepositoryError(f"File is not present in the repository tree: {path}")
        size = self.files[path]
        if size is not None and size > limit:
            raise RepositoryError(f"File exceeds the {limit}-byte inspection limit: {path}")
        # By object id, not HEAD:<path>, so git never interprets the path.
        return self._git("cat-file", ["blob", object_id], limit)

    def commits(self, max_commits: int, path: str | None = None) -> list[dict[str, Any]]:
        arguments = ["-n", str(max_commits), "--name-only", _LOG_FORMAT]
        if path is not None:
            arguments += ["--", path]
        output = self._git("log", arguments, self._max_log_bytes)
        return parse_git_log(output.decode("utf-8", errors="replace"))

    def close(self) -> None:
        if not _remove_tree(self._workspace):
            self.cleanup_note = _CLEANUP_NOTE


class LocalDirSource:
    """An existing directory on disk, used by tests and offline evaluation; never modified."""

    backend = "local"

    def __init__(self, root: Path, settings: Settings, default_branch: str):
        self.root = root
        self.default_branch = default_branch
        # A plain directory has no GitHub metadata.
        self.metadata: RepositoryMetadata | None = None
        self.fallback_reason: str | None = None
        self.cleanup_note: str | None = None
        self.files, self.truncated = _list_directory(root, settings.repository.max_listed_files)

    def read_bytes(self, path: str, limit: int) -> bytes:
        return _read_directory_file(self.root, path, limit)

    def commits(self, max_commits: int, path: str | None = None) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        """The directory belongs to the caller, so nothing is removed."""


def _list_directory(root: Path, max_listed_files: int) -> tuple[dict[str, int | None], bool]:
    resolved_root = root.resolve()
    files: dict[str, int | None] = {}
    for current, dirnames, filenames in os.walk(root):
        base = Path(current)
        # os.walk does not follow symlinks; the resolve check also stops Windows junctions.
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name != ".git" and (base / name).resolve().is_relative_to(resolved_root)
        )
        for name in sorted(filenames):
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            if len(files) >= max_listed_files:
                return files, True
            files[path.relative_to(root).as_posix()] = info.st_size
    return files, False


def _read_directory_file(root: Path, path: str, limit: int) -> bytes:
    parts = path.split("/")
    if (
        not path
        or "\x00" in path
        or "\\" in path
        or PurePosixPath(path).is_absolute()
        or PureWindowsPath(path).drive
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RepositoryError("File path must be a normalized repository-relative path.")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise RepositoryError(f"Refusing to follow a symbolic link: {path}")
    if not current.resolve().is_relative_to(root.resolve()):
        raise RepositoryError(f"File path leaves the repository checkout: {path}")
    try:
        info = current.lstat()
    except FileNotFoundError as exc:
        raise RepositoryError(f"File is not present in the repository tree: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RepositoryError(f"Not a regular file: {path}")
    try:
        with current.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError as exc:
        raise RepositoryError(f"File could not be read: {path}") from exc
    if len(data) > limit:
        raise RepositoryError(f"File exceeds the {limit}-byte inspection limit: {path}")
    return data


def _remove_tree(path: Path) -> bool:
    """Delete ``path``; returns False if it is still there after one retry.

    Never raises, so cleanup cannot hide the error that made the caller close early.
    The retry covers a git process on Windows that was still releasing file handles.
    """
    try:
        _rmtree(path)
    except OSError:
        try:
            _rmtree(path)
        except OSError:
            return False
    return True


def _rmtree(path: Path) -> None:
    if not path.exists():
        return
    # POSIX unlinks read-only files without help, and chmod there could follow a symlink.
    if os.name != "nt":
        shutil.rmtree(path)
    elif sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_clear_read_only)
    else:
        shutil.rmtree(path, onerror=_clear_read_only)


def _clear_read_only(function: Any, target: str, _error: Any) -> None:
    # Git writes pack files read-only, which rmtree cannot delete on Windows.
    os.chmod(target, stat.S_IWRITE)
    function(target)
