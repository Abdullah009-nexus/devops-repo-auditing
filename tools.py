"""Read-only tools for inspecting public GitHub repositories."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


API_ROOT = "https://api.github.com"
RAW_ROOT = "https://raw.githubusercontent.com"
REQUEST_TIMEOUT = 15
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 256 * 1024
MAX_LISTED_FILES = 5000
MAX_SEARCH_FILES = 100
MAX_SEARCH_MATCHES = 50
SEARCHABLE_SUFFIXES = {
    ".cfg", ".conf", ".css", ".dockerfile", ".env", ".example", ".go",
    ".ini", ".java", ".js", ".json", ".md", ".mk", ".php", ".properties",
    ".py", ".rb", ".rs", ".sh", ".sql", ".tf", ".toml", ".ts", ".txt",
    ".xml", ".yaml", ".yml", ".yml",
}
SENSITIVE_ASSIGNMENT = re.compile(
    r"""(?ix)
    (\b(?:api[_-]?(?:key|token)|client[_-]?secret|secret(?:[_-]?key)?|token|
        password|passwd|credential|private[_-]?key)\b\s*[:=]\s*)
    (?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;#]+)
    """
)
KNOWN_TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
BEARER_TOKEN = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}")
PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
    re.DOTALL,
)


class RepositoryError(Exception):
    """Raised when a repository cannot be inspected safely."""


def redact_secrets(text: str) -> str:
    """Remove likely credential values before content is returned to the agent."""
    text = PRIVATE_KEY_BLOCK.sub("[REDACTED PRIVATE KEY]", text)
    text = SENSITIVE_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = KNOWN_TOKEN.sub("[REDACTED TOKEN]", text)
    return BEARER_TOKEN.sub(r"\1[REDACTED]", text)


def _request_bytes(url: str, *, accept: str, limit: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "devops-repository-auditing-agent/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            data = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RepositoryError("GitHub resource was not found or is not public.") from exc
        if exc.code == 403:
            raise RepositoryError(
                "GitHub denied the request; check API rate limits and repository access."
            ) from exc
        raise RepositoryError(f"GitHub request failed with HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RepositoryError(f"GitHub request failed: {exc.reason if isinstance(exc, urllib.error.URLError) else 'timed out'}.") from exc
    if len(data) > limit:
        raise RepositoryError("GitHub response exceeded the configured size limit.")
    return data


def _request_json(url: str, *, limit: int = MAX_RESPONSE_BYTES) -> dict[str, Any]:
    raw = _request_bytes(
        url,
        accept="application/vnd.github+json",
        limit=limit,
    )
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RepositoryError("GitHub returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RepositoryError("GitHub returned an unexpected response.")
    return parsed


@dataclass(frozen=True)
class RepositoryAddress:
    owner: str
    name: str
    url: str

    @classmethod
    def parse(cls, value: str) -> "RepositoryAddress":
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
        if not owner or not name:
            raise RepositoryError("Repository owner and name are required.")
        canonical = f"https://github.com/{owner}/{name}"
        return cls(owner=owner, name=name, url=canonical)


class GitHubRepository:
    """Minimal read-only GitHub API client with bounded file inspection."""

    def __init__(self, address: RepositoryAddress):
        self.address = address
        repo_path = f"{API_ROOT}/repos/{address.owner}/{address.name}"
        metadata = _request_json(repo_path)
        if metadata.get("private") is True:
            raise RepositoryError("Only public GitHub repositories are supported.")
        default_branch = metadata.get("default_branch")
        if not isinstance(default_branch, str) or not default_branch:
            raise RepositoryError("GitHub repository has no discoverable default branch.")
        self.branch = default_branch
        encoded_branch = urllib.parse.quote(self.branch, safe="")
        tree_url = (
            f"{repo_path}/git/trees/{encoded_branch}?recursive=1"
        )
        tree = _request_json(tree_url)
        entries = tree.get("tree")
        if not isinstance(entries, list):
            raise RepositoryError("GitHub did not return a repository file tree.")
        self.tree_truncated = tree.get("truncated") is True
        self.files: dict[str, dict[str, Any]] = {}
        for entry in entries[:MAX_LISTED_FILES]:
            if (
                isinstance(entry, dict)
                and entry.get("type") == "blob"
                and isinstance(entry.get("path"), str)
            ):
                self.files[entry["path"]] = entry
        self.list_truncated = self.tree_truncated or len(entries) > MAX_LISTED_FILES
        self._read_cache: dict[str, str] = {}
        self._inspected_paths: set[str] = set()

    def list_files(self) -> dict[str, Any]:
        paths = sorted(self.files)
        result: dict[str, Any] = {
            "repository": self.address.url,
            "default_branch": self.branch,
            "files": paths,
            "file_count_returned": len(paths),
            "truncated": self.list_truncated,
        }
        if self.list_truncated:
            result["notice"] = (
                "The GitHub file tree was truncated; absence of a path is not "
                "evidence that the file does not exist."
            )
        return result

    @staticmethod
    def _validate_path(path: str) -> str:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise RepositoryError("A non-empty repository-relative file path is required.")
        normalized = path.replace("\\", "/")
        parts = normalized.split("/")
        if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise RepositoryError("File path must be a normalized repository-relative path.")
        return normalized

    def read_file(self, path: str) -> dict[str, Any]:
        normalized = self._validate_path(path)
        entry = self.files.get(normalized)
        if entry is None:
            raise RepositoryError(f"File is not present in the repository tree: {normalized}")
        size = entry.get("size")
        if isinstance(size, int) and size > MAX_FILE_BYTES:
            raise RepositoryError(
                f"File exceeds the {MAX_FILE_BYTES}-byte inspection limit: {normalized}"
            )
        if normalized not in self._read_cache:
            encoded_path = urllib.parse.quote(normalized, safe="/")
            encoded_branch = urllib.parse.quote(self.branch, safe="")
            url = (
                f"{RAW_ROOT}/{self.address.owner}/{self.address.name}/"
                f"{encoded_branch}/{encoded_path}"
            )
            raw = _request_bytes(
                url,
                accept="application/octet-stream",
                limit=MAX_FILE_BYTES,
            )
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RepositoryError(f"File is not UTF-8 text: {normalized}") from exc
            self._read_cache[normalized] = content
        self._inspected_paths.add(normalized)
        lines = self._read_cache[normalized].splitlines()
        rendered = "\n".join(
            f"{number}: {redact_secrets(line)}"
            for number, line in enumerate(lines, start=1)
        )
        return {
            "path": normalized,
            "line_count": len(lines),
            "content": rendered,
            "secrets_redacted": True,
        }

    def search_code(self, pattern: str) -> dict[str, Any]:
        if not isinstance(pattern, str) or not pattern.strip():
            raise RepositoryError("Search pattern must be a non-empty string.")
        if len(pattern) > 200:
            raise RepositoryError("Search pattern must not exceed 200 characters.")
        needle = pattern.casefold()
        candidates = [
            path
            for path, entry in sorted(self.files.items())
            if self._is_searchable(path, entry)
        ]
        scanned: list[str] = []
        matches: list[dict[str, Any]] = []
        unavailable: list[dict[str, str]] = []
        for path in candidates[:MAX_SEARCH_FILES]:
            try:
                self.read_file(path)
            except RepositoryError as exc:
                unavailable.append({"file": path, "reason": str(exc)})
                continue
            scanned.append(path)
            content = self._read_cache[path]
            for number, line in enumerate(content.splitlines(), start=1):
                if needle in line.casefold():
                    matches.append(
                        {
                            "file": path,
                            "line": number,
                            "text": redact_secrets(line),
                        }
                    )
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        return {
                            "pattern": pattern,
                            "matches": matches,
                            "files_scanned": len(scanned),
                            "unavailable_files": unavailable,
                            "scan_truncated": True,
                            "secrets_redacted": True,
                        }
        return {
            "pattern": pattern,
            "matches": matches,
            "files_scanned": len(scanned),
            "unavailable_files": unavailable,
            "scan_truncated": len(candidates) > len(scanned),
            "secrets_redacted": True,
        }

    @staticmethod
    def _is_searchable(path: str, entry: dict[str, Any]) -> bool:
        size = entry.get("size")
        suffix = "." + path.rsplit(".", 1)[-1].casefold() if "." in path else ""
        basename = path.rsplit("/", 1)[-1].casefold()
        text_candidate = (
            suffix in SEARCHABLE_SUFFIXES
            or basename in {
                "dockerfile", "makefile", "procfile", "jenkinsfile",
                ".dockerignore", ".gitignore",
            }
        )
        return (
            text_candidate
            and isinstance(size, int)
            and size <= MAX_FILE_BYTES
        )

    def git_log(self) -> dict[str, Any]:
        url = (
            f"{API_ROOT}/repos/{self.address.owner}/{self.address.name}"
            "/commits?per_page=10"
        )
        raw = _request_bytes(
            url,
            accept="application/vnd.github+json",
            limit=2 * 1024 * 1024,
        )
        try:
            commits = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RepositoryError("GitHub returned invalid commit history JSON.") from exc
        if not isinstance(commits, list):
            raise RepositoryError("GitHub returned an unexpected commit history response.")
        summaries = []
        for commit in commits:
            if not isinstance(commit, dict):
                continue
            details = commit.get("commit")
            if not isinstance(details, dict):
                continue
            author = details.get("author")
            summaries.append(
                {
                    "sha": str(commit.get("sha", ""))[:12],
                    "date": author.get("date") if isinstance(author, dict) else None,
                    "message": redact_secrets(str(details.get("message", ""))[:1000]),
                }
            )
        return {"commits": summaries}

    def validate_finding_location(self, path: str, line: int) -> bool:
        normalized = self._validate_path(path)
        if normalized not in self._inspected_paths or normalized not in self._read_cache:
            return False
        lines = self._read_cache[normalized].splitlines()
        return 1 <= line <= len(lines)

    def has_inspected_path(self, path: str) -> bool:
        normalized = self._validate_path(path)
        return normalized in self._inspected_paths

    def inspected_files(self) -> dict[str, str]:
        return {
            path: self._read_cache[path]
            for path in self._inspected_paths
            if path in self._read_cache
        }

    def supports_audit_finding(
        self,
        rule_id: str,
        path: str,
        line: int,
        evidence: str,
    ) -> bool:
        normalized = self._validate_path(path)
        if normalized not in self._inspected_paths:
            return False
        from audit import supports_finding

        return supports_finding(
            rule_id,
            normalized,
            line,
            evidence,
            {normalized: self._read_cache[normalized]},
        )


def build_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List paths in the public repository's default-branch tree.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 repository file; likely secrets are redacted.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search text files for a literal, case-insensitive string.",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_log",
                "description": "Read up to ten recent commit summaries from GitHub.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "audit_inspected_files",
                "description": (
                    "Run conservative security and CI checks on files already read. "
                    "Use after inspecting relevant configuration/source files; results "
                    "are partial evidence, not a complete repository audit."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]


class RepositoryTools:
    def __init__(self, repository: GitHubRepository):
        self.repository = repository

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "list_files":
            self._validate_arguments(name, arguments, set())
            return self.repository.list_files()
        if name == "read_file":
            self._validate_arguments(name, arguments, {"path"})
            path = arguments.get("path")
            if not isinstance(path, str):
                raise RepositoryError("Tool argument 'path' must be a string.")
            return self.repository.read_file(path)
        if name == "search_code":
            self._validate_arguments(name, arguments, {"pattern"})
            pattern = arguments.get("pattern")
            if not isinstance(pattern, str):
                raise RepositoryError("Tool argument 'pattern' must be a string.")
            return self.repository.search_code(pattern)
        if name == "git_log":
            self._validate_arguments(name, arguments, set())
            return self.repository.git_log()
        if name == "audit_inspected_files":
            self._validate_arguments(name, arguments, set())
            from audit import audit_inspected_files

            return audit_inspected_files(self.repository.inspected_files())
        raise RepositoryError(f"Unsupported inspection tool: {name}")

    @staticmethod
    def _validate_arguments(
        name: str,
        arguments: dict[str, Any],
        required: set[str],
    ) -> None:
        missing = required - arguments.keys()
        unexpected = arguments.keys() - required
        if missing:
            raise RepositoryError(
                f"Tool '{name}' is missing required arguments: {', '.join(sorted(missing))}."
            )
        if unexpected:
            raise RepositoryError(
                f"Tool '{name}' received unsupported arguments: "
                f"{', '.join(sorted(unexpected))}."
            )
