"""Read-only tools for inspecting public GitHub repositories."""

from __future__ import annotations

import fnmatch
import re
from types import TracebackType
from typing import Any, Callable

from auditor.checks import audit_inspected_files, supports_finding
from auditor.redaction import redact_secrets, redacted_lines
from auditor.repository import (
    RepositoryAddress,
    RepositoryError,
    RepositorySource,
    open_source,
)
from auditor.settings import GitLogSettings, Settings


# GitHub Actions workflow syntax, not tunables. YAML 1.1 reads a bare `on` as true,
# so the key is matched as written rather than parsed.
_TRIGGER_KEY = "on"
_JOBS_KEY = "jobs"
_PERMISSIONS_KEY = "permissions"
_YAML_KEY = re.compile(r"""^(?P<indent> *)["']?(?P<key>[A-Za-z0-9_][\w.-]*)["']?\s*:(?P<value>.*)$""")
_YAML_LIST_ITEM = re.compile(r"""^ *- +["']?(?P<key>[A-Za-z0-9_][\w-]*)["']?\s*(?:#.*)?$""")
_YAML_COMMENT = re.compile(r"\s+#.*$")


class GitHubRepository:
    """Bounded, redacted inspection of one repository snapshot.

    Use it as a context manager: closing it deletes the temporary clone.
    """

    def __init__(
        self,
        address: RepositoryAddress,
        settings: Settings,
        source: RepositorySource | None = None,
    ):
        self.address = address
        self.settings = settings
        self.source = source if source is not None else open_source(address, settings)
        self._read_cache: dict[str, str] = {}
        self._redacted_cache: dict[str, list[str]] = {}
        self._inspected_paths: set[str] = set()
        self._listing_complete = False

    def __enter__(self) -> GitHubRepository:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.source.close()

    def list_files(
        self,
        path_prefix: str | None = None,
        glob: str | None = None,
    ) -> dict[str, Any]:
        source = self.source
        prefix = self._validate_prefix(path_prefix) if path_prefix is not None else None
        if glob is not None:
            self._check_pattern_length(glob, "Tool argument 'glob'")
        matching = [
            path
            for path in sorted(source.files)
            if (prefix is None or path.startswith(prefix))
            and (glob is None or fnmatch.fnmatchcase(path, glob))
        ]
        if not source.truncated:
            self._listing_complete = True
        limit = self.settings.repository.max_list_entries
        returned = matching[:limit]
        result: dict[str, Any] = {
            "repository": self.address.url,
            "default_branch": source.default_branch,
            "backend": source.backend,
            "files": returned,
            "total_files": len(matching),
            "returned": len(returned),
            "truncated": source.truncated or len(matching) > len(returned),
            "tree_truncated": source.truncated,
        }
        if prefix is not None:
            result["path_prefix"] = prefix
        if glob is not None:
            result["glob"] = glob
        if source.fallback_reason is not None:
            result["fallback_reason"] = source.fallback_reason
        if source.truncated:
            result["notice"] = (
                "The file listing was truncated; absence of a path is not "
                "evidence that the file does not exist."
            )
        elif len(matching) > len(returned):
            result["notice"] = (
                f"Only the first {limit} matching paths are listed; narrow the "
                "listing with path_prefix or glob to see the rest."
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

    @staticmethod
    def _validate_prefix(prefix: str) -> str:
        normalized = prefix.replace("\\", "/")
        # A trailing slash is how a caller says "this directory only".
        parts = normalized.removesuffix("/").split("/")
        if (
            not normalized
            or "\x00" in normalized
            or normalized.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise RepositoryError("path_prefix must be a normalized repository-relative path.")
        return normalized

    def _check_pattern_length(self, pattern: str, name: str) -> None:
        if not pattern.strip():
            raise RepositoryError(f"{name} must be a non-empty string.")
        limit = self.settings.search.max_pattern_length
        if len(pattern) > limit:
            raise RepositoryError(f"{name} must not exceed {limit} characters.")

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        normalized = self._validate_path(path)
        repository = self.settings.repository
        ranged = start_line is not None or end_line is not None
        if ranged:
            self._check_size(normalized, repository.max_range_file_bytes, ranged=True)
            self._load(normalized, repository.max_range_file_bytes)
        else:
            self._load_whole(normalized)
        self._inspected_paths.add(normalized)
        lines = self._redacted(normalized)
        first = 1 if start_line is None else start_line
        if first < 1 or (end_line is not None and end_line < first):
            raise RepositoryError("Line range must satisfy 1 <= start_line <= end_line.")
        if first > max(len(lines), 1):
            raise RepositoryError(
                f"start_line {first} is past the end of {normalized} ({len(lines)} lines)."
            )
        requested_last = len(lines) if end_line is None else min(end_line, len(lines))
        last = min(requested_last, first + repository.max_read_lines - 1)
        rendered = "\n".join(
            f"{number}: {lines[number - 1]}" for number in range(first, last + 1)
        )
        result: dict[str, Any] = {
            "path": normalized,
            "line_count": len(lines),
            "start_line": first,
            "end_line": last,
            "content": rendered,
            "truncated": last < requested_last,
            "secrets_redacted": True,
        }
        if last < requested_last:
            result["next_start_line"] = last + 1
        return result

    def _check_size(self, path: str, limit: int, *, ranged: bool) -> None:
        if path not in self.source.files:
            raise RepositoryError(f"File is not present in the repository tree: {path}")
        size = self.source.files[path]
        if size is None and path in self._read_cache:
            size = len(self._read_cache[path].encode("utf-8"))
        if size is None or size <= limit:
            return
        if not ranged and size <= self.settings.repository.max_range_file_bytes:
            raise RepositoryError(
                f"File exceeds the {limit}-byte whole-file limit: {path}; read it "
                "with start_line/end_line instead."
            )
        raise RepositoryError(f"File exceeds the {limit}-byte inspection limit: {path}")

    def _load_whole(self, path: str) -> str:
        limit = self.settings.repository.max_file_bytes
        self._check_size(path, limit, ranged=False)
        return self._load(path, limit)

    def _load(self, path: str, limit: int) -> str:
        if path not in self._read_cache:
            raw = self.source.read_bytes(path, limit)
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RepositoryError(f"File is not UTF-8 text: {path}") from exc
            self._read_cache[path] = content
        return self._read_cache[path]

    def _redacted(self, path: str) -> list[str]:
        lines = self._redacted_cache.get(path)
        if lines is None:
            # Redacted as a whole: a private key spans lines, so per-line redaction misses it.
            lines = redacted_lines(self._read_cache[path])
            self._redacted_cache[path] = lines
        return lines

    def search_code(
        self,
        pattern: str,
        regex: bool = False,
        path_glob: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(pattern, str):
            raise RepositoryError("Search pattern must be a non-empty string.")
        self._check_pattern_length(pattern, "Search pattern")
        if path_glob is not None:
            self._check_pattern_length(path_glob, "Tool argument 'path_glob'")
        search = self.settings.search
        result: dict[str, Any] = {
            # The pattern may be the model echoing a secret it saw elsewhere.
            "pattern": redact_secrets(pattern),
            "mode": "regex" if regex else "literal",
        }
        if path_glob is not None:
            result["path_glob"] = path_glob
        if regex and re.search(search.alternation_repeat_pattern, pattern):
            result["error"] = (
                "Regular expression refused: a repeated group containing alternatives can take "
                "exponential time. Search for each alternative separately, or drop the repeat."
            )
            return result
        if regex and re.search(search.nested_repeat_pattern, pattern):
            result["error"] = (
                "Regular expression refused: a repeated group that repeats inside can take "
                "exponential time. Rewrite it without nested repetition, e.g. (ab)+ or a+b."
            )
            return result
        try:
            matches_line = _line_matcher(pattern, regex)
        except re.error as exc:
            result["error"] = redact_secrets(f"Invalid regular expression: {exc}.")
            return result
        candidates = [
            path
            for path, size in sorted(self.source.files.items())
            if self._is_searchable(path, size)
            and (path_glob is None or fnmatch.fnmatchcase(path, path_glob))
        ]
        scanned = 0
        skipped = 0
        matches: list[dict[str, Any]] = []
        unavailable: list[dict[str, str]] = []
        scan_truncated = len(candidates) > search.max_files
        for path in candidates[:search.max_files]:
            try:
                self._load_whole(path)
            except RepositoryError as exc:
                unavailable.append({"file": path, "reason": str(exc)})
                continue
            self._inspected_paths.add(path)
            scanned += 1
            # Matching the redacted text means a search cannot confirm a guess at a
            # secret value, and each file is redacted once however often it is searched.
            for number, line in enumerate(self._redacted(path), start=1):
                if regex and len(line) > search.max_scan_line_chars:
                    skipped += 1
                    continue
                if matches_line(line):
                    matches.append(self._match(path, number, line))
                    if len(matches) >= search.max_matches:
                        break
            if len(matches) >= search.max_matches:
                scan_truncated = True
                break
        result.update(
            {
                "matches": matches,
                "files_scanned": scanned,
                "unavailable_files": unavailable,
                "scan_truncated": scan_truncated,
                "secrets_redacted": True,
            }
        )
        if regex:
            result["lines_skipped"] = skipped
            if skipped:
                result["notice"] = (
                    f"{skipped} lines longer than {search.max_scan_line_chars} characters "
                    "were not searched in regex mode; use a literal search or read_file."
                )
        return result

    def _match(self, path: str, number: int, line: str) -> dict[str, Any]:
        limit = self.settings.search.max_line_chars
        match: dict[str, Any] = {"file": path, "line": number, "text": line[:limit]}
        if len(line) > limit:
            match["text_truncated"] = True
        return match

    def _is_searchable(self, path: str, size: int | None) -> bool:
        suffix = "." + path.rsplit(".", 1)[-1].casefold() if "." in path else ""
        basename = path.rsplit("/", 1)[-1].casefold()
        text_candidate = (
            suffix in self.settings.search.searchable_suffixes
            or basename in self.settings.search.searchable_filenames
        )
        return (
            text_candidate
            and size is not None
            and size <= self.settings.repository.max_file_bytes
        )

    def git_log(
        self,
        max_commits: int | None = None,
        path: str | None = None,
    ) -> dict[str, Any]:
        git_log = self.settings.git_log
        count = git_log.default_commits if max_commits is None else max_commits
        count = min(max(count, 1), git_log.max_commits)
        normalized = self._validate_path(path) if path is not None else None
        commits = self.source.commits(count, normalized)[:count]
        summaries = [self._commit_summary(commit) for commit in commits]
        authors = {commit["author"] for commit in commits if commit.get("author")}
        result: dict[str, Any] = {
            "backend": self.source.backend,
            "max_commits": count,
            "commits": summaries,
            "author_count": len(authors),
        }
        if normalized is not None:
            result["path"] = normalized
        if self.source.backend == "local":
            result["notice"] = "Commit history is not available for a local directory source."
        elif commits and all(commit.get("files") is None for commit in commits):
            result["notice"] = "Changed files per commit are listed only on the clone backend."
        return result

    def _commit_summary(self, commit: dict[str, Any]) -> dict[str, Any]:
        git_log = self.settings.git_log
        # Redact before cutting: a secret split at the limit would otherwise partly survive.
        message = redact_secrets(commit["message"])
        summary: dict[str, Any] = {
            "sha": commit["sha"][: git_log.sha_length],
            "date": commit["date"],
            "message": message[: git_log.message_max_chars],
        }
        if len(message) > git_log.message_max_chars:
            summary["message_truncated"] = True
        files = commit.get("files")
        if files is not None:
            limit = git_log.max_files_per_commit
            summary["files"] = [redact_secrets(name) for name in files[:limit]]
            if len(files) > limit:
                summary["files_omitted"] = len(files) - limit
        return summary

    def get_repo_metadata(self) -> dict[str, Any]:
        source = self.source
        metadata = source.metadata
        result: dict[str, Any] = {
            "repository": self.address.url,
            "backend": source.backend,
            "default_branch": self._metadata_text(source.default_branch),
            "available": metadata is not None,
        }
        if metadata is None:
            result["notice"] = (
                "GitHub metadata is not available for this source; only the default "
                "branch is known."
            )
            return result
        result.update(
            {
                "archived": metadata.archived,
                "fork": metadata.fork,
                "private": metadata.private,
                "visibility": self._metadata_text(metadata.visibility),
                "license_spdx_id": self._metadata_text(metadata.license_spdx_id),
                "pushed_at": self._metadata_text(metadata.pushed_at),
                "size_kb": metadata.size_kib,
            }
        )
        return result

    def _metadata_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        return redact_secrets(value)[: self.settings.github.metadata_max_chars]

    def list_workflows(self) -> dict[str, Any]:
        workflows = self.settings.workflows
        directory = workflows.directory.rstrip("/") + "/"
        paths = [
            path
            for path in sorted(self.source.files)
            if path.startswith(directory)
            and any(path.casefold().endswith(suffix) for suffix in workflows.suffixes)
        ]
        scanned: list[dict[str, Any]] = []
        unreadable: list[dict[str, str]] = []
        for path in paths[: workflows.max_files]:
            try:
                self._load_whole(path)
            except RepositoryError as exc:
                unreadable.append({"file": path, "reason": str(exc)})
                continue
            self._inspected_paths.add(path)
            scanned.append({"file": path, **self._scan_workflow(self._redacted(path))})
        for path in paths[workflows.max_files:]:
            unreadable.append(
                {"file": path, "reason": f"not scanned: over workflows.max_files ({workflows.max_files})"}
            )
        result: dict[str, Any] = {
            "directory": workflows.directory,
            "workflows": scanned,
            "unreadable_files": unreadable,
            "secrets_redacted": True,
        }
        if self.source.truncated:
            result["notice"] = (
                "The file listing was truncated; workflows outside it are not listed."
            )
        return result

    def _scan_workflow(self, lines: list[str]) -> dict[str, Any]:
        """Line numbers of the trigger block, permissions and job ids, by indentation only."""
        limit = self.settings.workflows.max_items_per_file
        text_limit = self.settings.search.max_line_chars
        trigger_line: int | None = None
        triggers: list[dict[str, Any]] = []
        permissions: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        section: str | None = None
        child_indent: int | None = None
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key_match = _YAML_KEY.match(line)
            indent = len(line) - len(line.lstrip(" "))
            if key_match is not None and key_match["key"] == _PERMISSIONS_KEY:
                permissions.append({"line": number, "text": stripped[:text_limit]})
            if indent == 0:
                section, child_indent = None, None
                if key_match is None:
                    continue
                if key_match["key"] == _TRIGGER_KEY:
                    trigger_line = number
                    section = _TRIGGER_KEY
                    triggers.extend(
                        {"name": name, "line": number}
                        for name in _inline_triggers(key_match["value"])
                    )
                elif key_match["key"] == _JOBS_KEY:
                    section = _JOBS_KEY
                continue
            if section is None:
                continue
            if child_indent is None:
                child_indent = indent
            if indent != child_indent:
                continue
            item = key_match if key_match is not None else _YAML_LIST_ITEM.match(line)
            if item is None:
                continue
            if section == _TRIGGER_KEY:
                triggers.append({"name": item["key"], "line": number})
            elif key_match is not None:
                jobs.append({"id": item["key"], "line": number})
        result: dict[str, Any] = {
            "line_count": len(lines),
            "on_line": trigger_line,
            "triggers": triggers[:limit],
            "permissions": permissions[:limit],
            "jobs": jobs[:limit],
        }
        if max(len(triggers), len(permissions), len(jobs)) > limit:
            result["items_truncated"] = True
        return result

    def validate_finding_location(self, path: str, line: int) -> bool:
        normalized = self._validate_path(path)
        if normalized not in self._inspected_paths or normalized not in self._read_cache:
            return False
        lines = self._read_cache[normalized].splitlines()
        return 1 <= line <= len(lines)

    def listing_complete(self) -> bool:
        """True once list_files has run on a tree that was not truncated."""
        return self._listing_complete

    def has_listed_path(self, path: str) -> bool:
        return self._validate_path(path) in self.source.files

    def redacted_window(self, path: str, first: int, last: int) -> list[str]:
        """Redacted lines first..last (1-based, inclusive) of a file that was read."""
        normalized = self._validate_path(path)
        if normalized not in self._inspected_paths or normalized not in self._read_cache:
            return []
        return self._redacted(normalized)[max(first, 1) - 1 : max(last, 0)]

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
        return supports_finding(
            rule_id,
            normalized,
            line,
            evidence,
            {normalized: self._read_cache[normalized]},
        )


def _inline_triggers(value: str) -> list[str]:
    """Event names from `on: push` or `on: [push, pull_request]`; a flow mapping gives none."""
    inline = _YAML_COMMENT.sub("", value).strip()
    if inline.startswith("{"):
        return []
    names = (name.strip().strip("'\"") for name in inline.strip("[]").split(","))
    return [name for name in names if name]


def _line_matcher(pattern: str, regex: bool) -> Callable[[str], object]:
    """Case-insensitive matcher; raises re.error for an invalid regex."""
    if regex:
        return re.compile(pattern, re.IGNORECASE).search
    needle = pattern.casefold()
    return lambda line: needle in line.casefold()


def _object(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def build_tool_schemas(git_log: GitLogSettings) -> list[dict[str, Any]]:
    return [
        _function(
            "list_files",
            "List paths in the repository's default-branch tree, optionally filtered. "
            "Reports total_files and whether the listing was truncated.",
            _object(
                {
                    "path_prefix": {
                        "type": "string",
                        "description": "Only paths starting with this, e.g. '.github/workflows/'.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "fnmatch pattern on the full path; '*' also matches '/'.",
                    },
                }
            ),
        ),
        _function(
            "read_file",
            "Read a UTF-8 repository file with numbered lines; likely secrets are redacted. "
            "Large files are returned in pages; use start_line/end_line (1-based, inclusive).",
            _object(
                {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                ("path",),
            ),
        ),
        _function(
            "search_code",
            "Search text files, case-insensitively, for a literal string or a Python regex. "
            "Matching runs on redacted text, so credential values cannot be searched for.",
            _object(
                {
                    "pattern": {"type": "string"},
                    "regex": {"type": "boolean", "description": "Treat pattern as a regex."},
                    "path_glob": {
                        "type": "string",
                        "description": "fnmatch pattern limiting which paths are searched.",
                    },
                },
                ("pattern",),
            ),
        ),
        _function(
            "git_log",
            f"Read recent commit summaries (default {git_log.default_commits}, "
            f"at most {git_log.max_commits}), optionally only those touching a path.",
            _object(
                {
                    "max_commits": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": git_log.max_commits,
                    },
                    "path": {"type": "string"},
                }
            ),
        ),
        _function(
            "get_repo_metadata",
            "GitHub metadata for the repository: default branch, archived, fork, "
            "visibility, license and last push time.",
            _object({}),
        ),
        _function(
            "list_workflows",
            "List GitHub Actions workflow files with the line numbers of their "
            "triggers, permissions entries and job ids.",
            _object({}),
        ),
        _function(
            "audit_inspected_files",
            "Run the deterministic detectors on files already read, plus the path "
            "checks (committed secret files, tracked generated files) on the whole "
            "file listing. Use after inspecting relevant configuration/source files; "
            "results are partial evidence, not a complete repository audit.",
            _object(
                {
                    "rule_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only run the detectors of these rule ids.",
                    }
                }
            ),
        ),
    ]


# Required and optional arguments with their JSON types, per tool.
_TOOL_ARGUMENTS: dict[str, tuple[dict[str, type], dict[str, type]]] = {
    "list_files": ({}, {"path_prefix": str, "glob": str}),
    "read_file": ({"path": str}, {"start_line": int, "end_line": int}),
    "search_code": ({"pattern": str}, {"regex": bool, "path_glob": str}),
    "git_log": ({}, {"max_commits": int, "path": str}),
    "get_repo_metadata": ({}, {}),
    "list_workflows": ({}, {}),
    "audit_inspected_files": ({}, {"rule_ids": list}),
}
_TYPE_NAMES = {str: "a string", int: "an integer", bool: "a boolean", list: "a list of strings"}


class RepositoryTools:
    def __init__(self, repository: GitHubRepository):
        self.repository = repository

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = _TOOL_ARGUMENTS.get(name)
        if spec is None:
            raise RepositoryError(f"Unsupported inspection tool: {name}")
        values = self._validate_arguments(name, arguments, *spec)
        repository = self.repository
        if name == "audit_inspected_files":
            rule_ids = values.get("rule_ids")
            if rule_ids is not None and not all(isinstance(item, str) for item in rule_ids):
                raise RepositoryError("Tool argument 'rule_ids' must be a list of strings.")
            result = audit_inspected_files(
                repository.inspected_files(),
                paths=repository.source.files,
                rule_ids=rule_ids,
            )
            if repository.source.truncated:
                result["notice"] = (
                    "The file listing was truncated; path-based checks did not see every file."
                )
            return result
        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "list_files": repository.list_files,
            "read_file": repository.read_file,
            "search_code": repository.search_code,
            "git_log": repository.git_log,
            "get_repo_metadata": repository.get_repo_metadata,
            "list_workflows": repository.list_workflows,
        }
        return handlers[name](**values)

    def _audit(self, rule_ids: list[Any] | None) -> dict[str, Any]:
        if rule_ids is not None and not all(isinstance(item, str) for item in rule_ids):
            raise RepositoryError("Tool argument 'rule_ids' must be a list of strings.")
        source = self.repository.source
        result = audit_inspected_files(
            self.repository.inspected_files(), paths=sorted(source.files), rule_ids=rule_ids
        )
        if source.truncated:
            result["tree_truncated"] = True
            result["tree_notice"] = (
                "The file listing was truncated; path-based rules cover only the listed paths."
            )
        return result

    @staticmethod
    def _validate_arguments(
        name: str,
        arguments: dict[str, Any],
        required: dict[str, type],
        optional: dict[str, type],
    ) -> dict[str, Any]:
        # Models often send null for an optional argument they mean to leave out.
        present = {key: value for key, value in arguments.items() if value is not None}
        missing = required.keys() - present.keys()
        unexpected = arguments.keys() - required.keys() - optional.keys()
        if missing:
            raise RepositoryError(
                f"Tool '{name}' is missing required arguments: {', '.join(sorted(missing))}."
            )
        if unexpected:
            raise RepositoryError(
                f"Tool '{name}' received unsupported arguments: "
                f"{', '.join(sorted(unexpected))}."
            )
        types = {**required, **optional}
        for key, value in present.items():
            expected = types[key]
            # bool is an int subclass; `start_line: true` is a mistake, not 1.
            if (expected is int and isinstance(value, bool)) or not isinstance(value, expected):
                raise RepositoryError(
                    f"Tool argument '{key}' must be {_TYPE_NAMES[expected]}."
                )
        return present
