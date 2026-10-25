"""Typed settings for one audit run, plus a minimal .env loader.

Values come from the packaged ``config/defaults.toml``, then an optional ``--config``
file, then ``AUDITOR_<SECTION>_<KEY>`` environment variables; each later source
overrides earlier ones key by key. The defaults file must define every key, so the
code never falls back to a literal, and anything missing, unknown, mistyped or out
of range raises ``ConfigError``.
"""

from __future__ import annotations

import dataclasses
import os
import re
import tomllib
import typing
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULTS_PATH = Path(__file__).with_name("config") / "defaults.toml"
ENV_PREFIX = "AUDITOR_"
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TRUE_WORDS = frozenset({"true", "1", "yes"})
_FALSE_WORDS = frozenset({"false", "0", "no"})
# A git object id is 40 hex characters; a longer prefix is meaningless.
_SHA_HEX_LENGTH = 40
# GitHub's commits endpoint returns at most 100 commits per page.
_GITHUB_MAX_PER_PAGE = 100
# An unquoted .env value ends where whitespace and a "#" start a comment.
_INLINE_COMMENT = re.compile(r"\s+#")


class ConfigError(Exception):
    """Raised when settings or the .env file are missing, malformed or out of range."""


def _bounded(minimum: float, maximum: float | None = None) -> Any:
    return field(metadata={"min": minimum, "max": maximum})


def _https_url() -> Any:
    return field(metadata={"https_url": True})


def _one_of(*choices: str) -> Any:
    return field(metadata={"choices": choices})


def _regex() -> Any:
    return field(metadata={"regex": True})


def _name_part() -> Any:
    return field(metadata={"name_part": True})


@dataclass(frozen=True)
class GitHubSettings:
    api_root: str = _https_url()
    raw_root: str = _https_url()
    clone_root: str = _https_url()
    owner_pattern: str = _regex()
    name_pattern: str = _regex()
    user_agent: str
    request_timeout_seconds: float = _bounded(1)
    max_response_bytes: int = _bounded(1)
    commit_history_bytes: int = _bounded(1)
    metadata_max_chars: int = _bounded(1)


@dataclass(frozen=True)
class RepositorySettings:
    max_file_bytes: int = _bounded(1)
    max_listed_files: int = _bounded(1)
    backend: str = _one_of("clone", "api")
    clone_depth: int = _bounded(1)
    clone_timeout_seconds: float = _bounded(1)
    max_repo_size_mb: int = _bounded(1)
    max_checkout_files: int = _bounded(1)
    max_checkout_mb: int = _bounded(1)
    max_tree_listing_bytes: int = _bounded(1)
    max_list_entries: int = _bounded(1)
    max_read_lines: int = _bounded(1)
    max_range_file_bytes: int = _bounded(1)
    git_executable: str
    git_env_allowlist: tuple[str, ...]
    temp_dir_prefix: str = _name_part()


@dataclass(frozen=True)
class SearchSettings:
    max_files: int = _bounded(1)
    max_matches: int = _bounded(1)
    max_pattern_length: int = _bounded(1)
    max_line_chars: int = _bounded(1)
    max_scan_line_chars: int = _bounded(1)
    nested_repeat_pattern: str = _regex()
    alternation_repeat_pattern: str = _regex()
    searchable_suffixes: tuple[str, ...]
    searchable_filenames: tuple[str, ...]


@dataclass(frozen=True)
class GitLogSettings:
    default_commits: int = _bounded(1, _GITHUB_MAX_PER_PAGE)
    max_commits: int = _bounded(1, _GITHUB_MAX_PER_PAGE)
    message_max_chars: int = _bounded(1)
    sha_length: int = _bounded(1, _SHA_HEX_LENGTH)
    max_output_bytes: int = _bounded(1)
    max_files_per_commit: int = _bounded(1)


@dataclass(frozen=True)
class WorkflowSettings:
    directory: str
    suffixes: tuple[str, ...]
    max_files: int = _bounded(1)
    max_items_per_file: int = _bounded(1)


@dataclass(frozen=True)
class AgentSettings:
    max_turns: int = _bounded(1)
    max_tool_calls: int = _bounded(1)
    max_tool_output_chars: int = _bounded(1)
    max_report_findings: int = _bounded(1)
    max_wall_seconds: float = _bounded(1)
    finalize_wall_seconds: float = _bounded(1)
    finalize_turns: int = _bounded(1)
    max_finding_field_chars: int = _bounded(1)
    max_text_replies: int = _bounded(1)
    ledger_max_paths: int = _bounded(1)
    ledger_max_searches: int = _bounded(1)
    max_repeat_replies: int = _bounded(0)


@dataclass(frozen=True)
class ContextSettings:
    keep_recent_tool_results: int = _bounded(1)
    max_total_chars: int = _bounded(1)
    max_stub_args_chars: int = _bounded(1)


@dataclass(frozen=True)
class TraceSettings:
    enabled: bool
    directory: str
    max_string_chars: int = _bounded(1)


@dataclass(frozen=True)
class EvalSettings:
    line_tolerance: int = _bounded(0)


@dataclass(frozen=True)
class LlmSettings:
    default_base_url: str = _https_url()
    user_agent: str
    request_timeout_seconds: float = _bounded(1)
    max_response_bytes: int = _bounded(1)
    temperature: float = _bounded(0.0, 2.0)
    max_retries: int = _bounded(0)
    backoff_base_seconds: float = _bounded(0.0)
    backoff_max_seconds: float = _bounded(0.0)


@dataclass(frozen=True)
class Settings:
    github: GitHubSettings
    repository: RepositorySettings
    search: SearchSettings
    git_log: GitLogSettings
    workflows: WorkflowSettings
    agent: AgentSettings
    context: ContextSettings
    trace: TraceSettings
    eval: EvalSettings
    llm: LlmSettings


def load_settings(config_path: Path | None = None) -> Settings:
    """Merge defaults, the optional config file and AUDITOR_* variables, then validate."""
    merged = _read_toml(DEFAULTS_PATH)
    _check_known_keys(merged, DEFAULTS_PATH)
    _check_complete(merged)
    if config_path is not None:
        overrides = _read_toml(config_path)
        _check_known_keys(overrides, config_path)
        for section, values in overrides.items():
            merged[section].update(values)
    _apply_environment(merged)
    return _build(merged)


def load_env_file(path: Path) -> None:
    """Copy KEY=VALUE lines from ``path`` into os.environ without replacing existing variables.

    Values are never echoed in errors, because the file usually holds API keys.
    """
    try:
        # utf-8-sig: Windows editors often save .env with a byte-order mark.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read environment file {path}.") from exc
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY.fullmatch(key):
            raise ConfigError(f"{path} line {number}: expected KEY=VALUE.")
        if key not in os.environ:
            os.environ[key] = _unquote(value.strip())


def _unquote(value: str) -> str:
    """The value without its quotes or an inline comment; a "#" inside quotes is kept."""
    if value[:1] in {'"', "'"}:
        closing = value.find(value[0], 1)
        rest = value[closing + 1:].strip() if closing > 0 else ""
        if closing > 0 and (not rest or rest.startswith("#")):
            return value[1:closing]
        return value
    if value.startswith("#"):
        return ""
    return _INLINE_COMMENT.split(value, maxsplit=1)[0]


def _section_types() -> dict[str, type]:
    return typing.get_type_hints(Settings)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc.strerror}.") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def _check_known_keys(document: dict[str, Any], source: Path) -> None:
    sections = _section_types()
    for section, values in document.items():
        if section not in sections:
            raise ConfigError(f"{source}: unknown section [{section}].")
        if not isinstance(values, dict):
            raise ConfigError(f"{source}: [{section}] must be a table.")
        known = {item.name for item in dataclasses.fields(sections[section])}
        for key in values:
            if key not in known:
                raise ConfigError(f"{source}: unknown setting {section}.{key}.")


def _check_complete(document: dict[str, Any]) -> None:
    for section, section_type in _section_types().items():
        if section not in document:
            raise ConfigError(f"{DEFAULTS_PATH}: missing section [{section}].")
        for item in dataclasses.fields(section_type):
            if item.name not in document[section]:
                raise ConfigError(f"{DEFAULTS_PATH}: missing setting {section}.{item.name}.")


def _apply_environment(merged: dict[str, Any]) -> None:
    targets: dict[str, tuple[str, str, Any]] = {}
    for section, section_type in _section_types().items():
        hints = typing.get_type_hints(section_type)
        for item in dataclasses.fields(section_type):
            name = f"{ENV_PREFIX}{section}_{item.name}".upper()
            targets[name] = (section, item.name, hints[item.name])
    for name, raw in os.environ.items():
        if not name.upper().startswith(ENV_PREFIX):
            continue
        # A typo such as AUDITOR_AGENT_MAX_TURN would otherwise be ignored silently.
        target = targets.get(name.upper())
        if target is None:
            raise ConfigError(f"Unknown setting in environment: {name}.")
        section, key, hint = target
        merged[section][key] = _coerce_env(name, raw, hint)


def _coerce_env(name: str, raw: str, hint: Any) -> Any:
    text = raw.strip()
    if hint is bool:
        lowered = text.lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        raise ConfigError(f"{name} must be one of true/false/1/0/yes/no.")
    if hint is int:
        try:
            return int(text)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer.") from exc
    if hint is float:
        try:
            return float(text)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number.") from exc
    if hint is str:
        return text
    return [item.strip() for item in text.split(",")]


def _build(merged: dict[str, Any]) -> Settings:
    sections: dict[str, Any] = {}
    for section, section_type in _section_types().items():
        hints = typing.get_type_hints(section_type)
        values = {
            item.name: _validated(
                f"{section}.{item.name}",
                hints[item.name],
                item.metadata,
                merged[section][item.name],
            )
            for item in dataclasses.fields(section_type)
        }
        sections[section] = section_type(**values)
    settings = Settings(**sections)
    if settings.repository.max_range_file_bytes < settings.repository.max_file_bytes:
        raise ConfigError(
            "repository.max_range_file_bytes must be at least repository.max_file_bytes."
        )
    if settings.llm.backoff_max_seconds < settings.llm.backoff_base_seconds:
        raise ConfigError("llm.backoff_max_seconds must be at least llm.backoff_base_seconds.")
    if settings.git_log.default_commits > settings.git_log.max_commits:
        raise ConfigError("git_log.default_commits must not exceed git_log.max_commits.")
    return settings


def _validated(name: str, hint: Any, metadata: Any, value: Any) -> Any:
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{name} must be a boolean.")
        return value
    if hint is int or hint is float:
        # bool is an int subclass; `max_turns = true` is a mistake, not 1.
        allowed = (int,) if hint is int else (int, float)
        if isinstance(value, bool) or not isinstance(value, allowed):
            raise ConfigError(f"{name} must be {'an integer' if hint is int else 'a number'}.")
        return hint(_in_range(name, value, metadata))
    if hint is str:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{name} must be a non-empty string.")
        if metadata.get("https_url"):
            _check_https_url(name, value)
        choices = metadata.get("choices")
        if choices and value not in choices:
            raise ConfigError(f"{name} must be one of: {', '.join(choices)}.")
        if metadata.get("regex"):
            _check_regex(name, value)
        if metadata.get("name_part"):
            _check_name_part(name, value)
        return value
    return _string_tuple(name, value)


def _in_range(name: str, value: float, metadata: Any) -> float:
    minimum = metadata["min"]
    maximum = metadata["max"]
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be at most {maximum}.")
    return value


def _check_https_url(name: str, value: str) -> None:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigError(f"{name} must be an HTTPS URL without credentials, query or fragment.")


def check_https_url(name: str, value: str) -> None:
    """Raise ConfigError unless ``value`` is an HTTPS URL without credentials, query or fragment."""
    _check_https_url(name, value)


def _check_regex(name: str, value: str) -> None:
    try:
        re.compile(value)
    except re.error as exc:
        raise ConfigError(f"{name} is not a valid regular expression: {exc}.") from exc


def _check_name_part(name: str, value: str) -> None:
    # Used as a mkdtemp prefix: anything path-like could place the clone elsewhere.
    if any(mark in value for mark in ("/", "\\", ":", "\x00")) or ".." in value:
        raise ConfigError(f"{name} must not contain path separators, drive letters or '..'.")


def _string_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list of strings.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigError(f"{name} must contain only non-empty strings.")
    if len(set(value)) != len(value):
        raise ConfigError(f"{name} contains duplicate entries.")
    return tuple(value)
