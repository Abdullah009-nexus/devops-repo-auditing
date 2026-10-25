"""Secret redaction applied to every piece of repository content that leaves a tool.

Redaction and the hard-coded credential check share one matcher, loaded from the
packaged ``config/rules.toml``, so a value the check reports is always a value
redaction hides. Line structure is never changed, so redacted line numbers match
the source.
"""

from __future__ import annotations

import functools
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from auditor.settings import ConfigError


RULES_PATH = Path(__file__).with_name("config") / "rules.toml"
# Digits are dropped so API_KEY2 and PASSWORD_1 read like their plain names.
_NAME_COMPONENT = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")
_SECTION_KEYS = {
    "secrets": {
        "key_suffixes",
        "exact_keys",
        "key_patterns",
        "element_pattern",
        "reference_pattern",
        "value_pattern",
        "placeholder_values",
        "placeholder_markers",
        "min_secret_length",
        "non_secret_values",
        "name_line_pattern",
        "value_line_pattern",
        "name_value_window",
        "block_scalar_pattern",
        "block_scalar_max_lines",
    },
    "redaction": {
        "max_line_chars",
        "token_patterns",
        "prefixed_patterns",
        "private_key_begin_pattern",
        "private_key_end_pattern",
        "max_private_key_chars",
        "value_marker",
        "token_marker",
        "private_key_marker",
    },
}


@dataclass(frozen=True)
class SecretAssignment:
    """A sensitive key with its value; ``start``/``end`` delimit the value, quotes included."""

    key: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class MultilineSecret:
    """A secret value on ``line`` (0-based) whose key is named on line ``anchor``."""

    key: str
    anchor: int
    line: int
    start: int
    end: int


@dataclass(frozen=True)
class SecretRules:
    key_suffixes: tuple[tuple[str, ...], ...]
    exact_keys: frozenset[tuple[str, ...]]
    # Any sensitive key contains one of these words; checking that first keeps a
    # line of thousands of harmless keys cheap.
    key_hint: re.Pattern[str]
    key_patterns: tuple[re.Pattern[str], ...]
    element: re.Pattern[str]
    reference: re.Pattern[str]
    value: re.Pattern[str]
    placeholder_values: frozenset[str]
    placeholder_markers: tuple[str, ...]
    min_secret_length: int
    non_secret_values: frozenset[str]
    name_line: re.Pattern[str]
    value_line: re.Pattern[str]
    name_value_window: int
    block_scalar: re.Pattern[str]
    block_scalar_max_lines: int
    max_line_chars: int
    token_patterns: tuple[re.Pattern[str], ...]
    prefixed_patterns: tuple[re.Pattern[str], ...]
    private_key_begin: re.Pattern[str]
    private_key_end: re.Pattern[str]
    max_private_key_chars: int
    value_marker: str
    token_marker: str
    private_key_marker: str

    def is_sensitive_key(self, key: str) -> bool:
        if self.key_hint.search(key) is None:
            return False
        components = name_components(key)
        if components in self.exact_keys:
            return True
        return any(
            len(components) >= len(suffix) and components[-len(suffix):] == suffix
            for suffix in self.key_suffixes
        )


def name_components(name: str) -> tuple[str, ...]:
    """Split ``dbPassword`` or ``OPENAI_API_KEY`` into lowercase words."""
    return tuple(part.lower() for part in _NAME_COMPONENT.findall(name))


def load_secret_rules(path: Path | None = None) -> SecretRules:
    """Read and validate the secret rules (packaged file by default); problems raise ``ConfigError``."""
    path = RULES_PATH if path is None else path
    document = _read_toml(path)
    secrets = _section(document, "secrets", path)
    redaction = _section(document, "redaction", path)

    def strings(section: str, key: str) -> tuple[str, ...]:
        return _strings(path, f"{section}.{key}", document[section][key])

    def pattern(section: str, key: str, groups: frozenset[str] = frozenset()) -> re.Pattern[str]:
        return _pattern(path, f"{section}.{key}", document[section][key], groups)

    def patterns(section: str, key: str, **required: Any) -> tuple[re.Pattern[str], ...]:
        return tuple(
            _pattern(path, f"{section}.{key}[{index}]", item, **required)
            for index, item in enumerate(strings(section, key))
        )

    key_suffixes = _components(path, "secrets.key_suffixes", strings("secrets", "key_suffixes"))
    exact_keys = _components(path, "secrets.exact_keys", strings("secrets", "exact_keys"))
    hint_words = sorted({components[-1] for components in key_suffixes + exact_keys})
    return SecretRules(
        key_suffixes=key_suffixes,
        exact_keys=frozenset(exact_keys),
        key_hint=re.compile("|".join(map(re.escape, hint_words)), re.IGNORECASE),
        key_patterns=patterns("secrets", "key_patterns", groups=frozenset({"key"})),
        element=pattern("secrets", "element_pattern", frozenset({"key", "value"})),
        reference=pattern("secrets", "reference_pattern"),
        value=pattern("secrets", "value_pattern"),
        placeholder_values=frozenset(
            value.casefold() for value in strings("secrets", "placeholder_values")
        ),
        placeholder_markers=tuple(
            marker.casefold() for marker in strings("secrets", "placeholder_markers")
        ),
        min_secret_length=_positive_int(
            path, "secrets.min_secret_length", secrets["min_secret_length"]
        ),
        non_secret_values=frozenset(
            value.casefold() for value in strings("secrets", "non_secret_values")
        ),
        name_line=pattern("secrets", "name_line_pattern", frozenset({"key"})),
        value_line=pattern("secrets", "value_line_pattern"),
        name_value_window=_positive_int(
            path, "secrets.name_value_window", secrets["name_value_window"]
        ),
        block_scalar=pattern("secrets", "block_scalar_pattern", frozenset({"key"})),
        block_scalar_max_lines=_positive_int(
            path, "secrets.block_scalar_max_lines", secrets["block_scalar_max_lines"]
        ),
        max_line_chars=_positive_int(path, "redaction.max_line_chars", redaction["max_line_chars"]),
        token_patterns=patterns("redaction", "token_patterns"),
        prefixed_patterns=patterns("redaction", "prefixed_patterns", group_count=1),
        private_key_begin=pattern("redaction", "private_key_begin_pattern"),
        private_key_end=pattern("redaction", "private_key_end_pattern"),
        max_private_key_chars=_positive_int(
            path, "redaction.max_private_key_chars", redaction["max_private_key_chars"]
        ),
        value_marker=_text(path, "redaction.value_marker", redaction["value_marker"]),
        token_marker=_text(path, "redaction.token_marker", redaction["token_marker"]),
        private_key_marker=_text(
            path, "redaction.private_key_marker", redaction["private_key_marker"]
        ),
    )


@functools.cache
def packaged_secret_rules() -> SecretRules:
    """The rules shipped with the package.

    Deliberately not affected by ``--config``: a user file must not be able to
    weaken what is hidden from the model, traces and reports.
    """
    return load_secret_rules()


def find_secret_assignments(line: str, rules: SecretRules) -> list[SecretAssignment]:
    """Return the values of sensitive keys in one line, in order and without overlaps.

    Pure references such as ``${{ secrets.X }}`` are not secrets and are skipped.
    Past ``max_line_chars`` the values are not parsed; everything after the first
    sensitive key is returned instead, so an oversized line fails closed.
    """
    if rules.key_hint.search(line) is None:
        return []
    if len(line) > rules.max_line_chars:
        starts = [start for _, start in _sensitive_keys(line, rules)]
        if not starts:
            return []
        start = min(starts)
        return [SecretAssignment("", line[start:], start, len(line))]
    return [
        SecretAssignment(key, line[start:end], start, end)
        for key, start, end in _without_overlaps(_sensitive_values(line, rules))
    ]


def redact_secrets(text: str) -> str:
    """Remove likely credential values before content is returned to the agent."""
    rules = packaged_secret_rules()
    pieces = _mask_private_keys(text, rules).splitlines(keepends=True)
    bodies = [piece.splitlines()[0] for piece in pieces]
    for secret in sorted(
        find_multiline_secrets(bodies, rules), key=lambda found: found.start, reverse=True
    ):
        body = bodies[secret.line]
        bodies[secret.line] = body[: secret.start] + rules.value_marker + body[secret.end :]
    return "".join(
        _redact_line(body, rules) + piece[len(piece.splitlines()[0]) :]
        for body, piece in zip(bodies, pieces)
    )


def redacted_lines(content: str) -> list[str]:
    """``content.splitlines()`` with secrets removed; private keys spanning lines included."""
    return redact_secrets(content).splitlines()


def find_multiline_secrets(lines: Sequence[str], rules: SecretRules) -> list[MultilineSecret]:
    """Secrets whose value is not on the line that names the key.

    Covers ``- name: DB_PASSWORD`` followed by ``value: x`` and YAML block scalars
    (``password: |`` and the lines below). References such as ``${{ secrets.X }}``
    are skipped; placeholders are not, because redaction hides them harmlessly.
    """
    found: list[MultilineSecret] = []
    claimed: set[int] = set()
    for index, line in enumerate(lines):
        block = rules.block_scalar.match(line)
        if block is not None and rules.is_sensitive_key(block["key"]):
            found.extend(_block_scalar_values(lines, index, block["key"], rules, claimed))
            continue
        named = rules.name_line.match(line)
        if named is None or not rules.is_sensitive_key(named["key"]):
            continue
        for offset in range(index + 1, min(index + 1 + rules.name_value_window, len(lines))):
            if rules.name_line.match(lines[offset]) is not None:
                break
            value = rules.value_line.match(lines[offset])
            if value is None or offset in claimed:
                continue
            span = _value_span(lines[offset], value.end(), rules)
            if span is not None:
                claimed.add(offset)
                found.append(MultilineSecret(named["key"], index, offset, *span))
            break
    return found


def _value_span(line: str, start: int, rules: SecretRules) -> tuple[int, int] | None:
    if len(line) > rules.max_line_chars:
        return (start, len(line)) if start < len(line) else None
    if rules.reference.match(line, start) is not None:
        return None
    value = rules.value.match(line, start)
    if value is None or value.end() == value.start() or _is_setting(value.group(), rules):
        return None
    return value.start(), value.end()


def _block_scalar_values(
    lines: Sequence[str],
    index: int,
    key: str,
    rules: SecretRules,
    claimed: set[int],
) -> list[MultilineSecret]:
    indent = len(lines[index]) - len(lines[index].lstrip())
    values = []
    for offset in range(index + 1, min(index + 1 + rules.block_scalar_max_lines, len(lines))):
        text = lines[offset]
        if not text.strip():
            continue
        if len(text) - len(text.lstrip()) <= indent:
            break
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        # A scalar that is only a reference stays readable.
        if rules.reference.fullmatch(text.strip()) is None and offset not in claimed:
            claimed.add(offset)
            values.append(MultilineSecret(key, index, offset, start, end))
    return values


def _redact_line(line: str, rules: SecretRules) -> str:
    for assignment in reversed(find_secret_assignments(line, rules)):
        line = line[: assignment.start] + rules.value_marker + line[assignment.end :]
    for pattern in rules.token_patterns:
        line = pattern.sub(rules.token_marker, line)
    for pattern in rules.prefixed_patterns:
        line = pattern.sub(lambda match: match.group(1) + rules.value_marker, line)
    return line


def _sensitive_keys(line: str, rules: SecretRules) -> Iterator[tuple[str, int]]:
    """Yield ``(key, value_start)`` for each sensitive key; never reads the value itself."""
    for pattern in rules.key_patterns:
        for match in pattern.finditer(line):
            # finditer resumes at the operator, so a harmless key's value is still
            # scanned (url=https://h/?token=abc) without reparsing it per key.
            if rules.is_sensitive_key(match["key"]):
                yield match["key"], match.end()
    for match in rules.element.finditer(line):
        if rules.is_sensitive_key(match["key"]):
            yield match["key"], match.start("value")


def _sensitive_values(line: str, rules: SecretRules) -> Iterator[tuple[str, int, int]]:
    for pattern in rules.key_patterns:
        position = 0
        while (match := pattern.search(line, position)) is not None:
            position = match.end()
            if not rules.is_sensitive_key(match["key"]):
                continue
            reference = rules.reference.match(line, position)
            if reference is not None:
                position = reference.end()
                continue
            value = rules.value.match(line, position)
            if value is None or value.end() == value.start():
                continue
            position = value.end()
            if not _is_setting(value.group(), rules):
                yield match["key"], value.start(), value.end()
    for match in rules.element.finditer(line):
        if (
            rules.is_sensitive_key(match["key"])
            and not rules.reference.fullmatch(match["value"])
            and not _is_setting(match["value"], rules)
        ):
            yield match["key"], match.start("value"), match.end("value")


def _is_setting(value: str, rules: SecretRules) -> bool:
    return value.strip().strip("\"'").casefold() in rules.non_secret_values


def _without_overlaps(spans: Iterator[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    # Two key patterns can claim the same value; splicing both would corrupt the line.
    merged: list[tuple[str, int, int]] = []
    for key, start, end in sorted(spans, key=lambda span: (span[1], -span[2])):
        if merged and start < merged[-1][2]:
            last_key, last_start, last_end = merged[-1]
            merged[-1] = (last_key, last_start, max(last_end, end))
            continue
        merged.append((key, start, end))
    return merged


def _mask_private_keys(text: str, rules: SecretRules) -> str:
    """Replace each line of a private key block with the marker, keeping line breaks."""
    parts = []
    position = 0
    while (begin := rules.private_key_begin.search(text, position)) is not None:
        window_end = begin.end() + rules.max_private_key_chars
        end = rules.private_key_end.search(text, begin.end(), window_end)
        # No END marker in reach: whatever follows may be key material.
        block_end = end.end() if end is not None else len(text)
        parts.append(text[position : begin.start()])
        parts.append(_marker_per_line(text[begin.start() : block_end], rules.private_key_marker))
        position = block_end
    parts.append(text[position:])
    return "".join(parts)


def _marker_per_line(block: str, marker: str) -> str:
    masked = []
    for piece in block.splitlines(keepends=True):
        body = piece.splitlines()[0]
        masked.append((marker if body else "") + piece[len(body):])
    return "".join(masked)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"Cannot read rules file {path}: {exc.strerror}.") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def _section(document: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    # Other sections of rules.toml belong to the rule catalog and are read elsewhere.
    values = document.get(name)
    if not isinstance(values, dict):
        raise ConfigError(f"{path}: missing section [{name}].")
    expected = _SECTION_KEYS[name]
    for key in values:
        if key not in expected:
            raise ConfigError(f"{path}: unknown setting {name}.{key}.")
    for key in sorted(expected):
        if key not in values:
            raise ConfigError(f"{path}: missing setting {name}.{key}.")
    return values


def _strings(path: Path, name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}: {name} must be a non-empty list of strings.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigError(f"{path}: {name} must contain only non-empty strings.")
    if len(set(value)) != len(value):
        raise ConfigError(f"{path}: {name} contains duplicate entries.")
    return tuple(value)


def _components(path: Path, name: str, phrases: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    components = tuple(name_components(phrase) for phrase in phrases)
    if not all(components):
        raise ConfigError(f"{path}: {name} entries must contain letters.")
    return components


def _text(path: Path, name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: {name} must be a non-empty string.")
    return value


def _positive_int(path: Path, name: str, value: Any) -> int:
    # bool is an int subclass; `min_secret_length = true` is a mistake, not 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{path}: {name} must be a positive integer.")
    return value


def _pattern(
    path: Path,
    name: str,
    value: Any,
    groups: frozenset[str] = frozenset(),
    group_count: int | None = None,
) -> re.Pattern[str]:
    try:
        compiled = re.compile(_text(path, name, value))
    except re.error as exc:
        raise ConfigError(f"{path}: {name} is not a valid regular expression: {exc}") from exc
    missing = set(groups) - set(compiled.groupindex)
    if missing:
        raise ConfigError(f"{path}: {name} needs named groups {', '.join(sorted(missing))}.")
    if group_count is not None and compiled.groups != group_count:
        raise ConfigError(f"{path}: {name} needs exactly {group_count} group.")
    return compiled
