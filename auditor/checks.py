"""Conservative, read-only DevOps detectors for files the agent already inspected.

Every pattern, file list and keyword comes from the ``[detectors.*]`` tables of
``config/rules.toml``; severity, description and recommendation come from the rule
catalog. A detector reports only what is visible in the text it is given.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from auditor.redaction import (
    SecretRules,
    find_multiline_secrets,
    find_secret_assignments,
    packaged_secret_rules,
    redact_secrets,
    redacted_lines,
)
from auditor.rules import get_rule, load_catalog
from auditor.settings import ConfigError


# A hit is (1-based line or None, evidence). Content detectors get one file,
# path detectors get the tracked path list and whatever file contents are known.
_Hit = tuple["int | None", str]


def _audit(key: str) -> Any:
    return _setting(load_catalog().audit, "audit", key)


def _detector(name: str, key: str) -> Any:
    return _setting(load_catalog().detectors[name], f"detectors.{name}", key)


def _setting(table: Mapping[str, Any], section: str, key: str) -> Any:
    try:
        return table[key]
    except KeyError:
        raise ConfigError(f"rules.toml: missing setting {section}.{key}.") from None


def _rx(name: str, key: str) -> re.Pattern[str]:
    # Patterns are matched against one line (or one command) at a time.
    return re.compile(_detector(name, key))


def _matches_any(name: str, globs: Iterable[str]) -> bool:
    folded = name.casefold()
    return any(fnmatch.fnmatchcase(folded, glob) for glob in globs)


def _base_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_fixture(path: str) -> bool:
    folded = path.casefold()
    return any(fragment in folded for fragment in _audit("fixture_fragments"))


def _is_workflow(path: str) -> bool:
    return path.casefold().startswith(_audit("workflow_directory"))


def _is_comment(line: str) -> bool:
    return re.match(_audit("comment_pattern"), line) is not None


def _run_lines(lines: Sequence[str]) -> list[int]:
    """Indexes of lines that are shell commands: an inline run: or the body of a run: | block."""
    key_pattern = re.compile(_audit("run_key_pattern"))
    block_start = re.compile(_audit("block_scalar_start_pattern"))
    indexes: list[int] = []
    position = 0
    while position < len(lines):
        match = key_pattern.match(lines[position])
        position += 1
        if match is None or _is_comment(lines[position - 1]):
            continue
        if block_start.match(match["rest"]) is None:
            indexes.append(position - 1)
            continue
        key_indent = len(match["lead"].replace("-", " "))
        while position < len(lines):
            text = lines[position]
            if text.strip() and len(text) - len(text.lstrip()) <= key_indent:
                break
            if text.strip():
                indexes.append(position)
            position += 1
    return indexes


@dataclass(frozen=True)
class _CredentialPolicy:
    code: bool
    prose_format: bool
    quoted_min_length: int
    skips: tuple[re.Pattern[str], ...]
    rules: SecretRules


def _credential_hits(path: str, content: str) -> list[_Hit]:
    name = _base_name(path)
    code = _matches_any(name, _detector("hardcoded_credential", "code_globs"))
    if not code and not _matches_any(name, _detector("hardcoded_credential", "unquoted_globs")):
        return []
    rules = packaged_secret_rules()
    policy = _CredentialPolicy(
        code=code,
        prose_format=_matches_any(name, _detector("hardcoded_credential", "prose_globs")),
        quoted_min_length=_detector("hardcoded_credential", "quoted_literal_min_length"),
        skips=tuple(re.compile(p) for p in _detector("hardcoded_credential", "literal_skip_patterns")),
        rules=rules,
    )
    lines = content.splitlines()
    shown = redacted_lines(content)
    hits: list[_Hit] = []
    for index, line in enumerate(lines):
        # Minified or generated lines are not parsed by redaction either.
        if len(line) > rules.max_line_chars:
            continue
        if any(_is_credential(a.key, a.value, policy) for a in find_secret_assignments(line, rules)):
            hits.append((index + 1, shown[index]))
    if not code:
        hits.extend(_multiline_hits(lines, shown, policy))
    return sorted(hits, key=lambda hit: hit[0] or 0)


def _is_credential(key: str, value: str, policy: _CredentialPolicy) -> bool:
    quoted = value[:1] in "\"'"
    # In source files an unquoted value is an expression (settings.KEY, os.environ[...]).
    if policy.code and not quoted:
        return False
    inner = value.strip("\"'").strip()
    if any(skip.search(inner) for skip in policy.skips):
        return False
    normalized = inner.casefold()
    if policy.prose_format and (" " in normalized or normalized == key.casefold()):
        return False
    minimum = policy.quoted_min_length if policy.code else policy.rules.min_secret_length
    return not (
        len(normalized) < minimum
        or normalized in policy.rules.placeholder_values
        or any(marker in normalized for marker in policy.rules.placeholder_markers)
    )


def _multiline_hits(
    lines: Sequence[str],
    shown: Sequence[str],
    policy: _CredentialPolicy,
) -> list[_Hit]:
    hits: list[_Hit] = []
    seen: set[int] = set()
    for secret in find_multiline_secrets(lines, policy.rules):
        if secret.anchor in seen:
            continue
        value = lines[secret.line][secret.start : secret.end]
        if not _is_credential("", value, policy):
            continue
        seen.add(secret.anchor)
        evidence = "\n".join(shown[secret.anchor : secret.line + 1])
        hits.append((secret.line + 1, evidence))
    return hits


def _token_hits(path: str, content: str) -> list[_Hit]:
    rules = packaged_secret_rules()
    markers = _detector("secret_token", "fake_markers")
    shown = redacted_lines(content)
    hits: list[_Hit] = []
    for index, line in enumerate(content.splitlines()):
        if any(marker in line.casefold() for marker in markers):
            continue
        if rules.private_key_begin.search(line) or any(p.search(line) for p in rules.token_patterns):
            hits.append((index + 1, shown[index]))
    return hits


def _secret_file_hits(
    paths: Sequence[str], files: Mapping[str, str]
) -> list[tuple[str, _Hit]]:
    rules = packaged_secret_rules()
    content_patterns = {
        name.casefold(): re.compile(pattern)
        for name, pattern in _detector("secret_file", "content_patterns").items()
    }
    hits = []
    for path in paths:
        name = _base_name(path).casefold()
        if _is_fixture(path) or name.endswith(tuple(_detector("secret_file", "safe_suffixes"))):
            continue
        content = files.get(path)
        if name in content_patterns:
            if content is not None and content_patterns[name].search(content):
                hits.append((path, (None, path)))
        elif _matches_any(name, _detector("secret_file", "private_key_globs")):
            # An unread key file, or a public certificate, proves nothing.
            if content is not None and rules.private_key_begin.search(content) is not None:
                hits.append((path, (None, path)))
        elif _matches_any(name, _detector("secret_file", "path_globs")) and not _is_env_template(name):
            # A read env file with nothing redactable holds only public settings.
            if content is None or not _is_env_file(name) or redact_secrets(content) != content:
                hits.append((path, (None, path)))
    return hits


def _is_env_file(name: str) -> bool:
    return _matches_any(name, _detector("secret_file", "env_globs"))


def _is_env_template(name: str) -> bool:
    if not _is_env_file(name):
        return False
    return not set(name.split(".")).isdisjoint(_detector("secret_file", "safe_env_parts"))


def _write_all_hits(path: str, content: str) -> list[_Hit]:
    pattern = _rx("workflow_write_all", "pattern")
    return [(i + 1, line) for i, line in enumerate(content.splitlines()) if pattern.match(line)]


def _pr_target_hits(path: str, content: str) -> list[_Hit]:
    if _rx("pr_target_checkout", "trigger_pattern").search(content) is None:
        return []
    lines = content.splitlines()
    checkout = _rx("pr_target_checkout", "checkout_pattern")
    ref = _rx("pr_target_checkout", "untrusted_ref_pattern")
    checkouts = [i for i, line in enumerate(lines) if checkout.match(line)]
    runs = _run_lines(lines)
    for index, line in enumerate(lines):
        if ref.match(line) and any(i < index for i in checkouts) and any(i > index for i in runs):
            note = _detector("pr_target_checkout", "related_evidence")
            return [(index + 1, f"{line}\n{note}")]
    return []


def _command_text(line: str) -> str:
    """The shell text of a run line: what follows an inline ``run:`` key, else the line itself."""
    key = re.match(_audit("run_key_pattern"), line)
    return key["rest"] if key is not None else line.strip()


def _ungated_hits(path: str, content: str) -> list[_Hit]:
    lines = content.splitlines()
    command = _rx("ungated_check", "command_pattern")
    override = _rx("ungated_check", "override_pattern")
    continue_pattern = _rx("ungated_check", "continue_on_error_pattern")
    step_start = _rx("ungated_check", "step_start_pattern")
    run_indexes = _run_lines(lines)
    hits: list[_Hit] = []
    for index in run_indexes:
        text = _command_text(lines[index])
        found = command.search(text)
        if found is not None and override.search(text, found.end()) is not None:
            hits.append((index + 1, lines[index]))
    for index, line in enumerate(lines):
        if not continue_pattern.match(line):
            continue
        start, end = _block_bounds(lines, index, step_start)
        if not step_start.match(lines[start]) and _is_experimental_job(lines, start, end, index):
            continue
        if any(start <= run < end and command.search(_command_text(lines[run])) for run in run_indexes):
            hits.append((index + 1, line))
    return sorted(hits, key=lambda hit: hit[0] or 0)


def _is_experimental_job(lines: Sequence[str], start: int, end: int, key_index: int) -> bool:
    """A job that may fail on purpose: named so, or driven by a matrix.experimental flag."""
    marker = _rx("ungated_check", "experimental_job_pattern")
    name = _rx("ungated_check", "name_pattern")
    depth = len(lines[key_index]) - len(lines[key_index].lstrip())
    own_names = [
        lines[i]
        for i in range(start, end)
        if name.match(lines[i]) and len(lines[i]) - len(lines[i].lstrip()) == depth
    ]
    return any(marker.search(text) for text in [lines[start], *own_names]) or any(
        "matrix.experimental" in lines[i] for i in range(start, end)
    )


def _block_bounds(lines: Sequence[str], index: int, step_start: re.Pattern[str]) -> tuple[int, int]:
    """The step or job that owns a key: from its first line to the next line at its level."""
    def indent(text: str) -> int:
        return len(text) - len(text.lstrip())

    if step_start.match(lines[index]):
        start = index
    else:
        start = index
        while start > 0:
            start -= 1
            if lines[start].strip() and indent(lines[start]) < indent(lines[index]):
                break
    level = indent(lines[start])
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or indent(lines[end]) > level):
        end += 1
    return start, end


def _injection_hits(path: str, content: str) -> list[_Hit]:
    lines = content.splitlines()
    expression = _rx("script_injection", "expression_pattern")
    return [(i + 1, lines[i]) for i in _run_lines(lines) if expression.search(lines[i])]


def _unpinned_hits(path: str, content: str) -> list[_Hit]:
    uses = _rx("unpinned_action", "uses_pattern")
    local = tuple(_detector("unpinned_action", "local_prefixes"))
    branches = _detector("unpinned_action", "branch_refs")
    hits: list[_Hit] = []
    for index, line in enumerate(content.splitlines()):
        match = uses.match(line)
        if match is None or match["ref"].startswith(local):
            continue
        _, separator, version = match["ref"].partition("@")
        if not separator or version.casefold() in branches:
            hits.append((index + 1, line))
    return hits


def _generated_prefix(path: str) -> str | None:
    """The generated directory a tracked path sits in, or None."""
    parts = path.split("/")
    top_level = _detector("generated_files", "top_level_directories")
    artefacts = tuple(_detector("generated_files", "artefact_suffixes"))
    if len(parts) > 1 and _matches_any(parts[0], top_level) and path.casefold().endswith(artefacts):
        return parts[0] + "/"
    for depth, part in enumerate(parts[:-1], start=1):
        if _matches_any(part, _detector("generated_files", "directory_globs")):
            return "/".join(parts[:depth]) + "/"
    return None


def _generated_hits(paths: Sequence[str], files: Mapping[str, str]) -> list[tuple[str, _Hit]]:
    file_globs = _detector("generated_files", "file_globs")
    hits: list[tuple[str, _Hit]] = []
    reported: set[str] = set()
    for path in paths:
        if _is_fixture(path):
            continue
        prefix = _generated_prefix(path)
        if prefix is not None:
            if prefix not in reported:
                reported.add(prefix)
                hits.append((path, (None, prefix)))
        elif _matches_any(_base_name(path), file_globs):
            hits.append((path, (None, path)))
    return hits


def _absolute_path_hits(path: str, content: str) -> list[_Hit]:
    if (
        _is_fixture(path)
        or _matches_any(_base_name(path), _detector("absolute_path", "skip_globs"))
        or path.casefold().startswith(tuple(_detector("absolute_path", "skip_prefixes")))
        or not set(path.casefold().split("/")[:-1]).isdisjoint(_detector("absolute_path", "skip_directories"))
    ):
        return []
    patterns = [re.compile(p) for p in _detector("absolute_path", "patterns")]
    ignored = _detector("absolute_path", "ignored_users")
    hits: list[_Hit] = []
    for index, line in enumerate(content.splitlines()):
        if _is_comment(line):
            continue
        if any(
            match["user"].casefold() not in ignored
            for pattern in patterns
            for match in pattern.finditer(line)
        ):
            hits.append((index + 1, line))
    return hits


def _dockerfile_hits(path: str, content: str) -> list[_Hit]:
    if not _matches_any(_base_name(path), _detector("dockerfile_risk", "file_globs")):
        return []
    from_line = _rx("dockerfile_risk", "from_pattern")
    pipe = _rx("dockerfile_risk", "pipe_pattern")
    allowed = _detector("dockerfile_risk", "allowed_images")
    markers = _detector("dockerfile_risk", "reference_markers")
    mutable = _detector("dockerfile_risk", "mutable_tag")
    stages: set[str] = set()
    hits: list[_Hit] = []
    for index, line in enumerate(content.splitlines()):
        if _is_comment(line):
            continue
        match = from_line.match(line)
        if match is not None:
            image = match["image"]
            known = image.casefold() in allowed or image.casefold() in stages
            if match["alias"]:
                stages.add(match["alias"].casefold())
            if known or any(marker in image for marker in markers):
                continue
            tag = image.rsplit("/", 1)[-1].partition(":")[2]
            if not tag or tag.casefold() == mutable:
                hits.append((index + 1, line))
        elif pipe.search(line):
            hits.append((index + 1, line))
    return hits


def _workflow_only(detector: Callable[[str, str], list[_Hit]]) -> Callable[[str, str], list[_Hit]]:
    def run(path: str, content: str) -> list[_Hit]:
        return detector(path, content) if _is_workflow(path) else []

    return run


_CONTENT_DETECTORS: dict[str, Callable[[str, str], list[_Hit]]] = {
    "hardcoded_credential": _credential_hits,
    "secret_token": _token_hits,
    "workflow_write_all": _workflow_only(_write_all_hits),
    "pr_target_checkout": _workflow_only(_pr_target_hits),
    "ungated_check": _workflow_only(_ungated_hits),
    "script_injection": _workflow_only(_injection_hits),
    "unpinned_action": _workflow_only(_unpinned_hits),
    "absolute_path": _absolute_path_hits,
    "dockerfile_risk": _dockerfile_hits,
}
_PATH_DETECTORS: dict[str, Callable[[Sequence[str], Mapping[str, str]], list[tuple[str, _Hit]]]] = {
    "secret_file": _secret_file_hits,
    "generated_files": _generated_hits,
}


def _check_registry() -> None:
    wired = _CONTENT_DETECTORS.keys() | _PATH_DETECTORS.keys()
    named = {rule.detector for rule in load_catalog().rules if rule.detector}
    if named - wired:
        raise ConfigError(f"rules.toml: no detector implements {', '.join(sorted(named - wired))}.")


_check_registry()
# Rules that must be reported exactly as a detector produced them.
AUDIT_RULE_IDS = frozenset(_audit("strict_rule_ids"))


def _finding(rule_id: str, path: str, line: int | None, evidence: str) -> dict[str, object]:
    rule = get_rule(rule_id)
    return {
        "rule_id": rule.id,
        "category": rule.category,
        "severity": rule.default_severity,
        "file": path,
        "line": line,
        "description": rule.title,
        "evidence": redact_secrets(evidence),
        "recommendation": rule.recommendation,
    }


def audit_inspected_files(
    files: Mapping[str, str],
    paths: Sequence[str] | None = None,
    rule_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run the detectors on files the model already read, and on listed paths if given."""
    wanted = None if rule_ids is None else set(rule_ids)
    by_detector = {
        rule.detector: rule.id
        for rule in load_catalog().rules
        if rule.detector and (wanted is None or rule.id in wanted)
    }
    checked_paths = sorted(files)
    tracked = sorted(set(paths or ()) | set(files))
    limit = _audit("max_findings_per_rule")
    findings: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    capped: set[str] = set()

    def add(detector: str, path: str, line: int | None, evidence: str) -> None:
        rule_id = by_detector[detector]
        counts[rule_id] = counts.get(rule_id, 0) + 1
        if counts[rule_id] > limit:
            capped.add(rule_id)
            return
        findings.append(_finding(rule_id, path, line, evidence))

    for path in checked_paths:
        for detector, run in _CONTENT_DETECTORS.items():
            if detector in by_detector:
                for line, evidence in run(path, files[path]):
                    add(detector, path, line, evidence)
    for detector, run in _PATH_DETECTORS.items():
        if detector in by_detector:
            for path, (line, evidence) in run(tracked, files):
                add(detector, path, line, evidence)
    result: dict[str, object] = {
        "checked_files": len(checked_paths),
        "checked_paths": checked_paths,
        "findings": findings,
        "scope_note": (
            "These checks are partial evidence: they cover only files already read "
            "by the agent, the tracked path list when it was supplied, and only the "
            "rules listed in these results. An empty result does not mean the "
            "repository has no other issues."
        ),
        "rules_run": sorted(by_detector.values()),
    }
    if capped:
        result["capped_rules"] = sorted(capped)
    return result


def supports_finding(
    rule_id: str,
    path: str,
    line: int,
    evidence: str,
    inspected_files: Mapping[str, str],
) -> bool:
    """Require known deterministic findings to match the checked source evidence."""
    if rule_id not in AUDIT_RULE_IDS or path not in inspected_files:
        return False
    findings = audit_inspected_files({path: inspected_files[path]}, rule_ids=[rule_id])["findings"]
    wanted = _normalize_evidence(evidence)
    return any(
        finding["rule_id"] == rule_id
        and finding["file"] == path
        and finding["line"] == line
        and _normalize_evidence(finding["evidence"]) == wanted
        for finding in findings
    )


def _normalize_evidence(text: str) -> str:
    # Models re-indent or re-wrap evidence they copy; only content must match.
    return "\n".join(" ".join(line.split()) for line in text.strip().splitlines())
