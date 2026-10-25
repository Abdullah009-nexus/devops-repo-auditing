"""Accepts or rejects candidate findings against what the agent actually read.

A rejection is never an exception: the reason is written for the model so it can
fix the finding or investigate further, which keeps one bad finding from costing
the whole report.
"""

from __future__ import annotations

import functools
import os
import re
import tomllib
from dataclasses import dataclass
from typing import Any, Mapping

from auditor.checks import AUDIT_RULE_IDS
from auditor.redaction import RULES_PATH, redact_secrets
from auditor.rules import get_rule, known_rule, load_catalog
from auditor.schema import CATEGORIES, REQUIRED_FIELDS, SEVERITIES, Finding, normalize_category
from auditor.settings import AgentSettings, ConfigError
from tools import GitHubRepository, RepositoryError


# Credentials the process holds; masked by exact value in case the pattern rules
# do not recognise their format.
_SECRET_ENV_NAMES = ("OPENAI_API_KEY", "GITHUB_TOKEN")
_MASK = "[REDACTED]"
_TEXT_FIELDS = ("rule_id", "category", "severity", "description", "evidence", "recommendation")

REPORT_FINDING = "report_finding"
FINISH_AUDIT = "finish_audit"


def build_finding_tools() -> list[dict[str, Any]]:
    """Schemas for the two tools that carry the model's conclusions."""
    text = {"type": "string", "minLength": 1}
    report_finding = {
        "type": "function",
        "function": {
            "name": REPORT_FINDING,
            "description": (
                "Record one violation you have verified in files you read. The call is "
                "checked against those files; if it is rejected, the reply says why, so "
                "fix it or read more before retrying."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": text,
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "file": {
                        "type": "string",
                        "description": "Repository path. Omit for an absence finding.",
                    },
                    "line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based line you read. Omit when file is omitted.",
                    },
                    "description": text,
                    "evidence": {
                        **text,
                        "description": "The exact line(s) as the tool showed them.",
                    },
                    "recommendation": text,
                    "quote": {
                        "type": "string",
                        "description": (
                            "Verbatim excerpt from within two lines of `line`. Required "
                            "when file and line are set (except SEC-001, CI-001, CI-002)."
                        ),
                    },
                },
                # file and line are optional: leaving them out means null in the report.
                "required": [name for name in REQUIRED_FIELDS if name not in ("file", "line")],
                "additionalProperties": False,
            },
        },
    }
    finish_audit = {
        "type": "function",
        "function": {
            "name": FINISH_AUDIT,
            "description": (
                "End the audit once every supported violation has been recorded. "
                "Call it with no findings pending."
            ),
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    }
    return [report_finding, finish_audit]


@dataclass(frozen=True)
class FindingDecision:
    accepted: bool
    reason: str | None = None
    finding: Finding | None = None
    # Kept off Finding so it can never reach the report; used later to check grounding.
    quote: str | None = None
    duplicate: bool = False


def _reject(reason: str, *, duplicate: bool = False) -> FindingDecision:
    return FindingDecision(False, reason, duplicate=duplicate)


@functools.cache
def _guards() -> dict[str, Any]:
    try:
        with RULES_PATH.open("rb") as handle:
            table = tomllib.load(handle).get("guards")
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read guards from {RULES_PATH}: {exc}") from exc
    if not isinstance(table, dict):
        raise ConfigError("rules.toml: missing section [guards].")
    return table


def _guard(key: str) -> Any:
    try:
        return _guards()[key]
    except KeyError:
        raise ConfigError(f"rules.toml: missing setting guards.{key}.") from None


@functools.cache
def _remote_setting_pattern() -> re.Pattern[str]:
    # Whole phrases only: a bare substring test also caught "required reviewers"
    # in CODEOWNERS findings, which are about files in the repository.
    phrases = (r"\s+".join(map(re.escape, k.split())) for k in _guard("remote_setting_keywords"))
    return re.compile(r"\b(?:" + "|".join(phrases) + r")(?:e?s)?\b")


def _squash(text: str) -> str:
    return " ".join(text.split())


class FindingStore:
    def __init__(
        self,
        repository: GitHubRepository,
        agent_settings: AgentSettings,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        self._repository = repository
        self._limit = agent_settings.max_report_findings
        self._max_field_chars = agent_settings.max_finding_field_chars
        self._secrets = [
            value for name in _SECRET_ENV_NAMES if (value := environ.get(name, "").strip())
        ]
        self._findings: list[Finding] = []
        # Root-cause key -> index in self._findings.
        self._keys: dict[tuple[Any, ...], int] = {}

    @property
    def findings(self) -> list[Finding]:
        return list(self._findings)

    def add(self, candidate: dict[str, Any]) -> FindingDecision:
        problem = self._field_problem(candidate)
        if problem:
            return _reject(problem)
        category = normalize_category(candidate["category"])
        if category is None:
            return _reject(
                f"category {candidate['category']!r} is not allowed; use one of: "
                + ", ".join(CATEGORIES)
                + "."
            )
        file_path = candidate["file"]
        line = candidate["line"]
        quote = candidate.get("quote")
        quote = quote if isinstance(quote, str) and quote.strip() else None
        problem = (
            self._catalog_problem(candidate, category)
            or self._location_problem(file_path, line)
            or self._guard_problem(candidate, file_path)
            or self._audit_rule_problem(candidate, file_path, line)
            or self._grounding_problem(candidate["rule_id"], file_path, line, quote)
        )
        if problem:
            return _reject(problem)
        finding = Finding(
            rule_id=self._scrub(candidate["rule_id"]),
            category=category,
            severity=candidate["severity"],
            file=self._scrub(file_path) if file_path is not None else None,
            line=line,
            description=self._scrub(candidate["description"]),
            evidence=self._scrub(candidate["evidence"]),
            recommendation=self._scrub(candidate["recommendation"]),
        )
        key = self._root_cause_key(finding)
        if key in self._keys:
            return self._merge_duplicate(key, finding)
        if len(self._findings) >= self._limit:
            return _reject(f"the report already holds the maximum of {self._limit} findings.")
        self._keys[key] = len(self._findings)
        self._findings.append(finding)
        return FindingDecision(True, finding=finding, quote=quote)

    @staticmethod
    def _root_cause_key(finding: Finding) -> tuple[Any, ...]:
        if finding.file is None:
            return ("absence", finding.rule_id, finding.category)
        return ("located", finding.category, finding.file, finding.line)

    def _merge_duplicate(self, key: tuple[Any, ...], finding: Finding) -> FindingDecision:
        index = self._keys[key]
        kept = self._findings[index]
        where = f"{kept.rule_id} at {kept.file or 'the repository'}" + (
            f":{kept.line}" if kept.line is not None else ""
        )
        # One root cause, one entry: a stronger reading of it replaces the weaker one.
        if SEVERITIES.index(finding.severity) < SEVERITIES.index(kept.severity):
            self._findings[index] = finding
            return _reject(
                f"duplicate of {where}; the entry was replaced because {finding.severity} "
                f"is higher than {kept.severity}.",
                duplicate=True,
            )
        return _reject(f"duplicate of {where}, which was already reported.", duplicate=True)

    def _catalog_problem(self, candidate: dict[str, Any], category: str) -> str | None:
        rule_id = candidate["rule_id"]
        if not known_rule(rule_id):
            return (
                f"rule_id {rule_id!r} is not in the rule catalog; use an id from the catalog, "
                "or the -099 rule of the category when nothing else fits."
            )
        rule = get_rule(rule_id)
        if category != rule.category:
            return f"{rule_id} belongs to category {rule.category}, not {category}."
        distance = abs(
            SEVERITIES.index(candidate["severity"]) - SEVERITIES.index(rule.default_severity)
        )
        if distance > _guard("severity_max_steps"):
            return f"default for {rule_id} is {rule.default_severity}; use it or a neighbouring level."
        return None

    def _guard_problem(self, candidate: dict[str, Any], file_path: str | None) -> str | None:
        rule_id = candidate["rule_id"]
        if file_path is not None:
            folded = file_path.casefold()
            if (
                rule_id.startswith(tuple(_guard("fixture_rule_prefixes")))
                and rule_id not in _guard("fixture_exempt_rules")
                and any(f in folded for f in load_catalog().audit["fixture_fragments"])
            ):
                return (
                    f"{file_path} is a test fixture or sample data; credentials there are "
                    "intentional. Report only a known token format (SEC-002) from such files."
                )
            if folded.endswith(tuple(_guard("template_suffixes"))):
                return f"{file_path} is a template file; its values are placeholders by design."
            if any(
                folded.startswith(f) or f"/{f}" in folded for f in _guard("vendored_fragments")
            ):
                return f"{file_path} is vendored third-party code; report it upstream instead."
            if rule_id.startswith(tuple(_guard("doc_rule_prefixes"))) and (
                folded.endswith(tuple(_guard("doc_suffixes")))
                or folded.startswith(tuple(_guard("doc_prefixes")))
            ):
                return f"{file_path} is documentation; snippets there are not live configuration."
        text = f"{candidate['description']} {candidate['evidence']}".casefold()
        if _remote_setting_pattern().search(text):
            return "remote settings can't be inferred from files; drop this finding."
        return None

    def _grounding_problem(
        self, rule_id: str, file_path: str | None, line: int | None, quote: str | None
    ) -> str | None:
        if file_path is None:
            if not self._repository.listing_complete():
                return (
                    "an absence finding needs a complete file listing: call list_files and "
                    "check that tree_truncated is false."
                )
            return None
        if line is None:
            if not self._repository.has_listed_path(file_path):
                return f"{file_path} does not appear in the file listing."
            return None
        # Strict rules are matched against the detector's own output instead.
        if rule_id in load_catalog().audit["strict_rule_ids"]:
            return None
        radius = _guard("grounding_radius")
        first, last = max(line - radius, 1), line + radius
        if quote is None:
            return (
                f"quote is required for a finding at {file_path}:{line}; copy the exact "
                f"text from lines {first}-{last}."
            )
        window = _squash(" ".join(self._repository.redacted_window(file_path, first, last)))
        if _squash(quote) not in window:
            return (
                f"quote was not found in lines {first}-{last} of {file_path}; copy it "
                "exactly as read_file showed it, or cite the right line."
            )
        return None

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, _MASK)
        return redact_secrets(text)

    def _field_problem(self, candidate: Any) -> str | None:
        if not isinstance(candidate, dict):
            return "a finding must be a JSON object."
        missing = [name for name in REQUIRED_FIELDS if name not in candidate]
        if missing:
            return "missing required fields: " + ", ".join(missing) + "."
        for name in _TEXT_FIELDS:
            value = candidate[name]
            if not isinstance(value, str) or not value.strip():
                return f"{name} must be a non-empty string."
            if len(value) > self._max_field_chars:
                return (
                    f"{name} is {len(value)} characters; shorten it to at most "
                    f"{self._max_field_chars}."
                )
        if candidate["severity"] not in SEVERITIES:
            return "severity must be one of: " + ", ".join(SEVERITIES) + "."
        file_path = candidate["file"]
        line = candidate["line"]
        if file_path is not None and not isinstance(file_path, str):
            return "file must be a string path or null."
        if line is not None and (not isinstance(line, int) or isinstance(line, bool) or line < 1):
            return "line must be a positive integer or null."
        if file_path is None and line is not None:
            return "line was given without a file; set file too, or set line to null."
        return None

    def _location_problem(self, file_path: str | None, line: int | None) -> str | None:
        if file_path is None:
            return None
        try:
            listed = line is None and self._repository.has_listed_path(file_path)
            if not listed and not self._repository.has_inspected_path(file_path):
                return f"{file_path} was not read; call read_file on {file_path} first."
            if line is not None and not self._repository.validate_finding_location(file_path, line):
                return (
                    f"line {line} is outside the part of {file_path} you read; "
                    f"call read_file on {file_path} and cite a line you saw."
                )
        except RepositoryError:
            return f"{file_path} is not a valid repository path."
        return None

    def _audit_rule_problem(
        self, candidate: dict[str, Any], file_path: str | None, line: int | None
    ) -> str | None:
        rule_id = candidate["rule_id"]
        if rule_id not in AUDIT_RULE_IDS:
            return None
        try:
            supported = (
                file_path is not None
                and line is not None
                and self._repository.supports_audit_finding(
                    rule_id, file_path, line, candidate["evidence"]
                )
            )
        except RepositoryError:
            supported = False
        if supported:
            return None
        return (
            f"{rule_id} is a deterministic audit rule and must match what audit_inspected_files "
            "returned: reuse its exact file, line and evidence, or use a different rule_id "
            "for your own finding."
        )
