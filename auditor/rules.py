"""The rule catalog: ids, default severities, evidence and recommendations.

The catalog lives in the packaged ``config/rules.toml`` next to the secret rules and,
like them, is not overridable by ``--config`` or the environment: the ids are part of
the report contract, so a user file must not be able to change them.
"""

from __future__ import annotations

import functools
import re
import tomllib
from dataclasses import dataclass
from typing import Any

from auditor.redaction import RULES_PATH
from auditor.schema import CATEGORIES, SEVERITIES
from auditor.settings import ConfigError


_ID_FORMAT = re.compile(r"^[A-Z]{2,3}-\d{3}$")
_RULE_KEYS = {
    "id",
    "category",
    "default_severity",
    "title",
    "evidence",
    "recommendation",
    "acceptable_alternatives",
    "detector",
}


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    default_severity: str
    title: str
    evidence: str
    recommendation: str
    acceptable_alternatives: tuple[str, ...]
    detector: str


@dataclass(frozen=True)
class Catalog:
    rules: tuple[Rule, ...]
    detectors: dict[str, dict[str, Any]]
    audit: dict[str, Any]

    def by_id(self) -> dict[str, Rule]:
        return {rule.id: rule for rule in self.rules}


def parse_catalog(document: dict[str, Any], source: str = "rules.toml") -> Catalog:
    """Validate a parsed rules document; problems raise ``ConfigError``."""
    raw_rules = document.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ConfigError(f"{source}: missing [[rules]] catalog.")
    detectors = document.get("detectors")
    if not isinstance(detectors, dict):
        raise ConfigError(f"{source}: missing [detectors] tables.")
    audit = document.get("audit")
    if not isinstance(audit, dict):
        raise ConfigError(f"{source}: missing section [audit].")
    shown = audit.get("prompt_alternatives_per_rule")
    if isinstance(shown, bool) or not isinstance(shown, int) or shown < 0:
        raise ConfigError(f"{source}: audit.prompt_alternatives_per_rule must be an integer of at least 0.")
    rules: list[Rule] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules):
        rule = _parse_rule(raw, index, source)
        if rule.id in seen:
            raise ConfigError(f"{source}: duplicate rule id {rule.id}.")
        seen.add(rule.id)
        if rule.detector and rule.detector not in detectors:
            raise ConfigError(
                f"{source}: {rule.id} names unknown detector '{rule.detector}'."
            )
        rules.append(rule)
    for name in audit.get("strict_rule_ids", ()):
        if name not in seen:
            raise ConfigError(f"{source}: audit.strict_rule_ids names unknown rule {name}.")
    return Catalog(tuple(rules), detectors, audit)


def _parse_rule(raw: Any, index: int, source: str) -> Rule:
    where = f"{source}: rules[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a table.")
    unknown = raw.keys() - _RULE_KEYS
    missing = _RULE_KEYS - raw.keys()
    if unknown:
        raise ConfigError(f"{where} has unknown fields: {', '.join(sorted(unknown))}.")
    if missing:
        raise ConfigError(f"{where} is missing fields: {', '.join(sorted(missing))}.")
    rule_id = raw["id"]
    if not isinstance(rule_id, str) or not _ID_FORMAT.match(rule_id):
        raise ConfigError(f"{where}: id {rule_id!r} must look like SEC-001.")
    where = f"{source}: {rule_id}"
    if raw["category"] not in CATEGORIES:
        raise ConfigError(f"{where}: unknown category {raw['category']!r}.")
    if raw["default_severity"] not in SEVERITIES:
        raise ConfigError(f"{where}: unknown severity {raw['default_severity']!r}.")
    for name in ("title", "evidence", "recommendation"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ConfigError(f"{where}: {name} must be a non-empty string.")
    alternatives = raw["acceptable_alternatives"]
    if not isinstance(alternatives, list) or any(not isinstance(a, str) for a in alternatives):
        raise ConfigError(f"{where}: acceptable_alternatives must be a list of strings.")
    if not isinstance(raw["detector"], str):
        raise ConfigError(f"{where}: detector must be a string.")
    return Rule(
        id=rule_id,
        category=raw["category"],
        default_severity=raw["default_severity"],
        title=raw["title"].strip(),
        evidence=raw["evidence"].strip(),
        recommendation=raw["recommendation"].strip(),
        acceptable_alternatives=tuple(alternatives),
        detector=raw["detector"],
    )


@functools.cache
def load_catalog() -> Catalog:
    try:
        with RULES_PATH.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"Cannot read rules file {RULES_PATH}: {exc.strerror}.") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {RULES_PATH}: {exc}") from exc
    return parse_catalog(document, str(RULES_PATH))


def known_rule(rule_id: str) -> bool:
    return rule_id in load_catalog().by_id()


def get_rule(rule_id: str) -> Rule:
    return load_catalog().by_id()[rule_id]


def render_catalog_for_prompt() -> str:
    """A compact table for the system prompt, grouped by category to save space."""
    lines = [
        "Use only these rule ids. Severity is the default; change it only when the "
        "evidence justifies it. Never flag the alternatives listed after 'ok:'. "
        "Use the category's -099 rule only for a verified problem no other rule covers."
    ]
    catalog = load_catalog()
    shown = catalog.audit["prompt_alternatives_per_rule"]
    for category in CATEGORIES:
        members = [rule for rule in catalog.rules if rule.category == category]
        if not members:
            continue
        lines.append(f"{category}:")
        for rule in members:
            alternatives = "; ".join(rule.acceptable_alternatives[:shown])
            suffix = f" | ok: {alternatives}" if alternatives else ""
            lines.append(f"{rule.id} {rule.default_severity} {rule.title}{suffix}")
    return "\n".join(lines)
