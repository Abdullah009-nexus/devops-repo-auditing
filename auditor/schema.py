"""Constants and the finding record fixed by the assignment's report contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SEVERITIES = ("critical", "high", "medium", "low")
REQUIRED_FIELDS = (
    "rule_id",
    "category",
    "severity",
    "file",
    "line",
    "description",
    "evidence",
    "recommendation",
)
CATEGORIES = (
    "repository_hygiene",
    "dependency_management",
    "code_quality",
    "testing",
    "ci_cd",
    "security",
    "configuration",
    "collaboration",
    "build_release",
)

_CATEGORY_SEPARATORS = re.compile(r"[\s\-_/]+")


def normalize_category(value: str) -> str | None:
    """The canonical category for a spelling such as "CI-CD" or "Build Release", else None."""
    candidate = _CATEGORY_SEPARATORS.sub("_", value.strip().lower()).strip("_")
    return candidate if candidate in CATEGORIES else None


@dataclass(frozen=True)
class Finding:
    rule_id: str
    category: str
    severity: str
    file: str | None
    line: int | None
    description: str
    evidence: str
    recommendation: str

    def to_report_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in REQUIRED_FIELDS}
