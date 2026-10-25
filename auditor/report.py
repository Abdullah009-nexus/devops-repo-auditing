"""Builds, checks and writes the final JSON report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from auditor.schema import REQUIRED_FIELDS, SEVERITIES, Finding


class ReportError(ValueError):
    """Raised when a report does not match the assignment's contract."""


def build_report(repository_url: str, findings: Iterable[Finding]) -> dict[str, Any]:
    return {
        "repository": repository_url,
        "violations": [finding.to_report_dict() for finding in findings],
    }


def validate_report_shape(report: Any) -> None:
    """Check keys and value types only; evidence grounding is the FindingStore's job."""
    if not isinstance(report, dict) or set(report) != {"repository", "violations"}:
        raise ReportError("Report must have exactly the keys repository and violations.")
    if not isinstance(report["repository"], str) or not report["repository"]:
        raise ReportError("Report repository must be a non-empty string.")
    if not isinstance(report["violations"], list):
        raise ReportError("Report violations must be an array.")
    for index, item in enumerate(report["violations"], start=1):
        _validate_violation(index, item)


def _validate_violation(index: int, item: Any) -> None:
    if not isinstance(item, dict) or set(item) != set(REQUIRED_FIELDS):
        raise ReportError(f"Violation {index} must have exactly the contract fields.")
    for name in ("rule_id", "category", "description", "evidence", "recommendation"):
        if not isinstance(item[name], str) or not item[name].strip():
            raise ReportError(f"Violation {index} has an invalid {name}.")
    if item["severity"] not in SEVERITIES:
        raise ReportError(f"Violation {index} has an unsupported severity.")
    if item["file"] is not None and not isinstance(item["file"], str):
        raise ReportError(f"Violation {index} has an invalid file.")
    line = item["line"]
    if line is not None and (not isinstance(line, int) or isinstance(line, bool) or line < 1):
        raise ReportError(f"Violation {index} has an invalid line.")


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write the report atomically so a crash never leaves a half-written file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
