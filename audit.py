"""Conservative, read-only DevOps checks for files inspected by the agent."""

from __future__ import annotations

import re
from typing import Mapping

from tools import redact_secrets


SENSITIVE_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(?:api[_-]?(?:key|token)|client[_-]?secret|secret(?:[_-]?key)?|token|
        password|passwd|credential|private[_-]?key)\b
    \s*[:=]\s*
    (?P<value>"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;#]+)
    """
)
PLACEHOLDER_VALUES = {
    "changeme",
    "change-me",
    "example",
    "dummy",
    "fake",
    "not-a-secret",
    "placeholder",
    "redacted",
    "replace-me",
    "test",
    "todo",
    "your-token",
    "your_api_key",
}
WORKFLOW_DIRECTORY = ".github/workflows/"
AUDIT_RULE_IDS = {"SEC-001", "CI-001", "CI-002"}
PR_TARGET = re.compile(r"\bpull_request_target\b", re.IGNORECASE)
CHECKOUT_ACTION = re.compile(
    r"^\s*(?:-\s*)?uses:\s*actions/checkout(?:@|\s)",
    re.IGNORECASE | re.MULTILINE,
)
UNTRUSTED_PULL_REF = re.compile(
    r"""(?ix)
    ^\s*ref:\s*
    ["']?\$\{\{\s*github\.event\.pull_request\.head\.(?:sha|ref)\s*\}\}["']?
    \s*(?:\#.*)?$
    """,
    re.MULTILINE,
)
EXECUTES_PROJECT_CODE = re.compile(
    r"""(?ix)
    ^\s*(?:-\s*)?run:\s*.*\b
    (?:pytest|tox|nox|npm\s+(?:ci|install|test|run)|pnpm\s+(?:install|test|run)|
       yarn(?:\s+(?:install|test|run))?|make|go\s+test|cargo\s+test|
       gradle|mvn|python(?:\d+(?:\.\d+)?)?\s+-m\s+(?:pytest|unittest))
    \b
    """,
    re.MULTILINE,
)


def _finding(
    *,
    rule_id: str,
    category: str,
    severity: str,
    path: str,
    line: int,
    description: str,
    evidence: str,
    recommendation: str,
) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "category": category,
        "severity": severity,
        "file": path,
        "line": line,
        "description": description,
        "evidence": redact_secrets(evidence),
        "recommendation": recommendation,
    }


def _credential_findings(path: str, content: str) -> list[dict[str, object]]:
    findings = []
    for number, line in enumerate(content.splitlines(), start=1):
        match = SENSITIVE_ASSIGNMENT.search(line)
        if match is None:
            continue
        value = match.group("value").strip("\"'").strip()
        normalized = value.casefold()
        if (
            len(value) < 8
            or normalized in PLACEHOLDER_VALUES
            or any(
                marker in normalized
                for marker in (
                    "${",
                    "$env:",
                    "$(",
                    "os.getenv",
                    "os.environ",
                    "environ.get",
                    "secretmanager",
                    "vault.",
                    "your-",
                )
            )
        ):
            continue
        findings.append(
            _finding(
                rule_id="SEC-001",
                category="security",
                severity="high",
                path=path,
                line=number,
                description="A credential-like value is assigned directly in a repository file.",
                evidence=line,
                recommendation=(
                    "Load credentials from a secret manager or environment variable, "
                    "and rotate any credential that may have been exposed."
                ),
            )
        )
    return findings


def _workflow_findings(path: str, content: str) -> list[dict[str, object]]:
    lines = content.splitlines()
    findings: list[dict[str, object]] = []

    for number, line in enumerate(lines, start=1):
        if re.match(r"^\s*permissions:\s*write-all\s*(?:#.*)?$", line, re.IGNORECASE):
            findings.append(
                _finding(
                    rule_id="CI-001",
                    category="security",
                    severity="high",
                    path=path,
                    line=number,
                    description="The workflow grants write access to all GitHub token permission scopes.",
                    evidence=line,
                    recommendation=(
                        "Declare only the minimum required permissions at the job or "
                        "workflow level."
                    ),
                )
            )

    if (
        PR_TARGET.search(content)
        and CHECKOUT_ACTION.search(content)
        and EXECUTES_PROJECT_CODE.search(content)
    ):
        for number, line in enumerate(lines, start=1):
            if UNTRUSTED_PULL_REF.match(line):
                findings.append(
                    _finding(
                        rule_id="CI-002",
                        category="security",
                        severity="high",
                        path=path,
                        line=number,
                        description=(
                            "A pull_request_target workflow checks out pull-request head "
                            "code and runs project commands in the privileged workflow context."
                        ),
                        evidence="\n".join(
                            (
                                line,
                                "Related workflow evidence: pull_request_target trigger, "
                                "actions/checkout, and a project test/build command.",
                            )
                        ),
                        recommendation=(
                            "Do not execute pull-request code in a pull_request_target "
                            "workflow. Use pull_request, or separate privileged actions "
                            "from untrusted code execution."
                        ),
                    )
                )
                break
    return findings


def audit_inspected_files(files: Mapping[str, str]) -> dict[str, object]:
    """Run evidence-producing checks only on files the model already inspected."""
    findings: list[dict[str, object]] = []
    checked_paths = sorted(files)
    for path in checked_paths:
        content = files[path]
        findings.extend(_credential_findings(path, content))
        if path.casefold().startswith(WORKFLOW_DIRECTORY):
            findings.extend(_workflow_findings(path, content))
    return {
        "checked_files": len(checked_paths),
        "checked_paths": checked_paths,
        "findings": findings,
        "scope_note": (
            "These checks are partial evidence: they cover only files already read "
            "by the agent and only the rules listed in these results. An empty result "
            "does not mean the repository has no other issues."
        ),
    }


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
    findings = audit_inspected_files({path: inspected_files[path]})["findings"]
    return any(
        finding["rule_id"] == rule_id
        and finding["file"] == path
        and finding["line"] == line
        and finding["evidence"] == evidence
        for finding in findings
    )
