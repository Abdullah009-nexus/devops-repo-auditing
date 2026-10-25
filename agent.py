"""Command-line entry point for the DevOps repository auditing agent.

Exit codes: 0 audit completed (an empty violations array counts), 1 runtime failure,
2 bad input or configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from auditor.llm_client import LLMClient, LLMError
from auditor.loop import AgentError, AuditAgent
from auditor.redaction import packaged_secret_rules
from auditor.report import ReportError, validate_report_shape, write_report
from auditor.settings import ConfigError, Settings, load_env_file, load_settings
from auditor.trace import Tracer, trace_path
from tools import GitHubRepository, RepositoryAddress, RepositoryError


def run_agent(
    repository: GitHubRepository, settings: Settings, client: LLMClient, tracer: Tracer
) -> dict[str, Any]:
    return AuditAgent(repository, settings, client, tracer).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a public GitHub repository for evidence-based DevOps issues."
    )
    parser.add_argument("--repo", required=True, help="Public GitHub repository URL")
    parser.add_argument("--output", required=True, help="Path for the JSON report")
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML file overriding the packaged defaults (AUDITOR_* variables still win)",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="Write the JSONL run trace here instead of the default logs directory",
    )
    parser.add_argument("--verbose", action="store_true", help="Log progress to stderr")
    return parser


def _fail(exc: Exception, code: int) -> int:
    print(f"Error: {exc}", file=sys.stderr)
    return code


def _check_output_writable(output: str) -> None:
    """Fail before the audit, not after it, when the report could not be saved."""
    destination = Path(output)
    if destination.is_dir():
        raise OSError(f"Output path {output} is a directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # write_report creates a sibling temp file, so probe the same way.
    with tempfile.TemporaryFile(dir=destination.parent):
        pass


def _log_progress() -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("auditor")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def _audit(args: argparse.Namespace) -> int:
    try:
        env_file = Path(".env")
        if env_file.is_file():
            load_env_file(env_file)
        settings = load_settings(args.config)
        # Loaded up front so a broken rules file is a config error, not a crash mid-run.
        packaged_secret_rules()
        # Built before the clone so a missing key costs nothing.
        client = LLMClient(settings.llm)
        address = RepositoryAddress.parse(args.repo, settings.github)
    except (ConfigError, LLMError, RepositoryError) as exc:
        return _fail(exc, 2)
    try:
        _check_output_writable(args.output)
    except OSError as exc:
        return _fail(exc, 1)
    tracer = Tracer(args.trace or trace_path(settings.trace, address), settings.trace.max_string_chars)
    try:
        # The context manager deletes the temporary clone on every exit path.
        with GitHubRepository(address, settings) as repository:
            report = run_agent(repository, settings, client, tracer)
        validate_report_shape(report)
        write_report(args.output, report)
    except (AgentError, RepositoryError, ReportError, OSError) as exc:
        return _fail(exc, 1)
    print(f"Audit report written to {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = _log_progress() if args.verbose else None
    try:
        return _audit(args)
    finally:
        if handler is not None:
            logging.getLogger("auditor").removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
