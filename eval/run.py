"""Scores audit findings against labelled synthetic repositories.

The default mode runs only the deterministic detectors over every file of a
case. That measures the detectors, not the agent. ``--live`` drives the real
agent and spends LLM quota, so it is never reached from the unit tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from auditor.checks import audit_inspected_files
from auditor.llm_client import LLMClient, LLMError
from auditor.loop import AgentError, AuditAgent
from auditor.repository import LocalDirSource, RepositoryError
from auditor.settings import ConfigError, load_env_file, load_settings
from auditor.trace import Tracer
from tools import GitHubRepository, RepositoryAddress

CASES_DIR = Path(__file__).parent / "cases"
DEFAULT_TRACE_DIR = Path("logs") / "eval"
# Models cite the line of the decisive statement, which can sit a couple of
# lines from the one a labeller picked (a key and its value, a step and its run).
LINE_TOLERANCE = load_settings().eval.line_tolerance
REPO_URL = "https://github.com/eval/{name}"

Violation = dict[str, Any]


class EvalError(Exception):
    """The harness itself is broken (bad case files, missing configuration)."""


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    expected: list[Violation]
    files: dict[str, str]
    root: Path


@dataclass
class Score:
    true_positives: int = 0
    false_positives: list[Violation] = field(default_factory=list)
    false_negatives: list[Violation] = field(default_factory=list)

    @property
    def precision(self) -> float:
        reported = self.true_positives + len(self.false_positives)
        return self.true_positives / reported if reported else 1.0

    @property
    def recall(self) -> float:
        labelled = self.true_positives + len(self.false_negatives)
        return self.true_positives / labelled if labelled else 1.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def merge(self, other: Score) -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives


def _rule_family(rule_id: str) -> str:
    return rule_id.partition("-")[0]


def _lines_match(expected: int | None, found: int | None) -> bool:
    if expected is None or found is None:
        return expected is None and found is None
    return abs(expected - found) <= LINE_TOLERANCE


def _matches(expected: Violation, found: Violation) -> bool:
    return (
        expected["category"] == found.get("category")
        and expected["file"] == found.get("file")
        and _rule_family(expected["rule_id"]) == _rule_family(str(found.get("rule_id", "")))
        and _lines_match(expected["line"], found.get("line"))
    )


def score_findings(expected: Sequence[Violation], found: Sequence[Violation]) -> Score:
    """Match each reported finding to at most one label; extras count as false positives."""
    score = Score()
    unmatched = list(expected)
    for finding in found:
        for label in unmatched:
            if _matches(label, finding):
                unmatched.remove(label)
                score.true_positives += 1
                break
        else:
            score.false_positives.append(finding)
    score.false_negatives = unmatched
    return score


def load_case(directory: Path) -> Case:
    try:
        label = json.loads((directory / "expected.json").read_text(encoding="utf-8"))
        aliases = _read_aliases(directory)
        files = _read_tree(directory / "repo", aliases)
    except (OSError, ValueError) as exc:
        raise EvalError(f"Cannot load case {directory.name}: {exc}") from exc
    return Case(directory.name, label["category"], label["violations"], files, directory)


def _read_aliases(directory: Path) -> dict[str, str]:
    path = directory / "aliases.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_tree(root: Path, aliases: dict[str, str]) -> dict[str, str]:
    """Read a case tree. Aliases restore names that a .gitignore would keep out of version control."""
    files: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stored = path.relative_to(root).as_posix()
        files[aliases.get(stored, stored)] = path.read_text(encoding="utf-8")
    return files


def load_cases(names: Sequence[str] | None = None) -> list[Case]:
    directories = sorted(p for p in CASES_DIR.iterdir() if p.is_dir())
    if names:
        known = {p.name for p in directories}
        missing = sorted(set(names) - known)
        if missing:
            raise EvalError(f"Unknown case: {', '.join(missing)}.")
        directories = [p for p in directories if p.name in names]
    return [load_case(p) for p in directories]


def detector_findings(case: Case) -> list[Violation]:
    """What the detectors report when the agent has read every file of the case."""
    result = audit_inspected_files(case.files, sorted(case.files))
    return list(result["findings"])  # type: ignore[arg-type]


def _write_tree(case: Case, destination: Path) -> None:
    for relative, text in case.files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")


def live_findings(case: Case, trace_dir: Path) -> list[Violation]:
    settings = load_settings()
    client = LLMClient(settings.llm)
    address = RepositoryAddress.parse(REPO_URL.format(name=case.name), settings.github)
    trace_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(trace_dir / f"{case.name}.jsonl", settings.trace.max_string_chars)
    with tempfile.TemporaryDirectory() as workdir:
        _write_tree(case, Path(workdir))
        source = LocalDirSource(Path(workdir), settings, "main")
        with GitHubRepository(address, settings, source) as repository:
            report = AuditAgent(repository, settings, client, tracer).run()
    return list(report["violations"])


def _describe(violation: Violation) -> str:
    return f"{violation.get('rule_id')} {violation.get('file')}:{violation.get('line')}"


def format_results(results: Sequence[tuple[Case, Score]], overall: Score, mode: str) -> str:
    width = max((len(case.name) for case, _ in results), default=4)
    lines = [
        mode,
        f"{'case':<{width}}  type  TP  FP  FN  precision  recall  f1",
    ]
    for case, score in results:
        lines.append(_row(case.name, case.category, score, width))
    lines.append(_row("overall", "-", overall, width))
    for case, score in results:
        for violation in score.false_positives:
            lines.append(f"FP {case.name}: {_describe(violation)}")
        for violation in score.false_negatives:
            lines.append(f"FN {case.name}: {_describe(violation)}")
    return "\n".join(lines)


def _row(name: str, category: str, score: Score, width: int) -> str:
    return (
        f"{name:<{width}}  {category:<4}  {score.true_positives:>2}  "
        f"{len(score.false_positives):>2}  {len(score.false_negatives):>2}  "
        f"{score.precision:>9.2f}  {score.recall:>6.2f}  {score.f1:.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__)
    parser.add_argument("--live", action="store_true", help="run the real agent (uses LLM quota)")
    parser.add_argument("--case", action="append", help="only this case; repeatable")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    return parser


def run(args: argparse.Namespace) -> str:
    cases = load_cases(args.case)
    if args.live:
        env_file = Path(".env")
        if env_file.is_file():
            load_env_file(env_file)
        mode = "live agent run"

        def found_by(case: Case) -> list[Violation]:
            return live_findings(case, args.trace_dir)

    else:
        mode = "detector baseline (no LLM); not agent accuracy"
        found_by = detector_findings
    results: list[tuple[Case, Score]] = []
    overall = Score()
    for case in cases:
        score = score_findings(case.expected, found_by(case))
        results.append((case, score))
        overall.merge(score)
    return format_results(results, overall, mode)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(run(args))
    except (EvalError, ConfigError, LLMError, AgentError, RepositoryError, OSError) as exc:
        print(f"eval harness failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
