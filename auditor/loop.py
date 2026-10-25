"""The audit loop: the model picks every tool call, this module only enforces limits.

Nothing in here is fatal except a provider that stays unreachable. A bad finding, a
bad tool call or an exhausted budget is turned into a message to the model, so a
run always ends with a valid, possibly partial, report.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from auditor.context import compact
from auditor.findings import (
    FINISH_AUDIT,
    REPORT_FINDING,
    FindingStore,
    build_finding_tools,
)
from auditor.llm_client import LLMClient, LLMError
from auditor.prompts import load_prompt
from auditor.redaction import redact_secrets
from auditor.report import build_report
from auditor.rules import render_catalog_for_prompt
from auditor.settings import Settings
from auditor.trace import Tracer
from tools import GitHubRepository, RepositoryError, RepositoryTools, build_tool_schemas


LOGGER = logging.getLogger(__name__)

_FENCE = "```"
_TRUNCATION_NOTICE = "The complete tool result exceeded the output limit."
_STUB_PREFIX = '{"compacted": true'
LEDGER_MARKER = "PROGRESS LEDGER"
_LEDGER_RULES = (
    "Don't repeat a call you have already made. Record each finding with report_finding as "
    "soon as you have confirmed evidence for it, instead of saving them for the end. When the "
    "checklist in your instructions is covered, call finish_audit."
)



class AgentError(Exception):
    """Raised when the provider fails for good, so no complete report exists."""


class AuditAgent:
    def __init__(
        self,
        repository: GitHubRepository,
        settings: Settings,
        client: LLMClient,
        tracer: Tracer,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._limits = settings.agent
        self._client = client
        self._tracer = tracer
        self._clock = clock
        self._tools = RepositoryTools(repository)
        self._store = FindingStore(repository, settings.agent)
        self._inspection_schemas = build_tool_schemas(settings.git_log)
        self._finding_schemas = build_finding_tools()
        self._messages: list[dict[str, Any]] = []
        self._turn = 0
        self._tool_calls = 0
        self._rejected = 0
        self._text_replies = 0
        self._started = 0.0
        # Insertion-ordered sets: what the model has looked at, for the per-turn ledger.
        self._read_paths: dict[str, None] = {}
        self._searches: dict[str, None] = {}
        # (tool, canonical arguments) -> (turn, tool_call_id) of the run that produced a result.
        self._executed: dict[tuple[str, str], tuple[int, str]] = {}
        self._repeat_replies = 0

    def run(self) -> dict[str, Any]:
        """Investigate until the model finishes or a budget ends, and return the report."""
        url = self._repository.address.url
        self._started = self._clock()
        self._messages = [
            {
                "role": "system",
                "content": load_prompt(
                    "system",
                    repository_url=url,
                    rule_catalog=render_catalog_for_prompt(),
                    max_tool_calls=self._limits.max_tool_calls,
                    max_turns=self._limits.max_turns,
                ),
            },
            {"role": "user", "content": load_prompt("start", repository_url=url)},
        ]
        source = self._repository.source
        self._tracer.event(
            "run_start",
            0,
            repository=url,
            backend=source.backend,
            fallback_reason=source.fallback_reason,
            model=self._client.model,
            budgets={
                "max_turns": self._limits.max_turns,
                "max_tool_calls": self._limits.max_tool_calls,
                "max_wall_seconds": self._limits.max_wall_seconds,
                "finalize_turns": self._limits.finalize_turns,
            },
        )
        try:
            status = self._investigate()
            if status != "finished":
                self._finalize(status)
        except AgentError:
            self._end("failed")
            raise
        self._end(status)
        return build_report(url, self._store.findings)

    def _investigate(self) -> str:
        investigation_turns = 0
        while True:
            exhausted = self._exhausted(investigation_turns)
            if exhausted:
                return exhausted
            investigation_turns += 1
            offered = self._inspection_schemas + self._finding_schemas
            if self._step(offered, allow_inspection=True):
                return "finished"

    def _exhausted(self, investigation_turns: int) -> str | None:
        if investigation_turns >= self._limits.max_turns:
            return "turn_limit"
        if self._tool_calls >= self._limits.max_tool_calls:
            return "tool_call_limit"
        if self._clock() - self._started >= self._limits.max_wall_seconds:
            return "wall_time_limit"
        return None

    def _finalize(self, reason: str) -> None:
        accepted = len(self._store.findings)
        self._tracer.event("finalize", self._turn, reason=reason, accepted=accepted)
        self._messages.append(
            {
                "role": "user",
                "content": f"{load_prompt('finalize')}\nFindings accepted so far: {accepted}.",
            }
        )
        # A fresh window: when the wall budget caused the finalize, it is already spent.
        deadline = self._clock() + self._limits.finalize_wall_seconds
        for _ in range(self._limits.finalize_turns):
            if self._clock() >= deadline:
                self._tracer.event("finalize", self._turn, reason="finalize_wall_time_limit")
                return
            if self._step(self._finding_schemas, allow_inspection=False):
                return

    def _step(self, offered: list[dict[str, Any]], *, allow_inspection: bool) -> bool:
        """Run one model turn; True when the model is done."""
        self._turn += 1
        self._messages = compact(self._messages, self._settings.context)
        # A model call can take many seconds; without this line a verbose run looks frozen.
        LOGGER.info("turn %d: asking the model (%d messages) ...", self._turn, len(self._messages))
        asked = self._clock()
        try:
            # The ledger is appended to the request only, so it never piles up in the history.
            message = self._client.chat(self._messages + [self._ledger()], offered)
        except LLMError as exc:
            raise AgentError(str(exc)) from exc
        LOGGER.info("turn %d: model replied in %.1f s", self._turn, self._clock() - asked)
        calls = _well_formed_calls(message)
        self._tracer.event(
            "llm_response",
            self._turn,
            tools=[call["function"]["name"] for call in calls],
            usage=self._client.last_usage,
        )
        if not calls:
            return self._text_reply(message)
        self._text_replies = 0
        # Only the fields every provider accepts are sent back; extras are not ours to replay.
        self._messages.append(
            {"role": "assistant", "content": _text_or_none(message.get("content")), "tool_calls": calls}
        )
        finished = False
        for call in calls:
            name = call["function"]["name"]
            LOGGER.info("turn %d: %s", self._turn, name)
            content = self._dispatch(
                name, call["function"]["arguments"], allow_inspection, call["id"]
            )
            self._messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )
            finished = finished or name == FINISH_AUDIT
        return finished

    def _dispatch(
        self, name: str, raw_arguments: str, allow_inspection: bool, call_id: str = ""
    ) -> str:
        if name == FINISH_AUDIT:
            return json.dumps({"finished": True, "accepted": len(self._store.findings)})
        arguments, problem = _parse_arguments(raw_arguments)
        if name == REPORT_FINDING:
            if arguments is not None:
                # file and line are optional in the schema; leaving them out means null.
                arguments = {"file": None, "line": None, **arguments}
            return json.dumps(self._record_finding(arguments, problem))
        if arguments is None:
            return self._tool_failure(name, None, problem or "Tool arguments were unusable.")
        if not allow_inspection:
            return self._tool_failure(
                name, arguments, "The inspection budget is used up; only report_finding and "
                "finish_audit are available."
            )
        if self._tool_calls >= self._limits.max_tool_calls:
            return self._tool_failure(name, arguments, "The tool call budget is used up.")
        repeated = self._repeat_reply(name, arguments)
        if repeated is not None:
            return repeated
        self._tool_calls += 1
        self._executed[(name, _canonical(arguments))] = (self._turn, call_id)
        self._note_inspection(name, arguments)
        try:
            outcome: dict[str, Any] = self._tools.call(name, arguments)
        except (RepositoryError, OSError) as exc:
            return self._tool_failure(name, arguments, str(exc))
        return self._bounded_result(name, arguments, outcome)

    def _repeat_reply(self, name: str, arguments: dict[str, Any]) -> str | None:
        """A short reply for an identical call whose full result is still in the context."""
        earlier = self._executed.get((name, _canonical(arguments)))
        if earlier is None or self._repeat_replies >= self._limits.max_repeat_replies:
            return None
        turn, call_id = earlier
        # A compacted result is gone from the model's view, so the call has to run again.
        if not self._is_in_full(call_id):
            return None
        self._repeat_replies += 1
        content = json.dumps(
            {
                "repeat": True,
                "note": f"Same call as turn {turn}; the result is unchanged and still above.",
            }
        )
        self._tracer.event(
            "tool_call",
            self._turn,
            name=name,
            arguments=arguments,
            result_chars=len(content),
            truncated=False,
            error="repeat_skipped",
        )
        return content

    def _is_in_full(self, call_id: str) -> bool:
        for message in self._messages:
            if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
                return not (message.get("content") or "").startswith(_STUB_PREFIX)
        return False

    def _note_inspection(self, name: str, arguments: dict[str, Any]) -> None:
        path = arguments.get("path")
        pattern = arguments.get("pattern")
        if name == "read_file" and isinstance(path, str):
            self._read_paths[path] = None
        elif name == "search_code" and isinstance(pattern, str):
            self._searches[pattern] = None

    def _ledger(self) -> dict[str, str]:
        paths = list(self._read_paths)[-self._limits.ledger_max_paths:]
        searches = list(self._searches)[-self._limits.ledger_max_searches:]
        found = [
            f"{item.rule_id} {item.file}:{item.line}" if item.file is not None else f"{item.rule_id} (no file)"
            for item in self._store.findings
        ]
        lines = [
            f"{LEDGER_MARKER}: turn {self._turn} of {self._limits.max_turns}; "
            f"tool calls used {self._tool_calls} of {self._limits.max_tool_calls}.",
            f"Files read so far ({len(self._read_paths)}): {', '.join(paths) or 'none'}",
            f"Searches already run ({len(self._searches)}): {'; '.join(searches) or 'none'}",
            f"Findings accepted so far ({len(found)}): {', '.join(found) or 'none'}",
            _LEDGER_RULES,
        ]
        # Paths and patterns come from the model and may quote a secret it searched for.
        return {"role": "user", "content": redact_secrets("\n".join(lines))}

    def _record_finding(self, candidate: Any, problem: str | None = None) -> dict[str, Any]:
        """Offer a finding to the store; a rejection is returned to the model, not raised."""
        if problem is None:
            decision = self._store.add(candidate)
            accepted, reason = decision.accepted, decision.reason
        else:
            accepted, reason = False, problem
        fields = candidate if isinstance(candidate, dict) else {}
        self._tracer.event(
            "finding",
            self._turn,
            accepted=accepted,
            rule_id=fields.get("rule_id"),
            file=fields.get("file"),
            line=fields.get("line"),
            reason=reason,
        )
        if accepted:
            return {"accepted": True, "findings_so_far": len(self._store.findings)}
        self._rejected += 1
        result: dict[str, Any] = {"accepted": False, "reason": reason}
        if problem is None and decision.duplicate:
            result["duplicate"] = True
        return result

    def _tool_failure(
        self, name: str, arguments: dict[str, Any] | None, message: str
    ) -> str:
        content = json.dumps({"error": message})
        self._tracer.event(
            "tool_call",
            self._turn,
            name=name,
            arguments=arguments,
            result_chars=len(content),
            truncated=False,
            error=message,
        )
        return content

    def _bounded_result(self, name: str, arguments: dict[str, Any], outcome: dict[str, Any]) -> str:
        serialized = json.dumps(outcome, ensure_ascii=True)
        limit = self._limits.max_tool_output_chars
        truncated = len(serialized) > limit
        if truncated:
            serialized = _truncated(serialized, limit)
        self._tracer.event(
            "tool_call",
            self._turn,
            name=name,
            arguments=arguments,
            result_chars=len(serialized),
            truncated=truncated,
            error=None,
        )
        return serialized

    def _text_reply(self, message: dict[str, Any]) -> bool:
        content = _text_or_none(message.get("content")) or ""
        self._messages.append({"role": "assistant", "content": content})
        candidates = _legacy_violations(content)
        if candidates is not None:
            # A model that ignores the tools and answers with the old JSON report still
            # gets every violation checked against the files it read.
            for candidate in candidates:
                self._record_finding(candidate)
            return True
        self._text_replies += 1
        # Repeated plain replies mean the model is done or cannot follow the tool
        # protocol; going on would only burn the remaining turns.
        if self._text_replies >= self._limits.max_text_replies:
            return True
        self._messages.append({"role": "user", "content": load_prompt("nudge")})
        return False

    def _end(self, status: str) -> None:
        self._tracer.event(
            "run_end",
            self._turn,
            status=status,
            accepted=len(self._store.findings),
            rejected=self._rejected,
            turns=self._turn,
            tool_calls=self._tool_calls,
            elapsed_seconds=round(self._clock() - self._started, 3),
        )


def _well_formed_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool calls that can be answered; a malformed one is dropped, not fatal."""
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    usable = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(call.get("id"), str) or not isinstance(name, str):
            continue
        if arguments is None:
            arguments = ""
        elif isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            continue
        usable.append(
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return usable


def _truncated(serialized: str, limit: int) -> str:
    """A JSON wrapper around as much of ``serialized`` as fits within ``limit`` characters."""

    def wrap(length: int) -> str:
        return json.dumps(
            {
                "result_truncated": True,
                "preview": serialized[:length],
                "notice": _TRUNCATION_NOTICE,
            }
        )

    length = max(limit - len(wrap(0)), 0)
    wrapped = wrap(length)
    # Quotes and backslashes in the preview are escaped again, so the wrapper can
    # come out longer than the slice; trim by the overshoot until it fits.
    while len(wrapped) > limit and length > 0:
        length = max(length - (len(wrapped) - limit), 0)
        wrapped = wrap(length)
    return wrapped


def _canonical(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=True)


def _text_or_none(content: Any) -> str | None:
    return content if isinstance(content, str) else None


def _parse_arguments(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    # Models send an empty string for tools that take no arguments.
    if not raw.strip():
        return {}, None
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        return None, "Tool arguments were not valid JSON."
    if not isinstance(arguments, dict):
        return None, "Tool arguments must be a JSON object."
    return arguments, None


def _legacy_violations(content: str) -> list[Any] | None:
    text = content.strip()
    if text.startswith(_FENCE):
        text = text.removeprefix(_FENCE).removeprefix("json").strip().removesuffix(_FENCE).strip()
    try:
        report = json.loads(text)
    except json.JSONDecodeError:
        return None
    violations = report.get("violations") if isinstance(report, dict) else None
    return violations if isinstance(violations, list) else None
