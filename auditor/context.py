"""Keeps the conversation inside a character budget by stubbing old tool results.

Only tool message contents are ever replaced. Every assistant tool_call keeps its
reply, because providers reject a tool_call without a matching tool message.
"""

from __future__ import annotations

import json
from typing import Any

from auditor.settings import ContextSettings


_STUB_PREFIX = '{"compacted": true'
# Without the arguments and this wording the model re-ran the same call to find out what was dropped.
_STUB_NOTE = (
    "Content dropped to save space. You already inspected this; rely on your notes or "
    "findings, and re-run only to re-check one specific line."
)


def compact(messages: list[dict[str, Any]], settings: ContextSettings) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with older tool results replaced by one-line stubs."""
    tool_indexes = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    # max(): with fewer results than the limit a negative start would select only the last few.
    recent = set(tool_indexes[max(0, len(tool_indexes) - settings.keep_recent_tool_results):])
    calls = _tool_calls(messages, settings.max_stub_args_chars)
    result = list(messages)
    for index in tool_indexes:
        if index not in recent:
            result[index] = _stubbed(result[index], calls)
    # Still over budget: spend the protected results too, oldest first.
    for index in tool_indexes:
        if _total_chars(result) <= settings.max_total_chars:
            break
        result[index] = _stubbed(result[index], calls)
    return result


def _tool_calls(messages: list[dict[str, Any]], max_args_chars: int) -> dict[str, tuple[str, Any]]:
    calls: dict[str, tuple[str, Any]] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            calls[call["id"]] = (function["name"], _short_args(function.get("arguments"), max_args_chars))
    return calls


def _short_args(raw: Any, limit: int) -> Any:
    """The call's arguments as a dict when they fit in ``limit`` characters, else a text prefix."""
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        text = json.dumps(parsed)
        if len(text) <= limit:
            return parsed
    else:
        text = raw if isinstance(raw, str) else ""
    return text[:limit]


def _stubbed(message: dict[str, Any], calls: dict[str, tuple[str, Any]]) -> dict[str, Any]:
    content = message.get("content") or ""
    if content.startswith(_STUB_PREFIX):
        return message
    name, args = calls.get(message.get("tool_call_id"), ("unknown", {}))
    stub = json.dumps(
        {
            "compacted": True,
            "tool": name,
            "args": args,
            "chars": len(content),
            "note": _STUB_NOTE,
        }
    )
    # Replacing a result that is already shorter than its stub would only grow the prompt.
    if len(stub) >= len(content):
        return message
    return {**message, "content": stub}


def _total_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            total += len(call["function"].get("arguments") or "")
    return total
