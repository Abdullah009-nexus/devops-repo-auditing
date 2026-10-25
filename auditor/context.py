"""Keeps the conversation inside a character budget by stubbing old tool results.

Only tool message contents are ever replaced. Every assistant tool_call keeps its
reply, because providers reject a tool_call without a matching tool message.
"""

from __future__ import annotations

import json
from typing import Any

from auditor.settings import ContextSettings


_STUB_PREFIX = '{"compacted": true'


def compact(messages: list[dict[str, Any]], settings: ContextSettings) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with older tool results replaced by one-line stubs."""
    tool_indexes = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    recent = set(tool_indexes[len(tool_indexes) - settings.keep_recent_tool_results:])
    names = _tool_names(messages)
    result = list(messages)
    for index in tool_indexes:
        if index not in recent:
            result[index] = _stubbed(result[index], names)
    # Still over budget: spend the protected results too, oldest first.
    for index in tool_indexes:
        if _total_chars(result) <= settings.max_total_chars:
            break
        result[index] = _stubbed(result[index], names)
    return result


def _tool_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            names[call["id"]] = call["function"]["name"]
    return names


def _stubbed(message: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    content = message.get("content") or ""
    if content.startswith(_STUB_PREFIX):
        return message
    stub = json.dumps(
        {
            "compacted": True,
            "tool": names.get(message.get("tool_call_id"), "unknown"),
            "chars": len(content),
            "note": "re-run the tool if needed",
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
