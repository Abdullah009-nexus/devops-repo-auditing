"""Chat-completions client for an OpenAI-compatible provider, using urllib only.

Error messages carry a status code or a fixed phrase and nothing else: the API key,
the request body and the provider's response body must never reach logs or reports.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from auditor.settings import ConfigError, LlmSettings, check_https_url


class LLMError(Exception):
    """Raised for missing credentials, unusable responses or exhausted retries."""


# Rate limiting and transient server faults; any other 4xx means the request itself is wrong.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A provider has no reason to redirect, and following one would resend the API key."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LLMError("LLM provider redirected the request; refusing to resend credentials.")


# Not urllib.request.urlopen: its default opener follows redirects.
_opener = urllib.request.build_opener(_NoRedirect)


class LLMClient:
    def __init__(
        self,
        settings: LlmSettings,
        environ: Mapping[str, str] = os.environ,
        *,
        sleep: Callable[[float], None] | None = None,
        random_fraction: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        # Looked up at call time rather than bound as defaults, so patching
        # time.sleep or random.random also reaches clients built elsewhere.
        self._sleep = sleep or (lambda seconds: time.sleep(seconds))
        self._random = random_fraction or (lambda: random.random())
        self.api_key = environ.get("OPENAI_API_KEY", "").strip()
        self.model = environ.get("OPENAI_MODEL", "").strip()
        base_url = environ.get("OPENAI_BASE_URL", settings.default_base_url).strip()
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is required.")
        if not self.model:
            raise LLMError("OPENAI_MODEL is required.")
        try:
            check_https_url("OPENAI_BASE_URL", base_url)
        except ConfigError as exc:
            raise LLMError(
                "OPENAI_BASE_URL must be an HTTPS URL without credentials, query or fragment."
            ) from exc
        self.base_url = base_url.rstrip("/")
        # The provider's token counts for the latest response, when it sent any.
        self.last_usage: dict[str, Any] | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Send one completion request and return the assistant message."""
        request = self._build_request(messages, tools, tool_choice)
        attempt = 0
        while True:
            try:
                message, usage = self._parse(self._send(request))
                self.last_usage = usage
                return message
            except _Transient as exc:
                if attempt >= self._settings.max_retries:
                    raise LLMError(exc.final_message) from exc
                self._sleep(self._delay(attempt, exc.retry_after))
                attempt += 1

    def _build_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str
    ) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self._settings.temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=True).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self._settings.user_agent,
            },
            method="POST",
        )

    def _send(self, request: urllib.request.Request) -> bytes:
        limit = self._settings.max_response_bytes
        try:
            with _opener.open(
                request, timeout=self._settings.request_timeout_seconds
            ) as response:
                raw = response.read(limit + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRYABLE_STATUS:
                raise _Transient(
                    f"LLM provider request failed with HTTP {exc.code}.",
                    _retry_after(exc.headers),
                ) from exc
            raise LLMError(f"LLM provider request failed with HTTP {exc.code}.") from exc
        except TimeoutError as exc:
            raise _Transient("LLM provider request timed out.") from exc
        except urllib.error.URLError as exc:
            # A timeout during connect arrives wrapped in URLError.
            if isinstance(exc.reason, TimeoutError):
                raise _Transient("LLM provider request timed out.") from exc
            raise _Transient("LLM provider could not be reached.") from exc
        except (OSError, http.client.HTTPException) as exc:
            # Connection resets and truncated bodies surface here, mid-request or mid-read.
            raise _Transient("LLM provider connection failed.") from exc
        if len(raw) > limit:
            raise LLMError("LLM provider response exceeded the configured size limit.")
        return raw

    @staticmethod
    def _parse(raw: bytes) -> tuple[dict[str, Any], dict[str, Any] | None]:
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMError("LLM provider returned invalid JSON.") from exc
        choices = result.get("choices") if isinstance(result, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LLMError("LLM provider response did not contain a completion choice.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise LLMError("LLM provider returned a malformed completion message.")
        usage = result.get("usage")
        return message, usage if isinstance(usage, dict) else None

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        cap = self._settings.backoff_max_seconds
        if retry_after is not None:
            return min(retry_after, cap)
        # Full jitter keeps parallel runs from retrying in lockstep.
        ceiling = min(cap, self._settings.backoff_base_seconds * 2**attempt)
        return self._random() * ceiling


class _Transient(Exception):
    def __init__(self, final_message: str, retry_after: float | None = None) -> None:
        super().__init__(final_message)
        self.final_message = final_message
        self.retry_after = retry_after


def _retry_after(headers: Any) -> float | None:
    """Seconds from a numeric Retry-After header; the HTTP-date form is ignored."""
    value = headers.get("Retry-After") if headers is not None else None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None
