"""JSONL trace of one audit run, written so it is safe to share.

Events carry names, counts, locations and short reasons, never message bodies or
tool output. Every string is redacted again on its way out because tool arguments
are model-written text and can echo a credential the model saw.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from auditor.redaction import redact_secrets
from auditor.settings import TraceSettings
from tools import RepositoryAddress


# Credentials the process itself holds; masked by exact value in case the pattern
# rules do not recognise their format.
_SECRET_ENV_NAMES = ("OPENAI_API_KEY", "GITHUB_TOKEN")
_MASK = "[REDACTED]"
_FILE_STAMP = "%Y%m%dT%H%M%SZ"


def trace_path(
    settings: TraceSettings, address: RepositoryAddress, now: datetime | None = None
) -> Path | None:
    """Default trace location for a run, or None when tracing is disabled."""
    if not settings.enabled:
        return None
    moment = now or datetime.now(timezone.utc)
    name = f"{address.owner}-{address.name}-{moment.strftime(_FILE_STAMP)}.jsonl"
    return Path(settings.directory) / name


class Tracer:
    def __init__(
        self,
        path: Path | str | None,
        max_string_chars: int,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._max_string_chars = max_string_chars
        self._secrets = [
            value for name in _SECRET_ENV_NAMES if (value := environ.get(name, "").strip())
        ]
        self._failed = False

    def event(self, kind: str, turn: int, **data: Any) -> None:
        """Append one line; a write failure disables tracing instead of failing the audit."""
        if self.path is None or self._failed:
            return
        line = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "turn": turn,
                "kind": kind,
                "data": self._clean(data),
            },
            ensure_ascii=True,
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            self._failed = True
            print(f"Warning: trace disabled, cannot write {self.path}: {exc.strerror}.", file=sys.stderr)

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clean_text(value)
        if isinstance(value, Mapping):
            return {self._clean_text(str(key)): self._clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(item) for item in value]
        return value

    def _clean_text(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, _MASK)
        text = redact_secrets(text)
        # Cut after redaction so a token is never left half-masked by the cut.
        if len(text) > self._max_string_chars:
            return text[: self._max_string_chars] + "..."
        return text
