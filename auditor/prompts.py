"""Loads prompt text from auditor/prompts/*.txt so wording is never embedded in code."""

from __future__ import annotations

from pathlib import Path
from string import Template


PROMPT_DIR = Path(__file__).with_name("prompts")


def load_prompt(name: str, **values: object) -> str:
    """Render ``prompts/<name>.txt``, filling ``$placeholders`` from ``values``.

    string.Template is used rather than str.format so the JSON braces in a prompt
    need no escaping. A placeholder without a value raises KeyError.
    """
    text = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return Template(text).substitute({key: str(value) for key, value in values.items()})
