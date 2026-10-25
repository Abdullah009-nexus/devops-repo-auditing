"""Shared helpers for tests that need a fully loaded Settings object."""

from __future__ import annotations

from auditor.settings import Settings, load_settings


def default_settings() -> Settings:
    return load_settings()
