"""Centralized LLM configuration loaded from the repository ``.env`` file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / ".env"

# Explicit process environment values take precedence over .env values.
load_dotenv(ENV_FILE, override=False)


def get_env(key: str, default: str | None = None) -> str:
    """Return a non-empty environment value or a supplied default.

    Empty values are treated as missing. This makes checked-in examples with
    blank secrets fail early instead of producing an HTTP authentication error.
    """

    value = os.getenv(key)
    if value is not None and value.strip():
        return value.strip()
    if default is not None:
        return default
    raise RuntimeError(
        f"Missing environment variable: {key}. "
        "Please create .env from .env.example and configure it."
    )


def get_optional_env(key: str) -> str | None:
    """Return a non-empty environment value, or ``None`` when unset."""

    value = os.getenv(key)
    return value.strip() if value is not None and value.strip() else None

