"""Typed registry of OpenRouter model strings, sourced from .env.

Swap any model by editing the corresponding env var — no code change required.
See .env.example for the canonical list of variables and 2026-04-26 defaults.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Required env var {name} is not set. See .env.example.")
    return val


MODEL_TRIAGE: str = _required("MODEL_TRIAGE")
MODEL_FUNDAMENTALS: str = _required("MODEL_FUNDAMENTALS")
MODEL_FILINGS: str = _required("MODEL_FILINGS")
MODEL_NEWS: str = _required("MODEL_NEWS")
MODEL_RISK: str = _required("MODEL_RISK")
MODEL_SYNTHESIS: str = _required("MODEL_SYNTHESIS")
MODEL_ROUTER: str = _required("MODEL_ROUTER")
MODEL_ADHOC_THESIS: str = _required("MODEL_ADHOC_THESIS")
MODEL_EMBEDDINGS: str = _required("MODEL_EMBEDDINGS")
