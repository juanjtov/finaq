"""OpenRouter client factory. All FINAQ LLM calls flow through this."""

from __future__ import annotations

import os

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MAX_RETRIES = 3


def get_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at OpenRouter.

    The OpenAI SDK has built-in retries; we set max_retries=3 here so callers
    don't need to wrap LLM calls with tenacity_retry separately.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, max_retries=OPENROUTER_MAX_RETRIES)
