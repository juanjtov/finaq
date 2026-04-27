"""Step 1 test plan — unit-style checks that do not require API keys.

Run via:  pytest tests/test_scaffolding.py
Integration smoke tests (real OpenRouter calls) live in test_scaffolding_integration.py
and are gated behind  pytest -m integration.
"""

from __future__ import annotations

import logging

import httpx
import pytest


def test_models_registry_exposes_nine_constants():
    """utils.models exposes the full set of model strings expected by the system."""
    from utils import models

    expected = [
        "MODEL_TRIAGE",
        "MODEL_FUNDAMENTALS",
        "MODEL_FILINGS",
        "MODEL_NEWS",
        "MODEL_RISK",
        "MODEL_SYNTHESIS",
        "MODEL_ROUTER",
        "MODEL_ADHOC_THESIS",
        "MODEL_EMBEDDINGS",
    ]
    for name in expected:
        val = getattr(models, name)
        assert isinstance(val, str) and val, f"{name} must be a non-empty string"


def test_models_registry_raises_on_missing_var(monkeypatch):
    """The _required helper raises a clear RuntimeError when an env var is missing."""
    from utils.models import _required

    monkeypatch.delenv("MODEL_DOES_NOT_EXIST", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_DOES_NOT_EXIST"):
        _required("MODEL_DOES_NOT_EXIST")


def test_openrouter_factory_requires_api_key(monkeypatch):
    """get_client() raises a clear error if OPENROUTER_API_KEY is missing."""
    from utils.openrouter import get_client

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_client()


def test_openrouter_factory_returns_client_with_api_key(monkeypatch):
    """With OPENROUTER_API_KEY set, get_client() returns an OpenAI client pointed at OpenRouter."""
    from utils.openrouter import OPENROUTER_BASE_URL, get_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-stub")
    client = get_client()
    assert str(client.base_url).startswith(OPENROUTER_BASE_URL)


def test_tenacity_retry_attempts_three_times_then_surfaces():
    """A function decorated with tenacity_retry retries 3x on httpx errors, then re-raises."""
    from utils import tenacity_retry

    call_count = {"n": 0}

    @tenacity_retry
    def always_fails():
        call_count["n"] += 1
        raise httpx.HTTPError("simulated")

    with pytest.raises(httpx.HTTPError, match="simulated"):
        always_fails()

    assert call_count["n"] == 3, f"expected 3 attempts, got {call_count['n']}"


def test_tenacity_retry_succeeds_after_transient_failures():
    """tenacity_retry returns the value once a retried call succeeds."""
    from utils import tenacity_retry

    call_count = {"n": 0}

    @tenacity_retry
    def flaky():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise httpx.HTTPError("transient")
        return "ok"

    assert flaky() == "ok"
    assert call_count["n"] == 2


def test_logger_writes_configured_format(caplog):
    """The shared logger emits at INFO and is named 'finaq'."""
    from utils import logger

    with caplog.at_level(logging.INFO, logger="finaq"):
        logger.info("hello scaffolding")

    assert any(
        "hello scaffolding" in record.message and record.name == "finaq"
        for record in caplog.records
    )
