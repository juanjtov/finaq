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


# --- humanize_amount -------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (215_938_000_000, "$215.94B"),
        (30_000_000_000, "$30.00B"),
        (1_500_000, "$1.50M"),
        (2_500_000_000_000, "$2.50T"),
        (213.17, "$213.17"),
        (0, "$0.00"),
        (-12_300_000, "$-12.30M"),  # negative passes through with sign
    ],
)
def test_humanize_amount_with_dollar_prefix(value, expected):
    from utils import humanize_amount

    assert humanize_amount(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (24_300_000_000, "24.30B"),
        (500_000_000, "500.00M"),
        (1_500, "1.50K"),  # K only kicks in when no $ prefix
    ],
)
def test_humanize_amount_no_prefix(value, expected):
    from utils import humanize_amount

    assert humanize_amount(value, prefix="") == expected


def test_humanize_amount_handles_none():
    from utils import humanize_amount

    assert humanize_amount(None) == "—"


def test_humanize_amount_handles_nan():
    from utils import humanize_amount

    assert humanize_amount(float("nan")) == "—"


def test_humanize_amount_handles_non_numeric():
    """Anything not coercible to float falls back to str()."""
    from utils import humanize_amount

    assert humanize_amount("oops") == "oops"


def test_humanize_amount_currency_in_thousands_stays_uncompacted():
    """$1,234 is more readable than $1.23K so we don't compact below 1M
    when prefix='$'. The K threshold only kicks in for unit-less values."""
    from utils import humanize_amount

    # Currency: full number, comma separators
    assert humanize_amount(1_234, prefix="$") == "$1,234.00"
    # No prefix: K kicks in
    assert humanize_amount(1_234, prefix="") == "1.23K"


def test_humanize_amount_precision_argument():
    from utils import humanize_amount

    assert humanize_amount(215_938_000_000, precision=0) == "$216B"
    assert humanize_amount(215_938_000_000, precision=3) == "$215.938B"
