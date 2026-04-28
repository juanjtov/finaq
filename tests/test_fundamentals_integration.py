"""Step 5a integration tests — real yfinance + real OpenRouter LLM call.

Run via:  pytest -m integration tests/test_fundamentals_integration.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.fundamentals import run
from utils.schemas import FundamentalsOutput

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")


@pytest.mark.asyncio
async def test_fundamentals_real_run_on_nvda_ai_cake():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = FundamentalsOutput.model_validate(result["fundamentals"])

    # Sanity-check projection ranges (3-sigma to public consensus).
    p = out.projections
    assert 0.0 <= p.revenue_growth_mean <= 0.50, p.revenue_growth_mean
    assert 0.0 < p.revenue_growth_std <= 0.20, p.revenue_growth_std
    assert 0.10 <= p.margin_mean <= 0.80, p.margin_mean
    assert 0.0 < p.margin_std <= 0.15, p.margin_std
    assert 5.0 <= p.exit_multiple_mean <= 100.0, p.exit_multiple_mean
    assert 0.0 < p.exit_multiple_std <= 30.0, p.exit_multiple_std

    # Evidence — at least 1 yfinance citation per the prompt instructions.
    assert any(e.source == "yfinance" for e in out.evidence)

    # Summary must mention something thesis-relevant.
    summary_lower = out.summary.lower()
    assert any(
        kw in summary_lower
        for kw in ("data center", "data-center", "hyperscaler", "ai", "gpu", "capex")
    ), f"summary does not appear thesis-aware: {out.summary!r}"


@pytest.mark.asyncio
async def test_fundamentals_real_run_on_avgo_ai_cake():
    """Second ticker for the same thesis — confirms the agent isn't NVDA-specific."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "AVGO", "thesis": thesis}
    result = await run(state)
    out = FundamentalsOutput.model_validate(result["fundamentals"])

    p = out.projections
    assert 0.0 <= p.revenue_growth_mean <= 0.50
    assert 5.0 <= p.exit_multiple_mean <= 100.0
    assert any(e.source == "yfinance" for e in out.evidence)
