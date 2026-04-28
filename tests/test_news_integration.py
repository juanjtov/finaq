"""Step 5c integration tests — real Tavily search + real LLM extraction.

Run via:  pytest -m integration tests/test_news_integration.py

Costs ~$0.005 (Tavily) + ~$0.02 (LLM) ≈ $0.025 per ticker run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.news import run
from utils.schemas import NewsOutput

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    if not os.environ.get("TAVILY_API_KEY", "").startswith("tvly-"):
        pytest.skip("TAVILY_API_KEY not set")


@pytest.mark.asyncio
async def test_news_real_run_on_nvda_ai_cake():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])

    if "STALE NEWS" in out.summary:
        pytest.skip("Tavily returned no fresh NVDA news today; skipping")

    # Sanity-check the output shape
    assert out.summary, "summary is empty"
    assert (
        3 <= len(out.catalysts) + len(out.concerns) <= 14
    ), f"unexpected total items: catalysts={len(out.catalysts)} concerns={len(out.concerns)}"

    # Every NewsItem must have a working URL
    for item in [*out.catalysts, *out.concerns]:
        assert item.url.startswith("http"), f"bad URL: {item.url}"
        assert item.sentiment in ("bull", "bear", "neutral")

    # Evidence must carry as_of (ISO date) for downstream Risk + Synthesis
    assert out.evidence, "no evidence emitted"
    for ev in out.evidence:
        assert ev.source == "tavily", ev.source
        # as_of optional but if present must be ISO-ish
        if ev.as_of:
            assert len(ev.as_of) >= 10, f"bad as_of format: {ev.as_of!r}"
