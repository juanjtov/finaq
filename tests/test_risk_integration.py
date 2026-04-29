"""Step 5d integration tests — Risk agent end-to-end with real LLM.

Run via:  pytest -m integration tests/test_risk_integration.py
Cost: ~$0.02 per ticker run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.risk import run
from utils.schemas import RISK_LEVEL_TO_SCORE, RiskOutput

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")


# A minimal but realistic worker-output fixture so we can exercise the Risk
# LLM call without first running the (slow) Fundamentals/Filings/News agents.
def _stub_state(ticker: str, thesis: dict) -> dict:
    return {
        "ticker": ticker,
        "thesis": thesis,
        "fundamentals": {
            "summary": f"{ticker} shows strong revenue growth (CAGR ~100%) but FCF yield 1.8% suggests overvaluation versus the thesis margin-of-safety threshold of 4%.",
            "kpis": {
                "revenue_5y_cagr": 1.0,
                "fcf_yield": 1.8,
                "operating_margin_5yr_avg": 0.49,
                "pe_trailing": 44.3,
            },
            "projections": {
                "revenue_growth_mean": 0.40,
                "revenue_growth_std": 0.10,
                "margin_mean": 0.65,
                "margin_std": 0.05,
                "exit_multiple_mean": 35.0,
                "exit_multiple_std": 5.0,
            },
        },
        "filings": {
            "summary": "Latest 10-K cites supply constraints on Blackwell ramp and AI Diffusion export-control headwinds.",
            "risk_themes": [
                "supply concentration",
                "export-control regulation",
                "customer concentration in hyperscalers",
            ],
            "mdna_quotes": [
                {
                    "text": "We continue to face supply constraints on advanced packaging capacity.",
                    "accession": "0001045810-25-000023",
                    "item": "Item 7. MD&A",
                }
            ],
        },
        "news": {
            "summary": "Recent coverage mixed — strong Q4 + China H200 approval offset by valuation premium concerns.",
            "catalysts": [
                {
                    "title": "China approves H200 sales",
                    "sentiment": "bull",
                    "as_of": "2026-03-18",
                }
            ],
            "concerns": [
                {
                    "title": "Valuation premium signal flashes red",
                    "sentiment": "bear",
                    "as_of": "2026-03-29",
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_risk_real_run_on_nvda_ai_cake():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = _stub_state("NVDA", thesis)
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])

    # Sanity-check the output shape
    assert out.level in RISK_LEVEL_TO_SCORE
    assert out.score_0_to_10 == RISK_LEVEL_TO_SCORE[out.level]
    assert 3 <= len(out.top_risks) <= 7, f"unexpected top_risks count: {out.top_risks}"
    assert out.summary, "summary is empty"

    # Severities must be 1-5
    for r in out.top_risks:
        assert 1 <= r.severity <= 5, f"bad severity: {r.severity}"

    # Each top_risk must reference at least one source
    for r in out.top_risks:
        assert r.sources, f"top_risk missing sources: {r.title}"
        for s in r.sources:
            assert s in ("fundamentals", "filings", "news"), f"unknown source: {s}"

    # The FCF-yield threshold from our fixture (1.8% < 4%) should fire
    fcf_breach = next((b for b in out.threshold_breaches if b.signal == "fcf_yield"), None)
    assert fcf_breach is not None, (
        f"expected fcf_yield threshold breach to fire (observed 1.8%, threshold <4%); "
        f"got: {[b.signal for b in out.threshold_breaches]}"
    )
