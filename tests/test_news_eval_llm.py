"""Tier 2 News evaluation — LLM-as-judge for relevance + sentiment accuracy.

For each catalyst/concern returned by the News agent, the judge model
decides: (a) is this article actually thesis-relevant? (b) is the
sentiment label correct?

Costs roughly $0.005 per item × ~10 items per drill-in ≈ $0.05/run.
Gated `pytest -m eval`. Persists results to data_cache/eval/runs/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.news import run
from utils.rag_eval import judge_news_quality, serialise_news_judge_report, write_eval_run
from utils.schemas import NewsOutput

pytestmark = pytest.mark.eval

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    if not os.environ.get("TAVILY_API_KEY", "").startswith("tvly-"):
        pytest.skip("TAVILY_API_KEY not set")
    judge = os.environ.get("MODEL_JUDGE", "")
    if not judge or judge == "test-stub-model":
        pytest.skip("MODEL_JUDGE not set in .env (see .env.example)")


@pytest.mark.asyncio
async def test_news_llm_judge_relevance_and_sentiment_for_nvda():
    """Run the News agent on NVDA, then have the judge LLM verify each
    catalyst/concern's relevance and sentiment label."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])

    if "STALE NEWS" in out.summary:
        pytest.skip("No fresh NVDA news today")

    items = [c.model_dump() for c in (out.catalysts + out.concerns)]
    if not items:
        pytest.skip("News agent returned no items")

    report = judge_news_quality(thesis_name=thesis["name"], ticker="NVDA", items=items)
    write_eval_run(
        {
            "tier": 2,
            "suite": "news_llm_judge",
            **serialise_news_judge_report(report),
        }
    )

    # Bars (calibrated for stochastic LLM judgments):
    #  - Relevance ≥ 0.7: most items are genuinely thesis-relevant
    #  - Sentiment accuracy ≥ 0.7: most sentiment labels agree with the judge
    assert report.relevance_rate >= 0.7, f"News relevance below 70%: {report.relevance_rate:.2%}"
    assert (
        report.sentiment_accuracy >= 0.7
    ), f"News sentiment accuracy below 70%: {report.sentiment_accuracy:.2%}"
