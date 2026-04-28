"""Tier 3 News evaluation — RAGAS framework metrics on the news synthesis.

Treats the News agent's `summary` as the answer, the Tavily-returned articles
as the contexts, and a meta-question as the user query. RAGAS computes:

  - faithfulness        : every claim in the summary grounded in the contexts
  - answer_relevancy    : does the summary address the meta-question?
  - context_precision   : are the top contexts truly relevant?

Heavy: pulls in langchain + datasets via ragas, costs ~$0.50–1.50 per run.
Gated `pytest -m eval`. Persists to data_cache/eval/runs/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.news import _company_name_for, run
from data.tavily import search_news
from utils.rag_eval import write_eval_run
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
        pytest.skip("MODEL_JUDGE not set in .env")


@pytest.mark.asyncio
async def test_ragas_evaluation_on_news_summary():
    """End-to-end RAGAS eval on the News agent's NVDA + AI cake output."""
    from utils.rag_ragas import evaluate_filings_run  # reuse — same shape

    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())

    # Re-fetch the same Tavily articles the agent used as RAGAS contexts.
    company = _company_name_for("NVDA")
    articles = search_news("NVDA", company, days=90, max_results=15)
    if not articles:
        pytest.skip("Tavily returned no NVDA news today")

    # Convert Tavily articles to the RAGAS context shape.
    contexts = [{"text": (a.get("content") or a.get("title") or "")[:1500]} for a in articles]

    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])

    if not out.summary or "STALE NEWS" in out.summary:
        pytest.skip("News agent produced no synthesis")

    meta_question = (
        f"What recent news catalysts and concerns affect NVDA's prospects "
        f"under the {thesis['name']} thesis?"
    )

    report = evaluate_filings_run(
        question=meta_question,
        retrieved_chunks=contexts,
        answer=out.summary,
        ground_truth=None,
    )

    write_eval_run(
        {
            "tier": 3,
            "suite": "ragas_news",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            **report.to_dict(),
            "raw": report.raw,
        }
    )

    if report.faithfulness is not None:
        assert report.faithfulness >= 0.6, f"RAGAS faithfulness too low: {report.faithfulness:.3f}"

    other_metrics = [
        m
        for m in (report.answer_relevancy, report.context_precision, report.context_recall)
        if m is not None
    ]
    if other_metrics:
        assert (
            max(other_metrics) >= 0.5
        ), f"all RAGAS non-faithfulness metrics below 0.5: {other_metrics}"
