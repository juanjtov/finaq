"""Step 5c news-quality eval suite — Tier 1 (deterministic, always-on).

Three test categories:
  - Golden-set recall (news agent surfaces expected events for known tickers)
  - Faithfulness (catalyst URLs trace back to Tavily; summaries supported by article text)
  - CEO-resignation regression: ensures the prompt-update kept event-attributed
    price moves rather than skipping them as "stock-price-only"

Tier 2 (LLM-as-judge) is in test_news_eval_llm.py — gated `pytest -m eval`.
Tier 3 (RAGAS) is in test_news_eval_ragas.py — also `pytest -m eval`.

Eval results persist to data_cache/eval/runs/ for the Mission Control panel.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from agents.news import NEWS_DAYS, run
from tests.eval.news_golden_queries import (
    NEWS_GOLDEN_QUERIES,
    NewsGoldenQuery,
    items_match_golden,
)
from utils.rag_eval import check_news_faithfulness, write_eval_run
from utils.schemas import NewsOutput

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    if not os.environ.get("TAVILY_API_KEY", "").startswith("tvly-"):
        pytest.skip("TAVILY_API_KEY not set")


# --- Golden-set recall ------------------------------------------------------


@pytest.mark.parametrize("gq", NEWS_GOLDEN_QUERIES, ids=lambda q: q.expected_keywords[0])
@pytest.mark.asyncio
async def test_news_golden_recall_for_query(gq: NewsGoldenQuery):
    """For each golden query, the News agent should surface at least one
    catalyst or concern containing one of the expected keywords."""
    thesis = json.loads((THESES_DIR / f"{gq.thesis_slug}.json").read_text())
    state = {"ticker": gq.ticker, "thesis": thesis}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])

    # Skip cleanly if Tavily had no fresh news today
    if "STALE NEWS" in out.summary:
        pytest.skip(f"No fresh {gq.ticker} news today; cannot evaluate")

    items = [{**c.model_dump()} for c in (out.catalysts + out.concerns)]
    passed, matched = items_match_golden(items, gq)
    assert passed, (
        f"News golden query failed: expected one of {gq.expected_keywords} "
        f"in catalysts+concerns. ({gq.description})"
    )


@pytest.mark.asyncio
async def test_news_golden_set_summary_recorded():
    """Run all golden queries and persist a summary so Mission Control can
    render the latest News quality score."""
    per_query: list[dict] = []
    pass_count = 0
    skip_count = 0

    for gq in NEWS_GOLDEN_QUERIES:
        thesis = json.loads((THESES_DIR / f"{gq.thesis_slug}.json").read_text())
        state = {"ticker": gq.ticker, "thesis": thesis}
        result = await run(state)
        out = NewsOutput.model_validate(result["news"])

        if "STALE NEWS" in out.summary:
            skip_count += 1
            per_query.append(
                {
                    "ticker": gq.ticker,
                    "thesis_slug": gq.thesis_slug,
                    "expected_keywords": list(gq.expected_keywords),
                    "passed": None,
                    "matched": None,
                    "skipped": "no fresh news",
                    "description": gq.description,
                }
            )
            continue

        items = [c.model_dump() for c in (out.catalysts + out.concerns)]
        passed, matched = items_match_golden(items, gq)
        if passed:
            pass_count += 1
        per_query.append(
            {
                "ticker": gq.ticker,
                "thesis_slug": gq.thesis_slug,
                "expected_keywords": list(gq.expected_keywords),
                "passed": passed,
                "matched_keyword": matched,
                "items_seen": len(items),
                "description": gq.description,
            }
        )

    evaluated = len(NEWS_GOLDEN_QUERIES) - skip_count
    recall = (pass_count / evaluated) if evaluated else None
    write_eval_run(
        {
            "tier": 1,
            "suite": "news_golden_recall",
            "queries": len(NEWS_GOLDEN_QUERIES),
            "evaluated": evaluated,
            "skipped": skip_count,
            "passed": pass_count,
            "recall": recall,
            "per_query": per_query,
        }
    )

    if recall is not None:
        # Soft bar — news availability fluctuates day-to-day, so we accept
        # 60% recall. Catastrophic regressions (recall < 0.6) signal real
        # prompt or retrieval problems.
        assert (
            recall >= 0.6
        ), f"news golden-set recall below 60%: {recall:.2%} ({pass_count}/{evaluated})"


# --- Faithfulness: catalyst URLs and summaries trace back to Tavily ---------


@pytest.mark.asyncio
async def test_news_items_url_grounded_in_tavily_results():
    """Every catalyst/concern URL must be in the Tavily-returned article list.
    A fabricated URL is a hard failure (the LLM hallucinated a citation)."""
    from agents.news import _company_name_for
    from data.tavily import search_news

    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    company = _company_name_for("NVDA")
    articles = search_news("NVDA", company, days=NEWS_DAYS, max_results=15)
    if not articles:
        pytest.skip("No fresh NVDA news today")

    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])
    items = [c.model_dump() for c in (out.catalysts + out.concerns)]
    if not items:
        pytest.skip("News agent returned no items today")

    faith = check_news_faithfulness(items, articles)
    write_eval_run(
        {
            "tier": 1,
            "suite": "news_faithfulness",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            **asdict(faith),
        }
    )

    assert faith.url_grounding_rate == 1.0, (
        f"fabricated URLs ({faith.items_grounded_url}/{faith.items_total}): "
        f"{faith.fabricated_urls}"
    )


# --- CEO-resignation regression: prompt update test -------------------------


@pytest.mark.asyncio
async def test_event_attributed_price_move_is_kept(monkeypatch):
    """The original prompt risked skipping 'stock plunged 20% on CEO resignation'
    as a 'stock-price-only' story. Updated prompt must KEEP event-attributed
    price moves. We feed a fixture article and assert the LLM surfaces it."""
    from agents import news as news_mod

    # Mock Tavily to return ONLY a CEO-resignation story so the agent's
    # output must include it (or be empty and fail the assertion).
    fake_articles = [
        {
            "title": "NVIDIA stock plunges 20% after CEO Jensen Huang resigns",
            "url": "https://example.com/nvda-ceo-resigns",
            "content": (
                "Shares of NVIDIA fell more than 20% in after-hours trading after the "
                "company announced that founder and CEO Jensen Huang will step down "
                "effective immediately. The board has named Colette Kress as interim CEO "
                "while it searches for a permanent successor. Analysts called the move "
                "the most consequential leadership change in NVIDIA's history."
            ),
            "score": 0.99,
            "published_date": "2026-04-25",
        },
        # An adjacent low-quality article that SHOULD be skipped so we can
        # also verify the skip rule still works for genuine stock-price-only
        # stories.
        {
            "title": "NVDA up 2.3% in midday trading",
            "url": "https://example.com/nvda-midday",
            "content": "NVDA rose 2.3% in midday trading on no specific news. Volume was average.",
            "score": 0.45,
            "published_date": "2026-04-26",
        },
    ]

    monkeypatch.setattr(news_mod, "search_news", lambda *a, **kw: fake_articles)
    monkeypatch.setattr(news_mod, "_company_name_for", lambda t: "NVIDIA Corporation")

    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])

    all_items = out.catalysts + out.concerns
    # The CEO-resignation article MUST be picked up.
    has_ceo_event = any(
        ("Huang" in it.title or "CEO" in it.title.upper() or "resign" in it.title.lower())
        and it.url == "https://example.com/nvda-ceo-resigns"
        for it in all_items
    )
    assert has_ceo_event, (
        "News agent failed to surface the event-attributed CEO-resignation article. "
        f"Got: {[(it.title, it.sentiment) for it in all_items]}"
    )

    # The bare price-move article SHOULD be skipped.
    has_bare_price_move = any(it.url == "https://example.com/nvda-midday" for it in all_items)
    assert (
        not has_bare_price_move
    ), "News agent kept a stock-price-only story with no underlying event"
