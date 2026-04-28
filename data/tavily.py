"""Tavily news-search wrapper.

Returns a JSON-friendly list of recent news articles for a (ticker, company)
pair, scoped to the last `days` days. Used by the News agent (Step 5c).

No caching — news is time-sensitive and we *want* every drill-in to surface
the latest catalysts. Cost is ~$0.001-0.005 per call depending on Tavily plan.
"""

from __future__ import annotations

import os
from typing import Any

from utils import logger, tenacity_retry

DEFAULT_DAYS = 90
DEFAULT_MAX_RESULTS = 15


# search news is a network-related function that's why I am using tenacity to retry in case it is needed
@tenacity_retry
def search_news(
    ticker: str,
    company_name: str | None = None,
    *,
    days: int = DEFAULT_DAYS,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Fetch recent news articles for a ticker.

    Returns a list of dicts with keys: `title`, `url`, `content`, `score`,
    `published_date` (when present). On persistent failure (after retries),
    returns an empty list and logs — the caller treats "no news" as a soft
    signal rather than crashing.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key or not api_key.startswith("tvly-"):
        logger.warning(f"[tavily] TAVILY_API_KEY not set; skipping news search for {ticker}")
        return []

    # Imported lazily so unit tests can run without the dep installed.
    from tavily import TavilyClient

    query = f"{ticker} {company_name}".strip() if company_name else ticker
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        topic="news",
        days=days,
        max_results=max_results,
        search_depth="advanced",
    )
    raw = response.get("results") or []

    out: list[dict[str, Any]] = []
    for r in raw:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score"),
                "published_date": r.get("published_date"),
            }
        )
    logger.info(f"[tavily] {ticker}: {len(out)} articles in last {days}d")
    return out
