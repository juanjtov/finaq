"""Tavily news-search wrapper.

Returns a JSON-friendly list of recent news articles for a (ticker, company)
pair, scoped to the last `days` days. Used by the News agent (Step 5c).

No caching for production — news is time-sensitive and we *want* every drill-in
to surface the latest catalysts. Backtest mode (Step B2) caches by
(ticker, as_of) tuple under `data_cache/tavily_backtest/` because historical
windows are immutable, the same backtest ticker should never re-pay Tavily
credits, and reproducibility matters for the demo.

Cost is ~$0.001-0.005 per call depending on Tavily plan.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from utils import logger, tenacity_retry

DEFAULT_DAYS = 90
DEFAULT_MAX_RESULTS = 15
BACKTEST_CACHE_DIR = Path("data_cache/tavily_backtest")


def _parse_as_of(as_of: str | date | None) -> date | None:
    if as_of is None:
        return None
    if isinstance(as_of, date) and not isinstance(as_of, datetime):
        return as_of
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, str):
        return datetime.fromisoformat(as_of).date()
    raise TypeError(f"as_of must be str, date, or None — got {type(as_of)!r}")


def _backtest_cache_path(ticker: str, as_of: date) -> Path:
    return BACKTEST_CACHE_DIR / f"{ticker}__as_of_{as_of.isoformat()}.json"


def _filter_results_by_published_date(
    results: list[dict[str, Any]],
    *,
    start_iso: str,
    end_iso: str,
) -> list[dict[str, Any]]:
    """Drop articles whose `published_date` is outside [start_iso, end_iso].

    Tavily's `start_date`/`end_date` are honored server-side, but defence in
    depth is cheap and protects against API behaviour drift. Articles
    without a `published_date` are dropped in backtest mode (we can't prove
    they're inside the window — same conservative posture as the EDGAR
    no-date-stamp filter).
    """
    out: list[dict[str, Any]] = []
    for r in results:
        pub = r.get("published_date") or ""
        if not pub:
            continue
        # Tavily returns ISO-8601 timestamps with a 'T' (e.g. "2025-09-04T14:23:00Z").
        # Lexicographic compare against YYYY-MM-DD start/end works because both
        # are date-prefixed in ISO-8601.
        pub_date = pub[:10]
        if start_iso <= pub_date <= end_iso:
            out.append(r)
    return out


# search news is a network-related function that's why I am using tenacity to retry in case it is needed
@tenacity_retry
def search_news(
    ticker: str,
    company_name: str | None = None,
    *,
    days: int = DEFAULT_DAYS,
    max_results: int = DEFAULT_MAX_RESULTS,
    as_of: str | date | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent news articles for a ticker.

    Returns a list of dicts with keys: `title`, `url`, `content`, `score`,
    `published_date` (when present). On persistent failure (after retries),
    returns an empty list and logs — the caller treats "no news" as a soft
    signal rather than crashing.

    Backtest mode (`as_of="YYYY-MM-DD"`): pulls articles published in
    `[as_of − days, as_of]` via Tavily's `start_date`/`end_date` params (N2
    strategy). Defence-in-depth filter rejects any article whose
    `published_date` falls outside the window. Cached by (ticker, as_of) at
    `data_cache/tavily_backtest/{TICKER}__as_of_{as_of}.json` for repeatable
    demos. If Tavily returns an empty result set the cache is still written
    (so subsequent calls don't re-pay credits to learn the same answer);
    callers should treat `[]` as a valid result.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key or not api_key.startswith("tvly-"):
        logger.warning(f"[tavily] TAVILY_API_KEY not set; skipping news search for {ticker}")
        return []

    as_of_d = _parse_as_of(as_of)

    # Backtest cache hit-path.
    if as_of_d is not None:
        cache = _backtest_cache_path(ticker, as_of_d)
        if cache.exists():
            try:
                cached = json.loads(cache.read_text())
                logger.info(
                    f"[tavily] {ticker}: cache hit for as_of={as_of_d} "
                    f"({len(cached)} articles)"
                )
                return cached
            except Exception as e:
                logger.warning(f"[tavily] backtest cache read failed: {e}; refetching")

    # Imported lazily so unit tests can run without the dep installed.
    from tavily import TavilyClient

    query = f"{ticker} {company_name}".strip() if company_name else ticker
    client = TavilyClient(api_key=api_key)
    if as_of_d is not None:
        start_d = as_of_d - timedelta(days=days)
        response = client.search(
            query=query,
            topic="news",
            start_date=start_d.isoformat(),
            end_date=as_of_d.isoformat(),
            max_results=max_results,
            search_depth="advanced",
        )
    else:
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

    if as_of_d is not None:
        # Defence in depth: filter by published_date even though Tavily
        # already honored start_date/end_date.
        before = len(out)
        out = _filter_results_by_published_date(
            out,
            start_iso=(as_of_d - timedelta(days=days)).isoformat(),
            end_iso=as_of_d.isoformat(),
        )
        if len(out) < before:
            logger.warning(
                f"[tavily] backtest filter dropped {before - len(out)}/{before} "
                f"articles outside window ending {as_of_d}"
            )
        # Persist (even when empty) so the demo is reproducible without
        # re-paying credits.
        BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _backtest_cache_path(ticker, as_of_d).write_text(json.dumps(out, indent=2))
        logger.info(
            f"[tavily] {ticker}: {len(out)} articles in window "
            f"[{(as_of_d - timedelta(days=days)).isoformat()}, {as_of_d.isoformat()}] "
            f"(cached for backtest)"
        )
    else:
        logger.info(f"[tavily] {ticker}: {len(out)} articles in last {days}d")

    return out
