"""Finnhub company-news wrapper — the News agent's and CIO planner's news source.

Replaced Tavily on 2026-09-08 after its monthly quota ran out (ARCHITECTURE
§3.11). Finnhub's free tier has no monthly cap (60 calls/min), takes explicit
`from`/`to` dates so backtest windows still work, and is ticker-native.

What Finnhub does *not* do is rank by relevance: `/company-news` returns every
article tagged with the symbol, newest first, capped at ~250 per call, and the
tag is loose (a quarter of NVDA's items actually mention NVIDIA — the rest are
market wraps that list it). So this module does the selection Tavily used to:

  1. fetch the window as log-spaced slices (last 3 days, 3-7, 7-14, 14-30,
     30-60, 60-90) — one call each, in parallel — so a mega-cap whose single
     page would cover one afternoon still yields news from across the quarter;
  2. keep only articles whose headline or summary names the ticker or company;
  3. drop duplicate headlines;
  4. share `max_results` across the slices (newest slices take the remainder),
     newest first, backfilling empty slots from the leftovers — a
     recency-tilted sample of the quarter;
  5. optionally fetch each selected article's body (`with_body=True`) — the
     drill-in reads excerpts, the CIO planner only reads headlines.

Backtest mode (`as_of="YYYY-MM-DD"`) caches by (ticker, as_of) under
`data_cache/news_backtest/` because historical windows are immutable. Finnhub's
free tier serves one year of history; older `as_of` dates come back empty.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from utils import logger, tenacity_retry

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
DEFAULT_DAYS = 90
DEFAULT_MAX_RESULTS = 15
PAGE_CAP = 200  # Finnhub returns ≤ ~250 items per call; a slice this full was truncated
SLICE_OFFSETS = (0, 3, 7, 14, 30, 60, 90)  # days back from `end`; consecutive pairs are slices
SLICE_WORKERS = 6
BODY_MAX_CHARS = 1500
BODY_TIMEOUT_S = 10.0
BODY_WORKERS = 6
BACKTEST_CACHE_DIR = Path("data_cache/news_backtest")
_NAME_STOPWORDS = frozenset(
    {
        "inc",
        "inc.",
        "corp",
        "corp.",
        "corporation",
        "co",
        "co.",
        "company",
        "ltd",
        "ltd.",
        "limited",
        "plc",
        "group",
        "holdings",
        "holding",
        "the",
        "and",
        "&",
        "sa",
        "nv",
        "ag",
        "se",
        "lp",
        "llc",
        "trust",
    }
)
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh) FINAQ/0.1"}


def _parse_as_of(as_of: str | date | None) -> date | None:
    if as_of is None:
        return None
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    if isinstance(as_of, str):
        return datetime.fromisoformat(as_of).date()
    raise TypeError(f"as_of must be str, date, or None — got {type(as_of)!r}")


def _backtest_cache_path(ticker: str, as_of: date) -> Path:
    return BACKTEST_CACHE_DIR / f"{ticker}__as_of_{as_of.isoformat()}.json"


@tenacity_retry
def _fetch_page(symbol: str, start: date, end: date, api_key: str) -> list[dict[str, Any]]:
    resp = httpx.get(
        FINNHUB_URL,
        params={
            "symbol": symbol,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": api_key,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, list) else []


def _slice_bounds(end: date, days: int) -> list[tuple[date, date]]:
    """Log-spaced (start, end) day pairs covering [end - days, end], newest first."""
    offsets = [o for o in SLICE_OFFSETS if o < days] + [days]
    return [
        (end - timedelta(days=b), end - timedelta(days=a))
        for a, b in zip(offsets, offsets[1:], strict=False)
    ]


def _fetch_window(symbol: str, end: date, days: int, api_key: str) -> list[dict[str, Any]]:
    """One call per slice, in parallel, de-duplicated by Finnhub id (adjacent
    slices share a boundary day). A full slice is logged as truncated: Finnhub
    returns its newest ~250 items, so the slice's older days are unseen."""
    bounds = _slice_bounds(end, days)
    with ThreadPoolExecutor(max_workers=SLICE_WORKERS) as pool:
        pages = list(pool.map(lambda b: _fetch_page(symbol, b[0], b[1], api_key), bounds))
    seen: set[Any] = set()
    out: list[dict[str, Any]] = []
    for (s_, e_), page in zip(bounds, pages, strict=True):
        if len(page) >= PAGE_CAP:
            logger.info(f"[finnhub] {symbol}: slice [{s_}, {e_}] truncated at {len(page)} items")
        for item in page:
            if item.get("id") in seen:
                continue
            seen.add(item.get("id"))
            out.append(item)
    return out


def _mention_pattern(ticker: str, company_name: str | None) -> re.Pattern[str]:
    """Match the ticker (case-sensitive) or the company's first distinctive
    name token (case-insensitive) on a word boundary. "NVIDIA Corporation" →
    NVIDIA; "Nu Holdings Ltd." → nothing usable, so the ticker alone."""
    parts = [re.escape(ticker)]
    for tok in (company_name or "").replace(",", " ").split():
        if tok.lower() not in _NAME_STOPWORDS and len(tok) >= 3:
            parts.append(f"(?i:{re.escape(tok)})")
            break
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])")


def _normalise(item: dict[str, Any]) -> dict[str, Any]:
    ts = item.get("datetime") or 0
    published = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None
    return {
        "title": item.get("headline", "") or "",
        "url": item.get("url", "") or "",
        "content": item.get("summary", "") or "",
        "source": item.get("source", "") or "",
        "score": None,
        "published_date": published,
    }


def _select(
    articles: list[dict[str, Any]], max_results: int, end: date, days: int
) -> list[dict[str, Any]]:
    """Share `max_results` across the slices — the newest slices take the
    remainder, so 15 over six slices is 3/3/3/2/2/2 — newest first within a
    slice; slots a thin slice can't fill go to the next-newest leftovers."""
    bounds = _slice_bounds(end, days)
    base, extra = divmod(max_results, len(bounds))
    quotas = [base + (1 if i < extra else 0) for i in range(len(bounds))]
    articles = sorted(articles, key=lambda a: a.get("published_date") or "", reverse=True)
    buckets: list[list[dict[str, Any]]] = [[] for _ in bounds]
    for art in articles:
        pub = date.fromisoformat((art.get("published_date") or "")[:10])
        for i, (s_, _) in enumerate(bounds):
            if pub >= s_:
                buckets[i].append(art)
                break
    taken: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for bucket, quota in zip(buckets, quotas, strict=True):
        taken.extend(bucket[:quota])
        leftovers.extend(bucket[quota:])
    if len(taken) < max_results:
        taken.extend(leftovers[: max_results - len(taken)])
    taken.sort(key=lambda a: a.get("published_date") or "", reverse=True)
    return taken[:max_results]


def _fetch_body(article: dict[str, Any]) -> dict[str, Any]:
    """Follow Finnhub's redirect to the publisher and pull the article
    paragraphs. Soft-fail: on any error the summary stands and the Finnhub
    link is kept. On success `url` becomes the publisher URL (what the
    report cites) and `content` gains the first BODY_MAX_CHARS of body."""
    from bs4 import BeautifulSoup

    try:
        resp = httpx.get(article["url"], follow_redirects=True, timeout=BODY_TIMEOUT_S, headers=_UA)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        body = " ".join(p for p in paras if len(p) > 60)
    except Exception as e:
        logger.debug(f"[finnhub] body fetch failed for {article.get('url')}: {e}")
        return article
    if len(body) > len(article["content"]):
        article["content"] = f"{article['content']} {body[:BODY_MAX_CHARS]}".strip()
    article["url"] = str(resp.url) or article["url"]
    return article


def search_news(
    ticker: str,
    company_name: str | None = None,
    *,
    days: int = DEFAULT_DAYS,
    max_results: int = DEFAULT_MAX_RESULTS,
    as_of: str | date | None = None,
    with_body: bool = True,
) -> list[dict[str, Any]]:
    """Recent news for a ticker, newest first.

    Returns dicts with `title`, `url`, `content`, `source`, `score` (always
    None — Finnhub has no relevance score), `published_date` (ISO-8601 UTC).
    Missing FINNHUB_API_KEY or a persistent API failure (after retries) yields
    an empty list — callers treat "no news" as a soft signal.

    `with_body=True` fetches the selected articles' bodies in parallel (the
    News agent reads excerpts); the CIO planner passes False since it only
    reads headlines.
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        logger.warning(f"[finnhub] FINNHUB_API_KEY not set; skipping news search for {ticker}")
        return []

    as_of_d = _parse_as_of(as_of)
    if as_of_d is not None:
        cache = _backtest_cache_path(ticker, as_of_d)
        if cache.exists():
            try:
                cached = json.loads(cache.read_text())
                logger.info(
                    f"[finnhub] {ticker}: cache hit for as_of={as_of_d} ({len(cached)} articles)"
                )
                return cached
            except Exception as e:
                logger.warning(f"[finnhub] backtest cache read failed: {e}; refetching")

    end = as_of_d or date.today()
    start = end - timedelta(days=days)
    raw = _fetch_window(ticker, end, days, api_key)
    pattern = _mention_pattern(ticker, company_name)
    start_iso, end_iso = start.isoformat(), end.isoformat()
    seen_titles: set[str] = set()
    relevant: list[dict[str, Any]] = []
    for item in raw:
        art = _normalise(item)
        pub = (art["published_date"] or "")[:10]
        if not pub or not (start_iso <= pub <= end_iso):
            continue
        if not pattern.search(f"{art['title']} {art['content']}"):
            continue
        key = re.sub(r"[^a-z0-9]", "", art["title"].lower())
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)
        relevant.append(art)

    out = _select(relevant, max_results, end, days)
    if with_body and out:
        with ThreadPoolExecutor(max_workers=BODY_WORKERS) as pool:
            out = list(pool.map(_fetch_body, out))
    logger.info(
        f"[finnhub] {ticker}: {len(raw)} tagged → {len(relevant)} mention {ticker} → "
        f"{len(out)} selected, window [{start_iso}, {end_iso}]"
    )

    if as_of_d is not None:
        BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _backtest_cache_path(ticker, as_of_d).write_text(json.dumps(out, indent=2))
    return out
