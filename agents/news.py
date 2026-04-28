"""News agent — LLM-driven thesis-aware news triage via Tavily.

Pipeline per drill-in:
  1. Pull last-90-day news from Tavily for "{ticker} {company_name}".
  2. Send the article list + thesis to the LLM.
  3. LLM extracts 3-7 catalysts (bull/neutral) and 3-7 concerns (bear/neutral),
     each tagged with sentiment, URL, and `as_of` (published_date).
  4. Validate against NewsOutput, surface partial failures via `errors`.

The exact model is configured by the MODEL_NEWS env var (see utils/models.py).

Run standalone:  python -m agents.news NVDA [thesis_slug]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

from data.tavily import search_news
from data.yfin import get_financials
from utils import logger
from utils.models import MODEL_NEWS
from utils.openrouter import get_client
from utils.schemas import NewsOutput
from utils.state import FinaqState

NODE = "news"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "news.md").read_text()
LLM_MAX_TOKENS = 2500
NEWS_DAYS = 90
NEWS_MAX_RESULTS = 15


# --- Prompt assembly ---------------------------------------------------------


def _format_article(idx: int, article: dict) -> str:
    title = article.get("title", "")
    url = article.get("url", "")
    pub = article.get("published_date") or "unknown"
    score = article.get("score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
    content = (article.get("content") or "")[:600]
    return (
        f"\n[article {idx}] published_date={pub} tavily_score={score_str}\n"
        f"  title: {title}\n"
        f"  url:   {url}\n"
        f"  excerpt: {content}"
    )


def _build_user_prompt(ticker: str, company_name: str, thesis: dict, articles: list[dict]) -> str:
    today_iso = date.today().isoformat()
    parts = [
        f"AS OF: {today_iso}",
        f"TICKER: {ticker}",
        f"COMPANY: {company_name or '(unknown)'}",
        f"ACTIVE THESIS: {thesis.get('name', 'unknown')}",
        f"THESIS SUMMARY: {thesis.get('summary', '')}",
        f"ANCHOR TICKERS: {', '.join(thesis.get('anchor_tickers', []))}",
        "",
        "MATERIAL THRESHOLDS THE TRIAGE SYSTEM IS WATCHING:",
        json.dumps(thesis.get("material_thresholds", []), indent=2),
        "",
        f"RECENT NEWS ARTICLES (last {NEWS_DAYS} days, top {len(articles)} by Tavily score):",
    ]
    if not articles:
        parts.append("(no articles retrieved)")
    else:
        for i, art in enumerate(articles, start=1):
            parts.append(_format_article(i, art))
    parts.append("\nPRODUCE YOUR ANALYSIS NOW. STRICT JSON ONLY.")
    return "\n".join(parts)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text


def _call_llm(ticker: str, company_name: str, thesis: dict, articles: list[dict]) -> dict:
    client = get_client()
    user = _build_user_prompt(ticker, company_name, thesis, articles)
    resp = client.chat.completions.create(
        model=MODEL_NEWS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return json.loads(_strip_code_fences(raw))


# --- Helpers -----------------------------------------------------------------


def _company_name_for(ticker: str) -> str:
    """Pull a friendly company name from the yfinance cache. Falls back to the
    ticker itself if `info.longName` isn't available."""
    try:
        financials = get_financials(ticker)
        info = financials.get("info") or {}
        return info.get("longName") or info.get("shortName") or ticker
    except Exception as e:
        logger.warning(f"[news] couldn't resolve company name for {ticker}: {e}")
        return ticker


# --- Graph node --------------------------------------------------------------


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    ticker = state.get("ticker", "")
    thesis = state.get("thesis") or {}
    errors: list[str] = []

    # Step 1 — resolve company name (cached yfinance call) + Tavily search
    company_name = await asyncio.to_thread(_company_name_for, ticker)
    try:
        articles = await asyncio.to_thread(
            search_news, ticker, company_name, days=NEWS_DAYS, max_results=NEWS_MAX_RESULTS
        )
    except Exception as e:
        logger.error(f"[news] Tavily search failed for {ticker}: {e}")
        errors.append(f"tavily: {e}")
        articles = []

    # Step 2 — LLM extraction
    if not articles:
        out = NewsOutput(
            summary=f"[STALE NEWS — no fresh coverage] No news articles retrieved for {ticker}.",
            catalysts=[],
            concerns=[],
            evidence=[],
            errors=errors + ["no articles retrieved"],
        )
    else:
        try:
            llm_out = await asyncio.to_thread(_call_llm, ticker, company_name, thesis, articles)
            out = NewsOutput.model_validate(llm_out)
            out.errors = errors
        except Exception as e:
            logger.error(f"[news] LLM extraction failed for {ticker}: {e}")
            errors.append(f"llm: {e}")
            out = NewsOutput(
                summary=f"LLM extraction failed for {ticker}; news search succeeded.",
                catalysts=[],
                concerns=[],
                evidence=[],
                errors=errors,
            )

    return {
        "news": out.model_dump(),
        "messages": [
            {
                "node": NODE,
                "event": "completed",
                "started_at": started_at,
                "completed_at": time.perf_counter(),
            }
        ],
    }


# --- CLI ---------------------------------------------------------------------


async def _cli(ticker: str, thesis_slug: str = "ai_cake") -> None:
    thesis = json.loads(Path(f"theses/{thesis_slug}.json").read_text())
    state: FinaqState = {"ticker": ticker, "thesis": thesis}
    result = await run(state)
    print(json.dumps(result["news"], indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agents.news TICKER [thesis_slug]", file=sys.stderr)
        sys.exit(1)
    ticker = sys.argv[1].upper()
    thesis_slug = sys.argv[2] if len(sys.argv) > 2 else "ai_cake"
    asyncio.run(_cli(ticker, thesis_slug))
