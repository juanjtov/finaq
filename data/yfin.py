"""yfinance wrapper with 24h JSON cache and tenacity retries.

`get_financials` returns a JSON-serialisable dict with the five fields the
Fundamentals agent expects. On persistent failure the dict still returns with
an `errors` field populated, so the LangGraph never crashes mid-run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yfinance as yf

from utils import logger, tenacity_retry

CACHE_DIR = Path("data_cache/yfin")
CACHE_TTL_SECONDS = 24 * 60 * 60
EXPECTED_KEYS = ("price_history_5y", "income_stmt", "balance_sheet", "cash_flow", "info")


def _df_to_jsonable(df: Any) -> Any:
    if df is None:
        return {}
    try:
        if hasattr(df, "empty") and df.empty:
            return {}
        return json.loads(df.to_json(orient="index", date_format="iso"))
    except Exception:
        return {}


@tenacity_retry
def _fetch_from_yfinance(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    return {
        "price_history_5y": _df_to_jsonable(t.history(period="5y")),
        "income_stmt": _df_to_jsonable(t.income_stmt),
        "balance_sheet": _df_to_jsonable(t.balance_sheet),
        "cash_flow": _df_to_jsonable(t.cash_flow),
        "info": dict(t.info or {}),
    }


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.json"


def _is_cache_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def get_financials(ticker: str) -> dict:
    """Fetch (or load from cache) 5 years of NPV-relevant data for a ticker.

    Returns a dict with keys: price_history_5y, income_stmt, balance_sheet,
    cash_flow, info. On persistent failure the dict has an `errors` field.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(ticker)

    if _is_cache_fresh(cache):
        logger.info(f"yfinance cache hit: {ticker}")
        return json.loads(cache.read_text())

    logger.info(f"yfinance fetch: {ticker}")
    try:
        data = _fetch_from_yfinance(ticker)
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return {k: {} for k in EXPECTED_KEYS} | {"errors": [str(e)]}

    cache.write_text(json.dumps(data, indent=2, default=str))
    return data
