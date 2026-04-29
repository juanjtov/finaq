"""US 10-year Treasury yield fetcher with 24h cache.

Used by the Monte Carlo engine as the risk-free rate (Buffett-simplified
discount-rate baseline). Pulls from yfinance ticker `^TNX`, which reports the
yield in percentage points (e.g. 4.3 for 4.30%).

Cached for 24 hours — long-run rates don't move enough intraday to matter for
the valuation distribution. The cache lives at data_cache/treasury.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yfinance as yf

from utils import logger, tenacity_retry

CACHE_PATH = Path("data_cache/treasury.json")
CACHE_TTL_SECONDS = 24 * 60 * 60
TREASURY_TICKER = "^TNX"
DEFAULT_FALLBACK = 0.045  # used only on persistent failure — typical 2025-26 range


@tenacity_retry
def _fetch_yield() -> float:
    """Fetch the latest 10y Treasury yield as a decimal (e.g. 0.043 for 4.30%)."""
    t = yf.Ticker(TREASURY_TICKER)
    hist = t.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"yfinance returned empty history for {TREASURY_TICKER}")
    pct = float(hist["Close"].iloc[-1])
    if pct <= 0 or pct > 25:
        raise RuntimeError(f"implausible 10y yield from yfinance: {pct}")
    return pct / 100.0  # ^TNX is in percentage points


def _is_cache_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    return (time.time() - CACHE_PATH.stat().st_mtime) < CACHE_TTL_SECONDS


def get_10y_treasury_yield() -> float:
    """Return the current 10-year US Treasury yield as a decimal.

    Cached 24h. On persistent failure, returns DEFAULT_FALLBACK and logs.
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _is_cache_fresh():
        try:
            return float(json.loads(CACHE_PATH.read_text())["yield"])
        except Exception as e:
            logger.warning(f"[treasury] cache read failed: {e}; refetching")

    try:
        y = _fetch_yield()
    except Exception as e:
        logger.error(f"[treasury] fetch failed: {e}; falling back to {DEFAULT_FALLBACK}")
        return DEFAULT_FALLBACK

    CACHE_PATH.write_text(json.dumps({"yield": y, "fetched_at": time.time()}))
    logger.info(f"[treasury] 10y yield = {y:.4f} ({y * 100:.2f}%)")
    return y
