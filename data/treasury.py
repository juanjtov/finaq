"""US 10-year Treasury yield fetcher with 24h cache.

Used by the Monte Carlo engine as the risk-free rate (Buffett-simplified
discount-rate baseline). Pulls from yfinance ticker `^TNX`, which reports the
yield in percentage points (e.g. 4.3 for 4.30%).

Cached for 24 hours — long-run rates don't move enough intraday to matter for
the valuation distribution. The cache lives at data_cache/treasury.json.

Backtest mode (Step B1): pass `as_of="YYYY-MM-DD"` to get the closing 10y
yield from the trading day at or before as_of. Backtest cache is keyed by
as_of date so different historical scenarios don't share state.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import yfinance as yf

from utils import logger, tenacity_retry

CACHE_PATH = Path("data_cache/treasury.json")
CACHE_DIR = Path("data_cache")
CACHE_TTL_SECONDS = 24 * 60 * 60
TREASURY_TICKER = "^TNX"
DEFAULT_FALLBACK = 0.045  # used only on persistent failure — typical 2025-26 range


@tenacity_retry
def _fetch_yield(as_of: date | None = None) -> float:
    """Fetch the 10y Treasury yield as a decimal (e.g. 0.043 for 4.30%).

    `as_of=None` → latest available close.
    `as_of=YYYY-MM-DD` → close of the most recent trading day at or before as_of.
    """
    t = yf.Ticker(TREASURY_TICKER)
    if as_of is None:
        hist = t.history(period="5d")
    else:
        # Pull a 10-day window ending at as_of so we hit at least one trading
        # day even if as_of itself was a weekend / holiday.
        start = (as_of - timedelta(days=10)).isoformat()
        end = (as_of + timedelta(days=1)).isoformat()
        hist = t.history(start=start, end=end)
        # Drop any rows past as_of (yfinance end is sometimes inclusive).
        if not hist.empty:
            try:
                import pandas as pd
                idx = hist.index
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx = idx.tz_localize(None)
                mask = idx <= pd.Timestamp(as_of)
                hist = hist.loc[mask]
            except Exception:
                pass
    if hist.empty:
        raise RuntimeError(
            f"yfinance returned empty history for {TREASURY_TICKER}"
            f"{f' at as_of={as_of}' if as_of else ''}"
        )
    pct = float(hist["Close"].iloc[-1])
    if pct <= 0 or pct > 25:
        raise RuntimeError(f"implausible 10y yield from yfinance: {pct}")
    return pct / 100.0  # ^TNX is in percentage points


def _backtest_cache_path(as_of: date) -> Path:
    return CACHE_DIR / f"treasury__as_of_{as_of.isoformat()}.json"


def _is_cache_fresh(path: Path = CACHE_PATH, *, ttl_seconds: int = CACHE_TTL_SECONDS) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl_seconds


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


def get_10y_treasury_yield(*, as_of: str | date | None = None) -> float:
    """Return the 10-year US Treasury yield as a decimal.

    Production: latest close, cached 24h at `data_cache/treasury.json`.
    Backtest: close at or before `as_of`, cached forever at
    `data_cache/treasury__as_of_{as_of}.json` (historical data is immutable).

    On persistent failure, returns DEFAULT_FALLBACK and logs.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    as_of_d = _parse_as_of(as_of)

    if as_of_d is None:
        cache = CACHE_PATH
        ttl = CACHE_TTL_SECONDS
    else:
        cache = _backtest_cache_path(as_of_d)
        # Historical close never changes → effectively immortal cache.
        ttl = 365 * 24 * 60 * 60 * 5

    if _is_cache_fresh(cache, ttl_seconds=ttl):
        try:
            return float(json.loads(cache.read_text())["yield"])
        except Exception as e:
            logger.warning(f"[treasury] cache read failed: {e}; refetching")

    try:
        y = _fetch_yield(as_of_d)
    except Exception as e:
        logger.error(
            f"[treasury] fetch failed{f' (as_of={as_of_d})' if as_of_d else ''}: "
            f"{e}; falling back to {DEFAULT_FALLBACK}"
        )
        return DEFAULT_FALLBACK

    cache.write_text(
        json.dumps({"yield": y, "fetched_at": time.time(),
                    "as_of": as_of_d.isoformat() if as_of_d else None})
    )
    logger.info(
        f"[treasury] 10y yield"
        f"{f' (as_of={as_of_d})' if as_of_d else ''} = {y:.4f} ({y * 100:.2f}%)"
    )
    return y
