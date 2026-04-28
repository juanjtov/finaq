"""yfinance wrapper with 24h JSON cache and tenacity retries.

`get_financials` returns a JSON-serialisable dict with the five financial fields
plus `fetched_at` (UTC ISO timestamp of the actual yfinance call). Downstream
consumers use `fetched_at` to reason about data freshness.

On persistent failure the dict still returns with an `errors` field populated,
so the LangGraph never crashes mid-run.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yfinance as yf

from utils import logger, tenacity_retry

CACHE_DIR = Path("data_cache/yfin")
CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_FORMAT_VERSION = 3  # bump 2 → 3: added `fetched_at` field
EXPECTED_KEYS = ("price_history_5y", "income_stmt", "balance_sheet", "cash_flow", "info")


def _df_to_jsonable(df: Any, *, transpose: bool = False) -> Any:
    """Serialize a yfinance DataFrame to JSON-friendly dict.

    yfinance's financial-statement DataFrames (income_stmt / balance_sheet /
    cash_flow) have *line items* as the row index and *dates* as columns —
    pass `transpose=True` for those so the resulting dict is keyed by date.
    Price history is already date-indexed; pass `transpose=False`.
    """
    if df is None:
        return {}
    try:
        if hasattr(df, "empty") and df.empty:
            return {}
        target = df.T if transpose else df
        return json.loads(target.to_json(orient="index", date_format="iso"))
    except Exception:
        return {}


@tenacity_retry
def _fetch_from_yfinance(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    return {
        "fetched_at": datetime.now(UTC).isoformat(),
        "price_history_5y": _df_to_jsonable(t.history(period="5y")),
        "income_stmt": _df_to_jsonable(t.income_stmt, transpose=True),
        "balance_sheet": _df_to_jsonable(t.balance_sheet, transpose=True),
        "cash_flow": _df_to_jsonable(t.cash_flow, transpose=True),
        "info": dict(t.info or {}),
    }


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.json"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists() or (time.time() - path.stat().st_mtime) >= CACHE_TTL_SECONDS:
        return False
    # Reject stale-format caches even if they're young — guards against silent breakage
    # when the on-disk shape changes (e.g. when we transposed financial statements).
    try:
        version = json.loads(path.read_text()).get("_format_version")
        return version == CACHE_FORMAT_VERSION
    except Exception:
        return False


def get_financials(ticker: str) -> dict:
    """Fetch (or load from cache) 5 years of NPV-relevant data for a ticker.

    Returns a dict with keys: price_history_5y, income_stmt, balance_sheet,
    cash_flow, info. On persistent failure the dict has an `errors` field.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(ticker)

    if _is_cache_fresh(cache):
        logger.info(f"yfinance cache hit: {ticker}")
        cached = json.loads(cache.read_text())
        cached.pop("_format_version", None)  # internal field, hide from consumers
        return cached

    logger.info(f"yfinance fetch: {ticker}")
    try:
        data = _fetch_from_yfinance(ticker)
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return {k: {} for k in EXPECTED_KEYS} | {"errors": [str(e)]}

    on_disk = {**data, "_format_version": CACHE_FORMAT_VERSION}
    cache.write_text(json.dumps(on_disk, indent=2, default=str))
    return data
