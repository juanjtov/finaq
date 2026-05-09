"""yfinance wrapper with 24h JSON cache and tenacity retries.

`get_financials` returns a JSON-serialisable dict with the five financial fields
plus `fetched_at` (UTC ISO timestamp of the actual yfinance call). Downstream
consumers use `fetched_at` to reason about data freshness.

On persistent failure the dict still returns with an `errors` field populated,
so the LangGraph never crashes mid-run.

Backtest mode (Step B1): pass `as_of=YYYY-MM-DD` to clip every output to data
dated on or before `as_of`. Price history fetches `start=as_of − 5y, end=as_of`;
financial-statement DataFrames are filtered to columns whose date ≤ as_of;
`info` is preserved as-is (yfinance's `.info` is a snapshot of current state
that has no historical analogue — backtest agents must NOT lean on its
forward-looking fields like `targetMeanPrice` when as_of is set).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf

from utils import logger, tenacity_retry

CACHE_DIR = Path("data_cache/yfin")
CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_FORMAT_VERSION = 3  # bump 2 → 3: added `fetched_at` field
EXPECTED_KEYS = ("price_history_5y", "income_stmt", "balance_sheet", "cash_flow", "info")

# Backtest cache: NEVER expires (historical data is immutable, the as_of date
# never moves), so we set TTL effectively infinite. Cache file lives at
# `data_cache/yfin/{TICKER}__as_of_{YYYYMMDD}.json` so it doesn't share TTL
# logic with the production cache and can't accidentally clobber it.
_BACKTEST_TTL_SECONDS = 365 * 24 * 60 * 60 * 5  # 5 years; effectively immortal


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


def _filter_df_by_as_of(df: Any, as_of: date, *, transpose: bool = False) -> Any:
    """Drop rows / columns whose timestamp is AFTER `as_of`.

    For price history (transpose=False), the date is on the index; we filter
    rows by `df.index.normalize() <= as_of`. For financial statements
    (transpose=True), dates are on the COLUMN axis and we filter columns.

    Date-only comparison: yfinance bars carry intraday timestamps; the user's
    intent for `as_of=2025-09-05` is "up to and including the close of
    Sep 5". `.normalize()` collapses the time-of-day so a 16:00 ET close
    bar dated Sep 5 is treated as Sep 5 (not Sep 5 16:00 → exclude).
    """
    if df is None:
        return df
    try:
        if hasattr(df, "empty") and df.empty:
            return df
        import numpy as np
        import pandas as pd

        as_of_ts = pd.Timestamp(as_of).normalize()

        if transpose:
            # Columns hold the dates.
            cols = pd.to_datetime(df.columns, errors="coerce")
            if hasattr(cols, "tz") and cols.tz is not None:
                cols = cols.tz_localize(None)
            # Normalize each column to its date; NaT → keep mask=False so
            # unparseable columns are dropped (cautious posture for backtest).
            cols_norm = cols.normalize()
            mask = np.where(pd.isna(cols_norm), False, cols_norm <= as_of_ts)
            return df.loc[:, mask]

        # else: rows hold the dates (price history shape).
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        idx_norm = idx.normalize()
        mask = np.where(pd.isna(idx_norm), False, idx_norm <= as_of_ts)
        return df.loc[mask]
    except Exception as e:
        from utils import logger
        logger.warning(f"[yfin] _filter_df_by_as_of fell back to original df: {e}")
        return df


def _parse_as_of(as_of: str | date | None) -> date | None:
    if as_of is None:
        return None
    if isinstance(as_of, date) and not isinstance(as_of, datetime):
        return as_of
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, str):
        # Accept "YYYY-MM-DD" — same shape we use everywhere else in the system.
        return datetime.fromisoformat(as_of).date()
    raise TypeError(f"as_of must be str, date, or None — got {type(as_of)!r}")


@tenacity_retry
def _fetch_from_yfinance(ticker: str, *, as_of: date | None = None) -> dict:
    t = yf.Ticker(ticker)
    if as_of is None:
        history = t.history(period="5y")
    else:
        # 5 years of trailing history ending AT as_of. yfinance accepts
        # `start`/`end` as ISO date strings; `end` is exclusive on yfinance's
        # side, so we add one day to make sure the as_of bar itself is
        # included in the result.
        start = (as_of - timedelta(days=365 * 5)).isoformat()
        end = (as_of + timedelta(days=1)).isoformat()
        history = t.history(start=start, end=end)
        # Defensive: drop any rows that landed past as_of (yfinance's `end`
        # behaviour is occasionally off by one in either direction).
        history = _filter_df_by_as_of(history, as_of)

    income_stmt = t.income_stmt
    balance_sheet = t.balance_sheet
    cash_flow = t.cash_flow
    if as_of is not None:
        income_stmt = _filter_df_by_as_of(income_stmt, as_of, transpose=True)
        balance_sheet = _filter_df_by_as_of(balance_sheet, as_of, transpose=True)
        cash_flow = _filter_df_by_as_of(cash_flow, as_of, transpose=True)

    return {
        "fetched_at": datetime.now(UTC).isoformat(),
        "as_of_date": as_of.isoformat() if as_of else None,
        "price_history_5y": _df_to_jsonable(history),
        "income_stmt": _df_to_jsonable(income_stmt, transpose=True),
        "balance_sheet": _df_to_jsonable(balance_sheet, transpose=True),
        "cash_flow": _df_to_jsonable(cash_flow, transpose=True),
        # `info` is yfinance's "current snapshot" (today's price, today's
        # analyst targets, etc.). It has no clean historical equivalent.
        # We retain it for backtest mode but agents MUST NOT use forward-
        # looking fields like targetMeanPrice / forwardEps when as_of is
        # set — the as-of context block in the prompt enforces that.
        "info": dict(t.info or {}),
    }


def _cache_path(ticker: str, *, as_of: date | None = None) -> Path:
    if as_of is None:
        return CACHE_DIR / f"{ticker}.json"
    # Backtest cache: keyed by as_of so different historical dates don't
    # share state, and so they don't pollute the production cache.
    return CACHE_DIR / f"{ticker}__as_of_{as_of.isoformat()}.json"


def _is_cache_fresh(path: Path, *, ttl_seconds: int = CACHE_TTL_SECONDS) -> bool:
    if not path.exists() or (time.time() - path.stat().st_mtime) >= ttl_seconds:
        return False
    # Reject stale-format caches even if they're young — guards against silent breakage
    # when the on-disk shape changes (e.g. when we transposed financial statements).
    try:
        version = json.loads(path.read_text()).get("_format_version")
        return version == CACHE_FORMAT_VERSION
    except Exception:
        return False


def get_financials(ticker: str, *, as_of: str | date | None = None) -> dict:
    """Fetch (or load from cache) 5 years of NPV-relevant data for a ticker.

    Returns a dict with keys: price_history_5y, income_stmt, balance_sheet,
    cash_flow, info. On persistent failure the dict has an `errors` field.

    Backtest mode: pass `as_of="YYYY-MM-DD"` (or a `date` object) and every
    DataFrame is filtered to ≤ as_of; price history fetches start=as_of − 5y,
    end=as_of. The cache file is keyed by as_of so the production cache is
    untouched.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    as_of_d = _parse_as_of(as_of)
    cache = _cache_path(ticker, as_of=as_of_d)
    ttl = _BACKTEST_TTL_SECONDS if as_of_d else CACHE_TTL_SECONDS

    if _is_cache_fresh(cache, ttl_seconds=ttl):
        logger.info(f"yfinance cache hit: {ticker}{f' (as_of={as_of_d})' if as_of_d else ''}")
        cached = json.loads(cache.read_text())
        cached.pop("_format_version", None)  # internal field, hide from consumers
        return cached

    logger.info(
        f"yfinance fetch: {ticker}"
        f"{f' (as_of={as_of_d})' if as_of_d else ''}"
    )
    try:
        data = _fetch_from_yfinance(ticker, as_of=as_of_d)
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return {k: {} for k in EXPECTED_KEYS} | {"errors": [str(e)]}

    on_disk = {**data, "_format_version": CACHE_FORMAT_VERSION}
    cache.write_text(json.dumps(on_disk, indent=2, default=str))
    return data
