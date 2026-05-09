"""SEC EDGAR filing fetcher.

Wraps `sec-edgar-downloader` with an idempotent `download_filings` that always
fetches the 2 most recent annual reports and 4 most recent interim reports per
ticker by default — covering both domestic filers (10-K + 10-Q) and foreign
private issuers (20-F + 6-K). Also exposes `parse_filed_date` which extracts
the SEC-reported filing date from a submission's SGML header — used by
ChromaDB metadata for freshness.

For tickers that file only one of the two (e.g. NVDA has no 20-Fs, NU has
no 10-Ks), `dl.get(kind, ticker)` is a no-op when no filings exist of that
kind. Cost is one wasted API call per missing kind, which EDGAR handles fine.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from sec_edgar_downloader import Downloader

from utils import logger, tenacity_retry

EDGAR_DIR = Path("data_cache/edgar")
# Domestic filers (10-K + 10-Q) and foreign private issuers (20-F + 6-K) are
# both covered. 6-K filings are press releases / interim disclosures attached
# as exhibits — they don't follow Item-X structure so each lands as one
# "misc" chunk pile via the fallback in data/chroma.py:_split_into_items.
DEFAULT_LIMITS: dict[str, int] = {"10-K": 2, "10-Q": 4, "20-F": 2, "6-K": 4}

# SEC SGML headers contain a "FILED AS OF DATE:	YYYYMMDD" line within the first
# ~50 lines of full-submission.txt.
_FILED_DATE_RE = re.compile(r"FILED AS OF DATE:\s*(\d{8})")
_HEADER_SCAN_LINES = 60


def parse_filed_date(filing_path: Path) -> str | None:
    """Extract the SEC-reported filing date from an SGML submission.

    Returns an ISO date string (YYYY-MM-DD) or None if not parseable.
    """
    try:
        with open(filing_path, encoding="utf-8", errors="ignore") as f:
            head = "".join(line for _, line in zip(range(_HEADER_SCAN_LINES), f, strict=False))
    except OSError:
        return None
    m = _FILED_DATE_RE.search(head)
    if not m:
        return None
    raw = m.group(1)  # e.g. "20240221"
    if len(raw) != 8:
        return None
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def _parse_user_agent() -> tuple[str, str]:
    ua = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC_EDGAR_USER_AGENT is not set. See .env.example.")
    parts = ua.split(maxsplit=1)
    company = parts[0]
    email = parts[1] if len(parts) > 1 else "user@example.com"
    return company, email


def _filings_dir(ticker: str, kind: str) -> Path:
    # sec-edgar-downloader saves to {root}/sec-edgar-filings/{ticker}/{kind}/{accession}/
    return EDGAR_DIR / "sec-edgar-filings" / ticker / kind


def has_filings_in_unsupported_kinds(ticker: str) -> list[str]:
    """Return any filing-kind directories on disk for this ticker that AREN'T
    in our supported set.

    Used by the Filings agent / dashboard ingest banner to detect cases where
    EDGAR returned only filing kinds we don't ingest — historically that meant
    foreign-issuer 20-F/6-K, but those are now first-class. Today the function
    is mostly a future hook for kinds we may still skip (8-K, S-1, etc).

    Returns the unsupported kinds present, or `[]` if the ticker either has
    no EDGAR cache OR has only-supported kinds.
    """
    ticker_root = EDGAR_DIR / "sec-edgar-filings" / ticker.upper()
    if not ticker_root.exists():
        return []
    supported = set(DEFAULT_LIMITS.keys())
    return sorted(
        p.name
        for p in ticker_root.iterdir()
        if p.is_dir() and p.name not in supported and any(p.iterdir())
    )


def _existing_filings(ticker: str, kind: str, *, as_of: str | None = None) -> list[Path]:
    """Return on-disk filing paths for (ticker, kind), most-recent-first.

    Backtest mode: when `as_of="YYYY-MM-DD"` is set, drop any filing whose
    SGML-header `FILED AS OF DATE` is AFTER `as_of`. A filing the SEC accepted
    on 2025-11-12 wasn't visible to investors on 2025-09-05 — including it
    would leak future information into the backtest.

    Filings without parseable filed_dates are kept in production mode but
    DROPPED in backtest mode (better to err on the side of completeness for
    live runs and on the side of safety for backtest).
    """
    folder = _filings_dir(ticker, kind)
    if not folder.exists():
        return []
    paths = sorted(folder.glob("*/full-submission.txt"))
    if as_of is None:
        return paths

    cutoff = as_of  # ISO YYYY-MM-DD; lexicographic compare matches calendar order
    kept: list[Path] = []
    for p in paths:
        filed = parse_filed_date(p)
        if filed is None:
            # Conservative for backtest: skip filings we can't date-stamp.
            logger.debug(f"[edgar] skipping {p} in backtest mode — no parseable filed_date")
            continue
        if filed <= cutoff:
            kept.append(p)
        else:
            logger.debug(f"[edgar] excluding {p} (filed {filed} > as_of {cutoff})")
    return kept


@tenacity_retry
def _fetch_kind(ticker: str, kind: str, limit: int) -> None:
    company, email = _parse_user_agent()
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(company, email, str(EDGAR_DIR))
    dl.get(kind, ticker, limit=limit)


def _download_sync(
    ticker: str,
    limits: dict[str, int],
    *,
    as_of: str | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for kind, limit in limits.items():
        # Production: count any on-disk filings toward `limit`. Backtest:
        # only count filings dated ≤ as_of toward `limit` so we don't
        # short-circuit out early when post-as_of filings exist on disk.
        existing = _existing_filings(ticker, kind, as_of=as_of)
        if len(existing) >= limit:
            mode = f" (as_of={as_of})" if as_of else ""
            logger.info(
                f"{ticker} {kind}: {len(existing)} on disk{mode} (>= {limit}), "
                f"skipping fetch"
            )
            paths.extend(existing[:limit])
            continue
        if as_of is None:
            try:
                logger.info(f"{ticker} {kind}: fetching {limit} from EDGAR")
                _fetch_kind(ticker, kind, limit)
            except Exception as e:
                logger.error(f"{ticker} {kind}: fetch failed after retries: {e}")
            # Always re-scan disk: even on partial failure we want what landed.
            paths.extend(_existing_filings(ticker, kind)[:limit])
        else:
            # Backtest mode: never fetch fresh from EDGAR. Doing so would
            # download filings that exist TODAY which by definition include
            # post-as_of content. We use whatever pre-as_of filings already
            # landed on disk via earlier production runs — and warn loudly
            # if there aren't enough.
            if len(existing) < limit:
                logger.warning(
                    f"{ticker} {kind}: only {len(existing)} pre-as_of {as_of} filings "
                    f"on disk (wanted {limit}). Backtest will run with reduced "
                    f"corpus. Re-run a production drill BEFORE the as_of date "
                    f"if you need more historical coverage."
                )
            paths.extend(existing[:limit])
    return paths


async def download_filings(
    ticker: str,
    limits: dict[str, int] | None = None,
    *,
    as_of: str | None = None,
) -> list[Path]:
    """Download recent SEC filings. Idempotent: skips if already on disk.

    Returns a list of paths to `full-submission.txt` files actually present on disk.
    Errors are logged, not raised — caller gets whatever was successfully fetched.

    Backtest mode (`as_of="YYYY-MM-DD"`): never fetches fresh from EDGAR (any
    fetch today returns post-as_of filings). Returns only filings whose
    SGML-header `FILED AS OF DATE` ≤ as_of. Logs a warning when corpus is
    thinner than `limits` requests.
    """
    return await asyncio.to_thread(
        _download_sync, ticker, limits or DEFAULT_LIMITS, as_of=as_of,
    )
