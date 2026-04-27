"""SEC EDGAR filing fetcher.

Wraps `sec-edgar-downloader` with an idempotent `download_filings` that always
fetches the 2 most recent 10-Ks and 4 most recent 10-Qs per ticker by default.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from sec_edgar_downloader import Downloader

from utils import logger, tenacity_retry

EDGAR_DIR = Path("data_cache/edgar")
DEFAULT_LIMITS: dict[str, int] = {"10-K": 2, "10-Q": 4}


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


def _existing_filings(ticker: str, kind: str) -> list[Path]:
    folder = _filings_dir(ticker, kind)
    if not folder.exists():
        return []
    return sorted(folder.glob("*/full-submission.txt"))


@tenacity_retry
def _fetch_kind(ticker: str, kind: str, limit: int) -> None:
    company, email = _parse_user_agent()
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(company, email, str(EDGAR_DIR))
    dl.get(kind, ticker, limit=limit)


def _download_sync(ticker: str, limits: dict[str, int]) -> list[Path]:
    paths: list[Path] = []
    for kind, limit in limits.items():
        existing = _existing_filings(ticker, kind)
        if len(existing) >= limit:
            logger.info(f"{ticker} {kind}: {len(existing)} on disk (>= {limit}), skipping fetch")
            paths.extend(existing[:limit])
            continue
        try:
            logger.info(f"{ticker} {kind}: fetching {limit} from EDGAR")
            _fetch_kind(ticker, kind, limit)
        except Exception as e:
            logger.error(f"{ticker} {kind}: fetch failed after retries: {e}")
        # Always re-scan disk: even on partial failure we want what landed.
        paths.extend(_existing_filings(ticker, kind)[:limit])
    return paths


async def download_filings(ticker: str, limits: dict[str, int] | None = None) -> list[Path]:
    """Download recent SEC filings. Idempotent: skips if already on disk.

    Returns a list of paths to `full-submission.txt` files actually present on disk.
    Errors are logged, not raised — caller gets whatever was successfully fetched.
    """
    return await asyncio.to_thread(_download_sync, ticker, limits or DEFAULT_LIMITS)
