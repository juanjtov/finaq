"""Bulk download SEC filings and ingest into ChromaDB.

Phase 0 default: AI cake universe (11 tickers). Phase 1 will extend to all 20.

Usage:
    python -m scripts.ingest_universe                       # AI cake universe
    python -m scripts.ingest_universe TICKER1 TICKER2 ...   # explicit list
"""

from __future__ import annotations

import asyncio
import sys

from data.chroma import ingest_filing
from data.edgar import download_filings
from utils import logger

AI_CAKE_UNIVERSE = (
    "NVDA",
    "AVGO",
    "TSM",
    "ASML",
    "MSFT",
    "GOOGL",
    "ORCL",
    "ANET",
    "VRT",
    "CEG",
    "PWR",
)


async def ingest_ticker(ticker: str) -> int:
    paths = await download_filings(ticker)
    if not paths:
        logger.warning(f"{ticker}: no filings on disk after download")
        return 0
    total = 0
    for path in paths:
        total += await asyncio.to_thread(ingest_filing, ticker, path)
    logger.info(f"{ticker}: ingested {total} chunks across {len(paths)} filings")
    return total


async def main(tickers: list[str]) -> None:
    grand_total = 0
    for ticker in tickers:
        try:
            grand_total += await ingest_ticker(ticker)
        except Exception as e:
            logger.error(f"{ticker}: ingest failed: {e}")
    logger.info(f"Done. Grand total: {grand_total} chunks across {len(tickers)} tickers.")


if __name__ == "__main__":
    args = sys.argv[1:] or list(AI_CAKE_UNIVERSE)
    asyncio.run(main(args))
