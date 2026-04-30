"""Bulk download SEC filings and ingest into ChromaDB.

The Filings agent's RAG index is empty for any ticker not ingested here.
A drill-in on a missing ticker will return `errors=["no chunks retrieved"]`
and produce a thesis-incomplete report. Run this script after adding
tickers to a thesis JSON or a new thesis file altogether.

Usage:
    python -m scripts.ingest_universe                       # all theses' universes (union)
    python -m scripts.ingest_universe NVDA AVGO             # explicit ticker list
    python -m scripts.ingest_universe --thesis ai_cake      # one thesis's universe
    python -m scripts.ingest_universe --list                # print the union, don't ingest

Filings corpus (per data/edgar.py): 2 most recent 10-Ks + 4 most recent 10-Qs
per ticker. 8-K (current report) is intentionally NOT included today —
adding it is tracked in docs/POSTPONED.md §2.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from data.chroma import ingest_filing
from data.edgar import download_filings
from utils import logger

THESES_DIR = Path("theses")


# --- Thesis universe resolution -------------------------------------------


def _load_all_thesis_universes() -> dict[str, list[str]]:
    """Map of `slug → universe[]` for every thesis JSON in /theses/."""
    out: dict[str, list[str]] = {}
    for path in sorted(THESES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            out[path.stem] = data.get("universe") or []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[ingest] could not read {path}: {e}")
    return out


def _all_tickers() -> list[str]:
    """Deduplicated union of every thesis's universe."""
    seen: set[str] = set()
    ordered: list[str] = []
    for universe in _load_all_thesis_universes().values():
        for ticker in universe:
            t = ticker.upper()
            if t not in seen:
                seen.add(t)
                ordered.append(t)
    return ordered


def _thesis_tickers(slug: str) -> list[str]:
    universes = _load_all_thesis_universes()
    if slug not in universes:
        raise SystemExit(
            f"Thesis '{slug}' not found. Available: {sorted(universes.keys())}"
        )
    return [t.upper() for t in universes[slug]]


# --- Per-ticker ingest ----------------------------------------------------


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
    logger.info(
        f"Done. Grand total: {grand_total} chunks across {len(tickers)} tickers."
    )


# --- CLI -------------------------------------------------------------------


def _parse_args() -> tuple[list[str], bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Explicit ticker list. If empty, ingests union of all theses' universes.",
    )
    parser.add_argument(
        "--thesis",
        help="Limit to one thesis's universe (slug, e.g. ai_cake / nvda_halo / construction).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the resolved ticker list and exit without ingesting.",
    )
    args = parser.parse_args()

    if args.thesis:
        return _thesis_tickers(args.thesis), args.list
    if args.tickers:
        return [t.upper() for t in args.tickers], args.list
    return _all_tickers(), args.list


if __name__ == "__main__":
    tickers, list_only = _parse_args()
    if list_only:
        print(f"{len(tickers)} tickers:")
        for t in tickers:
            print(f"  {t}")
        sys.exit(0)
    asyncio.run(main(tickers))
