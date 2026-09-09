"""Bulk download SEC filings and ingest into the filings index.

The Filings agent's RAG index is empty for any ticker not ingested here.
A drill-in on a missing ticker will return `errors=["no chunks retrieved"]`
and produce a thesis-incomplete report. Run this script after adding
tickers to a thesis JSON or a new thesis file altogether.

Usage:
    python -m scripts.ingest_universe                       # all theses' universes (union)
    python -m scripts.ingest_universe NVDA AVGO             # explicit ticker list
    python -m scripts.ingest_universe --thesis ai_cake      # one thesis's universe
    python -m scripts.ingest_universe --list                # print the union, don't ingest
    python -m scripts.ingest_universe NU --force            # forget the manifest, embed again

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

from data import state as state_db
from data.edgar import download_filings
from data.vectors import ingest_filing
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
        raise SystemExit(f"Thesis '{slug}' not found. Available: {sorted(universes.keys())}")
    return [t.upper() for t in universes[slug]]


# --- Per-ticker ingest ----------------------------------------------------


async def ingest_ticker(ticker: str, *, force_refresh: bool = False, force: bool = False) -> int:
    """Download + chunk + embed every recent filing for `ticker`. Returns
    total chunk count across all filings.

    `force_refresh=True` re-checks EDGAR even when on-disk count already
    satisfies `DEFAULT_LIMITS` — used by the drill-time freshness gate so
    a "ticker has the right number of filings but they're all old" case
    actually pulls the new accession.

    `force=True` clears the ticker's `ingested_filings` manifest rows first so
    every filing is embedded again — the recovery path after the Pinecone
    index was wiped or renamed (the manifest alone decides "already ingested").
    """
    if force:
        dropped = state_db.clear_ingested_filings(ticker)
        logger.info(f"{ticker}: --force cleared {dropped} manifest rows")
    paths = await download_filings(ticker, force_refresh=force_refresh)
    if not paths:
        logger.warning(f"{ticker}: no filings on disk after download")
        return 0
    total = 0
    for path in paths:
        total += await asyncio.to_thread(ingest_filing, ticker, path)
    logger.info(f"{ticker}: ingested {total} chunks across {len(paths)} filings")
    return total


async def main(tickers: list[str], *, force: bool = False) -> None:
    grand_total = 0
    for ticker in tickers:
        try:
            grand_total += await ingest_ticker(ticker, force=force)
        except Exception as e:
            logger.error(f"{ticker}: ingest failed: {e}")
    logger.info(f"Done. Grand total: {grand_total} chunks across {len(tickers)} tickers.")


# --- CLI -------------------------------------------------------------------


def _parse_args() -> tuple[list[str], bool, bool]:
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear each ticker's ingest manifest first so every filing is embedded again.",
    )
    args = parser.parse_args()

    if args.thesis:
        return _thesis_tickers(args.thesis), args.list, args.force
    if args.tickers:
        return [t.upper() for t in args.tickers], args.list, args.force
    return _all_tickers(), args.list, args.force


if __name__ == "__main__":
    tickers, list_only, force = _parse_args()
    if list_only:
        print(f"{len(tickers)} tickers:")
        for t in tickers:
            print(f"  {t}")
        sys.exit(0)
    asyncio.run(main(tickers, force=force))
