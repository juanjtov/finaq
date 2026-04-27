"""Step 2 integration tests — real EDGAR, yfinance, OpenRouter, ChromaDB.

Run via:  pytest -m integration tests/test_data_layer_integration.py

These tests download real SEC filings (idempotent), hit yfinance, and write to
the ChromaDB collection on disk under data_cache/chroma. Re-running is safe.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    if not os.environ.get("SEC_EDGAR_USER_AGENT"):
        pytest.skip("SEC_EDGAR_USER_AGENT not set")


@pytest.mark.asyncio
async def test_edgar_downloads_two_10ks_and_four_10qs_for_nvda():
    from data.edgar import download_filings

    paths = await download_filings("NVDA")
    kinds = [p.parts[-3] for p in paths]
    assert kinds.count("10-K") >= 2, f"expected >=2 10-Ks on disk, got: {kinds}"
    assert kinds.count("10-Q") >= 4, f"expected >=4 10-Qs on disk, got: {kinds}"


@pytest.mark.asyncio
async def test_edgar_second_call_is_idempotent_and_fast():
    from data.edgar import download_filings

    await download_filings("NVDA")  # warm the disk
    start = time.monotonic()
    paths = await download_filings("NVDA")
    elapsed = time.monotonic() - start
    assert paths, "second call returned no paths"
    assert elapsed < 5.0, f"second call should be fast (cache hit), took {elapsed:.1f}s"


def test_yfin_returns_all_five_keys_for_nvda():
    from data.yfin import EXPECTED_KEYS, get_financials

    data = get_financials("NVDA")
    for key in EXPECTED_KEYS:
        assert key in data
    assert data["info"], "yfinance info dict is empty for NVDA"


def test_yfin_cache_hit_on_second_call():
    from data.yfin import get_financials

    get_financials("NVDA")  # warm
    start = time.monotonic()
    get_financials("NVDA")
    assert time.monotonic() - start < 1.0, "second call should be <1s (cache hit)"


@pytest.mark.asyncio
async def test_chroma_ingest_and_query_round_trip_nvda():
    from data.chroma import ingest_filing, query
    from data.edgar import download_filings

    paths = await download_filings("NVDA")
    assert paths, "no NVDA filings on disk"
    # Pick the most recent 10-K to keep the test bounded.
    tenk = next((p for p in paths if "10-K" in p.parts), paths[0])

    chunks_added = await asyncio.to_thread(ingest_filing, "NVDA", tenk)
    assert chunks_added > 0

    results = await asyncio.to_thread(query, "NVDA", "data center capex outlook", 5)
    assert len(results) == 5
    for r in results:
        assert r["text"]
        assert r["metadata"]["ticker"] == "NVDA"


@pytest.mark.asyncio
async def test_chroma_item_filter_returns_only_matching_section():
    from data.chroma import query

    results = await asyncio.to_thread(
        query, "NVDA", "principal risks to the business", 5, "Item 1A"
    )
    assert results, "expected some Risk Factors chunks"
    for r in results:
        assert r["metadata"]["item_code"] == "1A", r["metadata"]


@pytest.mark.asyncio
async def test_chroma_cross_ticker_query_returns_chunks_from_multiple_tickers():
    """Without a ticker filter, queries should be able to surface chunks from any
    ingested ticker. Use two ticker-specific questions so semantic search picks up
    each — the point is that the data is queryable across tickers, not that one
    query returns balanced results."""
    from data.chroma import ingest_filing, query
    from data.edgar import download_filings

    paths = await download_filings("ANET")
    if paths:
        target = next((p for p in paths if "10-K" in p.parts), paths[0])
        await asyncio.to_thread(ingest_filing, "ANET", target)

    nvda_results = await asyncio.to_thread(
        query, None, "GPU revenue growth from data center customers", 5
    )
    anet_results = await asyncio.to_thread(
        query, None, "ethernet switch market share and networking products", 5
    )
    tickers_seen = {r["metadata"]["ticker"] for r in nvda_results + anet_results}
    assert {"NVDA", "ANET"}.issubset(
        tickers_seen
    ), f"expected NVDA and ANET to both be queryable, got: {tickers_seen}"
