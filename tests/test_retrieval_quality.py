"""Step 5b retrieval-quality evaluation suite — measurable bar for hybrid RAG.

Run via:  pytest -m integration tests/test_retrieval_quality.py

These tests fire known-relevant queries against the ingested NVDA corpus and
assert measurable properties of the top-K results. They catch regressions in:

  - chunking (mid-sentence cuts, missing item boundaries)
  - BM25 implementation (keyword recall)
  - RRF fusion (consistent items rise to top)
  - metadata pre-filter (other tickers / items don't leak)
  - filed_date metadata propagation

The bar is recall@K (does the top-K contain at least one chunk that mentions
the obvious topic?) rather than precision@K — a less brittle signal that flags
real regressions without false positives from arbitrary ranking changes.
"""

from __future__ import annotations

import os

import pytest

from data.chroma import query

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _require_keys_and_corpus():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    chunks = query("NVDA", "anything", k=1)
    if not chunks:
        pytest.skip("ChromaDB has no NVDA corpus; run data-layer integration first")


# --- Metadata pre-filter (must apply BEFORE retrieval) ----------------------


def test_metadata_pre_filter_excludes_other_tickers():
    """Filtering by ticker must exclude all chunks from other tickers."""
    chunks = query("NVDA", "any company commentary", k=10)
    for c in chunks:
        assert c["metadata"]["ticker"] == "NVDA", c["metadata"]


def test_metadata_pre_filter_to_item_1a_excludes_other_items():
    """item_filter='1A' must return chunks ONLY from Risk Factors section."""
    chunks = query("NVDA", "supply chain risks", k=5, item_filter="1A")
    assert chunks, "no chunks returned for Item 1A query"
    for c in chunks:
        assert c["metadata"]["item_code"] == "1A", c["metadata"]


def test_metadata_pre_filter_to_item_7_excludes_other_items():
    chunks = query("NVDA", "revenue trajectory", k=5, item_filter="7")
    assert chunks
    for c in chunks:
        assert c["metadata"]["item_code"] == "7", c["metadata"]


# --- Freshness markers must reach the agent ---------------------------------


def test_every_returned_chunk_has_filed_date():
    chunks = query("NVDA", "data center capex outlook", k=8)
    assert chunks
    for c in chunks:
        filed = c["metadata"].get("filed_date")
        assert filed, f"missing filed_date in metadata: {c['metadata']}"
        # Must be ISO format YYYY-MM-DD
        assert len(filed) == 10 and filed[4] == "-" and filed[7] == "-", filed


# --- Recall: well-known topics must surface --------------------------------


def test_recall_data_center_capex_topic():
    """A query for data-center capex should pull at least one chunk that
    mentions 'data center', 'capex', or 'capacity'."""
    chunks = query("NVDA", "data center capex outlook", k=8)
    assert chunks
    keywords = ("data center", "capex", "capacity", "compute")
    matched = sum(1 for c in chunks if any(kw in c["text"].lower() for kw in keywords))
    assert (
        matched >= 1
    ), f"no top-8 chunk mentioned a data-center keyword. texts: {[c['text'][:80] for c in chunks]}"


def test_recall_research_and_development_topic():
    chunks = query("NVDA", "research and development expenses", k=8)
    keywords = ("research and development", "r&d", "research")
    matched = sum(1 for c in chunks if any(kw in c["text"].lower() for kw in keywords))
    assert matched >= 1, "no top-8 chunk mentioned R&D"


# --- Hybrid contribution: BM25 must change the ranking --------------------


def test_keyword_only_query_changes_ranking_vs_pure_semantic():
    """A query with discriminative keywords should produce a different top-K
    when BM25 is enabled vs disabled — proving BM25 actually contributes."""
    q = "supply constraint Blackwell platform"

    sem_only = query("NVDA", q, k=8, use_keyword=False)
    hybrid = query("NVDA", q, k=8, use_keyword=True)

    sem_ids = [(c["metadata"]["accession"], c["metadata"]["item_code"]) for c in sem_only]
    hyb_ids = [(c["metadata"]["accession"], c["metadata"]["item_code"]) for c in hybrid]
    # Order should differ in at least one position (BM25 isn't a no-op)
    assert (
        sem_ids != hyb_ids
    ), "hybrid and semantic returned identical top-8 — BM25 isn't contributing"


# --- Top-K size honored ---------------------------------------------------


def test_query_returns_no_more_than_k_chunks():
    chunks = query("NVDA", "anything", k=5)
    assert len(chunks) <= 5


def test_query_returns_no_more_than_k_chunks_with_filter():
    chunks = query("NVDA", "anything", k=3, item_filter="1A")
    assert len(chunks) <= 3
