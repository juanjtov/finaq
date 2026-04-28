"""RAG retrieval-quality eval suite — Tier 1 (deterministic, always-on).

Three test categories:
  - Metadata pre-filter correctness (ticker / item_code don't leak)
  - Freshness markers (every chunk carries filed_date)
  - Golden-set recall@K (hand-curated discriminative queries)

Tier 2 (LLM-as-judge) lives in test_rag_eval_llm.py and is gated `pytest -m eval`.
Tier 3 (RAGAS) lives in test_rag_ragas.py, also gated `pytest -m eval`.

Run via:  pytest -m integration tests/test_retrieval_quality.py

Eval results from this file are persisted to data_cache/eval/runs/ so
Mission Control (Step 8) can render the latest scores + per-query pass/fail.
"""

from __future__ import annotations

import os

import pytest

from data.chroma import query
from tests.eval.golden_queries import GOLDEN_QUERIES, GoldenQuery, chunks_match_golden
from utils.rag_eval import write_eval_run

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
    chunks = query("NVDA", "any company commentary", k=10)
    for c in chunks:
        assert c["metadata"]["ticker"] == "NVDA", c["metadata"]


def test_metadata_pre_filter_to_item_1a_excludes_other_items():
    chunks = query("NVDA", "supply chain risks", k=5, item_filter="1A")
    assert chunks
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
        assert len(filed) == 10 and filed[4] == "-" and filed[7] == "-", filed


# --- Golden-set recall@K (parametrised across hand-curated queries) --------


@pytest.mark.parametrize("gq", GOLDEN_QUERIES, ids=lambda q: q.query[:40])
def test_golden_query_recall_at_k(gq: GoldenQuery):
    """Each golden query should retrieve at least one chunk containing one of
    its expected substrings within the top-K. This is a recall@K bar — a
    measurable quality signal that catches retrieval regressions."""
    chunks = query(gq.ticker, gq.query, k=8, item_filter=gq.item_filter)
    assert chunks, f"no chunks returned for golden query: {gq.query!r}"
    passed, matched, rank = chunks_match_golden(chunks, gq)
    assert (
        passed
    ), f"golden query failed (no expected substring in top-8): {gq.query!r}\n  expected one of: {gq.expected_substrings}"


def test_golden_set_summary_recorded_to_eval_dir():
    """Run all golden queries and persist a summary so Mission Control can
    render the latest scores. This is the SINGLE place that materialises a
    Tier-1 evaluation snapshot for the dashboard."""
    per_query: list[dict] = []
    pass_count = 0
    rank_sum = 0
    rank_count = 0
    for gq in GOLDEN_QUERIES:
        chunks = query(gq.ticker, gq.query, k=8, item_filter=gq.item_filter)
        passed, matched, rank = chunks_match_golden(chunks, gq)
        per_query.append(
            {
                "query": gq.query,
                "ticker": gq.ticker,
                "item_filter": gq.item_filter,
                "expected_substrings": list(gq.expected_substrings),
                "passed": passed,
                "matching_substring": matched,
                "rank_of_first_match": rank,
                "top_k_count": len(chunks),
                "description": gq.description,
            }
        )
        if passed:
            pass_count += 1
            if rank is not None:
                rank_sum += rank
                rank_count += 1

    total = len(GOLDEN_QUERIES)
    recall_at_k = pass_count / total if total else 0.0
    avg_rank = (rank_sum / rank_count) if rank_count else None

    write_eval_run(
        {
            "tier": 1,
            "suite": "golden_recall_at_k",
            "k": 8,
            "total_queries": total,
            "passed": pass_count,
            "recall_at_k": recall_at_k,
            "avg_rank_of_first_match": avg_rank,
            "per_query": per_query,
        }
    )

    # Bar: at least 75% of golden queries should pass. Below this is a regression.
    assert (
        recall_at_k >= 0.75
    ), f"golden-set recall@8 below 75% bar: {recall_at_k:.2%} ({pass_count}/{total})"


# --- Top-K size honored ---------------------------------------------------


def test_query_returns_no_more_than_k_chunks():
    chunks = query("NVDA", "anything", k=5)
    assert len(chunks) <= 5


def test_query_returns_no_more_than_k_chunks_with_filter():
    chunks = query("NVDA", "anything", k=3, item_filter="1A")
    assert len(chunks) <= 3
