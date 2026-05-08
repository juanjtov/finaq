"""Recall@K eval over the `synthesis_reports` ChromaDB corpus.

Reads `data_cache/chroma` directly — gated behind `pytest -m eval` since
it depends on a populated corpus (run
`python -m scripts.index_existing_reports` first to backfill).

For each golden query we call `cio.rag.query_past_reports(...)` and check
that at least one expected substring appears (case-insensitive) in any
of the top-K chunks. Failures dump the retrieved chunks so the user can
debug retrieval quality without re-running the eval.
"""

from __future__ import annotations

import os

import pytest

from cio.rag import query_past_reports
from tests.eval.cio_reports_golden_queries import (
    CIO_REPORTS_GOLDEN_QUERIES,
    CIOReportsGoldenQuery,
)

K = 5  # top-K to look at


def _has_chroma_corpus() -> bool:
    """Fast check whether a populated synthesis_reports collection exists."""
    try:
        from data.chroma import _get_collection
        coll = _get_collection(name="synthesis_reports")
        return coll.count() > 0
    except Exception:
        return False


@pytest.mark.eval
@pytest.mark.skipif(
    not _has_chroma_corpus(),
    reason=(
        "synthesis_reports collection empty or unavailable — run "
        "`python -m scripts.index_existing_reports` to populate."
    ),
)
@pytest.mark.parametrize("gq", CIO_REPORTS_GOLDEN_QUERIES, ids=lambda g: g.description[:60])
def test_recall_at_k_for_golden_query(gq: CIOReportsGoldenQuery):
    """Recall@K passes when ≥1 expected substring is found in any top-K
    chunk's text."""
    chunks = query_past_reports(
        question=gq.query,
        ticker=gq.ticker,
        thesis=gq.thesis,
        k=K,
    )
    assert chunks, (
        f"No chunks retrieved for ticker={gq.ticker!r} thesis={gq.thesis!r}. "
        f"Either the corpus doesn't have this ticker indexed yet, or the "
        f"metadata pre-filter is too strict."
    )

    haystack = "\n".join((c.get("text") or "").lower() for c in chunks)
    matched = [s for s in gq.expected_substrings if s.lower() in haystack]
    if not matched:
        retrieved_summary = "\n".join(
            f"  - section={c['metadata'].get('section')!r} "
            f"thesis={c['metadata'].get('thesis')!r} "
            f"date={c['metadata'].get('date')!r}\n    text: {(c.get('text') or '')[:200]!r}"
            for c in chunks
        )
        pytest.fail(
            f"\nNo expected substring matched for query={gq.query!r}.\n"
            f"Expected any of: {gq.expected_substrings}\n"
            f"Retrieved {len(chunks)} chunk(s):\n{retrieved_summary}\n"
            f"({gq.description})"
        )
