"""Tier 2 RAG evaluation — LLM-as-judge for precision@K, NDCG@K, MRR.

Runs the configured judge model (`MODEL_JUDGE` env var) over each top-K
chunk for the golden queries, scoring relevance 0–3. Computes:

  - precision@K  (fraction of chunks scored ≥ 2)
  - NDCG@K       (graded ranking quality)
  - MRR          (1 / rank of first relevant chunk)

Costs roughly $0.01 per query × 8 chunks × N queries.
For our 8-query golden set: ~$0.10–$0.15 per full Tier 2 run.

Gated behind `pytest -m eval` — never runs automatically.

Results persist to data_cache/eval/runs/ for the Mission Control panel
(Step 8). When state.db (Step 5z) lands, results also go there.

Run via:  pytest -m eval tests/test_rag_eval_llm.py
"""

from __future__ import annotations

import os

import pytest

from data.chroma import query
from tests.eval.golden_queries import GOLDEN_QUERIES, GoldenQuery
from utils.rag_eval import judge_relevance, serialise_judge_report, write_eval_run

pytestmark = pytest.mark.eval


@pytest.fixture(autouse=True)
def _require_keys_and_corpus():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    judge = os.environ.get("MODEL_JUDGE", "")
    # Skip if MODEL_JUDGE is missing or still the conftest stub.
    if not judge or judge == "test-stub-model":
        pytest.skip(
            "MODEL_JUDGE not set in .env (copy the line from .env.example, e.g. "
            "anthropic/claude-haiku-4.5)"
        )
    chunks = query("NVDA", "anything", k=1)
    if not chunks:
        pytest.skip("ChromaDB has no NVDA corpus; run data-layer integration first")


@pytest.mark.parametrize("gq", GOLDEN_QUERIES, ids=lambda q: q.query[:40])
def test_llm_judge_precision_per_query(gq: GoldenQuery):
    """For each golden query, retrieve top-8 and have the judge model score
    each chunk. Per-query bar is multi-signal: pass if ANY of (precision@8,
    NDCG@8, MRR) indicates retrieval surfaced relevant content. Per-query
    precision alone is noisy for some query types; the *aggregate* test
    enforces system-wide quality."""
    chunks = query(gq.ticker, gq.query, k=8, item_filter=gq.item_filter)
    assert chunks, f"no chunks retrieved for {gq.query!r}"
    report = judge_relevance(gq.query, chunks)
    write_eval_run(
        {
            "tier": 2,
            "suite": "llm_judge_per_query",
            "query": gq.query,
            "ticker": gq.ticker,
            "item_filter": gq.item_filter,
            **serialise_judge_report(report),
        }
    )
    # Pass if AT LEAST ONE quality signal succeeds — catches the catastrophic
    # case (retrieval found nothing relevant) while tolerating queries where
    # the judge is conservative about what counts as "PARTIAL or higher".
    signals = {
        "precision@8 >= 0.125 (≥1/8 relevant)": report.precision_at_k >= 0.125,
        "mrr >= 0.25 (first relevant in top-4)": report.mrr >= 0.25,
        "ndcg@8 >= 0.4 (overall ranking decent)": report.ndcg_at_k >= 0.4,
    }
    assert any(signals.values()), (
        f"all retrieval-quality signals failed for {gq.query!r}: "
        f"P@8={report.precision_at_k:.2%}, "
        f"NDCG@8={report.ndcg_at_k:.3f}, "
        f"MRR={report.mrr:.3f}"
    )


def test_llm_judge_aggregate_summary():
    """Run the judge across the entire golden set and persist an aggregate
    snapshot. The Mission Control panel reads from this file to render the
    headline 'Avg precision@8' / 'Avg NDCG@8' / 'Avg MRR' tiles."""
    per_query: list[dict] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []

    for gq in GOLDEN_QUERIES:
        chunks = query(gq.ticker, gq.query, k=8, item_filter=gq.item_filter)
        if not chunks:
            continue
        report = judge_relevance(gq.query, chunks)
        per_query.append(
            {
                "query": gq.query,
                **serialise_judge_report(report),
            }
        )
        precisions.append(report.precision_at_k)
        ndcgs.append(report.ndcg_at_k)
        mrrs.append(report.mrr)

    n = len(precisions)
    write_eval_run(
        {
            "tier": 2,
            "suite": "llm_judge_aggregate",
            "queries": n,
            "avg_precision_at_k": (sum(precisions) / n) if n else 0.0,
            "avg_ndcg_at_k": (sum(ndcgs) / n) if n else 0.0,
            "avg_mrr": (sum(mrrs) / n) if n else 0.0,
            "per_query": per_query,
        }
    )

    # Aggregate bar — at least one of the three metrics has to reach 0.5.
    # Looser than per-query because we're catching systemic regressions.
    if n > 0:
        avg_precision = sum(precisions) / n
        avg_ndcg = sum(ndcgs) / n
        avg_mrr = sum(mrrs) / n
        assert max(avg_precision, avg_ndcg, avg_mrr) >= 0.5, (
            f"all aggregate judge metrics below 0.5: "
            f"P@8={avg_precision:.3f}, NDCG@8={avg_ndcg:.3f}, MRR={avg_mrr:.3f}"
        )
