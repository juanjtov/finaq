"""Tier 3 RAG evaluation — RAGAS framework metrics.

Runs the full Filings drill-in, then evaluates the result with RAGAS:

  - faithfulness        : every claim in the synthesis grounded in the contexts
  - answer_relevancy    : does the synthesis address the question?
  - context_precision   : are the top contexts truly relevant?

Heavier than Tier 2 — pulls in langchain + datasets, runs multiple judge calls
per metric, costs roughly $0.50–$1.50 per evaluation.

Gated behind `pytest -m eval`. Persists to data_cache/eval/runs/ for Mission Control.

Run via:  pytest -m eval tests/test_rag_eval_ragas.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.filings import _build_subqueries, _retrieve_for_subquery, run
from data.chroma import query
from utils.rag_eval import write_eval_run
from utils.schemas import FilingsOutput

pytestmark = pytest.mark.eval

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys_and_corpus():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    judge = os.environ.get("MODEL_JUDGE", "")
    if not judge or judge == "test-stub-model":
        pytest.skip("MODEL_JUDGE not set in .env (see .env.example)")
    chunks = query("NVDA", "anything", k=1)
    if not chunks:
        pytest.skip("ChromaDB has no NVDA corpus; run data-layer integration first")


@pytest.mark.asyncio
async def test_ragas_evaluation_on_filings_drill_in():
    """End-to-end RAGAS eval on a real Filings drill-in for NVDA + AI cake.

    Bar: faithfulness ≥ 0.6 and at least one other metric ≥ 0.5. Loose because
    RAGAS scores are LLM-judged and stochastic.
    """
    from utils.rag_ragas import evaluate_filings_run

    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())

    # Re-run the same retrieval the agent uses, capture chunks for RAGAS contexts.
    subqueries = _build_subqueries("NVDA", thesis)
    all_chunks: list[dict] = []
    for sq in subqueries:
        all_chunks.extend(_retrieve_for_subquery("NVDA", sq))

    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = FilingsOutput.model_validate(result["filings"])

    if not out.summary or "no chunks" in out.summary.lower():
        pytest.skip("Filings agent produced no synthesis; check ingestion")

    # Use the synthesis summary as the "answer" RAGAS evaluates.
    # Question is a meta-question that subsumes the 3 subqueries.
    meta_question = (
        f"What does {state['ticker']}'s most recent SEC filing say about its "
        f"prospects under the {thesis['name']} thesis — covering risks, MD&A "
        "trajectory, and segment performance?"
    )

    report = evaluate_filings_run(
        question=meta_question,
        retrieved_chunks=all_chunks,
        answer=out.summary,
        ground_truth=None,  # Tier 3 without reference; context_recall will be None
    )

    write_eval_run(
        {
            "tier": 3,
            "suite": "ragas_filings",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            **report.to_dict(),
            "raw": report.raw,
        }
    )

    # Soft bar — RAGAS scores are noisy
    if report.faithfulness is not None:
        assert report.faithfulness >= 0.6, f"RAGAS faithfulness too low: {report.faithfulness:.3f}"

    other_metrics = [
        m
        for m in (report.answer_relevancy, report.context_precision, report.context_recall)
        if m is not None
    ]
    if other_metrics:
        assert (
            max(other_metrics) >= 0.5
        ), f"all RAGAS non-faithfulness metrics below 0.5: {other_metrics}"
