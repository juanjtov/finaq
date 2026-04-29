"""Tier 3 Risk evaluation — RAGAS faithfulness + answer_relevancy on the summary.

Treats the Risk agent's `summary` as the answer, the concatenation of the three
worker summaries as the context, and a meta-question as the user input. RAGAS
computes faithfulness (every claim grounded) + answer_relevancy + context_precision.

Heavy: pulls langchain + datasets via ragas, costs ~$0.50–$1.50 per run.
Gated `pytest -m eval`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.risk import run
from tests.test_risk_quality import _stub_state
from utils.rag_eval import write_eval_run
from utils.schemas import RiskOutput

pytestmark = pytest.mark.eval

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    judge = os.environ.get("MODEL_JUDGE", "")
    if not judge or judge == "test-stub-model":
        pytest.skip("MODEL_JUDGE not set in .env")


@pytest.mark.asyncio
async def test_ragas_evaluation_on_risk_summary():
    """End-to-end RAGAS eval on the Risk agent's NVDA + AI cake summary."""
    from utils.rag_ragas import evaluate_filings_run  # reuse — same metric shape

    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = _stub_state("NVDA", thesis)
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])

    if not out.summary:
        pytest.skip("Risk produced no summary")

    # Treat the three worker summaries as RAGAS contexts.
    contexts = [
        {"text": f"FUNDAMENTALS: {state['fundamentals'].get('summary', '')}"},
        {"text": f"FILINGS: {state['filings'].get('summary', '')}"},
        {"text": f"NEWS: {state['news'].get('summary', '')}"},
    ]

    meta_question = (
        f"What are the most important risks to NVDA under the {thesis['name']} thesis, "
        "given the latest fundamentals, filings, and news?"
    )

    report = evaluate_filings_run(
        question=meta_question,
        retrieved_chunks=contexts,
        answer=out.summary,
        ground_truth=None,
    )

    write_eval_run(
        {
            "tier": 3,
            "suite": "ragas_risk",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            **report.to_dict(),
            "raw": report.raw,
        }
    )

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
