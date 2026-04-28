"""Tier 1 faithfulness check — every Filings synthesis claim must be grounded.

Runs the real Filings agent end-to-end, then verifies:
  - mdna_quotes appear verbatim (substring match) in some retrieved chunk
  - evidence.accession values match accessions actually retrieved

Catches the most common RAG failure mode: the LLM fabricating quotes or
citations that look plausible but don't exist in the source. Pure string
matching, no LLM judge call.

Persists the result to data_cache/eval/runs/ for Mission Control.

Run via:  pytest -m integration tests/test_rag_eval_faithfulness.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from agents.filings import _build_subqueries, _retrieve_for_subquery, run
from data.chroma import query
from utils.rag_eval import check_faithfulness, write_eval_run
from utils.schemas import FilingsOutput

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys_and_corpus():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    chunks = query("NVDA", "anything", k=1)
    if not chunks:
        pytest.skip("ChromaDB has no NVDA corpus; run data-layer integration first")


@pytest.mark.asyncio
async def test_filings_synthesis_is_grounded_in_retrieved_chunks():
    """Every quote and citation in the Filings output must trace back to a chunk
    that retrieval actually returned. Catches LLM hallucination of source material."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())

    # Re-retrieve the same chunks the agent would use, so we can validate against them.
    subqueries = _build_subqueries("NVDA", thesis)
    all_chunks: list[dict] = []
    for sq in subqueries:
        all_chunks.extend(_retrieve_for_subquery("NVDA", sq))

    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = FilingsOutput.model_validate(result["filings"])

    if out.errors and any("no chunks" in e for e in out.errors):
        pytest.skip("ChromaDB has no NVDA chunks; run data-layer integration first")

    faith = check_faithfulness(
        mdna_quotes=[q.model_dump() for q in out.mdna_quotes],
        evidence=[e.model_dump() for e in out.evidence],
        retrieved_chunks=all_chunks,
    )

    write_eval_run(
        {
            "tier": 1,
            "suite": "filings_faithfulness",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            **asdict(faith),
        }
    )

    # Bars (calibrated for stochastic LLM output):
    #  - faithfulness_rate ≥ 0.8: most quotes verbatim-grounded; tolerates the
    #    LLM lightly normalising whitespace or punctuation in 1 of every 5 quotes.
    #  - citation_accuracy == 1.0: stricter, because fabricating an accession
    #    number is a hard failure. Real accessions are 18 chars, distinctive,
    #    and shouldn't be hallucinated even by a noisy LLM.
    assert faith.faithfulness_rate >= 0.8, (
        f"faithfulness below 80% bar ({faith.quotes_grounded}/{faith.quotes_total} = "
        f"{faith.faithfulness_rate:.0%}): ungrounded={faith.ungrounded_quotes}"
    )
    assert faith.citation_accuracy == 1.0, (
        f"fabricated accessions ({faith.citations_grounded}/{faith.citations_total}): "
        f"{faith.fabricated_accessions}"
    )
