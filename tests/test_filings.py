"""Step 5b unit tests — Filings agent prompt assembly + failure paths.

Pure-logic, no network. Real RAG runs in test_filings_integration.py and
test_retrieval_quality.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.filings import (
    SUBQUERY_K,
    _build_subqueries,
    _build_user_prompt,
    _format_chunk,
    _strip_code_fences,
    run,
)
from utils.schemas import FilingsOutput

THESES_DIR = Path(__file__).parents[1] / "theses"


# --- Subquery construction ---------------------------------------------------


def test_build_subqueries_returns_three_with_correct_item_filters():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    sqs = _build_subqueries("NVDA", thesis)
    assert len(sqs) == 3
    labels = {sq["label"] for sq in sqs}
    assert labels == {"risk_factors", "mdna_trajectory", "segment_performance"}

    by_label = {sq["label"]: sq for sq in sqs}
    assert by_label["risk_factors"]["item_filter"] == "1A"
    assert by_label["mdna_trajectory"]["item_filter"] == "7"
    assert by_label["segment_performance"]["item_filter"] is None


def test_build_subqueries_interpolates_thesis_name_and_ticker():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    sqs = _build_subqueries("NVDA", thesis)
    for sq in sqs:
        assert "NVDA" in sq["question"], sq
        assert "AI cake" in sq["question"], sq


# --- Prompt assembly ---------------------------------------------------------


def test_format_chunk_includes_filed_date_and_accession():
    chunk = {
        "text": "We are supply-constrained on Blackwell.",
        "metadata": {
            "accession": "0001045810-25-000023",
            "item_label": "Item 7. Management's Discussion",
            "filed_date": "2025-02-26",
        },
    }
    out = _format_chunk(1, chunk)
    assert "0001045810-25-000023" in out
    assert "filed_date=2025-02-26" in out
    assert "supply-constrained" in out


def test_build_user_prompt_includes_each_subquery_label_and_chunks():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    chunk = {
        "text": "Sample 10-K text.",
        "metadata": {
            "accession": "0001-25-001",
            "item_label": "Item 1A. Risk Factors",
            "filed_date": "2025-02-26",
        },
    }
    sqs = _build_subqueries("NVDA", thesis)
    prompt = _build_user_prompt(
        "NVDA", thesis, [(sqs[0], [chunk]), (sqs[1], [chunk]), (sqs[2], [])]
    )
    assert "AS OF" in prompt
    assert "risk_factors" in prompt
    assert "mdna_trajectory" in prompt
    assert "segment_performance" in prompt
    assert "(no chunks retrieved)" in prompt  # segment was empty
    assert "STRICT JSON ONLY" in prompt


# --- _strip_code_fences ------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"summary":"x"}', '{"summary":"x"}'),
        ('```json\n{"summary":"x"}\n```', '{"summary":"x"}'),
        ('```\n{"a":1}\n```', '{"a":1}'),
    ],
)
def test_strip_code_fences(raw, expected):
    assert _strip_code_fences(raw) == expected


# --- run() failure paths -----------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_ticker_not_ingested_message_when_chroma_empty(monkeypatch):
    """When ChromaDB has no chunks for the ticker AT ALL, surface a precise
    actionable error pointing at scripts.ingest_universe — not a generic
    'no chunks retrieved' message."""
    from agents import filings as f

    monkeypatch.setattr(f, "_retrieve_for_subquery", lambda *a, **kw: [])
    # has_ticker returns False → "ticker_not_ingested" path
    monkeypatch.setattr("data.chroma.has_ticker", lambda ticker: False)

    state = {"ticker": "ZZZZ", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FilingsOutput.model_validate(result["filings"])
    assert "ZZZZ" in out.summary
    assert any("ticker_not_ingested" in e for e in out.errors)
    assert any("scripts.ingest_universe" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_returns_empty_query_match_when_ticker_ingested_but_no_hits(
    monkeypatch,
):
    """When ChromaDB DOES have chunks for the ticker but the 3 subqueries all
    return 0 matches, emit a different (rare) error so we know the issue is
    the query templates, not the corpus."""
    from agents import filings as f

    monkeypatch.setattr(f, "_retrieve_for_subquery", lambda *a, **kw: [])
    monkeypatch.setattr("data.chroma.has_ticker", lambda ticker: True)
    monkeypatch.setattr("data.edgar.has_filings_in_unsupported_kinds", lambda t: [])

    state = {"ticker": "ABCD", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FilingsOutput.model_validate(result["filings"])
    assert "ABCD" in out.summary
    assert any("empty_query_match" in e for e in out.errors)
    # Should NOT recommend running the ingest script — this is not an ingest issue.
    assert not any("scripts.ingest_universe" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_returns_foreign_issuer_message_when_unsupported_kinds_present(
    monkeypatch,
):
    """Real failure mode: TSM + ASML file 20-F + 6-K, not 10-K + 10-Q. Our
    pipeline only ingests the latter; ChromaDB stays empty even after a
    successful download attempt. The agent should detect the EDGAR cache
    HAS filings in 20-F/6-K dirs and emit a precise message — NOT prompt
    the user to re-run scripts.ingest_universe (which won't help)."""
    from agents import filings as f

    monkeypatch.setattr(f, "_retrieve_for_subquery", lambda *a, **kw: [])
    monkeypatch.setattr("data.chroma.has_ticker", lambda ticker: False)
    monkeypatch.setattr(
        "data.edgar.has_filings_in_unsupported_kinds",
        lambda ticker: ["20-F", "6-K"] if ticker == "TSM" else [],
    )

    state = {"ticker": "TSM", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FilingsOutput.model_validate(result["filings"])

    # Foreign-issuer error path
    assert any("foreign_issuer" in e for e in out.errors)
    assert any("20-F" in e for e in out.errors)
    assert any("6-K" in e for e in out.errors)

    # Must NOT suggest re-running scripts.ingest_universe (won't help)
    assert not any(
        "scripts.ingest_universe" in e and "Run" in e for e in out.errors
    ), f"foreign-issuer error should not prompt re-ingest: {out.errors}"
    # Summary explains the situation
    assert "20-F" in out.summary
    assert "TSM" in out.summary


@pytest.mark.asyncio
async def test_run_falls_back_when_llm_fails(monkeypatch):
    """LLM raises but retrieval succeeded — output must still be schema-valid."""
    from agents import filings as f

    fake_chunks = [
        {
            "text": "stub filing chunk",
            "metadata": {
                "accession": "0001-25-001",
                "item_label": "Item 1A",
                "filed_date": "2025-02-26",
            },
        }
    ]
    monkeypatch.setattr(f, "_retrieve_for_subquery", lambda *a, **kw: fake_chunks)
    monkeypatch.setattr(
        f,
        "_call_llm",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated openrouter outage")),
    )

    state = {"ticker": "STUB", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FilingsOutput.model_validate(result["filings"])
    assert any("llm" in e for e in out.errors)
    # No retrieval errors — only the LLM error.
    retrieval_errors = [e for e in out.errors if e.startswith("retrieval/")]
    assert not retrieval_errors


@pytest.mark.asyncio
async def test_run_propagates_llm_output_when_call_succeeds(monkeypatch):
    from agents import filings as f

    fake_chunks = [
        {
            "text": "We continue to face supply constraints on Blackwell platforms.",
            "metadata": {
                "accession": "0001045810-25-000023",
                "item_label": "Item 7. MD&A",
                "filed_date": "2025-02-26",
            },
        }
    ]
    monkeypatch.setattr(f, "_retrieve_for_subquery", lambda *a, **kw: fake_chunks)

    fake_response = {
        "summary": "[stub] thesis-aware filings synthesis",
        "risk_themes": ["supply constraint", "customer concentration"],
        "mdna_quotes": [
            {
                "text": "We continue to face supply constraints on Blackwell platforms.",
                "accession": "0001045810-25-000023",
                "item": "Item 7. MD&A",
            }
        ],
        "evidence": [
            {
                "source": "edgar",
                "accession": "0001045810-25-000023",
                "item": "Item 7. MD&A",
                "excerpt": "supply constraints",
                "as_of": "2025-02-26",
            }
        ],
    }
    monkeypatch.setattr(f, "_call_llm", lambda *a, **kw: fake_response)

    state = {"ticker": "NVDA", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FilingsOutput.model_validate(result["filings"])
    assert "[stub]" in out.summary
    assert out.evidence[0].as_of == "2025-02-26"
    assert out.errors == []


def test_subquery_k_is_eight_per_spec():
    """CLAUDE.md §9.2: 'Each subquery returns top-8 chunks'."""
    assert SUBQUERY_K == 8
