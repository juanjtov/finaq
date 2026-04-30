"""Tier 1 unit tests for agents/qa.py — per-agent free-text Q&A.

No real LLM calls; the LLM is monkeypatched in every test that exercises a
full pipeline. Coverage focuses on:

  - Prompt assembly (each per-agent context builder produces the right shape)
  - Coercion (LLM JSON → AgentAnswer with citations)
  - Failure paths (empty state.<agent> → graceful "no data" answer; LLM raise
    → graceful failure answer)
  - Filings RAG fallback (no chunks retrieved → "ingest first" message)

Real-LLM Q&A is exercised via Tier 3 (integration) in a future test_qa_integration
test file when the dashboard integration is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.qa import (
    _coerce_answer,
    _filings_context_from_chunks,
    _fundamentals_context,
    _news_context,
    _risk_context,
    _strip_code_fences,
    ask,
)
from utils.schemas import AgentAnswer

THESES_DIR = Path(__file__).parents[1] / "theses"


# --- Fixtures ---------------------------------------------------------------


def _state_with(**fields) -> dict:
    """Build a minimal state with only the requested fields populated."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    base = {"ticker": "NVDA", "thesis": thesis}
    base.update(fields)
    return base


def _populated_fundamentals() -> dict:
    return {
        "summary": "NVDA: 47% YoY revenue growth, 65% margins.",
        "kpis": {
            "fcf_yield": 1.84,
            "pe_trailing": 44.3,
            "current_price": 200.0,
            "revenue_latest": 60.9e9,
        },
        "projections": {
            "revenue_growth_mean": 0.20,
            "revenue_growth_std": 0.05,
            "margin_mean": 0.65,
            "margin_std": 0.04,
            "tax_rate_mean": 0.18,
            "tax_rate_std": 0.02,
            "maintenance_capex_pct_rev_mean": 0.04,
            "maintenance_capex_pct_rev_std": 0.01,
            "da_pct_rev_mean": 0.03,
            "da_pct_rev_std": 0.01,
            "dilution_rate_mean": 0.005,
            "dilution_rate_std": 0.002,
            "exit_multiple_mean": 28.0,
            "exit_multiple_std": 4.0,
        },
    }


def _populated_news() -> dict:
    return {
        "summary": "Mixed signals.",
        "catalysts": [
            {
                "title": "MSFT raised AI capex guidance to $80B",
                "summary": "msft up",
                "sentiment": "bull",
                "url": "https://example.com/msft",
                "as_of": "2026-04-15",
            }
        ],
        "concerns": [
            {
                "title": "Sell-side flags valuation premium",
                "summary": "trims",
                "sentiment": "bear",
                "url": "https://example.com/val",
                "as_of": "2026-04-19",
            }
        ],
    }


def _populated_risk() -> dict:
    return {
        "level": "ELEVATED",
        "score_0_to_10": 6,
        "summary": "Multiple convergent signals.",
        "top_risks": [
            {
                "title": "Supply concentration",
                "severity": 4,
                "explanation": "TSM N3/N2 dependency.",
                "sources": ["fundamentals", "filings"],
            }
        ],
        "convergent_signals": [
            {
                "theme": "supply concentration",
                "sources": ["fundamentals", "filings"],
                "explanation": "Both surface same risk.",
            }
        ],
        "threshold_breaches": [
            {
                "signal": "fcf_yield",
                "operator": "<",
                "threshold_value": 4,
                "observed_value": 1.84,
                "explanation": "Below MoS.",
                "source": "fundamentals",
            }
        ],
    }


# --- Per-agent context builders --------------------------------------------


def test_fundamentals_context_includes_kpis_and_projections():
    state = _state_with(fundamentals=_populated_fundamentals())
    ctx = _fundamentals_context(state)
    assert "FUNDAMENTALS" in ctx
    assert "fcf_yield" in ctx
    assert "revenue_growth_mean" in ctx


def test_news_context_renders_catalysts_and_concerns():
    state = _state_with(news=_populated_news())
    ctx = _news_context(state)
    assert "MSFT raised" in ctx
    assert "Sell-side flags" in ctx
    assert "[bull " in ctx
    assert "[bear " in ctx


def test_risk_context_renders_top_risks_with_severity():
    state = _state_with(risk=_populated_risk())
    ctx = _risk_context(state)
    assert "level: ELEVATED" in ctx
    assert "Supply concentration" in ctx
    assert "convergent_signals" in ctx
    assert "threshold_breaches" in ctx
    assert "fcf_yield" in ctx


def test_filings_context_handles_empty_chunks():
    out = _filings_context_from_chunks([])
    assert "none" in out


def test_filings_context_renders_chunks_with_metadata():
    chunks = [
        {
            "text": "Capacity constraints persist across leading-edge nodes.",
            "metadata": {
                "accession": "0001045810-26-000123",
                "item_label": "Item 7",
                "filed_date": "2026-02-21",
            },
        }
    ]
    out = _filings_context_from_chunks(chunks)
    assert "0001045810-26-000123" in out
    assert "Item 7" in out
    assert "Capacity constraints" in out


# --- _strip_code_fences -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"answer":"x"}', '{"answer":"x"}'),
        ('```json\n{"answer":"x"}\n```', '{"answer":"x"}'),
        ('```\n{"a":1}\n```', '{"a":1}'),
    ],
)
def test_strip_code_fences(raw, expected):
    assert _strip_code_fences(raw) == expected


# --- _parse_llm_response — graceful JSON recovery --------------------------


def test_parse_llm_response_handles_strict_json():
    """The happy path — well-formed JSON returns the parsed dict."""
    from agents.qa import _parse_llm_response

    out = _parse_llm_response('{"answer": "ok", "citations": []}')
    assert out["answer"] == "ok"
    assert out["citations"] == []
    assert "_parse_recovery" not in out


def test_parse_llm_response_strips_code_fences_first():
    from agents.qa import _parse_llm_response

    out = _parse_llm_response('```json\n{"answer": "ok"}\n```')
    assert out["answer"] == "ok"


def test_parse_llm_response_recovers_from_unterminated_string():
    """Real failure mode: LLM hits max_tokens mid-citation. We regex-extract
    the answer field and surface a parse-recovery flag."""
    from agents.qa import _parse_llm_response

    truncated = (
        '{"answer": "Construction-sector hyperscaler exposure is limited.",\n'
        ' "citations": [\n'
        '  {"source": "tavily", "url": "https://example.com/very-long-url-that-was-cut-off-mid'
        # No closing quote, no closing bracket — JSON is broken
    )
    out = _parse_llm_response(truncated)
    assert "Construction-sector" in out["answer"]
    assert out["citations"] == []
    assert "_parse_recovery" in out
    assert "regex-extracted" in out["_parse_recovery"]


def test_parse_llm_response_falls_back_to_raw_text_when_unrecoverable():
    """Total failure: emit the raw text as the answer with a parse-recovery
    flag so the user sees something rather than a crash."""
    from agents.qa import _parse_llm_response

    out = _parse_llm_response("not json at all, just prose from the LLM")
    assert "not json" in out["answer"]
    assert out["citations"] == []
    assert "_parse_recovery" in out
    assert "unparseable" in out["_parse_recovery"]


def test_parse_llm_response_handles_empty_string():
    from agents.qa import _parse_llm_response

    out = _parse_llm_response("")
    assert out["citations"] == []
    assert "_parse_recovery" in out


def test_parse_llm_response_rejects_non_object_json():
    """If the LLM returns a list or string at the top level, we treat that
    as a parse failure and recover."""
    from agents.qa import _parse_llm_response

    out = _parse_llm_response('["a", "b"]')
    # Falls back to "raw text as answer" path — shows what we got
    assert "_parse_recovery" in out


# --- _coerce_answer surfaces _parse_recovery as an error -------------------


def test_coerce_answer_propagates_parse_recovery_to_errors():
    """When _parse_llm_response signals recovery, the user-visible error
    list includes a 'parse-recovery' note so the dashboard / Telegram can
    show 'partial answer' rather than confidently misreading."""
    out = _coerce_answer(
        "news",
        "anything?",
        {
            "answer": "Construction sector OK.",
            "citations": [],
            "_parse_recovery": "regex-extracted; citations dropped",
        },
        [],
    )
    assert "Construction sector OK." in out.answer
    assert any("parse-recovery" in e for e in out.errors)


# --- _coerce_answer ---------------------------------------------------------


def test_coerce_answer_happy_path():
    out = _coerce_answer(
        "fundamentals",
        "what is the revenue?",
        {
            "answer": "Latest revenue is $60.9B (Fund kpis: revenue_latest).",
            "citations": [
                {
                    "source": "fundamentals",
                    "note": "revenue_latest",
                    "excerpt": "$60.9B",
                }
            ],
        },
        [],
    )
    assert isinstance(out, AgentAnswer)
    assert out.agent == "fundamentals"
    assert out.question == "what is the revenue?"
    assert "60.9B" in out.answer
    assert len(out.citations) == 1
    assert out.citations[0].source == "fundamentals"
    assert out.errors == []


def test_coerce_answer_handles_empty_answer():
    out = _coerce_answer("risk", "q", {"answer": "", "citations": []}, [])
    assert "empty" in out.answer
    assert "empty answer" in out.errors


def test_coerce_answer_drops_malformed_citations():
    """A non-dict in citations must not crash the coerce — drop it."""
    out = _coerce_answer(
        "filings",
        "q",
        {
            "answer": "x",
            "citations": [
                "not-a-dict",
                {"source": "edgar", "accession": "abc"},
                42,
            ],
        },
        [],
    )
    assert len(out.citations) == 1
    assert out.citations[0].accession == "abc"


def test_coerce_answer_handles_missing_citations_field():
    out = _coerce_answer("news", "q", {"answer": "ok"}, [])
    assert out.citations == []


# --- ask() failure paths (no real LLM) -------------------------------------


@pytest.mark.asyncio
async def test_ask_rejects_unknown_agent():
    state = _state_with()
    with pytest.raises(ValueError, match="unknown agent"):
        await ask(state, "synthesis", "q")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_ask_rejects_empty_question():
    state = _state_with()
    with pytest.raises(ValueError, match="non-empty"):
        await ask(state, "fundamentals", "")


@pytest.mark.asyncio
async def test_ask_returns_no_data_message_when_state_empty_for_fundamentals():
    state = _state_with()  # no fundamentals
    out = await ask(state, "fundamentals", "what is the revenue?")
    assert "No Fundamentals data" in out.answer
    assert "state.fundamentals is empty" in out.errors


@pytest.mark.asyncio
async def test_ask_returns_no_data_message_when_state_empty_for_news():
    state = _state_with()
    out = await ask(state, "news", "any catalysts?")
    assert "No News data" in out.answer


@pytest.mark.asyncio
async def test_ask_returns_no_data_message_when_state_empty_for_risk():
    state = _state_with()
    out = await ask(state, "risk", "biggest concern?")
    assert "No Risk synthesis" in out.answer


@pytest.mark.asyncio
async def test_ask_propagates_llm_output_when_call_succeeds_for_fundamentals(monkeypatch):
    """Happy path with a mocked LLM response."""
    from agents import qa

    fake_resp = {
        "answer": "Revenue grew 47% YoY (Fund kpis).",
        "citations": [
            {"source": "fundamentals", "note": "revenue_latest", "excerpt": "47% YoY"}
        ],
    }
    monkeypatch.setattr(qa, "_call_llm", lambda system, user: fake_resp)

    state = _state_with(fundamentals=_populated_fundamentals())
    out = await ask(state, "fundamentals", "what is the revenue growth?")
    assert "47% YoY" in out.answer
    assert out.agent == "fundamentals"
    assert len(out.citations) == 1


@pytest.mark.asyncio
async def test_ask_returns_graceful_failure_when_llm_raises(monkeypatch):
    from agents import qa

    monkeypatch.setattr(
        qa, "_call_llm", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("openrouter outage"))
    )
    state = _state_with(risk=_populated_risk())
    out = await ask(state, "risk", "biggest concern?")
    assert "LLM call failed" in out.answer
    assert any("openrouter outage" in e for e in out.errors)


@pytest.mark.asyncio
async def test_ask_filings_returns_no_data_when_chroma_empty(monkeypatch):
    """When ChromaDB returns no chunks for the ticker, we surface that
    explicitly rather than asking the LLM to answer over no context."""

    monkeypatch.setattr("data.chroma.query", lambda *a, **kw: [])

    state = _state_with()  # ticker NVDA, no filings in state — that's fine for ask path
    out = await ask(state, "filings", "what about export controls?")
    assert "No filings chunks retrieved" in out.answer


@pytest.mark.asyncio
async def test_ask_filings_propagates_llm_when_chunks_present(monkeypatch):
    from agents import qa

    monkeypatch.setattr(
        "data.chroma.query",
        lambda *a, **kw: [
            {
                "text": "Capacity constraints persist across leading-edge nodes.",
                "metadata": {
                    "accession": "0001045810-26-000123",
                    "item_label": "Item 7",
                    "filed_date": "2026-02-21",
                },
            }
        ],
    )
    fake_resp = {
        "answer": "The 10-K says: 'Capacity constraints persist...' (Filings 10-K Item 7).",
        "citations": [
            {
                "source": "edgar",
                "accession": "0001045810-26-000123",
                "item": "Item 7",
                "excerpt": "Capacity constraints persist...",
            }
        ],
    }
    monkeypatch.setattr(qa, "_call_llm", lambda system, user: fake_resp)

    state = _state_with()  # ticker NVDA
    out = await ask(state, "filings", "are there supply constraints?")
    assert "Capacity constraints" in out.answer
    assert out.citations[0].accession == "0001045810-26-000123"


@pytest.mark.asyncio
async def test_ask_filings_handles_chroma_failure(monkeypatch):
    """ChromaDB raise → tenacity retry → finally surfaces in errors."""

    monkeypatch.setattr(
        "data.chroma.query",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("chroma down")),
    )
    state = _state_with()
    out = await ask(state, "filings", "anything?")
    assert "No filings chunks retrieved" in out.answer
    assert any("retrieval" in e for e in out.errors)
