"""Tier 1 unit tests for utils/live_eval.py — per-drill-in RAG eval sidecar.

Mocks the LLM-judge so we don't pay OpenRouter credits during unit tests.
Tier 3 (real LLM grading on a real drill-in) is gated behind `pytest -m eval`
and not exercised here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils import live_eval

# --- Env-var gating --------------------------------------------------------


def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv("EVAL_LIVE_DRILL_INS", raising=False)
    assert live_eval.is_enabled() is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "True"])
def test_is_enabled_truthy(monkeypatch, value):
    monkeypatch.setenv("EVAL_LIVE_DRILL_INS", value)
    assert live_eval.is_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "anything"])
def test_is_enabled_falsy(monkeypatch, value):
    monkeypatch.setenv("EVAL_LIVE_DRILL_INS", value)
    assert live_eval.is_enabled() is False


# --- evaluate_filings_retrieval --------------------------------------------


def _stub_judge_report():
    """Lightweight stand-in for utils.rag_eval.JudgeReport that satisfies
    serialise_judge_report's contract. Used to mock judge_relevance so the
    test doesn't hit OpenRouter."""
    from dataclasses import dataclass

    @dataclass
    class _Score:
        chunk_index: int
        label: str
        score: int
        rationale: str

    @dataclass
    class _Report:
        question: str
        k: int
        scores: list
        precision_at_k: float
        ndcg_at_k: float
        mrr: float

    return _Report(
        question="risk factors for NVDA",
        k=3,
        scores=[
            _Score(0, "HIGH", 3, "directly addresses"),
            _Score(1, "PARTIAL", 2, "tangential"),
            _Score(2, "NONE", 0, "off-topic"),
        ],
        precision_at_k=0.67,
        ndcg_at_k=0.85,
        mrr=1.0,
    )


def test_evaluate_filings_retrieval_persists_per_subquery(monkeypatch, tmp_path):
    """Three subqueries with chunks → three eval rows persisted to
    EVAL_OUTPUT_DIR. Each row has run_id, ticker, thesis, and the
    serialised judge report shape."""
    from utils import rag_eval

    # Redirect EVAL_OUTPUT_DIR so we don't pollute data_cache/eval/runs/.
    monkeypatch.setattr(rag_eval, "EVAL_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(live_eval, "judge_relevance", lambda q, c: _stub_judge_report())

    subqueries_with_chunks = [
        ({"label": "risk_factors", "question": "risks?"},
         [{"text": "supply concentration", "metadata": {}}]),
        ({"label": "mdna_trajectory", "question": "trajectory?"},
         [{"text": "revenue grew", "metadata": {}}]),
        ({"label": "segment_performance", "question": "segments?"},
         [{"text": "data center +47%", "metadata": {}}]),
    ]

    persisted = live_eval.evaluate_filings_retrieval(
        run_id="abcd-1234",
        ticker="NVDA",
        thesis_name="AI cake",
        subqueries_with_chunks=subqueries_with_chunks,
    )

    assert len(persisted) == 3
    for row in persisted:
        assert row["tier"] == 2
        assert row["suite"] == "filings_live"
        assert row["run_id"] == "abcd-1234"
        assert row["ticker"] == "NVDA"
        assert row["thesis"] == "AI cake"
        assert row["precision_at_k"] == pytest.approx(0.67)
        assert row["ndcg_at_k"] == pytest.approx(0.85)
        assert row["k"] == 3
        # subquery_label round-trips
        assert row["subquery_label"] in {
            "risk_factors", "mdna_trajectory", "segment_performance"
        }

    # Files actually written to disk
    files = list(Path(tmp_path).glob("*.json"))
    assert len(files) >= 3


def test_evaluate_filings_retrieval_empty_subquery_persists_zero_row(
    monkeypatch, tmp_path
):
    """A subquery with 0 retrieved chunks shouldn't crash the eval — it
    persists an empty row so Mission Control can show 'this subquery
    returned 0' as a data point."""
    from utils import rag_eval

    monkeypatch.setattr(rag_eval, "EVAL_OUTPUT_DIR", tmp_path)
    # Don't even need to mock judge_relevance — the empty path skips it.

    persisted = live_eval.evaluate_filings_retrieval(
        run_id="x",
        ticker="X",
        thesis_name="t",
        subqueries_with_chunks=[
            ({"label": "risk_factors", "question": "risks?"}, [])
        ],
    )
    assert len(persisted) == 1
    assert persisted[0]["k"] == 0
    assert persisted[0]["scores"] == []


def test_evaluate_filings_retrieval_handles_judge_failure(monkeypatch, tmp_path):
    """If judge_relevance raises (LLM outage), we log + skip that subquery
    rather than crash the whole eval pass."""
    from utils import rag_eval

    monkeypatch.setattr(rag_eval, "EVAL_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        live_eval,
        "judge_relevance",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("openrouter outage")),
    )

    persisted = live_eval.evaluate_filings_retrieval(
        run_id="x",
        ticker="X",
        thesis_name="t",
        subqueries_with_chunks=[
            ({"label": "risk_factors", "question": "risks?"},
             [{"text": "x", "metadata": {}}])
        ],
    )
    # Failure swallowed — nothing persisted, but no exception escaped.
    assert persisted == []


def test_evaluate_filings_retrieval_with_no_subqueries():
    """No subqueries → empty result, no errors."""
    persisted = live_eval.evaluate_filings_retrieval(
        run_id="x",
        ticker="X",
        thesis_name="t",
        subqueries_with_chunks=[],
    )
    assert persisted == []
