"""Unit tests for utils/rag_eval.py — pure-logic faithfulness + helpers.

No network. The LLM-as-judge pieces are integration-tested in test_rag_eval_llm.py
(gated `pytest -m eval`), since they require a real model.
"""

from __future__ import annotations

import pytest

from utils.rag_eval import _alnum_only, _normalise_text, check_faithfulness


def test_alnum_only_strips_punctuation_and_whitespace():
    assert _alnum_only("We continue to increase! Our supply...") == "wecontinuetoincreaseoursupply"


def test_alnum_only_handles_unicode_punctuation():
    """LLMs sometimes emit smart quotes / em-dashes — alnum should ignore them."""
    assert _alnum_only("Demand—per the filing—is up.") == "demandperthefilingisup"


def test_normalise_text_collapses_whitespace():
    assert _normalise_text("we   continue\n\nto   increase") == "we continue to increase"


def test_check_faithfulness_full_match():
    chunks = [
        {
            "text": "We continue to increase our supply and capacity purchases.",
            "metadata": {"accession": "0001-25-001"},
        }
    ]
    quotes = [{"text": "We continue to increase our supply and capacity purchases."}]
    evidence = [{"accession": "0001-25-001"}]
    result = check_faithfulness(quotes, evidence, chunks)
    assert result.faithfulness_rate == 1.0
    assert result.citation_accuracy == 1.0
    assert result.ungrounded_quotes == ()
    assert result.fabricated_accessions == ()


def test_check_faithfulness_tolerates_light_paraphrase_via_alnum_match():
    """LLM normalised whitespace and dropped a comma — should still match
    via alphanumeric-only prefix."""
    chunks = [
        {
            "text": "We continue, to increase our supply, and capacity purchases.",
            "metadata": {"accession": "0001-25-001"},
        }
    ]
    quotes = [{"text": "We continue to increase our supply and capacity purchases."}]
    result = check_faithfulness(quotes, [], chunks)
    assert result.faithfulness_rate == 1.0


def test_check_faithfulness_catches_fabricated_quote():
    """A quote whose first 60 alnum chars don't appear in any chunk is fabricated."""
    chunks = [{"text": "Real chunk text about NVIDIA.", "metadata": {"accession": "0001-25-001"}}]
    quotes = [{"text": "Completely fabricated quote about something else entirely."}]
    result = check_faithfulness(quotes, [], chunks)
    assert result.faithfulness_rate == 0.0
    assert "Completely fabricated" in result.ungrounded_quotes[0]


def test_check_faithfulness_catches_fabricated_accession():
    chunks = [{"text": "x", "metadata": {"accession": "0001-25-001"}}]
    evidence = [{"accession": "9999-99-FAKE"}]
    result = check_faithfulness([], evidence, chunks)
    assert result.citation_accuracy == 0.0
    assert "9999-99-FAKE" in result.fabricated_accessions


def test_check_faithfulness_partial_grounding_60_percent():
    chunks = [
        {"text": "First exact quote text appears here.", "metadata": {"accession": "A"}},
        {"text": "Second exact quote text appears here.", "metadata": {"accession": "B"}},
    ]
    quotes = [
        {"text": "First exact quote text appears here."},
        {"text": "Hallucinated quote that doesn't appear anywhere."},
        {"text": "Second exact quote text appears here."},
    ]
    result = check_faithfulness(quotes, [], chunks)
    assert result.faithfulness_rate == pytest.approx(2 / 3)
    assert len(result.ungrounded_quotes) == 1


def test_check_faithfulness_no_quotes_no_evidence():
    """Empty inputs: rates are 0/0 → 1.0 (vacuously true)."""
    chunks = [{"text": "x", "metadata": {"accession": "A"}}]
    result = check_faithfulness([], [], chunks)
    assert result.faithfulness_rate == 1.0
    assert result.citation_accuracy == 1.0
    assert result.quotes_total == 0
    assert result.citations_total == 0


def test_check_faithfulness_ignores_quotes_without_text():
    chunks = [{"text": "valid chunk", "metadata": {"accession": "A"}}]
    quotes = [
        {"text": ""},
        {"text": None},
        {"text": "not in any chunk and very long"},
    ]
    result = check_faithfulness(quotes, [], chunks)
    # Only the third quote counts
    assert result.quotes_total == 1


# --- Judge label parsing (Tier 2 prompt-engineering correctness) ------------


def test_label_to_score_maps_canonical_labels():
    from utils.rag_eval import _label_to_score

    assert _label_to_score("NONE") == 0
    assert _label_to_score("WEAK") == 1
    assert _label_to_score("PARTIAL") == 2
    assert _label_to_score("HIGH") == 3


def test_label_to_score_normalises_whitespace_and_case():
    from utils.rag_eval import _label_to_score

    assert _label_to_score(" partial ") == 2
    assert _label_to_score("high") == 3


def test_label_to_score_unknown_label_defaults_to_zero():
    """A garbled judge response should be treated as not-relevant — failsafe."""
    from utils.rag_eval import _label_to_score

    assert _label_to_score("MAYBE") == 0
    assert _label_to_score("") == 0
    assert _label_to_score(None) == 0  # type: ignore[arg-type]


def test_strip_judge_fences_handles_markdown_wrapped_json():
    from utils.rag_eval import _strip_judge_fences

    raw = '```json\n{"rationale": "x", "label": "HIGH"}\n```'
    assert _strip_judge_fences(raw) == '{"rationale": "x", "label": "HIGH"}'


def test_judge_one_chunk_parses_label_first_format(monkeypatch):
    """Validates the new prompt's expected response shape — rationale BEFORE label."""
    import utils.rag_eval as re

    class _FakeResp:
        class _Msg:
            content = '{"rationale": "Directly addresses Hopper demand drivers", "label": "HIGH"}'

        choices = [type("C", (), {"message": _Msg})()]

    class _FakeChat:
        def create(self, **kw):
            return _FakeResp()

    class _FakeClient:
        chat = type("CC", (), {"completions": _FakeChat()})()

    monkeypatch.setattr(re, "get_client", lambda: _FakeClient())
    label, score, rationale = re._judge_one_chunk("question", "chunk text")
    assert label == "HIGH"
    assert score == 3
    assert "Hopper" in rationale


def test_judge_one_chunk_handles_legacy_score_format(monkeypatch):
    """Older deployments may still emit `{"score": int}` — must back-translate
    to label so downstream metrics still work."""
    import utils.rag_eval as re

    class _FakeResp:
        class _Msg:
            content = '{"score": 2, "rationale": "okay match"}'

        choices = [type("C", (), {"message": _Msg})()]

    class _FakeClient:
        chat = type(
            "CC", (), {"completions": type("X", (), {"create": lambda self, **kw: _FakeResp()})()}
        )()

    monkeypatch.setattr(re, "get_client", lambda: _FakeClient())
    label, score, rationale = re._judge_one_chunk("q", "c")
    # score=2 → PARTIAL
    assert label == "PARTIAL"
    assert score == 2


def test_judge_one_chunk_falls_back_on_unparseable_response(monkeypatch):
    import utils.rag_eval as re

    class _FakeResp:
        class _Msg:
            content = "this is not JSON"

        choices = [type("C", (), {"message": _Msg})()]

    class _FakeClient:
        chat = type(
            "CC", (), {"completions": type("X", (), {"create": lambda self, **kw: _FakeResp()})()}
        )()

    monkeypatch.setattr(re, "get_client", lambda: _FakeClient())
    label, score, rationale = re._judge_one_chunk("q", "c")
    assert label == "NONE"
    assert score == 0
    assert "unparseable" in rationale


def test_judge_relevance_aggregates_metrics_from_judge_scores(monkeypatch):
    """End-to-end with mocked _judge_one_chunk: precision@K, NDCG@K, MRR all
    computed from the integer scores derived from labels."""
    import utils.rag_eval as re

    # Pretend the judge labels each chunk in turn: HIGH, NONE, PARTIAL, NONE
    fake_outputs = iter(
        [
            ("HIGH", 3, "rel"),
            ("NONE", 0, "off-topic"),
            ("PARTIAL", 2, "partial"),
            ("NONE", 0, "off-topic"),
        ]
    )
    monkeypatch.setattr(re, "_judge_one_chunk", lambda q, t: next(fake_outputs))

    chunks = [{"text": f"chunk {i}"} for i in range(4)]
    report = re.judge_relevance("any question", chunks)

    assert [s.label for s in report.scores] == ["HIGH", "NONE", "PARTIAL", "NONE"]
    # 2 of 4 are relevant (HIGH + PARTIAL)
    assert report.precision_at_k == 0.5
    # MRR: first relevant chunk at rank 1
    assert report.mrr == 1.0
    # NDCG should be < 1 because the worst chunk (NONE) is sandwiched between relevant ones
    assert 0.0 < report.ndcg_at_k < 1.0
