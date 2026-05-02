"""Tier 1 unit tests for agents/router.py — Telegram NL intent classifier.

We mock OpenRouter so the suite runs without network. Tier 3 (real LLM)
belongs in a future `test_router_integration.py` gated behind
`pytest -m integration`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agents import router as r
from utils.schemas import RouterDecision


# --- Helpers ---------------------------------------------------------------


def _stub_client_returning(payload: Any) -> SimpleNamespace:
    """Build a minimal OpenRouter-shaped client whose chat.completions.create
    returns `payload` (str or dict — dicts get json.dumped) as the LLM message
    content."""
    if isinstance(payload, dict):
        content = json.dumps(payload)
    else:
        content = str(payload)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )

    class _ChatCompletions:
        def create(self, **kwargs):
            return response

    class _Chat:
        completions = _ChatCompletions()

    return SimpleNamespace(chat=_Chat())


# --- Constants & basic shape ----------------------------------------------


def test_threshold_is_zero_point_seven():
    """The threshold is documented at 0.7 in docs/POSTPONED.md §2.
    Pin it so a future edit can't silently drift."""
    assert r.ROUTER_CONFIDENCE_THRESHOLD == 0.7


def test_should_dispatch_above_threshold(monkeypatch):
    decision = RouterDecision(intent="drill", args={"ticker": "NVDA"}, confidence=0.9)
    assert r.should_dispatch(decision) is True


def test_should_dispatch_below_threshold(monkeypatch):
    decision = RouterDecision(intent="drill", args={"ticker": "NVDA"}, confidence=0.5)
    assert r.should_dispatch(decision) is False


def test_should_dispatch_unknown_never_dispatches():
    """Even at confidence 1.0, an `unknown` intent is never dispatched —
    the bot has nowhere to route it."""
    decision = RouterDecision(intent="unknown", args={}, confidence=1.0)
    assert r.should_dispatch(decision) is False


def test_should_dispatch_at_exactly_threshold():
    """0.7 is the inclusive lower bound — the >= ensures a 0.7 response
    actually dispatches, not gets clarified."""
    decision = RouterDecision(intent="drill", args={"ticker": "X"}, confidence=0.7)
    assert r.should_dispatch(decision) is True


# --- classify() — happy paths ---------------------------------------------


@pytest.mark.asyncio
async def test_classify_drill_with_ticker(monkeypatch):
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning(
            {"intent": "drill", "args": {"ticker": "NVDA"}, "confidence": 0.92}
        ),
    )
    decision = await r.classify("what's NVDA looking like")
    assert decision.intent == "drill"
    assert decision.args == {"ticker": "NVDA"}
    assert decision.confidence == pytest.approx(0.92)
    assert r.should_dispatch(decision) is True


@pytest.mark.asyncio
async def test_classify_analyze_with_topic(monkeypatch):
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning(
            {
                "intent": "analyze",
                "args": {"topic": "defense semis"},
                "confidence": 0.95,
            }
        ),
    )
    decision = await r.classify("analyze defense semis")
    assert decision.intent == "analyze"
    assert decision.args == {"topic": "defense semis"}
    assert decision.confidence == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_classify_note_with_thesis_and_text(monkeypatch):
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning(
            {
                "intent": "note",
                "args": {"thesis": "ai_cake", "text": "trim 20% if Q3 misses $42B"},
                "confidence": 0.85,
            }
        ),
    )
    decision = await r.classify("trim my AI cake by 20% if Q3 misses 42B")
    assert decision.intent == "note"
    assert decision.args["thesis"] == "ai_cake"
    assert "trim" in decision.args["text"]


@pytest.mark.asyncio
async def test_classify_status_no_args(monkeypatch):
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning(
            {"intent": "status", "args": {}, "confidence": 1.0}
        ),
    )
    decision = await r.classify("status")
    assert decision.intent == "status"
    assert decision.args == {}


# --- classify() — robustness ----------------------------------------------


@pytest.mark.asyncio
async def test_classify_empty_input_returns_unknown(monkeypatch):
    """Empty / whitespace input never reaches the LLM. Returns unknown
    immediately so the bot can short-circuit to a clarification reply."""
    called = {"n": 0}

    def _get_client():
        called["n"] += 1
        return _stub_client_returning({})

    monkeypatch.setattr(r, "get_client", _get_client)
    decision = await r.classify("   ")
    assert decision.intent == "unknown"
    assert decision.confidence == 0.0
    assert called["n"] == 0, "empty input must not issue an LLM call"


@pytest.mark.asyncio
async def test_classify_strips_code_fences(monkeypatch):
    """Some models wrap JSON in ```json ... ``` despite the prompt. The
    parser must recover the JSON inside."""
    raw = '```json\n{"intent": "scan", "args": {}, "confidence": 0.85}\n```'
    monkeypatch.setattr(r, "get_client", lambda: _stub_client_returning(raw))
    decision = await r.classify("anything new today")
    assert decision.intent == "scan"
    assert decision.confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_classify_recovers_json_from_chatty_response(monkeypatch):
    """Defensive: when the LLM emits prose around the JSON, the regex
    fallback finds the JSON object and parses it."""
    raw = (
        'Sure, here you go:\n'
        '{"intent": "drill", "args": {"ticker": "AVGO"}, "confidence": 0.9}\n'
        'Hope that helps!'
    )
    monkeypatch.setattr(r, "get_client", lambda: _stub_client_returning(raw))
    decision = await r.classify("drill AVGO")
    assert decision.intent == "drill"
    assert decision.args == {"ticker": "AVGO"}


@pytest.mark.asyncio
async def test_classify_invalid_intent_falls_to_unknown(monkeypatch):
    """If the LLM hallucinates an intent outside the enum, Pydantic
    validation rejects it and we fall back to unknown rather than crash."""
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning(
            {"intent": "buy_stock", "args": {"ticker": "NVDA"}, "confidence": 0.9}
        ),
    )
    decision = await r.classify("buy NVDA for me")
    assert decision.intent == "unknown"
    assert decision.confidence == 0.0


@pytest.mark.asyncio
async def test_classify_invalid_confidence_falls_to_unknown(monkeypatch):
    """Confidence outside [0, 1] is invalid — fall back to unknown."""
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning(
            {"intent": "drill", "args": {"ticker": "NVDA"}, "confidence": 1.5}
        ),
    )
    decision = await r.classify("drill NVDA")
    assert decision.intent == "unknown"


@pytest.mark.asyncio
async def test_classify_non_json_response_falls_to_unknown(monkeypatch):
    """A fully prose response (router prompt drift) → unknown, no crash."""
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning("I think you want to drill into NVDA"),
    )
    decision = await r.classify("what's NVDA looking like")
    assert decision.intent == "unknown"
    assert decision.confidence == 0.0


@pytest.mark.asyncio
async def test_classify_openrouter_error_falls_to_unknown(monkeypatch):
    """OpenRouter outage / network error → unknown, no crash. The bot
    handler will see this and reply with a graceful error instead of
    the user seeing a stack trace."""

    class _BoomClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("openrouter 502")

    monkeypatch.setattr(r, "get_client", lambda: _BoomClient())
    decision = await r.classify("drill NVDA")
    assert decision.intent == "unknown"


@pytest.mark.asyncio
async def test_classify_drops_empty_arg_values(monkeypatch):
    """The LLM sometimes emits empty-string args. Coerce to absent — empty
    string would route as a real arg downstream and confuse the dispatcher."""
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning(
            {
                "intent": "drill",
                "args": {"ticker": "NVDA", "thesis": ""},
                "confidence": 0.9,
            }
        ),
    )
    decision = await r.classify("drill NVDA")
    assert decision.args == {"ticker": "NVDA"}


@pytest.mark.asyncio
async def test_classify_low_confidence_unknown_does_not_dispatch(monkeypatch):
    """End-to-end: ambiguous input → low-confidence unknown → bot would
    ask for clarification."""
    monkeypatch.setattr(
        r,
        "get_client",
        lambda: _stub_client_returning(
            {"intent": "unknown", "args": {}, "confidence": 0.1}
        ),
    )
    decision = await r.classify("hmm")
    assert decision.intent == "unknown"
    assert r.should_dispatch(decision) is False


# --- Prompt sanity ---------------------------------------------------------


def test_router_prompt_enumerates_all_intents():
    """The system prompt must list every intent the schema permits — drift
    between the two would mean either a missing intent or a hallucinated one."""
    prompt = r._SYSTEM_PROMPT
    for intent in ("drill", "analyze", "scan", "note", "thesis", "status", "help", "unknown"):
        assert f"`{intent}`" in prompt, (
            f"intent {intent!r} missing from agents/prompts/router.md"
        )


def test_router_prompt_mentions_threshold():
    """If we change the threshold and forget the prompt, the LLM may
    miscalibrate confidence. Pin that the prompt at least mentions 0.7."""
    assert "0.7" in r._SYSTEM_PROMPT, (
        "agents/prompts/router.md should mention the 0.7 dispatch threshold "
        "so the LLM calibrates confidence with that knowledge"
    )
