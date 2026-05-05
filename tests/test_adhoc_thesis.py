"""Tier 1 unit tests for `agents/adhoc_thesis.py` — Discovery-lite
synthesizer (Step 10e).

We mock the OpenRouter call so the suite runs without network. Tier 3
real-LLM smoke lives in `tests/test_adhoc_thesis_integration.py` (gated
behind `pytest -m integration`) — TODO when we add it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents import adhoc_thesis as at
from utils.schemas import Thesis


# --- Slug derivation ------------------------------------------------------


def test_adhoc_slug_from_topic_normalises_whitespace_and_case():
    """The Telegram handler may pass "Defense Semis" or "DEFENSE SEMIS";
    both must map to the same canonical slug so cache hits work across
    capitalisation variations."""
    assert at.adhoc_slug(topic="defense semis") == "adhoc_defense_semis"
    assert at.adhoc_slug(topic="Defense Semis") == "adhoc_defense_semis"
    assert at.adhoc_slug(topic="DEFENSE-SEMIS") == "adhoc_defense_semis"
    assert at.adhoc_slug(topic="defense  semis") == "adhoc_defense_semis"


def test_adhoc_slug_from_ticker_lowercased():
    assert at.adhoc_slug(ticker="AAPL") == "adhoc_aapl"
    assert at.adhoc_slug(ticker="aapl") == "adhoc_aapl"


def test_adhoc_slug_truncates_to_filesystem_safe_length():
    """Long topic strings shouldn't blow past filesystem name limits."""
    long = "the future of artificial general intelligence in healthcare 2030"
    slug = at.adhoc_slug(topic=long)
    # `adhoc_` + max 40 chars from the slug body = 46 chars — well under
    # any filesystem limit.
    assert len(slug) <= 50
    assert slug.startswith("adhoc_")


# --- Validation + persistence ---------------------------------------------


_VALID_THESIS_JSON = {
    "name": "Defense semis (ad-hoc)",
    "summary": "Defense-exposed semis with multi-year backlog visibility.",
    "anchor_tickers": ["MRCY", "KTOS"],
    "universe": ["MRCY", "KTOS", "LMT", "RTX", "NOC"],
    "relationships": [
        {"from": "LMT", "to": "MRCY", "type": "customer", "note": "F-35 boards"}
    ],
    "valuation": {
        "equity_risk_premium": 0.055,
        "erp_basis": "S&P 500 long-run + 0.5pp for cyclical defense risk",
        "terminal_growth_rate": 0.030,
        "terminal_growth_basis": "Tracks DoD modernisation budget growth",
        "discount_rate_floor": 0.080,
        "discount_rate_cap": 0.140,
    },
    "material_thresholds": [
        {"signal": "backlog_growth_yoy", "operator": ">", "value": 20, "unit": "percent"},
        {"signal": "roe_ttm", "operator": "<", "value": 12, "unit": "percent"},
        {"signal": "filing_mentions", "operator": "contains", "value": "going concern", "unit": "text"},
    ],
}


def _stub_llm_returning(payload: dict | str, monkeypatch):
    """Mock the OpenRouter client so `synthesize_adhoc_thesis` doesn't
    issue a real network call."""
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

    client = SimpleNamespace(chat=_Chat())
    monkeypatch.setattr(at, "get_client", lambda: client)


def _redirect_theses_dir(tmp_path, monkeypatch):
    """Persistence tests mustn't pollute the real /theses/ dir."""
    monkeypatch.setattr(at, "THESES_DIR", tmp_path)


# --- synthesize_adhoc_thesis ---------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_topic_writes_to_disk(tmp_path, monkeypatch):
    """Happy path: TOPIC mode → LLM returns valid JSON → validates →
    writes to `theses/adhoc_{slug}.json`."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    _stub_llm_returning(_VALID_THESIS_JSON, monkeypatch)
    result = await at.synthesize_adhoc_thesis(topic="defense semis")
    assert result.error is None
    assert result.slug == "adhoc_defense_semis"
    assert result.cached is False
    assert result.path == tmp_path / "adhoc_defense_semis.json"
    assert result.path.exists()
    # On-disk JSON must be re-loadable as a Thesis
    Thesis.model_validate_json(result.path.read_text())


@pytest.mark.asyncio
async def test_synthesize_ticker_mode(tmp_path, monkeypatch):
    """TICKER mode: ticker is the input, slug uses `adhoc_aapl`."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    _stub_llm_returning(_VALID_THESIS_JSON, monkeypatch)
    result = await at.synthesize_adhoc_thesis(ticker="AAPL")
    assert result.error is None
    assert result.slug == "adhoc_aapl"


@pytest.mark.asyncio
async def test_synthesize_cache_hit_skips_llm(tmp_path, monkeypatch):
    """Second call for the same topic must read from disk, not re-call
    the LLM. Caching at the synthesizer level is the whole point of
    the on-disk persistence — without it, every NL "analyze defense semis"
    pays $0.05 again."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    # Pre-populate the cache with a valid thesis.
    cached_path = tmp_path / "adhoc_defense_semis.json"
    cached_path.write_text(json.dumps(_VALID_THESIS_JSON, indent=2))

    # Stub LLM so we can detect it was called.
    call_count = {"n": 0}

    def _track(**kwargs):
        call_count["n"] += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )

    class _Stub:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _track(**kwargs)

    monkeypatch.setattr(at, "get_client", lambda: _Stub())

    result = await at.synthesize_adhoc_thesis(topic="defense semis")
    assert result.cached is True
    assert call_count["n"] == 0, "cache hit MUST NOT call the LLM"
    assert result.thesis.name == "Defense semis (ad-hoc)"


@pytest.mark.asyncio
async def test_synthesize_force_refresh_bypasses_cache(tmp_path, monkeypatch):
    """`force_refresh=True` re-synthesizes even when the cache is fresh.
    Used by `/analyze TOPIC --refresh` if we add that flag later."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    cached_path = tmp_path / "adhoc_defense_semis.json"
    cached_path.write_text(json.dumps(_VALID_THESIS_JSON, indent=2))

    new_payload = {**_VALID_THESIS_JSON, "name": "Defense semis (REFRESHED)"}
    _stub_llm_returning(new_payload, monkeypatch)
    result = await at.synthesize_adhoc_thesis(topic="defense semis", force_refresh=True)
    assert result.cached is False
    assert result.thesis.name == "Defense semis (REFRESHED)"


@pytest.mark.asyncio
async def test_synthesize_strips_code_fences(tmp_path, monkeypatch):
    """Some models wrap JSON in ```json ... ``` despite explicit
    instruction. The parser must recover."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    raw = "```json\n" + json.dumps(_VALID_THESIS_JSON) + "\n```"
    _stub_llm_returning(raw, monkeypatch)
    result = await at.synthesize_adhoc_thesis(topic="defense semis")
    assert result.error is None
    assert result.thesis is not None


@pytest.mark.asyncio
async def test_synthesize_handles_vague_input_error_marker(tmp_path, monkeypatch):
    """The system prompt instructs the LLM to return `{"error": "..."}`
    for vague inputs. The synthesizer must surface that as `result.error`
    rather than a Pydantic validation crash."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    _stub_llm_returning({"error": "input too vague", "_input": "stocks"}, monkeypatch)
    result = await at.synthesize_adhoc_thesis(topic="stocks")
    assert result.thesis is None
    assert result.error is not None
    assert "vague" in result.error.lower()


@pytest.mark.asyncio
async def test_synthesize_handles_invalid_thesis_json(tmp_path, monkeypatch):
    """If the LLM returns valid JSON that doesn't satisfy the Thesis
    schema (e.g. anchors not in universe), surface it as an error
    rather than crash."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    bad = {**_VALID_THESIS_JSON, "anchor_tickers": ["NOT_IN_UNIVERSE"]}
    _stub_llm_returning(bad, monkeypatch)
    result = await at.synthesize_adhoc_thesis(topic="x")
    assert result.thesis is None
    assert result.error is not None


@pytest.mark.asyncio
async def test_synthesize_handles_non_json_response(tmp_path, monkeypatch):
    """LLM emits prose instead of JSON → graceful failure, no crash.
    Error message must include a snippet of the raw response so the
    user gets actionable signal, not just 'non-JSON response'."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(at, "_RAW_FAIL_DIR", tmp_path / "fails")
    raw = "Sure, here's a thesis…"
    _stub_llm_returning(raw, monkeypatch)
    result = await at.synthesize_adhoc_thesis(topic="x")
    assert result.thesis is None
    assert result.error is not None
    # Error must include a snippet of the raw response so the user sees
    # what came back without having to dig in logs.
    assert "Sure" in result.error or raw[:50] in result.error
    # The full raw response must be stashed on disk for offline inspection.
    stashed = list((tmp_path / "fails").glob("*x*.txt"))
    assert len(stashed) == 1
    assert stashed[0].read_text() == raw


@pytest.mark.asyncio
async def test_synthesize_truncated_json_surfaces_truncation_hint(
    tmp_path, monkeypatch
):
    """When the LLM produced JSON-ish output that got cut off
    (e.g. response hit max_tokens mid-output), the error message should
    say so explicitly — that's a different failure mode from "model
    refused to return JSON" and the fix is bumping LLM_MAX_TOKENS."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(at, "_RAW_FAIL_DIR", tmp_path / "fails")
    # Looks JSON-ish but missing the closing brace.
    truncated = '{"name": "Defense", "summary": "Some text", "anchor_ticke'
    _stub_llm_returning(truncated, monkeypatch)
    result = await at.synthesize_adhoc_thesis(topic="defense")
    assert result.thesis is None
    assert result.error is not None
    # The hint that bumping LLM_MAX_TOKENS or shortening the topic
    # might fix it must appear so the user has an actionable next step.
    assert (
        "truncat" in result.error.lower()
        or "LLM_MAX_TOKENS" in result.error
    )


@pytest.mark.asyncio
async def test_synthesize_handles_llm_call_failure(tmp_path, monkeypatch):
    """OpenRouter outage → the synthesizer catches and surfaces the
    error string. `analyze_command` shows the user a friendly message
    rather than a stack trace."""
    _redirect_theses_dir(tmp_path, monkeypatch)

    class _BoomClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("openrouter 502")

    monkeypatch.setattr(at, "get_client", lambda: _BoomClient())
    result = await at.synthesize_adhoc_thesis(topic="x")
    assert result.thesis is None
    assert result.error is not None
    assert "LLM call failed" in result.error or "502" in result.error


@pytest.mark.asyncio
async def test_synthesize_rejects_missing_inputs(tmp_path, monkeypatch):
    _redirect_theses_dir(tmp_path, monkeypatch)
    result = await at.synthesize_adhoc_thesis()
    assert result.error is not None
    assert "required" in result.error.lower()


@pytest.mark.asyncio
async def test_synthesize_rejects_both_inputs(tmp_path, monkeypatch):
    """Both topic AND ticker is ambiguous — the prompt expects one or
    the other, not both."""
    _redirect_theses_dir(tmp_path, monkeypatch)
    result = await at.synthesize_adhoc_thesis(topic="x", ticker="AAPL")
    assert result.error is not None


# --- Prompt sanity --------------------------------------------------------


def test_prompt_loads_at_import_time():
    """If the prompt file is missing or unreadable, the module fails to
    import. Pin that the prompt is non-empty + mentions both modes."""
    assert at._SYSTEM_PROMPT
    assert "TOPIC mode" in at._SYSTEM_PROMPT
    assert "TICKER mode" in at._SYSTEM_PROMPT
    # Schema mention so the LLM knows what shape to emit.
    assert "valuation" in at._SYSTEM_PROMPT
    assert "material_thresholds" in at._SYSTEM_PROMPT


def test_disk_persistence_uses_theses_dir():
    """Pin that adhoc theses land in /theses/ (not a separate dir).
    The runner reads from /theses/{slug}.json, so anything else breaks
    `/drill TICKER adhoc_slug`."""
    assert at.THESES_DIR == Path("theses")
    assert at.ADHOC_PREFIX == "adhoc_"
