"""Step 5d unit tests — Risk agent prompt assembly + schema coercion + failure paths.

Pure-logic, no network. Real LLM in test_risk_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.risk import (
    FALLBACK_RISK,
    _build_user_prompt,
    _coerce_to_risk_output,
    _strip_code_fences,
    _summarise_worker,
    run,
)
from utils.schemas import RISK_LEVEL_TO_SCORE, RiskOutput

THESES_DIR = Path(__file__).parents[1] / "theses"


# --- Worker summarisation ---------------------------------------------------


def test_summarise_worker_renders_fundamentals_kpis():
    payload = {
        "summary": "NVDA looks strong",
        "kpis": {"fcf_yield": 1.84, "revenue_5y_cagr": 1.0},
    }
    out = _summarise_worker("fundamentals", payload)
    assert "summary" in out
    assert "FUNDAMENTALS" in out
    assert "fcf_yield" in out


def test_summarise_worker_renders_filings_quotes():
    payload = {
        "summary": "Risk factors discuss supply chain",
        "risk_themes": ["supply concentration", "export controls"],
        "mdna_quotes": [
            {"text": "Supply constraints persist...", "item": "Item 7"},
        ],
    }
    out = _summarise_worker("filings", payload)
    assert "supply concentration" in out
    assert "Supply constraints" in out


def test_summarise_worker_renders_news_with_sentiment():
    payload = {
        "summary": "Mixed signals",
        "catalysts": [{"title": "Catalyst A", "sentiment": "bull", "as_of": "2026-04-20"}],
        "concerns": [{"title": "Concern B", "sentiment": "bear", "as_of": "2026-04-19"}],
    }
    out = _summarise_worker("news", payload)
    assert "[bull] Catalyst A" in out
    assert "[bear] Concern B" in out


def test_summarise_worker_handles_empty_payload():
    """When a worker failed entirely, the prompt section says so explicitly."""
    out = _summarise_worker("filings", {})
    assert "no output" in out


# --- Prompt assembly --------------------------------------------------------


def test_build_user_prompt_includes_all_three_workers():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {
        "ticker": "NVDA",
        "thesis": thesis,
        "fundamentals": {"summary": "fundies", "kpis": {"fcf_yield": 1.8}},
        "filings": {"summary": "filings", "risk_themes": ["X"]},
        "news": {"summary": "news"},
    }
    prompt = _build_user_prompt("NVDA", thesis, state)
    assert "FUNDAMENTALS" in prompt
    assert "FILINGS" in prompt
    assert "NEWS" in prompt
    assert "STRICT JSON" in prompt
    assert "AI cake" in prompt


def test_build_user_prompt_includes_thesis_thresholds():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis, "fundamentals": {"summary": "x"}}
    prompt = _build_user_prompt("NVDA", thesis, state)
    # Thresholds rendered as JSON
    assert "fcf_yield" in prompt or "data_center_capex" in prompt


# --- _strip_code_fences -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"level":"LOW"}', '{"level":"LOW"}'),
        ('```json\n{"level":"LOW"}\n```', '{"level":"LOW"}'),
        ('```\n{"a":1}\n```', '{"a":1}'),
    ],
)
def test_strip_code_fences(raw, expected):
    assert _strip_code_fences(raw) == expected


# --- _coerce_to_risk_output -------------------------------------------------


def test_coerce_derives_score_from_level():
    raw = {
        "level": "ELEVATED",
        "summary": "x",
        "top_risks": [],
        "convergent_signals": [],
        "threshold_breaches": [],
    }
    out = _coerce_to_risk_output(raw)
    assert out["level"] == "ELEVATED"
    assert out["score_0_to_10"] == RISK_LEVEL_TO_SCORE["ELEVATED"]


def test_coerce_normalises_lowercase_level():
    """LLMs sometimes emit `low` instead of `LOW` despite instructions."""
    raw = {"level": "low", "summary": "x", "top_risks": []}
    out = _coerce_to_risk_output(raw)
    assert out["level"] == "LOW"
    assert out["score_0_to_10"] == 2


def test_coerce_defaults_to_moderate_on_unknown_level():
    """Unknown labels are coerced to MODERATE — failsafe for noisy LLM output."""
    raw = {"level": "DEFINITELY_BAD", "summary": "x", "top_risks": []}
    out = _coerce_to_risk_output(raw)
    assert out["level"] == "MODERATE"


def test_coerce_handles_missing_optional_fields():
    raw = {"level": "HIGH", "summary": "x", "top_risks": []}
    out = _coerce_to_risk_output(raw)
    assert out["convergent_signals"] == []
    assert out["threshold_breaches"] == []


# --- run() failure paths ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_fallback_when_all_workers_empty():
    """Risk has nothing to synthesize → return graceful fallback."""
    state = {
        "ticker": "ZZZZ",
        "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text()),
        "fundamentals": {},
        "filings": {},
        "news": {},
    }
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])
    assert out.level == "MODERATE"  # fallback level
    assert out.score_0_to_10 == 4
    assert any("upstream worker" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_returns_fallback_when_llm_fails(monkeypatch):
    """LLM raises but worker outputs exist — fallback keeps graph alive."""
    from agents import risk as r

    monkeypatch.setattr(
        r,
        "_call_llm",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("openrouter outage")),
    )
    state = {
        "ticker": "NVDA",
        "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text()),
        "fundamentals": {"summary": "x"},
        "filings": {"summary": "y"},
        "news": {"summary": "z"},
    }
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])
    assert out.level == "MODERATE"
    assert any("llm" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_propagates_llm_output_when_call_succeeds(monkeypatch):
    """Happy path: mocked LLM returns a valid level → derived score + structured fields."""
    from agents import risk as r

    fake_llm_out = {
        "rationale": "Multiple signals across sources",
        "level": "ELEVATED",
        "summary": "[stub] elevated risk",
        "top_risks": [
            {
                "title": "supply concentration",
                "severity": 4,
                "explanation": "Multiple sources flagged",
                "sources": ["fundamentals", "news"],
            }
        ],
        "convergent_signals": [
            {
                "theme": "supply concentration",
                "sources": ["fundamentals", "news"],
                "explanation": "Both flag the same risk",
            }
        ],
        "threshold_breaches": [
            {
                "signal": "fcf_yield",
                "operator": "<",
                "threshold_value": 4,
                "observed_value": 1.84,
                "explanation": "Below MoS threshold",
                "source": "fundamentals",
            }
        ],
    }
    # Patch the inner LLM call (not the coerce step) so we test the full pipeline
    monkeypatch.setattr(r, "_call_llm", lambda *a, **kw: r._coerce_to_risk_output(fake_llm_out))

    state = {
        "ticker": "NVDA",
        "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text()),
        "fundamentals": {"summary": "x"},
        "filings": {"summary": "y"},
        "news": {"summary": "z"},
    }
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])

    assert out.level == "ELEVATED"
    assert out.score_0_to_10 == 6  # canonical mapping
    assert len(out.convergent_signals) == 1
    assert out.threshold_breaches[0].observed_value == 1.84
    assert out.errors == []


# --- Fallback shape validity ------------------------------------------------


def test_fallback_risk_validates_against_schema():
    """The hardcoded FALLBACK_RISK must always validate so we never crash on
    its use as a graceful-degradation path."""
    out = RiskOutput.model_validate(FALLBACK_RISK)
    assert out.level == "MODERATE"
    assert out.score_0_to_10 == 4
