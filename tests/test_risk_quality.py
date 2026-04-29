"""Step 5d Risk-quality eval — Tier 1 (deterministic, always-on).

Three test categories:
  - Schema-level invariants (level↔score consistency, severity 1-5, sources valid)
  - Threshold-breach correctness (every fired breach references a real signal in
    the thesis JSON; observed_value type matches operator)
  - Source-attribution check (every top_risk lists at least one valid worker)

Tier 2 (LLM-judge) is in test_risk_eval_llm.py — gated `pytest -m eval`.
Tier 3 (RAGAS faithfulness) is in test_risk_eval_ragas.py — also gated.

Eval results persist to data_cache/eval/runs/ for the Mission Control panel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.risk import run
from utils.rag_eval import write_eval_run
from utils.schemas import OPERATORS, RISK_LEVEL_TO_SCORE, RiskOutput

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")


def _stub_state(ticker: str, thesis: dict) -> dict:
    """Same fixture shape as test_risk_integration. Kept lightweight so this
    test exercises Risk's synthesis logic without depending on slow upstream
    workers."""
    return {
        "ticker": ticker,
        "thesis": thesis,
        "fundamentals": {
            "summary": f"{ticker}: revenue CAGR ~100% but FCF yield 1.8% (threshold <4%).",
            "kpis": {
                "revenue_5y_cagr": 1.0,
                "fcf_yield": 1.8,
                "operating_margin_5yr_avg": 0.49,
                "pe_trailing": 44.3,
            },
            "projections": {
                "revenue_growth_mean": 0.40,
                "revenue_growth_std": 0.10,
                "margin_mean": 0.65,
                "margin_std": 0.05,
                "exit_multiple_mean": 35.0,
                "exit_multiple_std": 5.0,
            },
        },
        "filings": {
            "summary": "10-K cites supply constraints, export-control risk.",
            "risk_themes": ["supply concentration", "export-control regulation"],
            "mdna_quotes": [],
        },
        "news": {
            "summary": "Mixed — strong Q4 offset by valuation premium concerns.",
            "catalysts": [
                {"title": "Q4 earnings beat", "sentiment": "bull", "as_of": "2026-03-25"}
            ],
            "concerns": [
                {"title": "Valuation premium fading", "sentiment": "bear", "as_of": "2026-03-29"}
            ],
        },
    }


# --- Schema invariants (already enforced at Pydantic level; here we record metrics)


@pytest.mark.asyncio
async def test_risk_score_matches_level_canonical_mapping():
    """The model_validator already rejects mismatches; this test asserts that
    a real Risk run produces a consistent (level, score) pair."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = _stub_state("NVDA", thesis)
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])
    assert out.score_0_to_10 == RISK_LEVEL_TO_SCORE[out.level]


@pytest.mark.asyncio
async def test_every_top_risk_has_valid_source():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = _stub_state("NVDA", thesis)
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])

    valid = {"fundamentals", "filings", "news"}
    for r in out.top_risks:
        assert r.sources, f"top_risk has empty sources: {r.title}"
        for s in r.sources:
            assert s in valid, f"unknown source {s!r} on top_risk {r.title!r}"


# --- Threshold-breach correctness ------------------------------------------


@pytest.mark.asyncio
async def test_threshold_breaches_reference_real_thesis_signals():
    """Every threshold_breach.signal must correspond to a `signal` declared in
    the thesis's `material_thresholds`. Catches LLM fabrication of signal names."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = _stub_state("NVDA", thesis)
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])

    declared_signals = {t["signal"] for t in thesis["material_thresholds"]}
    for breach in out.threshold_breaches:
        assert breach.signal in declared_signals, (
            f"threshold_breach.signal {breach.signal!r} not declared in thesis "
            f"material_thresholds: {sorted(declared_signals)}"
        )


@pytest.mark.asyncio
async def test_threshold_breach_operators_are_valid():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = _stub_state("NVDA", thesis)
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])

    for breach in out.threshold_breaches:
        assert breach.operator in OPERATORS, f"bad operator {breach.operator!r}"


# --- Aggregate summary (Mission Control row) -------------------------------


@pytest.mark.asyncio
async def test_risk_quality_summary_recorded_to_eval_dir():
    """One Risk run, persist a structured snapshot to data_cache/eval/runs/
    so the Mission Control panel can render the latest level + counts."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = _stub_state("NVDA", thesis)
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])

    declared_signals = {t["signal"] for t in thesis["material_thresholds"]}
    valid_sources = {"fundamentals", "filings", "news"}

    invalid_breaches = [b for b in out.threshold_breaches if b.signal not in declared_signals]
    invalid_sources = [
        r for r in out.top_risks if not r.sources or any(s not in valid_sources for s in r.sources)
    ]

    write_eval_run(
        {
            "tier": 1,
            "suite": "risk_quality",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "level": out.level,
            "score_0_to_10": out.score_0_to_10,
            "top_risks_count": len(out.top_risks),
            "convergent_signals_count": len(out.convergent_signals),
            "threshold_breaches_count": len(out.threshold_breaches),
            "invalid_breaches": [b.signal for b in invalid_breaches],
            "top_risks_missing_sources": [r.title for r in invalid_sources],
        }
    )

    # All schema-level invariants must hold
    assert not invalid_breaches, f"breaches reference unknown signals: {invalid_breaches}"
    assert not invalid_sources, f"top_risks with missing/invalid sources: {invalid_sources}"
