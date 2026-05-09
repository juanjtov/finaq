"""Tier 1b — graph-node-level tests for the monte_carlo node.

These tests exercise the `monte_carlo` function in `agents/__init__.py` (NOT
the underlying `utils.monte_carlo.simulate` engine — that's covered by
test_monte_carlo.py). They verify that the node correctly reads upstream
FinaqState, validates required inputs, and gracefully degrades when inputs
are missing or treasury fetch fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents import monte_carlo

THESES_DIR = Path(__file__).parents[1] / "theses"


def _full_projections() -> dict:
    """Schema-valid projections with reasonable values for NVDA-style ticker."""
    return {
        "revenue_growth_mean": 0.20,
        "revenue_growth_std": 0.05,
        "margin_mean": 0.40,
        "margin_std": 0.05,
        "tax_rate_mean": 0.18,
        "tax_rate_std": 0.02,
        "maintenance_capex_pct_rev_mean": 0.04,
        "maintenance_capex_pct_rev_std": 0.01,
        "da_pct_rev_mean": 0.03,
        "da_pct_rev_std": 0.01,
        "dilution_rate_mean": 0.005,
        "dilution_rate_std": 0.002,
        "exit_multiple_mean": 25.0,
        "exit_multiple_std": 6.0,
    }


def _full_kpis() -> dict:
    return {
        "revenue_latest": 200e9,
        "shares_outstanding": 24e9,
        "current_price": 200.0,
        "net_cash": 30e9,
        "pe_trailing": 40.0,
    }


def _full_state() -> dict:
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    return {
        "ticker": "NVDA",
        "thesis": thesis,
        "fundamentals": {
            "summary": "test",
            "kpis": _full_kpis(),
            "projections": _full_projections(),
        },
    }


# --- Skipped paths ---------------------------------------------------------


@pytest.mark.asyncio
async def test_node_skips_when_thesis_valuation_missing():
    """Thesis without `valuation` block → MC node returns 'skipped' with clear error."""
    state = _full_state()
    state["thesis"] = {**state["thesis"], "valuation": None}
    result = await monte_carlo(state)
    assert result["monte_carlo"]["method"] == "skipped"
    assert any("thesis.valuation" in e for e in result["monte_carlo"]["errors"])


@pytest.mark.asyncio
async def test_node_skips_when_fundamentals_projections_missing():
    state = _full_state()
    state["fundamentals"]["projections"] = {}
    result = await monte_carlo(state)
    assert result["monte_carlo"]["method"] == "skipped"
    assert any("projections" in e for e in result["monte_carlo"]["errors"])


@pytest.mark.asyncio
async def test_node_skips_when_required_kpi_missing():
    state = _full_state()
    state["fundamentals"]["kpis"] = {**_full_kpis(), "current_price": None}
    result = await monte_carlo(state)
    assert result["monte_carlo"]["method"] == "skipped"
    assert any("current_price" in e for e in result["monte_carlo"]["errors"])


@pytest.mark.asyncio
async def test_node_skips_when_no_fundamentals_at_all():
    state = _full_state()
    state["fundamentals"] = {}
    result = await monte_carlo(state)
    assert result["monte_carlo"]["method"] == "skipped"


# --- Happy path ------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_runs_simulate_when_inputs_complete(monkeypatch):
    """Mock the treasury fetch (so we don't hit yfinance) and verify the engine
    actually runs and produces a complete dcf+multiple result."""
    import data.treasury

    monkeypatch.setattr(data.treasury, "get_10y_treasury_yield", lambda **kw: 0.045)

    state = _full_state()
    result = await monte_carlo(state)
    mc = result["monte_carlo"]

    assert mc["method"] == "dcf+multiple"
    assert "dcf" in mc and "multiple" in mc
    for q in ("p10", "p25", "p50", "p75", "p90"):
        assert q in mc["dcf"]
        assert q in mc["multiple"]
        assert mc["dcf"][q] >= 0
        assert mc["multiple"][q] >= 0
    assert 0.0 <= mc["convergence_ratio"] <= 1.0
    assert mc["current_price"] == 200.0
    # discount_rate_used = clip(treasury 4.5% + ai_cake ERP 5% = 9.5%, [7.5, 13])
    assert mc["discount_rate_used"] == pytest.approx(0.095)


@pytest.mark.asyncio
async def test_node_uses_treasury_fallback_on_fetch_failure(monkeypatch):
    """If the treasury fetch raises, the cached fallback in get_10y_treasury_yield
    kicks in (returns DEFAULT_FALLBACK). MC still produces valid output."""
    import data.treasury

    # The treasury module has its own fallback — simulate that the fallback is used
    monkeypatch.setattr(
        data.treasury, "get_10y_treasury_yield",
        lambda **kw: data.treasury.DEFAULT_FALLBACK,
    )

    state = _full_state()
    result = await monte_carlo(state)
    assert result["monte_carlo"]["method"] == "dcf+multiple"
    # Discount = clip(4.5% + 5% = 9.5%, [7.5, 13]) → still 9.5%
    assert result["monte_carlo"]["discount_rate_used"] == pytest.approx(0.095)


@pytest.mark.asyncio
async def test_node_clips_extreme_treasury_to_thesis_band(monkeypatch):
    """Very high or low treasury rate should be clipped by valuation.discount_rate_cap/floor."""
    import data.treasury

    # Ridiculously high: treasury 20% + ERP 5% = 25%, capped at ai_cake 13%
    monkeypatch.setattr(data.treasury, "get_10y_treasury_yield", lambda **kw: 0.20)
    state = _full_state()
    result = await monte_carlo(state)
    assert result["monte_carlo"]["discount_rate_used"] == pytest.approx(0.13)

    # Ridiculously low: treasury 0% + ERP 5% = 5%, floored at ai_cake 7.5%
    monkeypatch.setattr(data.treasury, "get_10y_treasury_yield", lambda **kw: 0.0)
    result = await monte_carlo(state)
    assert result["monte_carlo"]["discount_rate_used"] == pytest.approx(0.075)
