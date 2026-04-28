"""Step 5a unit tests — Fundamentals KPI math + prompt assembly + failure paths.

Pure-logic, no network. Real Sonnet calls live in test_fundamentals_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.fundamentals import (
    NULL_HYPOTHESIS_PROJECTIONS,
    _build_user_prompt,
    _derive_fallback_projections,
    _strip_code_fences,
    compute_kpis,
    run,
)
from utils.schemas import FundamentalsOutput, Projections

THESES_DIR = Path(__file__).parents[1] / "theses"


# --- KPI computation ---------------------------------------------------------


def _fake_financials(
    *,
    revenues: list[float] | None = None,
    op_incomes: list[float] | None = None,
    gross_profits: list[float] | None = None,
    net_incomes: list[float] | None = None,
    fcfs: list[float] | None = None,
    capex: list[float] | None = None,
    info: dict | None = None,
    price: float | None = None,
) -> dict:
    """Build a yfinance-shaped fixture from per-year arrays (oldest first)."""
    n = max(len(x) for x in [revenues, op_incomes, gross_profits, net_incomes, fcfs, capex] if x)
    income_stmt = {}
    cash_flow = {}
    for i in range(n):
        date = f"{2020 + i}-12-31"
        income_stmt[date] = {}
        if revenues and i < len(revenues):
            income_stmt[date]["Total Revenue"] = revenues[i]
        if op_incomes and i < len(op_incomes):
            income_stmt[date]["Operating Income"] = op_incomes[i]
        if gross_profits and i < len(gross_profits):
            income_stmt[date]["Gross Profit"] = gross_profits[i]
        if net_incomes and i < len(net_incomes):
            income_stmt[date]["Net Income"] = net_incomes[i]
        cash_flow[date] = {}
        if fcfs and i < len(fcfs):
            cash_flow[date]["Free Cash Flow"] = fcfs[i]
        if capex and i < len(capex):
            cash_flow[date]["Capital Expenditure"] = capex[i]
    return {
        "income_stmt": income_stmt,
        "cash_flow": cash_flow,
        "balance_sheet": {},
        "info": info or {},
        "price_history_5y": {"2024-12-31": {"Close": price}} if price else {},
    }


def test_compute_kpis_handles_empty_input():
    assert compute_kpis({}) == {}


def test_compute_kpis_revenue_cagr_5_years():
    fin = _fake_financials(revenues=[100.0, 120, 145, 175, 210])
    kpis = compute_kpis(fin)
    # 100 → 210 over 4 year-on-year periods. CAGR = (210/100)^(1/4) - 1 ≈ 0.2037
    assert "revenue_5y_cagr" in kpis
    assert 0.19 < kpis["revenue_5y_cagr"] < 0.22


def test_compute_kpis_latest_margins():
    fin = _fake_financials(
        revenues=[100, 110, 120],
        gross_profits=[40, 50, 60],
        op_incomes=[10, 15, 24],
    )
    kpis = compute_kpis(fin)
    assert kpis["gross_margin_latest"] == pytest.approx(0.5)
    assert kpis["operating_margin_latest"] == pytest.approx(0.2)


def test_compute_kpis_fcf_yield_uses_market_cap():
    fin = _fake_financials(
        revenues=[100, 200],
        fcfs=[10e9, 20e9],
        info={"marketCap": 200e9},
    )
    kpis = compute_kpis(fin)
    # Latest FCF 20B / market cap 200B = 10% yield
    assert kpis["fcf_yield"] == pytest.approx(10.0)


def test_compute_kpis_capex_to_revenue_avg():
    fin = _fake_financials(
        revenues=[100, 200, 400],
        capex=[-10, -30, -100],
    )
    kpis = compute_kpis(fin)
    # Pcts: 10%, 15%, 25% → avg 16.67%
    assert kpis["capex_to_revenue_5yr_avg"] == pytest.approx(16.666, rel=1e-2)


def test_compute_kpis_fcf_to_net_income_5yr():
    fin = _fake_financials(
        revenues=[100, 200],
        fcfs=[80, 120],
        net_incomes=[100, 100],
    )
    kpis = compute_kpis(fin)
    # sum(fcf)=200 / sum(ni)=200 = 1.0
    assert kpis["fcf_to_net_income_5yr"] == pytest.approx(1.0)


def test_compute_kpis_does_not_include_keys_when_data_missing():
    """Real yfinance is patchy — KPIs should only appear when computable."""
    fin = _fake_financials(revenues=[100, 200], info={})
    kpis = compute_kpis(fin)
    assert "fcf_yield" not in kpis  # no marketCap
    assert "fcf_to_net_income_5yr" not in kpis  # no NI / FCF


def test_compute_kpis_ignores_zero_division_in_revenue_cagr():
    fin = _fake_financials(revenues=[0, 100, 200])
    kpis = compute_kpis(fin)
    # First revenue is zero → CAGR undefined, key absent
    assert "revenue_5y_cagr" not in kpis


# --- Prompt assembly ---------------------------------------------------------


def test_build_user_prompt_includes_ticker_thesis_kpis():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    prompt = _build_user_prompt("NVDA", thesis, {"revenue_5y_cagr": 0.45})
    assert "NVDA" in prompt
    assert "AI cake" in prompt
    assert "anchor_tickers" not in prompt  # we render the list, not the field name
    assert "MSFT" in prompt  # anchor present
    assert "0.45" in prompt
    assert "STRICT JSON" in prompt


# --- _strip_code_fences ------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"summary": "x"}', '{"summary": "x"}'),
        ('```json\n{"summary": "x"}\n```', '{"summary": "x"}'),
        ('```\n{"summary": "x"}\n```', '{"summary": "x"}'),
        ('  ```json\n{"a":1}\n```  ', '{"a":1}'),
    ],
)
def test_strip_code_fences(raw, expected):
    assert _strip_code_fences(raw) == expected


# --- run() failure paths -----------------------------------------------------


# --- _derive_fallback_projections -------------------------------------------


def test_derive_fallback_projections_empty_kpis_returns_null_hypothesis():
    """No history at all → use the generic-mediocre-business baseline."""
    p = _derive_fallback_projections({})
    assert p == NULL_HYPOTHESIS_PROJECTIONS


def test_derive_fallback_projections_uses_revenue_cagr_when_present():
    """A computed CAGR replaces the 5% default and tightens the std."""
    p = _derive_fallback_projections({"revenue_5y_cagr": 0.32})
    assert p.revenue_growth_mean == pytest.approx(0.32)
    assert p.revenue_growth_std == pytest.approx(0.10)  # tighter than 0.20 baseline


def test_derive_fallback_projections_uses_op_margin_avg_when_present():
    p = _derive_fallback_projections({"operating_margin_5yr_avg": 0.55})
    assert p.margin_mean == pytest.approx(0.55)
    assert p.margin_std == pytest.approx(0.05)


def test_derive_fallback_projections_uses_pe_trailing_when_present():
    p = _derive_fallback_projections({"pe_trailing": 28.5})
    assert p.exit_multiple_mean == pytest.approx(28.5)


def test_derive_fallback_projections_mixes_derived_and_defaults():
    """Partial KPIs: each field independently picks history-or-baseline."""
    p = _derive_fallback_projections({"revenue_5y_cagr": 0.18})
    assert p.revenue_growth_mean == pytest.approx(0.18)  # derived
    assert p.margin_mean == pytest.approx(NULL_HYPOTHESIS_PROJECTIONS.margin_mean)  # baseline
    assert p.exit_multiple_mean == pytest.approx(
        NULL_HYPOTHESIS_PROJECTIONS.exit_multiple_mean
    )  # baseline


@pytest.mark.asyncio
async def test_run_returns_null_hypothesis_when_no_kpis(monkeypatch):
    """If yfinance returns nothing AND we have no history, we still emit a valid
    FundamentalsOutput so Monte Carlo and Synthesis don't crash. Projections are
    the generic-mediocre-business baseline."""
    from agents import fundamentals as f

    monkeypatch.setattr(f, "get_financials", lambda t: {})
    state = {"ticker": "ZZZZ", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FundamentalsOutput.model_validate(result["fundamentals"])
    assert Projections(**out.projections.model_dump()) == NULL_HYPOTHESIS_PROJECTIONS
    assert any("no kpis" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_uses_history_derived_fallback_on_llm_failure(monkeypatch):
    """LLM fails but compute_kpis succeeded — fallback uses *those* values, not
    the null hypothesis. Confirms approach (A): degrade with history when we have it."""
    from agents import fundamentals as f

    fixture = _fake_financials(
        revenues=[100, 200, 400, 700, 1200],
        op_incomes=[10, 30, 80, 200, 400],
        gross_profits=[40, 100, 200, 400, 700],
        net_incomes=[5, 25, 70, 180, 380],
        fcfs=[5, 30, 100, 250, 500],
        capex=[-5, -15, -40, -100, -200],
        info={"marketCap": 50e9, "trailingPE": 30, "sharesOutstanding": 1e9},
        price=120.0,
    )
    monkeypatch.setattr(f, "get_financials", lambda t: fixture)
    monkeypatch.setattr(
        f, "_call_llm", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    state = {"ticker": "STUB", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FundamentalsOutput.model_validate(result["fundamentals"])

    # Fallback should reflect the history we computed, not generic defaults.
    expected = _derive_fallback_projections(out.kpis)
    assert out.projections.revenue_growth_mean == pytest.approx(expected.revenue_growth_mean)
    assert out.projections.exit_multiple_mean == pytest.approx(expected.exit_multiple_mean)
    # Sanity: revenue_growth_mean is *not* the null-hypothesis 5%.
    assert out.projections.revenue_growth_mean != pytest.approx(
        NULL_HYPOTHESIS_PROJECTIONS.revenue_growth_mean
    )
    assert out.projections.exit_multiple_mean != pytest.approx(
        NULL_HYPOTHESIS_PROJECTIONS.exit_multiple_mean
    )
    assert out.kpis  # KPIs were still computed
    assert any("llm" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_propagates_llm_output_when_call_succeeds(monkeypatch):
    """Happy path with mocked LLM: output should be the LLM's response, schema-validated."""
    from agents import fundamentals as f

    fixture = _fake_financials(
        revenues=[100, 200, 400, 700, 1200],
        op_incomes=[10, 30, 80, 200, 400],
        gross_profits=[40, 100, 200, 400, 700],
        net_incomes=[5, 25, 70, 180, 380],
        fcfs=[5, 30, 100, 250, 500],
        capex=[-5, -15, -40, -100, -200],
        info={"marketCap": 50e9, "sharesOutstanding": 1e9},
        price=120.0,
    )
    monkeypatch.setattr(f, "get_financials", lambda t: fixture)

    fake_response = {
        "summary": "Fake thesis-aware NVDA take.",
        "kpis": {"echoed": True},
        "projections": {
            "revenue_growth_mean": 0.25,
            "revenue_growth_std": 0.07,
            "margin_mean": 0.55,
            "margin_std": 0.05,
            "exit_multiple_mean": 30.0,
            "exit_multiple_std": 5.0,
        },
        "evidence": [{"source": "yfinance", "note": "x", "excerpt": "y"}],
    }
    monkeypatch.setattr(f, "_call_llm", lambda *a, **kw: fake_response)

    state = {"ticker": "NVDA", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FundamentalsOutput.model_validate(result["fundamentals"])
    assert out.projections.revenue_growth_mean == pytest.approx(0.25)
    assert "Fake thesis-aware" in out.summary
    assert out.errors == []
