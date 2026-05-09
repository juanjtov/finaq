"""Step 5a unit tests — Fundamentals KPI math + prompt assembly + failure paths.

Pure-logic, no network. Real LLM calls live in test_fundamentals_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.fundamentals import (
    NULL_HYPOTHESIS_PROJECTIONS,
    STALE_DATA_DAYS,
    _build_user_prompt,
    _check_freshness,
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


def test_compute_kpis_skips_leading_nulls_for_revenue_cagr():
    """yfinance sometimes has nulls in the oldest record. CAGR should compute
    over the populated subset rather than fail."""
    fin = _fake_financials(revenues=[0, 100, 200])  # 0 treated as missing
    kpis = compute_kpis(fin)
    # CAGR over [100, 200] with n_years=1 → 100% growth
    assert kpis["revenue_5y_cagr"] == pytest.approx(1.0)


def test_compute_kpis_omits_revenue_cagr_when_only_one_period_has_data():
    fin = _fake_financials(revenues=[0, 0, 100])  # only one populated date
    kpis = compute_kpis(fin)
    assert "revenue_5y_cagr" not in kpis


def test_compute_kpis_resolves_alternate_revenue_field_name():
    """yfinance occasionally uses 'Operating Revenue' instead of 'Total Revenue'."""
    income = {
        "2023-12-31": {"Operating Revenue": 100, "Operating Income": 20},
        "2024-12-31": {"Operating Revenue": 200, "Operating Income": 50},
    }
    fin = {"income_stmt": income, "cash_flow": {}, "info": {}, "price_history_5y": {}}
    kpis = compute_kpis(fin)
    assert "revenue_latest" in kpis
    assert kpis["revenue_latest"] == 200
    assert "operating_margin_latest" in kpis  # 50/200 = 0.25
    assert kpis["operating_margin_latest"] == pytest.approx(0.25)


def test_compute_kpis_resolves_alternate_capex_field_name():
    """Some tickers report 'Capital Expenditures' (plural) instead of 'Capital Expenditure'."""
    income = {
        "2023-12-31": {"Total Revenue": 100},
        "2024-12-31": {"Total Revenue": 200},
    }
    cash_flow = {
        "2023-12-31": {"Capital Expenditures": -10},
        "2024-12-31": {"Capital Expenditures": -20},
    }
    fin = {
        "income_stmt": income,
        "cash_flow": cash_flow,
        "info": {},
        "price_history_5y": {},
    }
    kpis = compute_kpis(fin)
    assert "capex_to_revenue_5yr_avg" in kpis
    # 10% then 10% → avg 10%
    assert kpis["capex_to_revenue_5yr_avg"] == pytest.approx(10.0)


def test_compute_kpis_resolves_alternate_net_income_field_name():
    income = {
        "2023-12-31": {"Total Revenue": 100, "Net Income Common Stockholders": 50},
        "2024-12-31": {"Total Revenue": 200, "Net Income Common Stockholders": 80},
    }
    cash_flow = {
        "2023-12-31": {"Free Cash Flow": 60},
        "2024-12-31": {"Free Cash Flow": 100},
    }
    fin = {
        "income_stmt": income,
        "cash_flow": cash_flow,
        "info": {},
        "price_history_5y": {},
    }
    kpis = compute_kpis(fin)
    # sum(fcf)=160 / sum(ni)=130 ≈ 1.23
    assert kpis["fcf_to_net_income_5yr"] == pytest.approx(160 / 130)


def test_first_non_null_returns_first_match():
    from agents.fundamentals import _first_non_null

    row = {"Total Revenue": None, "Operating Revenue": 42, "Revenue": 99}
    assert _first_non_null(row, ["Total Revenue", "Operating Revenue", "Revenue"]) == 42


def test_first_non_null_filters_nan():
    """NaN values from pandas should be treated as missing."""
    from agents.fundamentals import _first_non_null

    nan = float("nan")
    row = {"Total Revenue": nan, "Operating Revenue": 100}
    assert _first_non_null(row, ["Total Revenue", "Operating Revenue"]) == 100


def test_first_non_null_returns_none_when_no_match():
    from agents.fundamentals import _first_non_null

    assert _first_non_null({"foo": 1}, ["bar", "baz"]) is None


# --- Freshness markers -------------------------------------------------------


def test_compute_kpis_surfaces_data_fetched_at_when_present():
    fin = {
        "fetched_at": "2026-04-27T18:30:00+00:00",
        "income_stmt": {"2024-12-31": {"Total Revenue": 100}},
        "cash_flow": {},
        "info": {},
        "price_history_5y": {},
    }
    kpis = compute_kpis(fin)
    assert kpis["data_fetched_at"] == "2026-04-27T18:30:00+00:00"


def test_compute_kpis_surfaces_latest_fiscal_period():
    fin = _fake_financials(revenues=[100, 200, 400])
    kpis = compute_kpis(fin)
    # Newest date in fixture is 2022-12-31 (index 2: 2020+2)
    assert kpis["latest_fiscal_period"] == "2022-12-31"


def test_compute_kpis_omits_freshness_keys_when_input_missing():
    """No fetched_at and no income data → no freshness keys at all (don't fabricate)."""
    kpis = compute_kpis({})
    assert "data_fetched_at" not in kpis
    assert "latest_fiscal_period" not in kpis


def test_check_freshness_returns_none_for_fresh_data():
    from datetime import UTC, datetime

    recent = datetime.now(UTC).isoformat()
    assert _check_freshness({"fetched_at": recent}) is None


def test_check_freshness_warns_when_data_older_than_threshold():
    from datetime import UTC, datetime, timedelta

    stale = (datetime.now(UTC) - timedelta(days=STALE_DATA_DAYS + 1)).isoformat()
    msg = _check_freshness({"fetched_at": stale})
    assert msg is not None
    assert "old" in msg.lower()


def test_check_freshness_handles_missing_fetched_at():
    assert _check_freshness({}) is None
    assert _check_freshness({"fetched_at": None}) is None


def test_check_freshness_handles_malformed_timestamp():
    assert _check_freshness({"fetched_at": "not a date"}) is None


def test_build_user_prompt_renders_as_of_block():
    """The freshness header must be at the top of the prompt so the LLM sees it
    before any KPI."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    prompt = _build_user_prompt(
        "NVDA",
        thesis,
        {"data_fetched_at": "2026-04-27T18:30:00Z", "latest_fiscal_period": "2026-01-31"},
    )
    assert "AS OF" in prompt
    assert "2026-04-27" in prompt  # fetched_at
    assert "2026-01-31" in prompt  # latest fiscal period
    assert prompt.index("AS OF") < prompt.index("HISTORICAL KPIs")


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

    monkeypatch.setattr(f, "get_financials", lambda t, **kw: {})
    state = {"ticker": "ZZZZ", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FundamentalsOutput.model_validate(result["fundamentals"])
    assert Projections(**out.projections.model_dump()) == NULL_HYPOTHESIS_PROJECTIONS
    assert any("no kpis" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_preserves_deterministic_kpis_when_llm_drops_them(monkeypatch):
    """Real-world bug: the LLM sometimes echoes the kpis dict but omits one-off
    keys like `shares_outstanding` or `current_price`. Both are required by
    Monte Carlo downstream — if the LLM drops them, MC silently skips.

    Fix: deterministic compute_kpis() values override anything the LLM
    emitted with the same name, AND any deterministic-only key is always
    present in the final output. This test enforces that contract."""
    from agents import fundamentals as f

    deterministic_kpis = {
        "revenue_latest": 215_938_000_000.0,
        "shares_outstanding": 24_300_000_000.0,
        "current_price": 213.17,
        "pe_trailing": 47.5,
        "fcf_yield": 1.8,
    }

    # LLM emits a kpis dict that DROPS shares_outstanding and current_price
    # (a real failure mode we observed in production).
    llm_payload = {
        "summary": "NVDA looks expensive on FCF yield but priced for growth.",
        "kpis": {
            "revenue_latest": 215_938_000_000.0,  # echoed
            "pe_trailing": 47.5,  # echoed
            "fcf_yield": 1.8,  # echoed
            "thesis_alignment_score": 0.85,  # LLM-only enrichment — must survive
            # NOTE: no shares_outstanding, no current_price
        },
        "projections": {
            "revenue_growth_mean": 0.20,
            "revenue_growth_std": 0.05,
            "margin_mean": 0.55,
            "margin_std": 0.05,
            "exit_multiple_mean": 32.0,
            "exit_multiple_std": 5.0,
        },
        "evidence": [
            {"source": "yfinance", "note": "fcf_yield", "excerpt": "1.8%"},
        ],
    }

    monkeypatch.setattr(f, "compute_kpis", lambda financials: deterministic_kpis)
    monkeypatch.setattr(f, "get_financials", lambda t, **kw: {"info": {"longName": "NVDA"}})
    monkeypatch.setattr(f, "_call_llm", lambda *args, **kwargs: llm_payload)

    state = {"ticker": "NVDA", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FundamentalsOutput.model_validate(result["fundamentals"])

    # Deterministic-only keys must be in the final kpis even though the LLM
    # dropped them.
    assert out.kpis.get("shares_outstanding") == 24_300_000_000.0, (
        "shares_outstanding lost to LLM drop"
    )
    assert out.kpis.get("current_price") == 213.17, "current_price lost to LLM drop"

    # LLM-only enrichment keys must also survive.
    assert out.kpis.get("thesis_alignment_score") == 0.85

    # Echoed keys still match deterministic values.
    assert out.kpis.get("revenue_latest") == 215_938_000_000.0


@pytest.mark.asyncio
async def test_run_deterministic_kpis_override_llm_hallucinations(monkeypatch):
    """If the LLM hallucinates a different value for a deterministic key,
    the deterministic value MUST win. Otherwise an LLM brain fart can corrupt
    Monte Carlo's input distribution."""
    from agents import fundamentals as f

    deterministic_kpis = {"shares_outstanding": 24_300_000_000.0, "current_price": 213.17}
    llm_payload = {
        "summary": "x",
        "kpis": {
            # Hallucinated values — should be overridden
            "shares_outstanding": 999_999.0,
            "current_price": 9999.99,
        },
        "projections": {
            "revenue_growth_mean": 0.20,
            "revenue_growth_std": 0.05,
            "margin_mean": 0.55,
            "margin_std": 0.05,
            "exit_multiple_mean": 32.0,
            "exit_multiple_std": 5.0,
        },
        "evidence": [],
    }
    monkeypatch.setattr(f, "compute_kpis", lambda financials: deterministic_kpis)
    monkeypatch.setattr(f, "get_financials", lambda t, **kw: {"info": {"longName": "NVDA"}})
    monkeypatch.setattr(f, "_call_llm", lambda *args, **kwargs: llm_payload)

    state = {"ticker": "NVDA", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = FundamentalsOutput.model_validate(result["fundamentals"])

    assert out.kpis.get("shares_outstanding") == 24_300_000_000.0
    assert out.kpis.get("current_price") == 213.17


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
    monkeypatch.setattr(f, "get_financials", lambda t, **kw: fixture)
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
    monkeypatch.setattr(f, "get_financials", lambda t, **kw: fixture)

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
