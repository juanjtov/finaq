"""Tier 1d — sensitivity diagnostic tests.

Verifies the `compute_sensitivity` helper produces sensible elasticities for
each input parameter. Pure-deterministic (no LLM, no network).

Elasticity = (% change in DCF P50) / (% change in input).
  > 0  : P50 moves with the input (growth, margin → positive)
  < 0  : P50 moves against the input (discount rate, dilution, capex → negative)
  ≈ 0  : input has no effect (degenerate or canceling)
"""

from __future__ import annotations

import math

from utils.monte_carlo import compute_sensitivity
from utils.schemas import Projections, ValuationConfig


def _baseline_projections() -> Projections:
    return Projections(
        revenue_growth_mean=0.10,
        revenue_growth_std=0.03,
        margin_mean=0.20,
        margin_std=0.03,
        tax_rate_mean=0.21,
        tax_rate_std=0.02,
        maintenance_capex_pct_rev_mean=0.05,
        maintenance_capex_pct_rev_std=0.01,
        da_pct_rev_mean=0.04,
        da_pct_rev_std=0.01,
        dilution_rate_mean=0.01,
        dilution_rate_std=0.005,
        exit_multiple_mean=18.0,
        exit_multiple_std=4.0,
    )


def _baseline_valuation() -> ValuationConfig:
    return ValuationConfig(
        equity_risk_premium=0.05,
        erp_basis="test",
        terminal_growth_rate=0.025,
        terminal_growth_basis="test",
        discount_rate_floor=0.07,
        discount_rate_cap=0.13,
    )


def _run() -> dict[str, float]:
    return compute_sensitivity(
        projections=_baseline_projections(),
        valuation=_baseline_valuation(),
        revenue_now=50e9,
        shares_now=1e9,
        current_price=100.0,
        net_cash=5e9,
        discount_rate=0.09,
    )


# --- Output shape ----------------------------------------------------------


def test_sensitivity_returns_elasticity_per_param():
    e = _run()
    expected = {
        "revenue_growth_mean",
        "margin_mean",
        "tax_rate_mean",
        "maintenance_capex_pct_rev_mean",
        "da_pct_rev_mean",
        "dilution_rate_mean",
        "exit_multiple_mean",
        "discount_rate",
    }
    assert set(e.keys()) == expected


# --- Sign of each elasticity ----------------------------------------------


def test_growth_elasticity_is_positive():
    """Higher revenue growth → higher DCF P50."""
    e = _run()
    assert e["revenue_growth_mean"] > 0


def test_margin_elasticity_is_positive():
    e = _run()
    assert e["margin_mean"] > 0


def test_tax_rate_elasticity_is_negative():
    """Higher tax rate → lower net income → lower DCF P50."""
    e = _run()
    assert e["tax_rate_mean"] < 0


def test_capex_elasticity_is_negative():
    """Higher maintenance capex → lower owner earnings → lower DCF P50."""
    e = _run()
    assert e["maintenance_capex_pct_rev_mean"] < 0


def test_da_elasticity_is_positive():
    """D&A is added back in owner earnings → higher D&A → higher P50.

    (This may seem counterintuitive — D&A is a non-cash expense; in owner
    earnings we add it back because cash isn't actually leaving the business.
    Higher D&A means more cash-equivalent earnings.)"""
    e = _run()
    assert e["da_pct_rev_mean"] > 0


def test_dilution_elasticity_is_negative():
    """Higher dilution → more shares → lower per-share value."""
    e = _run()
    assert e["dilution_rate_mean"] < 0


def test_discount_rate_elasticity_is_negative():
    """Higher discount rate → smaller present value → lower DCF P50."""
    e = _run()
    assert e["discount_rate"] < 0


def test_exit_multiple_has_negligible_dcf_elasticity():
    """The DCF model doesn't use exit_multiple at all — that's the multiple
    model's input. So shifting it shouldn't move DCF P50.

    (The function reports DCF elasticity, not multiple-model elasticity.
    A non-zero number here indicates a bug in the simulator.)"""
    e = _run()
    assert abs(e["exit_multiple_mean"]) < 0.05


# --- Magnitudes / sanity --------------------------------------------------


def test_growth_is_among_top_3_elasticities_for_growth_baseline():
    """For a baseline with 10% growth and 20% margins, growth and margin should
    be among the most consequential inputs (largest |elasticity|)."""
    e = _run()
    # Sort by absolute elasticity descending
    ranked = sorted(
        ((k, abs(v)) for k, v in e.items() if math.isfinite(v) and abs(v) > 1e-6),
        key=lambda x: x[1],
        reverse=True,
    )
    top_3 = {name for name, _ in ranked[:3]}
    assert (
        "revenue_growth_mean" in top_3
    ), f"growth not in top-3 elasticities; got ranking: {ranked}"


def test_all_elasticities_are_finite():
    """Catches edge cases where the simulation produced NaN/inf for any param."""
    e = _run()
    for k, v in e.items():
        assert math.isfinite(v), f"{k} elasticity is non-finite: {v}"


def test_elasticities_are_reproducible_under_fixed_seed():
    a = _run()
    b = _run()
    for k in a:
        if math.isfinite(a[k]):
            assert a[k] == b[k], f"{k}: {a[k]} != {b[k]} (non-reproducible)"
