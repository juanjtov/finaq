"""Step 6 unit tests — Monte Carlo engine math correctness, reproducibility, edge cases.

Pure-numerical, no network. The MC engine has no LLM, so there's no Tier 2/3
LLM-judge eval; correctness is fully captured by deterministic checks.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from utils.monte_carlo import (
    BOUNDS,
    DEFAULT_FORECAST_YEARS,
    DEFAULT_N_SIMS,
    _draw_lognormal,
    _draw_truncated_normal,
    compute_discount_rate,
    simulate,
)
from utils.schemas import Projections, ValuationConfig

# --- Fixtures ---------------------------------------------------------------


def _baseline_projections(**overrides) -> Projections:
    base = {
        "revenue_growth_mean": 0.10,
        "revenue_growth_std": 0.03,
        "margin_mean": 0.20,
        "margin_std": 0.03,
        "tax_rate_mean": 0.21,
        "tax_rate_std": 0.02,
        "maintenance_capex_pct_rev_mean": 0.05,
        "maintenance_capex_pct_rev_std": 0.01,
        "da_pct_rev_mean": 0.04,
        "da_pct_rev_std": 0.01,
        "dilution_rate_mean": 0.01,
        "dilution_rate_std": 0.005,
        "exit_multiple_mean": 18.0,
        "exit_multiple_std": 4.0,
    }
    base.update(overrides)
    return Projections(**base)


def _baseline_valuation(**overrides) -> ValuationConfig:
    base = {
        "equity_risk_premium": 0.05,
        "erp_basis": "test",
        "terminal_growth_rate": 0.025,
        "terminal_growth_basis": "test",
        "discount_rate_floor": 0.07,
        "discount_rate_cap": 0.13,
    }
    base.update(overrides)
    return ValuationConfig(**base)


def _run_baseline(**kwargs) -> dict:
    """Default baseline run, override individual args via kwargs."""
    args = {
        "projections": _baseline_projections(),
        "valuation": _baseline_valuation(),
        "revenue_now": 50e9,
        "shares_now": 1e9,
        "current_price": 100.0,
        "net_cash": 5e9,
        "discount_rate": 0.09,
    }
    args.update(kwargs)
    return simulate(**args)


# --- Distribution shape ----------------------------------------------------


def test_percentiles_are_monotonically_ordered():
    out = _run_baseline()
    for model in ("dcf", "multiple"):
        d = out[model]
        assert d["p10"] <= d["p25"] <= d["p50"] <= d["p75"] <= d["p90"], d


def test_dcf_p50_is_reasonable_for_baseline_inputs():
    """A profitable, growing, modestly-leveraged business should land in a sensible
    fair-value range. Catches gross math errors."""
    out = _run_baseline()
    dcf_p50 = out["dcf"]["p50"]
    mult_p50 = out["multiple"]["p50"]
    assert 10 < dcf_p50 < 5000, f"DCF P50 outside sanity range: {dcf_p50}"
    assert 10 < mult_p50 < 5000, f"Mult P50 outside sanity range: {mult_p50}"


# --- Reproducibility -------------------------------------------------------


def test_same_seed_produces_bit_identical_output():
    a = _run_baseline(seed=123)
    b = _run_baseline(seed=123)
    for model in ("dcf", "multiple"):
        for q in ("p10", "p25", "p50", "p75", "p90"):
            assert a[model][q] == b[model][q], f"{model}.{q}: {a[model][q]} != {b[model][q]}"


def test_different_seeds_produce_different_output():
    a = _run_baseline(seed=1)
    b = _run_baseline(seed=2)
    assert a["dcf"]["p50"] != b["dcf"]["p50"]


# --- Vectorisation perf ----------------------------------------------------


def test_100k_sims_under_2_seconds():
    """Vectorised NumPy must keep large runs fast — no Python loops over draws."""
    t0 = time.monotonic()
    _run_baseline(n_sims=100_000)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"100k sims took {elapsed:.2f}s; expected <2s"


# --- Edge cases ------------------------------------------------------------


def test_zero_std_inputs_produce_near_deterministic_output():
    """When all stds are zero, the distribution collapses to a single value
    (modulo tiny float error)."""
    p = _baseline_projections(
        revenue_growth_std=0.0,
        margin_std=0.0,
        tax_rate_std=0.0,
        maintenance_capex_pct_rev_std=0.0,
        da_pct_rev_std=0.0,
        dilution_rate_std=0.0,
        exit_multiple_std=0.0001,  # lognormal can't take exactly zero
    )
    out = _run_baseline(projections=p)
    # All percentiles of dcf should be essentially equal (tiny floating noise)
    p10, p90 = out["dcf"]["p10"], out["dcf"]["p90"]
    assert abs(p10 - p90) / max(p10, 1e-6) < 0.01, f"non-collapsing: P10={p10}, P90={p90}"


def test_negative_growth_clipping_prevents_negative_revenue():
    """A growth_mean of -200% would imply revenue going below zero. The bounds
    must clip the draws so revenue stays non-negative throughout the path."""
    p = _baseline_projections(revenue_growth_mean=-2.0, revenue_growth_std=0.0)
    out = _run_baseline(projections=p)
    # With growth clipped at -50%, revenue compounds down but stays positive.
    # DCF P50 should be small (low/no earnings) but not negative.
    assert out["dcf"]["p50"] >= 0


def test_high_dilution_reduces_per_share_value():
    """Higher dilution should mechanically reduce fair-value per share."""
    low_dilution = _run_baseline(
        projections=_baseline_projections(dilution_rate_mean=0.0, dilution_rate_std=0.0)
    )
    high_dilution = _run_baseline(
        projections=_baseline_projections(dilution_rate_mean=0.05, dilution_rate_std=0.0)
    )
    # Per-share value with 5%/y dilution must be lower than with 0% dilution
    assert high_dilution["dcf"]["p50"] < low_dilution["dcf"]["p50"]


def test_higher_discount_rate_reduces_dcf_value():
    a = _run_baseline(discount_rate=0.07)
    b = _run_baseline(discount_rate=0.12)
    assert b["dcf"]["p50"] < a["dcf"]["p50"]  # higher r → smaller PV


def test_net_cash_adds_to_equity_value():
    no_cash = _run_baseline(net_cash=0.0)
    big_cash = _run_baseline(net_cash=20e9)  # $20B net cash on 1B shares = +$20/sh
    delta = big_cash["dcf"]["p50"] - no_cash["dcf"]["p50"]
    # Cash-per-share is added directly to equity, before dilution applied.
    # With 1B shares now, ~$20/sh expected, but dilution over 10 years pushes
    # the post-dilution per-share contribution slightly lower.
    assert 15 < delta < 22, f"unexpected net-cash contribution: {delta}"


# --- Convergence ratio -----------------------------------------------------


def test_convergence_ratio_in_zero_to_one_range():
    out = _run_baseline()
    assert 0.0 <= out["convergence_ratio"] <= 1.0


# --- Input validation ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"revenue_now": -1e9},
        {"revenue_now": 0.0},
        {"shares_now": 0.0},
        {"current_price": 0.0},
        {"discount_rate": 0.0},
        {"discount_rate": 0.40},
    ],
)
def test_simulate_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        _run_baseline(**kwargs)


def test_simulate_rejects_discount_rate_below_terminal_growth():
    """Gordon-growth model requires r > g; otherwise terminal value is negative
    or infinite."""
    val = _baseline_valuation(terminal_growth_rate=0.05, discount_rate_floor=0.04)
    with pytest.raises(ValueError, match="discount_rate"):
        _run_baseline(valuation=val, discount_rate=0.04)


# --- Compute discount rate -------------------------------------------------


def test_compute_discount_rate_treasury_plus_erp_clipped():
    val = _baseline_valuation(
        equity_risk_premium=0.05, discount_rate_floor=0.07, discount_rate_cap=0.13
    )
    # Normal case
    assert compute_discount_rate(0.04, val) == pytest.approx(0.09)
    # Floor binds
    assert compute_discount_rate(0.005, val) == pytest.approx(0.07)
    # Cap binds
    assert compute_discount_rate(0.10, val) == pytest.approx(0.13)


# --- Draw helpers ----------------------------------------------------------


def test_truncated_normal_respects_bounds():
    rng = np.random.default_rng(42)
    samples = _draw_truncated_normal(rng, 0.5, 0.3, 0.0, 1.0, 10_000)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_lognormal_draws_positive_and_within_bounds():
    rng = np.random.default_rng(42)
    samples = _draw_lognormal(rng, 18.0, 4.0, *BOUNDS["multiple"], 10_000)
    assert (samples > 0).all()
    assert samples.min() >= BOUNDS["multiple"][0]
    assert samples.max() <= BOUNDS["multiple"][1]


def test_lognormal_recovers_target_mean_within_tolerance():
    """The lognormal helper takes mean/std *of the variable* and converts
    internally. Empirical mean of large sample should be close to the input mean."""
    rng = np.random.default_rng(42)
    target_mean = 22.0
    samples = _draw_lognormal(rng, target_mean, 5.0, 1.0, 200.0, 100_000)
    empirical_mean = float(samples.mean())
    # Truncation can shave a tiny amount; allow 5% tolerance.
    assert abs(empirical_mean - target_mean) / target_mean < 0.05


# --- Output shape ----------------------------------------------------------


def test_simulate_output_keys():
    out = _run_baseline()
    expected_keys = {
        "method",
        "dcf",
        "multiple",
        "convergence_ratio",
        "current_price",
        "discount_rate_used",
        "terminal_growth_used",
        "n_sims",
        "n_years",
    }
    assert set(out.keys()) == expected_keys
    assert out["method"] == "dcf+multiple"
    assert out["n_sims"] == DEFAULT_N_SIMS
    assert out["n_years"] == DEFAULT_FORECAST_YEARS
