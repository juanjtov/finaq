"""Monte Carlo fair-value engine — hybrid Owner-Earnings DCF + Multiple model.

Per draw out of n_sims:
  1. **DCF model** (primary)
     For each forecast year y in 1..N:
       revenue_y       = revenue_now × (1 + growth)^y
       op_income_y     = revenue_y × margin
       net_income_y    = op_income_y × (1 − tax_rate)
       da_y            = revenue_y × da_pct
       capex_y         = revenue_y × maintenance_capex_pct
       owner_earn_y    = net_income_y + da_y − capex_y
       pv_y            = owner_earn_y / (1 + discount_rate)^y

     terminal_oe       = owner_earn_N × (1 + terminal_growth)
     terminal_value    = terminal_oe / (discount_rate − terminal_growth)
     pv_terminal       = terminal_value / (1 + discount_rate)^N

     enterprise_value  = Σ pv_y + pv_terminal
     equity_value      = enterprise_value + net_cash
     fair_value_dcf    = equity_value / shares_year_N   (with dilution applied)

  2. **Multiple model** (secondary check, year-5 horizon)
     revenue_5         = revenue_now × (1 + growth)^5
     earnings_5        = revenue_5 × margin
     fair_value_mult   = (earnings_5 × exit_multiple) / shares_year_5

Both use the same growth/margin/dilution draws so the two distributions are
directly comparable. The convergence_ratio = min(dcf_p50, mult_p50)/max(...)
flags divergence (<0.7 = warn).

Distributions per parameter (see docs/FINANCE_ASSUMPTIONS.md §7):
  - revenue_growth, margin, tax, capex_pct, da_pct, dilution: truncated normal
  - exit_multiple:                                            lognormal
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.schemas import Projections, ValuationConfig

DEFAULT_N_SIMS = 10_000
DEFAULT_FORECAST_YEARS = 10
MULTIPLE_MODEL_HORIZON = 5

# Truncation bounds for each parameter — keeps draws inside economically
# sensible ranges. See docs/FINANCE_ASSUMPTIONS.md §7.
BOUNDS = {
    "growth": (-0.50, 2.00),
    "margin": (0.00, 0.95),
    "tax": (0.00, 0.50),
    "capex_pct": (0.00, 0.40),
    "da_pct": (0.00, 0.40),
    "dilution": (-0.05, 0.10),
    "multiple": (3.0, 100.0),
}


@dataclass(frozen=True)
class Distribution:
    """Percentile snapshot of one fair-value distribution."""

    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    samples: np.ndarray  # shape (n_sims,)

    def to_dict(self, include_samples: bool = False) -> dict:
        d = {
            "p10": float(self.p10),
            "p25": float(self.p25),
            "p50": float(self.p50),
            "p75": float(self.p75),
            "p90": float(self.p90),
        }
        if include_samples:
            d["samples"] = self.samples.tolist()
        return d


def _percentiles(samples: np.ndarray) -> Distribution:
    p10, p25, p50, p75, p90 = np.percentile(samples, [10, 25, 50, 75, 90])
    return Distribution(p10, p25, p50, p75, p90, samples)


def _draw_truncated_normal(
    rng: np.random.Generator, mean: float, std: float, lo: float, hi: float, n: int
) -> np.ndarray:
    """Truncated-normal draws via clip — fast, simple, slightly biased near the
    bounds vs proper rejection sampling. Acceptable for our parameter ranges."""
    return np.clip(rng.normal(mean, std, n), lo, hi)


def _draw_lognormal(
    rng: np.random.Generator, mean: float, std: float, lo: float, hi: float, n: int
) -> np.ndarray:
    """Lognormal draws specified via the *desired* mean and std of the variable
    itself (not of its log). Converts to underlying-normal parameters via the
    standard lognormal moment formulas."""
    # Avoid div-by-zero on degenerate inputs
    mean = max(mean, 1e-6)
    std = max(std, 1e-6)
    var = std**2
    log_mean = np.log(mean**2 / np.sqrt(var + mean**2))
    log_std = np.sqrt(np.log(1 + var / mean**2))
    samples = rng.lognormal(log_mean, log_std, n)
    return np.clip(samples, lo, hi)


def _validate_inputs(
    revenue_now: float, shares_now: float, current_price: float, discount_rate: float
) -> None:
    if revenue_now <= 0:
        raise ValueError(f"revenue_now must be positive, got {revenue_now}")
    if shares_now <= 0:
        raise ValueError(f"shares_now must be positive, got {shares_now}")
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")
    if not (0 < discount_rate < 0.30):
        raise ValueError(f"discount_rate must be in (0, 0.30), got {discount_rate}")


def simulate(
    *,
    projections: Projections,
    valuation: ValuationConfig,
    revenue_now: float,
    shares_now: float,
    current_price: float,
    net_cash: float,
    discount_rate: float,
    n_sims: int = DEFAULT_N_SIMS,
    n_years: int = DEFAULT_FORECAST_YEARS,
    seed: int = 42,
) -> dict:
    """Run hybrid DCF + Multiple Monte Carlo. Returns a dict suitable for the
    `monte_carlo` field in FinaqState.

    Args:
      projections: per-parameter mean/std (Fundamentals output).
      valuation:   per-thesis ERP / terminal-growth / discount caps.
      revenue_now: latest annual revenue (USD).
      shares_now:  diluted basic shares outstanding now.
      current_price: latest close price (USD).
      net_cash:    cash − total debt from balance sheet (USD; can be negative).
      discount_rate: pre-computed discount rate (e.g. treasury_10y + ERP), already
                     clipped to valuation.discount_rate_floor/cap by caller.
      n_sims:      Monte Carlo draws per model.
      n_years:     DCF forecast horizon.
      seed:        RNG seed for reproducibility.

    Returns:
      dict with keys: method, dcf, multiple, convergence_ratio, current_price,
                      discount_rate_used, terminal_growth_used, n_sims, n_years
    """
    _validate_inputs(revenue_now, shares_now, current_price, discount_rate)
    g = valuation.terminal_growth_rate
    if discount_rate <= g:
        raise ValueError(f"discount_rate ({discount_rate}) must exceed terminal_growth_rate ({g})")

    rng = np.random.default_rng(seed)

    # --- Draws (shared across both models so they're comparable) ---
    growth = _draw_truncated_normal(
        rng,
        projections.revenue_growth_mean,
        projections.revenue_growth_std,
        *BOUNDS["growth"],
        n_sims,
    )
    margin = _draw_truncated_normal(
        rng, projections.margin_mean, projections.margin_std, *BOUNDS["margin"], n_sims
    )
    tax = _draw_truncated_normal(
        rng, projections.tax_rate_mean, projections.tax_rate_std, *BOUNDS["tax"], n_sims
    )
    capex_pct = _draw_truncated_normal(
        rng,
        projections.maintenance_capex_pct_rev_mean,
        projections.maintenance_capex_pct_rev_std,
        *BOUNDS["capex_pct"],
        n_sims,
    )
    da_pct = _draw_truncated_normal(
        rng, projections.da_pct_rev_mean, projections.da_pct_rev_std, *BOUNDS["da_pct"], n_sims
    )
    dilution = _draw_truncated_normal(
        rng,
        projections.dilution_rate_mean,
        projections.dilution_rate_std,
        *BOUNDS["dilution"],
        n_sims,
    )
    # Lognormal for multiples — fat right tail matters for growth-name valuations
    multiple = _draw_lognormal(
        rng,
        projections.exit_multiple_mean,
        projections.exit_multiple_std,
        *BOUNDS["multiple"],
        n_sims,
    )

    years = np.arange(1, n_years + 1)  # (n_years,)

    # --- DCF model (vectorised) ---
    revenue_path = revenue_now * (1 + growth[:, None]) ** years[None, :]
    op_income_path = revenue_path * margin[:, None]
    net_income_path = op_income_path * (1 - tax[:, None])
    da_path = revenue_path * da_pct[:, None]
    capex_path = revenue_path * capex_pct[:, None]
    owner_earnings_path = net_income_path + da_path - capex_path  # may go negative

    discount_factors = 1.0 / (1 + discount_rate) ** years
    pv_oe = owner_earnings_path * discount_factors[None, :]
    sum_pv_explicit = pv_oe.sum(axis=1)

    oe_year_n = owner_earnings_path[:, -1]
    terminal_oe = oe_year_n * (1 + g)
    # Gordon growth model — discount_rate > g already validated above
    terminal_value = terminal_oe / (discount_rate - g)
    pv_terminal = terminal_value / (1 + discount_rate) ** n_years

    enterprise_value = sum_pv_explicit + pv_terminal
    equity_value = enterprise_value + net_cash

    shares_year_n = shares_now * (1 + dilution) ** n_years
    fair_value_dcf = equity_value / shares_year_n
    fair_value_dcf = np.maximum(
        fair_value_dcf, 0.0
    )  # cap at zero — negative fair value is meaningless

    # --- Multiple model (year-5 horizon, secondary check) ---
    revenue_5 = revenue_now * (1 + growth) ** MULTIPLE_MODEL_HORIZON
    earnings_5 = revenue_5 * margin
    shares_year_5 = shares_now * (1 + dilution) ** MULTIPLE_MODEL_HORIZON
    fair_value_mult = (earnings_5 * multiple) / shares_year_5
    fair_value_mult = np.maximum(fair_value_mult, 0.0)

    dcf_dist = _percentiles(fair_value_dcf)
    mult_dist = _percentiles(fair_value_mult)

    # Convergence: how close are the two median estimates?
    convergence_ratio = (
        min(dcf_dist.p50, mult_dist.p50) / max(dcf_dist.p50, mult_dist.p50)
        if max(dcf_dist.p50, mult_dist.p50) > 0
        else 0.0
    )

    return {
        "method": "dcf+multiple",
        "dcf": dcf_dist.to_dict(include_samples=False),
        "multiple": mult_dist.to_dict(include_samples=False),
        "convergence_ratio": float(convergence_ratio),
        "current_price": float(current_price),
        "discount_rate_used": float(discount_rate),
        "terminal_growth_used": float(g),
        "n_sims": int(n_sims),
        "n_years": int(n_years),
    }


def compute_discount_rate(treasury_10y: float, valuation: ValuationConfig) -> float:
    """Buffett-simplified discount rate: 10y Treasury + per-thesis ERP, clipped
    to the thesis's [floor, cap] band. See docs/FINANCE_ASSUMPTIONS.md §3."""
    raw = treasury_10y + valuation.equity_risk_premium
    return float(np.clip(raw, valuation.discount_rate_floor, valuation.discount_rate_cap))


# Parameters that get perturbed in sensitivity analysis. Each entry is the
# Projections field name; the discount_rate is handled separately because it
# lives outside Projections.
_SENSITIVITY_PARAMS = (
    "revenue_growth_mean",
    "margin_mean",
    "tax_rate_mean",
    "maintenance_capex_pct_rev_mean",
    "da_pct_rev_mean",
    "dilution_rate_mean",
    "exit_multiple_mean",
)
_SENSITIVITY_RELATIVE_SHIFT = 0.10  # 10% relative perturbation; gives interpretable elasticity


def compute_sensitivity(
    *,
    projections: Projections,
    valuation: ValuationConfig,
    revenue_now: float,
    shares_now: float,
    current_price: float,
    net_cash: float,
    discount_rate: float,
    n_sims: int = 5000,  # smaller than DEFAULT_N_SIMS — keeps the eight runs fast
    seed: int = 42,
) -> dict[str, float]:
    """Sensitivity diagnostic: how much does DCF P50 change per 1% change in each
    input parameter?

    Computes ∂(P50)/∂(param) via finite differences with a 10%-relative
    perturbation, then normalises to **elasticity** (% change in P50 per 1%
    change in input). Higher absolute value = more sensitive.

    Returns a dict mapping parameter name → elasticity. Eight params total: the
    seven Projections.*_mean fields plus discount_rate.
    """
    baseline = simulate(
        projections=projections,
        valuation=valuation,
        revenue_now=revenue_now,
        shares_now=shares_now,
        current_price=current_price,
        net_cash=net_cash,
        discount_rate=discount_rate,
        n_sims=n_sims,
        seed=seed,
    )
    baseline_p50 = baseline["dcf"]["p50"]
    if baseline_p50 <= 0:
        return dict.fromkeys((*_SENSITIVITY_PARAMS, "discount_rate"), 0.0)

    elasticities: dict[str, float] = {}
    for param in _SENSITIVITY_PARAMS:
        baseline_val = getattr(projections, param)
        shifted_val = baseline_val * (1 + _SENSITIVITY_RELATIVE_SHIFT)
        if abs(baseline_val) < 1e-9:
            # Param is at zero — additive shift instead of relative
            shifted_val = baseline_val + 0.001
            actual_shift = 0.001
        else:
            actual_shift = _SENSITIVITY_RELATIVE_SHIFT
        shifted_proj = projections.model_copy(update={param: shifted_val})
        try:
            shifted = simulate(
                projections=shifted_proj,
                valuation=valuation,
                revenue_now=revenue_now,
                shares_now=shares_now,
                current_price=current_price,
                net_cash=net_cash,
                discount_rate=discount_rate,
                n_sims=n_sims,
                seed=seed,
            )
            pct_change = (shifted["dcf"]["p50"] - baseline_p50) / baseline_p50
            elasticities[param] = pct_change / actual_shift
        except ValueError:
            elasticities[param] = float("nan")

    # Discount rate sensitivity (lives outside Projections)
    shifted_discount = discount_rate * (1 + _SENSITIVITY_RELATIVE_SHIFT)
    g = valuation.terminal_growth_rate
    if shifted_discount > g:  # Gordon-growth requires r > g
        try:
            shifted = simulate(
                projections=projections,
                valuation=valuation,
                revenue_now=revenue_now,
                shares_now=shares_now,
                current_price=current_price,
                net_cash=net_cash,
                discount_rate=shifted_discount,
                n_sims=n_sims,
                seed=seed,
            )
            pct_change = (shifted["dcf"]["p50"] - baseline_p50) / baseline_p50
            elasticities["discount_rate"] = pct_change / _SENSITIVITY_RELATIVE_SHIFT
        except ValueError:
            elasticities["discount_rate"] = float("nan")
    else:
        elasticities["discount_rate"] = float("nan")

    return elasticities
