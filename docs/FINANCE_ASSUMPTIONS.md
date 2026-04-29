# FINAQ Finance Assumptions

This document records every finance-modeling assumption used by the FINAQ
valuation pipeline. It exists so that when a fair-value distribution looks
surprising, the user can trace each component back to the choice that drives it.

If you change any assumption in code, update this doc in the same change.
If a change reflects a policy shift, also add a `Revised because:` block to
`docs/ARCHITECTURE.md`.

---

## §1 Valuation framework — hybrid DCF + Multiple

We compute fair value per share via **two parallel models** in every Monte
Carlo run, using shared random draws so they're directly comparable.

### §1.1 Owner-Earnings DCF (primary)

Buffett's preferred formula. For each forecast year `y` in `1..N`
(default `N=10`):

```
revenue_y      = revenue_now × (1 + growth)^y
op_income_y    = revenue_y × margin
net_income_y   = op_income_y × (1 − tax_rate)
da_y           = revenue_y × (D&A as % of revenue)
capex_y        = revenue_y × (maintenance capex as % of revenue)
owner_earn_y   = net_income_y + da_y − capex_y
pv_y           = owner_earn_y / (1 + discount_rate)^y
```

Terminal value via Gordon growth:

```
terminal_oe    = owner_earn_N × (1 + terminal_growth)
terminal_value = terminal_oe / (discount_rate − terminal_growth)
pv_terminal    = terminal_value / (1 + discount_rate)^N
```

Equity value:

```
enterprise_value = Σ pv_y + pv_terminal
equity_value     = enterprise_value + net_cash
shares_year_N    = shares_now × (1 + dilution_rate)^N
fair_value_dcf   = equity_value / shares_year_N
```

### §1.2 Multiple-based (secondary check, year-5 horizon)

The simpler textbook model, kept as a sanity check:

```
revenue_5         = revenue_now × (1 + growth)^5
earnings_5        = revenue_5 × margin
shares_year_5     = shares_now × (1 + dilution_rate)^5
fair_value_mult   = (earnings_5 × exit_multiple) / shares_year_5
```

### §1.3 Convergence ratio interpretation

```
convergence_ratio = min(dcf_p50, mult_p50) / max(dcf_p50, mult_p50)
```

| Range | Interpretation |
|---|---|
| `> 0.85` | Strong agreement — high confidence in the central estimate |
| `0.70–0.85` | Reasonable agreement |
| `< 0.70` | **Divergence flag.** One or both models is mis-specified for this ticker (e.g., growth-stage company where the multiple model overstates because earnings are still small but exit multiple is high). Synthesis should warn the user explicitly. |

---

## §2 Cash flow definition — Owner Earnings

`owner_earnings = net_income + D&A − maintenance_capex`

Buffett's definition. Why this rather than GAAP net income or FCF:

| Metric | What it captures | Why we don't use it as primary |
|---|---|---|
| GAAP net income | Accounting profit | Easily manipulated (non-cash items, impairments, accounting choices); divorced from actual cash a private owner could pull out |
| Free cash flow (CFO − capex) | Cash generated after all capex | Conflates *maintenance* capex (required to maintain ops) with *growth* capex (a discretionary investment for future earnings). Penalises growth investment at the moment it happens. |
| **Owner earnings** | Cash a private owner could pull out without impairing the business | Distinguishes maintenance from growth capex; adds back D&A (non-cash); standard in Buffett's letters |

**We approximate `maintenance_capex_pct_rev`** because separating maintenance
from growth capex is genuinely hard from yfinance data alone. The LLM is asked
to estimate it from historical capex intensity; if the LLM doesn't, we default
to 5% of revenue (conservative US-large-cap baseline).

**Working capital changes** (Δ-WC) are deliberately *omitted* from our owner
earnings formula. They typically average to zero over a multi-year window for
mature businesses; including them would add noise without adding signal at
our scale.

---

## §3 Discount rate — Buffett-simplified

`discount_rate = clip(treasury_10y + equity_risk_premium, floor, cap)`

### §3.1 Risk-free rate

The **10-year US Treasury yield**, refreshed daily from yfinance ticker `^TNX`.
Cached for 24 hours (long-run rates don't move enough intraday to matter for
fair-value distributions). Implementation in `data/treasury.py`.

On persistent fetch failure, falls back to `0.045` (typical 2025–26 range)
and logs a warning. This is the only place in the codebase where a magic
fallback value is silently used; it's logged so the next eval run will
expose the staleness.

### §3.2 Equity risk premium (per thesis)

Each thesis JSON declares its own ERP with documented reasoning:

| Thesis | ERP | Documented basis |
|---|---|---|
| AI cake | 5.0% | Long-run S&P 500 ERP ~4.5% + 0.5% premium for cyclical AI capex sensitivity |
| Halo · NVDA | 6.0% | S&P (4.5%) + 1.5% premium for single-anchor concentration risk |
| Construction | 3.5% | S&P (4.5%) − 1.0% for asset-backed defensive characteristics |

### §3.3 Discount-rate floor and cap

Each thesis declares `discount_rate_floor` and `discount_rate_cap` to clip
extreme treasury moves. Prevents a 1990s-style 8% Treasury or a 2020-style
0% Treasury from producing nonsense valuations.

| Thesis | Floor | Cap | Reasoning |
|---|---|---|---|
| AI cake | 7.5% | 13.0% | High-growth tech needs a meaningful discount; cap at typical late-cycle high |
| Halo · NVDA | 8.0% | 14.0% | Slightly higher band reflecting concentration risk |
| Construction | 6.0% | 11.0% | Lower band for defensive industrials |

### §3.4 Why not WACC

We use simple `treasury + ERP` not the more elaborate WACC formula
(weighted average cost of equity and after-tax debt). Two reasons:

1. **Buffett doesn't use WACC.** He uses Treasury + a small premium for
   businesses he understands; refuses to "compensate for risk" via a higher
   rate (he says: *"if it's risky, we just don't buy"*).
2. **WACC depends on the company's capital structure**, which can shift
   over the forecast horizon. Using a single rate avoids that complexity.

### §3.5 Why not random-draw the discount rate

The discount rate could itself be a Monte Carlo input (modelling rate
uncertainty). We treat it as fixed because:

- Long-run rates are policy-driven, not stochastic on the timescales we care about
- Adding a random discount rate adds a parameter without much insight at
  personal-tool scale
- Rate uncertainty is partially captured already via the floor/cap clip

---

## §4 Terminal growth (per thesis)

Each thesis JSON declares a single fixed `terminal_growth_rate` capped at
long-run real US GDP (~2%):

| Thesis | g | Documented basis |
|---|---|---|
| AI cake | 3.0% | Semiconductor + hyperscaler infra industry compound growth 1980–2025 minus 1pp for ex-AI normalisation |
| Halo · NVDA | 2.5% | Tied to NVDA cycle; reverts to broader IT sector long-run growth |
| Construction | 2.0% | Mature US industrial/infra GDP growth ~2% real long-run |

The Gordon growth model requires `discount_rate > terminal_growth`; the
engine validates this and raises if violated.

---

## §5 Exit multiple (data-driven)

For the secondary multiple-based model, `exit_multiple_mean` is derived
from data, not LLM-picked:

```
exit_multiple_mean = 0.4 × pe_5y_avg
                   + 0.3 × pe_trailing
                   + 0.3 × sector_pe
```

Rationale for the 0.4 / 0.3 / 0.3 weights:

- **0.4 × pe_5y_avg** — what the ticker historically traded at; strongest
  anchor against today's sentiment-driven mispricing
- **0.3 × pe_trailing** — current pricing acknowledged but not dominant
- **0.3 × sector_pe** — peer benchmark; mean-reversion toward sector centroid

`exit_multiple_std = pe_5y_std` from the same historical computation.

### §5.1 Sector P/E source

Currently a hardcoded JSON at `data/sector_multiples.json`, transcribed
manually from Damodaran's NYU Stern data tables. Refreshed quarterly.
**See `docs/POSTPONED.md` §2 — replacing with a live feed is on the
backlog.**

### §5.2 Lognormal distribution for the multiple

Per `§7`, the exit multiple is the only parameter we draw from a **lognormal**
distribution rather than truncated normal. The fat right tail matters for
growth-name valuations where the multiple realisation can be much higher
than the central estimate.

---

## §6 Shares outstanding (with dilution forecast)

Static `shares_now` understates fair-value-per-share for buyback-heavy names
and overstates it for SBC-heavy names over a 10-year horizon. We forecast:

```
shares_year_N = shares_now × (1 + dilution_rate)^N
```

Where:

- `dilution_rate_mean` is the **historical 5-year CAGR of share count**
  computed from yfinance balance-sheet data (negative for buyback-heavy
  names, positive for SBC-heavy)
- `dilution_rate_std` is the **5-year std of YoY dilution rate**

The Fundamentals agent estimates these from yfinance's 5y share-count
history; if not available, defaults are `mean=1%, std=0.5%`.

### §6.1 Net cash adjustment

`equity_value = enterprise_value + net_cash` where `net_cash = cash + ST
investments − total debt`, pulled from the most recent yfinance balance
sheet via `compute_kpis`. Adds (or subtracts) directly to per-share value.

---

## §7 Probability distributions per parameter

Choices below are **research-supported** but **practitioner-pragmatic** —
truncated normal everywhere except the exit multiple, where lognormal
genuinely matters.

| Parameter | Distribution | Bounds | Empirical basis |
|---|---|---|---|
| `revenue_growth` | Truncated normal | [−50%, +200%] | Penman (2010): annual growth follows lognormal in theory; truncated normal is 90% as good with intuitive parameter specification |
| `margin` (operating) | Truncated normal | [0, 0.95] | Beneish (2007), Damodaran (2012): bounded ratio, persistence within firm. Beta would be more correct; truncated normal is simpler |
| `tax_rate` | Truncated normal | [0, 0.5] | Dyreng et al (2008): bounded by statutory cap, firms cluster around effective rate |
| `maintenance_capex_pct_revenue` | Truncated normal | [0, 0.4] | Right-skewed in reality; tight stds make this matter less |
| `D&A_pct_revenue` | Truncated normal | [0, 0.4] | More stable than capex; tight clustering |
| `share_dilution_rate` | Truncated normal | [−5%, +10%] | Smooth in steady state; rare jumps |
| **`exit_multiple`** | **Lognormal** | [3, 100] | **Damodaran cross-sectional fit; long right tail matters for growth names** |
| `discount_rate` | Fixed | (computed) | Single number per drill-in (treasury + ERP, clipped) |
| `terminal_growth` | Fixed | (per thesis) | Single number from thesis JSON |

### §7.1 Lognormal helper specification

`utils.monte_carlo._draw_lognormal` takes the **desired mean and std of the
variable itself** (intuitive) and converts to the underlying normal's
parameters via the standard moment formulas:

```
log_mean = log(mean² / sqrt(var + mean²))
log_std  = sqrt(log(1 + var / mean²))
samples  = clip(rng.lognormal(log_mean, log_std, n), lo, hi)
```

This means the user passes `exit_multiple_mean=22, std=6` and the resulting
samples have approximately that mean/std (modulo truncation).

### §7.2 Truncation method

We use simple `np.clip` rather than rejection sampling for truncated normals.
Slightly biases the distribution near the bounds, but bounds are far from
the typical mean (e.g., `growth_mean=0.10` is nowhere near the `-50%` floor),
so the bias is negligible in practice. Rejection sampling would add 10× cost
for marginal accuracy.

---

## §8 What this model deliberately does NOT capture

These are honest gaps. Each one is logged with a trigger in `docs/POSTPONED.md` §2.

| Gap | Why we accept it |
|---|---|
| **Correlation between parameters** (growth ↔ margin ↔ multiple) | Independence simplifies the math; in reality high-growth names often see margin compression and lower terminal multiples. POSTPONED §2 trigger: empirical mis-calibration found via backtesting. |
| **Tail risk / catastrophic scenarios** (CEO exit, regulatory shutdown, fraud) | Gaussian model can't represent "5% chance of zero". POSTPONED §2 has Approach C — mixture distribution with catastrophic mode. |
| **Discount rate stochasticity** | Treated as fixed; rate uncertainty would just widen the histogram. |
| **Working capital dynamics** | Approximated to zero in owner earnings. |
| **Tax-rate jumps from policy changes** | Treated as a smooth random parameter, not a regime. |
| **Synthetic instruments** (convertibles, options) | Not modelled; share count is a simple scalar. |
| **Currency exposure** | All amounts assumed USD. International tickers would need FX modelling. |
| **Discount of fair value to present value** | The DCF model already discounts owner earnings *paths* to PV via `(1 + r)^y`. The **per-share output** is the present-value-of-future-cash-flows divided by year-N shares — interpretable as today's intrinsic value, not year-N nominal price. |

---

## §8.5 Sensitivity diagnostic (elasticity per input)

`utils.monte_carlo.compute_sensitivity()` answers "which input matters most for
this drill-in?" by perturbing each parameter mean by 10% (relative shift) and
measuring the resulting % change in DCF P50. The reported metric is **elasticity** —
the ratio of (% change in P50) to (% change in input).

Interpretation:

- **Positive elasticity** (revenue_growth, margin, D&A): higher input → higher P50.
- **Negative elasticity** (tax_rate, capex, dilution, discount_rate): higher input → lower P50.
- **Near-zero elasticity** (exit_multiple, since the DCF model doesn't use it): expected.

Magnitude tells you which assumption is most consequential for *this* ticker —
useful both for diagnostic ("if I'm wrong about growth by 1%, the P50 moves
~5%") and for the Synthesis report (which can highlight the top-3 most
sensitive inputs as the ones to watch).

Implementation: 9 forward runs (1 baseline + 8 perturbed) at `n_sims=5000`,
fixed seed for determinism. Adds ~500ms to a drill-in. Optional — Synthesis
can request it when desired.

## §9 What the model is and isn't useful for

**Useful for:**

- "Is this business clearly undervalued, fairly valued, or vastly overvalued?"
- Comparing the gap between current price and fair-value distribution
  (`current_price < dcf_p25` → potential margin of safety)
- Cross-ticker fair-value-per-thesis screens

**Not useful for:**

- Precision valuation (e.g., "fair value is $217.42")
- Short-horizon trade signals (the model is multi-year by design)
- Tickers outside the model's competency (banks, insurance, REITs, oil
  E&P — these need different valuation frameworks; the existing model
  will produce confidently-wrong numbers)
