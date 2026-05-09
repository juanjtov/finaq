"""Backtest scoring: predicted vs realised.

For each completed drill-in, we capture from the report:
  - `mc.dcf.{p10, p25, p50, p75, p90}` — Monte Carlo fair-value distribution
  - `mc.current_price` — price as of the drill date (= as_of)
  - `synthesis_confidence`
  - `risk.level`
  - The directional verdict from the Synthesis report's "What this means"
    (parsed heuristically from prose: "undervalued" / "fairly priced" /
    "overvalued").

Then we pull actual prices at +30 / +90 / +180 days and compute:
  - **Band coverage**: did `actual_price` fall inside [P10, P90]? In [P25, P75]?
  - **P50 magnitude error**: `abs(actual − P50) / P50` — how far off the central estimate was
  - **Direction**: did the model's verdict (cheap/fair/expensive) match the
    realised return direction?
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import yfinance as yf

from utils import logger

# --- Realised prices --------------------------------------------------------


def _ticker_close_at(ticker: str, target: date) -> float | None:
    """Return the close on the trading day ≤ target. None when yfinance
    has no data covering that range (delisted / pre-IPO / suspended)."""
    # Pull a 10-day window ending at target so we hit at least one trading
    # day even if target itself was a weekend or holiday.
    start = (target - timedelta(days=10)).isoformat()
    end = (target + timedelta(days=1)).isoformat()
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end)
    except Exception as e:
        logger.warning(f"[scorer] yfinance fetch failed for {ticker} @ {target}: {e}")
        return None
    if hist.empty:
        return None
    try:
        import pandas as pd
        idx = hist.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        mask = idx <= pd.Timestamp(target)
        hist = hist.loc[mask]
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"[scorer] price extract failed for {ticker} @ {target}: {e}")
        return None


def realised_prices(
    ticker: str,
    *,
    as_of_date: str,
    horizons: list[int],
) -> dict[str, dict[str, float | None]]:
    """For each horizon (in days), return the realised close price.

    Output:
      {
        "as_of":   {"date": "2025-09-05", "close": 21.85},
        "h_30":    {"date": "2025-10-05", "close": 22.10},
        "h_90":    {"date": "2025-12-04", "close": 19.80},
        "h_180":   {"date": "2026-03-04", "close": 24.90},
      }
    """
    as_of = datetime.fromisoformat(as_of_date).date()
    out: dict[str, dict[str, float | None]] = {
        "as_of": {
            "date": as_of.isoformat(),
            "close": _ticker_close_at(ticker, as_of),
        }
    }
    for h in horizons:
        target = as_of + timedelta(days=h)
        out[f"h_{h}"] = {
            "date": target.isoformat(),
            "close": _ticker_close_at(ticker, target),
        }
    return out


# --- Verdict extraction -----------------------------------------------------


# Directional verdict parsed from the synthesis report's "What this means"
# section. The persona prompt instructs the model to use one of these
# canonical phrasings (or close variants).
_UNDERVALUED_RE = re.compile(
    r"\b(under[- ]?valued|meaningfully cheap|trading below|"
    r"buy|adding|add (now|at|on))\b",
    re.IGNORECASE,
)
_OVERVALUED_RE = re.compile(
    r"\b(over[- ]?valued|expensive|stretched|pricey|"
    r"trim|reduce|lighten)\b",
    re.IGNORECASE,
)
_FAIR_RE = re.compile(
    r"\b(fairly priced|roughly fair|in line with|no action|"
    r"hold|monitor|watch|in[- ]line|neutral)\b",
    re.IGNORECASE,
)


def extract_verdict(report_markdown: str) -> str:
    """Heuristic verdict extraction. Returns one of:
    `undervalued | overvalued | fairly_priced | unknown`.

    Looks at the first ~1500 chars (post-header through "What this means")
    where the persona is instructed to commit to a directional verdict.
    """
    if not report_markdown:
        return "unknown"
    head = report_markdown[:1500]
    under = bool(_UNDERVALUED_RE.search(head))
    over = bool(_OVERVALUED_RE.search(head))
    fair = bool(_FAIR_RE.search(head))
    if under and not over:
        return "undervalued"
    if over and not under:
        return "overvalued"
    if fair and not (under and over):
        return "fairly_priced"
    return "unknown"


# --- Per-horizon scoring ----------------------------------------------------


def _band_coverage(price: float, lo: float | None, hi: float | None) -> bool | None:
    if price is None or lo is None or hi is None:
        return None
    return lo <= price <= hi


def _signed_pct_error(actual: float | None, p50: float | None) -> float | None:
    if actual is None or not p50 or p50 == 0:
        return None
    return (actual - p50) / p50


def _abs_pct_error(actual: float | None, p50: float | None) -> float | None:
    err = _signed_pct_error(actual, p50)
    return abs(err) if err is not None else None


def _direction_match(
    *,
    verdict: str,
    price_at_as_of: float | None,
    realised: float | None,
) -> bool | None:
    """Did the verdict match the realised direction?

    `undervalued` → expect realised > price_at_as_of
    `overvalued`  → expect realised < price_at_as_of
    `fairly_priced` → expect |Δ| < 5% (treat as "no big move")
    `unknown` → None (can't score)
    """
    if verdict == "unknown" or price_at_as_of is None or realised is None:
        return None
    delta_pct = (realised - price_at_as_of) / price_at_as_of
    if verdict == "undervalued":
        return delta_pct > 0
    if verdict == "overvalued":
        return delta_pct < 0
    if verdict == "fairly_priced":
        return abs(delta_pct) < 0.05
    return None


def score_run(
    *,
    ticker: str,
    as_of_date: str,
    horizons: list[int],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Compute calibration metrics for one drill-in.

    `state` is the final FinaqState dict (from `invoke_with_telemetry`).
    Pulls realised closes from yfinance live (no as_of filter — these
    are the future prices we're scoring against).
    """
    mc = (state.get("monte_carlo") or {})
    dcf = mc.get("dcf") or {}
    p10, p25, p50, p75, p90 = (
        dcf.get("p10"), dcf.get("p25"), dcf.get("p50"), dcf.get("p75"), dcf.get("p90")
    )
    confidence = state.get("synthesis_confidence")
    risk_level = (state.get("risk") or {}).get("level")
    report = state.get("report") or ""
    # Prefer the structured verdict synthesis emits as a side-channel; fall
    # back to regex-parsing the prose for legacy runs that pre-date the field.
    structured = state.get("synthesis_verdict")
    if structured in ("undervalued", "fairly_priced", "overvalued"):
        verdict = structured
    else:
        verdict = extract_verdict(report)

    prices = realised_prices(ticker, as_of_date=as_of_date, horizons=horizons)
    p_as_of = prices["as_of"]["close"]

    per_horizon: dict[str, dict] = {}
    for h in horizons:
        key = f"h_{h}"
        actual = prices[key]["close"]
        per_horizon[key] = {
            "horizon_days": h,
            "target_date": prices[key]["date"],
            "actual_close": actual,
            "in_p10_p90": _band_coverage(actual, p10, p90),
            "in_p25_p75": _band_coverage(actual, p25, p75),
            "abs_pct_err_vs_p50": _abs_pct_error(actual, p50),
            "signed_pct_err_vs_p50": _signed_pct_error(actual, p50),
            "actual_return_from_as_of": (
                (actual - p_as_of) / p_as_of
                if (actual is not None and p_as_of)
                else None
            ),
            "direction_match": _direction_match(
                verdict=verdict, price_at_as_of=p_as_of, realised=actual,
            ),
        }

    return {
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "verdict": verdict,
        "synthesis_confidence": confidence,
        "risk_level": risk_level,
        "mc": {"p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
               "current_price": mc.get("current_price"),
               "convergence_ratio": mc.get("convergence_ratio")},
        "prices": prices,
        "horizons": per_horizon,
    }
