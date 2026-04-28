"""Fundamentals agent — Sonnet-backed thesis-aware fundamental analysis.

Pipeline:
1. Pull yfinance bundle (cached 24h).
2. Compute thesis-relevant KPIs (revenue CAGR, margins, FCF yield, capex intensity, etc.).
3. Send the thesis + KPIs to Sonnet; receive a structured JSON analysis + Monte Carlo projections.
4. Validate against FundamentalsOutput, surface partial failures via the `errors` field.

Run standalone:  python -m agents.fundamentals NVDA [thesis_slug]
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from data.yfin import get_financials
from utils import logger
from utils.models import MODEL_FUNDAMENTALS
from utils.openrouter import get_client
from utils.schemas import FundamentalsOutput, Projections
from utils.state import FinaqState

NODE = "fundamentals"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "fundamentals.md").read_text()
LLM_MAX_TOKENS = 2000

# Generic-mediocre-business "no information" baseline — anchored to long-run
# US nominal GDP growth (~5%), industrial-average operating margin (~10%), and
# the long-run S&P 500 P/E (~15x). Used only when even historical KPIs are missing.
NULL_HYPOTHESIS_PROJECTIONS = Projections(
    revenue_growth_mean=0.05,
    revenue_growth_std=0.20,  # wide on purpose — we have nothing
    margin_mean=0.10,
    margin_std=0.10,
    exit_multiple_mean=15.0,
    exit_multiple_std=5.0,
)


def _derive_fallback_projections(kpis: dict) -> Projections:
    """Best-effort projections from whatever historical KPIs were computed.

    Used when the LLM call fails or returns invalid output. Each field falls
    back to the null-hypothesis only if its historical anchor is missing.
    Standard deviations tighten when an anchor exists (we trust history more
    than the generic prior).
    """
    has_growth = "revenue_5y_cagr" in kpis
    has_margin = "operating_margin_5yr_avg" in kpis
    has_pe = "pe_trailing" in kpis
    return Projections(
        revenue_growth_mean=kpis.get(
            "revenue_5y_cagr", NULL_HYPOTHESIS_PROJECTIONS.revenue_growth_mean
        ),
        revenue_growth_std=0.10 if has_growth else NULL_HYPOTHESIS_PROJECTIONS.revenue_growth_std,
        margin_mean=kpis.get("operating_margin_5yr_avg", NULL_HYPOTHESIS_PROJECTIONS.margin_mean),
        margin_std=0.05 if has_margin else NULL_HYPOTHESIS_PROJECTIONS.margin_std,
        exit_multiple_mean=kpis.get("pe_trailing", NULL_HYPOTHESIS_PROJECTIONS.exit_multiple_mean),
        exit_multiple_std=5.0 if has_pe else NULL_HYPOTHESIS_PROJECTIONS.exit_multiple_std,
    )


# --- KPI computation ---------------------------------------------------------


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    try:
        return num / den
    except (TypeError, ZeroDivisionError):
        return None


def _last(d: dict) -> dict | None:
    if not d:
        return None
    return d[max(d.keys())]


def compute_kpis(financials: dict) -> dict[str, Any]:
    """Distill a yfinance bundle into thesis-aware KPIs.

    Missing inputs degrade gracefully — keys only appear when there's enough
    data to compute them honestly.
    """
    kpis: dict[str, Any] = {}
    income = financials.get("income_stmt") or {}
    cash_flow = financials.get("cash_flow") or {}
    info = financials.get("info") or {}
    price_hist = financials.get("price_history_5y") or {}

    inc_dates = sorted(income.keys())
    cf_dates = sorted(cash_flow.keys())

    # 5-year revenue CAGR
    if len(inc_dates) >= 2:
        rev_first = income[inc_dates[0]].get("Total Revenue")
        rev_last = income[inc_dates[-1]].get("Total Revenue")
        n_years = len(inc_dates) - 1
        if rev_first and rev_last and rev_first > 0 and n_years > 0:
            with contextlib.suppress(ValueError, OverflowError):
                kpis["revenue_5y_cagr"] = (rev_last / rev_first) ** (1 / n_years) - 1

    # Latest revenue + margins
    last_inc = _last(income)
    if last_inc:
        rev = last_inc.get("Total Revenue")
        if rev:
            kpis["revenue_latest"] = rev
        kpis["gross_margin_latest"] = _safe_div(last_inc.get("Gross Profit"), rev)
        kpis["operating_margin_latest"] = _safe_div(last_inc.get("Operating Income"), rev)

    # 5y average margins
    gross_margins = []
    op_margins = []
    for d in inc_dates:
        row = income[d]
        rev = row.get("Total Revenue")
        if rev and rev > 0:
            if row.get("Gross Profit"):
                gross_margins.append(row["Gross Profit"] / rev)
            if row.get("Operating Income"):
                op_margins.append(row["Operating Income"] / rev)
    if gross_margins:
        kpis["gross_margin_5yr_avg"] = sum(gross_margins) / len(gross_margins)
    if op_margins:
        kpis["operating_margin_5yr_avg"] = sum(op_margins) / len(op_margins)

    # FCF — latest + 5y avg
    last_cf = _last(cash_flow)
    fcf_latest = last_cf.get("Free Cash Flow") if last_cf else None
    if fcf_latest:
        kpis["fcf_latest"] = fcf_latest

    fcfs = [
        cash_flow[d].get("Free Cash Flow") for d in cf_dates if cash_flow[d].get("Free Cash Flow")
    ]
    if fcfs:
        kpis["fcf_5yr_avg"] = sum(fcfs) / len(fcfs)

    # Buffett: FCF yield, FCF/NI, capex intensity
    market_cap = info.get("marketCap")
    if fcf_latest and market_cap and market_cap > 0:
        kpis["fcf_yield"] = (fcf_latest / market_cap) * 100  # percent

    nis = [income[d].get("Net Income") for d in inc_dates if income[d].get("Net Income")]
    if fcfs and nis and sum(nis) != 0:
        kpis["fcf_to_net_income_5yr"] = sum(fcfs) / sum(nis)

    capex_pct = []
    for d in cf_dates:
        capex = cash_flow[d].get("Capital Expenditure")
        # Find revenue for the same fiscal year (date strings start with YYYY)
        if capex is not None:
            year = d[:4]
            matching_rev = next(
                (income[d2].get("Total Revenue") for d2 in inc_dates if d2[:4] == year),
                None,
            )
            if matching_rev and matching_rev > 0:
                capex_pct.append(abs(capex) / matching_rev * 100)
    if capex_pct:
        kpis["capex_to_revenue_5yr_avg"] = sum(capex_pct) / len(capex_pct)

    # Multiples + price + shares
    if info.get("trailingPE"):
        kpis["pe_trailing"] = info["trailingPE"]
    if info.get("forwardPE"):
        kpis["pe_forward"] = info["forwardPE"]
    if info.get("sharesOutstanding"):
        kpis["shares_outstanding"] = info["sharesOutstanding"]
    if info.get("marketCap"):
        kpis["market_cap"] = info["marketCap"]

    if price_hist:
        latest_date = max(price_hist.keys())
        close = price_hist[latest_date].get("Close")
        if close:
            kpis["current_price"] = close

    return kpis


# --- LLM call ----------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Models occasionally wrap JSON in ```json ... ``` despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text


def _build_user_prompt(ticker: str, thesis: dict, kpis: dict) -> str:
    return (
        f"TICKER UNDER ANALYSIS: {ticker}\n\n"
        f"ACTIVE THESIS: {thesis.get('name', 'unknown')}\n"
        f"THESIS SUMMARY: {thesis.get('summary', '')}\n"
        f"ANCHOR TICKERS: {', '.join(thesis.get('anchor_tickers', []))}\n"
        f"FULL UNIVERSE: {', '.join(thesis.get('universe', []))}\n\n"
        f"MATERIAL THRESHOLDS THE TRIAGE SYSTEM IS WATCHING:\n"
        f"{json.dumps(thesis.get('material_thresholds', []), indent=2)}\n\n"
        f"HISTORICAL KPIs FOR {ticker} (computed from yfinance):\n"
        f"{json.dumps(kpis, indent=2, default=str)}\n\n"
        f"PRODUCE YOUR ANALYSIS NOW. STRICT JSON ONLY."
    )


def _call_llm(ticker: str, thesis: dict, kpis: dict) -> dict:
    client = get_client()
    user = _build_user_prompt(ticker, thesis, kpis)
    resp = client.chat.completions.create(
        model=MODEL_FUNDAMENTALS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return json.loads(_strip_code_fences(raw))


# --- Graph node --------------------------------------------------------------


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    ticker = state.get("ticker", "")
    thesis = state.get("thesis") or {}
    errors: list[str] = []

    # Step 1 — yfinance fetch (sync, off-loop)
    try:
        financials = await asyncio.to_thread(get_financials, ticker)
    except Exception as e:
        logger.error(f"[fundamentals] yfinance fetch failed for {ticker}: {e}")
        errors.append(f"yfinance: {e}")
        financials = {}
    if financials.get("errors"):
        errors.extend(financials["errors"])

    kpis = compute_kpis(financials)

    # Step 2 — LLM call (sync OpenAI client, off-loop)
    out: FundamentalsOutput
    if not kpis:
        out = FundamentalsOutput(
            summary=f"No fundamentals data available for {ticker}.",
            kpis={},
            projections=_derive_fallback_projections({}),  # all null-hypothesis
            errors=errors + ["no kpis computed; skipping LLM call"],
        )
    else:
        try:
            llm_out = await asyncio.to_thread(_call_llm, ticker, thesis, kpis)
            out = FundamentalsOutput.model_validate(llm_out)
            out.errors = errors  # propagate yfinance errors even on LLM success
        except Exception as e:
            logger.error(f"[fundamentals] LLM call failed for {ticker}: {e}")
            errors.append(f"llm: {e}")
            out = FundamentalsOutput(
                summary=f"LLM analysis failed for {ticker}; using history-derived fallback.",
                kpis=kpis,
                projections=_derive_fallback_projections(kpis),
                errors=errors,
            )

    return {
        "fundamentals": out.model_dump(),
        "messages": [
            {
                "node": NODE,
                "event": "completed",
                "started_at": started_at,
                "completed_at": time.perf_counter(),
            }
        ],
    }


# --- CLI ---------------------------------------------------------------------


def _load_thesis_for_cli(slug: str) -> dict:
    return json.loads(Path(f"theses/{slug}.json").read_text())


async def _cli(ticker: str, thesis_slug: str = "ai_cake") -> None:
    thesis = _load_thesis_for_cli(thesis_slug)
    state: FinaqState = {"ticker": ticker, "thesis": thesis}
    result = await run(state)
    print(json.dumps(result["fundamentals"], indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agents.fundamentals TICKER [thesis_slug]", file=sys.stderr)
        sys.exit(1)
    ticker = sys.argv[1].upper()
    thesis_slug = sys.argv[2] if len(sys.argv) > 2 else "ai_cake"
    asyncio.run(_cli(ticker, thesis_slug))
