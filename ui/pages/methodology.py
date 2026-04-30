"""Methodology page — auditable view of how the model values stocks.

Three panels:
  1. The full FINANCE_ASSUMPTIONS.md document, rendered.
  2. Per-thesis valuation parameters table — equity_risk_premium,
     terminal_growth_rate, discount_rate floor/cap, with the `_basis`
     strings shown so the user sees *why* each value was chosen.
  3. Most recent drill-in's MC inputs (revenue/margin/multiple projections
     from the Fundamentals agent) — only when a cached demo exists.

This page makes the model auditable: the user can always see "you used a
9.5% discount rate because Treasury was 4.5% + 5% ERP for AI cake, and the
prompt derived these projections from these specific KPIs." That's how
trust gets built.
"""

from __future__ import annotations

# Bootstrap (see ui/app.py for explanation).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import streamlit as st

from ui.components import page_header, section_divider

st.set_page_config(page_title="FINAQ — Methodology", page_icon="📐", layout="wide")

THESES_DIR = Path(__file__).parents[2] / "theses"
DEMO_DIR = Path(__file__).parents[2] / "data_cache" / "demos"
DOCS_DIR = Path(__file__).parents[2] / "docs"


@st.cache_data(show_spinner=False)
def _load_theses() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(THESES_DIR.glob("*.json"))}


@st.cache_data(show_spinner=False)
def _load_doc() -> str:
    path = DOCS_DIR / "FINANCE_ASSUMPTIONS.md"
    return path.read_text() if path.exists() else "(FINANCE_ASSUMPTIONS.md not found)"


def _render_valuation_table(slug: str, thesis: dict) -> None:
    val = thesis.get("valuation") or {}
    if not val:
        st.caption(f"No `valuation` block on thesis `{slug}`.")
        return

    rows = [
        (
            "Equity risk premium",
            f"{val.get('equity_risk_premium', '?'):.2%}"
            if isinstance(val.get("equity_risk_premium"), (int, float))
            else "—",
            val.get("erp_basis", ""),
        ),
        (
            "Terminal growth rate",
            f"{val.get('terminal_growth_rate', '?'):.2%}"
            if isinstance(val.get("terminal_growth_rate"), (int, float))
            else "—",
            val.get("terminal_growth_basis", ""),
        ),
        (
            "Discount rate floor",
            f"{val.get('discount_rate_floor', '?'):.2%}"
            if isinstance(val.get("discount_rate_floor"), (int, float))
            else "—",
            "Lower bound applied to Treasury + ERP — prevents unrealistically generous WACC during low-rate cycles.",
        ),
        (
            "Discount rate cap",
            f"{val.get('discount_rate_cap', '?'):.2%}"
            if isinstance(val.get("discount_rate_cap"), (int, float))
            else "—",
            "Upper bound applied to Treasury + ERP — prevents unrealistically punitive WACC during high-rate cycles.",
        ),
    ]
    for label, value, basis in rows:
        with st.container(border=True):
            cols = st.columns([2, 1, 5])
            cols[0].markdown(f"**{label}**")
            cols[1].markdown(f"`{value}`")
            cols[2].caption(basis or "_(no rationale provided)_")


def _render_thesis_universe(thesis: dict) -> None:
    universe = thesis.get("universe") or []
    anchors = set(thesis.get("anchor_tickers") or [])
    if not universe:
        st.caption("No universe defined.")
        return
    chips = []
    for ticker in universe:
        is_anchor = ticker in anchors
        bg = "#2D4F3A" if is_anchor else "#FBF5E8"
        fg = "#FFFFFF" if is_anchor else "#1A1611"
        chips.append(
            f"<span style='background:{bg}; color:{fg}; padding:0.2rem 0.7rem; "
            f"border-radius:999px; margin:0.2rem; font-size:0.85rem; "
            f"font-weight:600; display:inline-block;'>{ticker}"
            f"{' ★' if is_anchor else ''}</span>"
        )
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.caption("★ anchor ticker — the load-bearing names for this thesis.")


def _render_thresholds(thesis: dict) -> None:
    thresholds = thesis.get("material_thresholds") or []
    if not thresholds:
        st.caption("No material thresholds declared.")
        return
    for t in thresholds:
        signal = t.get("signal", "?")
        op = t.get("operator", "?")
        value = t.get("value", "?")
        unit = t.get("unit", "")
        st.markdown(
            f"- **`{signal}`** {op} `{value}` _{unit}_"
        )


def _render_current_run_inputs(slug: str) -> None:
    """If a cached demo exists for any ticker × this thesis, show the most
    recent run's MC inputs (Fundamentals projections)."""
    if not DEMO_DIR.exists():
        st.caption("No cached demo runs yet — run a drill-in to populate.")
        return
    matching = sorted(DEMO_DIR.glob(f"*__{slug}.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matching:
        st.caption(f"No cached drill-in for thesis `{slug}` yet.")
        return
    latest = matching[0]
    state = json.loads(latest.read_text())
    ticker = state.get("ticker", "?")
    fund = state.get("fundamentals") or {}
    mc = state.get("monte_carlo") or {}
    proj = fund.get("projections") or {}
    st.markdown(f"#### Most recent drill-in: **{ticker}** · _{latest.stem}_")
    if not proj:
        st.caption("No projections persisted in this cached run.")
        return
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Fundamentals projections (MC inputs)**")
        st.json(proj)
    with cols[1]:
        st.markdown("**Monte Carlo computed parameters**")
        if mc and mc.get("method") not in (None, "skipped"):
            st.markdown(
                f"- **Method:** {mc.get('method', '?')}\n"
                f"- **Discount rate used:** "
                f"{mc.get('discount_rate_used', '?'):.2%}"
                f" (Treasury + ERP, clipped to thesis band)\n"
                f"- **Terminal growth used:** "
                f"{mc.get('terminal_growth_used', '?'):.2%}\n"
                f"- **Convergence ratio:** {mc.get('convergence_ratio', '?')}\n"
                f"- **n_sims · n_years:** {mc.get('n_sims', '?')} × {mc.get('n_years', '?')}"
                if isinstance(mc.get("discount_rate_used"), (int, float))
                else "MC ran but inputs not all numeric."
            )
        else:
            st.caption("MC was skipped on this run.")


def _render_dcf_formula() -> None:
    """LaTeX rendering of the Owner-Earnings DCF formula. Streamlit supports
    LaTeX via `st.latex`."""
    st.markdown("### Owner-Earnings DCF")
    st.markdown(
        "Owner Earnings (Buffett-style) is the cash a business produces for "
        "its owners after maintenance capex but before growth investment:"
    )
    st.latex(r"OE_t = NI_t + D\&A_t - \text{maintenance\_capex}_t")
    st.markdown("Discounted to present value over a 10-year horizon:")
    st.latex(r"PV = \sum_{t=1}^{10} \frac{OE_t}{(1+r)^t} + \frac{TV}{(1+r)^{10}}")
    st.markdown("Terminal value via Gordon growth:")
    st.latex(r"TV = \frac{OE_{10} \cdot (1+g)}{r - g}")
    st.caption(
        "Where r is the discount rate (Treasury + thesis ERP, clipped) and "
        "g is the per-thesis terminal growth rate. Both are documented per "
        "thesis above with their `_basis` rationales."
    )


def main() -> None:
    page_header(
        "Methodology",
        subtitle=(
            "How FINAQ values stocks, what assumptions it makes, and why. "
            "Per-thesis parameters with their rationale; current-run MC inputs."
        ),
    )

    # Per-thesis valuation panels
    st.markdown("## Per-thesis valuation parameters")
    theses = _load_theses()
    if not theses:
        st.warning("No theses found in `/theses`.")
    else:
        slugs = list(theses.keys())
        chosen = st.selectbox("Thesis", slugs)
        thesis = theses[chosen]
        st.markdown(f"### {thesis.get('name', chosen)}")
        st.markdown(f"_{thesis.get('summary', '')}_")
        section_divider()

        st.markdown("#### Universe")
        _render_thesis_universe(thesis)
        section_divider()

        st.markdown("#### Discount-rate + terminal-growth parameters")
        _render_valuation_table(chosen, thesis)
        section_divider()

        st.markdown("#### Material thresholds")
        st.caption(
            "Phase 1 Triage uses these as monitoring rules. Synthesis "
            "references them by name in conditional action recommendations."
        )
        _render_thresholds(thesis)
        section_divider()

        st.markdown("#### Most recent run inputs")
        _render_current_run_inputs(chosen)

    section_divider()

    # Owner-Earnings DCF formula
    _render_dcf_formula()

    section_divider()

    # Full methodology document
    st.markdown("## Full methodology document")
    st.caption(
        "Source of truth: `docs/FINANCE_ASSUMPTIONS.md`. Every parameter "
        "above traces back to a section here with citations to Penman, "
        "Damodaran, and Buffett."
    )
    st.markdown(_load_doc())


main()
