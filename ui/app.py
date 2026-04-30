"""FINAQ Streamlit dashboard — main entrypoint.

Layout (CLAUDE.md §12, palette §13):

  Sidebar
    - Thesis dropdown (loads /theses/*.json)
    - Ticker input
    - Run drill-in (full graph) button
    - Run scan (Phase 0 fixture) button
    - Direct Agent shortcut → /direct_agent page link

  Main
    - Header: TICKER · thesis name · confidence badge
    - "What this means" section pulled from synthesis report
    - Bull / Bear / Top risks / Action / Watchlist sections
    - Monte Carlo histogram + scenario card + KPI grid
    - Per-agent expanders (Fundamentals · Filings · News · Risk)

  Footer
    - Download PDF
    - URL query-param deeplinks (Step 10 Telegram → tap → opens this view)

Pre-cached demo runs in `data_cache/demos/` are loaded instantly when the
ticker + thesis match a cached run — keeps the demo path under 2 seconds.
"""

from __future__ import annotations

# Bootstrap: ensure the project root is on sys.path so `from ui.* import ...`
# and `from agents.* import ...` resolve when Streamlit runs this script
# directly. Streamlit puts only the script's *directory* on sys.path, not
# the project root. Standard pattern for multi-file Streamlit apps.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import json
import re
import time
from datetime import date

import numpy as np
import streamlit as st

from ui.components import (
    agent_expander,
    hero_strip,
    kpi_grid,
    mc_chart,
    news_split,
    page_header,
    risk_gauge,
    scenario_card,
    section_divider,
    top_risks_chips,
    watchlist_card,
)
from utils.pdf_export import export as export_pdf

# --- Constants --------------------------------------------------------------

THESES_DIR = Path(__file__).parents[1] / "theses"
DEMO_DIR = Path(__file__).parents[1] / "data_cache" / "demos"
FIXTURES_DIR = Path(__file__).parents[1] / "data_cache" / "fixtures"
TRIAGE_FIXTURE = FIXTURES_DIR / "triage_alerts.json"

# Sections we extract from the synthesis markdown to render natively (instead
# of dumping the whole markdown blob into st.markdown). Keeps each section
# stylable and lets us slot the MC chart between MC-paragraph and scenarios.
_SECTION_NAMES = (
    "What this means",
    "Thesis statement",
    "Bull case",
    "Bear case",
    "Top risks",
    "Monte Carlo fair value",
    "Action recommendation",
    "Watchlist",
    "Evidence",
)

st.set_page_config(
    page_title="FINAQ — equity research advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helpers ----------------------------------------------------------------


@st.cache_data(show_spinner=False)
def list_thesis_slugs() -> list[str]:
    return sorted(p.stem for p in THESES_DIR.glob("*.json"))


@st.cache_data(show_spinner=False)
def load_thesis(slug: str) -> dict:
    return json.loads((THESES_DIR / f"{slug}.json").read_text())


def _section(md: str, header: str) -> str:
    lines = md.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"## {header}")), None
    )
    if start is None:
        return ""
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1 :], start=start + 1)
            if line.startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end]).strip()


def _confidence_from_markdown(md: str) -> str:
    m = re.search(r"\*\*Confidence:\*\*\s+(low|medium|high)", md, re.IGNORECASE)
    return m.group(1).lower() if m else "medium"


def _demo_path(ticker: str, thesis_slug: str) -> Path:
    return DEMO_DIR / f"{ticker.upper()}__{thesis_slug}.json"


def _try_load_demo(ticker: str, thesis_slug: str) -> dict | None:
    path = _demo_path(ticker, thesis_slug)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_demo(ticker: str, thesis_slug: str, final_state: dict) -> Path:
    """Persist a freshly-run drill-in so it can be re-rendered without
    re-running the graph. We strip the `messages` list to keep the file
    small (Streamlit doesn't need per-node timings to render)."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in final_state.items() if k != "messages"}
    path = _demo_path(ticker, thesis_slug)
    path.write_text(json.dumps(payload, default=str, indent=2))
    return path


def _run_full_graph(ticker: str, thesis_slug: str) -> dict:
    """Invoke the full LangGraph drill-in with state.db telemetry. Returns
    the final state. Slow (~3-5 min on a real run); the caller renders
    progress via st.status."""
    from agents import build_graph, invoke_with_telemetry

    thesis = load_thesis(thesis_slug)
    graph = build_graph()
    return asyncio.run(
        invoke_with_telemetry(graph, {"ticker": ticker.upper(), "thesis": thesis})
    )


def _format_currency(v: object) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


# --- Sidebar ---------------------------------------------------------------


def render_sidebar() -> dict:
    """Returns the user's current sidebar selection: ticker, thesis_slug,
    run_drill flag, run_scan flag, use_cached_demo flag.

    Default-ticker behaviour (real bug fixed late Step 8): when the user
    changes the thesis, the ticker is auto-set to the thesis's first anchor
    (NVDA → CAT for Construction, etc.) — picking a Construction thesis
    while still showing NVDA was a cross-thesis mismatch. Universe chips
    below the input let the user click any ticker the thesis tracks.
    """
    st.sidebar.markdown("### Drill-in")
    slugs = list_thesis_slugs()
    if not slugs:
        st.sidebar.error("No theses found. Add JSON files to /theses/.")
        return {"ticker": "", "thesis_slug": ""}

    # Read query params first — Telegram links land here.
    params = st.query_params
    default_thesis = params.get("thesis") or slugs[0]

    thesis_slug = st.sidebar.selectbox(
        "Thesis",
        slugs,
        index=slugs.index(default_thesis) if default_thesis in slugs else 0,
        key="sidebar_thesis",
    )

    # Resolve the active thesis once so we can default + render universe.
    thesis = load_thesis(thesis_slug)
    anchors = thesis.get("anchor_tickers") or []
    universe = thesis.get("universe") or anchors

    # Streamlit antipattern: passing BOTH `value=` and writing to
    # `session_state[key]` for the same widget triggers a warning. The
    # correct pattern is to write `session_state[key]` BEFORE the widget
    # is created (which Streamlit treats as initialization), then call
    # the widget WITHOUT a `value=` parameter.
    seed_ticker = (params.get("ticker") or (anchors[0] if anchors else "")).upper()
    if "sidebar_ticker" not in st.session_state:
        st.session_state["sidebar_ticker"] = seed_ticker

    # If the user has changed the thesis, reset to the new thesis's first
    # anchor. Detected by comparing against the last-seen thesis_slug.
    last_thesis = st.session_state.get("_last_thesis_slug")
    if last_thesis is not None and last_thesis != thesis_slug:
        st.session_state["sidebar_ticker"] = (anchors[0] if anchors else "").upper()
    st.session_state["_last_thesis_slug"] = thesis_slug

    # Honour any chip-click that landed in session_state since last rerun.
    if "_chip_pick" in st.session_state:
        st.session_state["sidebar_ticker"] = st.session_state.pop("_chip_pick")

    ticker = st.sidebar.text_input("Ticker", key="sidebar_ticker").upper()

    # Universe chips — anchors highlighted, click to select.
    if universe:
        st.sidebar.caption(f"{thesis.get('name', thesis_slug)} universe")
        chip_cols = st.sidebar.columns(min(4, len(universe)))
        for i, t in enumerate(universe):
            label = f"⭐ {t}" if t in anchors else t
            if chip_cols[i % len(chip_cols)].button(
                label,
                key=f"chip_{thesis_slug}_{t}",
                use_container_width=True,
            ):
                st.session_state["_chip_pick"] = t
                st.rerun()

    use_cached = st.sidebar.toggle(
        "Use cached demo if available",
        value=True,
        help="Faster preview — skips the LLM run if a cached drill-in exists.",
    )

    cols = st.sidebar.columns(2)
    run_drill = cols[0].button("🔍 Run drill-in", use_container_width=True)
    run_scan = cols[1].button("🛰️ Run scan", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pages")
    # `page_link` paths are relative to the entrypoint's directory (ui/),
    # not the project root. Streamlit auto-discovers pages from `pages/*.py`
    # so we only need to add the entrypoint itself by hand.
    st.sidebar.page_link("app.py", label="📊 Dashboard")
    return {
        "ticker": ticker,
        "thesis_slug": thesis_slug,
        "run_drill": run_drill,
        "run_scan": run_scan,
        "use_cached": use_cached,
    }


# --- Run helpers ------------------------------------------------------------


def _execute_drill(ticker: str, thesis_slug: str, use_cached: bool) -> dict:
    """Resolve the final state for the ticker × thesis pair. Either pulls
    from the cached demo or invokes the LangGraph end-to-end run, with
    progress streamed via `st.status()`."""
    if use_cached:
        cached = _try_load_demo(ticker, thesis_slug)
        if cached:
            st.toast("Loaded cached demo run.", icon="⚡")
            return cached

    with st.status(f"Running drill-in on {ticker} · {thesis_slug}", expanded=True) as status:
        status.update(label="Building graph + loading thesis", state="running")
        t0 = time.perf_counter()
        final = _run_full_graph(ticker, thesis_slug)
        elapsed = time.perf_counter() - t0
        status.update(
            label=f"Drill-in complete ({elapsed:.1f}s)",
            state="complete",
            expanded=False,
        )
    saved = _save_demo(ticker, thesis_slug, final)
    st.toast(f"Cached run saved to {saved.name}", icon="💾")
    return final


def _render_triage_fixture() -> None:
    """Phase 0 stand-in for the Triage agent (Step 11)."""
    if not TRIAGE_FIXTURE.exists():
        st.warning("No triage fixture found at data_cache/fixtures/triage_alerts.json.")
        return
    alerts = json.loads(TRIAGE_FIXTURE.read_text())
    st.subheader("Triage scan results")
    st.caption(
        "**Phase 0 fixture** — replaced by the real Triage agent in Step 11. "
        "These alerts are illustrative only."
    )
    if not alerts:
        st.info("No alerts.")
        return
    for a in alerts:
        st.markdown(
            f"**{a.get('ticker', '?')}** · _{a.get('thesis', '?')}_ · "
            f"severity {a.get('severity', '?')} · {a.get('signal', '')}"
        )
        if a.get("evidence_url"):
            st.markdown(f"  - [evidence]({a['evidence_url']})")


# --- Main render -----------------------------------------------------------


def _resolve_mc_samples(mc: dict) -> list[float]:
    """Return the MC sample array. If the cache stored only percentiles
    (no `samples` field), regenerate a visually-similar normal distribution
    from the P10/P50/P90 spread so the histogram still renders."""
    samples = mc.get("samples")
    if samples is not None and len(samples) > 0:
        return list(samples)
    dcf = mc.get("dcf") or {}
    if dcf:
        lo = dcf.get("p10") or 0
        hi = dcf.get("p90") or 0
        mid = dcf.get("p50") or (lo + hi) / 2
        std = max((hi - lo) / 2.6, 1.0)
        return list(np.random.normal(loc=mid, scale=std, size=8000))
    return []


def render_dashboard_view(state: dict) -> None:
    """The visual dashboard — what the user sees first.

    Storytelling order, top to bottom:
      1. Hero (TICKER · price · valuation badge · confidence)
      2. **Side-by-side**: Key Metrics (left, ~40%) + MC chart (right, ~60%).
         The chart was previously below the metrics — required scrolling.
         Now both fit above the fold on a typical laptop viewport.
      3. Bull/Base/Bear scenario cards (full-width row beneath)
      4. Risk dashboard — gauge + top-3 risks
      5. Catalysts vs Concerns — split view of news drivers
      6. Watchlist — forward-looking events to track
      7. Plain-English takeaway (what this means)
    """
    ticker = state.get("ticker", "?")
    thesis = state.get("thesis") or {}
    thesis_name = thesis.get("name", "unknown")
    report = state.get("report") or ""
    confidence = state.get("synthesis_confidence") or _confidence_from_markdown(report)
    fund = state.get("fundamentals") or {}
    kpis = fund.get("kpis") or {}
    mc = state.get("monte_carlo") or {}
    risk = state.get("risk") or {}
    news = state.get("news") or {}

    mc_ran = bool(mc) and mc.get("method") not in (None, "skipped")

    # 1. Hero
    hero_strip(
        ticker=ticker,
        thesis_name=thesis_name,
        current_price=kpis.get("current_price"),
        mc=mc,
        confidence=confidence,
    )
    section_divider()

    # 2. Side-by-side: Key Metrics (left) + MC chart (right).
    metrics_col, chart_col = st.columns([5, 7], gap="large")
    with metrics_col:
        st.markdown("### Key metrics")
        # 2 columns inside the narrower left half — keeps each card readable
        # while still showing 6+ KPIs in roughly the same vertical span as
        # the chart on the right.
        kpi_grid(kpis, columns=2)
    with chart_col:
        st.markdown("### Fair-value distribution")
        if mc_ran:
            samples = _resolve_mc_samples(mc)
            if samples:
                mc_chart(
                    samples,
                    current_price=mc.get("current_price"),
                    caption=(
                        f"Discount rate {mc.get('discount_rate_used', '?'):.2%} · "
                        f"Convergence ratio {mc.get('convergence_ratio', '?'):.2f}"
                        if isinstance(mc.get("discount_rate_used"), (int, float))
                        and isinstance(mc.get("convergence_ratio"), (int, float))
                        else None
                    ),
                )
        else:
            st.warning(
                "Monte Carlo was skipped on this run. "
                f"Reason: `{(mc.get('errors') or ['(unknown)'])[0]}`. "
                "Check the **Per-agent details** tab to see which Fundamentals "
                "KPIs were missing."
            )

    # 3. Scenario cards on a full-width row beneath the side-by-side block.
    if mc_ran:
        st.markdown("")  # spacer
        scenario_card(mc)
    section_divider()

    # 4. Risk
    st.markdown("### Risk view")
    rcol_a, rcol_b = st.columns([1, 2])
    with rcol_a:
        risk_gauge(risk.get("level"), risk.get("score_0_to_10"))
        if risk.get("summary"):
            st.markdown("")
            st.caption(risk["summary"])
    with rcol_b:
        st.markdown("**Top risks**")
        top_risks_chips(risk.get("top_risks") or [], limit=5)
    section_divider()

    # 5. Catalysts vs concerns
    st.markdown("### News drivers")
    news_split(news.get("catalysts") or [], news.get("concerns") or [])
    section_divider()

    # 6. Watchlist
    wcol, acol = st.columns([2, 3])
    with wcol:
        st.markdown("### 👀 Watchlist")
        st.caption("Forward-looking signals to track before the next drill-in.")
        watchlist_card(state.get("watchlist") or [])
    with acol:
        st.markdown("### 🎯 Action")
        action = _section(report, "Action recommendation")
        st.markdown(action or "_(no recommendation)_")

    section_divider()

    # 7. Plain English summary
    what_this_means = _section(report, "What this means")
    if what_this_means:
        st.markdown("### 💡 In plain English")
        st.info(what_this_means)


def render_report_view(state: dict) -> None:
    """The full Synthesis markdown report — for the user who wants the
    institutional brief end-to-end. This is what the PDF export captures."""
    report = state.get("report") or ""
    if not report.strip():
        st.warning("No synthesis report produced for this run.")
        return
    st.markdown(report)


def render_agent_details_view(state: dict) -> None:
    """Raw per-agent structured outputs in expanders — for drill-down debugging."""
    st.caption(
        "Each agent's full structured output. Useful when something in the "
        "Dashboard or Report seems off — drill in to see the raw payload."
    )
    for agent_name in ("fundamentals", "filings", "news", "risk"):
        agent_expander(agent_name, state.get(agent_name) or {})

    section_divider()
    mc = state.get("monte_carlo") or {}
    with st.expander("monte_carlo — full output"):
        if mc:
            st.json({k: v for k, v in mc.items() if k != "samples"})
            samples = mc.get("samples")
            if samples:
                st.caption(f"samples array elided ({len(samples)} draws)")
        else:
            st.caption("No Monte Carlo output for this run.")

    section_divider()
    st.markdown("**Top-level evidence (from synthesis):**")
    evidence_md = _section(state.get("report") or "", "Evidence")
    if evidence_md:
        st.markdown(evidence_md)
    else:
        st.caption("No Evidence section in the synthesis report.")


def render_pdf_download(state: dict) -> None:
    """A footer-style PDF download. Available regardless of which tab is open."""
    report = state.get("report") or ""
    if not report.strip():
        return
    ticker = state.get("ticker", "?")
    thesis = state.get("thesis") or {}
    confidence = state.get("synthesis_confidence") or _confidence_from_markdown(report)
    mc = state.get("monte_carlo") or {}
    fund = state.get("fundamentals") or {}
    kpis = fund.get("kpis") or {}
    try:
        tmp_pdf = _demo_path(ticker, "pdf").with_suffix(".pdf")
        tmp_pdf.parent.mkdir(parents=True, exist_ok=True)
        export_pdf(
            report,
            tmp_pdf,
            mc_samples=mc.get("samples") if mc else None,
            current_price=kpis.get("current_price"),
            kpis=kpis,
            confidence=confidence,
        )
        with tmp_pdf.open("rb") as f:
            st.download_button(
                label="📄 Download PDF report",
                data=f.read(),
                file_name=f"{ticker}_{thesis.get('name', 'thesis').replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        st.error(f"PDF export failed: {e}")


def render_report(state: dict) -> None:
    """Top-level renderer: header → 3 tabs (Dashboard / Report / Per-agent)
    → PDF download footer.

    The visual Dashboard is the FIRST thing the user sees — institutional
    research conventions put the answer up front (price + valuation badge
    + confidence) and let the reader drill into supporting detail."""
    ticker = state.get("ticker", "?")
    thesis = state.get("thesis") or {}
    thesis_name = thesis.get("name", "unknown")

    page_header(
        f"{ticker} — {thesis_name}",
        subtitle=f"Drill-in as of {date.today().isoformat()}",
    )

    tab_dashboard, tab_report, tab_details = st.tabs(
        ["📊 Dashboard", "📄 Full report", "🔍 Per-agent details"]
    )
    with tab_dashboard:
        render_dashboard_view(state)
    with tab_report:
        render_report_view(state)
    with tab_details:
        render_agent_details_view(state)

    section_divider()
    render_pdf_download(state)


def main() -> None:
    sel = render_sidebar()

    # Route the user's action
    if sel.get("run_scan"):
        _render_triage_fixture()
        return

    state: dict | None = None
    if sel.get("run_drill"):
        if not sel["ticker"]:
            st.warning("Enter a ticker first.")
            return
        state = _execute_drill(
            sel["ticker"], sel["thesis_slug"], use_cached=sel.get("use_cached", True)
        )
    else:
        # Default view — load cached demo if it exists for the current selection
        cached = _try_load_demo(sel["ticker"], sel["thesis_slug"]) if sel.get("ticker") else None
        if cached:
            state = cached
        else:
            page_header(
                "FINAQ",
                subtitle=(
                    "Personal equity research advisor. "
                    "Pick a thesis, enter a ticker, and click Run drill-in."
                ),
            )
            st.info(
                "No cached drill-in for this ticker × thesis. "
                "Click **Run drill-in** in the sidebar to generate one."
            )
            return

    if state is not None:
        render_report(state)


if __name__ == "__main__":
    main()
