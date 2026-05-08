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
from datetime import date, datetime

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


def list_thesis_slugs() -> list[str]:
    """Glob `theses/*.json`. NOT cached — adhoc theses created via the
    Telegram `/analyze` flow land in this directory while the dashboard
    is running, and we want them to appear in the dropdown without the
    user having to tap "🔄 Reload (clear cache)". The glob is microseconds
    so the cache wasn't worth the staleness."""
    return sorted(p.stem for p in THESES_DIR.glob("*.json"))


@st.cache_data(show_spinner=False)
def load_thesis(slug: str) -> dict:
    return json.loads((THESES_DIR / f"{slug}.json").read_text())


# `_md_safe` moved to `ui/components.py` (now `md_safe`) so the Direct Agent
# page can use the same escape. Aliased here so existing call sites stay
# unchanged.
from ui.components import md_safe as _md_safe  # noqa: E402


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


_RUN_ID_FILENAME_LEN = 8  # short prefix of UUID for filename brevity


def _demo_path(ticker: str, thesis_slug: str, run_id: str | None = None) -> Path:
    """Demo-cache filename. New format includes a short run_id suffix so each
    drill-in produces a NEW file (`__a1b2c3d4.json`) instead of overwriting.
    Legacy format (no run_id) is kept for backward-compat reads."""
    if run_id:
        return DEMO_DIR / f"{ticker.upper()}__{thesis_slug}__{run_id[:_RUN_ID_FILENAME_LEN]}.json"
    return DEMO_DIR / f"{ticker.upper()}__{thesis_slug}.json"


def _list_run_history(ticker: str, thesis_slug: str) -> list[dict]:
    """All cached drill-ins for this ticker × thesis, newest first.

    Each entry: `{run_id, path, mtime, started_at, confidence, convergence}`.
    Uses file mtime for ordering (no state.db cross-reference needed) and
    pulls the inline metadata from each file's JSON for display.
    """
    if not DEMO_DIR.exists():
        return []
    pattern = f"{ticker.upper()}__{thesis_slug}__*.json"
    files = sorted(DEMO_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    legacy = DEMO_DIR / f"{ticker.upper()}__{thesis_slug}.json"
    if legacy.exists():
        files.append(legacy)  # show legacy as oldest entry
    out: list[dict] = []
    for path in files:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        mc = data.get("monte_carlo") or {}
        out.append(
            {
                "run_id": data.get("run_id") or "(legacy)",
                "path": path,
                "mtime": path.stat().st_mtime,
                "confidence": data.get("synthesis_confidence", "?"),
                "convergence": (mc.get("convergence_ratio") if mc else None),
                "method": (mc.get("method") if mc else None),
            }
        )
    return out


def _try_load_demo(
    ticker: str, thesis_slug: str, run_id: str | None = None
) -> dict | None:
    """Load a cached drill-in.

    - If `run_id` is given: load that specific run's file.
    - If not: pick the most recent file for this ticker × thesis (across
      both new run_id-keyed format and legacy single-file format).
    """
    if run_id:
        path = _demo_path(ticker, thesis_slug, run_id=run_id)
        if path.exists():
            return json.loads(path.read_text())
        return None
    history = _list_run_history(ticker, thesis_slug)
    if not history:
        return None
    return json.loads(history[0]["path"].read_text())


def _save_demo(ticker: str, thesis_slug: str, final_state: dict) -> Path:
    """Persist a freshly-run drill-in so it can be re-rendered without
    re-running the graph. Files are keyed by run_id so each drill-in is
    preserved as its own snapshot — the previous run is no longer
    overwritten. We strip the `messages` list to keep the file small."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in final_state.items() if k != "messages"}
    run_id = final_state.get("run_id")
    path = _demo_path(ticker, thesis_slug, run_id=run_id)
    path.write_text(json.dumps(payload, default=str, indent=2))
    return path


def _run_full_graph(ticker: str, thesis_slug: str) -> dict:
    """Invoke the full LangGraph drill-in with state.db telemetry. Returns
    the final state. Slow (~3-5 min on a real run); the caller renders
    progress via st.status."""
    from agents import build_graph, invoke_with_telemetry

    thesis = load_thesis(thesis_slug)
    # Step 11.20 — see ui/_runner.py for the equivalent fix; the CIO's
    # cooldown gate joins by slug, so graph_runs.thesis must hold slug.
    if isinstance(thesis, dict):
        thesis.setdefault("slug", thesis_slug)
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

    # Run-history dropdown: lets the user load any past drill-in for this
    # ticker × thesis, not just the latest. Without this, the dashboard
    # always shows the most recent run and previous snapshots are inaccessible.
    history = _list_run_history(ticker, thesis_slug) if ticker else []
    history_choice: str | None = None
    # If the URL has `?run_id=...`, the user tapped a Telegram link
    # pointing at a specific snapshot — preselect that history entry so
    # the dashboard auto-loads exactly the run they were looking at,
    # not just the latest.
    url_run_id = (params.get("run_id") or "").strip()
    if history:
        labels = ["Latest"]
        # Build labels first so we can compute the default index that
        # matches `url_run_id` (if any).
        history_index_default = 0
        for i, entry in enumerate(history):
            ts = datetime.fromtimestamp(entry["mtime"]).strftime("%Y-%m-%d %H:%M")
            conv = entry.get("convergence")
            conv_str = f"conv={conv:.2f}" if isinstance(conv, (int, float)) else "—"
            labels.append(
                f"{ts} · {entry.get('confidence', '?')} · {conv_str} "
                f"· {entry['run_id'][:8]}"
            )
            # `url_run_id` may be the full UUID OR just the 8-char prefix
            # the bot embeds in URLs. Match both.
            if url_run_id and (
                entry["run_id"] == url_run_id
                or entry["run_id"].startswith(url_run_id)
            ):
                history_index_default = i + 1  # +1 because labels[0] is "Latest"
        if url_run_id and history_index_default == 0:
            # URL had a run_id but it doesn't match any saved file. Tell
            # the user instead of silently falling through to the latest.
            st.sidebar.warning(
                f"Requested run_id `{url_run_id[:8]}` not found on disk — "
                f"showing the latest instead."
            )
        picked = st.sidebar.selectbox(
            "Run history",
            labels,
            index=history_index_default,
            help="Load a past drill-in for this ticker × thesis. "
            "Each run is now stored as its own file (no overwriting).",
        )
        if picked != "Latest":
            # Map label back to the run_id portion (last token after '·').
            run_id_short = picked.split("·")[-1].strip()
            for entry in history:
                if entry["run_id"].startswith(run_id_short):
                    history_choice = entry["run_id"]
                    break

    use_cached = st.sidebar.toggle(
        "Auto-load latest cached run on first view",
        value=True,
        help=(
            "When ON, the dashboard auto-loads the most recent cached drill-in "
            "for this ticker × thesis as the default view. The 🔍 Run drill-in "
            "button ALWAYS runs the agents fresh — this toggle only affects the "
            "no-click default."
        ),
    )

    cols = st.sidebar.columns(2)
    run_drill = cols[0].button("🔍 Run drill-in", use_container_width=True)
    run_scan = cols[1].button("🛰️ Run scan", use_container_width=True)

    # Reload button: clears every @st.cache_data result + reruns the script.
    # Useful during development when a thesis JSON / source-data file has
    # changed and Streamlit's cache is serving stale results. Streamlit's
    # `runOnSave=true` auto-reruns on Python file edits but does NOT
    # invalidate cache_data; this button is the explicit nuke.
    if st.sidebar.button("🔄 Reload (clear cache)", use_container_width=True):
        st.cache_data.clear()
        st.toast("Cache cleared. Re-running…", icon="🔄")
        st.rerun()

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
        "history_run_id": history_choice,
    }


# --- Run helpers ------------------------------------------------------------


def _kick_off_drill(ticker: str, thesis_slug: str) -> None:
    """Start a drill-in in a background daemon thread and return immediately.

    The graph runs in `ui/_runner.py` so navigating to a different page
    (Architecture, Methodology, Mission Control, etc.) does NOT kill the
    in-flight `asyncio.run`. The dashboard polls `is_running()` on each
    rerun and renders a "🏃 Running…" panel until the cached state file
    appears on disk.
    """
    from ui._runner import is_running, kick_off_drill

    if is_running(ticker, thesis_slug):
        st.toast(
            f"Drill-in already running for {ticker} × {thesis_slug}.",
            icon="🏃",
        )
        return
    started = kick_off_drill(ticker, thesis_slug)
    if started:
        st.toast(
            f"Drill-in started for {ticker} × {thesis_slug}. "
            "You can switch tabs / pages — the run will continue in the background.",
            icon="🏃",
        )


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


# --- Running-status panel (background-thread runner) ----------------------


def _render_running_panel(ticker: str, thesis_slug: str) -> None:
    """Render a sticky 'Running…' panel while a background drill-in is in
    flight. Schedules an auto-rerun every 2s so elapsed time updates.

    This is what the user sees instead of the report when they've clicked
    Run drill-in but the agents aren't done yet. They can switch tabs /
    navigate to other pages and the run will keep going."""
    from ui._runner import elapsed_seconds, get_run_status

    elapsed = elapsed_seconds(ticker, thesis_slug) or 0.0
    rec = get_run_status(ticker, thesis_slug)
    error = rec.get("error") if rec else None

    page_header(
        f"🏃 Running drill-in on {ticker}",
        subtitle=f"Thesis: {thesis_slug} · Elapsed: {int(elapsed)}s",
    )

    if error:
        st.error(
            f"Drill-in failed: {error}\n\n"
            "Click 🔍 Run drill-in again to retry."
        )
        # Don't auto-rerun — let the user read the error and decide.
        from ui._runner import clear_record

        clear_record(ticker, thesis_slug)
        return

    st.info(
        "The drill-in is executing the LangGraph pipeline (Fundamentals → "
        "Filings → News → Risk → Monte Carlo → Synthesis). Expected "
        "duration: 60-180 seconds. **Switch tabs or pages freely — the "
        "run will continue in the background.** When complete, the report "
        "will load automatically."
    )

    # Per-stage progress: read state.db node_runs to show which agent is
    # currently running. Approximation — the run_id may not be set yet.
    from data import state as state_db

    rec_runs = state_db.recent_runs(limit=5)
    matching = [
        r
        for r in rec_runs
        if r["ticker"] == ticker.upper() and r["status"] == "running"
    ]
    if matching:
        run_id = matching[0]["run_id"]
        nodes = state_db.all_node_runs_for(run_id)
        if nodes:
            done = [n["node"] for n in nodes if n["status"] == "completed"]
            if done:
                st.success(f"✓ Completed: {', '.join(done)}")

    # Schedule a polling rerun. 2s is short enough that elapsed time feels
    # live, long enough to avoid hammering Streamlit's reactivity loop.
    time.sleep(2)
    st.rerun()


# --- Pre-flight ingest check (Filings corpus) ------------------------------


def _run_ingestion_for(ticker: str) -> tuple[bool, str]:
    """Synchronous wrapper around scripts.ingest_universe.ingest_ticker.
    Returns (success, message). Used by the dashboard's 'Ingest now' button.
    """
    import asyncio as _asyncio

    from scripts.ingest_universe import ingest_ticker

    try:
        chunks = _asyncio.run(ingest_ticker(ticker))
        if chunks > 0:
            return True, f"Ingested {chunks} chunks for {ticker}."
        return False, f"Ingestion ran but produced 0 chunks for {ticker}."
    except Exception as e:
        return False, f"Ingestion failed for {ticker}: {e}"


def _render_ingest_banner(ticker: str) -> bool:
    """If `ticker` isn't in ChromaDB, render a banner explaining what to do.

    Three branches:
      - Foreign issuer (e.g. TSM, ASML): files 20-F/6-K, not 10-K/10-Q.
        Re-running ingestion won't help — the kinds aren't supported yet.
        Render an info banner pointing at POSTPONED.md (no "Ingest now" button).
      - Not ingested + supported kinds expected: render the standard
        "📥 Ingest now" button to download + chunk + embed.
      - Already ingested: short-circuit, return False.

    Result is cached in session_state so we don't recheck ChromaDB on every
    rerun.
    """
    from data.chroma import has_ticker
    from data.edgar import has_filings_in_unsupported_kinds

    cache_key = f"_ingest_check::{ticker}"
    if st.session_state.get(cache_key) == "ingested":
        return False
    ingested = has_ticker(ticker)
    if ingested:
        st.session_state[cache_key] = "ingested"
        return False

    unsupported_kinds = has_filings_in_unsupported_kinds(ticker)

    if unsupported_kinds:
        # Foreign-issuer path — re-ingestion won't help.
        kinds_str = ", ".join(unsupported_kinds)
        st.markdown(
            f"""
            <div style="background:#FBF5E8; border:1px solid #E0D5C2;
                border-left:6px solid #E0D5C2; border-radius:6px;
                padding:1rem 1.2rem; margin-bottom:1rem;">
                <div style="color:#1A1611; font-weight:700; font-size:1rem;
                    letter-spacing:0.04em;">🌍 FOREIGN ISSUER · UNSUPPORTED CORPUS</div>
                <div style="color:#1A1611; margin-top:0.4rem; line-height:1.45;">
                    <b>{ticker}</b> files <b>{kinds_str}</b> with the SEC,
                    not 10-K/10-Q. The current ingest pipeline only handles
                    10-K + 10-Q, so {ticker}'s filings are NOT in ChromaDB.
                </div>
                <div style="color:#1A1611; opacity:0.75; font-size:0.85rem;
                    margin-top:0.5rem;">
                    Re-running <code>scripts.ingest_universe</code> will not
                    help. 20-F + 6-K support is tracked in
                    <code>docs/POSTPONED.md</code>. Drill-ins on {ticker} will
                    still run, but Filings will be empty and Synthesis will
                    flag this as a coverage gap.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return True

    # Standard path: ticker just hasn't been ingested yet.
    st.markdown(
        f"""
        <div style="background:#FBF5E8; border:1px solid #2D4F3A;
            border-left:6px solid #2D4F3A; border-radius:6px;
            padding:1rem 1.2rem; margin-bottom:1rem;">
            <div style="color:#2D4F3A; font-weight:700; font-size:1rem;
                letter-spacing:0.04em;">📥 INGEST REQUIRED</div>
            <div style="color:#1A1611; margin-top:0.4rem; line-height:1.45;">
                <b>{ticker}</b> isn't in ChromaDB yet. Filings retrieval
                will return zero chunks → Risk and Synthesis will run on a
                Filings-less state and the report will be incomplete.
            </div>
            <div style="color:#1A1611; opacity:0.75; font-size:0.85rem;
                margin-top:0.5rem;">
                Click below to download {ticker}'s recent 10-K + 10-Q from SEC
                EDGAR, chunk + embed them, and add to ChromaDB. ~5-10 min the
                first time; subsequent ingests of the same ticker are no-ops.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(f"📥 Ingest {ticker} now", type="primary", key=f"ingest_btn_{ticker}"):
        with st.status(f"Ingesting {ticker} — downloading + chunking + embedding…", expanded=True) as status:
            ok, msg = _run_ingestion_for(ticker)
            if ok:
                status.update(label=msg, state="complete")
                st.session_state[cache_key] = "ingested"
                st.toast(msg, icon="✅")
                st.rerun()
            else:
                status.update(label=msg, state="error")
                st.error(msg)
    return True


# --- Main render -----------------------------------------------------------


# Sample resolution moved to utils.charts.resolve_mc_samples so the bot's
# /drill MC-photo path can reuse the same logic as the dashboard. Aliased
# here for readability — the dashboard call site doesn't change shape.
from utils.charts import resolve_mc_samples as _resolve_mc_samples


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
            st.caption(_md_safe(risk["summary"]))
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
        st.markdown(_md_safe(action) if action else "_(no recommendation)_")

    section_divider()

    # 7. Plain English summary
    what_this_means = _section(report, "What this means")
    if what_this_means:
        st.markdown("### 💡 In plain English")
        st.info(_md_safe(what_this_means))


def render_report_view(state: dict) -> None:
    """The full Synthesis markdown report — for the user who wants the
    institutional brief end-to-end. This is what the PDF export captures."""
    report = state.get("report") or ""
    if not report.strip():
        st.warning("No synthesis report produced for this run.")
        return
    st.markdown(_md_safe(report))


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
        st.markdown(_md_safe(evidence_md))
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
    history_run_id = sel.get("history_run_id")  # may be None ("Latest")

    # Pre-flight check: if the selected ticker isn't ingested, render the
    # ingestion banner BEFORE the drill-in or cached-load logic. The banner
    # is non-blocking — the user can still click Run drill-in, but they're
    # warned that Filings will come back empty without ingestion.
    if sel.get("ticker") and history_run_id is None:
        # Skip the banner when loading a historical run — that run already
        # has its filings data baked in, and re-ingesting the ticker now
        # wouldn't change what the historical view shows.
        _render_ingest_banner(sel["ticker"])

    # If a run is in flight (started this session OR a previous tab) for the
    # current ticker × thesis, render the running panel instead of trying to
    # load a (possibly-stale) cached state. The panel auto-reruns every 2s.
    from ui._runner import is_running as _is_running

    if sel.get("ticker") and _is_running(sel["ticker"], sel["thesis_slug"]):
        _render_running_panel(sel["ticker"], sel["thesis_slug"])
        return

    if sel.get("run_drill"):
        if not sel["ticker"]:
            st.warning("Enter a ticker first.")
            return
        # Run drill-in always runs the agents fresh — the use_cached toggle
        # only governs the no-click default-view path below.
        _kick_off_drill(sel["ticker"], sel["thesis_slug"])
        # Render the running panel immediately; subsequent reruns will
        # poll until the file appears on disk.
        _render_running_panel(sel["ticker"], sel["thesis_slug"])
        return
    elif sel.get("use_cached", True):
        # No-click default view — load a cached demo if one exists. Honors
        # run-history selection: if the user picked an older run, load that
        # specific file; otherwise load the most recent file for this
        # ticker × thesis.
        cached = (
            _try_load_demo(sel["ticker"], sel["thesis_slug"], run_id=history_run_id)
            if sel.get("ticker")
            else None
        )
        if cached:
            state = cached
            if history_run_id:
                st.toast(
                    f"Loaded historical run {history_run_id[:8]}", icon="🕘"
                )
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
    else:
        # Auto-load toggle is OFF and no Run drill-in was clicked — show the
        # landing message instead of silently presenting stale state.
        page_header(
            "FINAQ",
            subtitle=(
                "Auto-load disabled. Click 🔍 Run drill-in to generate a "
                "fresh analysis."
            ),
        )
        return

    if state is not None:
        render_report(state)


if __name__ == "__main__":
    main()
