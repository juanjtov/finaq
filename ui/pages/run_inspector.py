"""Run Inspector — drill-down view of a single graph run (Step 10c.8).

Pick a run from the sidebar dropdown → see:
  - Run summary header (run_id, ticker, thesis, status, duration, total cost)
  - Per-node table (start/end/duration/status/tokens/cost/error)
  - Errors log scoped to this run_id
  - Per-agent payloads from the saved demo state (collapsible expanders)
  - LangSmith deep-link button so you jump from a node row to the
    full prompt + response for its underlying LLM call

The page uses ONLY data already persisted by the rest of the system:
  - `data_cache/state.db.graph_runs` — top-level row (run header)
  - `data_cache/state.db.node_runs` — per-node timeline + tokens/cost
  - `data_cache/state.db.errors` — error events scoped to run_id
  - `data_cache/demos/{TICKER}__{slug}__{run_id[:8]}.json` — agent payloads
  - LangSmith — deep-link only; no API calls (we don't have the LangSmith
    project's run_id mapping in our DB; the link uses our own run_id as a
    free-text search filter, which works because LangSmith indexes
    metadata tags by default).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os
from datetime import datetime

import streamlit as st

from data import state as state_db
from ui.components import (
    EGGSHELL,
    INK,
    SAGE,
    TAUPE,
    md_safe,
    page_header,
    section_divider,
)

st.set_page_config(
    page_title="FINAQ — Run Inspector",
    page_icon="🔍",
    layout="wide",
)

DEMO_DIR = Path(__file__).parents[2] / "data_cache" / "demos"


# --- Run discovery --------------------------------------------------------


def _list_runs() -> list[dict]:
    """Most-recent-first list of every run with enough metadata to label
    the sidebar dropdown. We DO NOT cache — the user wants the dropdown
    to reflect runs that just landed (e.g. they're inspecting a drill
    that just finished while the page is open)."""
    return state_db.recent_runs(limit=100)


def _load_demo_state(ticker: str, thesis_slug: str, run_id: str) -> dict | None:
    """Find the saved demo JSON matching this run_id. The runner saves to
    `{TICKER}__{slug}__{run_id[:8]}.json`, but legacy non-suffixed files
    may also exist — try both."""
    if not DEMO_DIR.exists():
        return None
    prefix = f"{ticker.upper()}__{thesis_slug}"
    short = run_id[:8] if run_id else ""
    # Most specific match first.
    candidates: list[Path] = []
    if short:
        exact = DEMO_DIR / f"{prefix}__{short}.json"
        if exact.exists():
            candidates.append(exact)
    if not candidates:
        # Fall back to any `{prefix}*.json` and pick the most-recently-modified
        candidates = list(DEMO_DIR.glob(f"{prefix}*.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(latest.read_text())
    except Exception as e:
        st.warning(f"Couldn't parse {latest.name}: {e}")
        return None


# --- LangSmith deep-link --------------------------------------------------


def _langsmith_url_for_run(run_id: str) -> str | None:
    """Build a LangSmith URL that filters the project view to this run_id.
    LangSmith indexes metadata tags, and our LangGraph runs propagate the
    `run_id` field — so a `metadata` filter on `run_id` finds the trace.

    Returns None when LANGSMITH_TRACING isn't enabled (no traces would
    exist).
    """
    if not os.environ.get("LANGSMITH_TRACING"):
        return None
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    # The user's project URL pattern is opaque to us — link to the project
    # page with a free-text search for the run_id, which falls back
    # gracefully even if LangSmith's URL scheme changes. Trade-off: the
    # user lands on the search results page rather than the exact trace.
    return (
        f"https://smith.langchain.com/projects/p/{project}"
        f"?searchModel=%7B%22filter%22%3A%22{run_id}%22%7D"
    )


# --- Render helpers -------------------------------------------------------


def _render_run_header(run: dict, total_cost: float, total_tokens: int) -> None:
    """Sage-bordered card with the run-level summary."""
    started = (run.get("started_at") or "")[:19].replace("T", " ")
    duration = run.get("duration_s") or 0.0
    status = run.get("status") or "?"
    confidence = run.get("confidence") or "—"
    ticker = run.get("ticker") or "?"
    thesis = run.get("thesis") or "?"
    run_id = run.get("run_id") or "?"
    err = run.get("error") or ""
    err_html = (
        f"<div style='color:#1A1611; opacity:0.8; margin-top:0.4rem;'>"
        f"⚠ {md_safe(err)}</div>"
        if err else ""
    )
    badge_color = SAGE if status == "completed" else TAUPE
    st.markdown(
        f"""
        <div style="background:{EGGSHELL}; border:1px solid {TAUPE};
            border-left:6px solid {badge_color}; border-radius:6px;
            padding:1rem 1.2rem; margin-bottom:1rem;">
            <div style="color:{INK}; font-size:1.05rem;">
                <b>{ticker}</b> × <code>{thesis}</code> · {status}
                · {duration:.1f}s · confidence <b>{confidence}</b>
            </div>
            <div style="color:{INK}; opacity:0.7; font-size:0.85rem;
                margin-top:0.3rem;">
                run_id <code>{run_id}</code> · started {started} UTC ·
                total cost <b>${total_cost:.4f}</b> · {total_tokens:,} tokens
            </div>
            {err_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_node_table(nodes: list[dict]) -> None:
    """Per-node timeline. Streamlit's `st.dataframe` gives us sortable
    columns + horizontal scroll for free."""
    if not nodes:
        st.caption("No node_runs recorded for this run.")
        return
    rows = []
    for n in nodes:
        rows.append({
            "node": n.get("node") or "?",
            "status": n.get("status") or "?",
            "duration_s": round(n.get("duration_s") or 0.0, 2),
            "n_calls": n.get("n_calls") or 0,
            "tokens_in": n.get("tokens_in") or 0,
            "tokens_out": n.get("tokens_out") or 0,
            "cost_usd": round(n.get("cost_usd") or 0.0, 4),
            "error": (n.get("error") or "")[:100],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_errors(errors: list[dict]) -> None:
    if not errors:
        st.caption("No errors logged for this run.")
        return
    for e in errors:
        ts = (e.get("ts") or "")[:19].replace("T", " ")
        agent = e.get("agent") or "?"
        msg = e.get("message") or ""
        st.markdown(
            f"<div style='border-left:3px solid {TAUPE}; padding:0.4rem 0.8rem; "
            f"margin-bottom:0.4rem; background:{EGGSHELL};'>"
            f"<b>{md_safe(agent)}</b> @ {md_safe(ts)}<br>"
            f"<code style='font-size:0.85rem;'>{md_safe(msg)}</code></div>",
            unsafe_allow_html=True,
        )


def _render_agent_payload(state: dict, agent: str) -> None:
    """One agent's saved output, rendered as JSON. Wrapped in an expander
    so the page doesn't sprawl. Falls back to "(empty)" if the agent
    didn't run."""
    payload = state.get(agent)
    with st.expander(f"{agent} — saved payload"):
        if not payload:
            st.caption("(no payload — agent didn't produce output)")
            return
        if isinstance(payload, dict):
            # Show errors first if present so the user sees them at the top.
            errors = payload.get("errors") or []
            if errors:
                st.error("Errors emitted by agent:")
                for err in errors:
                    st.markdown(f"- `{err}`")
            st.json(payload, expanded=False)
        else:
            st.write(payload)


# --- Main page ------------------------------------------------------------


def main() -> None:
    page_header(
        "🔍 Run Inspector",
        subtitle=(
            "Drill-down view of a single graph run: per-node timeline, "
            "tokens + cost, error log, agent payloads, LangSmith deep-link."
        ),
    )

    runs = _list_runs()
    if not runs:
        st.warning(
            "No graph runs in `data_cache/state.db` yet. Run a drill-in from "
            "the dashboard or Telegram first."
        )
        return

    # Run picker — labels show `ticker × thesis · status · started`. Newest
    # first so the most recent run is the default.
    labels: list[str] = []
    for r in runs:
        ts = (r.get("started_at") or "")[:16].replace("T", " ")
        labels.append(
            f"{r.get('ticker', '?')} × {r.get('thesis', '?')} · "
            f"{r.get('status', '?')} · {ts} · {(r.get('run_id') or '')[:8]}"
        )

    chosen_label = st.sidebar.selectbox("Run", labels, index=0)
    chosen_idx = labels.index(chosen_label)
    run = runs[chosen_idx]
    run_id = run.get("run_id") or ""

    # Pull all the run-scoped data up front so the rendering below is dumb.
    nodes = state_db.node_runs_for_run(run_id)
    errors = state_db.errors_for_run(run_id)
    total_cost = sum((n.get("cost_usd") or 0.0) for n in nodes)
    total_tokens = sum(
        (n.get("tokens_in") or 0) + (n.get("tokens_out") or 0) for n in nodes
    )

    _render_run_header(run, total_cost=total_cost, total_tokens=total_tokens)

    # LangSmith deep-link as a sidebar button — visible at the same time
    # as the run picker so the user can pivot fast.
    ls_url = _langsmith_url_for_run(run_id)
    if ls_url:
        st.sidebar.link_button(
            "🔗 Open in LangSmith", ls_url, use_container_width=True
        )
    else:
        st.sidebar.caption(
            "Set `LANGSMITH_TRACING=true` in `.env` to enable LangSmith deep-link."
        )

    # Per-node timeline first — this is where the user starts when
    # something looks wrong (which node failed, which was slow, what cost).
    st.markdown("### Node timeline")
    _render_node_table(nodes)

    section_divider()

    st.markdown("### Errors logged for this run")
    _render_errors(errors)

    section_divider()

    # Agent payloads — only render if we have a saved demo state for this run.
    state = _load_demo_state(
        ticker=run.get("ticker") or "",
        thesis_slug=run.get("thesis") or "",
        run_id=run_id,
    )
    if state is None:
        st.markdown("### Agent payloads")
        st.caption(
            "No saved demo state on disk for this run. The runner writes "
            "`data_cache/demos/{TICKER}__{slug}__{run_id[:8]}.json` AFTER "
            "the graph completes; this run may have been recorded in "
            "`state.db` but not saved (older runner code, or the run "
            "failed before `_save_run_to_demo_dir` could fire)."
        )
        return

    st.markdown("### Agent payloads (from saved state)")
    for agent in (
        "fundamentals", "filings", "news", "risk", "monte_carlo", "synthesis"
    ):
        _render_agent_payload(state, agent)

    section_divider()
    _render_cio_cycles()


# --- CIO cycles + actions (Step 11.15) -----------------------------------


def _render_cio_action_row(a: dict) -> None:
    """One CIO action row, with deep-link to the related drill if any."""
    drill_id = a.get("drill_run_id") or ""
    reuse_id = a.get("reuse_run_id") or ""
    confidence = str(a.get("confidence") or "—")
    action = str(a.get("action") or "?")
    icon = {"drill": "📈", "reuse": "♻️", "dismiss": "🪦"}.get(action, "•")
    ticker = str(a.get("ticker") or "?")
    thesis = str(a.get("thesis") or "—")
    rationale = str(a.get("rationale") or "")[:240]

    line = (
        f"{icon} **{ticker}** / `{thesis}` — {action.upper()} "
        f"_(confidence: {confidence})_  \n"
        f"  {rationale}"
    )
    if drill_id:
        line += f"  \n  ↳ drill_run_id: `{drill_id[:8]}`"
    elif reuse_id:
        line += f"  \n  ↳ reuse_run_id: `{reuse_id[:8]}`"
    st.markdown(line)


def _render_cio_cycles() -> None:
    """Render the last 10 CIO cycles with their actions inline.

    The CIO planner LLM call doesn't go through `_safe_node` (it's a
    standalone OpenRouter call inside `cio.planner.decide`), so it
    doesn't write to `node_runs`. Its telemetry footprint is the
    `cio_actions` row + the corresponding `cio_runs` rollup. This panel
    surfaces both so the user can audit the CIO's reasoning history
    next to the drill-in inspector they already use.
    """
    st.markdown("### 🤖 CIO Cycles")
    st.caption(
        "CIO heartbeat / on-demand cycles. Each cycle is one decision pass "
        "across (ticker, thesis) candidates — drills you see in the run "
        "picker above are typically *executed by* a CIO action below."
    )
    runs = state_db.recent_cio_runs(limit=10)
    if not runs:
        st.caption(
            "No CIO cycles recorded yet. Run `/cio` from Telegram or wait "
            "for the heartbeat (5am + 1pm PT)."
        )
        return

    for run in runs:
        run_id = run.get("run_id") or ""
        started = str(run.get("started_at") or "")[:19].replace("T", " ")
        trigger = str(run.get("trigger") or "")
        status = str(run.get("status") or "")
        duration = (
            f"{run['duration_s']:.1f}s" if run.get("duration_s") else "—"
        )
        n_drilled = int(run.get("n_drilled") or 0)
        n_reused = int(run.get("n_reused") or 0)
        n_dismissed = int(run.get("n_dismissed") or 0)
        title = (
            f"{trigger} · {started} · {n_drilled}d / {n_reused}r / "
            f"{n_dismissed}x · {duration} · {status}"
        )
        # Default-expand the most recent cycle so it's visible without a click.
        is_first = run_id == (runs[0].get("run_id") or "")
        with st.expander(title, expanded=is_first):
            actions = state_db.recent_cio_actions(limit=200)
            scoped = [a for a in actions if a.get("cio_run_id") == run_id]
            if not scoped:
                st.caption("No actions recorded for this cycle.")
                continue
            for a in scoped:
                _render_cio_action_row(a)
            if run.get("summary"):
                st.markdown("**Summary**")
                st.code(str(run["summary"])[:2000], language="text")


main()
