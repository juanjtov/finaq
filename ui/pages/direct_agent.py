"""Direct Agent page — invoke a single agent in isolation.

Two modes per agent:

  - **Run** (no question): re-runs the agent's full `run()` step on a fresh
    state. Useful for "I just re-ingested filings, re-run Filings on NVDA
    without paying for the full graph."
  - **Ask** (free-text question): runs `agents.qa.ask()` over the agent's
    most recent cached drill-in (or runs Filings RAG fresh if no cached
    state exists). Cheaper than `run()` because it uses MODEL_AGENT_QA
    (Haiku-tier).

Agents supported: fundamentals, filings, news, risk. Synthesis and Monte
Carlo are excluded — they're integrators / pure compute, no useful direct
invocation surface.

Cost guard: a daily $-ceiling check (TODO: lands with Step 5z `state.db`).
For now, the page just warns when more than 5 calls have happened in the
current Streamlit session.
"""

from __future__ import annotations

# Bootstrap (see ui/app.py for explanation).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import json
import time

import streamlit as st

from agents.qa import AGENT_NAMES, ask
from ui.components import evidence_list, page_header, section_divider

st.set_page_config(page_title="FINAQ — Direct Agent", page_icon="🎯", layout="wide")

THESES_DIR = Path(__file__).parents[2] / "theses"
DEMO_DIR = Path(__file__).parents[2] / "data_cache" / "demos"


# Per-agent helpful prompts shown to the user as starting points.
SUGGESTED_QUESTIONS: dict[str, list[str]] = {
    "fundamentals": [
        "What is the current FCF yield and is it concerning vs the thesis threshold?",
        "How does revenue growth compare to the 5-year CAGR?",
        "What's the projected exit multiple and how was it derived?",
    ],
    "filings": [
        "Does the latest 10-Q mention supply constraints or capacity issues?",
        "What does management say about export-control risk?",
        "Are there new segment-level disclosures vs the prior 10-K?",
    ],
    "news": [
        "Are there any recent hyperscaler capex announcements?",
        "What are the bear-case concerns from the last 30 days?",
        "Has there been any management commentary on Q3?",
    ],
    "risk": [
        "What is the most severe risk and which sources surfaced it?",
        "Are any thesis material thresholds currently breached?",
        "Are there divergent signals between News and Filings?",
    ],
    "synthesis": [
        "Why is the confidence label what it is?",
        "Expand on the most important bear-case bullet.",
        "What would change the action recommendation?",
        "Reconcile the tension between bull and bear cases.",
    ],
}


@st.cache_data(show_spinner=False)
def _list_thesis_slugs() -> list[str]:
    return sorted(p.stem for p in THESES_DIR.glob("*.json"))


def _try_load_demo(ticker: str, thesis_slug: str) -> dict | None:
    path = DEMO_DIR / f"{ticker.upper()}__{thesis_slug}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data(show_spinner=False)
def _load_thesis(slug: str) -> dict:
    return json.loads((THESES_DIR / f"{slug}.json").read_text())


def _build_state_for(ticker: str, thesis_slug: str) -> dict:
    """Resolve a state dict suitable for `ask()`. Prefers a cached demo
    (full state with upstream payloads) so Q&A has context to answer over."""
    cached = _try_load_demo(ticker, thesis_slug)
    if cached:
        return cached
    # No cached demo — return a minimal state. Filings ask() can still RAG
    # without it; the other agents will say "no data".
    return {"ticker": ticker.upper(), "thesis": _load_thesis(thesis_slug)}


def _track_call(agent: str) -> int:
    """Lightweight per-session call counter — surfaces a soft cost warning
    until Step 5z lands real telemetry."""
    if "direct_agent_calls" not in st.session_state:
        st.session_state["direct_agent_calls"] = []
    st.session_state["direct_agent_calls"].append(
        {"agent": agent, "ts": time.time()}
    )
    return len(st.session_state["direct_agent_calls"])


def _render_run_form(thesis_slug: str, ticker: str, agent: str) -> None:
    """Re-run the agent's full `run()` (no Q&A). Useful when the user wants
    a fresh structured output without paying for the full graph."""
    if st.button(f"Re-run {agent}", type="primary"):
        if not ticker:
            st.warning("Enter a ticker first.")
            return
        thesis = _load_thesis(thesis_slug)
        state = _build_state_for(ticker, thesis_slug)
        state.update({"thesis": thesis})

        # Import the right agent module dynamically
        agent_module_map = {
            "fundamentals": "agents.fundamentals",
            "filings": "agents.filings",
            "news": "agents.news",
            "risk": "agents.risk",
        }
        module_name = agent_module_map[agent]
        with st.status(f"Running {agent} on {ticker}", expanded=True) as status:
            t0 = time.perf_counter()
            try:
                module = __import__(module_name, fromlist=["run"])
                result = asyncio.run(module.run(state))
                elapsed = time.perf_counter() - t0
                status.update(
                    label=f"{agent} complete ({elapsed:.1f}s)",
                    state="complete",
                )
            except Exception as e:
                status.update(label=f"{agent} failed", state="error")
                st.error(str(e))
                return

        _track_call(agent)
        payload = result.get(agent) or {}
        st.markdown(f"### {agent.capitalize()} output")
        st.json(payload)


def _render_ask_form(thesis_slug: str, ticker: str, agent: str) -> None:
    """Free-text Q&A on the agent. Uses MODEL_AGENT_QA (cheap-tier)."""
    suggestions = SUGGESTED_QUESTIONS.get(agent, [])
    if suggestions:
        st.caption("Example questions:")
        for s in suggestions:
            if st.button(s, key=f"{agent}_suggest_{hash(s)}"):
                st.session_state[f"{agent}_question"] = s

    question = st.text_area(
        "Question",
        value=st.session_state.get(f"{agent}_question", ""),
        key=f"{agent}_question",
        height=80,
        placeholder=f"What would you like to ask {agent}?",
    )

    if st.button(f"Ask {agent}", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
            return
        if not ticker:
            st.warning("Enter a ticker first.")
            return

        state = _build_state_for(ticker, thesis_slug)
        with st.status(f"Asking {agent} on {ticker}", expanded=True) as status:
            t0 = time.perf_counter()
            try:
                answer = asyncio.run(ask(state, agent, question))
                elapsed = time.perf_counter() - t0
                status.update(
                    label=f"{agent} responded ({elapsed:.1f}s)",
                    state="complete",
                )
            except Exception as e:
                status.update(label=f"{agent} failed", state="error")
                st.error(str(e))
                return

        n_calls = _track_call(agent)
        if n_calls > 5:
            st.warning(
                f"You've made {n_calls} direct-agent calls this session. "
                "Step 5z will add a per-day $ ceiling."
            )

        st.markdown(f"### {agent.capitalize()} says")
        st.markdown(answer.answer)
        if answer.citations:
            st.markdown("#### Citations")
            evidence_list([c.model_dump() for c in answer.citations])
        if answer.errors:
            st.warning(f"Errors during call: {answer.errors}")


def main() -> None:
    page_header(
        "Direct Agent",
        subtitle=(
            "Talk to one agent at a time. "
            "Re-run a single step on a fresh ticker, or ask a free-text question "
            "scoped to a single agent's structured output."
        ),
    )

    slugs = _list_thesis_slugs()
    if not slugs:
        st.error("No theses found in /theses/.")
        return

    cols = st.columns([1, 1, 2])
    with cols[0]:
        thesis_slug = st.selectbox("Thesis", slugs)
    with cols[1]:
        # Ticker is a dropdown limited to the active thesis's universe — no
        # more arbitrary text entry that defaults to NVDA on every thesis.
        thesis = _load_thesis(thesis_slug)
        anchors = thesis.get("anchor_tickers") or []
        universe = thesis.get("universe") or anchors
        if not universe:
            st.warning(f"Thesis `{thesis_slug}` has an empty universe.")
            return
        # Render anchors with a ⭐ prefix so the user can spot them at a glance.
        formatted = [f"⭐ {t}" if t in anchors else t for t in universe]

        def _strip_chip(label: str) -> str:
            return label.lstrip("⭐ ").strip()

        choice = st.selectbox(
            "Ticker",
            formatted,
            help="Tickers limited to the active thesis's universe. ⭐ marks anchors.",
        )
        ticker = _strip_chip(choice).upper()
    with cols[2]:
        agent = st.selectbox(
            "Agent",
            list(AGENT_NAMES),
            help=(
                "fundamentals: KPIs + projections · "
                "filings: SEC RAG · "
                "news: 90d Tavily · "
                "risk: cross-modal synthesis · "
                "synthesis: explain the report"
            ),
        )

    section_divider()

    mode = st.radio(
        "Mode",
        ["Ask (free-text Q&A)", "Re-run (full agent step)"],
        horizontal=True,
    )

    if mode.startswith("Ask"):
        _render_ask_form(thesis_slug, ticker, agent)
    else:
        _render_run_form(thesis_slug, ticker, agent)


main()
