"""LangGraph drill-in graph: load_thesis → {fundamentals, filings, news} → risk → monte_carlo → synthesis.

Each worker node runs through `_safe_node`, which catches any exception, logs
it, and routes it into `state.errors` instead of crashing the whole graph.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents import filings, fundamentals, news, risk, synthesis
from utils import logger
from utils.schemas import Thesis
from utils.state import FinaqState

THESES_DIR = Path("theses")

NodeFn = Callable[[FinaqState], Awaitable[dict[str, Any]]]


def _safe_node(name: str, run_fn: NodeFn) -> NodeFn:
    """Wrap an agent's run() so a raise becomes a state.errors entry instead of a crash."""

    async def wrapped(state: FinaqState) -> dict[str, Any]:
        try:
            return await run_fn(state)
        except Exception as e:  # surface, do not propagate
            logger.error(f"{name}: {e}")
            return {
                "errors": [f"{name}: {e}"],
                "messages": [
                    {"node": name, "event": "failed", "error": str(e), "ts": time.perf_counter()}
                ],
            }

    return wrapped


async def load_thesis(state: FinaqState) -> dict[str, Any]:
    """Resolve and validate the thesis. Accepts either a dict (already parsed) or a slug string."""
    raw = state.get("thesis")
    if isinstance(raw, str):
        path = THESES_DIR / f"{raw}.json"
        raw = json.loads(path.read_text())
    Thesis.model_validate(raw)
    return {
        "thesis": raw,
        "messages": [{"node": "load_thesis", "event": "completed", "ts": time.perf_counter()}],
    }


async def monte_carlo(state: FinaqState) -> dict[str, Any]:
    """Step 4 stub. Real vectorized engine lands in Step 6."""
    return {
        "monte_carlo": {
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "samples": [],
        },
        "messages": [{"node": "monte_carlo", "event": "completed", "ts": time.perf_counter()}],
    }


def build_graph():
    """Compile the FinaqState graph with the topology from CLAUDE.md §8."""
    g: StateGraph = StateGraph(FinaqState)

    g.add_node("load_thesis", _safe_node("load_thesis", load_thesis))
    g.add_node("fundamentals", _safe_node("fundamentals", fundamentals.run))
    g.add_node("filings", _safe_node("filings", filings.run))
    g.add_node("news", _safe_node("news", news.run))
    g.add_node("risk", _safe_node("risk", risk.run))
    g.add_node("monte_carlo", _safe_node("monte_carlo", monte_carlo))
    g.add_node("synthesis", _safe_node("synthesis", synthesis.run))

    g.add_edge(START, "load_thesis")

    # Fan out: load_thesis → {fundamentals, filings, news}
    g.add_edge("load_thesis", "fundamentals")
    g.add_edge("load_thesis", "filings")
    g.add_edge("load_thesis", "news")

    # Fan in: all 3 must complete before risk runs
    g.add_edge("fundamentals", "risk")
    g.add_edge("filings", "risk")
    g.add_edge("news", "risk")

    g.add_edge("risk", "monte_carlo")
    g.add_edge("monte_carlo", "synthesis")
    g.add_edge("synthesis", END)

    return g.compile()
