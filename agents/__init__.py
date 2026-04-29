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
    """Step 6 — Hybrid DCF + Multiple Monte Carlo. Synthesis-only (no LLM, no
    external calls). Reads Fundamentals' projections + KPIs and the thesis
    valuation block; emits two fair-value distributions plus a convergence ratio.
    """
    from data.treasury import get_10y_treasury_yield
    from utils.monte_carlo import compute_discount_rate, simulate
    from utils.schemas import Projections, Thesis

    fundamentals = state.get("fundamentals") or {}
    thesis_dict = state.get("thesis") or {}
    kpis = fundamentals.get("kpis") or {}
    projections_dict = fundamentals.get("projections") or {}

    # Validate required upstream inputs are present
    missing: list[str] = []
    if not projections_dict:
        missing.append("fundamentals.projections")
    for required_kpi in ("revenue_latest", "shares_outstanding", "current_price"):
        if not kpis.get(required_kpi):
            missing.append(f"fundamentals.kpis.{required_kpi}")
    thesis = Thesis.model_validate(thesis_dict) if thesis_dict else None
    if thesis is None or thesis.valuation is None:
        missing.append("thesis.valuation")

    if missing:
        logger.warning(f"[monte_carlo] missing inputs: {missing}; emitting empty MC")
        return {
            "monte_carlo": {
                "method": "skipped",
                "errors": [f"missing inputs: {missing}"],
            },
            "messages": [{"node": "monte_carlo", "event": "skipped", "ts": time.perf_counter()}],
        }

    projections = Projections.model_validate(projections_dict)
    treasury = get_10y_treasury_yield()
    discount_rate = compute_discount_rate(treasury, thesis.valuation)

    mc = simulate(
        projections=projections,
        valuation=thesis.valuation,
        revenue_now=float(kpis["revenue_latest"]),
        shares_now=float(kpis["shares_outstanding"]),
        current_price=float(kpis["current_price"]),
        net_cash=float(kpis.get("net_cash", 0.0)),
        discount_rate=discount_rate,
    )

    return {
        "monte_carlo": mc,
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
