"""LangGraph drill-in graph: load_thesis → {fundamentals, filings, news} → risk → monte_carlo → synthesis.

Each worker node runs through `_safe_node`, which catches any exception, logs
it, and routes it into `state.errors` instead of crashing the whole graph.
Step 5z added per-node telemetry: `_safe_node` writes a `node_runs` row on
every entry/exit (read by the Mission Control dashboard).

Use `invoke_with_telemetry(graph, state)` (not `graph.ainvoke(state)`
directly) when you want the run captured in `data_cache/state.db`. The
plain `ainvoke` path still works for tests / one-offs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents import filings, fundamentals, news, risk, synthesis
from data import state as state_db
from utils import logger
from utils.schemas import Thesis
from utils.state import FinaqState

THESES_DIR = Path("theses")

NodeFn = Callable[[FinaqState], Awaitable[dict[str, Any]]]


def _safe_node(name: str, run_fn: NodeFn) -> NodeFn:
    """Wrap an agent's run() so a raise becomes a state.errors entry instead of
    a crash, and so every entry/exit writes a `node_runs` telemetry row.

    The telemetry write is best-effort: if state.db is unwritable (e.g. disk
    full), we log a warning and continue. The graph itself NEVER fails because
    telemetry failed — observability is supposed to be invisible to the
    happy path.
    """

    async def wrapped(state: FinaqState) -> dict[str, Any]:
        run_id = state_db.current_run_id.get()
        started_iso = datetime.now(UTC).isoformat()
        t0 = time.perf_counter()
        error_msg: str | None = None
        try:
            result = await run_fn(state)
            # Surface "soft failures" — agents that return successfully but
            # carry non-empty `errors` lists in their payload. The canonical
            # case: Filings returns {filings: {errors: ["no chunks retrieved"]}}
            # when ChromaDB is empty for the ticker — completes without
            # raising, so was previously invisible in Mission Control.
            try:
                for value in (result or {}).values():
                    if not isinstance(value, dict):
                        continue
                    for err in value.get("errors") or []:
                        if err:
                            state_db.record_error(name, str(err), run_id=run_id)
            except Exception as scan_exc:
                logger.warning(
                    f"[telemetry] payload-error scan failed for {name}: {scan_exc}"
                )
            return result
        except Exception as e:  # raised exception — escalates to node-run 'failed'
            error_msg = str(e)
            logger.error(f"{name}: {e}")
            return {
                "errors": [f"{name}: {e}"],
                "messages": [
                    {"node": name, "event": "failed", "error": str(e), "ts": time.perf_counter()}
                ],
            }
        finally:
            ended_iso = datetime.now(UTC).isoformat()
            duration_s = time.perf_counter() - t0
            try:
                state_db.record_node_run(
                    run_id=run_id,
                    node=name,
                    started_at=started_iso,
                    ended_at=ended_iso,
                    duration_s=duration_s,
                    status="failed" if error_msg else "completed",
                    error=error_msg,
                )
                if error_msg:
                    state_db.record_error(name, error_msg, run_id=run_id)
            except Exception as telemetry_exc:
                logger.warning(
                    f"[telemetry] node_runs write failed for {name}: {telemetry_exc}"
                )

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
    """Compile the FinaqState graph.

    Topology (revised late Step 8 — see ARCHITECTURE.md §1.1):

        START → load_thesis ──┬──→ fundamentals ──┬──→ risk ────→ synthesis → END
                              │                   │                ↑
                              ├──→ filings ───────┤                │
                              │                   │                │
                              └──→ news ──────────┘                │
                                                                   │
                                  fundamentals ────→ monte_carlo ──┘

    Why this shape (not a single linear chain after Risk):
      - Monte Carlo only reads `fundamentals.projections + kpis` and
        `thesis.valuation`. It does NOT need Risk's output. Forcing MC to
        wait for Risk made the diagram falsely imply MC reads from Risk.
      - Synthesis reads from EVERY upstream output (fundamentals, filings,
        news, risk, monte_carlo). The previous shape only showed the MC
        edge into Synthesis, hiding the fact that Risk feeds it too.
      - Net effect: MC now runs **in parallel with Risk** (both fan out
        from Fundamentals), shaving ~10-15s off a typical drill-in. The
        diagram correctly shows MC's input as Fundamentals, not Risk, and
        Synthesis as a 2-way fan-in from Risk + MC.

    State semantics: LangGraph state is shared, so any node that runs
    after another can still read its output even without an explicit
    edge. Filings/News reach Synthesis via Risk → Synthesis (transitive);
    Fundamentals reaches Synthesis via both Risk and MC.
    """
    g: StateGraph = StateGraph(FinaqState)

    g.add_node("load_thesis", _safe_node("load_thesis", load_thesis))
    g.add_node("fundamentals", _safe_node("fundamentals", fundamentals.run))
    g.add_node("filings", _safe_node("filings", filings.run))
    g.add_node("news", _safe_node("news", news.run))
    g.add_node("risk", _safe_node("risk", risk.run))
    g.add_node("monte_carlo", _safe_node("monte_carlo", monte_carlo))
    g.add_node("synthesis", _safe_node("synthesis", synthesis.run))

    g.add_edge(START, "load_thesis")

    # Fan out: load_thesis → {fundamentals, filings, news} (3 parallel workers)
    g.add_edge("load_thesis", "fundamentals")
    g.add_edge("load_thesis", "filings")
    g.add_edge("load_thesis", "news")

    # Fan in to Risk: all 3 worker outputs must be present before Risk runs.
    g.add_edge("fundamentals", "risk")
    g.add_edge("filings", "risk")
    g.add_edge("news", "risk")

    # Monte Carlo runs in PARALLEL with Risk, both fan out from Fundamentals.
    # MC's data dependency is fundamentals.projections — not Risk's output.
    g.add_edge("fundamentals", "monte_carlo")

    # Fan in to Synthesis: needs both the threat-side view (Risk) AND the
    # fair-value distribution (MC). Filings/News reach it transitively
    # through Risk; Fundamentals through both.
    g.add_edge("risk", "synthesis")
    g.add_edge("monte_carlo", "synthesis")

    g.add_edge("synthesis", END)

    return g.compile()


# --- Telemetry-wrapped invocation -------------------------------------------


async def invoke_with_telemetry(graph: Any, initial_state: dict[str, Any]) -> dict[str, Any]:
    """Run the graph end-to-end with full state.db telemetry.

    Sets the `current_run_id` contextvar so that `_safe_node` writes can be
    attached to a parent `graph_runs` row. Always closes the row on exit
    (status = completed | failed) so we never leave dangling 'running' rows.

    Use this from the Streamlit app + the Telegram bot. Plain
    `graph.ainvoke(state)` still works for tests where telemetry is noise.
    """
    ticker = (initial_state.get("ticker") or "").upper() or "?"
    thesis = initial_state.get("thesis") or {}
    thesis_name = (
        thesis.get("name") if isinstance(thesis, dict) else str(thesis) if thesis else "?"
    )
    run_id = state_db.start_graph_run(ticker, thesis_name)
    token = state_db.current_run_id.set(run_id)
    t0 = time.perf_counter()

    try:
        final = await graph.ainvoke(initial_state)
    except Exception as e:
        # Even if the graph itself raises (rare — _safe_node catches per-node
        # errors), close the run row as failed so Mission Control reflects it.
        state_db.finish_graph_run(
            run_id, "failed", error=str(e), duration_s=time.perf_counter() - t0
        )
        raise
    finally:
        state_db.current_run_id.reset(token)

    # Determine the run's status from the accumulated state.errors. If any
    # node failed, the run is "failed" — even though the graph completed (the
    # _safe_node wrapper turned the exception into a state.errors entry).
    state_errors = final.get("errors") or []
    status = "failed" if state_errors else "completed"
    confidence = final.get("synthesis_confidence")
    state_db.finish_graph_run(
        run_id,
        status,
        error="; ".join(state_errors)[:1000] if state_errors else None,
        confidence=confidence,
        duration_s=time.perf_counter() - t0,
    )

    # Stamp the run_id onto the final state so downstream consumers (the
    # Streamlit dashboard's run-history cache) can key files by it without
    # having to re-query state.db.
    final["run_id"] = run_id
    return final
