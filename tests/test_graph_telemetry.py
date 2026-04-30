"""Step 5z — telemetry wiring tests.

Confirms that `_safe_node` writes a `node_runs` row per graph node, and
that `invoke_with_telemetry` writes a single `graph_runs` row per ainvoke.
The autouse `_isolated_state_db` fixture in conftest.py points
`data.state.DB_PATH` at a tmp file so we don't touch the real telemetry DB.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agents import build_graph, invoke_with_telemetry
from data import state as state_db

THESES_DIR = Path(__file__).parents[1] / "theses"


# Reuse the same stub fixture from test_graph.py — replaces real agent run()
# functions with fast deterministic stubs so we don't hit OpenRouter.
@pytest.fixture(autouse=True)
def _stub_agents(monkeypatch):
    import time as _time

    import agents as agents_pkg
    from agents import filings, fundamentals, news, risk, synthesis

    def _make_stub(node: str, payload_factory):
        async def stub_run(state):
            started = _time.perf_counter()
            return {
                **payload_factory(state),
                "messages": [
                    {
                        "node": node,
                        "event": "completed",
                        "started_at": started,
                        "completed_at": _time.perf_counter(),
                    }
                ],
            }

        return stub_run

    monkeypatch.setattr(
        fundamentals,
        "run",
        _make_stub(
            "fundamentals",
            lambda s: {
                "fundamentals": {
                    "summary": "stub",
                    "kpis": {},
                    "projections": {
                        "revenue_growth_mean": 0.20,
                        "revenue_growth_std": 0.05,
                        "margin_mean": 0.65,
                        "margin_std": 0.04,
                        "exit_multiple_mean": 28.0,
                        "exit_multiple_std": 4.0,
                    },
                    "evidence": [],
                }
            },
        ),
    )
    monkeypatch.setattr(
        filings,
        "run",
        _make_stub(
            "filings",
            lambda s: {
                "filings": {
                    "summary": "stub",
                    "risk_themes": [],
                    "mdna_quotes": [],
                    "evidence": [],
                }
            },
        ),
    )
    monkeypatch.setattr(
        news,
        "run",
        _make_stub(
            "news",
            lambda s: {
                "news": {
                    "summary": "stub",
                    "catalysts": [],
                    "concerns": [],
                    "evidence": [],
                }
            },
        ),
    )
    monkeypatch.setattr(
        risk,
        "run",
        _make_stub(
            "risk",
            lambda s: {
                "risk": {
                    "level": "MODERATE",
                    "score_0_to_10": 4,
                    "top_risks": [
                        {
                            "title": "stub",
                            "severity": 3,
                            "explanation": "x",
                            "sources": [],
                        }
                    ],
                    "convergent_signals": [],
                    "threshold_breaches": [],
                    "summary": "stub",
                }
            },
        ),
    )
    monkeypatch.setattr(
        synthesis,
        "run",
        _make_stub(
            "synthesis",
            lambda s: {
                "report": "# stub\n\n## Thesis statement\nstub.",
                "synthesis_confidence": "medium",
                "gaps": [],
                "watchlist": [],
            },
        ),
    )

    async def stub_mc(state):
        return {
            "monte_carlo": {
                "method": "stub",
                "dcf": {"p10": 50, "p25": 70, "p50": 100, "p75": 130, "p90": 160},
                "multiple": {"p10": 60, "p25": 80, "p50": 110, "p75": 140, "p90": 170},
                "convergence_ratio": 0.91,
                "current_price": 100.0,
                "discount_rate_used": 0.09,
                "terminal_growth_used": 0.025,
                "n_sims": 10000,
                "n_years": 10,
            },
            "messages": [
                {
                    "node": "monte_carlo",
                    "event": "completed",
                    "started_at": time.perf_counter(),
                    "completed_at": time.perf_counter(),
                },
            ],
        }

    monkeypatch.setattr(agents_pkg, "monte_carlo", stub_mc)


@pytest.fixture
def ai_cake() -> dict:
    return json.loads((THESES_DIR / "ai_cake.json").read_text())


# --- Telemetry assertions --------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_with_telemetry_writes_one_graph_runs_row(ai_cake):
    graph = build_graph()
    final = await invoke_with_telemetry(graph, {"ticker": "NVDA", "thesis": ai_cake})
    runs = state_db.recent_runs(limit=10)
    assert len(runs) == 1
    assert runs[0]["ticker"] == "NVDA"
    assert runs[0]["thesis"] == "AI cake"
    assert runs[0]["status"] == "completed"
    assert runs[0]["confidence"] == "medium"
    assert runs[0]["duration_s"] is not None
    assert final.get("report")  # graph still produced its output


@pytest.mark.asyncio
async def test_invoke_with_telemetry_writes_one_node_run_per_node(ai_cake):
    graph = build_graph()
    await invoke_with_telemetry(graph, {"ticker": "NVDA", "thesis": ai_cake})
    runs = state_db.recent_runs(limit=1)
    assert len(runs) == 1
    run_id = runs[0]["run_id"]
    nodes = state_db.all_node_runs_for(run_id)
    seen = {n["node"] for n in nodes}
    expected = {"load_thesis", "fundamentals", "filings", "news", "risk", "monte_carlo", "synthesis"}
    assert seen == expected, f"unexpected node set: {seen}"
    # All nodes should have status='completed' on the happy path
    assert all(n["status"] == "completed" for n in nodes), [
        (n["node"], n["status"], n["error"]) for n in nodes if n["status"] != "completed"
    ]


@pytest.mark.asyncio
async def test_failing_node_records_failed_status_and_error_row(monkeypatch, ai_cake):
    """When an agent raises, _safe_node still writes a node_runs row with
    status='failed' AND records an entry in the errors table."""
    from agents import filings as filings_mod

    async def boom(state):
        raise RuntimeError("simulated filings outage")

    monkeypatch.setattr(filings_mod, "run", boom)

    graph = build_graph()
    await invoke_with_telemetry(graph, {"ticker": "NVDA", "thesis": ai_cake})

    runs = state_db.recent_runs(limit=1)
    run_id = runs[0]["run_id"]

    # Run-level: status is 'failed' because state.errors is non-empty
    assert runs[0]["status"] == "failed"
    assert "filings" in (runs[0]["error"] or "")

    # Node-level: filings is the failed node
    node_rows = state_db.all_node_runs_for(run_id)
    failed = [n for n in node_rows if n["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["node"] == "filings"
    assert "simulated filings outage" in failed[0]["error"]

    # Errors table: one row, attributed to the run
    errs = state_db.recent_errors()
    assert any(e["agent"] == "filings" and e["run_id"] == run_id for e in errs)


@pytest.mark.asyncio
async def test_two_consecutive_runs_have_distinct_run_ids(ai_cake):
    graph = build_graph()
    await invoke_with_telemetry(graph, {"ticker": "NVDA", "thesis": ai_cake})
    await invoke_with_telemetry(graph, {"ticker": "AVGO", "thesis": ai_cake})
    runs = state_db.recent_runs(limit=10)
    assert len(runs) == 2
    assert runs[0]["run_id"] != runs[1]["run_id"]
    # Tickers are routed correctly (most-recent-first)
    assert {r["ticker"] for r in runs} == {"NVDA", "AVGO"}


@pytest.mark.asyncio
async def test_node_runs_have_increasing_started_at(ai_cake):
    """Sanity: node start times respect the dependency edges (revised
    topology — see ARCHITECTURE.md §1.1):
      load_thesis < {fundamentals, filings, news}
      max(fundamentals, filings, news) ≤ risk
      fundamentals ≤ monte_carlo  (MC parallel to Risk, both downstream of Fundamentals)
      max(risk, monte_carlo) ≤ synthesis
    Risk and MC may interleave — they run in parallel."""
    graph = build_graph()
    await invoke_with_telemetry(graph, {"ticker": "NVDA", "thesis": ai_cake})
    runs = state_db.recent_runs(limit=1)
    nodes = state_db.all_node_runs_for(runs[0]["run_id"])

    by_node = {n["node"]: n["started_at"] for n in nodes}
    assert by_node["load_thesis"] <= by_node["fundamentals"]
    assert by_node["load_thesis"] <= by_node["filings"]
    assert by_node["load_thesis"] <= by_node["news"]
    assert max(by_node["fundamentals"], by_node["filings"], by_node["news"]) <= by_node["risk"]
    assert by_node["fundamentals"] <= by_node["monte_carlo"]
    assert max(by_node["risk"], by_node["monte_carlo"]) <= by_node["synthesis"]


@pytest.mark.asyncio
async def test_plain_ainvoke_still_works_without_telemetry_setup(ai_cake):
    """Plain `graph.ainvoke(...)` (without invoke_with_telemetry) still
    works for tests / one-offs. _safe_node writes node_runs with run_id=None
    when no current_run_id contextvar is set — those rows are still useful."""
    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": ai_cake})
    assert final.get("report")
    # node_runs should still have been written (with run_id=None)
    nodes = state_db.recent_node_runs(limit=20)
    orphan = [n for n in nodes if n["run_id"] is None]
    assert len(orphan) >= 7, f"expected ≥7 orphan node rows, got {len(orphan)}"
