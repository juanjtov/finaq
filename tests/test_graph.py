"""Graph wiring tests — fast topology / routing / failure-path checks.

These tests assert on the *graph* (fan-out, fan-in, error routing, schema
contracts), not on individual agent behaviour. To keep the suite fast and
deterministic, an autouse fixture replaces real agent `run` functions with
lightweight stubs. Real agent behaviour is covered in test_{agent}.py and
test_{agent}_integration.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agents import build_graph
from utils.state import FinaqState

THESES_DIR = Path(__file__).parents[1] / "theses"
EXPECTED_NODES = (
    "load_thesis",
    "fundamentals",
    "filings",
    "news",
    "risk",
    "monte_carlo",
    "synthesis",
)


def _stub_payload(node: str, ticker: str) -> dict:
    """Standard stub return shared across the graph tests — Pydantic-valid for each schema."""
    payloads = {
        "fundamentals": {
            "fundamentals": {
                "summary": f"[stub] fundamentals for {ticker}",
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
        "filings": {
            "filings": {
                "summary": f"[stub] filings for {ticker}",
                "risk_themes": ["macro", "supply chain"],
                "mdna_quotes": [],
                "evidence": [],
            }
        },
        "news": {
            "news": {
                "summary": f"[stub] news for {ticker}",
                "catalysts": [],
                "concerns": [],
                "evidence": [],
            }
        },
        "risk": {
            "risk": {
                "level": "MODERATE",
                "score_0_to_10": 4,
                "top_risks": [
                    {
                        "title": "[stub]",
                        "severity": 3,
                        "explanation": "stub",
                        "sources": [],
                    }
                ],
                "convergent_signals": [],
                "threshold_breaches": [],
                "summary": f"[stub] risk for {ticker}",
            }
        },
    }
    return payloads.get(node, {})


@pytest.fixture(autouse=True)
def _stub_real_agents(monkeypatch):
    """Replace real agent `run` callables with fast deterministic stubs so graph tests
    don't hit OpenRouter / yfinance. Tests that explicitly want a different stub
    (e.g. a raising one) override via their own monkeypatch *after* this fixture runs."""

    def _make_stub(node: str):
        async def stub_run(state):
            started = time.perf_counter()
            ticker = state.get("ticker", "?")
            return {
                **_stub_payload(node, ticker),
                "messages": [
                    {
                        "node": node,
                        "event": "completed",
                        "started_at": started,
                        "completed_at": time.perf_counter(),
                    }
                ],
            }

        return stub_run

    from agents import filings, fundamentals, news, risk

    monkeypatch.setattr(fundamentals, "run", _make_stub("fundamentals"))
    monkeypatch.setattr(filings, "run", _make_stub("filings"))
    monkeypatch.setattr(news, "run", _make_stub("news"))
    monkeypatch.setattr(risk, "run", _make_stub("risk"))


@pytest.fixture(scope="module")
def ai_cake() -> dict:
    return json.loads((THESES_DIR / "ai_cake.json").read_text())


# --- Topology ----------------------------------------------------------------


def test_graph_compiles_without_errors():
    graph = build_graph()
    assert graph is not None


def test_graph_topology_includes_all_nodes_and_edges():
    graph = build_graph().get_graph()
    nodes = set(graph.nodes.keys())
    for n in EXPECTED_NODES:
        assert n in nodes, f"missing node: {n}"

    mermaid = graph.draw_mermaid()
    assert "load_thesis" in mermaid
    assert "fundamentals" in mermaid
    assert "synthesis" in mermaid


def test_langgraph_json_points_to_compiled_graph():
    """langgraph.json's drill_in entrypoint must resolve to a callable that builds a compiled graph."""
    cfg = json.loads((THESES_DIR.parent / "langgraph.json").read_text())
    assert "graphs" in cfg
    assert "drill_in" in cfg["graphs"]
    assert cfg["graphs"]["drill_in"].endswith(":build_graph")
    # Sanity-check the entrypoint actually compiles to a graph.
    from agents import build_graph as factory

    assert callable(factory)
    assert factory().get_graph() is not None


# --- End-to-end stub run -----------------------------------------------------


@pytest.mark.asyncio
async def test_graph_runs_end_to_end_with_stubs(ai_cake):
    graph = build_graph()
    final: FinaqState = await graph.ainvoke({"ticker": "NVDA", "thesis": ai_cake})

    for key in (
        "ticker",
        "thesis",
        "fundamentals",
        "filings",
        "news",
        "risk",
        "monte_carlo",
        "report",
        "messages",
    ):
        assert key in final, f"final state missing key: {key}"

    assert final["ticker"] == "NVDA"
    assert final["thesis"]["name"] == "AI cake"
    assert final["fundamentals"]["projections"]["revenue_growth_mean"] == 0.20
    assert "[stub]" in final["report"]
    assert "AI cake" in final["report"]


@pytest.mark.asyncio
async def test_graph_loads_thesis_from_slug(ai_cake):
    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": "ai_cake"})
    assert final["thesis"]["name"] == ai_cake["name"]


# --- Fan-out / fan-in behaviour ----------------------------------------------


@pytest.mark.asyncio
async def test_three_workers_run_in_parallel(ai_cake):
    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": ai_cake})

    starts = {
        m["node"]: m["started_at"]
        for m in final["messages"]
        if m["node"] in {"fundamentals", "filings", "news"} and "started_at" in m
    }
    assert set(starts.keys()) == {"fundamentals", "filings", "news"}
    spread = max(starts.values()) - min(starts.values())
    assert spread < 0.1, f"workers did not start in parallel: spread={spread:.3f}s"


@pytest.mark.asyncio
async def test_risk_starts_only_after_all_three_workers_complete(ai_cake):
    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": ai_cake})
    msgs_by_node = {
        m["node"]: m
        for m in final["messages"]
        if m["node"] in {"fundamentals", "filings", "news", "risk"}
    }
    worker_completions = max(
        msgs_by_node[n]["completed_at"] for n in ("fundamentals", "filings", "news")
    )
    risk_start = msgs_by_node["risk"]["started_at"]
    assert risk_start >= worker_completions, "risk started before all 3 workers completed"


@pytest.mark.asyncio
async def test_astream_emits_events_in_dependency_order(ai_cake):
    graph = build_graph()
    seen: list[str] = []
    async for event in graph.astream({"ticker": "NVDA", "thesis": ai_cake}):
        seen.extend(event.keys())

    # load_thesis must precede each worker; risk must come after all workers; etc.
    assert "load_thesis" in seen
    for worker in ("fundamentals", "filings", "news"):
        assert seen.index("load_thesis") < seen.index(worker)
    for worker in ("fundamentals", "filings", "news"):
        assert seen.index(worker) < seen.index("risk")
    assert seen.index("risk") < seen.index("monte_carlo")
    assert seen.index("monte_carlo") < seen.index("synthesis")


# --- Failure-path / safe-node wrapper ----------------------------------------


@pytest.mark.asyncio
async def test_failing_worker_does_not_crash_graph(monkeypatch, ai_cake):
    """If one worker raises, the graph still completes and routes the error into state."""
    from agents import filings as filings_mod

    async def boom(state):
        raise RuntimeError("simulated filings outage")

    monkeypatch.setattr(filings_mod, "run", boom)

    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": ai_cake})

    assert "errors" in final
    assert any("filings" in e and "simulated" in e for e in final["errors"])
    # Other state keys still populated despite the failure.
    assert final["fundamentals"]
    assert final["news"]
    assert final["risk"]
    assert final["report"]


# --- Schema-contract guardrails ---------------------------------------------


@pytest.mark.asyncio
async def test_each_agent_output_validates_against_pydantic_schema(ai_cake):
    """Every agent's stub must already conform to its §9 Pydantic shape — locks the
    contract before Step 5 replaces stubs with real LLM calls."""
    from utils.schemas import (
        FilingsOutput,
        FundamentalsOutput,
        NewsOutput,
        RiskOutput,
        SynthesisOutput,
        Thesis,
    )

    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": ai_cake})

    Thesis.model_validate(final["thesis"])
    FundamentalsOutput.model_validate(final["fundamentals"])
    FilingsOutput.model_validate(final["filings"])
    NewsOutput.model_validate(final["news"])
    RiskOutput.model_validate(final["risk"])
    SynthesisOutput.model_validate({"report": final["report"]})


@pytest.mark.asyncio
async def test_messages_list_contains_one_entry_per_active_node(ai_cake):
    """Every node along the path must append a message — proves all nodes ran and
    the operator.add reducer is wired correctly."""
    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": ai_cake})

    nodes_seen = {m["node"] for m in final["messages"]}
    assert nodes_seen == {
        "load_thesis",
        "fundamentals",
        "filings",
        "news",
        "risk",
        "monte_carlo",
        "synthesis",
    }, f"unexpected nodes_seen: {nodes_seen}"


@pytest.mark.asyncio
async def test_clean_run_has_empty_errors_list(ai_cake):
    """A run with no failures should leave errors empty."""
    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": ai_cake})
    assert final.get("errors", []) == []


@pytest.mark.asyncio
async def test_triage_stub_returns_empty_dict(ai_cake):
    """Phase 0 triage is a no-op stub; real impl lands in Step 11."""
    from agents import triage

    result = await triage.run({"ticker": "NVDA", "thesis": ai_cake})
    assert result == {}


@pytest.mark.asyncio
async def test_load_thesis_rejects_invalid_thesis_dict():
    """load_thesis must validate via Pydantic — caught at the safe-node wrapper."""
    graph = build_graph()
    bad = {"name": "bad", "summary": "missing required fields"}  # no anchor_tickers, no universe
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": bad})
    # The wrapper catches the ValidationError and routes to state.errors.
    assert any("load_thesis" in e for e in final.get("errors", []))
