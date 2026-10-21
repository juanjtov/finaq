"""Step 5z — telemetry wiring tests.

Confirms that `_safe_node` writes a `node_runs` row per graph node, and
that `invoke_with_telemetry` writes a single `graph_runs` row per ainvoke.
The autouse `_isolated_state_db` fixture in conftest.py points
`data.state.DB_PATH` at a tmp file so we don't touch the real telemetry DB.
"""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

from agents import build_graph, invoke_with_telemetry
from data import state as state_db


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
    """Minimal valid thesis, built inline. Was `theses/ai_cake.json`, but
    thesis JSONs are user-managed (renamed/archived over time) — telemetry
    tests must not depend on which theses currently exist on disk."""
    return {
        "name": "AI cake",
        "slug": "ai_cake",
        "summary": "Stub thesis for telemetry wiring tests.",
        "anchor_tickers": ["NVDA"],
        "universe": ["NVDA", "MSFT"],
    }


# --- Telemetry assertions --------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_with_telemetry_writes_one_graph_runs_row(ai_cake):
    graph = build_graph()
    final = await invoke_with_telemetry(graph, {"ticker": "NVDA", "thesis": ai_cake})
    runs = state_db.recent_runs(limit=10)
    assert len(runs) == 1
    assert runs[0]["ticker"] == "NVDA"
    # invoke_with_telemetry stores the SLUG (canonical join key for the
    # CIO cooldown gate), falling back to name only when slug is absent.
    assert runs[0]["thesis"] == "ai_cake"
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


@pytest.mark.asyncio
async def test_soft_failure_in_agent_payload_is_recorded_in_errors_table(
    monkeypatch, ai_cake
):
    """Real failure mode (CRDO case): Filings returns SUCCESSFULLY but with
    `errors=['no chunks retrieved']` in its payload because the filings index has no
    ingest for that ticker. Previously this completed silently (status='completed',
    no error row). The scan in _safe_node should now surface it."""
    from agents import filings as filings_mod

    async def filings_with_soft_error(state):
        return {
            "filings": {
                "summary": f"No filings data available for {state.get('ticker')}.",
                "risk_themes": [],
                "mdna_quotes": [],
                "evidence": [],
                "errors": ["no chunks retrieved"],
            },
            "messages": [{"node": "filings", "event": "completed"}],
        }

    monkeypatch.setattr(filings_mod, "run", filings_with_soft_error)

    graph = build_graph()
    await invoke_with_telemetry(graph, {"ticker": "ZZZZ", "thesis": ai_cake})

    # The filings node should still be 'completed' (no exception raised)…
    runs = state_db.recent_runs(limit=1)
    run_id = runs[0]["run_id"]
    nodes = state_db.all_node_runs_for(run_id)
    filings_node = next(n for n in nodes if n["node"] == "filings")
    assert filings_node["status"] == "completed"

    # … but the errors table should contain the soft-failure message.
    errs = state_db.recent_errors()
    soft_errs = [e for e in errs if e["agent"] == "filings"]
    assert any("no chunks retrieved" in (e["message"] or "") for e in soft_errs), (
        f"soft failure not propagated. Recorded errors for filings: "
        f"{[e['message'] for e in soft_errs]}"
    )


# --- Schema v9 — two-phase node rows + source ---------------------------------


def _payload(node: str) -> dict:
    """Minimal valid payload for one of the three parallel workers."""
    if node == "fundamentals":
        return {
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
        }
    if node == "filings":
        return {
            "filings": {"summary": "stub", "risk_themes": [], "mdna_quotes": [], "evidence": []}
        }
    return {"news": {"summary": "stub", "catalysts": [], "concerns": [], "evidence": []}}


@pytest.mark.asyncio
async def test_node_row_is_running_while_inside_the_node_and_completed_after(monkeypatch, ai_cake):
    """The row written at entry is what a live view sees mid-node: status
    `running`, no end time, this process's pid. The same row is closed at exit."""
    from agents import news as news_mod

    seen: dict = {}

    async def spying_news(state):
        run_id = state_db.current_run_id.get()
        seen["rows"] = {n["node"]: n for n in state_db.all_node_runs_for(run_id)}
        return _payload("news")

    monkeypatch.setattr(news_mod, "run", spying_news)
    await invoke_with_telemetry(build_graph(), {"ticker": "NVDA", "thesis": ai_cake})

    mid = seen["rows"]["news"]
    assert (mid["status"], mid["ended_at"], mid["pid"]) == ("running", None, os.getpid())
    assert seen["rows"]["load_thesis"]["status"] == "completed"
    run_id = state_db.recent_runs(limit=1)[0]["run_id"]
    rows = state_db.all_node_runs_for(run_id)
    assert len(rows) == 7
    assert all(r["status"] == "completed" and r["ended_at"] for r in rows)
    news = next(r for r in rows if r["node"] == "news")
    assert news["id"] == mid["id"] and news["duration_s"] is not None


@pytest.mark.asyncio
async def test_live_snapshot_shows_the_running_node_mid_run(monkeypatch, ai_cake):
    from agents import risk as risk_mod

    seen: dict = {}

    async def spying_risk(state):
        seen["snapshot"] = state_db.live_snapshot()
        return {
            "risk": {
                "level": "MODERATE",
                "score_0_to_10": 4,
                "top_risks": [],
                "convergent_signals": [],
                "threshold_breaches": [],
                "summary": "stub",
            }
        }

    monkeypatch.setattr(risk_mod, "run", spying_risk)
    await invoke_with_telemetry(
        build_graph(), {"ticker": "NVDA", "thesis": ai_cake}, source="dashboard"
    )

    snap = seen["snapshot"]
    assert snap["vitals"]["active"] == 1
    [run] = snap["runs"]
    assert run["source"] == "dashboard" and run["status"] == "running"
    statuses = {n["node"]: n["status"] for n in run["nodes"]}
    assert statuses["risk"] == "running"
    assert statuses["load_thesis"] == "completed"
    assert "synthesis" not in statuses  # not entered yet
    assert state_db.live_snapshot()["vitals"]["active"] == 0


@pytest.mark.asyncio
async def test_parallel_workers_each_see_their_own_running_row(monkeypatch, ai_cake):
    """fundamentals / filings / news fan out together; each must find its own
    entry row in `running` while it works, and its siblings' rows already
    opened. A barrier holds every worker inside its node until all three have
    entered, so the check does not depend on timing."""
    from agents import filings as filings_mod
    from agents import fundamentals as fundamentals_mod
    from agents import news as news_mod

    seen: dict[str, dict[str, str]] = {}
    all_entered = asyncio.Barrier(3)

    def _spy(node: str):
        async def run(state):
            await all_entered.wait()
            run_id = state_db.current_run_id.get()
            seen[node] = {n["node"]: n["status"] for n in state_db.all_node_runs_for(run_id)}
            return _payload(node)

        return run

    monkeypatch.setattr(fundamentals_mod, "run", _spy("fundamentals"))
    monkeypatch.setattr(filings_mod, "run", _spy("filings"))
    monkeypatch.setattr(news_mod, "run", _spy("news"))
    await invoke_with_telemetry(build_graph(), {"ticker": "NVDA", "thesis": ai_cake})

    workers = ("fundamentals", "filings", "news")
    for node in workers:
        assert seen[node][node] == "running", f"{node} did not see its own running row"
        assert set(workers) <= set(seen[node]), f"{node} saw only {sorted(seen[node])}"
    run_id = state_db.recent_runs(limit=1)[0]["run_id"]
    rows = [n for n in state_db.all_node_runs_for(run_id) if n["node"] in workers]
    assert len(rows) == 3 and len({n["id"] for n in rows}) == 3
    assert all(n["status"] == "completed" for n in rows)


@pytest.mark.asyncio
async def test_source_is_stamped_on_graph_runs(ai_cake):
    graph = build_graph()
    await invoke_with_telemetry(graph, {"ticker": "NVDA", "thesis": ai_cake}, source="telegram")
    await invoke_with_telemetry(graph, {"ticker": "AVGO", "thesis": ai_cake})
    by_ticker = {r["ticker"]: r["source"] for r in state_db.recent_runs(limit=5)}
    assert by_ticker == {"NVDA": "telegram", "AVGO": "cli"}


@pytest.mark.asyncio
async def test_entry_write_failure_falls_back_to_one_shot_node_rows(monkeypatch, ai_cake):
    """If the entry write fails, the exit still lands the node row via the
    one-shot path — and the graph never notices."""

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(state_db, "start_node_run", _boom)
    final = await invoke_with_telemetry(build_graph(), {"ticker": "NVDA", "thesis": ai_cake})
    assert final.get("report")
    run_id = state_db.recent_runs(limit=1)[0]["run_id"]
    rows = state_db.all_node_runs_for(run_id)
    assert len(rows) == 7 and all(r["status"] == "completed" and r["pid"] is None for r in rows)


@pytest.mark.asyncio
async def test_llm_calls_made_inside_a_node_are_attributed_with_full_messages(monkeypatch, ai_cake):
    """Graph + interceptor together: a call made from inside a node lands with
    the run id, the node name and its full messages."""
    from agents import news as news_mod
    from utils.openrouter import _install_telemetry_interceptor

    def _create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="headline"))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2),
        )

    client = _install_telemetry_interceptor(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    )

    async def news_with_llm(state):
        client.chat.completions.create(
            model="stub-model",
            messages=[
                {"role": "system", "content": "news persona"},
                {"role": "user", "content": "NVDA"},
            ],
        )
        return _payload("news")

    monkeypatch.setattr(news_mod, "run", news_with_llm)
    await invoke_with_telemetry(build_graph(), {"ticker": "NVDA", "thesis": ai_cake})

    run_id = state_db.recent_runs(limit=1)[0]["run_id"]
    [call] = state_db.llm_calls_for_run(run_id)
    assert (call["node"], call["status"], call["tokens_in"]) == ("news", "completed", 7)
    assert [(m["role"], m["text"]) for m in state_db.llm_messages_for_call(call["id"])] == [
        ("system", "news persona"),
        ("user", "NVDA"),
        ("assistant", "headline"),
    ]
    news = next(n for n in state_db.all_node_runs_for(run_id) if n["node"] == "news")
    assert news["n_calls"] == 1 and news["tokens_in"] == 7
    [node] = [n for n in state_db.run_snapshot(run_id)["nodes"] if n["node"] == "news"]
    assert node["calls"][0]["response"] == "headline"
