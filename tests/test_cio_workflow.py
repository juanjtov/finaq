"""Workflow tests for `cio/cio.py` — the orchestrator that decides,
applies the drill budget, and executes drills.

Stubs out:
  - `planner.decide`            — canned CIODecisions per call
  - `_drill_one`                — returns a fake run_id without invoking the graph
  - `_fetch_news`               — empty list (Tavily not exercised)
  - `_curated_candidates`       — fixed list of pairs

This lets us assert the cycle's contract end-to-end:
  - drill budget cap applied
  - cio_runs row opened + closed with correct counts
  - cio_actions rows persisted with proper trigger / decision / drill_run_id
  - on-demand exempt from drill budget
  - catch-up tag flows through to the cio_runs row
  - LLM error → planner returns dismiss → cycle continues
  - rerun yo-yo guard short-circuits to dismiss without drill
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from cio import cio as cio_mod
from cio import planner as cio_planner
from cio.planner import CIODecision, Plan
from data import state as state_db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    state_db.init_db(db)
    return db


@pytest.fixture
def fake_thesis_dir(tmp_path, monkeypatch):
    """A 2-thesis dir: `curated_a` (3 tickers) + `curated_b` (2 tickers)
    + one `adhoc_x` to verify heartbeat skips adhocs."""
    theses = tmp_path / "theses"
    theses.mkdir()

    def _write(slug: str, universe: list[str], anchors: list[str]):
        (theses / f"{slug}.json").write_text(
            json.dumps(
                {
                    "name": slug,
                    "summary": f"{slug} summary",
                    "anchor_tickers": anchors,
                    "universe": universe,
                    "relationships": [],
                    "material_thresholds": [],
                }
            )
        )

    _write("curated_a", ["AAA", "BBB", "CCC"], ["AAA"])
    _write("curated_b", ["DDD", "EEE"], ["DDD"])
    _write("adhoc_x", ["ZZZ"], ["ZZZ"])  # MUST be skipped on heartbeat

    monkeypatch.setattr(cio_mod, "THESES_DIR", theses)
    return theses


@pytest.fixture
def stub_news(monkeypatch):
    monkeypatch.setattr(cio_mod, "_fetch_news", lambda t, n=None: [])


def _stub_decide_factory(canned: dict[tuple[str, str], CIODecision]):
    """Returns a `planner.decide` stub that looks up its response by
    (ticker, thesis_slug). Default to dismiss for unmapped pairs."""

    def _stub(*, ticker, thesis, **kw):
        slug = (thesis or {}).get("slug") if isinstance(thesis, dict) else None
        key = (ticker.upper(), slug or "")
        if key in canned:
            return canned[key]
        return CIODecision(
            action="dismiss", ticker=ticker.upper(), thesis=slug,
            rationale="default stub", confidence="low",
        )

    return _stub


# --- Heartbeat: curated-only sweep ----------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_skips_adhoc_theses(isolated_db, fake_thesis_dir, stub_news, monkeypatch):
    """Adhoc theses must NOT be in the candidate list for heartbeat."""
    candidates = cio_mod._curated_candidates()
    slugs = {slug for _, slug in candidates}
    assert "adhoc_x" not in slugs
    assert {"curated_a", "curated_b"}.issubset(slugs)


@pytest.mark.asyncio
async def test_heartbeat_decides_drills_records(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """Happy path: 5 candidates, 2 drills proposed (within budget=3).
    Expect 2 drills executed, 3 dismisses recorded, cio_runs row closed."""
    canned = {
        ("AAA", "curated_a"): CIODecision(
            action="drill", ticker="AAA", thesis="curated_a",
            rationale="drill it", confidence="high",
        ),
        ("DDD", "curated_b"): CIODecision(
            action="drill", ticker="DDD", thesis="curated_b",
            rationale="drill it too", confidence="high",
        ),
    }
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))
    drilled: list[str] = []

    async def _fake_drill(ticker, thesis):
        drilled.append(ticker)
        return f"fake-run-{ticker}"

    monkeypatch.setattr(cio_mod, "_drill_one", _fake_drill)

    plan, summary = await cio_mod.run_heartbeat()

    assert plan.n_drilled == 2
    assert plan.n_dismissed == 3
    assert plan.drills_capped == 0
    assert set(drilled) == {"AAA", "DDD"}
    assert "drilled, 3 reused" not in summary  # dismiss tally lives in summary
    assert "2 drilled" in summary

    # cio_runs row is closed with completed status.
    runs = state_db.recent_cio_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["n_drilled"] == 2
    assert runs[0]["n_dismissed"] == 3

    # cio_actions rows: 5 total, 2 drills, 3 dismisses.
    actions = state_db.recent_cio_actions(limit=20)
    assert len(actions) == 5
    drill_actions = [a for a in actions if a["action"] == "drill"]
    assert len(drill_actions) == 2
    assert all(a["drill_run_id"] is not None for a in drill_actions)


@pytest.mark.asyncio
async def test_heartbeat_drill_budget_caps_at_three(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """5 drill proposals, budget=3 → 2 demoted to dismiss (no prior drills
    so reuse path can't trigger)."""
    canned = {
        ("AAA", "curated_a"): CIODecision(action="drill", ticker="AAA", thesis="curated_a",
                                           rationale="x", confidence="low"),
        ("BBB", "curated_a"): CIODecision(action="drill", ticker="BBB", thesis="curated_a",
                                           rationale="x", confidence="high"),
        ("CCC", "curated_a"): CIODecision(action="drill", ticker="CCC", thesis="curated_a",
                                           rationale="x", confidence="medium"),
        ("DDD", "curated_b"): CIODecision(action="drill", ticker="DDD", thesis="curated_b",
                                           rationale="x", confidence="high"),
        ("EEE", "curated_b"): CIODecision(action="drill", ticker="EEE", thesis="curated_b",
                                           rationale="x", confidence="low"),
    }
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    drilled: list[str] = []

    async def _fake_drill(ticker, thesis):
        drilled.append(ticker)
        return f"fake-run-{ticker}"

    monkeypatch.setattr(cio_mod, "_drill_one", _fake_drill)

    plan, _ = await cio_mod.run_heartbeat(drill_budget=3)

    assert plan.n_drilled == 3
    assert plan.drills_capped == 2
    # Demoted drills become dismiss (no prior runs on disk for reuse).
    demoted = [d for d in plan.decisions if d.action == "dismiss" and "budget cap" in d.rationale]
    assert len(demoted) == 2
    # Highest-confidence drills survived (BBB, CCC, DDD all high or medium).
    assert set(drilled) == {"BBB", "CCC", "DDD"}


@pytest.mark.asyncio
async def test_heartbeat_planner_exception_demotes_to_dismiss(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """A planner crash for one ticker must NOT poison the cycle — the
    orchestrator records a deterministic dismiss and continues."""
    counter = {"n": 0}

    def _flaky_decide(*, ticker, thesis, **kw):
        counter["n"] += 1
        if counter["n"] == 1:
            raise RuntimeError("synthetic planner crash")
        return CIODecision(
            action="dismiss", ticker=ticker.upper(),
            thesis=(thesis or {}).get("slug"),
            rationale="ok", confidence="low",
        )

    monkeypatch.setattr(cio_planner, "decide", _flaky_decide)

    async def _no_drill(*a, **k):
        pytest.fail("must not drill on a planner-error pair")

    monkeypatch.setattr(cio_mod, "_drill_one", _no_drill)

    plan, _ = await cio_mod.run_heartbeat()

    # Cycle survived: every candidate produced a decision (5 total).
    assert len(plan.decisions) == 5
    # The first one became a deterministic dismiss with planner-error rationale.
    first_dismiss = next(
        (d for d in plan.decisions if "planner error" in d.rationale.lower()),
        None,
    )
    assert first_dismiss is not None
    assert first_dismiss.action == "dismiss"


# --- Catch-up: same as heartbeat, different trigger ----------------------


@pytest.mark.asyncio
async def test_catchup_writes_catchup_trigger(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    canned = {("AAA", "curated_a"): CIODecision(
        action="drill", ticker="AAA", thesis="curated_a",
        rationale="x", confidence="high",
    )}
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    async def _fake_drill(ticker, thesis):
        return f"fake-{ticker}"

    monkeypatch.setattr(cio_mod, "_drill_one", _fake_drill)

    plan, _ = await cio_mod.run_catchup()

    runs = state_db.recent_cio_runs()
    assert len(runs) == 1
    assert runs[0]["trigger"] == "catchup"
    actions = state_db.recent_cio_actions()
    drill_action = next(a for a in actions if a["action"] == "drill")
    assert drill_action["trigger"] == "catchup"


# --- On-demand: single-pair, exempt from budget --------------------------


@pytest.mark.asyncio
async def test_on_demand_single_pair(isolated_db, fake_thesis_dir, stub_news, monkeypatch):
    """`/cio NVDA ai_cake` → exactly one decision."""
    canned = {("AAA", "curated_a"): CIODecision(
        action="drill", ticker="AAA", thesis="curated_a",
        rationale="x", confidence="high",
    )}
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    async def _fake_drill(ticker, thesis):
        return "fake-run"

    monkeypatch.setattr(cio_mod, "_drill_one", _fake_drill)

    plan, _ = await cio_mod.run_on_demand("AAA", "curated_a")
    assert len(plan.decisions) == 1
    assert plan.decisions[0].action == "drill"

    runs = state_db.recent_cio_runs()
    assert runs[0]["trigger"] == "on_demand"


@pytest.mark.asyncio
async def test_on_demand_resolves_thesis_from_universe(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """`/cio AAA` (no thesis) → finds every thesis whose universe has
    AAA. Here AAA is only in curated_a, so we get 1 candidate."""
    canned = {("AAA", "curated_a"): CIODecision(
        action="reuse", ticker="AAA", thesis="curated_a",
        rationale="still applies", confidence="medium",
        reuse_run_id="prior-run-id",
    )}
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    plan, _ = await cio_mod.run_on_demand("AAA")
    assert len(plan.decisions) == 1
    assert plan.decisions[0].thesis == "curated_a"
    assert plan.decisions[0].action == "reuse"


@pytest.mark.asyncio
async def test_on_demand_exempt_from_default_drill_budget(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """A multi-thesis ticker on-demand can drill more than 3 times if the
    user explicitly asks (drill_budget defaults to len(candidates))."""
    # Make a ticker present in all 5 candidates of curated_a (3 tickers)
    # by overriding _curated_candidates indirectly. Simpler: rely on the
    # fact run_on_demand defaults drill_budget to len(candidates).
    canned = {
        ("AAA", "curated_a"): CIODecision(action="drill", ticker="AAA", thesis="curated_a",
                                           rationale="x", confidence="high"),
        ("BBB", "curated_a"): CIODecision(action="drill", ticker="BBB", thesis="curated_a",
                                           rationale="x", confidence="high"),
    }
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    drilled: list[str] = []

    async def _fake_drill(ticker, thesis):
        drilled.append(ticker)
        return f"fake-{ticker}"

    monkeypatch.setattr(cio_mod, "_drill_one", _fake_drill)

    # Force candidates by passing thesis explicitly.
    plan_a, _ = await cio_mod.run_on_demand("AAA", "curated_a")
    plan_b, _ = await cio_mod.run_on_demand("BBB", "curated_a")

    # Both single-pair cycles drilled with budget=1 each (not capped).
    assert plan_a.drills_capped == 0
    assert plan_b.drills_capped == 0
    assert "AAA" in drilled and "BBB" in drilled


# --- _drill_one failure path ---------------------------------------------


@pytest.mark.asyncio
async def test_drill_failure_is_recorded_with_null_run_id(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """When the graph crashes, _drill_one returns None — the cio_actions
    row still records the decision, just without a drill_run_id."""
    canned = {("AAA", "curated_a"): CIODecision(
        action="drill", ticker="AAA", thesis="curated_a",
        rationale="x", confidence="high",
    )}
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    async def _broken_drill(ticker, thesis):
        return None  # _drill_one returns None on internal exception

    monkeypatch.setattr(cio_mod, "_drill_one", _broken_drill)

    plan, _ = await cio_mod.run_on_demand("AAA", "curated_a")
    assert plan.decisions[0].action == "drill"

    actions = state_db.recent_cio_actions(ticker="AAA", thesis="curated_a")
    assert len(actions) == 1
    assert actions[0]["action"] == "drill"
    assert actions[0]["drill_run_id"] is None  # graph crashed → None recorded


# --- Cycle completion + summary ------------------------------------------


@pytest.mark.asyncio
async def test_cycle_summary_lists_each_decision(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    canned = {("AAA", "curated_a"): CIODecision(
        action="drill", ticker="AAA", thesis="curated_a",
        rationale="capex announcement landed", confidence="high",
    )}
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    async def _fake_drill(ticker, thesis):
        return f"r-{ticker}"

    monkeypatch.setattr(cio_mod, "_drill_one", _fake_drill)

    plan, summary = await cio_mod.run_heartbeat()

    assert "AAA" in summary
    assert "DRILL" in summary or "drill" in summary
    runs = state_db.recent_cio_runs()
    assert runs[0]["summary"] is not None
    assert "AAA" in runs[0]["summary"]


# --- Cooldown / reuse interaction ----------------------------------------


@pytest.mark.asyncio
async def test_reuse_decision_records_reuse_run_id(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """When the LLM picks `reuse`, the decision's reuse_run_id must land
    on the cio_actions row so Mission Control can render a deep-link."""
    rid = state_db.start_graph_run("AAA", "curated_a")
    state_db.finish_graph_run(rid, "completed")

    canned = {("AAA", "curated_a"): CIODecision(
        action="reuse", ticker="AAA", thesis="curated_a",
        rationale="still applies as of today", confidence="high",
        reuse_run_id=rid,
    )}
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    async def _no_drill(*a, **k):
        pytest.fail("must NOT drill on a reuse decision")

    monkeypatch.setattr(cio_mod, "_drill_one", _no_drill)

    await cio_mod.run_heartbeat()

    actions = state_db.recent_cio_actions(ticker="AAA", thesis="curated_a")
    reuse_action = next(a for a in actions if a["action"] == "reuse")
    assert reuse_action["reuse_run_id"] == rid
    assert reuse_action["drill_run_id"] is None


@pytest.mark.asyncio
async def test_low_confidence_reuse_recorded_with_low_confidence(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """The `confidence` field must round-trip onto cio_actions so the
    notify formatter can render the qualifier ('still applies' for
    high, 1-line confirm for low)."""
    rid = state_db.start_graph_run("AAA", "curated_a")
    state_db.finish_graph_run(rid, "completed")

    canned = {("AAA", "curated_a"): CIODecision(
        action="reuse", ticker="AAA", thesis="curated_a",
        rationale="ok-ish, watch next quarter", confidence="low",
        reuse_run_id=rid,
    )}
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    await cio_mod.run_heartbeat()

    actions = state_db.recent_cio_actions(ticker="AAA", thesis="curated_a")
    a = next(x for x in actions if x["action"] == "reuse")
    assert a["confidence"] == "low"


# --- Drill-budget demotion to reuse when prior drill exists --------------


@pytest.mark.asyncio
async def test_budget_cap_demotes_to_reuse_when_prior_drill_exists(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """When budget caps a drill but the pair has a recent completed run,
    the orchestrator demotes to reuse (not dismiss) so the user still
    sees relevant content."""
    # Seed prior drill on AAA so cooldown has a run_id to fall back to.
    rid = state_db.start_graph_run("AAA", "curated_a")
    state_db.finish_graph_run(rid, "completed")

    canned = {
        ("AAA", "curated_a"): CIODecision(action="drill", ticker="AAA", thesis="curated_a",
                                            rationale="x", confidence="low"),  # likely demoted
        ("BBB", "curated_a"): CIODecision(action="drill", ticker="BBB", thesis="curated_a",
                                            rationale="x", confidence="high"),
        ("CCC", "curated_a"): CIODecision(action="drill", ticker="CCC", thesis="curated_a",
                                            rationale="x", confidence="high"),
        ("DDD", "curated_b"): CIODecision(action="drill", ticker="DDD", thesis="curated_b",
                                            rationale="x", confidence="high"),
    }
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    async def _fake_drill(ticker, thesis):
        return f"new-{ticker}"

    monkeypatch.setattr(cio_mod, "_drill_one", _fake_drill)

    plan, _ = await cio_mod.run_heartbeat(drill_budget=3)
    assert plan.drills_capped == 1

    aaa_action = next(d for d in plan.decisions if d.ticker == "AAA")
    assert aaa_action.action == "reuse"
    assert aaa_action.reuse_run_id == rid


# --- Stable ordering when actions are ties --------------------------------


@pytest.mark.asyncio
async def test_decisions_preserve_candidate_order(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """Decisions in the plan must land in the same order as candidates
    were proposed so the exec summary lists anchors first (which is
    the order `_curated_candidates` returns them)."""
    candidates = cio_mod._curated_candidates()
    canned = {
        (t, s): CIODecision(action="dismiss", ticker=t, thesis=s,
                             rationale="quiet", confidence="low")
        for (t, s) in candidates
    }
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    plan, _ = await cio_mod.run_heartbeat()

    expected_pairs = [(t, s) for (t, s) in candidates]
    actual_pairs = [(d.ticker, d.thesis) for d in plan.decisions]
    assert actual_pairs == expected_pairs
