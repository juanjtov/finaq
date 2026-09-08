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


@pytest.fixture(autouse=True)
def stub_news(monkeypatch):
    """Never hit Tavily or EDGAR from this suite. Autouse because the
    orchestrator now runs the freshness check for every candidate
    ticker before deciding."""
    monkeypatch.setattr(cio_mod, "_fetch_news", lambda t: [])
    monkeypatch.setattr(cio_mod, "_probe_freshness", lambda t: None)

    async def _no_ingest(ticker, report):
        pytest.fail("no test in this suite expects an auto-ingest unless it stubs one")

    monkeypatch.setattr(cio_mod, "_auto_ingest", _no_ingest)


def _stale_report(ticker: str, edgar_date: str = "2026-08-05"):
    """A FreshnessReport saying EDGAR has a 10-Q the corpus lacks."""
    from data.freshness import FormDiff, FreshnessReport

    return FreshnessReport(
        ticker=ticker, is_stale=True,
        per_form=[FormDiff(form="10-Q", edgar_date=edgar_date, ingested_date=None, behind_days=0)],
    )


def _stub_decide_factory(canned: dict[tuple[str, str], CIODecision]):
    """Returns a `planner.decide` stub that looks up its response by
    (ticker, thesis_slug). Default to dismiss for unmapped pairs.

    Step 11.19 — `decide` now returns `(CIODecision, telemetry_dict)`.
    The stub returns a synthetic telemetry payload so the orchestrator's
    cost-aggregation path is exercised in tests too.
    """

    def _stub(*, ticker, thesis, **kw):
        slug = (thesis or {}).get("slug") if isinstance(thesis, dict) else None
        key = (ticker.upper(), slug or "")
        if key in canned:
            decision = canned[key]
        else:
            decision = CIODecision(
                action="dismiss", ticker=ticker.upper(), thesis=slug,
                rationale="default stub", confidence="low",
            )
        telemetry = {
            "model_used": "stub-model",
            "tokens_in": 100,
            "tokens_out": 50,
            "cost_usd": 0.001,
            "latency_s": 0.5,
        }
        return decision, telemetry

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
        return (
            CIODecision(
                action="dismiss", ticker=ticker.upper(),
                thesis=(thesis or {}).get("slug"),
                rationale="ok", confidence="low",
            ),
            {"model_used": "stub-model", "tokens_in": 0, "tokens_out": 0,
             "cost_usd": 0.0, "latency_s": 0.1},
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


# --- Step 11.19 — telemetry roundtrip (model + cost + latency) -----------


@pytest.mark.asyncio
async def test_cycle_aggregates_per_action_cost_onto_cio_runs(
    isolated_db, fake_thesis_dir, stub_news, monkeypatch,
):
    """Each cio_actions row carries its own (model, tokens, cost, latency).
    cio_runs.total_cost_usd should equal sum(cio_actions.cost_usd) for the
    cycle, and cio_runs.model_used should be the planner's MODEL_CIO."""
    canned = {
        ("AAA", "curated_a"): CIODecision(action="dismiss", ticker="AAA",
                                            thesis="curated_a",
                                            rationale="quiet", confidence="low"),
        ("BBB", "curated_a"): CIODecision(action="drill", ticker="BBB",
                                            thesis="curated_a",
                                            rationale="x", confidence="high"),
    }

    # Stub returns a custom cost per pair so we can verify aggregation.
    def _stub(*, ticker, thesis, **kw):
        slug = (thesis or {}).get("slug") if isinstance(thesis, dict) else None
        decision = canned.get(
            (ticker.upper(), slug or ""),
            CIODecision(action="dismiss", ticker=ticker.upper(), thesis=slug,
                        rationale="default", confidence="low"),
        )
        # Distinct costs per ticker so the sum is provable, not a coincidence.
        cost = 0.001 if ticker.upper() == "AAA" else 0.002
        telemetry = {
            "model_used": "openai/gpt-5.4-mini",
            "tokens_in": 2500, "tokens_out": 80,
            "cost_usd": cost, "latency_s": 1.2,
        }
        return decision, telemetry

    monkeypatch.setattr(cio_planner, "decide", _stub)

    async def _fake_drill(ticker, thesis):
        return f"r-{ticker}"

    monkeypatch.setattr(cio_mod, "_drill_one", _fake_drill)

    plan, _ = await cio_mod.run_heartbeat()

    # Inspect what landed.
    runs = state_db.recent_cio_runs(limit=1)
    assert runs[0]["model_used"] == "openai/gpt-5.4-mini"

    actions = state_db.recent_cio_actions(limit=20)
    expected_total = sum((a.get("cost_usd") or 0.0) for a in actions)
    assert runs[0]["total_cost_usd"] == pytest.approx(expected_total, rel=1e-9)
    assert runs[0]["total_cost_usd"] > 0.0  # something was spent

    # Each row's per-call telemetry round-trips.
    aaa = next(a for a in actions if a["ticker"] == "AAA")
    bbb = next(a for a in actions if a["ticker"] == "BBB")
    assert aaa["model_used"] == "openai/gpt-5.4-mini"
    assert aaa["cost_usd"] == pytest.approx(0.001)
    assert bbb["cost_usd"] == pytest.approx(0.002)
    assert aaa["latency_s"] == pytest.approx(1.2)


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


# --- Sept 2026 regressions: freshness before decide, lazy news, source ---


@pytest.mark.asyncio
async def test_freshness_check_runs_before_decide_once_per_ticker(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    """The EDGAR probe must run for every candidate ticker BEFORE
    `planner.decide` sees it, and only once per ticker even when the
    ticker sits in two theses. Its report reaches `decide` as
    `edgar_freshness`. Previously freshness ran inside `_drill_one`,
    i.e. after the decision it was meant to inform."""
    events: list[tuple[str, str]] = []
    reports = {"AAA": _stale_report("AAA"), "BBB": None}

    def _probe(ticker):
        events.append(("probe", ticker))
        return reports[ticker]

    monkeypatch.setattr(cio_mod, "_probe_freshness", _probe)
    seen_freshness: dict[str, object] = {}

    def _decide(*, ticker, thesis, edgar_freshness=None, **kw):
        events.append(("decide", ticker.upper()))
        seen_freshness[ticker.upper()] = edgar_freshness
        return (
            CIODecision(action="dismiss", ticker=ticker.upper(),
                        thesis=thesis.get("slug"), rationale="q", confidence="low"),
            cio_planner._empty_telemetry(),
        )

    monkeypatch.setattr(cio_planner, "decide", _decide)
    # AAA in both theses → cross-thesis multiplicity. Neither is an anchor
    # of curated_a in the candidate override below, so nothing ingests.
    monkeypatch.setattr(
        cio_mod, "_curated_candidates",
        lambda: [("BBB", "curated_a"), ("AAA", "curated_a"), ("AAA", "curated_b")],
    )
    monkeypatch.setattr(cio_mod, "_anchor_tickers", lambda: set())

    await cio_mod.run_heartbeat()

    probes = [t for kind, t in events if kind == "probe"]
    assert probes == ["BBB", "AAA"]  # once per ticker, candidate order
    for t in ("AAA", "BBB"):
        assert events.index(("probe", t)) < events.index(("decide", t))
    assert seen_freshness["AAA"] is reports["AAA"]
    assert seen_freshness["BBB"] is None


@pytest.mark.asyncio
async def test_news_is_fetched_lazily_and_once_per_ticker(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    """Tavily must fire only when the planner asks (via `news_fetcher`),
    and at most once per ticker per cycle — 57 eager calls twice a day
    burned the monthly quota in four days."""
    fetched: list[str] = []
    monkeypatch.setattr(cio_mod, "_fetch_news", lambda t: fetched.append(t) or [])

    def _decide(*, ticker, thesis, news_fetcher=None, **kw):
        # AAA takes the LLM path and asks twice (still one Tavily call);
        # BBB is a gate shortcut and never asks.
        if ticker.upper() == "AAA":
            assert news_fetcher() == []
            news_fetcher()
        return (
            CIODecision(action="dismiss", ticker=ticker.upper(),
                        thesis=thesis.get("slug"), rationale="q", confidence="low"),
            cio_planner._empty_telemetry(),
        )

    monkeypatch.setattr(cio_planner, "decide", _decide)
    monkeypatch.setattr(
        cio_mod, "_curated_candidates",
        lambda: [("AAA", "curated_a"), ("BBB", "curated_a"), ("AAA", "curated_b")],
    )

    await cio_mod.run_heartbeat()
    assert fetched == ["AAA"]


@pytest.mark.asyncio
async def test_decision_source_lands_on_cio_actions(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    """`source` round-trips from CIODecision to the cio_actions row —
    including the orchestrator's own planner-error fallback."""
    n = {"i": 0}

    def _decide(*, ticker, thesis, **kw):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("boom")
        src = "gate" if ticker.upper() == "BBB" else "llm"
        return (
            CIODecision(action="dismiss", ticker=ticker.upper(),
                        thesis=thesis.get("slug"), rationale="q",
                        confidence="low", source=src),
            cio_planner._empty_telemetry(),
        )

    monkeypatch.setattr(cio_planner, "decide", _decide)
    await cio_mod.run_heartbeat()

    by_ticker = {a["ticker"]: a for a in state_db.recent_cio_actions(limit=20)}
    assert by_ticker["AAA"]["source"] == "fallback"  # first call raised
    assert by_ticker["BBB"]["source"] == "gate"
    assert by_ticker["CCC"]["source"] == "llm"


# --- Anchors-only auto-ingest (user decision 2026-09-07) -----------------


@pytest.mark.asyncio
async def test_heartbeat_auto_ingests_anchor_tickers_only(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    """Every ticker is stale at EDGAR, but only the anchors (AAA for
    curated_a, DDD for curated_b) get the expensive ingest. The full
    sweep on 2026-09-07 ran for hours embedding 30 tickers nobody asked for."""
    monkeypatch.setattr(cio_mod, "_probe_freshness", lambda t: _stale_report(t))
    ingested: list[str] = []

    async def _ingest(ticker, report):
        ingested.append(ticker)

    monkeypatch.setattr(cio_mod, "_auto_ingest", _ingest)
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory({}))

    await cio_mod.run_heartbeat()

    assert sorted(ingested) == ["AAA", "DDD"]
    assert cio_mod._anchor_tickers() == {"AAA", "DDD"}


@pytest.mark.asyncio
async def test_heartbeat_skips_ingest_for_fresh_anchor(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    """An anchor whose corpus already matches EDGAR is not ingested."""
    from data.freshness import FreshnessReport

    monkeypatch.setattr(
        cio_mod, "_probe_freshness",
        lambda t: FreshnessReport(ticker=t, is_stale=False, per_form=[]),
    )
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory({}))
    await cio_mod.run_heartbeat()  # autouse `_auto_ingest` stub fails if called


@pytest.mark.asyncio
async def test_on_demand_auto_ingests_the_requested_ticker(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    """`/cio BBB curated_a` — BBB is not an anchor, but the user asked."""
    monkeypatch.setattr(cio_mod, "_probe_freshness", lambda t: _stale_report(t))
    ingested: list[str] = []

    async def _ingest(ticker, report):
        ingested.append(ticker)

    monkeypatch.setattr(cio_mod, "_auto_ingest", _ingest)
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory({}))

    await cio_mod.run_on_demand("BBB", "curated_a")
    assert ingested == ["BBB"]


@pytest.mark.asyncio
async def test_drill_on_stale_non_anchor_ingests_right_before_drill(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    """The planner drills BBB (non-anchor, stale): ingest happens once,
    after the decision and before `_drill_one`. AAA (anchor, stale) was
    already ingested in the probe pass and must not be ingested twice."""
    monkeypatch.setattr(cio_mod, "_probe_freshness", lambda t: _stale_report(t))
    events: list[tuple[str, str]] = []

    async def _ingest(ticker, report):
        events.append(("ingest", ticker))

    async def _drill(ticker, thesis):
        events.append(("drill", ticker))
        return f"run-{ticker}"

    monkeypatch.setattr(cio_mod, "_auto_ingest", _ingest)
    monkeypatch.setattr(cio_mod, "_drill_one", _drill)
    canned = {
        ("AAA", "curated_a"): CIODecision(action="drill", ticker="AAA", thesis="curated_a",
                                           rationale="x", confidence="high"),
        ("BBB", "curated_a"): CIODecision(action="drill", ticker="BBB", thesis="curated_a",
                                           rationale="x", confidence="high"),
    }
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory(canned))

    await cio_mod.run_heartbeat()

    assert events.count(("ingest", "AAA")) == 1
    assert events.count(("ingest", "BBB")) == 1
    assert events.index(("ingest", "BBB")) < events.index(("drill", "BBB"))
    # BBB's ingest is drill-time: it comes after every probe-pass event.
    assert events.index(("ingest", "BBB")) > events.index(("ingest", "AAA"))
    assert ("drill", "CCC") not in events and ("ingest", "CCC") not in events


# --- Overdue-thesis nag in the cycle summary (2026-09-07) ------------------


@pytest.mark.asyncio
async def test_cycle_flags_theses_overdue_for_review(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    """curated_a was last reviewed in January → overdue; curated_b today →
    fine. The plan carries the list and the stored summary names it."""
    from data import theses as theses_lifecycle

    def _stamp(slug: str, day: str) -> None:
        p = fake_thesis_dir / f"{slug}.json"
        d = json.loads(p.read_text())
        d["last_reviewed"] = day
        p.write_text(json.dumps(d))

    _stamp("curated_a", "2026-01-01")
    _stamp("curated_b", __import__("datetime").date.today().isoformat())
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory({}))

    plan, summary = await cio_mod.run_heartbeat()

    assert [o["slug"] for o in plan.overdue_theses] == ["curated_a"]
    assert plan.overdue_theses[0]["age_days"] > theses_lifecycle.REVIEW_MAX_DAYS
    assert "overdue for review" in summary
    assert "curated_a" in summary
    assert "curated_b" not in summary.split("Decisions:")[0]
    assert "overdue for review" in state_db.recent_cio_runs(limit=1)[0]["summary"]


@pytest.mark.asyncio
async def test_cycle_summary_silent_when_all_theses_reviewed(
    isolated_db, fake_thesis_dir, monkeypatch,
):
    today = __import__("datetime").date.today().isoformat()
    for slug in ("curated_a", "curated_b"):
        p = fake_thesis_dir / f"{slug}.json"
        d = json.loads(p.read_text())
        d["last_reviewed"] = today
        p.write_text(json.dumps(d))
    monkeypatch.setattr(cio_planner, "decide", _stub_decide_factory({}))

    plan, summary = await cio_mod.run_heartbeat()
    assert plan.overdue_theses == []
    assert "overdue" not in summary.lower()
