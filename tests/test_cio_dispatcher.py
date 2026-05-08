"""Tests for `cio/dispatcher.py` — CLI entry, mode resolution, catch-up logic."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cio import dispatcher as cio_dispatcher
from cio.planner import CIODecision, Plan
from data import state as state_db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    state_db.init_db(db)
    return db


@pytest.fixture
def stub_cycles(monkeypatch):
    """Stub the three cycle entry points so dispatcher tests don't run
    real graphs / planners."""
    from cio import cio as cio_orchestrator

    fake_plan = Plan(
        decisions=[
            CIODecision(
                action="dismiss", ticker="X", thesis="t",
                rationale="stub", confidence="low",
            )
        ],
        drill_budget=3,
    )
    captured: dict = {"calls": []}

    async def _heartbeat(**kw):
        captured["calls"].append("heartbeat")
        return fake_plan, "heartbeat summary"

    async def _catchup(**kw):
        captured["calls"].append("catchup")
        return fake_plan, "catchup summary"

    async def _on_demand(ticker, thesis_slug=None, **kw):
        captured["calls"].append(("on_demand", ticker, thesis_slug))
        return fake_plan, "on_demand summary"

    monkeypatch.setattr(cio_orchestrator, "run_heartbeat", _heartbeat)
    monkeypatch.setattr(cio_orchestrator, "run_catchup", _catchup)
    monkeypatch.setattr(cio_orchestrator, "run_on_demand", _on_demand)
    return captured


@pytest.fixture
def stub_notify(monkeypatch):
    from cio import notify as cio_notify

    captured: dict = {}

    def _stub(plan, **kw):
        captured["plan"] = plan
        captured["trigger"] = kw["trigger"]
        return {"telegram_sent": True, "notion_url": None, "telegram_chars": 100}

    monkeypatch.setattr(cio_notify, "notify_cycle", _stub)
    return captured


# --- _hours_since ---------------------------------------------------------


def test_hours_since_handles_none():
    assert cio_dispatcher._hours_since(None) is None
    assert cio_dispatcher._hours_since("") is None


def test_hours_since_returns_age_for_recent_iso():
    iso = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    age = cio_dispatcher._hours_since(iso)
    assert age is not None
    assert 1.9 < age < 2.1


def test_hours_since_handles_invalid_iso():
    assert cio_dispatcher._hours_since("not-an-iso") is None


# --- _resolve_auto_mode --------------------------------------------------


def test_resolve_auto_first_run_returns_catchup(isolated_db):
    """No prior cycle on disk → boot path picks catchup."""
    assert cio_dispatcher._resolve_auto_mode() == "catchup"


def test_resolve_auto_recent_cycle_returns_heartbeat(isolated_db):
    rid = state_db.start_cio_run("heartbeat")
    state_db.finish_cio_run(rid, "completed")
    assert cio_dispatcher._resolve_auto_mode() == "heartbeat"


def test_resolve_auto_stale_cycle_returns_catchup(isolated_db, monkeypatch):
    """Manually back-date the most recent cycle to >8h ago → auto picks catchup."""
    import sqlite3

    rid = state_db.start_cio_run("heartbeat")
    state_db.finish_cio_run(rid, "completed")
    old_iso = (datetime.now(UTC) - timedelta(hours=10)).isoformat()
    with sqlite3.connect(isolated_db) as conn:
        conn.execute("UPDATE cio_runs SET ended_at = ? WHERE run_id = ?", (old_iso, rid))
    assert cio_dispatcher._resolve_auto_mode() == "catchup"


# --- main() / _run() -----------------------------------------------------


def test_main_auto_first_run_invokes_catchup(isolated_db, stub_cycles, stub_notify):
    """First-ever invocation → auto resolves to catchup → calls run_catchup."""
    code = cio_dispatcher.main(["--mode", "auto"])
    assert code == 0
    assert stub_cycles["calls"] == ["catchup"]


def test_main_force_heartbeat_overrides_resolution(isolated_db, stub_cycles, stub_notify):
    """`--mode heartbeat` always runs heartbeat — no freshness check."""
    code = cio_dispatcher.main(["--mode", "heartbeat"])
    assert code == 0
    assert stub_cycles["calls"] == ["heartbeat"]


def test_main_force_catchup(isolated_db, stub_cycles, stub_notify):
    code = cio_dispatcher.main(["--mode", "catchup"])
    assert code == 0
    assert stub_cycles["calls"] == ["catchup"]


def test_main_on_demand_routes_with_ticker(isolated_db, stub_cycles, stub_notify):
    code = cio_dispatcher.main(["--mode", "on_demand", "--ticker", "NVDA"])
    assert code == 0
    assert stub_cycles["calls"] == [("on_demand", "NVDA", None)]


def test_main_on_demand_routes_with_ticker_and_thesis(isolated_db, stub_cycles, stub_notify):
    code = cio_dispatcher.main(
        ["--mode", "on_demand", "--ticker", "NVDA", "--thesis", "ai_cake"]
    )
    assert code == 0
    assert stub_cycles["calls"] == [("on_demand", "NVDA", "ai_cake")]


def test_main_on_demand_without_ticker_returns_2(isolated_db, stub_cycles, stub_notify):
    """Missing --ticker for on_demand → exit code 2 (config error)."""
    code = cio_dispatcher.main(["--mode", "on_demand"])
    assert code == 2
    assert stub_cycles["calls"] == []


def test_main_invokes_notify_with_correct_trigger(isolated_db, stub_cycles, stub_notify):
    """The exec summary must be sent under the auto-resolved trigger
    (catchup), not the original 'auto' label — so dashboards know the
    cycle was a catchup."""
    cio_dispatcher.main(["--mode", "auto"])
    assert stub_notify.get("trigger") == "catchup"


def test_main_unknown_mode_returns_2(isolated_db, monkeypatch):
    """Argparse rejects unknown choices before _run is reached."""
    with pytest.raises(SystemExit):
        cio_dispatcher.main(["--mode", "bogus"])


def test_main_propagates_cycle_failure_as_exit_1(
    isolated_db, monkeypatch, stub_notify,
):
    """When the orchestrator raises, dispatcher returns 1, not 0."""
    from cio import cio as cio_orchestrator

    async def _explode(**kw):
        raise RuntimeError("synthetic cycle crash")

    monkeypatch.setattr(cio_orchestrator, "run_heartbeat", _explode)
    code = cio_dispatcher.main(["--mode", "heartbeat"])
    assert code == 1
