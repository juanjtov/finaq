"""Tests for `cio/dispatcher.py` — CLI entry, mode resolution, catch-up logic."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

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


# --- Schema v9 — stale sweep at cycle start + notify span --------------------


def test_run_sweeps_stale_rows_before_the_cycle(isolated_db, stub_cycles, stub_notify):
    """A graph run left `running` by a killed process (here: too old for the
    window, with no node row to vouch for a live pid) is closed as failed
    before the cycle starts."""
    stale = state_db.start_graph_run("AAA", "t")
    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "UPDATE graph_runs SET started_at = ? WHERE run_id = ?",
            ((datetime.now(UTC) - timedelta(hours=2)).isoformat(), stale),
        )
    assert cio_dispatcher.main(["--mode", "heartbeat"]) == 0
    run = state_db.get_graph_run(stale)
    assert (run["status"], run["error"]) == ("failed", state_db.STALE_ERROR)


def test_run_attaches_the_notify_span_to_the_latest_cycle(isolated_db, stub_cycles, stub_notify):
    """The stubbed plan carries no cio_run_id, so the dispatcher falls back to
    the most recent cycle row; without any cycle no span is written."""
    assert cio_dispatcher.main(["--mode", "heartbeat"]) == 0
    with sqlite3.connect(isolated_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 0

    rid = state_db.start_cio_run("heartbeat")
    state_db.finish_cio_run(rid, "completed")
    assert cio_dispatcher.main(["--mode", "heartbeat"]) == 0
    [notify] = state_db.spans_for("cio_run", rid)
    assert (notify["name"], notify["status"]) == ("notify", "completed")
    assert notify["meta"] == {"telegram_sent": True, "notion": False}


# --- Model preflight (2026-11-11) -------------------------------------------


def test_preflight_alerts_and_the_cycle_still_runs(
    isolated_db, stub_cycles, stub_notify, monkeypatch
):
    """A configured model OpenRouter no longer lists → one Telegram alert +
    an errors row, and the cycle proceeds (its drills get flagged degraded)."""
    from cio import notify as cio_notify

    sent: list[str] = []
    monkeypatch.setattr(
        cio_dispatcher, "unavailable_models", lambda: {"MODEL_RISK": "vendor/gone"}
    )
    monkeypatch.setattr(cio_notify, "send_telegram_message", lambda text, **kw: sent.append(text))

    assert cio_dispatcher.main(["--mode", "heartbeat"]) == 0

    assert stub_cycles["calls"] == ["heartbeat"]
    assert len(sent) == 1 and "MODEL_RISK" in sent[0] and "vendor/gone" in sent[0]
    [err] = state_db.recent_errors()
    assert err["agent"] == "cio.preflight" and "MODEL_RISK=vendor/gone" in err["message"]


def test_preflight_is_silent_when_every_model_is_listed(
    isolated_db, stub_cycles, stub_notify, monkeypatch
):
    from cio import notify as cio_notify

    monkeypatch.setattr(cio_dispatcher, "unavailable_models", lambda: {})
    monkeypatch.setattr(
        cio_notify, "send_telegram_message", lambda *a, **k: pytest.fail("no alert expected")
    )
    assert cio_dispatcher.main(["--mode", "heartbeat"]) == 0
    assert state_db.recent_errors() == []


# --- Process exit (2026-11-13) ----------------------------------------------


def test_exit_arms_a_daemon_watchdog_then_exits_normally(monkeypatch):
    """Clean path: an ordinary SystemExit, with a daemon timer armed so a
    stuck worker thread cannot hold the process past the grace period."""
    armed: dict = {}

    class _Timer:
        def __init__(self, interval, fn):
            armed.update(interval=interval, fn=fn)

        def start(self):
            armed.update(started=True, daemon=self.daemon)

    monkeypatch.setattr(cio_dispatcher.threading, "Timer", _Timer)

    with pytest.raises(SystemExit) as exc:
        cio_dispatcher._exit(2, grace_s=7.0)

    assert exc.value.code == 2
    assert armed["interval"] == 7.0 and armed["started"] and armed["daemon"] is True


def test_exit_leaves_despite_a_stuck_worker_thread():
    """Regression (2026-11-12): a worker blocked in a network read kept the
    finished dispatcher alive for 32 hours, and launchd skipped every cycle
    behind it. Real process, real non-daemon thread that never returns."""
    script = (
        "import threading\n"
        "threading.Thread(target=threading.Event().wait).start()\n"
        "from cio import dispatcher\n"
        "dispatcher._exit(3, grace_s=0.5)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 3, proc.stderr[-500:]
    assert "exiting without it" in proc.stderr
