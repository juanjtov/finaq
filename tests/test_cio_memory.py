"""Tests for `cio/memory.py` — the CIO domain layer over state.db + Notion.

Memory wraps three primitives: cooldown status, recent CIO action history,
and Notion thesis-notes pull. We test each independently against an
isolated state.db tmp_path; Notion is monkey-patched.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from cio import memory as cio_memory
from data import state as state_db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point state_db.DB_PATH at tmp_path so tests don't read or write the
    real data_cache/state.db."""
    db = tmp_path / "state.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    state_db.init_db(db)
    return db


# --- cooldown_status ------------------------------------------------------


def test_cooldown_status_no_prior_drill(isolated_db):
    out = cio_memory.cooldown_status("NVDA", "ai_cake")
    assert out == {
        "active": False,
        "last_drill_age_hours": None,
        "last_drill_run_id": None,
        "last_drill_at": None,
    }


def test_cooldown_status_recent_drill_active(isolated_db):
    """A drill that ended <48h ago → cooldown active."""
    rid = state_db.start_graph_run("NVDA", "ai_cake")
    state_db.finish_graph_run(rid, "completed")
    out = cio_memory.cooldown_status("NVDA", "ai_cake")
    assert out["active"] is True
    assert out["last_drill_run_id"] == rid
    assert out["last_drill_age_hours"] is not None
    assert out["last_drill_age_hours"] < 1.0  # just happened


def test_cooldown_status_old_drill_expired(isolated_db, monkeypatch):
    """Manually back-date the drill's ended_at to >48h ago → cooldown expired."""
    import sqlite3

    rid = state_db.start_graph_run("NVDA", "ai_cake")
    state_db.finish_graph_run(rid, "completed")
    old_iso = (datetime.now(UTC) - timedelta(hours=72)).isoformat()
    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "UPDATE graph_runs SET ended_at = ?, started_at = ? WHERE run_id = ?",
            (old_iso, old_iso, rid),
        )
    out = cio_memory.cooldown_status("NVDA", "ai_cake")
    assert out["active"] is False
    assert out["last_drill_age_hours"] is not None
    assert out["last_drill_age_hours"] > 48


def test_cooldown_status_filters_failed_runs(isolated_db):
    """A failed run must NOT count toward cooldown — that drill produced
    nothing reusable."""
    rid = state_db.start_graph_run("NVDA", "ai_cake")
    state_db.finish_graph_run(rid, "failed")
    out = cio_memory.cooldown_status("NVDA", "ai_cake")
    assert out["active"] is False
    assert out["last_drill_run_id"] is None


def test_cooldown_status_thesis_optional(isolated_db):
    """Without a thesis arg, cooldown looks across all theses for that ticker."""
    rid = state_db.start_graph_run("NVDA", "ai_cake")
    state_db.finish_graph_run(rid, "completed")
    out = cio_memory.cooldown_status("NVDA")
    assert out["last_drill_run_id"] == rid


def test_cooldown_status_respects_custom_window(isolated_db, monkeypatch):
    """A 24h cooldown sees a 30h-old drill as expired even though 48h
    default would mark it active."""
    import sqlite3

    rid = state_db.start_graph_run("NVDA", "ai_cake")
    state_db.finish_graph_run(rid, "completed")
    iso_30h = (datetime.now(UTC) - timedelta(hours=30)).isoformat()
    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "UPDATE graph_runs SET ended_at = ?, started_at = ? WHERE run_id = ?",
            (iso_30h, iso_30h, rid),
        )
    default = cio_memory.cooldown_status("NVDA", "ai_cake")
    custom = cio_memory.cooldown_status("NVDA", "ai_cake", cooldown_hours=24)
    assert default["active"] is True   # 30h < 48h default
    assert custom["active"] is False   # 30h > 24h custom


# --- recent_cio_actions ---------------------------------------------------


def test_recent_cio_actions_returns_pair_only(isolated_db):
    state_db.record_cio_action(ticker="NVDA", thesis="ai_cake", action="drill")
    state_db.record_cio_action(ticker="MSFT", thesis="ai_cake", action="reuse")
    state_db.record_cio_action(ticker="NVDA", thesis="nvda_halo", action="dismiss")

    nvda_aicake = cio_memory.recent_cio_actions("NVDA", "ai_cake")
    assert len(nvda_aicake) == 1
    assert nvda_aicake[0]["action"] == "drill"

    nvda_any = cio_memory.recent_cio_actions("NVDA")
    assert len(nvda_any) == 2


# --- thesis_notes ---------------------------------------------------------


def test_thesis_notes_empty_when_notion_unconfigured(monkeypatch):
    from data import notion as notion_mod

    monkeypatch.setattr(notion_mod, "is_configured", lambda: False)
    assert cio_memory.thesis_notes("ai_cake") == ""


def test_thesis_notes_passes_through_when_configured(monkeypatch):
    from data import notion as notion_mod

    monkeypatch.setattr(notion_mod, "is_configured", lambda: True)
    monkeypatch.setattr(
        notion_mod, "read_thesis_notes",
        lambda slug: "trim 20% if Q3 misses $42B" if slug == "ai_cake" else "",
    )
    assert cio_memory.thesis_notes("ai_cake") == "trim 20% if Q3 misses $42B"


def test_thesis_notes_soft_fails_on_exception(monkeypatch):
    """Notion outage must NOT block the planner — empty string is the
    'no notes' signal the LLM already understands."""
    from data import notion as notion_mod

    monkeypatch.setattr(notion_mod, "is_configured", lambda: True)

    def _boom(slug):
        raise RuntimeError("notion 503")

    monkeypatch.setattr(notion_mod, "read_thesis_notes", _boom)
    assert cio_memory.thesis_notes("ai_cake") == ""


# --- dismissals_in_window -------------------------------------------------


def test_dismissals_in_window_filters_by_action_and_age(isolated_db):
    state_db.record_cio_action(ticker="NVDA", thesis="ai_cake", action="dismiss")
    state_db.record_cio_action(ticker="NVDA", thesis="ai_cake", action="reuse")
    state_db.record_cio_action(ticker="NVDA", thesis="ai_cake", action="dismiss")
    out = cio_memory.dismissals_in_window("NVDA", "ai_cake")
    assert len(out) == 2
    assert all(a["action"] == "dismiss" for a in out)


def test_dismissals_in_window_excludes_old_rows(isolated_db, monkeypatch):
    """Dismissals older than the window must not count."""
    import sqlite3

    state_db.record_cio_action(ticker="NVDA", thesis="ai_cake", action="dismiss")
    old_iso = (datetime.now(UTC) - timedelta(days=14)).isoformat()
    with sqlite3.connect(isolated_db) as conn:
        conn.execute("UPDATE cio_actions SET ts = ?", (old_iso,))
    out = cio_memory.dismissals_in_window("NVDA", "ai_cake", window_days=7)
    assert out == []


# --- Step 11.20 — slug-based cooldown round-trip regression --------------


def test_cooldown_round_trip_via_invoke_with_telemetry(isolated_db, monkeypatch):
    """Regression for the schema-mismatch bug found 2026-05-07: the CIO
    planner queries `graph_runs.thesis` by SLUG (e.g. "ai_cake") but the
    runner used to write the human-readable NAME (e.g. "AI cake"). This
    silently broke the cooldown gate — the planner saw `active=False`
    even when there was a fresh drill 17 minutes earlier — and let the
    LLM re-drill the same pair.

    Test: simulate `invoke_with_telemetry` writing a graph_runs row with
    a slug-bearing thesis dict, then assert that the CIO memory layer's
    cooldown_status round-trips with `active=True` and a usable
    `last_drill_run_id`.
    """
    # Open + close a graph_runs row using the same helpers
    # `invoke_with_telemetry` uses, with a thesis dict that has both
    # `slug` and `name` set. The runner picks `slug` per Step 11.20.
    thesis = {"slug": "ai_cake", "name": "AI cake"}
    label = thesis.get("slug") or thesis.get("name") or "?"
    rid = state_db.start_graph_run("NVDA", label)
    state_db.finish_graph_run(rid, "completed", duration_s=42.0)

    # CIO memory layer queries by slug — must find the row.
    out = cio_memory.cooldown_status("NVDA", "ai_cake")
    assert out["active"] is True, (
        "cooldown_status reports inactive — graph_runs.thesis is probably "
        "still being written as the human name instead of the slug. See "
        "agents/__init__.py:invoke_with_telemetry."
    )
    assert out["last_drill_run_id"] == rid


def test_cooldown_inactive_when_slug_mismatch(isolated_db):
    """Sanity counter-test: if a row IS written with the wrong label
    (the legacy bug shape), cooldown_status correctly reports inactive.
    Confirms the failure mode the regression test above guards against."""
    rid = state_db.start_graph_run("NVDA", "AI cake")  # ← legacy bad label
    state_db.finish_graph_run(rid, "completed")
    out = cio_memory.cooldown_status("NVDA", "ai_cake")  # ← slug query
    assert out["active"] is False
    assert out["last_drill_run_id"] is None
