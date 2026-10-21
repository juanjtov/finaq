"""Step 5z unit tests — data/state.py SQLite telemetry layer.

Every test uses a tmp_path-scoped DB so the tests don't pollute
data_cache/state.db (and so they're isolated from each other).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from data import state as st

# --- Schema migration ------------------------------------------------------


def test_init_db_creates_all_tables(tmp_path: Path):
    db = tmp_path / "test.db"
    st.init_db(db)
    with sqlite3.connect(db) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    expected = {
        "meta", "graph_runs", "node_runs", "alerts", "triage_runs", "errors",
        "cio_runs", "cio_actions",  # Step 11.5
        "llm_calls",  # schema v6
    }
    assert expected.issubset(names), f"missing tables: {expected - names}"


def test_init_db_is_idempotent(tmp_path: Path):
    """Calling init_db twice must not raise and must not duplicate rows."""
    db = tmp_path / "test.db"
    st.init_db(db)
    st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.init_db(db)  # second call must not wipe data
    runs = st.recent_runs(limit=10, db_path=db)
    assert len(runs) == 1


def test_init_db_writes_schema_version(tmp_path: Path):
    db = tmp_path / "test.db"
    st.init_db(db)
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert row is not None
    assert int(row[0]) == st.SCHEMA_VERSION


def test_init_db_creates_parent_dir(tmp_path: Path):
    """If the data_cache/ dir doesn't exist, init_db creates it."""
    db = tmp_path / "missing" / "nested" / "test.db"
    st.init_db(db)
    assert db.exists()


# --- graph_runs lifecycle --------------------------------------------------


def test_start_and_finish_graph_run_round_trip(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    row = st.get_graph_run(run_id, db_path=db)
    assert row is not None
    assert row["ticker"] == "NVDA"
    assert row["status"] == "running"
    assert row["started_at"] is not None
    assert row["ended_at"] is None

    st.finish_graph_run(run_id, "completed", confidence="medium", duration_s=12.5, db_path=db)
    row = st.get_graph_run(run_id, db_path=db)
    assert row["status"] == "completed"
    assert row["confidence"] == "medium"
    assert row["duration_s"] == pytest.approx(12.5)
    assert row["ended_at"] is not None


def test_finish_graph_run_records_failure(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.finish_graph_run(run_id, "failed", error="LLM outage", db_path=db)
    row = st.get_graph_run(run_id, db_path=db)
    assert row["status"] == "failed"
    assert row["error"] == "LLM outage"


def test_finish_graph_run_rejects_unknown_status(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    with pytest.raises(ValueError, match="status must be"):
        st.finish_graph_run(run_id, "weird", db_path=db)


# --- node_runs --------------------------------------------------------------


def test_record_node_run_round_trip(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    nid = st.record_node_run(
        run_id,
        "fundamentals",
        st._now_iso(),
        st._now_iso(),
        2.4,
        "completed",
        db_path=db,
    )
    assert nid > 0
    rows = st.all_node_runs_for(run_id, db_path=db)
    assert len(rows) == 1
    assert rows[0]["node"] == "fundamentals"
    assert rows[0]["duration_s"] == pytest.approx(2.4)
    # Step 10c.8 added telemetry columns. Defaults must be 0 so legacy
    # callers (and existing tests) don't have to know about them.
    assert rows[0]["tokens_in"] == 0
    assert rows[0]["tokens_out"] == 0
    assert rows[0]["cost_usd"] == pytest.approx(0.0)
    assert rows[0]["n_calls"] == 0


def test_record_node_run_persists_telemetry_fields(tmp_path: Path):
    """Step 10c.8 — `_safe_node` writes per-node tokens + cost from the
    ContextVar accumulator. The DB must round-trip those values."""
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.record_node_run(
        run_id,
        "filings",
        st._now_iso(),
        st._now_iso(),
        17.5,
        "completed",
        tokens_in=1234,
        tokens_out=567,
        cost_usd=0.0234,
        n_calls=3,
        db_path=db,
    )
    rows = st.all_node_runs_for(run_id, db_path=db)
    assert rows[0]["tokens_in"] == 1234
    assert rows[0]["tokens_out"] == 567
    assert rows[0]["cost_usd"] == pytest.approx(0.0234)
    assert rows[0]["n_calls"] == 3


def test_init_db_migrates_v1_to_v2(tmp_path: Path):
    """A DB created before Step 10c.8 has no tokens_in/tokens_out/cost_usd/
    n_calls columns. The forward-only migration in `init_db` must add
    them WITHOUT dropping existing rows. Simulates an upgrade by
    creating a v1-shaped table manually, then calling init_db."""
    import sqlite3

    db = tmp_path / "legacy.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    # Create a v1 node_runs table with the exact pre-migration schema.
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE node_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            node TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            duration_s REAL,
            status TEXT NOT NULL,
            error TEXT
        );
        INSERT INTO node_runs (run_id, node, started_at, status)
        VALUES ('legacy-run', 'fundamentals', '2026-04-01T00:00:00Z', 'completed');
        """
    )
    conn.commit()
    conn.close()
    # Run the migration.
    st.init_db(db_path=db)
    # Existing row preserved + new columns default to 0.
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("SELECT * FROM node_runs"))
    assert len(rows) == 1
    assert rows[0]["node"] == "fundamentals"
    assert rows[0]["tokens_in"] == 0
    assert rows[0]["cost_usd"] == 0.0
    conn.close()
    # New inserts populate cleanly with the new fields.
    st.record_node_run(
        "legacy-run", "filings", "2026-04-01T00:00:01Z", "2026-04-01T00:00:02Z",
        1.0, "completed", tokens_in=100, tokens_out=50, cost_usd=0.0015,
        n_calls=1, db_path=db,
    )
    rows = st.all_node_runs_for("legacy-run", db_path=db)
    by_node = {r["node"]: r for r in rows}
    assert by_node["filings"]["tokens_in"] == 100
    assert by_node["fundamentals"]["tokens_in"] == 0  # legacy row untouched


def test_node_telemetry_var_default_is_none():
    """ContextVar default must be None so direct ad-hoc LLM calls (outside
    the graph) don't accumulate ghost telemetry rows."""
    assert st.node_telemetry_var.get() is None


def test_new_node_telemetry_returns_fresh_dict():
    """Each node entry binds a fresh accumulator — repeated calls must
    return distinct dicts (not a shared singleton)."""
    a = st.new_node_telemetry()
    b = st.new_node_telemetry()
    assert a is not b
    assert a == {
        "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "n_calls": 0, "node": "",
    }
    # Schema v6 — the accumulator carries the node name so llm_calls rows
    # written mid-node can be attributed to the right agent.
    assert st.new_node_telemetry(node="filings")["node"] == "filings"


def test_daily_cost_aggregates_node_runs(tmp_path: Path):
    """`daily_cost(days=N)` must group by date and sum cost_usd /
    tokens_in / tokens_out / n_calls per day. Used by /status and the
    Mission Control cost chart."""
    from datetime import UTC, datetime, timedelta

    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    today = datetime.now(UTC).date().isoformat()
    yesterday = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    # Today: two LLM calls totalling $0.05.
    st.record_node_run(
        run_id, "fundamentals", f"{today}T10:00:00+00:00", f"{today}T10:00:01+00:00",
        1.0, "completed", tokens_in=100, tokens_out=50, cost_usd=0.02, n_calls=1,
        db_path=db,
    )
    st.record_node_run(
        run_id, "synthesis", f"{today}T10:01:00+00:00", f"{today}T10:01:05+00:00",
        5.0, "completed", tokens_in=2000, tokens_out=500, cost_usd=0.03, n_calls=1,
        db_path=db,
    )
    # Yesterday: one call.
    st.record_node_run(
        run_id, "filings", f"{yesterday}T10:00:00+00:00", f"{yesterday}T10:00:02+00:00",
        2.0, "completed", tokens_in=300, tokens_out=100, cost_usd=0.01, n_calls=1,
        db_path=db,
    )
    rolled = st.daily_cost(days=7, db_path=db)
    by_date = {r["date"]: r for r in rolled}
    assert today in by_date and yesterday in by_date
    assert by_date[today]["cost_usd"] == pytest.approx(0.05)
    assert by_date[today]["n_calls"] == 2
    assert by_date[today]["tokens_in"] == 2100
    assert by_date[yesterday]["cost_usd"] == pytest.approx(0.01)


def test_cost_today_returns_zero_when_no_runs(tmp_path: Path):
    db = tmp_path / "empty.db"
    st.init_db(db_path=db)
    today = st.cost_today(db_path=db)
    assert today == {"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "n_calls": 0}


def test_node_runs_for_run_filters_by_id(tmp_path: Path):
    db = tmp_path / "test.db"
    a = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    b = st.start_graph_run("MSFT", "ai_cake", db_path=db)
    st.record_node_run(a, "fundamentals", st._now_iso(), st._now_iso(), 1.0, "completed", db_path=db)
    st.record_node_run(a, "filings", st._now_iso(), st._now_iso(), 2.0, "completed", db_path=db)
    st.record_node_run(b, "fundamentals", st._now_iso(), st._now_iso(), 1.5, "completed", db_path=db)
    a_nodes = st.node_runs_for_run(a, db_path=db)
    b_nodes = st.node_runs_for_run(b, db_path=db)
    assert len(a_nodes) == 2
    assert len(b_nodes) == 1
    assert {n["node"] for n in a_nodes} == {"fundamentals", "filings"}


def test_errors_for_run_filters_by_id(tmp_path: Path):
    db = tmp_path / "test.db"
    a = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    b = st.start_graph_run("MSFT", "ai_cake", db_path=db)
    st.record_error("risk", "validation failed", run_id=a, db_path=db)
    st.record_error("risk", "different error", run_id=b, db_path=db)
    a_errs = st.errors_for_run(a, db_path=db)
    assert len(a_errs) == 1
    assert "validation failed" in a_errs[0]["message"]


def test_record_node_run_with_failure(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.record_node_run(
        run_id,
        "filings",
        st._now_iso(),
        st._now_iso(),
        0.1,
        "failed",
        error="the filings index connection refused",
        db_path=db,
    )
    rows = st.all_node_runs_for(run_id, db_path=db)
    assert rows[0]["status"] == "failed"
    assert "the filings index" in rows[0]["error"]


def test_record_node_run_rejects_unknown_status(tmp_path: Path):
    db = tmp_path / "test.db"
    with pytest.raises(ValueError, match="status must be"):
        st.record_node_run(
            None,
            "fundamentals",
            st._now_iso(),
            st._now_iso(),
            1.0,
            "in_progress",
            db_path=db,
        )


def test_node_runs_attached_to_correct_run_id(tmp_path: Path):
    db = tmp_path / "test.db"
    a = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    b = st.start_graph_run("EME", "construction", db_path=db)
    for run_id, nodes in (
        (a, ["fundamentals", "filings", "news"]),
        (b, ["fundamentals", "risk"]),
    ):
        for n in nodes:
            st.record_node_run(
                run_id, n, st._now_iso(), st._now_iso(), 1.0, "completed", db_path=db
            )
    assert len(st.all_node_runs_for(a, db_path=db)) == 3
    assert len(st.all_node_runs_for(b, db_path=db)) == 2


# --- recent_runs -----------------------------------------------------------


def test_recent_runs_orders_most_recent_first(tmp_path: Path):
    db = tmp_path / "test.db"
    a = st.start_graph_run("AAA", "x", db_path=db)
    time.sleep(0.01)  # make sure timestamps differ at sub-millisecond resolution
    b = st.start_graph_run("BBB", "x", db_path=db)
    runs = st.recent_runs(limit=5, db_path=db)
    assert [r["run_id"] for r in runs[:2]] == [b, a]


def test_recent_runs_includes_node_count(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    for n in ("fundamentals", "filings", "news"):
        st.record_node_run(
            run_id, n, st._now_iso(), st._now_iso(), 1.0, "completed", db_path=db
        )
    runs = st.recent_runs(limit=5, db_path=db)
    assert runs[0]["node_runs_count"] == 3
    assert runs[0]["failed_nodes"] == 0


def test_recent_runs_counts_failed_nodes(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.record_node_run(
        run_id, "fundamentals", st._now_iso(), st._now_iso(), 1.0, "completed", db_path=db
    )
    st.record_node_run(
        run_id, "filings", st._now_iso(), st._now_iso(), 1.0, "failed",
        error="boom", db_path=db,
    )
    runs = st.recent_runs(limit=5, db_path=db)
    assert runs[0]["failed_nodes"] == 1


def test_recent_runs_returns_empty_when_db_missing(tmp_path: Path):
    """If state.db never existed (first-run scenario), recent_runs returns
    an empty list rather than raising."""
    db = tmp_path / "never_created.db"
    assert st.recent_runs(db_path=db) == []


# --- alerts -----------------------------------------------------------------


def test_record_alert_and_update_status(tmp_path: Path):
    db = tmp_path / "test.db"
    aid = st.record_alert(
        "NVDA", "ai_cake", 4, "supply concentration",
        evidence_url="https://sec.gov/...", db_path=db,
    )
    assert aid > 0
    rows = st.recent_alerts(db_path=db)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"

    st.update_alert_status(aid, "actioned", db_path=db)
    rows = st.recent_alerts(db_path=db)
    assert rows[0]["status"] == "actioned"


def test_recent_alerts_filters_by_status(tmp_path: Path):
    db = tmp_path / "test.db"
    st.record_alert("AAA", "x", 3, "sig1", db_path=db)
    aid = st.record_alert("BBB", "x", 4, "sig2", db_path=db)
    st.update_alert_status(aid, "dismissed", db_path=db)
    pending = st.recent_alerts(status="pending", db_path=db)
    dismissed = st.recent_alerts(status="dismissed", db_path=db)
    assert len(pending) == 1
    assert len(dismissed) == 1
    assert pending[0]["ticker"] == "AAA"
    assert dismissed[0]["ticker"] == "BBB"


def test_update_alert_status_rejects_unknown(tmp_path: Path):
    db = tmp_path / "test.db"
    aid = st.record_alert("X", "y", 1, "s", db_path=db)
    with pytest.raises(ValueError, match="unknown alert status"):
        st.update_alert_status(aid, "approved", db_path=db)


# --- triage_runs -----------------------------------------------------------


def test_record_triage_run_round_trip(tmp_path: Path):
    db = tmp_path / "test.db"
    rid = st.record_triage_run(123, 2, 45.0, db_path=db)
    assert rid > 0
    rows = st.recent_triage_runs(db_path=db)
    assert len(rows) == 1
    assert rows[0]["items_scanned"] == 123
    assert rows[0]["alerts_emitted"] == 2
    assert rows[0]["duration_s"] == pytest.approx(45.0)


# --- errors -----------------------------------------------------------------


def test_record_error_round_trip(tmp_path: Path):
    db = tmp_path / "test.db"
    eid = st.record_error("filings", "the filings index unreachable", db_path=db)
    assert eid > 0
    rows = st.recent_errors(db_path=db)
    assert len(rows) == 1
    assert rows[0]["agent"] == "filings"
    assert "the filings index" in rows[0]["message"]


def test_record_error_with_run_id_links(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.record_error("filings", "boom", run_id=run_id, db_path=db)
    rows = st.recent_errors(db_path=db)
    assert rows[0]["run_id"] == run_id


# --- daily_run_counts + health_summary -------------------------------------


def test_daily_run_counts_aggregates_by_day(tmp_path: Path):
    db = tmp_path / "test.db"
    # 3 runs on the current day (test machine clock — just rely on
    # SQLite's date('now') to bucket them).
    for ticker in ("AAA", "BBB", "CCC"):
        run_id = st.start_graph_run(ticker, "x", db_path=db)
        st.finish_graph_run(run_id, "completed", db_path=db)
    one_failed = st.start_graph_run("DDD", "x", db_path=db)
    st.finish_graph_run(one_failed, "failed", error="boom", db_path=db)

    rows = st.daily_run_counts(days=2, db_path=db)
    assert len(rows) >= 1
    today = rows[-1]
    assert today["total"] == 4
    assert today["completed"] == 3
    assert today["failed"] == 1


def test_health_summary_empty_db(tmp_path: Path):
    db = tmp_path / "never.db"
    out = st.health_summary(db_path=db)
    assert out == {"total_runs": 0, "last_run_at": None, "failure_rate_7d": None}


def test_health_summary_with_runs(tmp_path: Path):
    db = tmp_path / "test.db"
    a = st.start_graph_run("AAA", "x", db_path=db)
    st.finish_graph_run(a, "completed", db_path=db)
    b = st.start_graph_run("BBB", "x", db_path=db)
    st.finish_graph_run(b, "failed", error="x", db_path=db)
    summary = st.health_summary(db_path=db)
    assert summary["total_runs"] == 2
    assert summary["last_run_at"] is not None
    assert summary["failure_rate_7d"] == 0.5  # 1 of 2 failed


# --- Contextvar -------------------------------------------------------------


def test_current_run_id_contextvar_starts_none():
    """ContextVar default → None when no graph run is active."""
    assert st.current_run_id.get() is None


def test_current_run_id_contextvar_set_and_get():
    token = st.current_run_id.set("test-run-123")
    try:
        assert st.current_run_id.get() == "test-run-123"
    finally:
        st.current_run_id.reset(token)
    assert st.current_run_id.get() is None


# --- CIO cycle + action recording (Step 11.5) -----------------------------


def test_start_cio_run_returns_uuid_and_inserts_running_row(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_cio_run("heartbeat", db_path=db)
    assert run_id and len(run_id) == 36  # uuid4 string

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT trigger, status, ended_at FROM cio_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "heartbeat"
    assert row[1] == "running"
    assert row[2] is None  # not yet finished


def test_start_cio_run_rejects_unknown_trigger(tmp_path: Path):
    with pytest.raises(ValueError, match="trigger"):
        st.start_cio_run("scheduled", db_path=tmp_path / "test.db")


def test_finish_cio_run_updates_counts_and_summary(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_cio_run("on_demand", db_path=db)
    st.finish_cio_run(
        run_id, "completed",
        duration_s=12.4, n_actions=4, n_drilled=1, n_reused=2, n_dismissed=1,
        summary="Reused MSFT, drilled NVDA, dismissed AAPL+GOOGL.",
        db_path=db,
    )
    runs = st.recent_cio_runs(db_path=db)
    assert len(runs) == 1
    r = runs[0]
    assert r["status"] == "completed"
    assert r["n_actions"] == 4 and r["n_drilled"] == 1
    assert r["n_reused"] == 2 and r["n_dismissed"] == 1
    assert r["duration_s"] == 12.4
    assert "MSFT" in (r["summary"] or "")


def test_finish_cio_run_rejects_unknown_status(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_cio_run("heartbeat", db_path=db)
    with pytest.raises(ValueError, match="status"):
        st.finish_cio_run(run_id, "in_progress", db_path=db)


def test_record_cio_action_round_trip_with_drill(tmp_path: Path):
    db = tmp_path / "test.db"
    cio_run_id = st.start_cio_run("heartbeat", db_path=db)
    drill_run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)

    aid = st.record_cio_action(
        cio_run_id=cio_run_id,
        ticker="NVDA",
        thesis="ai_cake",
        action="drill",
        rationale="Latest 10-Q lands tomorrow; refresh now",
        drill_run_id=drill_run_id,
        confidence="high",
        decision_json='{"action":"drill","ticker":"NVDA"}',
        db_path=db,
    )
    assert aid > 0

    actions = st.recent_cio_actions(db_path=db)
    assert len(actions) == 1
    a = actions[0]
    assert a["ticker"] == "NVDA"
    assert a["action"] == "drill"
    assert a["confidence"] == "high"
    assert a["drill_run_id"] == drill_run_id
    assert a["cio_run_id"] == cio_run_id


def test_record_cio_action_rejects_unknown_action(tmp_path: Path):
    db = tmp_path / "test.db"
    with pytest.raises(ValueError, match="action"):
        st.record_cio_action(
            ticker="NVDA", action="postpone",  # not allowed
            db_path=db,
        )


def test_record_cio_action_rejects_unknown_confidence(tmp_path: Path):
    db = tmp_path / "test.db"
    with pytest.raises(ValueError, match="confidence"):
        st.record_cio_action(
            ticker="NVDA", action="dismiss", confidence="strong",  # not allowed
            db_path=db,
        )


def test_recent_cio_actions_filters_by_ticker_and_thesis(tmp_path: Path):
    db = tmp_path / "test.db"
    for ticker, thesis in (
        ("NVDA", "ai_cake"),
        ("MSFT", "ai_cake"),
        ("NVDA", "nvda_halo"),
    ):
        st.record_cio_action(
            ticker=ticker, thesis=thesis, action="reuse", db_path=db,
        )

    only_nvda = st.recent_cio_actions(ticker="NVDA", db_path=db)
    assert {a["thesis"] for a in only_nvda} == {"ai_cake", "nvda_halo"}

    only_pair = st.recent_cio_actions(ticker="NVDA", thesis="ai_cake", db_path=db)
    assert len(only_pair) == 1
    assert only_pair[0]["thesis"] == "ai_cake"


def test_last_drill_for_returns_most_recent_completed(tmp_path: Path):
    db = tmp_path / "test.db"
    # Two completed graph_runs for NVDA on ai_cake — older first.
    rid_old = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.finish_graph_run(rid_old, "completed", db_path=db)
    time.sleep(0.01)  # ensures distinct timestamps
    rid_new = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.finish_graph_run(rid_new, "completed", db_path=db)

    # And one failed run that must NOT be returned.
    rid_failed = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.finish_graph_run(rid_failed, "failed", db_path=db)

    last = st.last_drill_for("NVDA", "ai_cake", db_path=db)
    assert last is not None
    assert last["run_id"] == rid_new
    assert last["status"] == "completed"


def test_last_drill_for_thesis_optional(tmp_path: Path):
    """Without a thesis arg, returns the most recent completed run for
    the ticker across all theses."""
    db = tmp_path / "test.db"
    rid = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.finish_graph_run(rid, "completed", db_path=db)

    last = st.last_drill_for("NVDA", db_path=db)
    assert last is not None and last["run_id"] == rid


def test_last_drill_for_returns_none_when_no_run(tmp_path: Path):
    db = tmp_path / "test.db"
    st.init_db(db)
    assert st.last_drill_for("NVDA", "ai_cake", db_path=db) is None


def test_last_successful_cio_run_at_returns_iso_or_none(tmp_path: Path):
    db = tmp_path / "test.db"
    st.init_db(db)
    assert st.last_successful_cio_run_at(db_path=db) is None

    cio_run_id = st.start_cio_run("heartbeat", db_path=db)
    # Still 'running' → not counted.
    assert st.last_successful_cio_run_at(db_path=db) is None

    st.finish_cio_run(cio_run_id, "completed", db_path=db)
    ts = st.last_successful_cio_run_at(db_path=db)
    assert isinstance(ts, str)
    assert ts.startswith("20")  # ISO-like


def test_recent_cio_runs_orders_most_recent_first(tmp_path: Path):
    db = tmp_path / "test.db"
    rids: list[str] = []
    for trigger in ("heartbeat", "on_demand", "catchup"):
        rids.append(st.start_cio_run(trigger, db_path=db))
        time.sleep(0.01)
    runs = st.recent_cio_runs(limit=10, db_path=db)
    assert [r["run_id"] for r in runs] == list(reversed(rids))


def test_recent_cio_runs_empty_when_db_missing(tmp_path: Path):
    """No DB on disk → empty list, no exception. Mirrors recent_runs."""
    assert st.recent_cio_runs(db_path=tmp_path / "nope.db") == []


def test_recent_cio_actions_empty_when_db_missing(tmp_path: Path):
    assert st.recent_cio_actions(db_path=tmp_path / "nope.db") == []


# --- Step 11.19 schema migration + per-model performance rollup ----------


def test_init_db_adds_step_11_19_columns_to_cio_actions(tmp_path: Path):
    """Forward-only migration: a fresh init_db must add the per-call
    telemetry columns (model_used, tokens_in/out, cost_usd, latency_s)
    to cio_actions."""
    db = tmp_path / "test.db"
    st.init_db(db)
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(cio_actions)")}
    for required in {"model_used", "tokens_in", "tokens_out", "cost_usd", "latency_s"}:
        assert required in cols, f"cio_actions missing column {required!r}"


def test_init_db_adds_step_11_19_columns_to_cio_runs(tmp_path: Path):
    db = tmp_path / "test.db"
    st.init_db(db)
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(cio_runs)")}
    for required in {"model_used", "total_cost_usd"}:
        assert required in cols, f"cio_runs missing column {required!r}"


def test_init_db_v3_to_v4_migration_preserves_existing_rows(tmp_path: Path):
    """Simulate a pre-Step-11.19 database: drop the new columns and
    re-run init_db. Existing rows must survive the ALTER TABLE additions."""
    db = tmp_path / "test.db"
    st.init_db(db)
    rid = st.start_cio_run("heartbeat", db_path=db)
    st.record_cio_action(ticker="NVDA", thesis="ai_cake", action="drill",
                         cio_run_id=rid, db_path=db)

    # Pretend the new cols never existed: delete them via a copy table.
    # SQLite doesn't support DROP COLUMN until 3.35; we use the table
    # copy idiom that mirrors how a real legacy DB would look.
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE cio_actions_v3 AS
            SELECT id, cio_run_id, ts, trigger, ticker, thesis, action,
                   rationale, drill_run_id, reuse_run_id, confidence, decision_json
              FROM cio_actions;
            DROP TABLE cio_actions;
            ALTER TABLE cio_actions_v3 RENAME TO cio_actions;

            CREATE TABLE cio_runs_v3 AS
            SELECT run_id, trigger, started_at, ended_at, duration_s, status,
                   error, n_actions, n_drilled, n_reused, n_dismissed, summary
              FROM cio_runs;
            DROP TABLE cio_runs;
            ALTER TABLE cio_runs_v3 RENAME TO cio_runs;
            """
        )

    # Re-run init_db: the v4 migration must add the new columns and keep data.
    st.init_db(db)
    with sqlite3.connect(db) as conn:
        action_cols = {row[1] for row in conn.execute("PRAGMA table_info(cio_actions)")}
        run_cols = {row[1] for row in conn.execute("PRAGMA table_info(cio_runs)")}
        action_count = conn.execute("SELECT COUNT(*) FROM cio_actions").fetchone()[0]

    assert "model_used" in action_cols
    assert "cost_usd" in action_cols
    assert "total_cost_usd" in run_cols
    assert action_count == 1, "pre-existing row was lost during migration"


def test_cio_model_performance_groups_by_model(tmp_path: Path):
    """Multiple models in cio_actions roll up to one row per model,
    with parse_fail counted from the rationale prefix."""
    db = tmp_path / "test.db"

    # Three calls on gpt-mini (one parse-fail), two on haiku (clean).
    st.record_cio_action(
        ticker="A", action="drill", model_used="openai/gpt-5.4-mini",
        tokens_in=1000, tokens_out=50, cost_usd=0.001, latency_s=1.0,
        db_path=db,
    )
    st.record_cio_action(
        ticker="B", action="dismiss", model_used="openai/gpt-5.4-mini",
        tokens_in=2000, tokens_out=80, cost_usd=0.002, latency_s=1.5,
        db_path=db,
    )
    st.record_cio_action(
        ticker="C", action="dismiss",
        rationale="LLM response unparseable: empty",
        model_used="openai/gpt-5.4-mini",
        tokens_in=1500, tokens_out=0, cost_usd=0.001, latency_s=2.0,
        db_path=db,
    )
    st.record_cio_action(
        ticker="D", action="reuse", model_used="anthropic/claude-haiku-4.5",
        tokens_in=900, tokens_out=70, cost_usd=0.0009, latency_s=0.8,
        db_path=db,
    )
    st.record_cio_action(
        ticker="E", action="dismiss", model_used="anthropic/claude-haiku-4.5",
        tokens_in=950, tokens_out=60, cost_usd=0.001, latency_s=0.9,
        db_path=db,
    )

    perf = st.cio_model_performance(days=30, db_path=db)
    by_model = {p["model_used"]: p for p in perf}

    assert "openai/gpt-5.4-mini" in by_model
    assert "anthropic/claude-haiku-4.5" in by_model

    gpt = by_model["openai/gpt-5.4-mini"]
    assert gpt["n_calls"] == 3
    assert gpt["n_drills"] == 1
    assert gpt["n_dismisses"] == 2
    assert gpt["n_parse_fails"] == 1
    assert gpt["total_cost_usd"] == pytest.approx(0.004)

    hk = by_model["anthropic/claude-haiku-4.5"]
    assert hk["n_calls"] == 2
    assert hk["n_reuses"] == 1
    assert hk["n_parse_fails"] == 0


def test_cio_model_performance_excludes_old_rows(tmp_path: Path):
    """Rows older than `days` must not appear in the rollup."""
    db = tmp_path / "test.db"
    st.record_cio_action(
        ticker="A", action="dismiss", model_used="openai/gpt-5.4-mini",
        cost_usd=0.001, db_path=db,
    )
    # Back-date that row to 100 days ago.
    with sqlite3.connect(db) as conn:
        old_iso = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        conn.execute("UPDATE cio_actions SET ts = ?", (old_iso,))

    perf = st.cio_model_performance(days=30, db_path=db)
    assert perf == [], "old rows must be filtered by the days window"


def test_cio_model_performance_empty_when_db_missing(tmp_path: Path):
    assert st.cio_model_performance(db_path=tmp_path / "nope.db") == []


def test_finish_cio_run_persists_model_and_total_cost(tmp_path: Path):
    """Step 11.19 — `finish_cio_run` round-trips the new fields."""
    db = tmp_path / "test.db"
    rid = st.start_cio_run("heartbeat", db_path=db)
    st.finish_cio_run(
        rid, "completed",
        model_used="openai/gpt-5.4-mini",
        total_cost_usd=0.1312,
        n_actions=38, n_drilled=3, n_reused=0, n_dismissed=35,
        duration_s=390.0,
        db_path=db,
    )
    runs = st.recent_cio_runs(db_path=db)
    assert runs[0]["model_used"] == "openai/gpt-5.4-mini"
    assert runs[0]["total_cost_usd"] == pytest.approx(0.1312)


def test_finish_cio_run_defaults_zero_cost_when_unset(tmp_path: Path):
    """Backwards-compat: callers that don't pass model/cost still work."""
    db = tmp_path / "test.db"
    rid = st.start_cio_run("heartbeat", db_path=db)
    st.finish_cio_run(rid, "completed", db_path=db)  # no kwargs
    runs = st.recent_cio_runs(db_path=db)
    assert runs[0]["model_used"] is None
    assert runs[0]["total_cost_usd"] == 0.0


def test_record_cio_action_persists_telemetry_columns(tmp_path: Path):
    db = tmp_path / "test.db"
    aid = st.record_cio_action(
        ticker="NVDA", thesis="ai_cake", action="drill",
        model_used="openai/gpt-5.4-mini",
        tokens_in=2500, tokens_out=80,
        cost_usd=0.0025, latency_s=1.5,
        db_path=db,
    )
    assert aid > 0
    actions = st.recent_cio_actions(db_path=db)
    a = actions[0]
    assert a["model_used"] == "openai/gpt-5.4-mini"
    assert a["tokens_in"] == 2500
    assert a["tokens_out"] == 80
    assert a["cost_usd"] == pytest.approx(0.0025)
    assert a["latency_s"] == pytest.approx(1.5)


# --- Schema v5 — cio_actions.source ---------------------------------------


def test_init_db_adds_source_column_to_cio_actions(tmp_path: Path):
    """Schema v5: `source` (llm | gate | budget_cap | fallback), '' default."""
    db = tmp_path / "test.db"
    st.init_db(db)
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(cio_actions)")}
    assert "source" in cols


def test_init_db_v4_to_v5_migration_backfills_source_as_empty(tmp_path: Path):
    """A pre-v5 DB gains the column; legacy rows read back as source=''
    so the yo-yo guard never mistakes them for LLM judgements."""
    db = tmp_path / "test.db"
    st.init_db(db)
    st.record_cio_action(ticker="NVDA", thesis="ai_cake", action="dismiss",
                         source="llm", db_path=db)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE cio_actions_legacy AS
                SELECT id, cio_run_id, ts, trigger, ticker, thesis, action, rationale,
                       drill_run_id, reuse_run_id, confidence, decision_json,
                       model_used, tokens_in, tokens_out, cost_usd, latency_s
                FROM cio_actions;
            DROP TABLE cio_actions;
            ALTER TABLE cio_actions_legacy RENAME TO cio_actions;
            """
        )
    st.init_db(db)
    rows = st.recent_cio_actions(db_path=db)
    assert len(rows) == 1
    assert rows[0]["source"] == ""


def test_record_cio_action_persists_source(tmp_path: Path):
    db = tmp_path / "test.db"
    for src in ("llm", "gate", "budget_cap", "fallback"):
        st.record_cio_action(ticker="NVDA", thesis="ai_cake", action="dismiss",
                             source=src, db_path=db)
    st.record_cio_action(ticker="NVDA", thesis="ai_cake", action="dismiss", db_path=db)
    sources = sorted(a["source"] for a in st.recent_cio_actions(db_path=db))
    assert sources == ["", "budget_cap", "fallback", "gate", "llm"]


# --- Schema v6 — llm_calls trace rows + run rollups ------------------------


def test_record_llm_call_and_query_roundtrip(tmp_path: Path):
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    st.record_llm_call(
        run_id=run_id, node="fundamentals", model="test/model-a",
        latency_s=1.5, tokens_in=1000, tokens_out=200, cost_usd=0.003,
        prompt_excerpt='[{"role": "user", "content": "hi"}]',
        response_excerpt="hello",
        db_path=db,
    )
    st.record_llm_call(
        run_id=run_id, node="synthesis", model="test/model-b",
        latency_s=12.0, tokens_in=40000, tokens_out=5000, cost_usd=0.16,
        db_path=db,
    )
    calls = st.llm_calls_for_run(run_id, db_path=db)
    assert len(calls) == 2
    # Call order preserved (by autoincrement id).
    assert calls[0]["node"] == "fundamentals"
    assert calls[1]["node"] == "synthesis"
    assert calls[0]["model"] == "test/model-a"
    assert calls[0]["tokens_in"] == 1000
    assert calls[0]["prompt_excerpt"].startswith('[{"role"')
    assert calls[0]["response_excerpt"] == "hello"
    # Missing excerpts stored as NULL, not empty string.
    assert calls[1]["prompt_excerpt"] is None


def test_record_llm_call_truncates_excerpts(tmp_path: Path):
    db = tmp_path / "test.db"
    long_text = "x" * (st.LLM_EXCERPT_MAX_CHARS + 500)
    st.record_llm_call(
        run_id=None, node="", model="m", latency_s=0.1,
        prompt_excerpt=long_text, response_excerpt=long_text,
        db_path=db,
    )
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT prompt_excerpt, response_excerpt, run_id FROM llm_calls"
        ).fetchone()
    assert len(row[0]) == st.LLM_EXCERPT_MAX_CHARS
    assert len(row[1]) == st.LLM_EXCERPT_MAX_CHARS
    # run_id None is allowed — calls outside a graph run (CIO planner, Q&A).
    assert row[2] is None


def test_llm_calls_for_run_empty_when_db_missing(tmp_path: Path):
    assert st.llm_calls_for_run("nope", db_path=tmp_path / "missing.db") == []


def test_init_db_v5_to_v6_adds_llm_calls_table(tmp_path: Path):
    """A DB created before v6 (no llm_calls) must gain the table on the
    next init_db, without touching existing rows."""
    db = tmp_path / "test.db"
    st.init_db(db)
    run_id = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE llm_calls")
    st.init_db(db)  # re-migrate
    with sqlite3.connect(db) as conn:
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "llm_calls" in names
    # Pre-existing rows untouched.
    assert st.get_graph_run(run_id, db_path=db) is not None


def test_recent_runs_rolls_up_cost_tokens_and_failed_names(tmp_path: Path):
    """Mission Control's runs table renders cost / tokens / failed-agent
    names straight from recent_runs — the rollup must be correct."""
    db = tmp_path / "test.db"
    run_id = st.start_graph_run("WEN", "wen", db_path=db)
    now = st._now_iso()
    st.record_node_run(
        run_id, "fundamentals", now, now, 46.0, "completed",
        tokens_in=21000, tokens_out=2400, cost_usd=0.06, n_calls=4, db_path=db,
    )
    st.record_node_run(
        run_id, "filings", now, now, 122.0, "failed",
        error="chroma timeout", tokens_in=18000, tokens_out=1100,
        cost_usd=0.04, n_calls=2, db_path=db,
    )
    st.record_node_run(
        run_id, "news", now, now, 93.0, "failed",
        error="finnhub 429", db_path=db,
    )
    st.record_error("filings", "chroma timeout", run_id=run_id, db_path=db)
    st.finish_graph_run(run_id, "failed", duration_s=214.0, db_path=db)

    runs = st.recent_runs(db_path=db)
    assert len(runs) == 1
    r = runs[0]
    assert r["node_runs_count"] == 3
    assert r["failed_nodes"] == 2
    assert set((r["failed_node_names"] or "").split(",")) == {"filings", "news"}
    assert r["tokens_in"] == 39000
    assert r["tokens_out"] == 3500
    assert abs(r["cost_usd"] - 0.10) < 1e-9
    assert r["n_calls"] == 6
    assert r["n_errors"] == 1


def test_recent_runs_rollups_zero_for_run_without_nodes(tmp_path: Path):
    db = tmp_path / "test.db"
    st.start_graph_run("NU", "nu", db_path=db)
    r = st.recent_runs(db_path=db)[0]
    assert r["node_runs_count"] == 0
    assert (r["failed_node_names"] or None) is None
    assert r["n_errors"] == 0


def test_cio_action_for_run_finds_drill_and_reuse(tmp_path: Path):
    db = tmp_path / "test.db"
    cycle = st.start_cio_run("heartbeat", db_path=db)
    drill_run = st.start_graph_run("NU", "nu", db_path=db)
    reused_run = st.start_graph_run("EME", "construction", db_path=db)
    st.record_cio_action(
        ticker="NU", thesis="nu", action="drill", cio_run_id=cycle,
        drill_run_id=drill_run, confidence="high", db_path=db,
    )
    st.record_cio_action(
        ticker="EME", thesis="construction", action="reuse", cio_run_id=cycle,
        reuse_run_id=reused_run, confidence="medium", db_path=db,
    )
    a = st.cio_action_for_run(drill_run, db_path=db)
    assert a is not None
    assert a["action"] == "drill"
    assert a["cycle_trigger"] == "heartbeat"
    assert a["cycle_started_at"]  # joined from the parent cycle
    b = st.cio_action_for_run(reused_run, db_path=db)
    assert b is not None and b["action"] == "reuse"


def test_cio_action_for_run_none_for_manual_runs(tmp_path: Path):
    db = tmp_path / "test.db"
    manual_run = st.start_graph_run("NVDA", "ai_cake", db_path=db)
    assert st.cio_action_for_run(manual_run, db_path=db) is None
    assert st.cio_action_for_run("", db_path=db) is None


def test_llm_calls_for_run_tolerates_pre_v6_db(tmp_path: Path):
    """A DB created before schema v6 has no llm_calls table. A read must
    return [] (not raise, not migrate) — the live dashboard reads DBs that
    only get migrated on their next write. Regression for the Run
    Inspector crashing with 'no such table: llm_calls'."""
    db = tmp_path / "old.db"
    st.init_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE llm_calls")
    assert st.llm_calls_for_run("any-run", db_path=db) == []
    # And the missing table stays missing — reads never migrate.
    with sqlite3.connect(db) as conn:
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "llm_calls" not in names


# --- ingested_filings manifest (schema v7) -----------------------------------


def test_ingested_filings_manifest_round_trip(tmp_path: Path):
    """`record_ingested_filing` upserts per (ticker, accession); reads come
    back newest filed_date first; the chunk probe distinguishes 'never
    ingested' (None) from 'ingested with N chunks'."""
    db = tmp_path / "test.db"
    assert st.ingested_filings("NKE", db_path=db) == []
    assert st.ingested_filing_chunks("NKE", "acc-1", db_path=db) is None

    st.record_ingested_filing(
        ticker="nke", accession="acc-1", filing_type="10-K",
        filed_date="2025-07-24", chunks=310, db_path=db,
    )
    st.record_ingested_filing(
        ticker="NKE", accession="acc-2", filing_type="10-Q",
        filed_date="2025-10-01", chunks=120, db_path=db,
    )
    # Re-ingest of the same accession replaces the row (new chunk count).
    st.record_ingested_filing(
        ticker="NKE", accession="acc-1", filing_type="10-K",
        filed_date="2025-07-24", chunks=305, db_path=db,
    )

    rows = st.ingested_filings("nke", db_path=db)
    assert [(r["accession"], r["chunks"]) for r in rows] == [("acc-2", 120), ("acc-1", 305)]
    assert all(r["ticker"] == "NKE" and r["ingested_at"] for r in rows)
    assert st.ingested_filing_chunks("NKE", "acc-1", db_path=db) == 305


def test_ingest_manifest_summary(tmp_path: Path):
    db = tmp_path / "test.db"
    empty = st.ingest_manifest_summary(db_path=db)
    assert empty == {"tickers": 0, "filings": 0, "chunks": 0, "last_ingested_at": None}

    st.record_ingested_filing(
        ticker="NKE", accession="a", filing_type="10-K", filed_date="", chunks=10, db_path=db
    )
    st.record_ingested_filing(
        ticker="COUR", accession="b", filing_type="10-Q", filed_date="", chunks=5, db_path=db
    )
    summary = st.ingest_manifest_summary(db_path=db)
    assert (summary["tickers"], summary["filings"], summary["chunks"]) == (2, 2, 15)
    assert summary["last_ingested_at"]


def test_ingested_filings_reads_tolerate_pre_v7_db(tmp_path: Path):
    """Reads never migrate: a DB without the manifest table reads as
    "nothing ingested" instead of raising (the dashboard reads DBs that only
    get migrated on their next write)."""
    db = tmp_path / "old.db"
    st.init_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE ingested_filings")
    assert st.ingested_filings("NKE", db_path=db) == []
    assert st.ingested_filing_chunks("NKE", "acc", db_path=db) is None
    assert st.ingest_manifest_summary(db_path=db)["filings"] == 0
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "ingested_filings" not in names


def test_clear_ingested_filings_forgets_one_ticker(tmp_path: Path):
    db = tmp_path / "test.db"
    for ticker in ("NKE", "COUR"):
        st.record_ingested_filing(
            ticker=ticker, accession="a", filing_type="10-K", filed_date="", chunks=3, db_path=db
        )
    assert st.clear_ingested_filings("nke", db_path=db) == 1
    assert st.ingested_filings("NKE", db_path=db) == []
    assert len(st.ingested_filings("COUR", db_path=db)) == 1


def test_manifest_reads_do_not_create_a_missing_db(tmp_path: Path):
    """Mission Control's state.db panel treats "file exists" as "recording";
    a read-only manifest probe must not conjure an empty file."""
    db = tmp_path / "absent.db"
    assert st.ingested_filings("NKE", db_path=db) == []
    assert st.ingested_filing_chunks("NKE", "acc", db_path=db) is None
    assert st.ingest_manifest_summary(db_path=db)["tickers"] == 0
    assert not db.exists()


# --- Schema v9 — Mission Control v2 Phase 1 ---------------------------------
# Two-phase node + LLM rows, prompt blobs with retention, spans, the stale
# sweeper, medians and the live / run snapshots (docs/MISSION_CONTROL_PLAN.md §3).

V9_TABLES = {"prompt_blobs", "llm_messages", "spans"}


def _table_names(db: Path) -> set[str]:
    with sqlite3.connect(db) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _columns(db: Path, table: str) -> set[str]:
    with sqlite3.connect(db) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _one(db: Path, sql: str, *args) -> dict:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute(sql, args).fetchone())


def _hours_ago(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


def test_init_db_v9_creates_new_tables_and_columns(tmp_path: Path):
    db = tmp_path / "v9.db"
    st.init_db(db)
    assert st.SCHEMA_VERSION == 9
    assert V9_TABLES.issubset(_table_names(db))
    assert "pid" in _columns(db, "node_runs")
    assert "source" in _columns(db, "graph_runs")
    assert {"status", "ended_at", "error", "attempt"}.issubset(_columns(db, "llm_calls"))


def test_connect_applies_wal_and_busy_timeout(tmp_path: Path):
    db = tmp_path / "pragmas.db"
    st.init_db(db)
    conn = st._connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()
    # journal_mode is persistent on the file: a plain connection sees WAL too.
    with sqlite3.connect(db) as plain:
        assert plain.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


_V8_DDL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE graph_runs (
    run_id TEXT PRIMARY KEY, ticker TEXT, thesis TEXT, started_at TEXT NOT NULL,
    ended_at TEXT, duration_s REAL, status TEXT NOT NULL, error TEXT, confidence TEXT);
CREATE TABLE node_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, node TEXT NOT NULL,
    started_at TEXT NOT NULL, ended_at TEXT, duration_s REAL, status TEXT NOT NULL, error TEXT,
    tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0, n_calls INTEGER NOT NULL DEFAULT 0);
CREATE TABLE llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, node TEXT NOT NULL DEFAULT '',
    ts TEXT NOT NULL, model TEXT, latency_s REAL NOT NULL DEFAULT 0.0,
    tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0, prompt_excerpt TEXT, response_excerpt TEXT);
INSERT INTO meta VALUES ('schema_version', '8');
INSERT INTO graph_runs (run_id, ticker, thesis, started_at, status)
    VALUES ('r1', 'NVDA', 'ai_cake', '2026-10-01T00:00:00+00:00', 'completed');
INSERT INTO node_runs (run_id, node, started_at, ended_at, duration_s, status)
    VALUES ('r1', 'fundamentals', '2026-10-01T00:00:00+00:00', '2026-10-01T00:01:00+00:00',
            60.0, 'completed');
INSERT INTO llm_calls (run_id, node, ts, model, latency_s, response_excerpt)
    VALUES ('r1', 'fundamentals', '2026-10-01T00:01:00+00:00', 'm', 1.0, 'ok');
INSERT INTO llm_calls (run_id, node, ts, model, latency_s, response_excerpt)
    VALUES ('r1', 'synthesis', '2026-10-01T00:02:00+00:00', 'm', 1.0, 'CALL FAILED: upstream 500');
"""


def test_init_db_v8_to_v9_migration_keeps_rows_and_backfills(tmp_path: Path):
    """A hand-built v8 file (no v9 columns / tables) migrates in place: every
    row survives, new columns get their defaults, and the two backfills run —
    pre-v9 `CALL FAILED` rows become status=failed with the error, and
    `ended_at` = `ts` (the old rows were written after the response)."""
    db = tmp_path / "v8.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(_V8_DDL)
    st.init_db(db)
    assert V9_TABLES.issubset(_table_names(db))
    assert _one(db, "SELECT COUNT(*) AS n FROM graph_runs")["n"] == 1
    assert _one(db, "SELECT COUNT(*) AS n FROM node_runs")["n"] == 1
    assert _one(db, "SELECT COUNT(*) AS n FROM llm_calls")["n"] == 2
    assert _one(db, "SELECT * FROM graph_runs")["source"] == ""
    assert _one(db, "SELECT * FROM node_runs")["pid"] is None
    ok = _one(db, "SELECT * FROM llm_calls WHERE node = 'fundamentals'")
    assert (ok["status"], ok["attempt"], ok["error"]) == ("completed", 1, None)
    assert ok["ended_at"] == ok["ts"]
    failed = _one(db, "SELECT * FROM llm_calls WHERE node = 'synthesis'")
    assert (failed["status"], failed["error"]) == ("failed", "upstream 500")
    assert _one(db, "SELECT value FROM meta WHERE key = 'schema_version'")["value"] == "9"
    # Idempotent: a second init_db on the migrated file changes nothing.
    st.init_db(db)
    assert _one(db, "SELECT COUNT(*) AS n FROM llm_calls")["n"] == 2


def _has_tables(db: Path) -> bool:
    """False for an empty / unreadable file — a stray 0-byte state.db must
    make the migration test skip, not fail."""
    if not db.is_file() or db.stat().st_size == 0:
        return False
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            return conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] > 0
    except sqlite3.DatabaseError:
        return False


def _real_state_db() -> Path | None:
    """The user's live telemetry DB. FINAQ_MIGRATION_FIXTURE_DB wins whenever
    it is set (a worktree has no data_cache/); otherwise the repo-relative
    default path. None when neither is a populated database. The test only
    ever reads it."""
    override = os.environ.get("FINAQ_MIGRATION_FIXTURE_DB")
    candidate = Path(override) if override else Path("data_cache") / "state.db"
    return candidate if _has_tables(candidate) else None


def _row_counts(db: Path) -> dict[str, int]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


def test_migrating_a_copy_of_the_real_state_db_keeps_every_row(tmp_path: Path):
    """Copy the real data_cache/state.db to a tmp path, migrate the COPY, and
    check no table lost a row. Skips when no DB is around."""
    source = _real_state_db()
    if source is None:
        pytest.skip("no data_cache/state.db here — set FINAQ_MIGRATION_FIXTURE_DB to point at one")
    copy = tmp_path / "state_copy.db"
    # SQLite's online backup copies a consistent snapshot, WAL content included,
    # and never writes the source (opened read-only).
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(copy)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    source_mtime = source.stat().st_mtime_ns
    before = _row_counts(copy)
    assert before, "fixture DB has no tables"

    st.init_db(copy)

    after = _row_counts(copy)
    for table, n in before.items():
        if table == "meta":
            # init_db adds its own bookkeeping key (last_prune); the old keys stay.
            assert after[table] >= n
            continue
        assert after[table] == n, f"{table}: {n} rows before migration, {after[table]} after"
    assert V9_TABLES.issubset(after)
    assert {"status", "ended_at", "error", "attempt"}.issubset(_columns(copy, "llm_calls"))
    assert "source" in _columns(copy, "graph_runs") and "pid" in _columns(copy, "node_runs")
    assert _one(copy, "SELECT value FROM meta WHERE key = 'schema_version'")["value"] == str(
        st.SCHEMA_VERSION
    )
    with sqlite3.connect(copy) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert source.stat().st_mtime_ns == source_mtime  # the real file was never written


# --- v9 node rows -----------------------------------------------------------


def test_start_and_finish_node_run_round_trip(tmp_path: Path):
    db = tmp_path / "t.db"
    run_id = st.start_graph_run("NVDA", "ai_cake", source="dashboard", db_path=db)
    assert st.get_graph_run(run_id, db_path=db)["source"] == "dashboard"
    nid = st.start_node_run(run_id, "fundamentals", db_path=db)
    [row] = st.all_node_runs_for(run_id, db_path=db)
    assert row["id"] == nid
    assert row["status"] == "running"
    assert row["ended_at"] is None and row["duration_s"] is None
    assert row["pid"] == os.getpid()

    st.finish_node_run(
        nid,
        "completed",
        duration_s=2.5,
        tokens_in=10,
        tokens_out=4,
        cost_usd=0.01,
        n_calls=1,
        db_path=db,
    )
    [row] = st.all_node_runs_for(run_id, db_path=db)
    assert row["id"] == nid  # the same row was updated, not a second one inserted
    assert row["status"] == "completed"
    assert row["ended_at"] is not None
    assert row["duration_s"] == pytest.approx(2.5)
    assert (row["tokens_in"], row["tokens_out"], row["n_calls"]) == (10, 4, 1)
    assert row["cost_usd"] == pytest.approx(0.01)


def test_finish_node_run_records_failure_and_rejects_unknown_status(tmp_path: Path):
    db = tmp_path / "t.db"
    nid = st.start_node_run(None, "news", db_path=db)
    with pytest.raises(ValueError, match="status must be"):
        st.finish_node_run(nid, "running", db_path=db)
    st.finish_node_run(nid, "failed", error="Finnhub 500", duration_s=0.3, db_path=db)
    row = _one(db, "SELECT * FROM node_runs WHERE id = ?", nid)
    assert (row["status"], row["error"], row["run_id"]) == ("failed", "Finnhub 500", None)


def test_record_node_run_one_shot_path_still_works_without_pid(tmp_path: Path):
    """Old callers (and the entry-write-failed fallback) keep the one-shot
    insert; such rows simply have no pid."""
    db = tmp_path / "t.db"
    nid = st.record_node_run(
        None, "risk", st._now_iso(), st._now_iso(), 1.0, "completed", db_path=db
    )
    row = _one(db, "SELECT * FROM node_runs WHERE id = ?", nid)
    assert row["status"] == "completed" and row["pid"] is None


def test_running_node_is_visible_in_live_snapshot_between_start_and_finish(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(st, "DB_PATH", tmp_path / "live.db")
    run_id = st.start_graph_run("NVDA", "ai_cake", source="dashboard")
    nid = st.start_node_run(run_id, "news")
    cid = st.start_llm_call(
        run_id=run_id,
        node="news",
        model="m",
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
    )

    snap = st.live_snapshot()
    assert snap["poll_s"] == 1 and snap["now"]
    assert snap["vitals"]["active"] == 1 and snap["vitals"]["runs_today"] == 1
    [run] = snap["runs"]
    assert (run["run_id"], run["source"], run["status"]) == (run_id, "dashboard", "running")
    [node] = run["nodes"]
    assert (node["node"], node["status"], node["ended_at"]) == ("news", "running", None)
    [call] = node["calls"]
    assert (call["id"], call["status"], call["response"]) == (cid, "running", None)
    assert call["messages"] == [{"role": "system", "text": "S"}, {"role": "user", "text": "U"}]

    st.finish_llm_call(cid, status="completed", latency_s=0.5, response_text="A")
    st.finish_node_run(nid, "completed", duration_s=1.0)
    st.finish_graph_run(run_id, "completed", duration_s=1.0)
    snap = st.live_snapshot()
    assert snap["runs"] == [] and snap["vitals"]["active"] == 0
    [recent] = snap["recent"]
    assert recent["run_id"] == run_id and recent["nodes"] == {"news": "completed"}
    assert snap["medians"] == {"news": 1.0, "run": 1.0}


def test_live_snapshot_is_empty_shaped_when_db_missing(tmp_path: Path, monkeypatch):
    missing = tmp_path / "nope.db"
    monkeypatch.setattr(st, "DB_PATH", missing)
    snap = st.live_snapshot()
    assert snap["runs"] == [] and snap["recent"] == [] and snap["cio"] == []
    assert snap["medians"] == {}
    assert snap["vitals"]["active"] == 0 and snap["vitals"]["next_heartbeat"]
    assert not missing.exists()  # reads never create the file


# --- v9 LLM calls + prompt blobs --------------------------------------------


def test_start_llm_call_stores_full_messages_and_finish_appends_response(tmp_path: Path):
    db = tmp_path / "t.db"
    cid = st.start_llm_call(
        run_id=None,
        node="cio.planner",
        model="m",
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
        prompt_excerpt="[x]",
        db_path=db,
    )
    row = _one(db, "SELECT * FROM llm_calls WHERE id = ?", cid)
    assert (row["status"], row["ended_at"], row["prompt_excerpt"], row["attempt"]) == (
        "running",
        None,
        "[x]",
        1,
    )
    msgs = st.llm_messages_for_call(cid, db_path=db)
    assert [(m["seq"], m["role"], m["text"]) for m in msgs] == [
        (0, "system", "SYS"),
        (1, "user", "USER"),
    ]

    st.finish_llm_call(
        cid,
        status="completed",
        latency_s=1.5,
        tokens_in=10,
        tokens_out=3,
        cost_usd=0.002,
        response_excerpt="ANSWER",
        response_text="ANSWER",
        db_path=db,
    )
    row = _one(db, "SELECT * FROM llm_calls WHERE id = ?", cid)
    assert row["status"] == "completed" and row["ended_at"] is not None and row["error"] is None
    assert (row["tokens_in"], row["tokens_out"]) == (10, 3)
    assert row["latency_s"] == pytest.approx(1.5) and row["cost_usd"] == pytest.approx(0.002)
    assert row["response_excerpt"] == "ANSWER"
    msgs = st.llm_messages_for_call(cid, db_path=db)
    assert (msgs[-1]["seq"], msgs[-1]["role"], msgs[-1]["text"]) == (2, "assistant", "ANSWER")


def test_finish_llm_call_failed_keeps_error_and_no_assistant_message(tmp_path: Path):
    db = tmp_path / "t.db"
    cid = st.start_llm_call(
        run_id=None,
        node="news",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        db_path=db,
    )
    with pytest.raises(ValueError, match="status must be"):
        st.finish_llm_call(cid, status="running", latency_s=0.0, db_path=db)
    st.finish_llm_call(
        cid,
        status="failed",
        latency_s=0.2,
        response_excerpt="CALL FAILED: 500",
        error="500",
        db_path=db,
    )
    row = _one(db, "SELECT * FROM llm_calls WHERE id = ?", cid)
    assert (row["status"], row["error"], row["response_excerpt"]) == (
        "failed",
        "500",
        "CALL FAILED: 500",
    )
    assert [m["role"] for m in st.llm_messages_for_call(cid, db_path=db)] == ["user"]


def test_prompt_blobs_dedupe_the_system_prompt_across_calls(tmp_path: Path):
    db = tmp_path / "t.db"
    system = {"role": "system", "content": "You are the analyst." * 50}
    first = st.start_llm_call(
        run_id=None,
        node="n",
        model="m",
        messages=[system, {"role": "user", "content": "one"}],
        db_path=db,
    )
    second = st.start_llm_call(
        run_id=None,
        node="n",
        model="m",
        messages=[system, {"role": "user", "content": "two"}],
        db_path=db,
    )
    assert _one(db, "SELECT COUNT(*) AS n FROM prompt_blobs")["n"] == 3  # system, one, two
    assert _one(db, "SELECT COUNT(*) AS n FROM llm_messages")["n"] == 4
    sha = st.llm_messages_for_call(first, db_path=db)[0]["sha256"]
    assert st.llm_messages_for_call(second, db_path=db)[0]["sha256"] == sha
    blob = _one(db, "SELECT * FROM prompt_blobs WHERE sha256 = ?", sha)
    assert blob["seen_count"] == 2
    assert blob["last_seen"] >= blob["first_seen"]
    assert blob["n_chars"] == len(system["content"])


def test_record_llm_messages_appends_after_existing_rows(tmp_path: Path):
    db = tmp_path / "t.db"
    cid = st.record_llm_call(run_id=None, node="qa", model="m", latency_s=0.1, db_path=db)
    assert st.llm_messages_for_call(cid, db_path=db) == []
    n = st.record_llm_messages(cid, [{"role": "user", "content": "hi"}], response="yo", db_path=db)
    assert n == 2
    assert [
        (m["seq"], m["role"], m["text"]) for m in st.llm_messages_for_call(cid, db_path=db)
    ] == [
        (0, "user", "hi"),
        (1, "assistant", "yo"),
    ]
    assert st.record_llm_messages(cid, [], db_path=db) == 0
    # Non-string content (multi-part messages) is serialised, not dropped.
    st.record_llm_messages(
        cid, [{"role": "user", "content": [{"type": "text", "text": "x"}]}], db_path=db
    )
    assert (
        st.llm_messages_for_call(cid, db_path=db)[-1]["text"] == '[{"type": "text", "text": "x"}]'
    )


def test_prune_blobs_removes_only_unreferenced_blobs_older_than_the_window(tmp_path: Path):
    db = tmp_path / "t.db"
    system = {"role": "system", "content": "shared system prompt"}
    old = st.start_llm_call(
        run_id=None,
        node="n",
        model="m",
        messages=[system, {"role": "user", "content": "old user prompt"}],
        db_path=db,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE llm_calls SET ts = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=100)).isoformat(), old),
        )
    recent = st.start_llm_call(
        run_id=None,
        node="n",
        model="m",
        messages=[system, {"role": "user", "content": "recent user prompt"}],
        db_path=db,
    )

    assert st.prune_blobs(days=90, db_path=db) == {"messages": 2, "blobs": 1}

    assert st.llm_messages_for_call(old, db_path=db) == []
    assert [m["text"] for m in st.llm_messages_for_call(recent, db_path=db)] == [
        "shared system prompt",
        "recent user prompt",
    ]
    with sqlite3.connect(db) as conn:
        remaining = {r[0] for r in conn.execute("SELECT content FROM prompt_blobs")}
    assert remaining == {"shared system prompt", "recent user prompt"}
    # The call row itself (telemetry + excerpt) is never pruned.
    assert _one(db, "SELECT COUNT(*) AS n FROM llm_calls")["n"] == 2


def test_init_db_prunes_at_most_once_per_day(tmp_path: Path, monkeypatch):
    db = tmp_path / "t.db"
    calls: list[int] = []
    real = st._prune_blobs_conn

    def _counting(conn, days):
        calls.append(days)
        return real(conn, days)

    monkeypatch.setattr(st, "_prune_blobs_conn", _counting)
    st.init_db(db)
    st.init_db(db)
    st.start_graph_run("NVDA", "t", db_path=db)  # every write path calls init_db
    assert calls == [st.PROMPT_RETENTION_DAYS]
    today = datetime.now(UTC).date().isoformat()
    assert _one(db, "SELECT value FROM meta WHERE key = 'last_prune'")["value"] == today

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value = '2000-01-01' WHERE key = 'last_prune'")
    st.init_db(db)
    assert len(calls) == 2
    assert _one(db, "SELECT value FROM meta WHERE key = 'last_prune'")["value"] == today


def test_prompt_retention_days_reads_env_with_90_day_default(monkeypatch):
    monkeypatch.delenv("PROMPT_RETENTION_DAYS", raising=False)
    assert st._retention_days() == 90
    monkeypatch.setenv("PROMPT_RETENTION_DAYS", "30")
    assert st._retention_days() == 30
    monkeypatch.setenv("PROMPT_RETENTION_DAYS", "junk")
    assert st._retention_days() == 90


# --- v9 spans ---------------------------------------------------------------


def test_span_records_running_then_completed_with_meta(tmp_path: Path):
    db = tmp_path / "t.db"
    with st.span("cio_run", "c1", "probe", label="NVDA", db_path=db) as sp:
        [row] = st.spans_for("cio_run", "c1", db_path=db)
        assert row["id"] == sp.id
        assert (row["status"], row["ended_at"], row["label"]) == ("running", None, "NVDA")
        sp.meta["stale"] = True
    [row] = st.spans_for("cio_run", "c1", db_path=db)
    assert row["status"] == "completed" and row["ended_at"] is not None
    assert row["meta"] == {"stale": True} and row["error"] is None
    assert st.spans_for("cio_run", "other", db_path=db) == []


def test_span_records_failure_and_reraises(tmp_path: Path):
    db = tmp_path / "t.db"
    with pytest.raises(RuntimeError, match="boom"), st.span(
        "cio_run", "c1", "decide", label="NVDA", db_path=db
    ):
        raise RuntimeError("boom")
    [row] = st.spans_for("cio_run", "c1", db_path=db)
    assert (row["status"], row["error"]) == ("failed", "boom")
    assert row["ended_at"] is not None


def test_span_soft_failure_via_handle_error(tmp_path: Path):
    db = tmp_path / "t.db"
    with st.span("cio_run", "c1", "drill", label="NVDA", db_path=db) as sp:
        sp.meta["drill_run_id"] = None
        sp.error = "drill failed"
    [row] = st.spans_for("cio_run", "c1", db_path=db)
    assert (row["status"], row["error"], row["meta"]) == (
        "failed",
        "drill failed",
        {"drill_run_id": None},
    )


def test_span_is_noop_without_parent_and_survives_db_failures(tmp_path: Path, monkeypatch):
    db = tmp_path / "t.db"
    ran = False
    with st.span("cio_run", None, "notify", db_path=db) as sp:
        ran = True
    assert ran and sp.id is None
    assert not db.exists()  # nothing was written

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(st, "start_span", _boom)
    ran = False
    with st.span("cio_run", "c9", "probe", db_path=db) as sp:
        ran = True
    assert ran and sp.id is None  # telemetry failure, not the caller's


# --- v9 stale sweeper --------------------------------------------------------


def _dead_pid() -> int:
    """A pid that no longer exists: a child that has already exited."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_sweep_stale_runs_closes_dead_process_rows_and_keeps_live_ones(tmp_path: Path):
    db = tmp_path / "t.db"
    dead = st.start_graph_run("AAA", "t", db_path=db)
    dead_node = st.start_node_run(dead, "news", pid=_dead_pid(), db_path=db)
    dead_call = st.start_llm_call(run_id=dead, node="news", model="m", db_path=db)
    live = st.start_graph_run("BBB", "t", db_path=db)
    live_node = st.start_node_run(live, "news", db_path=db)  # this process → alive
    live_call = st.start_llm_call(run_id=live, node="news", model="m", db_path=db)

    counts = st.sweep_stale_runs(db_path=db)
    assert counts == {"node_runs": 1, "graph_runs": 1, "cio_runs": 0, "spans": 0, "llm_calls": 1}

    run = st.get_graph_run(dead, db_path=db)
    assert (run["status"], run["error"]) == ("failed", st.STALE_ERROR) and run["ended_at"]
    assert _one(db, "SELECT status FROM node_runs WHERE id = ?", dead_node)["status"] == "failed"
    assert _one(db, "SELECT status, error FROM llm_calls WHERE id = ?", dead_call) == {
        "status": "failed",
        "error": st.STALE_ERROR,
    }
    assert st.get_graph_run(live, db_path=db)["status"] == "running"
    assert _one(db, "SELECT status FROM node_runs WHERE id = ?", live_node)["status"] == "running"
    assert _one(db, "SELECT status FROM llm_calls WHERE id = ?", live_call)["status"] == "running"
    # Idempotent.
    assert all(v == 0 for v in st.sweep_stale_runs(db_path=db).values())


_SWEEP_TABLES = (
    ("graph_runs", "run_id", "started_at"),
    ("node_runs", "id", "started_at"),
    ("cio_runs", "run_id", "started_at"),
    ("spans", "id", "started_at"),
    ("llm_calls", "id", "ts"),
)


def _backdate_everything(db: Path, iso: str) -> None:
    with sqlite3.connect(db) as conn:
        for table, _, col in _SWEEP_TABLES:
            conn.execute(f"UPDATE {table} SET {col} = ?", (iso,))


def test_sweep_trusts_a_live_pid_and_uses_the_window_only_without_one(tmp_path: Path):
    """Real drill-ins and heartbeats exceed 30 minutes, so age alone never
    closes a row whose process is alive. Strip the pids (pre-v9 rows) and the
    window decides, children following their parents."""
    db = tmp_path / "t.db"
    run = st.start_graph_run("AAA", "t", db_path=db)
    node = st.start_node_run(run, "fundamentals", db_path=db)  # this process → alive
    cycle = st.start_cio_run("heartbeat", db_path=db)  # this process → alive
    span_id = st.start_span("cio_run", cycle, "drill", label="AAA", db_path=db)
    call = st.start_llm_call(run_id=run, node="fundamentals", model="m", db_path=db)
    assert _one(db, "SELECT pid FROM cio_runs WHERE run_id = ?", cycle)["pid"] == os.getpid()
    _backdate_everything(db, _hours_ago(2))

    assert all(v == 0 for v in st.sweep_stale_runs(minutes=30, db_path=db).values())

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE node_runs SET pid = NULL")
        conn.execute("UPDATE cio_runs SET pid = NULL")
    assert st.sweep_stale_runs(minutes=30, db_path=db) == {
        "node_runs": 1,
        "graph_runs": 1,
        "cio_runs": 1,
        "spans": 1,
        "llm_calls": 1,
    }
    keys = {
        "graph_runs": run,
        "node_runs": node,
        "cio_runs": cycle,
        "spans": span_id,
        "llm_calls": call,
    }
    for table, key_col, _ in _SWEEP_TABLES:
        row = _one(db, f"SELECT status, error FROM {table} WHERE {key_col} = ?", keys[table])
        assert (row["status"], row["error"]) == ("failed", st.STALE_ERROR), table


def test_sweep_hard_cap_closes_even_a_live_pid(tmp_path: Path):
    """Past STALE_MAX_MINUTES a running row is closed regardless of its pid,
    which may belong to an unrelated process by then."""
    db = tmp_path / "t.db"
    run = st.start_graph_run("AAA", "t", db_path=db)
    st.start_node_run(run, "news", db_path=db)
    st.start_cio_run("heartbeat", db_path=db)
    _backdate_everything(db, _hours_ago(st.STALE_MAX_MINUTES / 60 + 1))
    assert st.sweep_stale_runs(db_path=db) == {
        "node_runs": 1,
        "graph_runs": 1,
        "cio_runs": 1,
        "spans": 0,
        "llm_calls": 0,
    }


def test_sweep_follows_a_dead_cycle_down_to_its_steps_and_calls(tmp_path: Path):
    """A cycle whose process is gone is swept with its running span and the
    planner call attached to it; a live cycle keeps both however old. A pre-v9
    cycle (no pid) counts as alive while it keeps opening steps."""
    db = tmp_path / "t.db"
    old = _hours_ago(2)
    alive = st.start_cio_run("heartbeat", db_path=db)
    alive_span = st.start_span("cio_run", alive, "drill", label="NVDA", db_path=db)
    alive_call = st.start_llm_call(run_id=alive, node="cio.planner", model="m", db_path=db)
    dead = st.start_cio_run("heartbeat", db_path=db)
    dead_span = st.start_span("cio_run", dead, "drill", label="NVDA", db_path=db)
    dead_call = st.start_llm_call(run_id=dead, node="cio.planner", model="m", db_path=db)
    _backdate_everything(db, old)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE cio_runs SET pid = ? WHERE run_id = ?", (_dead_pid(), dead))

    counts = st.sweep_stale_runs(minutes=30, db_path=db)
    assert counts == {"node_runs": 0, "graph_runs": 0, "cio_runs": 1, "spans": 1, "llm_calls": 1}
    for table, key, expected in (
        ("cio_runs", dead, "failed"),
        ("cio_runs", alive, "running"),
    ):
        assert _one(db, f"SELECT status FROM {table} WHERE run_id = ?", key)["status"] == expected
    assert _one(db, "SELECT status FROM spans WHERE id = ?", dead_span)["status"] == "failed"
    assert _one(db, "SELECT status FROM spans WHERE id = ?", alive_span)["status"] == "running"
    assert _one(db, "SELECT status FROM llm_calls WHERE id = ?", dead_call)["status"] == "failed"
    assert _one(db, "SELECT status FROM llm_calls WHERE id = ?", alive_call)["status"] == "running"

    legacy = st.start_cio_run("heartbeat", db_path=db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE cio_runs SET pid = NULL, started_at = ? WHERE run_id = ?", (old, legacy)
        )
    st.start_span("cio_run", legacy, "decide", label="AAA", db_path=db)  # a fresh step
    assert all(v == 0 for v in st.sweep_stale_runs(minutes=30, db_path=db).values())


def test_sweep_closes_orphans_of_a_finished_parent_after_the_window(tmp_path: Path):
    """A call (or span) left `running` after its parent finished — the phase-2
    write failed — is closed once it is older than the window."""
    db = tmp_path / "t.db"
    run = st.start_graph_run("AAA", "t", db_path=db)
    call = st.start_llm_call(run_id=run, node="news", model="m", db_path=db)
    st.finish_graph_run(run, "completed", db_path=db)
    assert st.sweep_stale_runs(minutes=30, db_path=db)["llm_calls"] == 0  # still fresh
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE llm_calls SET ts = ? WHERE id = ?", (_hours_ago(2), call))
    assert st.sweep_stale_runs(minutes=30, db_path=db)["llm_calls"] == 1
    assert _one(db, "SELECT status FROM llm_calls WHERE id = ?", call)["status"] == "failed"


def test_init_db_is_cheap_after_the_first_call_in_a_process(tmp_path: Path):
    """The Live tab's per-second sweeper goes through init_db; after the first
    call in a process it must not write (another connection's data_version is
    unchanged). A replaced file is migrated again."""
    db = tmp_path / "t.db"
    st.init_db(db)
    probe = sqlite3.connect(db)
    try:
        before = probe.execute("PRAGMA data_version").fetchone()[0]
        for _ in range(5):
            st.init_db(db)
            st.sweep_stale_runs(db_path=db)
            st.live_snapshot(db_path=db)
        assert probe.execute("PRAGMA data_version").fetchone()[0] == before
    finally:
        probe.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    st.init_db(db)
    assert V9_TABLES.issubset(_table_names(db))


def test_live_snapshot_never_migrates_a_pre_v9_file(tmp_path: Path):
    """Reads never migrate: a v8 file polled before any write reads as an empty
    snapshot and stays at schema 8."""
    db = tmp_path / "v8.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(_V8_DDL)
    snap = st.live_snapshot(db_path=db)
    assert snap["runs"] == [] and snap["cio"] == [] and snap["recent"] == []
    assert _one(db, "SELECT value FROM meta WHERE key = 'schema_version'")["value"] == "8"
    assert not V9_TABLES & _table_names(db)


async def test_span_records_a_cancelled_step_as_failed(tmp_path: Path):
    """asyncio cancellation is not an Exception; the span must still close."""
    db = tmp_path / "t.db"

    async def step() -> None:
        with st.span("cio_run", "c1", "drill", label="NVDA", db_path=db):
            await asyncio.sleep(10)

    task = asyncio.create_task(step())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [row] = st.spans_for("cio_run", "c1", db_path=db)
    assert (row["status"], row["error"]) == ("failed", "CancelledError")


# --- v9 medians + run snapshot -----------------------------------------------


def _completed_run(db: Path, ticker: str, node_s: float, run_s: float) -> str:
    run_id = st.start_graph_run(ticker, "t", db_path=db)
    nid = st.start_node_run(run_id, "fundamentals", db_path=db)
    st.finish_node_run(nid, "completed", duration_s=node_s, db_path=db)
    st.finish_graph_run(run_id, "completed", duration_s=run_s, db_path=db)
    time.sleep(0.002)  # distinct started_at ordering
    return run_id


def test_node_medians_over_the_last_completed_runs(tmp_path: Path):
    db = tmp_path / "t.db"
    assert st.node_medians(db_path=db) == {}
    _completed_run(db, "A", node_s=10.0, run_s=100.0)
    _completed_run(db, "B", node_s=90.0, run_s=300.0)
    _completed_run(db, "C", node_s=20.0, run_s=200.0)
    failed = st.start_graph_run("D", "t", db_path=db)  # excluded: not completed
    nid = st.start_node_run(failed, "fundamentals", db_path=db)
    st.finish_node_run(nid, "completed", duration_s=999.0, db_path=db)
    st.finish_graph_run(failed, "failed", duration_s=999.0, db_path=db)
    st.start_graph_run("E", "t", db_path=db)  # excluded: still running

    assert st.node_medians(db_path=db) == {"fundamentals": 20.0, "run": 200.0}
    assert st.node_medians(limit=1, db_path=db) == {"fundamentals": 20.0, "run": 200.0}


def test_run_snapshot_returns_nodes_calls_messages_and_cio_lineage(tmp_path: Path):
    db = tmp_path / "t.db"
    cycle = st.start_cio_run("heartbeat", db_path=db)
    run_id = st.start_graph_run("NVDA", "ai_cake", source="cio", db_path=db)
    nid = st.start_node_run(run_id, "risk", db_path=db)
    cid = st.start_llm_call(
        run_id=run_id,
        node="risk",
        model="m",
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        db_path=db,
    )
    st.finish_llm_call(
        cid, status="completed", latency_s=1.0, tokens_in=5, response_text="R", db_path=db
    )
    st.finish_node_run(nid, "completed", duration_s=1.0, n_calls=1, db_path=db)
    st.record_error("risk", "soft issue", run_id=run_id, db_path=db)
    st.finish_graph_run(run_id, "completed", duration_s=2.0, confidence="high", db_path=db)
    st.record_cio_action(
        ticker="NVDA", action="drill", cio_run_id=cycle, drill_run_id=run_id, db_path=db
    )

    snap = st.run_snapshot(run_id, db_path=db)
    assert (snap["source"], snap["status"], snap["confidence"]) == ("cio", "completed", "high")
    [node] = snap["nodes"]
    assert node["node"] == "risk" and node["pid"] == os.getpid()
    [call] = node["calls"]
    assert call["messages"] == [{"role": "system", "text": "S"}, {"role": "user", "text": "U"}]
    assert call["response"] == "R" and call["tokens_in"] == 5 and call["status"] == "completed"
    assert [e["message"] for e in snap["errors"]] == ["soft issue"]
    assert snap["cio"]["cio_run_id"] == cycle and snap["cio"]["action"] == "drill"
    assert st.run_snapshot("missing", db_path=db) is None

    live = st.live_snapshot(db_path=db)
    [cio] = live["cio"]
    assert cio["run_id"] == cycle and cio["actions"][0]["drill_run_id"] == run_id
    assert cio["steps"] == []
    assert live["vitals"]["degraded_24h"] == 1  # completed, but with a run-scoped error


def test_next_heartbeat_picks_the_next_configured_hour():
    local = datetime.now().astimezone().tzinfo
    at = lambda h: datetime(2026, 10, 16, h, 0, tzinfo=local)  # noqa: E731
    assert st.next_heartbeat(at(4)).startswith("2026-10-16T05:00")
    assert st.next_heartbeat(at(6)).startswith("2026-10-16T13:00")
    assert st.next_heartbeat(at(14)).startswith("2026-10-17T05:00")
