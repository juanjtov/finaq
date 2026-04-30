"""Step 5z unit tests — data/state.py SQLite telemetry layer.

Every test uses a tmp_path-scoped DB so the tests don't pollute
data_cache/state.db (and so they're isolated from each other).
"""

from __future__ import annotations

import sqlite3
import time
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
    expected = {"meta", "graph_runs", "node_runs", "alerts", "triage_runs", "errors"}
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
        error="ChromaDB connection refused",
        db_path=db,
    )
    rows = st.all_node_runs_for(run_id, db_path=db)
    assert rows[0]["status"] == "failed"
    assert "ChromaDB" in rows[0]["error"]


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
    eid = st.record_error("filings", "ChromaDB unreachable", db_path=db)
    assert eid > 0
    rows = st.recent_errors(db_path=db)
    assert len(rows) == 1
    assert rows[0]["agent"] == "filings"
    assert "ChromaDB" in rows[0]["message"]


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
