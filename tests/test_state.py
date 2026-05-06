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
    assert a == {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "n_calls": 0}


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
