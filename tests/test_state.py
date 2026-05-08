"""Step 5z unit tests — data/state.py SQLite telemetry layer.

Every test uses a tmp_path-scoped DB so the tests don't pollute
data_cache/state.db (and so they're isolated from each other).
"""

from __future__ import annotations

import sqlite3
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
