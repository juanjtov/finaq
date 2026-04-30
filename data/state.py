"""SQLite telemetry layer (Step 5z).

Single source of truth for "what happened" in FINAQ — graph runs, node
runs, alerts, triage scans, errors. Lives at `data_cache/state.db`.

Why SQLite (not Postgres / external observability):
  - Single-user single-box system. Postgres adds nothing here.
  - File-based — backs up cleanly with the rest of `data_cache/`.
  - No daemon, no port. Crashing FINAQ doesn't leave orphan state.

What we DO record (centrally queryable):
  - Graph-level: run_id, ticker, thesis, status, duration, errors.
  - Node-level: which agent ran, when, how long, did it fail.
  - Alerts (Phase 1 Triage).
  - Triage runs (Phase 1).
  - Standalone error events (centralised log).

What we DO NOT record:
  - Per-LLM-call tokens / USD cost / full prompt / response. **LangSmith**
    auto-instruments these when LANGSMITH_TRACING=true; duplicating that
    work in our SQLite would just diverge. The dashboard's Mission Control
    panel will deep-link to LangSmith for per-call detail; state.db is for
    aggregate-and-historical queries.

Run-ID propagation via `contextvars` — the graph wrapper sets
`current_run_id` once per ainvoke, and `_safe_node` reads it without
touching the FinaqState shape (keeps the state clean).
"""

from __future__ import annotations

import contextvars
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path("data_cache/state.db")
SCHEMA_VERSION = 1

# Contextvars propagated across asyncio tasks within a single graph invocation.
# Set by `invoke_with_telemetry`; read by `_safe_node` to attach node rows to
# the parent graph run.
current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)


# --- Schema migration -------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS graph_runs (
    run_id      TEXT PRIMARY KEY,
    ticker      TEXT,
    thesis      TEXT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL,
    status      TEXT NOT NULL,                  -- running | completed | failed
    error       TEXT,
    confidence  TEXT                            -- low | medium | high (from synthesis)
);

CREATE TABLE IF NOT EXISTS node_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    node        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL,
    status      TEXT NOT NULL,                  -- completed | failed
    error       TEXT,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_node_runs_run_id ON node_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_graph_runs_started_at ON graph_runs(started_at);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    ticker        TEXT,
    thesis        TEXT,
    severity      INTEGER,
    signal        TEXT,
    evidence_url  TEXT,
    notion_url    TEXT,
    status        TEXT NOT NULL DEFAULT 'pending'  -- pending | acked | dismissed | actioned
);

CREATE TABLE IF NOT EXISTS triage_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    items_scanned   INTEGER,
    alerts_emitted  INTEGER,
    duration_s      REAL,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS errors (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    agent     TEXT,
    message   TEXT NOT NULL,
    run_id    TEXT
);
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection. Idempotent; creates parent dir if needed."""
    target = Path(db_path) if db_path is not None else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    # `check_same_thread=False` is safe here because Streamlit's reruns happen
    # in the same process; we don't share connections across threads.
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create the schema if missing. Idempotent — calling twice is a no-op."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )


# --- Time helpers -----------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_id() -> str:
    return str(uuid.uuid4())


# --- Graph + node recording -------------------------------------------------


def start_graph_run(
    ticker: str,
    thesis: str,
    *,
    db_path: Path | None = None,
) -> str:
    """Insert a graph_runs row in the `running` state and return the run_id.
    Caller must call `finish_graph_run` after the run completes."""
    init_db(db_path)
    run_id = _new_run_id()
    started_at = _now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO graph_runs (run_id, ticker, thesis, started_at, status)
            VALUES (?, ?, ?, ?, 'running')
            """,
            (run_id, ticker, thesis, started_at),
        )
    return run_id


def finish_graph_run(
    run_id: str,
    status: str,
    *,
    error: str | None = None,
    confidence: str | None = None,
    duration_s: float | None = None,
    db_path: Path | None = None,
) -> None:
    """Update the graph_runs row at completion. `status` ∈ {completed, failed}."""
    if status not in ("completed", "failed"):
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    ended_at = _now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE graph_runs
               SET ended_at = ?, status = ?, error = ?, confidence = ?, duration_s = ?
             WHERE run_id = ?
            """,
            (ended_at, status, error, confidence, duration_s, run_id),
        )


def record_node_run(
    run_id: str | None,
    node: str,
    started_at: str,
    ended_at: str,
    duration_s: float,
    status: str,
    *,
    error: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert a node_runs row. `status` ∈ {completed, failed}."""
    if status not in ("completed", "failed"):
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO node_runs (run_id, node, started_at, ended_at, duration_s, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, node, started_at, ended_at, duration_s, status, error),
        )
        return cur.lastrowid or 0


def record_alert(
    ticker: str,
    thesis: str,
    severity: int,
    signal: str,
    *,
    evidence_url: str | None = None,
    notion_url: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert an alert row (Phase 1 Triage). Status starts as 'pending'."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO alerts (ts, ticker, thesis, severity, signal, evidence_url, notion_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (_now_iso(), ticker, thesis, severity, signal, evidence_url, notion_url),
        )
        return cur.lastrowid or 0


def update_alert_status(alert_id: int, status: str, *, db_path: Path | None = None) -> None:
    if status not in ("pending", "acked", "dismissed", "actioned"):
        raise ValueError(f"unknown alert status {status!r}")
    with _connect(db_path) as conn:
        conn.execute("UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id))


def record_triage_run(
    items_scanned: int,
    alerts_emitted: int,
    duration_s: float,
    *,
    error: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert one triage_runs row. Used by Phase 1 scripts/run_triage.py."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO triage_runs (ts, items_scanned, alerts_emitted, duration_s, error)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_now_iso(), items_scanned, alerts_emitted, duration_s, error),
        )
        return cur.lastrowid or 0


def record_error(
    agent: str,
    message: str,
    *,
    run_id: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Centralised error log. Agents already log via `utils.logger`; this
    additionally records to state.db so the Mission Control panel can list
    recent errors without grepping log files."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO errors (ts, agent, message, run_id)
            VALUES (?, ?, ?, ?)
            """,
            (_now_iso(), agent, message, run_id),
        )
        return cur.lastrowid or 0


# --- Queries ----------------------------------------------------------------


def recent_runs(limit: int = 20, *, db_path: Path | None = None) -> list[dict]:
    """Most-recent-first list of graph runs, with completed-node count."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT g.*, COUNT(n.id) AS node_runs_count,
                   SUM(CASE WHEN n.status = 'failed' THEN 1 ELSE 0 END) AS failed_nodes
              FROM graph_runs g
         LEFT JOIN node_runs n ON n.run_id = g.run_id
          GROUP BY g.run_id
          ORDER BY g.started_at DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_node_runs(limit: int = 100, *, db_path: Path | None = None) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM node_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_errors(limit: int = 50, *, db_path: Path | None = None) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM errors ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def recent_alerts(
    limit: int = 50,
    *,
    status: str | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    sql = "SELECT * FROM alerts"
    args: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def recent_triage_runs(limit: int = 50, *, db_path: Path | None = None) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM triage_runs ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def daily_run_counts(days: int = 14, *, db_path: Path | None = None) -> list[dict]:
    """Per-day counts of graph_runs (completed + failed). Used by the Mission
    Control daily-runs chart. NOTE: cost-tracking lives in LangSmith for
    Phase 0; if state.db ever needs cost columns, add a small price table
    and compute on insert. For now this is a count-only chart."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT date(started_at) AS day,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
              FROM graph_runs
             WHERE started_at >= date('now', '-{int(days)} days')
          GROUP BY day
          ORDER BY day
            """,
        ).fetchall()
    return [dict(r) for r in rows]


def health_summary(*, db_path: Path | None = None) -> dict:
    """Top-of-Mission-Control snapshot: total runs, last run timestamp,
    failure rate over the last 7 days."""
    if not Path(db_path or DB_PATH).exists():
        return {"total_runs": 0, "last_run_at": None, "failure_rate_7d": None}
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM graph_runs").fetchone()["n"]
        last = conn.execute(
            "SELECT started_at FROM graph_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        last_run_at = last["started_at"] if last else None
        last7 = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
              FROM graph_runs
             WHERE started_at >= date('now', '-7 days')
            """
        ).fetchone()
        failure_rate = (
            (last7["failed"] or 0) / last7["total"] if last7 and last7["total"] else None
        )
    return {
        "total_runs": total,
        "last_run_at": last_run_at,
        "failure_rate_7d": failure_rate,
    }


# --- Iteration helper for tests --------------------------------------------


def all_node_runs_for(run_id: str, *, db_path: Path | None = None) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM node_runs WHERE run_id = ? ORDER BY started_at",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_graph_run(run_id: str, *, db_path: Path | None = None) -> dict | None:
    if not Path(db_path or DB_PATH).exists():
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM graph_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None
